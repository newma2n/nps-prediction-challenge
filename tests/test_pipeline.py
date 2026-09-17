"""Tests de non-régression sur les invariants du projet.

    python -m pytest tests -q

Ils protègent ce qui, s'il casse silencieusement, invalide toute l'étude : la jointure sans
perte, l'exclusion des fuites, la construction de la cible, le protocole de non-réponse, la
règle de publication des métriques et la tolérance de l'application aux entrées incomplètes.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import charger_config, charger_tables, colonnes_exclues, joindre, preparer
from src.evaluate import metriques, precision_at_k
from src.features import colonnes_modele, construire
from src.protocol import decouper, diagnostic
from src.target import ORDRE, construire_cibles, nps

RACINE = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg():
    return charger_config()


@pytest.fixture(scope="session")
def df(cfg):
    d, _ = preparer(cfg)
    return construire(d)


# --- Données -------------------------------------------------------------------------
def test_jointure_sans_perte(cfg):
    assert len(joindre(charger_tables(cfg))) == 7043


def test_aucun_doublon_identifiant(df):
    assert df["Customer ID"].is_unique


def test_manquants_structurels_encodes(df):
    assert df["Offer"].isna().sum() == 0 and (df["Offer"] == "Aucune offre").sum() == 3877
    assert df["Internet Type"].isna().sum() == 0 and (df["Internet Type"] == "Pas d'internet").sum() == 1526


# --- Fuites --------------------------------------------------------------------------
def test_aucune_colonne_churn_dans_les_features(df, cfg):
    num, cat = colonnes_modele(df, cfg)
    interdites = {"Churn Label", "Churn Value", "Churn Score", "Churn Category", "Churn Reason",
                  "Customer Status", "CLTV", "Satisfaction Score", "Customer ID"}
    assert interdites.isdisjoint(set(num) | set(cat))


def test_demographie_exclue_des_features(df, cfg):
    num, cat = colonnes_modele(df, cfg)
    for c in ("Gender", "Age", "Senior Citizen", "Married", "Dependents", "Zip Code", "Latitude", "Longitude"):
        assert c not in num + cat


def test_fait_fondateur_churn_par_satisfaction(cfg):
    d = joindre(charger_tables(cfg))
    taux = d.groupby("Satisfaction Score")["Churn Value"].mean()
    assert taux[1] == 1.0 and taux[2] == 1.0 and taux[4] == 0.0 and taux[5] == 0.0


# --- Cible ---------------------------------------------------------------------------
def test_mappings_et_nps(df, cfg):
    cibles, arb = construire_cibles(df, cfg, cfg["seed"])
    for m in ("M1", "M2", "M3"):
        assert set(cibles[m].dropna().unique()) <= set(ORDRE)
    assert nps(cibles["M1"]) == pytest.approx(-42.0, abs=0.1)
    assert arb["n_ambigus"] == 2665
    assert 0.0 < arb["part_3_vers_detracteur"] < 1.0
    # Le modèle d'arbitrage doit séparer les extrêmes sans être irréel (pas de fuite)
    assert 0.70 <= arb["exactitude_sur_extremes"] <= 0.90


def test_indicateur_profil_detracteur_reserve_aux_ambigus(df, cfg):
    cibles, _ = construire_cibles(df, cfg, cfg["seed"])
    amb = df["Satisfaction Score"] == 3
    assert cibles.loc[amb, "profil_detracteur_si_ambigu"].notna().all()
    assert cibles.loc[~amb, "profil_detracteur_si_ambigu"].isna().all()


# --- Protocole -----------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["S1_MCAR", "S2_MAR", "S3_MNAR"])
def test_taux_de_reponse_proche_de_15_pct(df, cfg, scenario):
    dec = decouper(df, cfg, scenario, cfg["seed"])
    assert 0.12 <= dec["taux_reponse_obtenu"] <= 0.18
    assert not (dec["repondants"] & dec["silencieux"]).any()


def test_mnar_surrepresente_les_extremes(df, cfg):
    d1 = diagnostic(df, cfg, decouper(df, cfg, "S1_MCAR", cfg["seed"]))
    d3 = diagnostic(df, cfg, decouper(df, cfg, "S3_MNAR", cfg["seed"]))
    assert d3["part_extremes_repondants"] > d1["part_extremes_repondants"] + 0.15


def test_poids_ipw_normalises(df, cfg):
    dec = decouper(df, cfg, "S3_MNAR", cfg["seed"])
    assert dec["poids_ipw"][dec["repondants"]].mean() == pytest.approx(1.0)
    assert (dec["poids_ipw"][dec["silencieux"]] == 0).all()


# --- Évaluation ----------------------------------------------------------------------
def test_metriques_toujours_avec_detail_par_classe():
    y = np.array(ORDRE * 10); p = np.array(ORDRE * 10)
    m = metriques(y, p)
    assert set(m["par_classe"]) == set(ORDRE)
    assert m["kappa_quadratique"] == pytest.approx(1.0)


def test_precision_at_k_borne_et_lift():
    y = np.array(["Detracteur"] * 20 + ["Promoteur"] * 80)
    proba = np.r_[np.linspace(0.9, 0.6, 20), np.linspace(0.5, 0.0, 80)]
    r = precision_at_k(y, proba, 20)
    assert r["precision_at_k"] == 1.0 and r["lift"] == pytest.approx(5.0)


# --- Artefacts et application --------------------------------------------------------
@pytest.mark.skipif(not (RACINE / "models" / "modele_final.joblib").exists(), reason="pipeline non exécuté")
def test_modele_tolere_entrees_vides_et_modalites_inconnues():
    art = joblib.load(RACINE / "models" / "modele_final.joblib")
    vide = pd.DataFrame([{c: np.nan for c in art["num"] + art["cat"]}])
    vide["Contract"] = "MODALITE_INEXISTANTE"
    proba = art["pipeline"].predict_proba(vide)[0]
    assert proba.shape == (3,) and proba.sum() == pytest.approx(1.0, abs=1e-6)
    assert str(art["pipeline_decision"].predict(vide)[0]) in ORDRE


@pytest.mark.skipif(not (RACINE / "reports" / "resultats.json").exists(), reason="pipeline non exécuté")
def test_resultats_pas_de_fuite_et_trois_classes_predites():
    R = json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
    assert R["diagnostic_fuite"]["alarme"] is False
    assert R["diagnostic_fuite"]["exactitude_max_observee"] < 0.85
    for c in ORDRE:
        assert R["modele_final"]["metriques_retenues"]["par_classe"][c]["rappel"] > 0.05
