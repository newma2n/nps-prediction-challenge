"""Tests de non-régression sur les invariants du projet.

    python -m pytest tests -q

Ils protègent ce qui, s'il casse silencieusement, invalide toute l'étude : la jointure sans perte,
l'exclusion des fuites, la construction de la cible, le protocole de non-réponse, le catalogue de
modèles (chacun s'ajuste, prédit trois classes, se clone), la règle de publication des métriques, la
sélection sans regarder le test, et la tolérance de l'application aux entrées incomplètes.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.data import charger_config, charger_tables, colonnes_exclues, joindre, preparer
from src.evaluate import metriques, precision_at_k, nps_de, seuils_par_groupe
from src.features import GROUPES_OPTIONNELS, HYPOTHESES_DERIVEES, colonnes_avec_groupes, colonnes_modele, construire
from src.models import CATALOGUE, NOMS, SIMPLICITE, construire_modeles, entrainer, espace_recherche
from src.protocol import decouper, diagnostic, estimer_propension
from src.target import ORDRE, construire_cibles, nps

RACINE = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg():
    return charger_config()


@pytest.fixture(scope="session")
def df(cfg):
    d, _ = preparer(cfg)
    return construire(d)


@pytest.fixture(scope="session")
def jeu(df, cfg):
    """Petit jeu d'entraînement / test (répondants S3, cible M3) pour les tests de modèles."""
    cibles, _ = construire_cibles(df, cfg, cfg["seed"])
    num, cat = colonnes_modele(df, cfg)
    dec = decouper(df, cfg, "S3_MNAR", cfg["seed"])
    rep = dec["repondants"]
    y = cibles["M3"].astype(str).values
    sat = pd.to_numeric(df[cfg["colonnes"]["cible"]]).values.astype(float)
    return dict(X=df[num + cat], y=y, sat=sat, rep=rep, num=num, cat=cat)


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


def test_demographie_et_geographie_exclues_des_features(df, cfg):
    num, cat = colonnes_modele(df, cfg)
    for c in ("Gender", "Age", "Senior Citizen", "Married", "Dependents", "Zip Code", "Latitude", "Longitude", "Population"):
        assert c not in num + cat


def test_groupes_optionnels_reintegrables_pour_ablation(df, cfg):
    num, cat = colonnes_avec_groupes(df, cfg, ["demographie", "geographie"])
    for c in GROUPES_OPTIONNELS["demographie"] + GROUPES_OPTIONNELS["geographie"]:
        assert c in num + cat


def test_fait_fondateur_churn_par_satisfaction(cfg):
    d = joindre(charger_tables(cfg))
    taux = d.groupby("Satisfaction Score")["Churn Value"].mean()
    assert taux[1] == 1.0 and taux[2] == 1.0 and taux[4] == 0.0 and taux[5] == 0.0


# --- Features ------------------------------------------------------------------------
def test_chaque_variable_derivee_a_une_hypothese(df, cfg):
    num, cat = colonnes_modele(df, cfg)
    for v in HYPOTHESES_DERIVEES:
        assert v in num + cat, f"{v} déclarée mais absente des features"
    assert set(df["paiement_automatique"].unique()) <= {0, 1}


# --- Cible ---------------------------------------------------------------------------
def test_mappings_et_nps(df, cfg):
    cibles, arb = construire_cibles(df, cfg, cfg["seed"])
    for m in ("M1", "M2", "M3"):
        assert set(cibles[m].dropna().unique()) <= set(ORDRE)
    assert nps(cibles["M1"]) == pytest.approx(-42.0, abs=0.1)
    assert nps(cibles["M2"]) == pytest.approx(-4.1, abs=0.2)
    assert arb["n_ambigus"] == 2665
    assert 0.0 < arb["part_3_vers_detracteur"] < 1.0
    assert 0.70 <= arb["exactitude_sur_extremes"] <= 0.90


def test_arbitrage_des_3_verrouille(df, cfg):
    """La décision la plus lourde de l'étude est verrouillée par un test, pas seulement racontée.

    Le bloc des 2 665 clients à satisfaction « 3 » part chez les Passifs parce que 27,9 % seulement
    penchent détracteur — bien en deçà du seuil de bascule de 50 %. Si un changement de données, de
    features ou de graine déplaçait cette valeur, le NPS de référence changerait de plus de
    60 points sans que personne ne s'en aperçoive. Ce test rend ce glissement impossible en silence.
    """
    cibles, arb = construire_cibles(df, cfg, cfg["seed"])
    assert arb["destination_des_3"] == "Passif"
    assert arb["part_3_vers_detracteur"] == pytest.approx(0.2788, abs=0.02)
    assert arb["seuil_bascule"] == 0.5
    assert arb["issues_possibles"] == ["Detracteur", "Passif"]   # un milieu d'échelle ne se promeut pas
    assert arb["marge_a_la_bascule"] > 0.15                      # la décision n'est pas sur le fil
    assert nps(cibles["M3"]) == pytest.approx(21.3, abs=1.0)
    # Les trois mappings doivent rester distincts : sinon la sensibilité au mapping ne teste rien.
    assert nps(cibles["M1"]) < nps(cibles["M2"]) < nps(cibles["M3"])


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


def test_propension_estimee_est_positive_et_normalisee(df, cfg, jeu):
    w = estimer_propension(df, jeu["rep"], jeu["num"], jeu["cat"], cfg["seed"])
    assert (w[jeu["rep"]] > 0).all() and (w[~jeu["rep"]] == 0).all()
    assert w[jeu["rep"]].mean() == pytest.approx(1.0)


# --- Catalogue de modèles ------------------------------------------------------------
def test_catalogue_complet_et_justifie():
    assert len(CATALOGUE) == 15
    familles = {m["famille"] for m in CATALOGUE}
    assert len(familles) == 7
    for m in CATALOGUE:
        assert m["pourquoi"] and m["hypothese"] and m["reference"], m["nom"]
        assert m["nom"] in SIMPLICITE
    assert {m["formulation"] for m in CATALOGUE} == {"classification nominale", "classification ordinale", "régression 1-5 puis seuillage"}


@pytest.mark.parametrize("nom", NOMS)
def test_chaque_modele_s_ajuste_predit_trois_classes_et_se_clone(jeu, cfg, nom):
    X, y, sat, rep = jeu["X"], jeu["y"], jeu["sat"], jeu["rep"]
    idx = np.where(rep)[0][:400]
    m = construire_modeles(jeu["num"], jeu["cat"], cfg["seed"], [nom])[nom]
    m2 = clone(m)  # RandomizedSearchCV et la validation croisée exigent des estimateurs clonables
    entrainer(m2, X.iloc[idx], y[idx], y_continu=sat[idx])
    p = m2.predict_proba(X.iloc[:20])
    assert p.shape == (20, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.asarray(m2.predict(X.iloc[:20])).astype(str)) <= set(ORDRE)
    esp = espace_recherche(nom)
    if esp:
        for k in esp:
            assert k.startswith("modele__")


# --- Évaluation ----------------------------------------------------------------------
def test_metriques_toujours_avec_detail_par_classe():
    y = np.array(ORDRE * 10); p = np.array(ORDRE * 10)
    m = metriques(y, p)
    assert set(m["par_classe"]) == set(ORDRE)
    assert m["kappa_quadratique"] == pytest.approx(1.0)


def test_metriques_avec_probabilites():
    y = np.array(["Detracteur"] * 30 + ["Passif"] * 30 + ["Promoteur"] * 30)
    proba = np.tile(np.eye(3), (30, 1)); proba = np.vstack([proba[::3], proba[1::3], proba[2::3]])
    pred = np.array(ORDRE)[proba.argmax(1)]
    m = metriques(y, pred, proba, ORDRE)
    assert m["auc_detracteur"] == pytest.approx(1.0) and m["brier_detracteur"] == pytest.approx(0.0) and 0 <= m["ece_detracteur"] <= 1


def test_precision_at_k_borne_et_lift():
    y = np.array(["Detracteur"] * 20 + ["Promoteur"] * 80)
    proba = np.r_[np.linspace(0.9, 0.6, 20), np.linspace(0.5, 0.0, 80)]
    r = precision_at_k(y, proba, 20)
    assert r["precision_at_k"] == 1.0 and r["lift"] == pytest.approx(5.0)


def test_nps_et_seuils_par_groupe():
    assert nps_de(["Promoteur"] * 6 + ["Detracteur"] * 2 + ["Passif"] * 2) == pytest.approx(40.0)
    y = np.array(["Detracteur"] * 20 + ["Promoteur"] * 20); proba = np.r_[np.linspace(0.9, 0.5, 20), np.linspace(0.6, 0.0, 20)]
    g = np.array(["A"] * 20 + ["B"] * 20)
    y2 = np.array(["Detracteur"] * 10 + ["Promoteur"] * 10 + ["Detracteur"] * 10 + ["Promoteur"] * 10)
    r = seuils_par_groupe(y2, proba, g, rappel_cible=0.8)
    assert set(r) == {"A", "B"} and all(v["rappel"] >= 0.8 for v in r.values())


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
def test_resultats_pas_de_fuite_trois_classes_et_selection_sans_test():
    R = json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
    assert R["diagnostic_fuite"]["alarme"] is False
    assert R["diagnostic_fuite"]["exactitude_max_observee"] < 0.85
    for c in ORDRE:
        assert R["modele_final"]["metriques_retenues"]["par_classe"][c]["rappel"] > 0.05
    sel = R["selection_modele_final"]
    # Le modèle retenu est le plus simple des candidats dans la marge de parcimonie — jamais choisi sur les silencieux.
    marge = [c for c in sel["classement"] if c["dans_la_marge"]]
    assert sel["retenu"] == min(marge, key=lambda c: (SIMPLICITE[c["modele"]], -c["kappa_cv_ipw"]))["modele"]
    assert len(sel["classement"]) == len(NOMS)


@pytest.mark.skipif(not (RACINE / "reports" / "resultats.json").exists(), reason="pipeline non exécuté")
def test_criteres_de_succes_chiffres_sont_verifies(cfg):
    """Les critères de succès de l'étape 1 sont vérifiés par le pipeline ET par un test.

    Un critère fixé a priori qui n'est jamais confronté au résultat n'est pas un critère : c'est une
    intention. On relit donc ici les seuils de `config.yaml` et on les compare aux chiffres publiés.
    """
    R = json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
    cs = cfg["criteres_succes"]
    pk = R["evaluation_metier"]["precision_at_k"]
    eco = R["evaluation_metier"]["economie"]
    assert pk["lift"] >= cs["lift_min"]
    assert pk["precision_at_k"] >= cs["precision_at_k_min"]
    assert eco["gain_net_eur"] > cs["gain_net_min_eur"]
    assert eco["gain_net_eur"] > eco["gain_net_ciblage_aleatoire_eur"]   # le modèle bat le hasard
    for c in ORDRE:
        assert R["modele_final"]["metriques_retenues"]["par_classe"][c]["rappel"] >= cs["rappel_min_par_classe"]
    # Le bloc de vérification doit exister dans les résultats et dire la même chose que ce test.
    v = R["criteres_succes"]["verification"]
    assert len(v) == 5 and all(x["atteint"] for x in v)
    # Un écart d'équité au-delà du seuil doit être PUBLIÉ, pas masqué.
    signale = [x for x in v if "signalement" in x][0]
    if signale["valeur"] is not None and signale["valeur"] > cs["ecart_equite_signalement"]:
        assert signale["signalement"], "écart d'équité au-dessus du seuil mais aucune dimension signalée"


@pytest.mark.skipif(not (RACINE / "reports" / "resultats.json").exists(), reason="pipeline non exécuté")
def test_leviers_et_drivers_sont_exploitables():
    """Trois garde-fous d'utilité, pas de performance.

    (1) Le premier levier est très concentré, et c'est un fait de la base, pas un défaut de la
        règle : presque tous les détracteurs prédits sont en contrat mensuel. Ce qu'on verrouille
        ici, c'est que la concentration soit **publiée** et que le **second** levier, lui,
        différencie réellement les clients — sans quoi la recommandation ne trierait rien.
    (2) Les contributions doivent être agrégées par variable d'origine : la présence d'un nom de
        modalité one-hot dans le classement signale une régression de l'agrégation.
    (3) Le sens de chaque driver doit être publié pour toutes les variables classées.
    """
    R = json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
    lev = R["leviers"]
    assert len(lev["distribution_detracteurs_predits"]) >= 2
    assert lev["concentration_du_levier_dominant"] is not None      # publiée, jamais tue
    assert lev["n_leviers_distincts_en_second"] >= 3                # le second levier trie
    assert len(lev["distribution_paires"]) >= 3
    assert lev["concentration_de_la_paire_dominante"] < 0.80        # le couple n'est pas dégénéré
    dr = R["drivers"]
    assert "erreur" not in dr
    assert set(dr["sens"]) <= set(dr["importance_detracteur"])
    assert not [v for v in dr["importance_detracteur"] if v.startswith("Contract_")]


@pytest.mark.skipif(not (RACINE / "data" / "processed" / "clients_scores.parquet").exists(), reason="pipeline non exécuté")
def test_scores_clients_complets():
    s = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet")
    assert len(s) == 7043 and s["proba_detracteur"].between(0, 1).all()
    assert set(s["prediction"].unique()) <= set(ORDRE) and s["levier_recommande"].notna().all()


# --- Livrables (contraintes explicites de l'énoncé § 6) -------------------------------
@pytest.mark.skipif(not (RACINE / "livrable" / "write_up.pdf").exists(), reason="write-up non généré")
def test_le_write_up_tient_dans_les_3_a_6_pages_demandees():
    """L'énoncé impose « a short write-up (3–6 pages) ». C'est une contrainte chiffrée, donc
    vérifiable : elle mérite un test plutôt qu'une relecture à l'œil. Un document qui déborde
    n'est pas un document plus riche, c'est un document hors cahier des charges."""
    import fitz
    doc = fitz.open(RACINE / "livrable" / "write_up.pdf")
    n = doc.page_count
    doc.close()
    assert 3 <= n <= 6, f"write_up.pdf fait {n} pages ; l'énoncé en demande 3 à 6"
