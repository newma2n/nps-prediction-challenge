"""Étapes 12 et 15 — feature engineering et pipeline de préparation.

Chaque variable dérivée est adossée à une hypothèse métier, écrite en commentaire.
Aucune transformation n'est appliquée hors du Pipeline : sinon les statistiques
calculées sur tout le jeu fuiraient vers l'évaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import colonnes_exclues

SERVICES_BINAIRES = [
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection Plan", "Premium Tech Support",
    "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data",
]


def _oui(serie: pd.Series) -> pd.Series:
    return (serie.astype(str).str.strip().str.lower() == "yes").astype(int)


def construire(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables dérivées. Ne retire rien."""
    d = df.copy()

    # Intensité de la relation commerciale : combien de services souscrits.
    d["nb_services"] = sum(_oui(d[c]) for c in SERVICES_BINAIRES if c in d)

    # Perception du rapport qualité-prix : ce que coûte chaque service.
    d["charge_par_service"] = d["Monthly Charge"] / d["nb_services"].replace(0, np.nan)

    # L'insatisfaction d'un nouveau client n'a pas la même nature que celle d'un ancien.
    d["tranche_anciennete"] = pd.cut(
        d["Tenure in Months"], bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6m", "6-12m", "1-2a", "2-4a", "4a+"],
    ).astype(str)

    # Un geste commercial a-t-il été fait ?
    d["a_recu_offre"] = (d["Offer"] != "Aucune offre").astype(int)

    # Traces d'incidents de facturation.
    d["a_eu_remboursement"] = (d["Total Refunds"] > 0).astype(int)
    d["a_frais_supplementaires"] = (d["Total Extra Data Charges"] > 0).astype(int)

    # Revenu moyen par mois d'ancienneté : intensité économique de la relation.
    d["revenu_par_mois"] = d["Total Revenue"] / d["Tenure in Months"].replace(0, np.nan)

    # Le parrainage est un proxy comportemental DIRECT de la recommandation,
    # donc du NPS lui-même. Variable la plus proche de la cible sans être une fuite :
    # c'est un comportement passé observable, pas une conséquence de la note.
    d["a_parraine"] = _oui(d["Referred a Friend"])

    return d


def colonnes_modele(df: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str]]:
    """Sépare les variables explicatives retenues en numériques et catégorielles."""
    exclues = set(colonnes_exclues(cfg))
    # Colonnes dérivées de variables exclues : retirées aussi.
    exclues |= {"Referred a Friend"}

    gardees = [c for c in df.columns if c not in exclues]
    num = [c for c in gardees if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in gardees if c not in num]
    return num, cat


def pipeline_preparation(num: list[str], cat: list[str], normaliser: bool) -> ColumnTransformer:
    """Préprocessing encapsulé. `normaliser` : True pour les modèles linéaires et
    ordinaux, False pour les modèles à arbres qui n'en ont pas besoin."""
    etapes_num = [("imputation", SimpleImputer(strategy="median"))]
    if normaliser:
        etapes_num.append(("echelle", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(etapes_num), num),
        ("cat", Pipeline([
            ("imputation", SimpleImputer(strategy="most_frequent")),
            # handle_unknown='ignore' : l'interface Streamlit doit tolérer une
            # modalité inconnue sans planter.
            ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat),
    ], remainder="drop")


def noms_apres_encodage(prep: ColumnTransformer) -> list[str]:
    """Noms des colonnes en sortie du préprocessing, pour SHAP et l'interprétabilité."""
    noms: list[str] = []
    for nom, transformeur, colonnes in prep.transformers_:
        if nom == "num":
            noms += list(colonnes)
        elif nom == "cat":
            enc = transformeur.named_steps["encodage"]
            noms += list(enc.get_feature_names_out(colonnes))
    return noms
