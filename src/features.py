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


# Pourquoi chaque variable brute est gardée — la justification est écrite ici, pas dans un
# rapport recopié à la main (§ 4.3 : « the clarity with which you explain why each feature
# should help predict NPS »).
JUSTIFICATION_BRUTES = {
    "Contract": "l'engagement contractuel est le premier marqueur de la relation ; un client mensuel peut partir demain",
    "Tenure in Months": "l'ancienneté cumule l'expérience vécue ; les premiers mois concentrent les déceptions",
    "Offer": "un geste commercial reçu signale à la fois une politique de l'opérateur et un client jugé fragile",
    "Number of Referrals": "recommander est le comportement que le NPS mesure : proxy comportemental direct, observable, antérieur",
    "Monthly Charge": "le prix payé chaque mois ; la perception du rapport qualité-prix passe par lui",
    "Total Charges / Total Revenue": "l'intensité économique cumulée de la relation",
    "Total Refunds / Total Extra Data Charges / Total Long Distance Charges": "traces d'incidents de facturation et d'usages hors forfait — sources classiques de friction",
    "Avg Monthly GB Download / Avg Monthly Long Distance Charges": "intensité d'usage : un gros consommateur subit davantage les défauts de qualité",
    "Internet Service / Internet Type": "la fibre et le DSL n'ont ni la même qualité ni la même clientèle (Elero : vitesse et stabilité = premiers drivers)",
    "Services optionnels (sécurité, sauvegarde, protection, support premium, streaming, données illimitées)": "la composition du bouquet dit ce que le client attend et ce qu'il a accepté de payer",
    "Phone Service / Multiple Lines": "périmètre du service souscrit",
    "Paperless Billing": "proxy d'engagement numérique cité par l'énoncé",
    "Payment Method": "le mode de paiement révèle l'automatisation de la relation (auto-pay) et le canal préféré",
}

# Hypothèse métier derrière chaque variable dérivée.
HYPOTHESES_DERIVEES = {
    "nb_services": "intensité de la relation commerciale : plus de services, plus de raisons de rester — ou de se plaindre",
    "charge_par_service": "perception du rapport qualité-prix : ce que coûte chaque service souscrit",
    "tranche_anciennete": "l'insatisfaction d'un nouveau client (attentes déçues) n'a pas la même nature que celle d'un ancien (usure)",
    "a_recu_offre": "un geste commercial a-t-il été fait ? (indicateur binaire, complémentaire du type d'offre)",
    "a_eu_remboursement": "trace d'un incident de facturation reconnu par l'opérateur",
    "a_frais_supplementaires": "trace d'un dépassement de forfait — friction tarifaire",
    "revenu_par_mois": "intensité économique moyenne de la relation, indépendante de l'ancienneté",
    "a_parraine": "proxy comportemental direct de la recommandation, donc du NPS lui-même ; comportement passé observable, pas une conséquence de la note",
    "paiement_automatique": "auto-pay (énoncé § 4.3) : un paiement automatisé signale une relation installée et sans friction",
}


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

    # Proxy d'engagement cité par l'énoncé (§ 4.3, « auto-pay ») : prélèvement bancaire ou
    # carte = paiement automatique ; chèque posté = paiement manuel, relation plus distante.
    d["paiement_automatique"] = (~d["Payment Method"].astype(str).str.contains("Check", case=False)).astype(int)

    return d


# Groupes de variables EXCLUES des features par défaut, testés en ablation (phase 5) pour
# chiffrer ce que leur exclusion coûte — l'énoncé exige que l'arbitrage soit explicite (§ 4.7).
GROUPES_OPTIONNELS = {
    "demographie": ["Gender", "Age", "Under 30", "Senior Citizen", "Married", "Dependents", "Number of Dependents"],
    "geographie": ["Latitude", "Longitude", "Population"],
}


def colonnes_avec_groupes(df: pd.DataFrame, cfg: dict, groupes: list[str]) -> tuple[list[str], list[str]]:
    """Features de base + un ou plusieurs groupes optionnels (ablation)."""
    num, cat = colonnes_modele(df, cfg)
    for g in groupes:
        for c in GROUPES_OPTIONNELS[g]:
            if c in df.columns:
                (num if pd.api.types.is_numeric_dtype(df[c]) else cat).append(c)
    return num, cat


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
    if hasattr(prep, "num") and hasattr(prep, "cat") and not hasattr(prep, "transformers_"):
        return list(prep.num) + list(prep.cat)  # PreparationCatBoost : aucun encodage
    noms: list[str] = []
    for nom, transformeur, colonnes in prep.transformers_:
        if nom == "num":
            noms += list(colonnes)
        elif nom == "cat":
            enc = transformeur.named_steps["encodage"]
            noms += list(enc.get_feature_names_out(colonnes))
    return noms
