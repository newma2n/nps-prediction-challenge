"""Étapes 4 à 9 — collecte, description, qualité, registre de fuites, sélection.

Charge les cinq tables IBM Telco, les réunit en une table client unique, contrôle la
qualité, et applique le registre de fuites défini dans config.yaml.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

RACINE = Path(__file__).resolve().parent.parent


def charger_config() -> dict:
    with open(RACINE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Étape 4 : collecte et jointure -------------------------------------------
def charger_tables(cfg: dict) -> dict[str, pd.DataFrame]:
    brut = RACINE / cfg["chemins"]["brut"]
    noms = {
        "demographics": "Telco_customer_churn_demographics.xlsx",
        "location": "Telco_customer_churn_location.xlsx",
        "population": "Telco_customer_churn_population.xlsx",
        "services": "Telco_customer_churn_services.xlsx",
        "status": "Telco_customer_churn_status.xlsx",
    }
    return {k: pd.read_excel(brut / v) for k, v in noms.items()}


def joindre(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Jointure des 4 tables client sur Customer ID, puis population via Zip Code.

    Jointure GAUCHE à partir de `demographics` (la table qui définit la population de référence),
    contrôlée par deux assertions : le nombre de lignes ne bouge pas (aucun doublon créé) et
    aucun client n'arrive sans ligne de services, de statut ou de localisation (aucune perte
    silencieuse). Une jointure interne masquerait le second cas en supprimant les clients
    incomplets ; ici on veut qu'un client incomplet fasse échouer le pipeline.
    """
    df = tables["demographics"].drop(columns=["Count"])
    for nom in ("location", "services", "status"):
        t = tables[nom].drop(columns=[c for c in ("Count", "Quarter") if c in tables[nom]])
        df = df.merge(t, on="Customer ID", how="left", validate="one_to_one")

    pop = tables["population"][["Zip Code", "Population"]].drop_duplicates("Zip Code")
    df = df.merge(pop, on="Zip Code", how="left")

    # Portes de contrôle de l'étape 4 : ni ligne perdue ou dupliquée, ni client incomplet.
    assert len(df) == 7043, f"Jointure : {len(df)} lignes au lieu de 7043 — bug de jointure."
    temoins = {"services": "Tenure in Months", "status": "Satisfaction Score", "location": "Zip Code",
               "population": "Population"}
    manquants = {t: int(df[c].isna().sum()) for t, c in temoins.items() if df[c].isna().any()}
    assert not manquants, f"Jointure : clients sans ligne dans {manquants} — table incomplète."
    return df


# --- Étape 7 : qualité des données --------------------------------------------
def controler_qualite(df: pd.DataFrame) -> dict:
    """Les cinq contrôles de l'étape 7. Retourne un rapport, ne modifie rien."""
    rapport: dict = {"lignes": len(df), "colonnes": df.shape[1]}

    # 1. Manquants
    manquants = df.isna().sum()
    rapport["manquants"] = manquants[manquants > 0].sort_values(ascending=False).to_dict()

    # 2. Doublons
    rapport["doublons_id"] = int(df["Customer ID"].duplicated().sum())
    rapport["doublons_lignes"] = int(df.drop(columns=["Customer ID"]).duplicated().sum())

    # 3. Aberrantes (écart interquartile)
    num = ["Tenure in Months", "Monthly Charge", "Total Charges", "Total Revenue",
           "Avg Monthly GB Download"]
    aberrantes = {}
    for c in num:
        if c not in df:
            continue
        q1, q3 = df[c].quantile([0.25, 0.75])
        ecart = q3 - q1
        bas, haut = q1 - 1.5 * ecart, q3 + 1.5 * ecart
        aberrantes[c] = int(((df[c] < bas) | (df[c] > haut)).sum())
    rapport["aberrantes"] = aberrantes

    # 4. Cohérence interne : Total Charges vs Monthly Charge x Tenure
    attendu = df["Monthly Charge"] * df["Tenure in Months"]
    ecart_rel = (df["Total Charges"] - attendu).abs() / attendu.replace(0, pd.NA)
    rapport["incoherence_facturation"] = int((ecart_rel > 0.5).sum())
    rapport["anciennete_zero"] = int((df["Tenure in Months"] == 0).sum())

    # 5. Colonnes constantes
    rapport["constantes"] = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    return rapport


# --- Étapes 8 et 9 : registre de fuites et sélection ---------------------------
def registre_fuites(cfg: dict) -> dict[str, list[str]]:
    c = cfg["colonnes"]
    return {
        "cible": [c["cible"]],
        "fuite_consequence": c["fuite_consequence"],
        "fuite_disponibilite": c["fuite_disponibilite"],
        "fuite_modele_amont": c["fuite_modele_amont"],
        "techniques": c["techniques"],
        "audit_seulement": c["audit_seulement"],
    }


def colonnes_exclues(cfg: dict) -> list[str]:
    """Tout ce qui ne doit jamais entrer dans le modèle."""
    reg = registre_fuites(cfg)
    exclues = []
    for cle in ("cible", "fuite_consequence", "fuite_disponibilite",
                "fuite_modele_amont", "techniques", "audit_seulement"):
        exclues += reg[cle]
    return exclues


def preparer(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Chaîne complète étapes 4 → 9. Retourne la table client et le rapport qualité."""
    cfg = cfg or charger_config()
    tables = charger_tables(cfg)
    df = joindre(tables)
    rapport = controler_qualite(df)

    # Manquants structurels → catégorie explicite, jamais imputés (étape 11).
    if "Offer" in df:
        df["Offer"] = df["Offer"].fillna("Aucune offre")
    if "Internet Type" in df:
        df["Internet Type"] = df["Internet Type"].fillna("Pas d'internet")

    interim = RACINE / cfg["chemins"]["interim"]
    interim.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim / "clients.parquet", index=False)
    return df, rapport


if __name__ == "__main__":
    cfg = charger_config()
    df, rapport = preparer(cfg)
    print(f"Table client : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("Doublons ID :", rapport["doublons_id"])
    print("Colonnes constantes :", rapport["constantes"])
    print("Manquants :", rapport["manquants"])
