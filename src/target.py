"""Étape 10 — construction de la cible NPS, et arbitrage empirique des « 3 ».

Trois mappings :
  M1  mapping baseline de l'énoncé  (<=3 Détracteur, 4 Passif, 5 Promoteur)
  M2  recentré                      (<=2 Détracteur, 3-4 Passif, 5 Promoteur)
  M3  arbitré par les données       (les « 3 » sont tranchés par la mesure, pas par décret)

MÉTHODE D'ARBITRAGE (d'après Elero et al., 2026)
-------------------------------------------------
On entraîne un modèle UNIQUEMENT sur les extrêmes non ambigus — satisfaction 1-2
contre 4-5 — puis on fait passer les 2 665 clients à satisfaction 3, jamais vus, à
travers ce modèle. La proportion qui bascule de chaque côté est une MESURE, pas un
choix.

PIÈGE ÉVITÉ — pourquoi on n'étiquette pas chaque « 3 » individuellement
-----------------------------------------------------------------------
Il serait tentant d'affecter à chaque client à satisfaction 3 la classe prédite par
ce modèle. Ce serait une faute : la cible deviendrait une fonction des variables
explicatives, et le modèle aval réapprendrait sa propre sortie. C'est une forme de
circularité, cousine de la fuite de modèle amont qu'on reproche à `Churn Score`.

On utilise donc le résultat au niveau AGRÉGÉ pour trancher le bloc des « 3 », et on
conserve la prédiction individuelle comme simple INDICATEUR MÉTIER (« ce passif a un
profil de détracteur »), jamais comme étiquette d'entraînement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from .features import colonnes_modele, pipeline_preparation

ORDRE = ["Detracteur", "Passif", "Promoteur"]


def appliquer_mapping(satisfaction: pd.Series, mapping: dict) -> pd.Series:
    return satisfaction.map(mapping).astype("category").cat.set_categories(ORDRE, ordered=True)


def nps(labels: pd.Series) -> float:
    """Score NPS = %Promoteurs - %Détracteurs, en points."""
    p = (labels == "Promoteur").mean()
    d = (labels == "Detracteur").mean()
    return round((p - d) * 100, 1)


def arbitrer_les_trois(df: pd.DataFrame, cfg: dict, seed: int = 42) -> dict:
    """Entraîne sur les extrêmes non ambigus, classe les « 3 », mesure la bascule."""
    num, cat = colonnes_modele(df, cfg)
    satisfaction = df[cfg["colonnes"]["cible"]]

    extremes = satisfaction.isin([1, 2, 4, 5])
    ambigus = satisfaction == 3

    X_ext = df.loc[extremes, num + cat]
    y_ext = satisfaction[extremes].map(cfg["cible"]["m3_extremes"])
    X_amb = df.loc[ambigus, num + cat]

    modele = Pipeline([
        ("preparation", pipeline_preparation(num, cat, normaliser=False)),
        ("modele", HistGradientBoostingClassifier(random_state=seed,
                                                  class_weight="balanced")),
    ])

    # Contrôle : le modèle d'arbitrage sépare-t-il vraiment les extrêmes ?
    pred_cv = cross_val_predict(modele, X_ext, y_ext, cv=5)
    exactitude_extremes = float((pred_cv == y_ext).mean())

    modele.fit(X_ext, y_ext)
    pred_amb = modele.predict(X_amb)
    proba_amb = modele.predict_proba(X_amb)
    i_det = list(modele.classes_).index("Detracteur")

    part_detracteur = float((pred_amb == "Detracteur").mean())

    return {
        "n_extremes": int(extremes.sum()),
        "n_ambigus": int(ambigus.sum()),
        "exactitude_sur_extremes": round(exactitude_extremes, 4),
        "part_3_vers_detracteur": round(part_detracteur, 4),
        "part_3_vers_promoteur": round(1 - part_detracteur, 4),
        # Indicateur métier individuel — jamais utilisé comme étiquette.
        "index_ambigus": df.index[ambigus],
        "proba_detracteur_des_3": proba_amb[:, i_det],
        "modele": modele,
    }


def construire_cibles(df: pd.DataFrame, cfg: dict, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Retourne un DataFrame des trois cibles et le diagnostic d'arbitrage."""
    satisfaction = df[cfg["colonnes"]["cible"]]
    cibles = pd.DataFrame(index=df.index)
    cibles["M1"] = appliquer_mapping(satisfaction, cfg["cible"]["m1"])
    cibles["M2"] = appliquer_mapping(satisfaction, cfg["cible"]["m2"])

    arb = arbitrer_les_trois(df, cfg, seed)

    # Décision de bloc, prise par la mesure : si la majorité des « 3 » ressemble à des
    # détracteurs, le bloc rejoint les détracteurs ; sinon il reste passif.
    destination_des_3 = "Detracteur" if arb["part_3_vers_detracteur"] > 0.5 else "Passif"
    mapping_m3 = dict(cfg["cible"]["m3_extremes"])
    mapping_m3[3] = destination_des_3
    cibles["M3"] = appliquer_mapping(satisfaction, mapping_m3)
    arb["destination_des_3"] = destination_des_3

    # Indicateur métier : parmi les clients à satisfaction 3, lesquels ont un profil
    # de détracteur ? Utilisable dans la couche de décision, pas à l'entraînement.
    cibles["profil_detracteur_si_ambigu"] = np.nan
    cibles.loc[arb["index_ambigus"], "profil_detracteur_si_ambigu"] = arb["proba_detracteur_des_3"]

    arb["nps"] = {m: nps(cibles[m]) for m in ("M1", "M2", "M3")}
    arb["distributions"] = {
        m: cibles[m].value_counts(normalize=True).reindex(ORDRE).round(4).to_dict()
        for m in ("M1", "M2", "M3")
    }
    return cibles, arb
