"""Étape 16 — protocole de validation 15 % répondants / 85 % silencieux.

Un train_test_split stratifié raterait l'intention de l'énoncé : les répondants à une
enquête NPS ne sont pas un échantillon aléatoire. On simule donc trois mécanismes de
non-réponse, du plus optimiste au plus réaliste.

  S1  MCAR  15 % tirés au hasard                          -> borne haute optimiste
  S2  MAR   propension = f(covariables observables)        -> biais de covariables seul
  S3  MNAR  propension = f(covariables, SATISFACTION)      -> le cas réaliste

L'écart de performance entre S1 et S3 est le résultat scientifique du projet : il
mesure ce qu'aucun praticien ne peut mesurer en production, puisqu'ici on connaît la
satisfaction des 100 % des clients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _centrer_reduire(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").astype(float).values
    v = np.nan_to_num(v, nan=np.nanmedian(v))
    ecart = v.std() or 1.0
    return (v - v.mean()) / ecart


def propensions(df: pd.DataFrame, cfg: dict, scenario: str, seed: int) -> np.ndarray:
    """Probabilité, pour chaque client, de répondre à l'enquête."""
    rng = np.random.default_rng(seed)
    n = len(df)

    if scenario == "S1_MCAR":
        return np.full(n, cfg["protocole"]["taux_reponse"])

    # Covariables observables plausiblement liées à la propension à répondre :
    # ancienneté (un client installé répond plus), facturation dématérialisée
    # (habitude du canal numérique), engagement contractuel.
    score = (
        0.45 * _centrer_reduire(df["Tenure in Months"])
        + 0.35 * (df["Paperless Billing"].astype(str).str.lower() == "yes").astype(float).values
        + 0.25 * (df["Contract"].astype(str) != "Month-to-Month").astype(float).values
    )
    score *= cfg["protocole"]["force_mar"]

    if scenario == "S3_MNAR":
        # Le cœur du MNAR : les très mécontents et les très contents répondent
        # davantage que les indifférents. La propension dépend donc de la
        # satisfaction elle-même, qui est précisément ce qu'on cherche à prédire.
        satisfaction = pd.to_numeric(df[cfg["colonnes"]["cible"]]).values
        extremite = np.abs(satisfaction - 3.0)  # 0 au centre, 2 aux extrêmes
        score += cfg["protocole"]["force_mnar"] * _centrer_reduire(pd.Series(extremite))

    # Calage : on ajuste l'ordonnée à l'origine pour atteindre le taux de réponse visé.
    cible = cfg["protocole"]["taux_reponse"]
    bas, haut = -20.0, 20.0
    for _ in range(80):
        milieu = (bas + haut) / 2
        if _sigmoide(score + milieu).mean() > cible:
            haut = milieu
        else:
            bas = milieu
    p = _sigmoide(score + (bas + haut) / 2)
    return np.clip(p, 1e-4, 1 - 1e-4)


def decouper(df: pd.DataFrame, cfg: dict, scenario: str, seed: int) -> dict:
    """Tire les répondants selon le scénario. Retourne masques et poids IPW."""
    rng = np.random.default_rng(seed)
    p = propensions(df, cfg, scenario, seed)
    repondants = rng.random(len(df)) < p

    # Garde-fou : on veut un échantillon de répondants exploitable.
    if repondants.sum() < 300:
        idx = np.argsort(-p)[:int(cfg["protocole"]["taux_reponse"] * len(df))]
        repondants = np.zeros(len(df), dtype=bool)
        repondants[idx] = True

    # Repondération par l'inverse de la propension : chaque répondant représente
    # 1/p clients de la base complète.
    poids = np.where(repondants, 1.0 / p, 0.0)
    poids = poids / poids[repondants].mean()

    return {
        "scenario": scenario,
        "repondants": repondants,
        "silencieux": ~repondants,
        "poids_ipw": poids,
        "taux_reponse_obtenu": float(repondants.mean()),
    }


def diagnostic(df: pd.DataFrame, cfg: dict, decoupe: dict) -> dict:
    """Le mécanisme mord-il ? Compare la satisfaction des répondants à la base."""
    sat = pd.to_numeric(df[cfg["colonnes"]["cible"]])
    rep = decoupe["repondants"]
    return {
        "scenario": decoupe["scenario"],
        "taux_reponse": round(decoupe["taux_reponse_obtenu"], 4),
        "satisfaction_moyenne_base": round(float(sat.mean()), 3),
        "satisfaction_moyenne_repondants": round(float(sat[rep].mean()), 3),
        "ecart": round(float(sat[rep].mean() - sat.mean()), 3),
        "part_extremes_base": round(float(sat.isin([1, 2, 5]).mean()), 4),
        "part_extremes_repondants": round(float(sat[rep].isin([1, 2, 5]).mean()), 4),
    }


def estimer_propension(df: pd.DataFrame, repondants: np.ndarray, num: list[str], cat: list[str], seed: int) -> np.ndarray:
    """Propension à répondre ESTIMÉE à partir des seules covariables observables — ce qu'un
    praticien peut faire en production (il sait qui a répondu, il connaît les covariables de
    tous). Hypothèse MAR : la propension vraie sous S3 dépend aussi de la satisfaction, que cette
    estimation ne voit pas ; l'écart mesure la limite de l'IPW face au MNAR.

    Retourne des poids 1/p̂ normalisés à moyenne 1 sur les répondants (0 ailleurs).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from .features import pipeline_preparation
    modele = Pipeline([("preparation", pipeline_preparation(num, cat, normaliser=True)),
                       ("modele", LogisticRegression(max_iter=2000, C=0.5, random_state=seed))])
    modele.fit(df[num + cat], repondants.astype(int))
    p_hat = np.clip(modele.predict_proba(df[num + cat])[:, 1], 0.02, 0.98)
    poids = np.where(repondants, 1.0 / p_hat, 0.0)
    poids[repondants] /= poids[repondants].mean()
    return poids
