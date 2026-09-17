"""Étapes 17 à 24 — baseline, ordinal, boosting, calibration.

Trois familles, comme l'exige le § 4.5 de l'énoncé :
  - baseline   régression logistique multinomiale (le plancher honnête)
  - ordinale   régression sur 1-5 puis seuils optimisés (respecte l'ordre des classes)
  - boosting   HistGradientBoosting (livré avec scikit-learn, premier sur neuf
               algorithmes chez Elero et al. 2026 pour une cible NPS à trois classes)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline

from .features import colonnes_modele, pipeline_preparation
from .target import ORDRE


class OrdinalParSeuils(BaseEstimator, ClassifierMixin):
    """Régression sur la satisfaction 1-5, puis découpage en 3 classes par deux
    seuils optimisés sur le kappa quadratique pondéré.

    Pourquoi : un classifieur nominal traite une confusion Promoteur/Détracteur
    comme équivalente à Promoteur/Passif. C'est faux métier. Ici l'ordre est
    respecté par construction.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        # y est la classe ; on a besoin du score continu sous-jacent, passé via y_brut.
        self.classes_ = np.array(ORDRE)
        self.regresseur_ = HistGradientBoostingRegressor(random_state=self.seed)
        self.regresseur_.fit(X, self._y_continu, sample_weight=sample_weight)
        scores = self.regresseur_.predict(X)

        meilleur, self.seuils_ = -1.0, (2.5, 4.5)
        grille = np.quantile(scores, np.linspace(0.05, 0.95, 40))
        for s1 in grille:
            for s2 in grille[grille > s1]:
                pred = self._decouper(scores, (s1, s2))
                k = cohen_kappa_score(y, pred, weights="quadratic", labels=ORDRE)
                if k > meilleur:
                    meilleur, self.seuils_ = k, (s1, s2)
        self.kappa_entrainement_ = meilleur
        return self

    def _decouper(self, scores, seuils):
        s1, s2 = seuils
        return np.where(scores < s1, ORDRE[0], np.where(scores < s2, ORDRE[1], ORDRE[2]))

    def predict(self, X):
        return self._decouper(self.regresseur_.predict(X), self.seuils_)

    def predict_proba(self, X):
        """Probabilités approchées par la distance aux seuils : suffisant pour
        classer, insuffisant pour une décision fine — d'où la calibration."""
        scores = self.regresseur_.predict(X)
        s1, s2 = self.seuils_
        largeur = max((s2 - s1), 1e-6)
        p_det = 1 / (1 + np.exp((scores - s1) / (largeur / 4)))
        p_pro = 1 / (1 + np.exp(-(scores - s2) / (largeur / 4)))
        p_pas = np.clip(1 - p_det - p_pro, 1e-6, None)
        p = np.vstack([p_det, p_pas, p_pro]).T
        return p / p.sum(axis=1, keepdims=True)


def construire_modeles(num, cat, seed: int) -> dict:
    """Les trois familles, chacune avec son préprocessing adapté.

    `normaliser=True` pour le modèle linéaire (indispensable), `False` pour les
    arbres (inutile).
    """
    return {
        "baseline_logistique": Pipeline([
            ("preparation", pipeline_preparation(num, cat, normaliser=True)),
            ("modele", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=seed)),
        ]),
        "ordinal_seuils": Pipeline([
            ("preparation", pipeline_preparation(num, cat, normaliser=False)),
            ("modele", OrdinalParSeuils(seed=seed)),
        ]),
        "boosting_hgb": Pipeline([
            ("preparation", pipeline_preparation(num, cat, normaliser=False)),
            ("modele", HistGradientBoostingClassifier(random_state=seed,
                                                      class_weight="balanced",
                                                      max_iter=300,
                                                      learning_rate=0.08)),
        ]),
    }


def espace_recherche(nom: str) -> dict | None:
    """Étape 22 — espaces d'hyperparamètres. Modestes à dessein : ~1 050 lignes
    d'entraînement, le risque de surajuster la recherche dépasse vite le gain."""
    if nom == "baseline_logistique":
        return {"modele__C": np.logspace(-2.5, 1.5, 12)}
    if nom == "boosting_hgb":
        return {"modele__learning_rate": [0.03, 0.05, 0.08, 0.12],
                "modele__max_iter": [100, 200, 300, 500],
                "modele__max_leaf_nodes": [7, 15, 31],
                "modele__l2_regularization": [0.0, 0.5, 2.0],
                "modele__min_samples_leaf": [10, 20, 40]}
    return None  # ordinal_seuils : sa recherche de seuils interne fait déjà office de réglage


def entrainer(modele, X, y, y_continu=None, poids=None):
    """Entraîne en passant la cible continue au modèle ordinal si besoin."""
    etape = modele.named_steps["modele"]
    if isinstance(etape, OrdinalParSeuils):
        etape._y_continu = np.asarray(y_continu, dtype=float)
    modele.fit(X, y)
    return modele


def calibrer(modele, X, y, seed: int, methode: str = "sigmoid"):
    """Étape 24 — recalibration. Les probabilités servent à prioriser des appels :
    un modèle mal calibré fait mal allouer le budget.

    Platt (sigmoid) plutôt qu'isotonique : sur ~1 000 répondants répartis en 3 plis,
    l'isotonique a écrasé la classe intermédiaire (rappel Passif = 0). Platt est
    paramétrique, donc bien plus stable sur de petits effectifs. Retourne None en
    cas d'échec — l'appelant décide, rien n'est masqué.
    """
    try:
        cal = CalibratedClassifierCV(modele, method=methode, cv=3)
        cal.fit(X, y)
        return cal
    except Exception:
        return None
