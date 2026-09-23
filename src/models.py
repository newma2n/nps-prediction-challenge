"""Étapes 17 à 24 — catalogue des modèles, espaces d'hyperparamètres, calibration.

Quinze modèles répartis en sept familles. Chacun est là pour une raison écrite (colonne
`pourquoi`) et teste une hypothèse nommée (colonne `hypothese`) : on ne compare pas des
algorithmes pour décorer un tableau, on compare des hypothèses sur les données.

Les trois formulations de l'énoncé (§ 2) sont toutes représentées :
  - classification nominale       logistique, arbres, boosting, KNN, SVM, MLP, NB
  - classification ordinale       logistique ordinale à seuils (mord, all-threshold)
  - régression 1-5 puis seuillage ridge_seuils, hgb_seuils

CORAL / CORN (cités par l'énoncé) sont des architectures d'apprentissage profond pour cibles
ordinales : sur ~1 000 lignes d'entraînement tabulaires, elles n'ont pas de sens ; écartées avec
argument, pas ignorées. Les modèles de fondation (TabPFN, TabICL) sont évalués à part
(`src/fondation_eval.py`) : ils ne s'entraînent pas au sens classique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (AdaBoostClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .features import pipeline_preparation
from .target import ORDRE

_IDX = {c: i for i, c in enumerate(ORDRE)}


def _vers_entiers(y) -> np.ndarray:
    return np.array([_IDX[str(v)] for v in np.asarray(y)])


def _proba_3_classes(p, classes_entieres) -> np.ndarray:
    """Réaligne une matrice de probabilités sur les trois classes de ORDRE, même si une classe
    était absente de l'entraînement (colonne à zéro)."""
    p = np.asarray(p); out = np.zeros((p.shape[0], len(ORDRE)))
    for k, c in enumerate(np.asarray(classes_entieres).ravel()):
        out[:, int(c)] = p[:, k]
    return out


# =============================================================================
# Formulation 3 — régression sur la note 1-5, puis deux seuils optimisés sur le kappa
# =============================================================================
class OrdinalParSeuils(BaseEstimator, ClassifierMixin):
    """Régression sur la satisfaction 1-5 (si fournie via `y_continu`, sinon sur l'indice de
    classe 0-2), puis découpage en 3 classes par deux seuils choisis pour maximiser le kappa
    quadratique pondéré sur l'entraînement.

    Pourquoi : un classifieur nominal traite la confusion Promoteur/Détracteur comme équivalente
    à Promoteur/Passif. C'est faux métier. Ici l'ordre est respecté par construction, et la note
    fine 1-5 — plus riche que 3 classes — est exploitée à l'entraînement.
    """

    def __init__(self, regresseur: str = "hgb", seed: int = 42, equilibrer: bool = True):
        self.regresseur = regresseur
        self.seed = seed
        self.equilibrer = equilibrer

    def fit(self, X, y, y_continu=None, sample_weight=None):
        self.classes_ = np.array(ORDRE)
        cible = np.asarray(y_continu, dtype=float) if y_continu is not None else _vers_entiers(y).astype(float)
        # Pondération de classes « balanced » sur la régression : chaque classe pèse autant dans
        # l'ajustement du score latent, sinon la classe rare (Passif) ne pèse rien sur les seuils.
        w = compute_sample_weight("balanced", _vers_entiers(y)) if self.equilibrer else None
        if sample_weight is not None:
            w = np.asarray(sample_weight) if w is None else w * np.asarray(sample_weight)
        sample_weight = w
        if self.regresseur == "ridge":
            self.regresseur_ = Ridge(alpha=1.0)
        else:
            self.regresseur_ = HistGradientBoostingRegressor(random_state=self.seed, max_iter=200,
                                                             learning_rate=0.06, min_samples_leaf=20)
        self.regresseur_.fit(X, cible, sample_weight=sample_weight)
        scores = self.regresseur_.predict(X)

        y_cls = np.asarray(y).astype(str)
        meilleur, self.seuils_ = -1.0, (np.quantile(scores, 0.3), np.quantile(scores, 0.6))
        grille = np.quantile(scores, np.linspace(0.05, 0.95, 40))
        for s1 in grille:
            for s2 in grille[grille > s1]:
                k = cohen_kappa_score(y_cls, self._decouper(scores, (s1, s2)), weights="quadratic", labels=ORDRE)
                if k > meilleur:
                    meilleur, self.seuils_ = k, (float(s1), float(s2))
        self.kappa_entrainement_ = meilleur
        return self

    def _decouper(self, scores, seuils):
        s1, s2 = seuils
        return np.where(scores < s1, ORDRE[0], np.where(scores < s2, ORDRE[1], ORDRE[2]))

    def predict(self, X):
        return self._decouper(self.regresseur_.predict(X), self.seuils_)

    def predict_proba(self, X):
        """Probabilités approchées par la distance aux seuils. Suffisant pour classer des
        clients entre eux ; insuffisant pour lire une fréquence — d'où la calibration."""
        scores = self.regresseur_.predict(X)
        s1, s2 = self.seuils_
        largeur = max(s2 - s1, 1e-6)
        p_det = 1 / (1 + np.exp((scores - s1) / (largeur / 4)))
        p_pro = 1 / (1 + np.exp(-(scores - s2) / (largeur / 4)))
        p_pas = np.clip(1 - p_det - p_pro, 1e-6, None)
        p = np.vstack([p_det, p_pas, p_pro]).T
        return p / p.sum(axis=1, keepdims=True)


# =============================================================================
# Formulation 2 — logistique ordinale à seuils (mord, all-threshold)
# =============================================================================
class LogistiqueOrdinale(BaseEstimator, ClassifierMixin):
    """Enveloppe scikit-learn autour de `mord.LogisticAT`.

    Un seul vecteur de coefficients et deux seuils sur une variable latente : trois fois moins de
    paramètres que la logistique multinomiale, et l'ordre Détracteur < Passif < Promoteur est
    imposé par la structure du modèle, pas espéré.
    """

    def __init__(self, alpha: float = 1.0, equilibrer: bool = True):
        self.alpha = alpha
        self.equilibrer = equilibrer

    def fit(self, X, y, sample_weight=None):
        from mord import LogisticAT
        self.classes_ = np.array(ORDRE)
        self.modele_ = LogisticAT(alpha=self.alpha, max_iter=5000)
        yi = _vers_entiers(y)
        # Pondération de classes « balanced », comme les autres modèles du catalogue : sans elle,
        # la classe Passif (7 % des répondants sous MNAR) est écrasée par construction.
        w = compute_sample_weight("balanced", yi) if self.equilibrer else None
        if sample_weight is not None:
            w = np.asarray(sample_weight) if w is None else w * np.asarray(sample_weight)
        try:
            self.modele_.fit(X, yi, sample_weight=w)
        except TypeError:
            self.modele_.fit(X, yi)
        return self

    def predict(self, X):
        return self.classes_[self.modele_.predict(X).astype(int)]

    def predict_proba(self, X):
        p = self.modele_.predict_proba(X)
        if p.shape[1] < 3:  # une classe absente de l'entraînement : on complète par zéro
            plein = np.zeros((p.shape[0], 3)); plein[:, :p.shape[1]] = p; p = plein
        return p / p.sum(axis=1, keepdims=True)

    @property
    def coef_(self):
        return np.vstack([-self.modele_.coef_, np.zeros_like(self.modele_.coef_), self.modele_.coef_])


# =============================================================================
# Enveloppes utilitaires
# =============================================================================
class ClassifieurEntiers(BaseEstimator, ClassifierMixin):
    """Pour les librairies qui exigent des étiquettes entières (XGBoost). Ajoute la pondération
    de classes « balanced » via sample_weight, comme les autres modèles du catalogue."""

    def __init__(self, base=None, equilibrer: bool = True):
        self.base = base
        self.equilibrer = equilibrer

    def fit(self, X, y, sample_weight=None):
        from sklearn.base import clone
        self.classes_ = np.array(ORDRE)
        yi = _vers_entiers(y)
        w = compute_sample_weight("balanced", yi) if self.equilibrer else None
        if sample_weight is not None:
            w = sample_weight if w is None else w * sample_weight
        self.modele_ = clone(self.base).fit(X, yi, sample_weight=w)
        return self

    def predict(self, X):
        return self.classes_[self.modele_.predict(X).astype(int)]

    def predict_proba(self, X):
        return _proba_3_classes(self.modele_.predict_proba(X), self.modele_.classes_)

    # SHAP TreeExplainer lit le modèle sous-jacent.
    @property
    def modele_arbre(self):
        return self.modele_


class PreparationCatBoost(BaseEstimator, TransformerMixin):
    """Préparation minimale pour CatBoost : imputation médiane des numériques, catégorielles
    laissées en chaînes (imputées par une modalité explicite). Aucun encodage : c'est CatBoost qui
    encode les catégorielles (statistiques de cible ordonnées) — c'est l'hypothèse qu'on teste."""

    def __init__(self, num, cat):
        self.num = num
        self.cat = cat

    def fit(self, X, y=None):
        self.medianes_ = X[self.num].median()
        return self

    def transform(self, X):
        d = pd.DataFrame(index=X.index)
        for c in self.num:
            d[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.medianes_[c])
        for c in self.cat:
            d[c] = X[c].astype(object).where(X[c].notna(), "manquant").astype(str)
        return d

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.num) + list(self.cat))


class CatBoostNatif(BaseEstimator, ClassifierMixin):
    """CatBoost avec encodage natif des catégorielles, enveloppé pour être clonable par
    scikit-learn (le constructeur de CatBoost modifie `cat_features`, ce que `clone` refuse).
    Les catégorielles sont les `n_cat` dernières colonnes de la matrice préparée."""

    def __init__(self, n_cat: int = 0, iterations: int = 300, depth: int = 5, learning_rate: float = 0.05,
                 l2_leaf_reg: float = 3.0, seed: int = 42):
        self.n_cat = n_cat
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        from catboost import CatBoostClassifier
        self.classes_ = np.array(ORDRE)
        n_col = X.shape[1]
        self.modele_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                                          l2_leaf_reg=self.l2_leaf_reg, auto_class_weights="Balanced", random_seed=self.seed,
                                          verbose=0, allow_writing_files=False, thread_count=4,
                                          cat_features=list(range(n_col - self.n_cat, n_col)))
        self.modele_.fit(X, _vers_entiers(y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.modele_.predict(X)).ravel().astype(int)]

    def predict_proba(self, X):
        return _proba_3_classes(self.modele_.predict_proba(X), self.modele_.classes_)


# =============================================================================
# Le catalogue — quinze modèles, sept familles, une raison chacun
# =============================================================================
CATALOGUE: list[dict] = [
    dict(nom="logistique", famille="Linéaire", formulation="classification nominale",
         pourquoi="la baseline exigée par l'énoncé (§ 4.5) ; ses coefficients sont une explication exacte et additive",
         hypothese="les frontières entre classes sont proches du linéaire dans l'espace des variables standardisées",
         reference="Chong et al. 2023 (LR 79,2 % sur ce dataset) ; Kannan et al. 2022"),
    dict(nom="logistique_ordinale", famille="Linéaire", formulation="classification ordinale",
         pourquoi="la formulation ordinale demandée au § 2 : un seul vecteur de coefficients et deux seuils (mord, all-threshold), trois fois moins de paramètres que la multinomiale",
         hypothese="l'ordre Détracteur < Passif < Promoteur est porté par une seule direction latente",
         reference="énoncé § 5 (mord) ; Agresti, Categorical Data Analysis"),
    dict(nom="ridge_seuils", famille="Régression + seuils", formulation="régression 1-5 puis seuillage",
         pourquoi="la troisième formulation de l'énoncé, en version linéaire : régression ridge sur la note 1-5, puis deux seuils optimisés sur le kappa quadratique",
         hypothese="la note fine 1-5 contient plus d'information que les trois classes et mérite d'être exploitée à l'entraînement",
         reference="énoncé § 2 (« regression with thresholding »)"),
    dict(nom="hgb_seuils", famille="Régression + seuils", formulation="régression 1-5 puis seuillage",
         pourquoi="même formulation avec un régresseur non linéaire (gradient boosting d'histogrammes)",
         hypothese="la relation variables → note comporte des interactions qu'un modèle linéaire ne capte pas",
         reference="énoncé § 2"),
    dict(nom="arbre_decision", famille="Arbres et bagging", formulation="classification nominale",
         pourquoi="des règles lisibles par un non-technicien ; le modèle de Kannan et al. et le CART de Mustafa et al. ; témoin de surapprentissage d'un arbre seul",
         hypothese="quelques règles contractuelles (contrat, ancienneté, parrainage) suffisent à séparer les classes",
         reference="Kannan et al. 2022 ; Mustafa et al. 2021"),
    dict(nom="foret_aleatoire", famille="Arbres et bagging", formulation="classification nominale",
         pourquoi="l'ensemble d'arbres « calibré » cité par l'énoncé ; co-premier chez Elero et al. sur une cible NPS à trois classes ; robuste avec peu de réglage",
         hypothese="le bagging stabilise les arbres sur un petit échantillon bruité mieux que le boosting ne le fait",
         reference="Elero et al. 2026 ; Chong et al. 2023"),
    dict(nom="hist_gradient_boosting", famille="Boosting", formulation="classification nominale",
         pourquoi="le boosting natif de scikit-learn, sans dépendance ; premier sur neuf algorithmes chez Elero et al. (modèle 3, trois classes)",
         hypothese="des interactions et non-linéarités sont apprenables avec ~1 000 lignes",
         reference="Elero et al. 2026"),
    dict(nom="lightgbm", famille="Boosting", formulation="classification nominale",
         pourquoi="cité par l'énoncé ; croissance feuille par feuille et régularisation fine (min_child_samples) — le boosting le plus déployé en production tabulaire",
         hypothese="même famille que HGB : l'écart entre les deux mesure la variance d'implémentation, pas une différence de méthode",
         reference="énoncé § 4.5 ; Ke et al. 2017"),
    dict(nom="xgboost", famille="Boosting", formulation="classification nominale",
         pourquoi="cité par l'énoncé ; le boosting de référence de la littérature churn (meilleur modèle chez Chong et al., 79,7 %)",
         hypothese="idem LightGBM ; régularisation L2 des feuilles utile sur petit échantillon",
         reference="Chong et al. 2023 ; Chen & Guestrin 2016"),
    dict(nom="catboost", famille="Boosting", formulation="classification nominale",
         pourquoi="cité par l'énoncé ; encode nativement les catégorielles par statistiques de cible ordonnées — le seul modèle du catalogue qui ne passe pas par le one-hot",
         hypothese="le one-hot fait perdre de l'information sur `Offer`, `Contract`, `Payment Method` ; un encodage cible ordonné fait mieux sur petit échantillon",
         reference="énoncé § 4.5 ; Prokhorenkova et al. 2018"),
    dict(nom="adaboost", famille="Boosting", formulation="classification nominale",
         pourquoi="le boosting historique ; meilleur modèle chez Lalwani et al. (81,7 %, cité par Chong) ; teste un boosting d'arbres courts",
         hypothese="repondérer les erreurs suffit, sans gradient ni régularisation",
         reference="Chong et al. 2023 (revue) ; Elero et al. 2026"),
    dict(nom="knn", famille="Voisinage et noyaux", formulation="classification nominale",
         pourquoi="présent dans les quatre études de référence ; « les clients qui se ressemblent se notent pareil » ; standardisé, sinon dominé par les montants",
         hypothese="la structure locale (voisinage) porte l'information ; si KNN égale les arbres, des « jumeaux » existent dans les données — signal de fuite",
         reference="Chong et al. 2023 ; Elero et al. 2026 ; Mustafa et al. 2021"),
    dict(nom="svm_rbf", famille="Voisinage et noyaux", formulation="classification nominale",
         pourquoi="noyau gaussien, marge maximale ; présent chez Elero, Chong, Mustafa ; probabilités par Platt interne",
         hypothese="la marge maximale est plus robuste au bruit d'étiquettes que les modèles qui ajustent chaque point",
         reference="Elero et al. 2026 ; Chong et al. 2023"),
    dict(nom="mlp", famille="Réseau de neurones", formulation="classification nominale",
         pourquoi="le meilleur modèle de Kannan et al. (F 0,876) et l'ANN d'Elero et al. ; deux couches, arrêt précoce",
         hypothese="une représentation apprise bat les variables construites à la main — attendu faux sur ~1 000 lignes, inclus pour trancher, pas pour décorer",
         reference="Kannan et al. 2022 ; Elero et al. 2026"),
    dict(nom="naive_bayes", famille="Probabiliste", formulation="classification nominale",
         pourquoi="le GaussianNB de Mustafa et al. ; coût nul ; témoin bas",
         hypothese="l'indépendance conditionnelle des variables — fausse ici (colinéarité mesurée en phase 3), ce qui rend sa contre-performance informative",
         reference="Mustafa et al. 2021"),
]
NOMS = [m["nom"] for m in CATALOGUE]

# Ordre de simplicité (1 = le plus simple), fixé A PRIORI pour la règle de parcimonie de la
# sélection. Trois critères, dans cet ordre : (a) nombre de paramètres libres ajustés sur les
# données ; (b) le modèle produit-il nativement des probabilités de classe — la liste d'appels et
# l'application en dépendent ; (c) l'explication est-elle exacte et lisible (coefficients) ou
# approchée et coûteuse (SHAP, permutation).
SIMPLICITE = {"logistique": 1, "logistique_ordinale": 1, "naive_bayes": 1, "ridge_seuils": 2, "arbre_decision": 2,
              "knn": 3, "hgb_seuils": 4, "foret_aleatoire": 4, "adaboost": 4, "hist_gradient_boosting": 5,
              "lightgbm": 5, "xgboost": 5, "catboost": 5, "svm_rbf": 5, "mlp": 6}

JUSTIFICATION_SIMPLICITE = {
    1: "un vecteur de coefficients ajusté directement, probabilités natives, explication exacte et additive",
    2: "coefficients simples mais une étape d'ajustement supplémentaire (deux seuils optimisés a posteriori sur le kappa) ou une structure d'arbre à lire",
    3: "aucun paramètre appris mais aucune explication intrinsèque : la prédiction dépend de l'échantillon entier",
    4: "ensemble d'arbres (centaines de règles) ; explication par SHAP, approchée et coûteuse",
    5: "ensemble d'arbres régularisé à nombreux hyperparamètres, ou noyau non linéaire sans explication native",
    6: "réseau de neurones : représentation apprise, aucune lecture directe des poids",
}
FAMILLE = {m["nom"]: m["famille"] for m in CATALOGUE}
FORMULATION = {m["nom"]: m["formulation"] for m in CATALOGUE}

# Pondération de classes « balanced » : appliquée partout où l'estimateur l'accepte (class_weight ou
# sample_weight). Deux exceptions structurelles, assumées et notées dans l'étude : KNN (pas de notion
# de poids d'entraînement — `weights='distance'` pondère les voisins, pas les classes) et Naive Bayes
# (les a priori de classe sont estimés sur l'échantillon ; les forcer uniformes reviendrait à `priors=`,
# testé sans effet notable).
SANS_PONDERATION = {"knn", "naive_bayes"}

# Familles qui exigent des variables standardisées
_NORMALISER = {"logistique", "logistique_ordinale", "ridge_seuils", "knn", "svm_rbf", "mlp", "naive_bayes"}


def catalogue_df() -> pd.DataFrame:
    return pd.DataFrame(CATALOGUE)[["famille", "nom", "formulation", "pourquoi", "hypothese", "reference"]]


def construire_modeles(num, cat, seed: int, noms: list[str] | None = None) -> dict:
    """Instancie les modèles demandés (tous par défaut), chacun avec le préprocessing adapté :
    standardisation pour les modèles à distance ou linéaires, rien pour les arbres, préparation
    sans encodage pour CatBoost."""
    def prep(nom):
        if nom == "catboost":
            return PreparationCatBoost(num, cat)
        return pipeline_preparation(num, cat, normaliser=nom in _NORMALISER)

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    estimateurs = {
        "logistique": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "logistique_ordinale": LogistiqueOrdinale(alpha=1.0),
        "ridge_seuils": OrdinalParSeuils(regresseur="ridge", seed=seed),
        "hgb_seuils": OrdinalParSeuils(regresseur="hgb", seed=seed),
        "arbre_decision": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=seed),
        "foret_aleatoire": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                                  class_weight="balanced_subsample", random_state=seed, n_jobs=4),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                                 class_weight="balanced", random_state=seed),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                                   random_state=seed, verbose=-1, n_jobs=4),
        "xgboost": ClassifieurEntiers(XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                    objective="multi:softprob", random_state=seed, verbosity=0, n_jobs=4)),
        "catboost": CatBoostNatif(n_cat=len(cat), seed=seed),
        "adaboost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced"),
                                       n_estimators=200, learning_rate=0.5, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "svm_rbf": SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=600,
                             random_state=seed),
        "naive_bayes": GaussianNB(),
    }
    noms = noms or NOMS
    return {n: Pipeline([("preparation", prep(n)), ("modele", estimateurs[n])]) for n in noms}


def espace_recherche(nom: str) -> dict | None:
    """Étape 22 — espaces d'hyperparamètres. Modestes à dessein : ~1 000 lignes d'entraînement,
    le risque de surajuster la recherche dépasse vite le gain. `None` = pas de réglage (les
    modèles à seuils s'optimisent en interne, Naive Bayes n'a rien à régler)."""
    E = {
        "logistique": {"modele__C": np.logspace(-2.5, 1.5, 12)},
        "logistique_ordinale": {"modele__alpha": np.logspace(-2, 2.5, 12)},
        "arbre_decision": {"modele__max_depth": [3, 4, 6, 8, None], "modele__min_samples_leaf": [5, 10, 20, 40]},
        "foret_aleatoire": {"modele__n_estimators": [300, 600], "modele__max_depth": [None, 6, 10, 16],
                            "modele__min_samples_leaf": [1, 3, 5, 10], "modele__max_features": ["sqrt", 0.3, 0.5]},
        "hist_gradient_boosting": {"modele__learning_rate": [0.03, 0.05, 0.08, 0.12], "modele__max_iter": [100, 200, 300, 500],
                                   "modele__max_leaf_nodes": [7, 15, 31], "modele__l2_regularization": [0.0, 0.5, 2.0],
                                   "modele__min_samples_leaf": [10, 20, 40]},
        "lightgbm": {"modele__n_estimators": [100, 200, 400], "modele__learning_rate": [0.02, 0.05, 0.1],
                     "modele__num_leaves": [7, 15, 31], "modele__min_child_samples": [10, 20, 40],
                     "modele__reg_lambda": [0.0, 1.0, 5.0]},
        "xgboost": {"modele__base__n_estimators": [100, 200, 400], "modele__base__learning_rate": [0.02, 0.05, 0.1],
                    "modele__base__max_depth": [3, 4, 6], "modele__base__min_child_weight": [1, 5, 10],
                    "modele__base__reg_lambda": [0.5, 1.0, 5.0]},
        "catboost": {"modele__iterations": [200, 400], "modele__depth": [4, 6], "modele__learning_rate": [0.03, 0.06, 0.1],
                     "modele__l2_leaf_reg": [1.0, 3.0, 10.0]},
        "adaboost": {"modele__n_estimators": [100, 200, 400], "modele__learning_rate": [0.3, 0.6, 1.0]},
        "knn": {"modele__n_neighbors": [11, 25, 51, 101], "modele__weights": ["uniform", "distance"]},
        "svm_rbf": {"modele__C": [0.3, 1.0, 3.0, 10.0], "modele__gamma": ["scale", 0.01, 0.03]},
        "mlp": {"modele__hidden_layer_sizes": [(64,), (64, 32), (128, 64)], "modele__alpha": [1e-4, 1e-3, 1e-2],
                "modele__learning_rate_init": [1e-3, 3e-3]},
    }
    return E.get(nom)


def budget_recherche(nom: str) -> int:
    """Nombre de configurations tirées : moins pour les modèles lents."""
    return {"catboost": 6, "svm_rbf": 8, "mlp": 8, "adaboost": 8, "foret_aleatoire": 10}.get(nom, 12)


def entrainer(modele: Pipeline, X, y, y_continu=None, poids=None):
    """Entraîne en passant la note continue aux modèles à seuils et les poids à ceux qui les
    acceptent. Rien n'est masqué : une erreur remonte."""
    params = {}
    if isinstance(modele.named_steps["modele"], OrdinalParSeuils) and y_continu is not None:
        params["modele__y_continu"] = np.asarray(y_continu, dtype=float)
    if poids is not None:
        params["modele__sample_weight"] = np.asarray(poids, dtype=float)
    modele.fit(X, y, **params)
    return modele


def calibrer(modele, X, y, seed: int, methode: str = "sigmoid"):
    """Étape 24 — recalibration. Les probabilités servent à prioriser des appels : un modèle mal
    calibré fait mal allouer le budget.

    Platt (sigmoid) plutôt qu'isotonique : sur ~1 000 répondants répartis en 3 plis,
    l'isotonique a écrasé la classe intermédiaire (rappel Passif = 0). Platt est paramétrique,
    donc plus stable sur de petits effectifs. Retourne None en cas d'échec — l'appelant décide.
    """
    try:
        cal = CalibratedClassifierCV(modele, method=methode, cv=3)
        cal.fit(X, y)
        return cal
    except Exception:
        return None


def modele_pour_shap(pipe: Pipeline):
    """Retourne (objet à expliquer, type) : 'lineaire' (coefficients), 'arbre' (TreeExplainer)
    ou 'boite_noire' (explication par permutation)."""
    m = pipe.named_steps["modele"]
    if isinstance(m, ClassifieurEntiers):
        m = m.modele_
    if isinstance(m, (LogisticRegression, LogistiqueOrdinale)):
        return m, "lineaire"
    nom = type(m).__name__
    if nom in ("RandomForestClassifier", "DecisionTreeClassifier", "HistGradientBoostingClassifier",
               "LGBMClassifier", "XGBClassifier", "CatBoostClassifier"):
        return m, "arbre"
    return m, "boite_noire"
