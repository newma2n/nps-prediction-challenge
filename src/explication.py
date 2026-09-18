"""Étape 29 — contributions individuelles à la classe Détracteur, quel que soit le modèle.

Un seul point d'entrée, utilisé par le pipeline (drivers globaux et par segment) ET par
l'application (explication d'un client) : les deux affichent donc exactement la même chose.

  - modèle linéaire (logistique, logistique ordinale, ridge + seuils) : coefficient × valeur,
    explication exacte et additive ;
  - modèle à arbres (forêt, boosting, arbre, HGB + seuils)           : SHAP TreeExplainer ;
  - autres (KNN, SVM, MLP, Naive Bayes)                                 : SHAP par permutation,
    sur un échantillon (coûteux).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import noms_apres_encodage
from .models import CatBoostNatif, ClassifieurEntiers, LogistiqueOrdinale, OrdinalParSeuils
from .target import ORDRE

_ARBRES = ("RandomForestClassifier", "DecisionTreeClassifier", "HistGradientBoostingClassifier",
           "LGBMClassifier", "XGBClassifier", "CatBoostClassifier", "HistGradientBoostingRegressor")


def genre_explication(pipe) -> str:
    m = pipe.named_steps["modele"]
    if isinstance(m, (ClassifieurEntiers, CatBoostNatif)):
        m = m.modele_
    if isinstance(m, OrdinalParSeuils):
        return "seuils_lineaire" if type(m.regresseur_).__name__ == "Ridge" else "seuils_arbre"
    if isinstance(m, LogistiqueOrdinale) or type(m).__name__ == "LogisticRegression":
        return "lineaire"
    if type(m).__name__ in _ARBRES:
        return "arbre"
    return "boite_noire"


def libelle_methode(genre: str) -> str:
    return {"lineaire": "coefficients × valeur (exact, additif)",
            "seuils_lineaire": "coefficients × valeur du score latent (signe inversé : score bas = détraction)",
            "seuils_arbre": "SHAP TreeExplainer sur le régresseur du score latent (signe inversé)",
            "arbre": "SHAP TreeExplainer", "boite_noire": "SHAP par permutation (échantillon)"}[genre]


def contributions_detracteur(pipe, X: pd.DataFrame, seed: int = 42, max_lignes: int = 1500) -> tuple[pd.DataFrame, str]:
    """Matrice (n × variables préparées) : contribution de chaque variable au score de détraction
    de chaque client. Positif = pousse vers Détracteur."""
    prep = pipe.named_steps["preparation"]
    noms = noms_apres_encodage(prep)
    Xt = prep.transform(X)
    genre = genre_explication(pipe)
    m = pipe.named_steps["modele"]
    if isinstance(m, (ClassifieurEntiers, CatBoostNatif)):
        m = m.modele_

    if genre == "lineaire":
        j = list(pipe.named_steps["modele"].classes_).index("Detracteur")
        C = np.asarray(Xt, dtype=float) * m.coef_[j]
        return pd.DataFrame(C, columns=noms, index=X.index), libelle_methode(genre)
    if genre == "seuils_lineaire":
        C = -np.asarray(Xt, dtype=float) * m.regresseur_.coef_
        return pd.DataFrame(C, columns=noms, index=X.index), libelle_methode(genre)

    import shap
    n = min(len(X), max_lignes)
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False) if len(X) > n else np.arange(len(X))
    Xs = Xt.iloc[idx] if hasattr(Xt, "iloc") else Xt[idx]
    if genre == "seuils_arbre":
        sv = shap.TreeExplainer(m.regresseur_).shap_values(Xs)
        return pd.DataFrame(-np.asarray(sv), columns=noms, index=X.index[idx]), libelle_methode(genre)
    if genre == "arbre":
        cls = list(pipe.named_steps["modele"].classes_)
        j = cls.index("Detracteur")
        sv = shap.TreeExplainer(m).shap_values(Xs)
        sv_d = sv[j] if isinstance(sv, list) else (sv[:, :, j] if np.ndim(sv) == 3 else sv)
        return pd.DataFrame(np.asarray(sv_d), columns=noms, index=X.index[idx]), libelle_methode(genre)

    # boîte noire : permutation sur un échantillon réduit
    cls = list(pipe.named_steps["modele"].classes_)
    j = cls.index("Detracteur")
    fond = Xs[:100] if not hasattr(Xs, "iloc") else Xs.iloc[:100]
    expl = shap.Explainer(lambda Z: m.predict_proba(Z)[:, j], shap.maskers.Independent(fond, max_samples=100), seed=seed)
    n2 = min(n, 400)
    Xs2 = Xs[:n2] if not hasattr(Xs, "iloc") else Xs.iloc[:n2]
    return pd.DataFrame(expl(Xs2).values, columns=noms, index=X.index[idx][:n2]), libelle_methode(genre)


def libelle_variable(nom: str, cat: list[str]) -> str:
    """`Contract_Month-to-Month` -> `Contract = Month-to-Month`."""
    for c in cat:
        if nom.startswith(c + "_"):
            return f"{c} = {nom[len(c) + 1:]}"
    return nom
