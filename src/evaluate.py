"""Étapes 25, 27 et 31 — évaluation technique, métier, et audit d'équité.

RÈGLE ABSOLUE DU PROJET : aucune métrique globale n'est produite sans son détail par
classe. Kannan et al. (2022) annoncent 0,876 de F-score en résumé alors que leur
classe minoritaire est à 0,126 — c'est exactement ce que l'énoncé sanctionne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from .target import ORDRE


def metriques(y_vrai, y_pred) -> dict:
    """Globales ET détail par classe, toujours ensemble."""
    res = {
        "exactitude": float((np.asarray(y_vrai) == np.asarray(y_pred)).mean()),
        "exactitude_equilibree": float(balanced_accuracy_score(y_vrai, y_pred)),
        "macro_f1": float(f1_score(y_vrai, y_pred, average="macro", labels=ORDRE, zero_division=0)),
        # Kappa quadratique : pénalise davantage une erreur de deux crans qu'une
        # erreur de un. C'est le comportement métier correct sur une cible ordonnée.
        "kappa_quadratique": float(cohen_kappa_score(y_vrai, y_pred, weights="quadratic",
                                                     labels=ORDRE)),
        "par_classe": {},
    }
    for i, classe in enumerate(ORDRE):
        res["par_classe"][classe] = {
            "precision": float(precision_score(y_vrai, y_pred, labels=[classe],
                                               average="macro", zero_division=0)),
            "rappel": float(recall_score(y_vrai, y_pred, labels=[classe],
                                         average="macro", zero_division=0)),
            "f1": float(f1_score(y_vrai, y_pred, labels=[classe],
                                 average="macro", zero_division=0)),
            "effectif": int((np.asarray(y_vrai) == classe).sum()),
        }
    res["matrice_confusion"] = confusion_matrix(y_vrai, y_pred, labels=ORDRE).tolist()
    return res


def precision_at_k(y_vrai, proba_detracteur, k: int) -> dict:
    """Étape 27 — la métrique métier.

    La vraie question n'est pas « quel taux de bonnes réponses » mais « sur les K
    clients que l'équipe peut appeler ce mois-ci, combien sont réellement des
    détracteurs ».
    """
    y_vrai = np.asarray(y_vrai)
    proba = np.asarray(proba_detracteur)
    k = min(k, len(y_vrai))

    ordre = np.argsort(-proba)[:k]
    vrais_detracteurs_cibles = int((y_vrai[ordre] == "Detracteur").sum())
    total_detracteurs = int((y_vrai == "Detracteur").sum())
    taux_base = total_detracteurs / len(y_vrai) if len(y_vrai) else 0.0
    precision = vrais_detracteurs_cibles / k if k else 0.0

    return {
        "k": k,
        "detracteurs_captes": vrais_detracteurs_cibles,
        "detracteurs_total": total_detracteurs,
        "precision_at_k": round(precision, 4),
        "rappel_at_k": round(vrais_detracteurs_cibles / total_detracteurs, 4)
        if total_detracteurs else 0.0,
        "taux_de_base": round(taux_base, 4),
        # Lift : combien de fois mieux que d'appeler au hasard.
        "lift": round(precision / taux_base, 2) if taux_base else 0.0,
    }


def valeur_economique(res_k: dict, cfg: dict) -> dict:
    """Traduction en euros — le langage du directeur Expérience Client."""
    m = cfg["metier"]
    cout = res_k["k"] * m["cout_appel_eur"]
    retenus = res_k["detracteurs_captes"] * m["taux_succes_appel"]
    gain = retenus * m["valeur_client_retenu_eur"]

    # Contrefactuel : la même campagne en appelant au hasard.
    captes_hasard = res_k["k"] * res_k["taux_de_base"]
    gain_hasard = captes_hasard * m["taux_succes_appel"] * m["valeur_client_retenu_eur"]

    return {
        "cout_campagne_eur": round(cout),
        "clients_retenus_estimes": round(retenus, 1),
        "gain_brut_eur": round(gain),
        "gain_net_eur": round(gain - cout),
        "gain_net_ciblage_aleatoire_eur": round(gain_hasard - cout),
        "gain_apporte_par_le_modele_eur": round(gain - gain_hasard),
        "roi": round((gain - cout) / cout, 2) if cout else 0.0,
    }


def audit_equite(df: pd.DataFrame, y_vrai, y_pred, dimensions: list[str]) -> dict:
    """Étape 31 — rappel Détracteur par sous-groupe.

    Les variables démographiques sont EXCLUES du modèle (étape 9) et servent
    uniquement ici. L'énoncé : capter 80 % des jeunes détracteurs mais 50 % des
    seniors est inacceptable, même à exactitude globale élevée.
    """
    y_vrai = pd.Series(np.asarray(y_vrai), index=df.index)
    y_pred = pd.Series(np.asarray(y_pred), index=df.index)
    resultats = {}

    for dim in dimensions:
        if dim not in df.columns:
            continue
        groupes = {}
        for valeur, sous in df.groupby(dim, observed=True):
            if len(sous) < 50:
                continue
            vrai, pred = y_vrai.loc[sous.index], y_pred.loc[sous.index]
            n_det = int((vrai == "Detracteur").sum())
            if n_det < 10:
                continue
            groupes[str(valeur)] = {
                "effectif": len(sous),
                "detracteurs": n_det,
                "rappel_detracteur": round(float(
                    recall_score(vrai, pred, labels=["Detracteur"], average="macro",
                                 zero_division=0)), 4),
            }
        if len(groupes) >= 2:
            rappels = [g["rappel_detracteur"] for g in groupes.values()]
            resultats[dim] = {
                "groupes": groupes,
                "ecart_max": round(max(rappels) - min(rappels), 4),
            }
    return resultats
