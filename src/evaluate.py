"""Étapes 25, 27 et 31 — évaluation technique, métier, équité, robustesse.

RÈGLE ABSOLUE DU PROJET : aucune métrique globale n'est produite sans son détail par classe.
Kannan et al. (2022) annoncent 0,876 de F-score en résumé alors que leur classe minoritaire est à
0,126 — c'est exactement ce que l'énoncé sanctionne (« Accuracy alone is not enough »).

Chaque métrique est là pour une raison (voir `JUSTIFICATION_METRIQUES`, repris dans l'étude).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

from .target import ORDRE

JUSTIFICATION_METRIQUES = [
    ("Kappa quadratique pondéré", "métrique principale de sélection",
     "cible ordonnée : une erreur de deux crans (Promoteur prédit Détracteur) doit coûter plus qu'une erreur d'un cran ; le kappa quadratique le fait, l'exactitude et le F1 ne le font pas ; corrigé du hasard"),
    ("Macro-F1", "départage",
     "moyenne non pondérée des F1 des trois classes : une classe minoritaire abandonnée fait chuter le score, ce que l'exactitude masque"),
    ("Exactitude équilibrée", "lecture",
     "moyenne des rappels par classe ; même logique que le macro-F1 côté rappel"),
    ("Rappel / précision / F1 par classe", "obligatoire à chaque publication",
     "la règle du projet : le Détracteur est la classe qui compte pour le métier, le Passif celle qu'un modèle abandonne en premier"),
    ("AUC un-contre-tous et AP (Détracteur)", "qualité du classement",
     "l'application classe les clients par P(Détracteur) : l'AUC et l'aire précision-rappel mesurent ce classement indépendamment du seuil ; l'AP est préférée sous déséquilibre"),
    ("Brier et ECE (Détracteur)", "qualité des probabilités",
     "les probabilités servent à allouer un budget d'appels : Brier mesure leur erreur quadratique, l'ECE l'écart entre probabilité annoncée et fréquence observée"),
    ("Précision@K, rappel@K, lift", "métrique métier",
     "la question de l'équipe rétention : « sur les K clients qu'on peut appeler ce mois-ci, combien sont réellement des détracteurs ? »"),
    ("Exactitude brute", "rapportée pour montrer qu'elle trompe",
     "prédire toujours Passif donne ~0,43 ; l'énoncé la disqualifie comme critère"),
]


def _ece(y_bin: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error, découpage en quantiles de probabilité."""
    bords = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(bords) < 3:
        return float(abs(p.mean() - y_bin.mean()))
    idx = np.clip(np.searchsorted(bords, p, side="right") - 1, 0, len(bords) - 2)
    ece = 0.0
    for b in range(len(bords) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y_bin[m].mean())
    return float(ece)


def metriques(y_vrai, y_pred, proba=None, classes=None) -> dict:
    """Globales ET détail par classe, toujours ensemble. Si `proba` (n × 3, ordre `classes`) est
    fourni, ajoute les métriques de classement et de calibration."""
    y_vrai = np.asarray(y_vrai).astype(str); y_pred = np.asarray(y_pred).astype(str)
    res = {
        "exactitude": float((y_vrai == y_pred).mean()),
        "exactitude_equilibree": float(balanced_accuracy_score(y_vrai, y_pred)),
        "macro_f1": float(f1_score(y_vrai, y_pred, average="macro", labels=ORDRE, zero_division=0)),
        "kappa_quadratique": float(cohen_kappa_score(y_vrai, y_pred, weights="quadratic", labels=ORDRE)),
        "par_classe": {},
    }
    for classe in ORDRE:
        res["par_classe"][classe] = {
            "precision": float(precision_score(y_vrai, y_pred, labels=[classe], average="macro", zero_division=0)),
            "rappel": float(recall_score(y_vrai, y_pred, labels=[classe], average="macro", zero_division=0)),
            "f1": float(f1_score(y_vrai, y_pred, labels=[classe], average="macro", zero_division=0)),
            "effectif": int((y_vrai == classe).sum()),
        }
    res["matrice_confusion"] = confusion_matrix(y_vrai, y_pred, labels=ORDRE).tolist()

    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        classes = list(classes or ORDRE)
        i_det = classes.index("Detracteur")
        y_det = (y_vrai == "Detracteur").astype(int)
        try:
            res["auc_ovr_macro"] = float(roc_auc_score(y_vrai, proba[:, [classes.index(c) for c in ORDRE]],
                                                       multi_class="ovr", average="macro", labels=ORDRE))
        except ValueError:
            res["auc_ovr_macro"] = float("nan")
        res["auc_detracteur"] = float(roc_auc_score(y_det, proba[:, i_det])) if 0 < y_det.sum() < len(y_det) else float("nan")
        res["ap_detracteur"] = float(average_precision_score(y_det, proba[:, i_det])) if y_det.sum() else float("nan")
        res["brier_detracteur"] = float(brier_score_loss(y_det, np.clip(proba[:, i_det], 0, 1)))
        res["ece_detracteur"] = _ece(y_det, np.clip(proba[:, i_det], 0, 1))
        try:
            res["log_loss"] = float(log_loss(y_vrai, np.clip(proba, 1e-9, 1), labels=classes))
        except ValueError:
            res["log_loss"] = float("nan")
    return res


def resume(m: dict) -> dict:
    """Les colonnes qu'on met dans les tableaux comparatifs — toujours avec le rappel par classe."""
    r = {"kappa": round(m["kappa_quadratique"], 4), "macro_f1": round(m["macro_f1"], 4),
         "exactitude_equilibree": round(m["exactitude_equilibree"], 4),
         "rappel_detracteur": round(m["par_classe"]["Detracteur"]["rappel"], 4),
         "rappel_passif": round(m["par_classe"]["Passif"]["rappel"], 4),
         "rappel_promoteur": round(m["par_classe"]["Promoteur"]["rappel"], 4),
         "precision_detracteur": round(m["par_classe"]["Detracteur"]["precision"], 4)}
    for k in ("auc_detracteur", "ap_detracteur", "brier_detracteur", "ece_detracteur"):
        if k in m:
            r[k] = round(m[k], 4)
    return r


def precision_at_k(y_vrai, proba_detracteur, k: int) -> dict:
    """Étape 27 — la métrique métier : sur les K clients appelables ce mois-ci, combien sont
    réellement des détracteurs ?"""
    y_vrai = np.asarray(y_vrai).astype(str)
    proba = np.asarray(proba_detracteur)
    k = min(k, len(y_vrai))
    ordre = np.argsort(-proba)[:k]
    captes = int((y_vrai[ordre] == "Detracteur").sum())
    total = int((y_vrai == "Detracteur").sum())
    taux_base = total / len(y_vrai) if len(y_vrai) else 0.0
    precision = captes / k if k else 0.0
    return {"k": k, "detracteurs_captes": captes, "detracteurs_total": total,
            "precision_at_k": round(precision, 4),
            "rappel_at_k": round(captes / total, 4) if total else 0.0,
            "taux_de_base": round(taux_base, 4),
            "lift": round(precision / taux_base, 2) if taux_base else 0.0}


def valeur_economique(res_k: dict, cfg: dict) -> dict:
    """Traduction en euros — le langage du directeur Expérience Client. Contrefactuel : la même
    campagne en appelant au hasard."""
    m = cfg["metier"]
    cout = res_k["k"] * m["cout_appel_eur"]
    retenus = res_k["detracteurs_captes"] * m["taux_succes_appel"]
    gain = retenus * m["valeur_client_retenu_eur"]
    captes_hasard = res_k["k"] * res_k["taux_de_base"]
    gain_hasard = captes_hasard * m["taux_succes_appel"] * m["valeur_client_retenu_eur"]
    return {"cout_campagne_eur": round(cout), "clients_retenus_estimes": round(retenus, 1),
            "gain_brut_eur": round(gain), "gain_net_eur": round(gain - cout),
            "gain_net_ciblage_aleatoire_eur": round(gain_hasard - cout),
            "gain_apporte_par_le_modele_eur": round(gain - gain_hasard),
            "roi": round((gain - cout) / cout, 2) if cout else 0.0}


def nps_de(labels) -> float:
    l = pd.Series(np.asarray(labels).astype(str))
    return round(float(((l == "Promoteur").mean() - (l == "Detracteur").mean()) * 100), 1)


def nps_simule(y_vrai_sil, pred_sil, y_rep, pred_tous, seed: int, n_boot: int = 500) -> dict:
    """§ 2 de l'énoncé : « simulate the expected NPS of the silent 85 % ». Compare ce que
    l'opérateur voit (NPS des répondants), ce que le modèle estime pour les silencieux, et — ici
    seulement, parce que le dataset est complet — la vérité. Intervalle par bootstrap."""
    rng = np.random.default_rng(seed)
    pred_sil = np.asarray(pred_sil).astype(str)
    boots = [nps_de(pred_sil[rng.integers(0, len(pred_sil), len(pred_sil))]) for _ in range(n_boot)]
    return {"nps_repondants_observe": nps_de(y_rep),
            "nps_silencieux_predit": nps_de(pred_sil),
            "nps_silencieux_predit_ic95": [round(float(np.percentile(boots, 2.5)), 1), round(float(np.percentile(boots, 97.5)), 1)],
            "nps_silencieux_vrai": nps_de(y_vrai_sil),
            "nps_base_complete_predit": nps_de(pred_tous),
            "part_classes_silencieux_predit": pd.Series(pred_sil).value_counts(normalize=True).reindex(ORDRE).fillna(0).round(4).to_dict(),
            "part_classes_silencieux_vrai": pd.Series(np.asarray(y_vrai_sil).astype(str)).value_counts(normalize=True).reindex(ORDRE).fillna(0).round(4).to_dict()}


def audit_equite(df: pd.DataFrame, y_vrai, y_pred, dimensions: list[str], min_groupe: int = 50) -> dict:
    """Étape 31 — rappel ET précision Détracteur par sous-groupe, plus taux de sélection (part
    du groupe prédite détracteur : c'est ce qui alloue le budget d'appels). Les variables
    démographiques sont EXCLUES du modèle (étape 9) et servent uniquement ici."""
    y_vrai = pd.Series(np.asarray(y_vrai).astype(str), index=df.index)
    y_pred = pd.Series(np.asarray(y_pred).astype(str), index=df.index)
    resultats = {}
    for dim in dimensions:
        if dim not in df.columns:
            continue
        groupes = {}
        for valeur, sous in df.groupby(dim, observed=True):
            if len(sous) < min_groupe:
                continue
            vrai, pred = y_vrai.loc[sous.index], y_pred.loc[sous.index]
            n_det = int((vrai == "Detracteur").sum())
            if n_det < 10:
                continue
            groupes[str(valeur)] = {
                "effectif": len(sous), "detracteurs": n_det,
                "taux_detracteurs_reel": round(n_det / len(sous), 4),
                "taux_selection": round(float((pred == "Detracteur").mean()), 4),
                "rappel_detracteur": round(float(recall_score(vrai, pred, labels=["Detracteur"], average="macro", zero_division=0)), 4),
                "precision_detracteur": round(float(precision_score(vrai, pred, labels=["Detracteur"], average="macro", zero_division=0)), 4),
            }
        if len(groupes) >= 2:
            rappels = [g["rappel_detracteur"] for g in groupes.values()]
            sel = [g["taux_selection"] for g in groupes.values()]
            resultats[dim] = {"groupes": groupes, "ecart_max": round(max(rappels) - min(rappels), 4),
                              "ecart_selection": round(max(sel) - min(sel), 4)}
    return resultats


def audit_proxies(df: pd.DataFrame, X_prep: np.ndarray, attributs: dict[str, np.ndarray], seed: int) -> dict:
    """§ 4.7 : les features retenues « proxent »-elles un attribut protégé ? On entraîne, pour
    chaque attribut, un classifieur sur les features du modèle et on mesure l'AUC en validation
    croisée : 0,5 = aucune information, 1 = l'attribut est entièrement reconstructible."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    res = {}
    for nom, cible in attributs.items():
        cible = np.asarray(cible).astype(int)
        if cible.sum() < 30 or (len(cible) - cible.sum()) < 30:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        p = cross_val_predict(clf, X_prep, cible, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
        res[nom] = {"auc": round(float(roc_auc_score(cible, p)), 4), "prevalence": round(float(cible.mean()), 4)}
    return res


def seuils_par_groupe(y_vrai, proba_det, groupes, rappel_cible: float) -> dict:
    """Mitigation testée (§ 4.7) : un seuil de décision Détracteur par groupe, choisi pour que
    chaque groupe atteigne le même rappel. On mesure ce que ça coûte en précision et en volume
    d'appels — l'arbitrage est ensuite au métier, pas au modèle."""
    y_vrai = np.asarray(y_vrai).astype(str); proba_det = np.asarray(proba_det); groupes = np.asarray(groupes).astype(str)
    res = {}
    for g in np.unique(groupes):
        m = groupes == g
        det = (y_vrai[m] == "Detracteur")
        if det.sum() < 10:
            continue
        p_det = np.sort(proba_det[m][det])[::-1]
        seuil = float(p_det[min(int(np.ceil(rappel_cible * len(p_det))) - 1, len(p_det) - 1)])
        pred = proba_det[m] >= seuil
        res[g] = {"seuil": round(seuil, 4), "rappel": round(float(pred[det].mean()), 4),
                  "precision": round(float(det[pred].mean()) if pred.any() else 0.0, 4),
                  "taux_selection": round(float(pred.mean()), 4), "effectif": int(m.sum())}
    return res
