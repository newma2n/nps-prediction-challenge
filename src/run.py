"""Orchestrateur — rejoue le pipeline complet, des fichiers bruts au modèle final.

    python -m src.run

Produit : data/processed/clients_scores.parquet, models/modele_final.joblib, reports/resultats.json.
Durée indicative : 15 à 20 minutes (quinze modèles, recherche d'hyperparamètres, études de
robustesse). Tout est déterministe (seed unique dans config.yaml).

Étapes, dans l'ordre CRISP-DM :
  [1] données, qualité, registre de fuites, features
  [2] cible (trois mappings, arbitrage empirique des « 3 »)
  [3] grille : 3 scénarios de non-réponse × 3 mappings × 15 modèles
  [4] S3/M3 : validation croisée sur les répondants (pondérée IPW), hyperparamètres, sélection
  [5] robustesse : IPW à l'entraînement, déséquilibre, bruit d'étiquettes, ablation de features
  [6] modèle final : calibration vérifiée, évaluation technique et métier, NPS simulé
  [7] équité (sous-groupes, proxies, mitigation), drivers globaux et par segment, leviers
  [8] persistance
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from .data import RACINE, charger_config, preparer, registre_fuites
from .evaluate import (appliquer_seuils, audit_equite, audit_proxies, bruiter_ordinal, metriques, mitigation_seuils,
                       nps_de, nps_simule, precision_at_k, quantification, resume, seuils_par_groupe, valeur_economique)
from .features import (GROUPES_OPTIONNELS, HYPOTHESES_DERIVEES, colonnes_avec_groupes, colonnes_modele,
                       construire, noms_apres_encodage)
from .models import (CATALOGUE, FAMILLE, FORMULATION, NOMS, SIMPLICITE, OrdinalParSeuils, budget_recherche, calibrer,
                     construire_modeles, entrainer, espace_recherche, modele_pour_shap)
from .protocol import decouper, diagnostic, estimer_propension
from .explication import BLOCS_DRIVERS, agreger_par_variable, contributions_detracteur, genre_explication
from .target import ORDRE, construire_cibles

warnings.filterwarnings("ignore")

SEUIL_GAIN_REGLAGE = 0.005   # un réglage n'est retenu que s'il gagne plus que ça en CV


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Index, pd.Series)):
        return o.tolist()
    return o


def _ajuster(nom, num, cat, seed, X, y, sat, poids=None, modele=None):
    m = modele if modele is not None else construire_modeles(num, cat, seed, [nom])[nom]
    return entrainer(m, X, y, y_continu=sat, poids=poids)


def _params_fit(modele, sat):
    """Paramètres de fit à router vers l'étape `modele` (cible continue pour les modèles à seuils)."""
    if isinstance(modele.named_steps["modele"], OrdinalParSeuils):
        return {"modele__y_continu": np.asarray(sat, dtype=float)}
    return {}


def _cv_repondants(modele, X, y, sat, poids, seed):
    """Validation croisée stratifiée à 5 plis sur les répondants. Retourne kappa moyen (± écart-type),
    kappa pondéré IPW (chaque répondant compte pour 1/p̂ : estimation de la performance sur la
    population entière) et macro-F1."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    ks, fs = [], []
    pred_all = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        m = clone(modele)
        entrainer(m, X.iloc[tr], y[tr], y_continu=sat[tr])
        p = np.asarray(m.predict(X.iloc[te])).astype(str)
        pred_all[te] = p
        ks.append(cohen_kappa_score(y[te], p, weights="quadratic", labels=ORDRE))
        fs.append(f1_score(y[te], p, average="macro", labels=ORDRE, zero_division=0))
    pred_all = pred_all.astype(str)
    return {"kappa_cv": round(float(np.mean(ks)), 4), "kappa_cv_et": round(float(np.std(ks)), 4),
            "kappa_cv_ipw": round(float(cohen_kappa_score(y, pred_all, weights="quadratic", labels=ORDRE, sample_weight=poids)), 4),
            "macro_f1_cv": round(float(np.mean(fs)), 4),
            "macro_f1_cv_ipw": round(float(f1_score(y, pred_all, average="macro", labels=ORDRE, sample_weight=poids, zero_division=0)), 4)}


def _reconstruire(nom, n_, c_, seed, ref_pipe, regle: bool):
    """Même modèle (et mêmes hyperparamètres si la version réglée est retenue) sur un autre jeu
    de colonnes — pour l'ablation. CatBoost : les indices de catégorielles sont recalculés."""
    ma = construire_modeles(n_, c_, seed, [nom])[nom]
    if regle:
        params = ref_pipe.named_steps["modele"].get_params(deep=False)
        params.pop("cat_features", None)
        if "n_cat" in params:
            params["n_cat"] = len(c_)
        if "base" in params:
            params["base"] = clone(params["base"])
        ma.named_steps["modele"].set_params(**params)
    return ma


# Ce que le métier peut changer, et les variables du modèle qui le portent. L'ordre de la liste ne
# décide de rien : c'est la contribution individuelle du client qui choisit (voir `levier`).
LEVIERS_ACTIONNABLES = {
    "Engagement": {"libelle": "Proposer un engagement 12 mois avec avantage",
                   "variables": ["Contract"],
                   "condition": lambda r, med: str(r.get("Contract")) == "Month-to-Month"},
    "Prix": {"libelle": "Revue tarifaire / bundle (charge par service élevée)",
             "variables": ["Facture mensuelle", "Monthly Charge", "charge_par_service", "revenu_par_mois",
                           "Total Charges", "Total Revenue", "Total Long Distance Charges"],
             "condition": lambda r, med: pd.notna(r.get("charge_par_service")) and float(r["charge_par_service"]) > med},
    "Accueil": {"libelle": "Parcours d'accueil renforcé (client récent)",
                "variables": ["Ancienneté", "Tenure in Months", "tranche_anciennete"],
                "condition": lambda r, med: float(r.get("Tenure in Months", 99)) < 24},
    "Support": {"libelle": "Proposer le support technique premium ou la sécurité en ligne",
                "variables": ["Premium Tech Support", "Online Security", "Online Backup", "Device Protection Plan",
                              "n_services", "Internet Type"],
                "condition": lambda r, med: str(r.get("Premium Tech Support")) == "No" or str(r.get("Online Security")) == "No"},
    "Parrainage": {"libelle": "Inviter au programme de parrainage",
                   "variables": ["Parrainage", "Number of Referrals", "a_parraine"],
                   "condition": lambda r, med: float(r.get("Number of Referrals", 0)) == 0},
}
LEVIER_DEFAUT = "Appel de courtoisie et diagnostic"


def leviers_ordonnes(row: pd.Series, mediane_charge: float, contributions: pd.Series | None = None) -> list[str]:
    """Tous les leviers actionnables chez ce client, classés par contribution décroissante au risque.

    On rend la liste et pas seulement le premier, parce que le premier ne différencie presque rien
    dans cette base : le contrat mensuel est le premier facteur de risque ET la situation de la
    quasi-totalité des détracteurs prédits. Le second levier est celui qui sépare deux clients
    également mensuels, et c'est lui qui rend la recommandation exploitable.
    """
    ordre, vus = [], set()
    if contributions is not None and len(contributions):
        for var in contributions[contributions > 0].sort_values(ascending=False).index:
            for nom, lev in LEVIERS_ACTIONNABLES.items():
                if nom not in vus and var in lev["variables"] and lev["condition"](row, mediane_charge):
                    vus.add(nom); ordre.append(lev["libelle"])
    for nom, lev in LEVIERS_ACTIONNABLES.items():   # complément : actionnabilité décroissante
        if nom not in vus and lev["condition"](row, mediane_charge):
            vus.add(nom); ordre.append(lev["libelle"])
    return ordre or [LEVIER_DEFAUT]


def levier(row: pd.Series, mediane_charge: float, contributions: pd.Series | None = None) -> str:
    """Levier d'action d'un client = le **premier levier actionnable parmi ses propres facteurs de
    risque**, classés par contribution décroissante à P(Détracteur) — et non une cascade fixe
    identique pour tout le monde.

    Pourquoi ce changement : une cascade qui teste le contrat en premier attribue « engagement »
    à la quasi-totalité des détracteurs prédits, puisqu'ils sont presque tous en mensuel. La
    recommandation devient vraie mais inutilisable — elle ne distingue plus deux clients. Ici,
    deux clients mensuels reçoivent des leviers différents si ce qui pèse chez eux diffère.

    `contributions` : contributions signées du client, agrégées par variable d'origine
    (`explication.agreger_par_variable`). Repli sur la cascade d'actionnabilité si absentes.
    """
    return leviers_ordonnes(row, mediane_charge, contributions)[0]


def _bruit_uniforme(y, taux: float, rng) -> np.ndarray:
    """Borne pessimiste : une étiquette corrompue part vers n'importe quelle autre classe."""
    y = np.asarray(y).astype(str).copy()
    for i in np.flatnonzero(rng.random(len(y)) < taux):
        y[i] = rng.choice([c for c in ORDRE if c != y[i]])
    return y


def main() -> dict:
    t0 = time.time()
    cfg = charger_config()
    seed = cfg["seed"]
    R: dict = {"seed": seed}
    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
    import os
    rapide = os.environ.get("NPS_RAPIDE") == "1"   # test de fumée : 3 modèles, 1 scénario
    noms_actifs = ["logistique", "hgb_seuils", "lightgbm"] if rapide else NOMS
    scenarios = ["S3_MNAR"] if rapide else cfg["protocole"]["scenarios"]

    # ======================================================================== [1] données
    print("[1/8] Ingestion, qualité, registre de fuites, features")
    df, qualite = preparer(cfg)
    df = construire(df)
    R["qualite"] = qualite
    R["registre_fuites"] = registre_fuites(cfg)
    num, cat = colonnes_modele(df, cfg)
    R["features"] = {"numeriques": num, "categorielles": cat, "derivees": HYPOTHESES_DERIVEES,
                     "groupes_optionnels": GROUPES_OPTIONNELS}
    X = df[num + cat]
    satisfaction = pd.to_numeric(df[cfg["colonnes"]["cible"]]).values.astype(float)

    # ======================================================================== [2] cible
    print("[2/8] Cible : trois mappings, arbitrage empirique des « 3 »")
    cibles, arb = construire_cibles(df, cfg, seed)
    R["arbitrage_des_3"] = {k: v for k, v in arb.items() if k not in ("modele", "index_ambigus", "proba_detracteur_des_3")}
    off = df.groupby("Offer").agg(n=("Customer ID", "count"), taux_depart=("Churn Value", "mean"),
                                  satisfaction=("Satisfaction Score", "mean"))
    R["analyse_offer"] = off.round(3).reset_index().to_dict(orient="records")

    # ======================================================================== [3] grille
    print("[3/8] Grille : 3 scénarios × 3 mappings × 15 modèles")
    R["catalogue"] = CATALOGUE
    grille, R["diagnostic_protocole"] = [], []
    modeles_entraines = {}
    decoupes = {}
    for scenario in scenarios:
        dec = decouper(df, cfg, scenario, seed)
        decoupes[scenario] = dec
        R["diagnostic_protocole"].append(diagnostic(df, cfg, dec))
        rep, sil = dec["repondants"], dec["silencieux"]
        for mapping in ("M1", "M2", "M3"):
            y = cibles[mapping].astype(str).values
            for nom in noms_actifs:
                t = time.time()
                try:
                    m = _ajuster(nom, num, cat, seed, X[rep], y[rep], satisfaction[rep])
                    pr = m.predict_proba(X[sil]); pred = m.predict(X[sil])
                    r = metriques(y[sil], pred, pr, list(m.classes_))
                except Exception as e:  # on trace, on ne masque pas
                    r = {"erreur": f"{type(e).__name__}: {e}"}
                    m = None
                r.update({"scenario": scenario, "mapping": mapping, "modele": nom, "famille": FAMILLE[nom],
                          "formulation": FORMULATION[nom], "duree_fit_s": round(time.time() - t, 2)})
                grille.append(r)
                modeles_entraines[(scenario, mapping, nom)] = m
                if "kappa_quadratique" in r:
                    print(f"   {scenario:8s} {mapping} {nom:24s} kappa={r['kappa_quadratique']:.3f} "
                          f"macroF1={r['macro_f1']:.3f} rappelDet={r['par_classe']['Detracteur']['rappel']:.3f}")
    R["grille"] = grille

    # Diagnostic de fuite : exactitude irréelle, ou écart arbres / linéaire (signature Mustafa et al.)
    def _get(sc, mp, md, cle):
        for g in grille:
            if g["scenario"] == sc and g["mapping"] == mp and g["modele"] == md:
                return g.get(cle)
    ex_max = max(g.get("exactitude", 0) for g in grille)
    sc_ref = scenarios[0]
    arbres = [_get(sc_ref, "M3", n, "exactitude") or 0 for n in ("foret_aleatoire", "hist_gradient_boosting", "lightgbm", "xgboost", "catboost")]
    ecart = max(arbres) - (_get(sc_ref, "M3", "logistique", "exactitude") or 0)
    R["diagnostic_fuite"] = {"exactitude_max_observee": round(ex_max, 4), "seuil_alarme": 0.85,
                             "ecart_arbres_lineaire_S1_M3": round(ecart, 4), "seuil_ecart": 0.30,
                             "alarme": bool(ex_max > 0.85 or ecart > 0.30)}

    # ======================================================================== [4] S3/M3 : CV, réglage, sélection
    print("[4/8] S3/M3 — validation croisée sur les répondants, hyperparamètres, sélection")
    sc_final, mp_final = "S3_MNAR", "M3"
    dec = decoupes[sc_final]
    rep, sil = dec["repondants"], dec["silencieux"]
    y = cibles[mp_final].astype(str).values
    Xr, yr, sr = X[rep], y[rep], satisfaction[rep]
    poids_ipw = estimer_propension(df, rep, num, cat, seed)
    wr = poids_ipw[rep]
    R["ipw_estime"] = {"methode": "logistique P(répondre | covariables) ajustée sur les 7 043 clients — hypothèse MAR",
                       "poids_min": round(float(wr.min()), 3), "poids_max": round(float(wr.max()), 3),
                       "correlation_avec_propension_vraie": round(float(np.corrcoef(1 / wr, dec["poids_ipw"][rep] ** -1)[0, 1]), 3)}

    comparatif = {}
    versions = {}
    for nom in noms_actifs:
        t = time.time()
        defaut = construire_modeles(num, cat, seed, [nom])[nom]
        cv_def = _cv_repondants(defaut, Xr, yr, sr, wr, seed)
        m_def = modeles_entraines[(sc_final, mp_final, nom)]
        sil_def = resume(metriques(y[sil], m_def.predict(X[sil]), m_def.predict_proba(X[sil]), list(m_def.classes_)))
        ligne = {"modele": nom, "famille": FAMILLE[nom], "formulation": FORMULATION[nom],
                 "defaut": {"cv": cv_def, "silencieux": sil_def}}
        espace = espace_recherche(nom)
        if espace:
            try:
                rs = RandomizedSearchCV(construire_modeles(num, cat, seed, [nom])[nom], espace, n_iter=2 if rapide else budget_recherche(nom),
                                        scoring=kappa_scorer, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                        random_state=seed, n_jobs=1, refit=True)
                rs.fit(Xr, yr)
                regle = rs.best_estimator_
                cv_reg = _cv_repondants(regle, Xr, yr, sr, wr, seed)
                sil_reg = resume(metriques(y[sil], regle.predict(X[sil]), regle.predict_proba(X[sil]), list(regle.classes_)))
                params = {k.replace("modele__", "").replace("base__", ""): (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else str(v))
                          for k, v in rs.best_params_.items()}
                ligne["regle"] = {"cv": cv_reg, "silencieux": sil_reg, "meilleurs_params": params,
                                  "n_configurations": int(len(rs.cv_results_["params"]))}
                gain = cv_reg["kappa_cv_ipw"] - cv_def["kappa_cv_ipw"]
                ligne["gain_reglage_cv_ipw"] = round(gain, 4)
                if gain > SEUIL_GAIN_REGLAGE:
                    ligne["version_retenue"] = "réglée"; versions[nom] = regle
                else:
                    ligne["version_retenue"] = "par défaut"; versions[nom] = m_def
            except Exception as e:
                ligne["regle"] = {"erreur": f"{type(e).__name__}: {str(e)[:200]}"}
                ligne["version_retenue"] = "par défaut"; versions[nom] = m_def
        else:
            ligne["version_retenue"] = "par défaut (pas d'espace de réglage)"; versions[nom] = m_def
        ligne["duree_s"] = round(time.time() - t, 1)
        comparatif[nom] = ligne
        v = ligne[("regle" if ligne["version_retenue"] == "réglée" else "defaut")]
        print(f"   {nom:24s} CV kappa {cv_def['kappa_cv']:.3f} (ipw {cv_def['kappa_cv_ipw']:.3f}) -> retenu {ligne['version_retenue']:11s} "
              f"CV ipw {v['cv']['kappa_cv_ipw']:.3f} · silencieux {v['silencieux']['kappa']:.3f}  [{ligne['duree_s']}s]")
    R["comparatif_s3_m3"] = comparatif

    # Sélection PAR LA CV DES RÉPONDANTS (seule information disponible en production), pondérée IPW.
    # Les silencieux sont le test final : ils ne participent jamais au choix.
    classement = []
    for nom, l in comparatif.items():
        v = l["regle"] if l["version_retenue"] == "réglée" else l["defaut"]
        # Précision@K de CHAQUE modèle, et pas seulement du retenu : c'est la métrique qui décide
        # de l'usage réel (une capacité d'appel fixe), et sans elle on ne peut pas répondre à
        # « la baseline capte plus de détracteurs, pourquoi ne pas la prendre ? ».
        try:
            mv = versions[nom]
            prv = mv.predict_proba(X[sil])[:, list(mv.classes_).index("Detracteur")]
            pkv = precision_at_k(y[sil], prv, cfg["metier"]["k_appels"])
            prec_k, lift_k = pkv["precision_at_k"], pkv["lift"]
        except Exception:
            prec_k = lift_k = None
        classement.append({"modele": nom, "famille": l["famille"], "formulation": l["formulation"], "version": l["version_retenue"],
                           "kappa_cv_ipw": v["cv"]["kappa_cv_ipw"], "kappa_cv": v["cv"]["kappa_cv"], "kappa_cv_et": v["cv"]["kappa_cv_et"],
                           "macro_f1_cv": v["cv"]["macro_f1_cv"],
                           "kappa_silencieux": v["silencieux"]["kappa"], "macro_f1_silencieux": v["silencieux"]["macro_f1"],
                           "rappel_detracteur_silencieux": v["silencieux"]["rappel_detracteur"],
                           "rappel_passif_silencieux": v["silencieux"]["rappel_passif"],
                           "precision_at_k_silencieux": prec_k, "lift_silencieux": lift_k,
                           "auc_detracteur_silencieux": v["silencieux"].get("auc_detracteur")})
    classement.sort(key=lambda d: (-d["kappa_cv_ipw"], -d["macro_f1_cv"]))
    for i, c in enumerate(classement, 1):
        c["rang_cv"] = i
    rang_sil = {c["modele"]: i for i, c in enumerate(sorted(classement, key=lambda d: -d["kappa_silencieux"]), 1)}
    for c in classement:
        c["rang_silencieux"] = rang_sil[c["modele"]]
    # Règle de parcimonie (« one standard error rule », Hastie, Tibshirani & Friedman, ESL § 7.10) :
    # parmi les modèles dont le kappa CV est à moins d'un écart-type (entre plis) du meilleur, on
    # retient le plus simple selon l'ordre fixé a priori dans models.SIMPLICITE. Un modèle complexe
    # n'est retenu que s'il gagne plus que le bruit de la validation.
    meilleur = classement[0]
    seuil_parcimonie = meilleur["kappa_cv_ipw"] - meilleur["kappa_cv_et"]
    candidats = [c for c in classement if c["kappa_cv_ipw"] >= seuil_parcimonie]
    retenu = min(candidats, key=lambda c: (SIMPLICITE[c["modele"]], -c["kappa_cv_ipw"]))
    md_final = retenu["modele"]
    final = versions[md_final]
    for c in classement:
        c["simplicite"] = SIMPLICITE[c["modele"]]
        c["dans_la_marge"] = c["kappa_cv_ipw"] >= seuil_parcimonie
    rho_ipw = spearmanr([c["kappa_cv_ipw"] for c in classement], [c["kappa_silencieux"] for c in classement]).correlation
    rho_cv = spearmanr([c["kappa_cv"] for c in classement], [c["kappa_silencieux"] for c in classement]).correlation
    R["selection_modele_final"] = {
        "critere": "kappa quadratique en validation croisée (5 plis) sur les répondants, pondéré par l'inverse de la propension estimée ; départage par macro-F1 CV ; les silencieux ne servent qu'au test final",
        "classement": classement, "retenu": md_final, "version_retenue": comparatif[md_final]["version_retenue"],
        "meilleur_cv_brut": meilleur["modele"], "seuil_parcimonie": round(seuil_parcimonie, 4),
        "candidats_dans_la_marge": [c["modele"] for c in candidats],
        "regle_parcimonie": "parmi les modèles à moins d'un écart-type (entre plis) du meilleur kappa CV, le plus simple l'emporte (ESL § 7.10)",
        "gagnant_silencieux": min(classement, key=lambda c: c["rang_silencieux"])["modele"],
        "spearman_cv_ipw_vs_silencieux": round(float(rho_ipw), 3), "spearman_cv_vs_silencieux": round(float(rho_cv), 3),
        "ecart_cv_silencieux_du_retenu": round(classement[0]["kappa_cv"] - classement[0]["kappa_silencieux"], 4),
    }
    print(f"   meilleur kappa CV brut : {meilleur['modele']} ; retenu après parcimonie : {md_final} ({comparatif[md_final]['version_retenue']}) ; "
          f"meilleur sur les silencieux : {R['selection_modele_final']['gagnant_silencieux']}")

    # Meilleur modèle par formulation (§ 2 : classification / ordinale / régression + seuils)
    R["formulations"] = {}
    for f in sorted(set(FORMULATION.values())):
        cands = [c for c in classement if c["formulation"] == f]
        if not cands:
            continue
        best = min(cands, key=lambda c: c["rang_cv"])
        R["formulations"][f] = {"meilleur_modele": best["modele"], "kappa_cv_ipw": best["kappa_cv_ipw"],
                                "kappa_silencieux": best["kappa_silencieux"], "macro_f1_silencieux": best["macro_f1_silencieux"],
                                "n_modeles": len(cands)}

    # ======================================================================== [5] robustesse
    print("[5/8] Robustesse : IPW à l'entraînement, déséquilibre, bruit d'étiquettes, ablation")
    base_final = construire_modeles(num, cat, seed, [md_final])[md_final]
    if comparatif[md_final]["version_retenue"] == "réglée":
        base_final = clone(final)

    # IPW comme poids d'entraînement (les modèles qui ne les acceptent pas remontent une erreur, tracée)
    try:
        m_ipw = entrainer(clone(base_final), Xr, yr, y_continu=sr, poids=wr)
        r_ipw = resume(metriques(y[sil], m_ipw.predict(X[sil]), m_ipw.predict_proba(X[sil]), list(m_ipw.classes_)))
        r_sans = resume(metriques(y[sil], final.predict(X[sil]), final.predict_proba(X[sil]), list(final.classes_)))
        R["effet_ipw"] = {"sans_ipw": r_sans, "avec_ipw": r_ipw,
                          "retenu": "avec IPW" if r_ipw["kappa"] > r_sans["kappa"] + SEUIL_GAIN_REGLAGE else "sans IPW",
                          "note": "poids estimés sous hypothèse MAR ; la propension vraie sous S3 dépend de la satisfaction, invisible"}
        if R["effet_ipw"]["retenu"] == "avec IPW":
            final = m_ipw
    except Exception as e:
        R["effet_ipw"] = {"erreur": f"{type(e).__name__}: {str(e)[:200]}"}

    # Déséquilibre : stratégies comparées sur la baseline logistique (référence commune)
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.linear_model import LogisticRegression
    from .features import pipeline_preparation
    strategies = {
        "aucune (logistique non pondérée)": ("aucune", None),
        "pondération de classes 'balanced' (retenue)": ("poids", None),
        "sur-échantillonnage aléatoire": ("ech", RandomOverSampler(random_state=seed)),
        "SMOTE (k=5)": ("ech", SMOTE(random_state=seed, k_neighbors=5)),
        "sous-échantillonnage aléatoire": ("ech", RandomUnderSampler(random_state=seed)),
    }
    R["desequilibre"] = {"distribution_repondants": pd.Series(yr).value_counts(normalize=True).reindex(ORDRE).round(4).to_dict(),
                         "distribution_silencieux": pd.Series(y[sil]).value_counts(normalize=True).reindex(ORDRE).round(4).to_dict(),
                         "distribution_base": pd.Series(y).value_counts(normalize=True).reindex(ORDRE).round(4).to_dict(),
                         "strategies": []}
    for lib, (genre, sampler) in strategies.items():
        try:
            lr = LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced" if genre == "poids" else None)
            if genre == "ech":
                pipe = ImbPipeline([("preparation", pipeline_preparation(num, cat, normaliser=True)), ("echantillonnage", sampler), ("modele", lr)])
            else:
                from sklearn.pipeline import Pipeline
                pipe = Pipeline([("preparation", pipeline_preparation(num, cat, normaliser=True)), ("modele", lr)])
            pipe.fit(Xr, yr)
            r = resume(metriques(y[sil], pipe.predict(X[sil]), pipe.predict_proba(X[sil]), list(pipe.classes_)))
            R["desequilibre"]["strategies"].append({"strategie": lib, **r})
        except Exception as e:
            R["desequilibre"]["strategies"].append({"strategie": lib, "erreur": f"{type(e).__name__}: {str(e)[:120]}"})

    # On corrompt une part des étiquettes d'entraînement et on mesure la dégradation sur les
    # silencieux. Le bruit ordinal est le seul réaliste pour une note : un promoteur mal mesuré
    # devient passif plus souvent que détracteur. Le bruit uniforme sert de borne pessimiste.
    R["bruit_etiquettes"] = []
    R["bruit_etiquettes_note"] = ("bruit ordinal : 80 % des bascules vont vers une classe adjacente, 20 % vers l'opposée, et un "
                                  "client à satisfaction 3 est trois fois plus exposé qu'un autre. Le bruit uniforme, borne "
                                  "pessimiste, tire la classe de remplacement au hasard parmi les deux autres.")
    poids_ambigu = 1.0 + 2.0 * (sr == 3).astype(float)
    for taux in (0.0, 0.05, 0.10, 0.20, 0.30):
        for genre_bruit in ("ordinal", "uniforme"):
            if taux == 0 and genre_bruit == "uniforme":
                continue
            yb = (bruiter_ordinal(yr, taux, np.random.default_rng(seed + 1), poids=poids_ambigu) if genre_bruit == "ordinal"
                  else _bruit_uniforme(yr, taux, np.random.default_rng(seed + 2)))
            try:
                mb = entrainer(clone(base_final), Xr, yb, y_continu=sr)
                r = resume(metriques(y[sil], mb.predict(X[sil]), mb.predict_proba(X[sil]), list(mb.classes_)))
                R["bruit_etiquettes"].append({"taux_bruit": taux, "genre": genre_bruit,
                                              "etiquettes_modifiees": int((np.asarray(yb) != np.asarray(yr)).sum()), **r})
            except Exception as e:
                R["bruit_etiquettes"].append({"taux_bruit": taux, "genre": genre_bruit, "erreur": str(e)[:120]})

    # Ablation de features : ce que coûte ou rapporte chaque bloc (§ 4.3, § 4.7)
    derivees = list(HYPOTHESES_DERIVEES)
    configs = {
        "base (retenue)": (num, cat),
        "+ démographie": colonnes_avec_groupes(df, cfg, ["demographie"]),
        "+ géographie (lat, lon, population)": colonnes_avec_groupes(df, cfg, ["geographie"]),
        "+ démographie + géographie": colonnes_avec_groupes(df, cfg, ["demographie", "geographie"]),
        "− variables dérivées": ([c for c in num if c not in derivees], [c for c in cat if c not in derivees]),
        "− parrainage (Number of Referrals, a_parraine)": ([c for c in num if c not in ("Number of Referrals", "a_parraine")], cat),
        "− Offer et a_recu_offre": ([c for c in num if c != "a_recu_offre"], [c for c in cat if c != "Offer"]),
        "− Contract": (num, [c for c in cat if c != "Contract"]),
    }
    # Ce que coûte le retrait des variables qui reconstruisent le mieux l'âge.
    try:
        from sklearn.linear_model import LogisticRegression as _LR
        _prep_p = construire_modeles(num, cat, seed, ["logistique"])["logistique"].named_steps["preparation"].fit(X[rep])
        _Xp = np.asarray(_prep_p.transform(X[rep]), dtype=float); _noms_p = noms_apres_encodage(_prep_p)
        _lr = _LR(max_iter=2000, class_weight="balanced", random_state=seed).fit(_Xp, (df.loc[rep, "Age"] < 30).astype(int).values)
        proxies_age, vus = [], set()
        for i in np.argsort(-np.abs(_lr.coef_[0])):
            base = next((c for c in cat if _noms_p[i].startswith(c + "_")), _noms_p[i])
            if base in (num + cat) and base not in vus:
                vus.add(base); proxies_age.append(base)
            if len(proxies_age) == 3:
                break
        R["proxies_age_retires"] = proxies_age
        configs["− top-3 proxies d'âge (" + ", ".join(proxies_age) + ")"] = ([c for c in num if c not in proxies_age],
                                                                             [c for c in cat if c not in proxies_age])
    except Exception as e:
        R["proxies_age_retires"] = {"erreur": f"{type(e).__name__}: {str(e)[:120]}"}
    R["ablation_features"] = []
    for lib, (n_, c_) in configs.items():
        try:
            ma = _reconstruire(md_final, n_, c_, seed, final, comparatif[md_final]["version_retenue"] == "réglée")
            entrainer(ma, df.loc[rep, n_ + c_], yr, y_continu=sr)
            r = resume(metriques(y[sil], ma.predict(df.loc[sil, n_ + c_]), ma.predict_proba(df.loc[sil, n_ + c_]), list(ma.classes_)))
            R["ablation_features"].append({"configuration": lib, "n_variables": len(n_) + len(c_), **r})
        except Exception as e:
            R["ablation_features"].append({"configuration": lib, "erreur": f"{type(e).__name__}: {str(e)[:120]}"})

    # ======================================================================== [6] modèle final
    print("[6/8] Modèle final : calibration vérifiée, évaluation technique et métier, NPS simulé")
    final_cal = calibrer(final, Xr, yr, seed, methode="sigmoid") or final
    classes = list(final_cal.classes_); i_det = classes.index("Detracteur")
    proba_sil = final_cal.predict_proba(X[sil]); proba_brut_sil = final.predict_proba(X[sil])
    pred_cal = np.array(classes)[proba_sil.argmax(axis=1)]
    pred_brut = np.asarray(final.predict(X[sil])).astype(str)
    m_cal = metriques(y[sil], pred_cal, proba_sil, classes)
    m_brut = metriques(y[sil], pred_brut, proba_brut_sil, list(final.classes_))
    passif_ecrase = m_cal["par_classe"]["Passif"]["rappel"] < 0.05
    if passif_ecrase or m_cal["macro_f1"] < m_brut["macro_f1"] - 0.03:
        pipeline_decision, pred_sil = final, pred_brut
        strategie = ("hybride — classe : modèle non calibré (la calibration écrasait la classe Passif) ; "
                     "P(Détracteur) : modèle calibré (Platt), pour le classement des appels")
    else:
        pipeline_decision, pred_sil = final_cal, pred_cal
        strategie = "modèle calibré pour la classe et pour le classement"
    R["modele_final"] = {
        "scenario": sc_final, "mapping": mp_final, "modele": md_final, "famille": FAMILLE[md_final],
        "formulation": FORMULATION[md_final], "version": comparatif[md_final]["version_retenue"],
        "n_entrainement": int(rep.sum()), "n_evaluation": int(sil.sum()), "strategie_decision": strategie,
        "metriques_retenues": metriques(y[sil], pred_sil, proba_sil, classes),
        "metriques_calibre": m_cal, "metriques_non_calibre": m_brut,
        "calibration": {"non_calibre": {"brier": round(m_brut["brier_detracteur"], 4), "ece": round(m_brut["ece_detracteur"], 4)},
                        "calibre_platt": {"brier": round(m_cal["brier_detracteur"], 4), "ece": round(m_cal["ece_detracteur"], 4)},
                        "passif_ecrase": bool(passif_ecrase)},
    }
    k = cfg["metier"]["k_appels"]
    res_k = precision_at_k(y[sil], proba_sil[:, i_det], k)
    R["evaluation_metier"] = {"precision_at_k": res_k, "economie": valeur_economique(res_k, cfg),
                              "courbe_k": [precision_at_k(y[sil], proba_sil[:, i_det], kk) for kk in (100, 250, 500, 750, 1000, 1500, 2000)]}
    pred_tous = np.asarray(pipeline_decision.predict(X)).astype(str)
    R["nps_simule"] = nps_simule(y[sil], pred_sil, yr, pred_tous, seed)
    # Quantification : quatre estimateurs de la composition des silencieux, comparés à la vérité.
    # La matrice de confusion vient d'une validation croisée sur les répondants — la seule
    # disponible en production ; ce qu'elle vaut sous MNAR est mesuré, pas supposé.
    skf_q = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred_cv_q = np.empty(len(yr), dtype=object); proba_cv_q = np.zeros((len(yr), 3))
    for tr, te in skf_q.split(Xr, yr):
        mq = clone(final); entrainer(mq, Xr.iloc[tr], yr[tr], y_continu=sr[tr])
        pred_cv_q[te] = np.asarray(mq.predict(Xr.iloc[te])).astype(str)
        pq = mq.predict_proba(Xr.iloc[te]); cq = list(mq.classes_); proba_cv_q[te] = pq[:, [cq.index(c) for c in ORDRE]]
    proba_sil_ordre = proba_brut_sil[:, [list(final.classes_).index(c) for c in ORDRE]]
    R["nps_simule"]["quantification"] = quantification(yr, pred_cv_q.astype(str), proba_cv_q, pred_sil, proba_sil_ordre, seed)
    R["nps_simule"]["quantification"]["verite"] = {"parts": {c: round(float((y[sil] == c).mean()), 4) for c in ORDRE}, "nps": nps_de(y[sil])}
    # NPS par segment : prédit vs vrai (silencieux)
    seg_nps = []
    df_sil = df[sil]
    for lib, masque in (("Contrat mensuel", df_sil["Contract"] == "Month-to-Month"), ("Contrat 1–2 ans", df_sil["Contract"] != "Month-to-Month"),
                        ("Ancienneté < 12 mois", df_sil["Tenure in Months"] < 12), ("Ancienneté ≥ 24 mois", df_sil["Tenure in Months"] >= 24),
                        ("Fibre", df_sil["Internet Type"] == "Fiber Optic"), ("DSL", df_sil["Internet Type"] == "DSL"),
                        ("Sans internet", df_sil["Internet Type"] == "Pas d'internet")):
        mk_ = masque.values
        if mk_.sum() < 50:
            continue
        seg_nps.append({"segment": lib, "clients": int(mk_.sum()), "nps_predit": nps_de(pred_sil[mk_]), "nps_vrai": nps_de(y[sil][mk_]),
                        "part_detracteurs_predite": round(float((pred_sil[mk_] == "Detracteur").mean()), 4),
                        "part_detracteurs_vraie": round(float((y[sil][mk_] == "Detracteur").mean()), 4)})
    R["nps_simule"]["par_segment"] = seg_nps

    # ======================================================================== [7] équité, drivers, leviers
    print("[7/8] Équité (sous-groupes, proxies, mitigation), drivers globaux et par segment, leviers")
    df_sil = df[sil].copy()
    # right=False : « <30 » = [0 ; 30[ et « 60+ » = [60 ; 120[. Avec les bornes fermées à droite,
    # un client de 30 ans tombait dans « <30 » et un client de 60 ans dans « 45-60 » : les libellés
    # mentaient sur leur contenu, et l'audit d'équité portait sur des groupes mal nommés.
    df_sil["tranche_age"] = pd.cut(df_sil["Age"], [0, 30, 45, 60, 120], right=False,
                                   labels=["<30", "30-45", "45-60", "60+"]).astype(str)
    tertiles = df["Population"].quantile([1 / 3, 2 / 3]).values
    df_sil["zone"] = pd.cut(df_sil["Population"], [-1, tertiles[0], tertiles[1], np.inf],
                            labels=["faible densité", "densité moyenne", "forte densité"]).astype(str)
    dims = ["Gender", "Senior Citizen", "Married", "Dependents", "tranche_age", "zone"]
    R["audit_equite"] = audit_equite(df_sil, y[sil], pred_sil, dims)
    R["audit_equite_note"] = "zone = tertiles de la population du code postal (proxy géographique de densité) ; tranche_age construite sur Age"

    prep_f = final.named_steps["preparation"]
    Xt_sil = prep_f.transform(X[sil])
    Xt_sil = np.asarray(Xt_sil.select_dtypes("number") if hasattr(Xt_sil, "select_dtypes") else Xt_sil, dtype=float) \
        if not hasattr(Xt_sil, "select_dtypes") else None
    # Pour l'audit des proxies on utilise les features préparées par la logistique (numériques, standardisées)
    prep_lin = construire_modeles(num, cat, seed, ["logistique"])["logistique"].named_steps["preparation"].fit(X[rep])
    Xt_lin = np.asarray(prep_lin.transform(X[sil]), dtype=float)
    R["audit_proxies"] = audit_proxies(df_sil, Xt_lin, {
        "Moins de 30 ans": (df_sil["Age"] < 30).values, "Senior (65+)": (df_sil["Senior Citizen"].astype(str) == "Yes").values,
        "Femme": (df_sil["Gender"].astype(str) == "Female").values, "Marié(e)": (df_sil["Married"].astype(str) == "Yes").values,
        "A des dépendants": (df_sil["Dependents"].astype(str) == "Yes").values,
        "Zone de forte densité": (df_sil["zone"] == "forte densité").values}, seed,
        noms_features=noms_apres_encodage(prep_lin))
    R["audit_proxies_note"] = ("AUC de reconstruction d'un attribut protégé à partir des seules variables du modèle, en validation "
                               "croisée : 0,5 = aucune information, 1 = attribut entièrement reconstructible. Les variables qui le "
                               "reconstruisent sont nommées pour que leur retrait soit chiffrable — voir l'ablation « − top-3 proxies d'âge ».")

    # Seuil de décision par tranche d'âge. Les seuils sont appris sur les répondants puis appliqués
    # aux silencieux : les caler sur les silencieux égaliserait les rappels par construction. La
    # variante in-sample est calculée pour montrer cet écart. Deux politiques sont chiffrées,
    # nivellement et rattrapage ; l'arbitrage revient au métier.
    rappel_global = R["modele_final"]["metriques_retenues"]["par_classe"]["Detracteur"]["rappel"]
    avant = {g: v for g, v in R["audit_equite"].get("tranche_age", {}).get("groupes", {}).items()}
    ages_rep = pd.cut(df.loc[rep, "Age"], [0, 30, 45, 60, 120], right=False,
                      labels=["<30", "30-45", "45-60", "60+"]).astype(str).values
    ages_sil = df_sil["tranche_age"].values
    cls_dec = list(pipeline_decision.classes_)
    proba_rep_det = pipeline_decision.predict_proba(Xr)[:, cls_dec.index("Detracteur")]
    mit = mitigation_seuils(yr, proba_rep_det, ages_rep, y[sil], proba_sil[:, i_det], ages_sil,
                            pred_sil, rappel_cible=round(rappel_global, 2))
    pred_niv = mit.pop("pred_hors_echantillon"); pred_ins = mit.pop("pred_in_sample")

    groupe_faible = min(avant, key=lambda g: avant[g]["rappel_detracteur"]) if avant else None
    seuils_rat = {g: v for g, v in seuils_par_groupe(yr, proba_rep_det, ages_rep, round(rappel_global, 2)).items() if g == groupe_faible}
    pred_rat = appliquer_seuils(pred_sil, proba_sil[:, i_det], ages_sil, seuils_rat)

    def _bilan(pred):
        mm_ = cfg["metier"]
        n = int((pred == "Detracteur").sum())
        vrais = int(((pred == "Detracteur") & (y[sil] == "Detracteur")).sum())
        rappels = {g: round(float(((pred == "Detracteur") & (y[sil] == "Detracteur") & (ages_sil == g)).sum()
                                  / max(int(((y[sil] == "Detracteur") & (ages_sil == g)).sum()), 1)), 4) for g in sorted(avant)}
        return {"global": resume(metriques(y[sil], pred, proba_sil, classes)), "rappels_par_groupe": rappels,
                "ecart_max_rappel": round(max(rappels.values()) - min(rappels.values()), 4) if rappels else None,
                "appels": n, "vrais_detracteurs_appeles": vrais,
                "gain_net_eur": round(vrais * mm_["taux_succes_appel"] * mm_["valeur_client_retenu_eur"] - n * mm_["cout_appel_eur"])}

    R["mitigation_seuils_age"] = {
        "rappel_cible": round(rappel_global, 4), "avant": avant, **mit,
        "politiques": {
            "aucune (seuil unique, classe la plus probable)": _bilan(pred_sil),
            "nivellement — seuils appris sur les répondants, tous les groupes visent le rappel global": _bilan(pred_niv),
            "rattrapage — seul le groupe le moins bien servi (" + str(groupe_faible) + ") est relevé": _bilan(pred_rat),
            "nivellement in-sample — seuils calés sur les silencieux (optimiste, pour comparaison)": _bilan(pred_ins)},
        "note_methode": ("les seuils retenus sont appris sur les répondants et appliqués aux silencieux ; la variante in-sample, "
                         "calée sur les silencieux eux-mêmes, égalise les rappels par construction et surestime le bénéfice réel"),
        # clés conservées pour les rapports existants
        "apres": mit["seuils_hors_echantillon"], "global_avant": resume(R["modele_final"]["metriques_retenues"]),
        "global_apres": resume(metriques(y[sil], pred_niv, proba_sil, classes)),
        "appels_avant": int((pred_sil == "Detracteur").sum()), "appels_apres": int((pred_niv == "Detracteur").sum())}

    # Contributions à la classe Détracteur, agrégées par variable d'origine : les colonnes one-hot
    # d'une même variable sont sommées, sinon ses modalités apparaissent comme des variables
    # opposées. Le sens dépend de la nature de la variable : signe de moyenne(valeur × contribution)
    # pour une numérique, modalité la plus à risque pour une catégorielle.
    contributions_clients = None
    try:
        C_brut, methode = contributions_detracteur(final, X[sil], seed)
        C = agreger_par_variable(C_brut, cat)
        imp = C.abs().mean().sort_values(ascending=False)
        moy = C.mean()
        # Sens de l'effet, calculé selon la nature de la variable (voir la note ci-dessus).
        Xt_brut = pd.DataFrame(np.asarray(prep_f.transform(X.loc[C_brut.index]), dtype=float),
                               index=C_brut.index, columns=list(C_brut.columns))
        origine = {}
        for col in C_brut.columns:
            base = next((c for c in cat if col.startswith(c + "_")), col)
            for nom_bloc, membres in BLOCS_DRIVERS.items():
                if base in membres:
                    base = nom_bloc
            origine.setdefault(base, []).append(col)

        sens, detail_sens = {}, {}
        for var, cols in origine.items():
            est_cat = any(any(c.startswith(cc + "_") for cc in cat) for c in cols)
            if est_cat:
                # Une variable catégorielle n'a pas de sens global : ce sont ses modalités qui
                # poussent ou protègent. On nomme les deux extrêmes.
                effets = {}
                for c in cols:
                    m_ = Xt_brut[c] > 0.5
                    if m_.sum() >= 30:
                        prefixe = next((cc for cc in cat if c.startswith(cc + "_")), "")
                        effets[c[len(prefixe) + 1:] if prefixe else c] = float(C_brut.loc[m_, c].mean())
                if not effets:
                    continue
                pousse = max(effets, key=effets.get); protege = min(effets, key=effets.get)
                sens[var] = f"▲ « {pousse} » pousse vers la détraction"
                detail_sens[var] = {"nature": "catégorielle", "modalite_la_plus_a_risque": pousse,
                                    "effet_de_cette_modalite": round(effets[pousse], 4),
                                    "modalite_la_plus_protectrice": protege,
                                    "effet_de_cette_modalite_protectrice": round(effets[protege], 4)}
            else:
                # Numérique : signe de moyenne(valeur × contribution) — celui du coefficient.
                pente = float(sum((Xt_brut[c] * C_brut[c]).mean() for c in cols))
                sens[var] = ("▲ une valeur élevée pousse vers la détraction" if pente > 0
                             else "▼ une valeur élevée protège")
                detail_sens[var] = {"nature": "numérique", "effet_signe": round(pente, 4)}
        segs = {"Contrat mensuel": (df.loc[C.index, "Contract"] == "Month-to-Month", ["Contract"]),
                "Contrat 1–2 ans": (df.loc[C.index, "Contract"] != "Month-to-Month", ["Contract"]),
                "Ancienneté < 12 mois": (df.loc[C.index, "Tenure in Months"] < 12, ["Ancienneté"]),
                "Ancienneté ≥ 24 mois": (df.loc[C.index, "Tenure in Months"] >= 24, ["Ancienneté"]),
                "Fibre": (df.loc[C.index, "Internet Type"] == "Fiber Optic", ["Internet Type"]),
                "DSL": (df.loc[C.index, "Internet Type"] == "DSL", ["Internet Type"])}
        par_segment = {}
        for lib, (mk_, definissantes) in segs.items():
            sub = C[mk_.values]
            if len(sub) < 30:
                continue
            colonnes = [c for c in sub.columns if c not in definissantes and float(sub[c].std()) > 1e-12]
            top = sub[colonnes].abs().mean().sort_values(ascending=False).head(8)
            # Pas de sens propre au segment : le modèle est additif, seul le poids varie.
            par_segment[lib] = {"clients": int(len(sub)), "variables_exclues": definissantes,
                                "drivers": {v: {"importance": round(float(top[v]), 4),
                                                "sens_global": sens.get(v, "—")}
                                            for v in top.index}}
        R["drivers"] = {
            "methode": methode,
            "agregation": "contributions sommées par variable d'origine (modalités one-hot regroupées ; blocs parrainage, ancienneté, facture mensuelle)",
            "perimetre": (str(int(len(C))) + " clients silencieux ; segments d'ancienneté « < 12 mois » et « ≥ 24 mois » (la bande 12–24 mois "
                          "est volontairement écartée pour contraster) ; accès « Fibre » et « DSL » (câble et sans internet non représentés)"),
            "importance_detracteur": imp.head(20).round(4).to_dict(),
            "contribution_moyenne": {v: round(float(moy[v]), 4) for v in imp.index[:20]},
            "sens": {v: sens.get(v, "—") for v in imp.index[:20]},
            "detail_sens": {v: detail_sens.get(v, {}) for v in imp.index[:20]},
            "legende_sens": ("▲ = pousse vers la détraction, ▼ = protège. Pour une variable numérique, le sens est celui "
                             "de l'effet d'une valeur élevée ; pour une variable catégorielle, la modalité en cause est "
                             "nommée, car « Contract » ne pousse ni ne protège en soi — c'est « Month-to-Month » qui pousse. "
                             "⚠️ La colonne « contribution moyenne » ne donne PAS le sens : pour une variable numérique "
                             "standardisée elle vaut environ zéro par construction."),
            "avertissement_segments": ("le modèle retenu est additif, sans interaction : le coefficient d'une variable est le même dans "
                                       "tous les segments. Ce tableau indique où chaque variable PÈSE le plus (effet de composition), pas "
                                       "un effet propre au segment ; la variable qui définit le segment est exclue de son classement."),
            "par_segment": par_segment, "n_lignes_expliquees": int(len(C))}
        contributions_clients = C
    except Exception as e:
        R["drivers"] = {"erreur": f"{type(e).__name__}: {str(e)[:200]}"}

    # Leviers recommandés : dérivés des contributions individuelles, donc réellement discriminants
    # entre deux clients au même contrat (voir la docstring de `levier`).
    med_charge = float(df["charge_par_service"].median())
    if contributions_clients is not None:
        C_tous = agreger_par_variable(contributions_detracteur(final, X, seed, max_lignes=len(X))[0], cat)
        rangs = [leviers_ordonnes(df.loc[i], med_charge, C_tous.loc[i] if i in C_tous.index else None) for i in df.index]
        origine_levier = "contribution individuelle à P(Détracteur), agrégée par variable d'origine"
    else:
        rangs = [leviers_ordonnes(r, med_charge) for _, r in df.iterrows()]
        origine_levier = "repli : cascade d'actionnabilité (contributions indisponibles)"
    leviers = pd.Series([r[0] for r in rangs], index=df.index)
    leviers2 = pd.Series([r[1] if len(r) > 1 else "—" for r in rangs], index=df.index)
    det = pred_sil == "Detracteur"
    dist = leviers[sil][det].value_counts()
    dist2 = leviers2[sil][det].value_counts()
    paires = (leviers[sil][det] + "  ▸  " + leviers2[sil][det]).value_counts()
    R["leviers"] = {
        "origine": origine_levier,
        "regle": ["le levier retenu est le premier levier actionnable parmi les facteurs de risque propres au client, "
                  "classés par contribution décroissante à P(Détracteur)",
                  *[nom + " → " + lev["libelle"] + " — déclenché par : " + ", ".join(lev["variables"][:3])
                    for nom, lev in LEVIERS_ACTIONNABLES.items()],
                  "aucun levier actionnable chez ce client → " + LEVIER_DEFAUT],
        "distribution_detracteurs_predits": dist.to_dict(),
        "distribution_levier_secondaire": dist2.to_dict(),
        "distribution_paires": paires.head(10).to_dict(),
        "concentration_du_levier_dominant": round(float(dist.iloc[0] / dist.sum()), 4) if len(dist) else None,
        "concentration_de_la_paire_dominante": round(float(paires.iloc[0] / paires.sum()), 4) if len(paires) else None,
        "n_leviers_distincts_en_second": int(len(dist2)),
        "lecture": ("le premier levier est concentré parce que la base l'est : le contrat mensuel est à la fois le "
                    "premier facteur de risque et la situation de la quasi-totalité des détracteurs prédits. La "
                    "recommandation reste vraie, mais elle ne trie pas — c'est le SECOND levier qui différencie deux "
                    "clients également mensuels, et c'est pourquoi les deux sont livrés."),
        "mediane_charge_par_service": round(med_charge, 2),
        "avertissement": ("heuristique dérivée des drivers — donc d'associations, pas d'un effet causal mesuré. Un levier est une "
                          "hypothèse à tester en campagne, avec groupe de contrôle (10 à 20 % des ciblés non appelés).")}

    # Le modèle retenu est réajusté sous M1, M2 et M3 sur les mêmes répondants. Les kappas ne se
    # comparent pas d'un mapping à l'autre ; seuls le sens des effets et l'effet métier le peuvent.
    try:
        coefs, effet = {}, {}
        for mp in ("M1", "M2", "M3"):
            ym = cibles[mp].astype(str).values
            mm = clone(base_final); entrainer(mm, Xr, ym[rep], y_continu=sr)
            coefs[mp] = agreger_par_variable(contributions_detracteur(mm, X[sil], seed)[0], cat).mean()
            prm = mm.predict_proba(X[sil]); cm = list(mm.classes_)
            pkm = precision_at_k(ym[sil], prm[:, cm.index("Detracteur")], cfg["metier"]["k_appels"])
            effet[mp] = {"part_detracteurs_predits": round(float((np.asarray(mm.predict(X[sil])).astype(str) == "Detracteur").mean()), 4),
                         "precision_at_k": pkm["precision_at_k"], "lift": pkm["lift"], "taux_de_base": pkm["taux_de_base"]}
        S = pd.DataFrame(coefs).dropna()
        top = S.loc[S["M3"].abs().sort_values(ascending=False).head(12).index]
        instables = [v for v in top.index if len(set(np.sign(top.loc[v].values))) > 1]
        R["sensibilite_mapping"] = {
            "methode": "modèle retenu réajusté sous chaque mapping, mêmes répondants S3 ; contributions moyennes signées agrégées par variable",
            "coefficients": top.round(4).to_dict(orient="index"),
            "n_variables_examinees": int(len(top)), "variables_instables": instables, "n_stables": int(len(top) - len(instables)),
            "effet_metier": effet,
            "avertissement": ("les kappas ne se comparent pas entre mappings — la difficulté de la tâche change ; seuls le sens des "
                              "effets et l'effet métier se comparent")}
    except Exception as e:
        R["sensibilite_mapping"] = {"erreur": f"{type(e).__name__}: {str(e)[:200]}"}

    # Critères de succès fixés dans config.yaml avant la modélisation, vérifiés un par un.
    cs = cfg["criteres_succes"]
    eco = R["evaluation_metier"]["economie"]; pk = R["evaluation_metier"]["precision_at_k"]
    rappels_classes = {c: R["modele_final"]["metriques_retenues"]["par_classe"][c]["rappel"] for c in ORDRE}
    ecarts_eq = {d: v["ecart_max"] for d, v in R["audit_equite"].items()}
    verif = [
        {"critere": "lift ≥ " + str(cs["lift_min"]), "valeur": pk["lift"], "seuil": cs["lift_min"],
         "atteint": bool(pk["lift"] >= cs["lift_min"])},
        {"critere": "précision@" + str(pk["k"]) + " ≥ " + f"{cs['precision_at_k_min']:.0%}", "valeur": pk["precision_at_k"],
         "seuil": cs["precision_at_k_min"], "atteint": bool(pk["precision_at_k"] >= cs["precision_at_k_min"])},
        {"critere": "gain net > " + str(cs["gain_net_min_eur"]) + " €", "valeur": eco["gain_net_eur"],
         "seuil": cs["gain_net_min_eur"], "atteint": bool(eco["gain_net_eur"] > cs["gain_net_min_eur"])},
        {"critere": "rappel ≥ " + f"{cs['rappel_min_par_classe']:.0%}" + " pour les trois classes",
         "valeur": round(min(rappels_classes.values()), 4), "seuil": cs["rappel_min_par_classe"],
         "atteint": bool(min(rappels_classes.values()) >= cs["rappel_min_par_classe"]), "detail": rappels_classes},
        {"critere": "écart d'équité signalé au-delà de " + f"{cs['ecart_equite_signalement']:.0%}",
         "valeur": round(max(ecarts_eq.values()), 4) if ecarts_eq else None, "seuil": cs["ecart_equite_signalement"],
         "atteint": True, "signalement": {d: e for d, e in ecarts_eq.items() if e > cs["ecart_equite_signalement"]},
         "note": "critère de signalement, pas de rejet : un écart supérieur au seuil doit être publié et arbitré, jamais masqué"},
    ]
    R["criteres_succes"] = {"definis_a_l_etape_1": cs, "verification": verif,
                            "tous_atteints": bool(all(v["atteint"] for v in verif))}

    # ======================================================================== [8] persistance
    print("[8/8] Persistance")
    traite = RACINE / cfg["chemins"]["traite"]; modeles = RACINE / cfg["chemins"]["modeles"]; rapports = RACINE / cfg["chemins"]["rapports"]
    for d in (traite, modeles, rapports):
        d.mkdir(parents=True, exist_ok=True)
    sortie = df.copy()
    for mp in ("M1", "M2", "M3"):
        sortie[f"cible_{mp}"] = cibles[mp].astype(str)
    sortie["cible_profil_detracteur_si_ambigu"] = cibles["profil_detracteur_si_ambigu"]
    sortie["repondant_S3"] = rep
    sortie["proba_detracteur"] = final_cal.predict_proba(X)[:, i_det]
    sortie["prediction"] = pred_tous
    sortie["levier_recommande"] = leviers.values
    sortie["levier_secondaire"] = leviers2.values
    sortie["tranche_age"] = pd.cut(sortie["Age"], [0, 30, 45, 60, 120], right=False,
                                   labels=["<30", "30-45", "45-60", "60+"]).astype(str)
    sortie.to_parquet(traite / "clients_scores.parquet", index=False)

    joblib.dump({"pipeline": final_cal, "pipeline_brut": final, "pipeline_decision": pipeline_decision, "classes": classes,
                 "num": num, "cat": cat, "mapping": mp_final, "scenario": sc_final, "seed": seed, "modele": md_final,
                 "famille": FAMILLE[md_final], "version": comparatif[md_final]["version_retenue"],
                 "type_explication": genre_explication(final), "mediane_charge_par_service": med_charge},
                modeles / "modele_final.joblib")

    R["duree_s"] = round(time.time() - t0, 1)
    with open(rapports / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(R), f, ensure_ascii=False, indent=2)
    print(f"\nTerminé en {R['duree_s']} s — reports/resultats.json, models/modele_final.joblib, data/processed/clients_scores.parquet")
    return R


if __name__ == "__main__":
    main()
