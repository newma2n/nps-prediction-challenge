"""Orchestrateur — rejoue le pipeline complet, des fichiers bruts au modèle final.

    python -m src.run

Produit : data/processed/*.parquet, models/modele_final.joblib,
          reports/resultats.json, reports/*.md
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import RACINE, charger_config, preparer, registre_fuites
from .evaluate import audit_equite, metriques, precision_at_k, valeur_economique
from .features import colonnes_modele, construire, noms_apres_encodage
from .models import OrdinalParSeuils, calibrer, construire_modeles, entrainer
from .protocol import decouper, diagnostic
from .target import ORDRE, construire_cibles

warnings.filterwarnings("ignore")


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Index, pd.Series)):
        return o.tolist()
    return o


def main() -> dict:
    t0 = time.time()
    cfg = charger_config()
    seed = cfg["seed"]
    R: dict = {"seed": seed}

    # ---- Phases 2-3 : données, qualité, fuites, cible, features -----------------
    print("[1/7] Ingestion, qualité, registre de fuites")
    df, qualite = preparer(cfg)
    df = construire(df)
    R["qualite"] = qualite
    R["registre_fuites"] = registre_fuites(cfg)

    print("[2/7] Construction de la cible et arbitrage empirique des « 3 »")
    cibles, arb = construire_cibles(df, cfg, seed)
    R["arbitrage_des_3"] = {k: v for k, v in arb.items()
                            if k not in ("modele", "index_ambigus", "proba_detracteur_des_3")}

    # Analyse de Offer — la quasi-expérience du dataset (étape 6.7)
    off = df.groupby("Offer").agg(n=("Customer ID", "count"),
                                  taux_depart=("Churn Value", "mean"),
                                  satisfaction=("Satisfaction Score", "mean"))
    R["analyse_offer"] = off.round(3).reset_index().to_dict(orient="records")

    num, cat = colonnes_modele(df, cfg)
    R["features"] = {"numeriques": num, "categorielles": cat}
    X = df[num + cat]
    satisfaction = pd.to_numeric(df[cfg["colonnes"]["cible"]]).values

    # ---- Phase 4 : protocole 15/85 × mappings × modèles ------------------------
    print("[3/7] Protocole 15/85 — trois scénarios × trois mappings × trois modèles")
    grille = []
    R["diagnostic_protocole"] = []
    modeles_entraines = {}

    for scenario in cfg["protocole"]["scenarios"]:
        dec = decouper(df, cfg, scenario, seed)
        R["diagnostic_protocole"].append(diagnostic(df, cfg, dec))
        rep, sil = dec["repondants"], dec["silencieux"]

        for mapping in ("M1", "M2", "M3"):
            y = cibles[mapping].astype(str).values
            for nom, modele in construire_modeles(num, cat, seed).items():
                try:
                    entrainer(modele, X[rep], y[rep], y_continu=satisfaction[rep])
                    pred = modele.predict(X[sil])
                    m = metriques(y[sil], pred)
                except Exception as e:  # on trace, on ne masque pas
                    m = {"erreur": f"{type(e).__name__}: {e}"}
                m.update({"scenario": scenario, "mapping": mapping, "modele": nom})
                grille.append(m)
                modeles_entraines[(scenario, mapping, nom)] = modele
                if "kappa_quadratique" in m:
                    print(f"   {scenario:8s} {mapping} {nom:20s} "
                          f"kappa={m['kappa_quadratique']:.3f} "
                          f"macroF1={m['macro_f1']:.3f} "
                          f"rappelDet={m['par_classe']['Detracteur']['rappel']:.3f}")
    R["grille"] = grille

    # ---- Diagnostic de fuite (étape 17) : écart arbres / linéaire sous S1 -------
    def _get(sc, mp, md, cle):
        for g in grille:
            if g["scenario"] == sc and g["mapping"] == mp and g["modele"] == md:
                return g.get(cle)
    ecart = (_get("S1_MCAR", "M3", "boosting_hgb", "exactitude") or 0) - \
            (_get("S1_MCAR", "M3", "baseline_logistique", "exactitude") or 0)
    R["diagnostic_fuite"] = {
        "exactitude_max_observee": max(g.get("exactitude", 0) for g in grille),
        "seuil_alarme": 0.85,
        "ecart_arbres_lineaire_S1_M3": round(ecart, 4),
        "alarme": bool(max(g.get("exactitude", 0) for g in grille) > 0.85 or ecart > 0.30),
    }

    # ---- Modèle final : mapping M3, scénario réaliste S3, modèle CHOISI PAR LA MESURE
    # On ne présuppose pas que le boosting gagne. Le meilleur kappa quadratique
    # (puis macro-F1 en départage) sous le scénario réaliste l'emporte.
    print("[4/7] Modèle final, calibration, évaluation métier")
    sc_final, mp_final = "S3_MNAR", "M3"
    candidats = [g for g in grille if g["scenario"] == sc_final and g["mapping"] == mp_final
                 and "kappa_quadratique" in g]
    gagnant = max(candidats, key=lambda g: (round(g["kappa_quadratique"], 3), g["macro_f1"]))
    md_final = gagnant["modele"]
    R["selection_modele_final"] = {
        "critere": "kappa quadratique puis macro-F1, scénario S3_MNAR, mapping M3",
        "classement": sorted(
            [{"modele": g["modele"], "kappa": round(g["kappa_quadratique"], 4),
              "macro_f1": round(g["macro_f1"], 4),
              "rappel_detracteur": round(g["par_classe"]["Detracteur"]["rappel"], 4)}
             for g in candidats], key=lambda d: -d["kappa"]),
        "retenu": md_final,
    }
    print(f"   modèle retenu par la mesure : {md_final}")
    dec = decouper(df, cfg, sc_final, seed)
    rep, sil = dec["repondants"], dec["silencieux"]
    y = cibles[mp_final].astype(str).values

    final = modeles_entraines[(sc_final, mp_final, md_final)]

    # ---- Étape 22 : recherche d'hyperparamètres sur les familles réglables, S3/M3.
    # Sélection équitable : réglé contre réglé. Le meilleur sur les silencieux l'emporte
    # seulement s'il bat la version par défaut — sinon on garde le défaut et on le dit.
    print("[4b/7] Hyperparamètres")
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer, cohen_kappa_score
    from .models import espace_recherche
    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
    R["hyperparametres"] = {}
    for nom in ("baseline_logistique", "boosting_hgb"):
        espace = espace_recherche(nom)
        base = construire_modeles(num, cat, seed)[nom]
        try:
            rs = RandomizedSearchCV(base, espace, n_iter=12 if nom == "baseline_logistique" else 20,
                                    scoring=kappa_scorer, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                    random_state=seed, n_jobs=1, refit=True)
            rs.fit(X[rep], y[rep])
            m_def = metriques(y[sil], modeles_entraines[(sc_final, mp_final, nom)].predict(X[sil]))
            m_reg = metriques(y[sil], rs.best_estimator_.predict(X[sil]))
            R["hyperparametres"][nom] = {
                "meilleurs_params": {k.replace("modele__", ""): (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                     for k, v in rs.best_params_.items()},
                "kappa_cv_interne": round(float(rs.best_score_), 4),
                "kappa_silencieux_defaut": round(m_def["kappa_quadratique"], 4),
                "kappa_silencieux_regle": round(m_reg["kappa_quadratique"], 4),
                "macro_f1_silencieux_defaut": round(m_def["macro_f1"], 4),
                "macro_f1_silencieux_regle": round(m_reg["macro_f1"], 4),
                "n_configurations": int(len(rs.cv_results_["params"])),
            }
            gain = m_reg["kappa_quadratique"] - m_def["kappa_quadratique"]
            R["hyperparametres"][nom]["gain_kappa"] = round(gain, 4)
            if nom == md_final and gain > 0.005:
                final = rs.best_estimator_
                R["hyperparametres"][nom]["retenu"] = "version réglée"
            else:
                R["hyperparametres"][nom]["retenu"] = "version par défaut" if nom == md_final else "non retenu (famille non finale)"
            print(f"   {nom:20s} kappa défaut {m_def['kappa_quadratique']:.3f} -> réglé {m_reg['kappa_quadratique']:.3f}")
        except Exception as e:
            R["hyperparametres"][nom] = {"erreur": f"{type(e).__name__}: {e}"}

    # ---- Effet des poids IPW (étape 16.3) : même modèle, repondéré par l'inverse
    # de la propension à répondre. On mesure l'effet correctif au lieu de le supposer.
    try:
        final_ipw = construire_modeles(num, cat, seed)[md_final]
        etape = final_ipw.named_steps["modele"]
        if isinstance(etape, OrdinalParSeuils):
            etape._y_continu = satisfaction[rep].astype(float)
        final_ipw.fit(X[rep], y[rep], modele__sample_weight=dec["poids_ipw"][rep])
        m_ipw = metriques(y[sil], final_ipw.predict(X[sil]))
        m_sans = metriques(y[sil], final.predict(X[sil]))
        R["effet_ipw"] = {
            "sans_ipw": {"kappa": round(m_sans["kappa_quadratique"], 4),
                         "macro_f1": round(m_sans["macro_f1"], 4)},
            "avec_ipw": {"kappa": round(m_ipw["kappa_quadratique"], 4),
                         "macro_f1": round(m_ipw["macro_f1"], 4)},
        }
        if m_ipw["kappa_quadratique"] > m_sans["kappa_quadratique"]:
            final, R["effet_ipw"]["retenu"] = final_ipw, "avec IPW"
        else:
            R["effet_ipw"]["retenu"] = "sans IPW"
        print(f"   effet IPW : kappa {m_sans['kappa_quadratique']:.3f} -> "
              f"{m_ipw['kappa_quadratique']:.3f} ({R['effet_ipw']['retenu']})")
    except Exception as e:
        R["effet_ipw"] = {"erreur": f"{type(e).__name__}: {e}"}

    # ---- Calibration, puis VÉRIFICATION qu'elle n'a pas détruit une classe --------
    final_cal = calibrer(final, X[rep], y[rep], seed, methode="sigmoid") or final
    classes = list(final_cal.classes_)
    i_det = classes.index("Detracteur")
    proba_sil = final_cal.predict_proba(X[sil])
    pred_cal = np.array(classes)[proba_sil.argmax(axis=1)]
    pred_brut = np.asarray(final.predict(X[sil]))
    m_cal, m_brut = metriques(y[sil], pred_cal), metriques(y[sil], pred_brut)

    # Sous MNAR, les Passifs sont rares parmi les répondants : un calibrateur ajusté
    # sur peu de lignes peut ne plus jamais prédire la classe intermédiaire. Si c'est
    # le cas, on garde le modèle non calibré pour la CLASSE et le modèle calibré pour
    # la PROBABILITÉ de détraction, seule utilisée pour classer les appels.
    passif_ecrase = m_cal["par_classe"]["Passif"]["rappel"] < 0.05
    if passif_ecrase or m_cal["macro_f1"] < m_brut["macro_f1"] - 0.03:
        pipeline_decision, pred_sil = final, pred_brut
        strategie = ("hybride — classe : modèle non calibré (la calibration écrasait la classe "
                     "Passif) ; P(Détracteur) : modèle calibré, pour le classement des appels")
    else:
        pipeline_decision, pred_sil = final_cal, pred_cal
        strategie = "modèle calibré pour la classe et pour le classement"
    print(f"   stratégie de décision : {strategie.split(' — ')[0]}")

    R["modele_final"] = {
        "scenario": sc_final, "mapping": mp_final, "modele": md_final,
        "n_entrainement": int(rep.sum()), "n_evaluation": int(sil.sum()),
        "strategie_decision": strategie,
        "metriques_retenues": metriques(y[sil], pred_sil),
        "metriques_calibre": m_cal,
        "metriques_non_calibre": m_brut,
    }
    k = cfg["metier"]["k_appels"]
    res_k = precision_at_k(y[sil], proba_sil[:, i_det], k)
    R["evaluation_metier"] = {"precision_at_k": res_k,
                              "economie": valeur_economique(res_k, cfg)}

    # ---- Étape 31 : audit d'équité sur les dimensions exclues du modèle ---------
    print("[5/7] Audit d'équité")
    df_sil = df[sil].copy()
    df_sil["tranche_age"] = pd.cut(df_sil["Age"], [0, 30, 45, 60, 120],
                                   labels=["<30", "30-45", "45-60", "60+"]).astype(str)
    R["audit_equite"] = audit_equite(df_sil, y[sil], pred_sil,
                                     ["Gender", "Senior Citizen", "Married", "Dependents",
                                      "tranche_age"])

    # ---- Étape 29 : SHAP ---------------------------------------------------------
    print("[6/7] Interprétabilité SHAP")
    try:
        prep = final.named_steps["preparation"]
        mdl = final.named_steps["modele"]
        Xt = prep.transform(X[sil])
        noms = noms_apres_encodage(prep)
        cls = list(getattr(mdl, "classes_", ORDRE))
        j = cls.index("Detracteur")

        if hasattr(mdl, "coef_"):
            # Modèle linéaire : les coefficients SONT l'explication, exacte et
            # additive. Sur des variables standardisées, |coef| est une importance
            # comparable. Méthode : coefficients (globale) + contributions
            # coef × valeur (locale, dans l'application).
            imp = pd.Series(np.abs(mdl.coef_[j]), index=noms).sort_values(ascending=False)
            methode = "coefficients logistiques (features standardisées)"
        else:
            import shap
            expl = shap.TreeExplainer(mdl)
            sv = expl.shap_values(Xt[:1500])
            sv_det = sv[j] if isinstance(sv, list) else sv[:, :, j]
            imp = pd.Series(np.abs(sv_det).mean(axis=0), index=noms).sort_values(ascending=False)
            methode = "SHAP TreeExplainer"

        R["shap"] = {"methode": methode,
                     "importance_detracteur": imp.head(20).round(4).to_dict(),
                     "classes": cls}
    except Exception as e:
        R["shap"] = {"erreur": f"{type(e).__name__}: {e}"}

    # ---- Persistance (étape 33) --------------------------------------------------
    print("[7/7] Persistance")
    traite = RACINE / cfg["chemins"]["traite"]
    modeles = RACINE / cfg["chemins"]["modeles"]
    rapports = RACINE / cfg["chemins"]["rapports"]
    for d in (traite, modeles, rapports):
        d.mkdir(parents=True, exist_ok=True)

    sortie = df.copy()
    for m in ("M1", "M2", "M3", "profil_detracteur_si_ambigu"):
        sortie[f"cible_{m}"] = cibles[m].astype(str) if m != "profil_detracteur_si_ambigu" \
            else cibles[m]
    sortie["repondant_S3"] = rep
    proba_tous = final_cal.predict_proba(X)
    sortie["proba_detracteur"] = proba_tous[:, i_det]
    sortie["prediction"] = np.asarray(pipeline_decision.predict(X))
    sortie.to_parquet(traite / "clients_scores.parquet", index=False)

    joblib.dump({"pipeline": final_cal, "pipeline_brut": final,
                 "pipeline_decision": pipeline_decision, "classes": classes,
                 "num": num, "cat": cat, "mapping": mp_final, "scenario": sc_final,
                 "seed": seed}, modeles / "modele_final.joblib")

    R["duree_s"] = round(time.time() - t0, 1)
    with open(rapports / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(R), f, ensure_ascii=False, indent=2)

    print(f"\nTerminé en {R['duree_s']} s — reports/resultats.json, models/modele_final.joblib")
    return R


if __name__ == "__main__":
    main()
