"""Étape 20 — modèle de fondation tabulaire (bonus § 4.5).

    python -m src.tabpfn_eval

TabPFN sur le même découpage S3/M3 que le modèle final, mêmes features préparées.
Consigne de l'énoncé : « do not frame it as the winner if it is not ». On mesure performance,
latence et empreinte, et on écrit ce qu'on trouve.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from .data import RACINE, charger_config
from .evaluate import metriques


def main():
    cfg = charger_config()
    art = joblib.load(RACINE / "models" / "modele_final.joblib")
    df = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet")
    rep = df["repondant_S3"].values; sil = ~rep
    y = df["cible_M3"].astype(str).values
    prep = art["pipeline_brut"].named_steps["preparation"]
    Xt = prep.transform(df[art["num"] + art["cat"]]).astype(np.float32)

    res: dict = {"n_entrainement": int(rep.sum()), "n_evaluation": int(sil.sum()), "n_features": int(Xt.shape[1])}
    try:
        import tabpfn
        from tabpfn import TabPFNClassifier
        res["version"] = getattr(tabpfn, "__version__", "?")
        t0 = time.time()
        clf = TabPFNClassifier(device="cpu")
        clf.fit(Xt[rep], y[rep])
        res["duree_fit_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        proba = clf.predict_proba(Xt[sil])
        res["duree_inference_s"] = round(time.time() - t0, 2)
        res["latence_par_client_ms"] = round(1000 * res["duree_inference_s"] / int(sil.sum()), 3)
        classes = list(clf.classes_)
        pred = np.array(classes)[proba.argmax(1)]
        m = metriques(y[sil], pred)
        res["metriques"] = {"kappa": round(m["kappa_quadratique"], 4), "macro_f1": round(m["macro_f1"], 4),
                            "par_classe": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                                           for k, v in m["par_classe"].items()}}
        res["statut"] = "ok"
    except Exception as e:
        res["statut"] = "indisponible"
        res["erreur"] = f"{type(e).__name__}: {str(e)[:300]}"

    # Point de comparaison : le modèle final et sa latence
    t0 = time.time(); _ = art["pipeline"].predict_proba(df.loc[sil, art["num"] + art["cat"]])
    res["reference_logistique"] = {"kappa": round(json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
                                                  ["modele_final"]["metriques_retenues"]["kappa_quadratique"], 4),
                                   "duree_inference_s": round(time.time() - t0, 3)}
    with open(RACINE / "reports" / "tabpfn.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "metriques"}, ensure_ascii=False))
    if "metriques" in res:
        print("kappa TabPFN =", res["metriques"]["kappa"], "| kappa logistique =", res["reference_logistique"]["kappa"])


if __name__ == "__main__":
    main()
