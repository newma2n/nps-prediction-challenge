"""Étape 20 — modèles de fondation tabulaires (bonus § 4.5 de l'énoncé).

    python -m src.fondation_eval

Deux candidats cités par l'énoncé, évalués sur le même découpage S3/M3 et les mêmes features
préparées que le modèle final :
  - TabICL  (Qu et al., ICML 2025 ; poids publics sur Hugging Face, non soumis à acceptation)
  - TabPFN  (Hollmann et al., Nature 2025 ; TabPFN-2.5 : dépôt Hugging Face à accès contrôlé —
             l'acceptation des conditions et `huggingface-cli login` sont nécessaires)

Consigne de l'énoncé : « do not frame it as the winner if it is not ». On mesure performance,
latence d'ajustement et d'inférence, et on écrit ce qu'on trouve. Un modèle indisponible est
déclaré indisponible avec la raison, pas passé sous silence.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from .data import RACINE, charger_config
from .evaluate import metriques, resume
from .features import pipeline_preparation

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Sans compte Hugging Face, TabPFN tente de demander un jeton au clavier ; en exécution non
# interactive cela échoue avec une erreur obscure. Un jeton factice force une réponse HTTP claire.
os.environ.setdefault("HF_TOKEN", "hf_aucun_compte_disponible")


def _message_clair(err: str) -> str:
    e = err.lower()
    if any(k in e for k in ("gated", "api key", "socket", "401", "403", "authentication", "hf_")):
        return ("dépôt Hugging Face à accès contrôlé (gated) : l'acceptation des conditions d'utilisation et "
                "`huggingface-cli login` sont requis pour télécharger les poids — " + err[:160])
    return err


def _evaluer(nom, constructeur, Xr, yr, Xs, ys):
    res = {"nom": nom}
    try:
        t0 = time.time()
        clf = constructeur()
        clf.fit(Xr, yr)
        res["duree_fit_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        proba = clf.predict_proba(Xs)
        res["duree_inference_s"] = round(time.time() - t0, 2)
        res["latence_par_client_ms"] = round(1000 * res["duree_inference_s"] / len(Xs), 3)
        classes = [str(c) for c in clf.classes_]
        pred = np.array(classes)[proba.argmax(1)]
        m = metriques(ys, pred, proba, classes)
        res["metriques"] = resume(m)
        res["par_classe"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in m["par_classe"].items()}
        res["statut"] = "ok"
    except Exception as e:
        res["statut"] = "indisponible"
        res["erreur"] = _message_clair(f"{type(e).__name__}: {str(e)[:400]}")
    return res


def main():
    cfg = charger_config(); seed = cfg["seed"]
    art = joblib.load(RACINE / "models" / "modele_final.joblib")
    df = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet")
    rep = df["repondant_S3"].values; sil = ~rep
    y = df["cible_M3"].astype(str).values
    num, cat = art["num"], art["cat"]
    # Même préparation que la logistique (numérique, standardisée, one-hot) : les modèles de
    # fondation attendent une matrice numérique.
    prep = pipeline_preparation(num, cat, normaliser=True).fit(df.loc[rep, num + cat])
    Xr = np.asarray(prep.transform(df.loc[rep, num + cat]), dtype=np.float32)
    Xs = np.asarray(prep.transform(df.loc[sil, num + cat]), dtype=np.float32)

    R = json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
    res: dict = {"n_entrainement": int(rep.sum()), "n_evaluation": int(sil.sum()), "n_features": int(Xr.shape[1]),
                 "materiel": "CPU", "modeles": []}

    def tabicl():
        from tabicl import TabICLClassifier
        return TabICLClassifier(device="cpu", random_state=seed)

    def tabpfn():
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu", random_state=seed)

    for nom, ctor in (("TabICL", tabicl), ("TabPFN", tabpfn)):
        r = _evaluer(nom, ctor, Xr, y[rep], Xs, y[sil])
        try:
            import importlib
            mod = importlib.import_module(nom.lower())
            r["version"] = getattr(mod, "__version__", "?")
        except Exception:
            pass
        res["modeles"].append(r)
        print(nom, r["statut"], r.get("metriques", {}).get("kappa"), r.get("erreur", "")[:120])

    # Point de comparaison : le modèle final retenu et sa latence
    t0 = time.time(); _ = art["pipeline"].predict_proba(df.loc[sil, num + cat])
    mf = R["modele_final"]
    res["reference"] = {"modele": mf["modele"], "kappa": round(mf["metriques_retenues"]["kappa_quadratique"], 4),
                        "macro_f1": round(mf["metriques_retenues"]["macro_f1"], 4),
                        "rappel_detracteur": round(mf["metriques_retenues"]["par_classe"]["Detracteur"]["rappel"], 4),
                        "duree_inference_s": round(time.time() - t0, 3)}
    # Meilleur boosting du comparatif, pour le « trade-off vs gradient boosting » demandé
    boost = [c for c in R["selection_modele_final"]["classement"] if c["famille"] == "Boosting"]
    if boost:
        b = max(boost, key=lambda c: c["kappa_silencieux"])
        res["reference_boosting"] = {"modele": b["modele"], "kappa": b["kappa_silencieux"], "macro_f1": b["macro_f1_silencieux"]}

    with open(RACINE / "reports" / "fondation.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("reports/fondation.json écrit")


if __name__ == "__main__":
    main()
