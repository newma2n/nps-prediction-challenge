"""Étape 20 — modèles de fondation tabulaires (bonus § 4.5 de l'énoncé).

    python -m src.fondation_eval

Trois modèles évalués sur le même découpage S3/M3 et les mêmes features préparées que le modèle
final, plus deux points de comparaison (le modèle retenu, et le meilleur boosting du catalogue —
c'est le « trade-off vs gradient boosting » que l'énoncé demande de chiffrer) :

  - TabICL       (Qu et al., ICML 2025) — poids publics sur Hugging Face.
  - TabPFN v2    (Hollmann et al., Nature 2025) — poids publics `Prior-Labs/TabPFN-v2-clf`,
                 chargés explicitement via `model_path`, donc sans acceptation de licence.
  - TabPFN 2.5   (PriorLabs, arXiv 2511.08667) — le paquet `tabpfn` 9.x sert ce modèle sous le
                 nom de dépôt `Prior-Labs/tabpfn_3_5` (le numéro du dépôt suit la génération
                 interne, pas le nom commercial : c'est bien la 2.5). Ce dépôt n'est PAS « gated »
                 côté Hugging Face ; ce qui bloque est la licence PriorLabs, dont l'acceptation
                 passe par un compte et une clé API. Si `TABPFN_TOKEN` est présent (dans `.env`,
                 jamais dans le dépôt), la 2.5 est évaluée aussi ; sinon elle est déclarée
                 indisponible avec la cause exacte et la trace brute, sans extrapolation.

Consigne de l'énoncé : « do not frame it as the winner if it is not ». On mesure la performance,
la latence d'ajustement et d'inférence, et on écrit ce qu'on trouve.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from .data import RACINE, charger_config
from .evaluate import metriques, resume
from .features import pipeline_preparation
from .models import construire_modeles, entrainer

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Sans ce garde-fou, `tabpfn` attend une clé API au clavier et échoue sur une erreur illisible en
# exécution non interactive.
os.environ.setdefault("TABPFN_NO_BROWSER", "1")

DEPOT_TABPFN_V2 = "Prior-Labs/TabPFN-v2-clf"
FICHIER_TABPFN_V2 = "tabpfn-v2-classifier.ckpt"


def _version(paquet: str) -> str:
    """Version réellement installée, lue dans les métadonnées de distribution.

    `module.__version__` n'existe pas dans tous les paquets (TabICL n'en expose pas) et renvoyait
    « ? » : un tableau de bonus qui ne sait pas dire quelle version il a testée ne vaut rien.
    """
    try:
        return importlib.metadata.version(paquet)
    except Exception:
        return "non installé"


def _diagnostic(err: Exception) -> dict:
    """Sépare ce qu'on SAIT (la trace brute) de ce qu'on INTERPRÈTE (la cause probable).

    La version précédente concluait « dépôt Hugging Face à accès contrôlé » dès que le mot
    « socket » apparaissait. C'était faux : `Prior-Labs/tabpfn_3_5` est public (l'API HF répond
    `gated: false`), et l'erreur venait du flux de licence PriorLabs, interactif. On distingue
    donc désormais la cause, l'action attendue, et la trace exacte.
    """
    brut = f"{type(err).__name__}: {err}"
    e = brut.lower()
    if "licen" in e or "tabpfnlicense" in e:
        return {"cause": "licence PriorLabs non acceptée : le téléchargement des poids exige un compte PriorLabs et une clé API",
                "action": "créer un compte sur ux.priorlabs.ai, accepter la licence, puis renseigner TABPFN_TOKEN dans .env (hors dépôt)",
                "erreur_brute": brut[:400]}
    if "winerror 10038" in e or ("select" in e and "socket" in e):
        return {"cause": "flux d'authentification interactif : la bibliothèque attend une clé API au clavier, ce qui échoue en exécution non interactive sous Windows",
                "action": "définir TABPFN_NO_BROWSER=1 pour obtenir une erreur de licence explicite, puis fournir TABPFN_TOKEN",
                "erreur_brute": brut[:400]}
    if "401" in e or "403" in e or "gated" in e:
        return {"cause": "accès refusé au dépôt de poids (HTTP 401/403)",
                "action": "vérifier l'acceptation des conditions du dépôt et `huggingface-cli login`",
                "erreur_brute": brut[:400]}
    if "connect" in e or "timeout" in e or "resolve" in e:
        return {"cause": "poids non téléchargeables (réseau)", "action": "réessayer avec un accès réseau",
                "erreur_brute": brut[:400]}
    return {"cause": "échec non catégorisé — voir la trace", "action": "lire la trace brute", "erreur_brute": brut[:400]}


def _evaluer(nom: str, constructeur, Xr, yr, Xs, ys) -> dict:
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
        res["par_classe"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                             for k, v in m["par_classe"].items()}
        res["statut"] = "ok"
    except Exception as e:
        res["statut"] = "indisponible"
        res.update(_diagnostic(e))
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

    # Ces modèles rejouent le jeu d'entraînement à chaque prédiction : scorer tous les silencieux
    # sur processeur prend plusieurs minutes par modèle. L'évaluation porte donc sur un échantillon
    # fixé par la graine, le même pour tous les modèles comparés.
    N_ECH = int(os.environ.get("NPS_FONDATION_N", "1500"))
    idx_sil = np.flatnonzero(sil)
    if 0 < N_ECH < len(idx_sil):
        idx_sil = np.sort(np.random.default_rng(seed).choice(idx_sil, N_ECH, replace=False))
    masque_eval = np.zeros(len(df), dtype=bool); masque_eval[idx_sil] = True
    Xs = np.asarray(prep.transform(df.loc[masque_eval, num + cat]), dtype=np.float32)
    sil_eval = masque_eval

    R = json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
    res: dict = {"n_entrainement": int(rep.sum()), "n_evaluation": int(sil_eval.sum()),
                 "n_silencieux_total": int(sil.sum()),
                 "echantillonnage": (f"évaluation sur {int(sil_eval.sum())} silencieux tirés au hasard (graine {seed}) parmi "
                                     f"{int(sil.sum())} : les modèles de fondation rejouent le jeu d'entraînement à chaque "
                                     f"prédiction, et l'objet de cette section est de comparer des modèles entre eux, sur le "
                                     f"même échantillon, pas de produire des prédictions de production"
                                     if int(sil_eval.sum()) < int(sil.sum()) else "évaluation sur tous les silencieux"),
                 "n_features": int(Xr.shape[1]),
                 "materiel": "CPU", "versions": {p: _version(p) for p in ("tabicl", "tabpfn", "torch", "lightgbm")},
                 "note_nom_tabpfn": ("le paquet `tabpfn` 9.x sert le modèle TabPFN-2.5 ; son dépôt de poids s'appelle "
                                     "`Prior-Labs/tabpfn_3_5` (numérotation interne). Ce dépôt est public : ce qui bloque "
                                     "est l'acceptation de la licence PriorLabs, pas un contrôle d'accès Hugging Face."),
                 "modeles": []}

    def tabicl():
        from tabicl import TabICLClassifier
        return TabICLClassifier(device="cpu", random_state=seed)

    def tabpfn_v2():
        """Poids publics, chargés explicitement : aucune licence interactive n'est déclenchée."""
        from huggingface_hub import hf_hub_download
        from tabpfn import TabPFNClassifier
        chemin = hf_hub_download(DEPOT_TABPFN_V2, FICHIER_TABPFN_V2)
        return TabPFNClassifier(device="cpu", random_state=seed, model_path=chemin, ignore_pretraining_limits=True)

    def tabpfn_25():
        """Dernière génération : nécessite TABPFN_TOKEN (licence PriorLabs acceptée)."""
        from tabpfn import TabPFNClassifier
        if not os.environ.get("TABPFN_TOKEN"):
            raise RuntimeError("TabPFNLicenseError : TABPFN_TOKEN absent — licence PriorLabs non acceptée sur cette machine")
        return TabPFNClassifier(device="cpu", random_state=seed, ignore_pretraining_limits=True)

    candidats = (("TabICL", "tabicl", tabicl),
                 ("TabPFN v2", "tabpfn", tabpfn_v2),
                 ("TabPFN 2.5", "tabpfn", tabpfn_25))
    for nom, paquet, ctor in candidats:
        r = _evaluer(nom, ctor, Xr, y[rep], Xs, y[sil_eval])
        r["version_paquet"] = _version(paquet)
        r["poids"] = {"TabICL": "poids publics Hugging Face (jingang/TabICL)",
                      "TabPFN v2": f"{DEPOT_TABPFN_V2} / {FICHIER_TABPFN_V2} (public, sans licence à accepter)",
                      "TabPFN 2.5": "Prior-Labs/tabpfn_3_5 (public, mais licence PriorLabs à accepter : compte + TABPFN_TOKEN)"}[nom]
        res["modeles"].append(r)
        print(f"{nom:12s} {r['statut']:12s} kappa={r.get('metriques', {}).get('kappa')} {r.get('cause', '')[:90]}")

    # ---------------------------------------------------------------- points de comparaison
    # (1) Le modèle retenu : performance et latence d'inférence sur les mêmes silencieux.
    t0 = time.time(); _ = art["pipeline"].predict_proba(df.loc[sil_eval, num + cat])
    duree_inf_ref = round(time.time() - t0, 3)
    mf = R["modele_final"]
    res["reference"] = {"modele": mf["modele"], "kappa": round(mf["metriques_retenues"]["kappa_quadratique"], 4),
                        "macro_f1": round(mf["metriques_retenues"]["macro_f1"], 4),
                        "rappel_detracteur": round(mf["metriques_retenues"]["par_classe"]["Detracteur"]["rappel"], 4),
                        "duree_inference_s": duree_inf_ref,
                        "latence_par_client_ms": round(1000 * duree_inf_ref / int(sil_eval.sum()), 3)}

    # (2) Le meilleur boosting du catalogue, RÉAJUSTÉ ET CHRONOMÉTRÉ ici : l'énoncé demande le
    #     compromis coût / latence « vs gradient boosting ». Reprendre le kappa du comparatif sans
    #     mesurer son temps laissait la moitié de la comparaison vide.
    boost = [c for c in R["selection_modele_final"]["classement"] if c["famille"] == "Boosting"]
    if boost:
        b = max(boost, key=lambda c: c["kappa_silencieux"])
        ref_b = {"modele": b["modele"], "kappa": b["kappa_silencieux"], "macro_f1": b["macro_f1_silencieux"]}
        try:
            sat = pd.to_numeric(df.loc[rep, cfg["colonnes"]["cible"]]).values.astype(float)
            pipe_b = construire_modeles(num, cat, seed, [b["modele"]])[b["modele"]]
            t0 = time.time(); entrainer(pipe_b, df.loc[rep, num + cat], y[rep], y_continu=sat)
            ref_b["duree_fit_s"] = round(time.time() - t0, 2)
            t0 = time.time(); _ = pipe_b.predict_proba(df.loc[sil_eval, num + cat])
            ref_b["duree_inference_s"] = round(time.time() - t0, 3)
            ref_b["latence_par_client_ms"] = round(1000 * ref_b["duree_inference_s"] / int(sil_eval.sum()), 3)
        except Exception as e:
            ref_b["erreur_chronometrage"] = f"{type(e).__name__}: {str(e)[:200]}"
        res["reference_boosting"] = ref_b

    ok = [m for m in res["modeles"] if m["statut"] == "ok"]
    if ok:
        meilleur = max(ok, key=lambda m: m["metriques"]["kappa"])
        res["lecture"] = {
            "meilleur_fondation": meilleur["nom"], "kappa": meilleur["metriques"]["kappa"],
            "ecart_au_modele_retenu": round(meilleur["metriques"]["kappa"] - res["reference"]["kappa"], 4),
            "n_evaluation": int(sil_eval.sum()),
            "rapport_de_latence": (round(meilleur["latence_par_client_ms"] / res["reference"]["latence_par_client_ms"], 1)
                                   if res["reference"]["latence_par_client_ms"] else None),
            "conclusion": ("aucun modèle de fondation ne dépasse le modèle retenu sur ce jeu ; ils coûtent en plus une latence "
                           "d'inférence de plusieurs ordres de grandeur supérieure"
                           if meilleur["metriques"]["kappa"] <= res["reference"]["kappa"] else
                           "un modèle de fondation dépasse le modèle retenu en kappa — à mettre en regard de sa latence")}

    with open(RACINE / "reports" / "fondation.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("reports/fondation.json écrit")


if __name__ == "__main__":
    main()
