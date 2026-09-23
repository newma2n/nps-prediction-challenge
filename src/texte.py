"""Étape 21 — le texte apporte-t-il quelque chose ? Et pourquoi la réponse est piégée.

    python -m src.texte

Trois représentations (TF-IDF · TF-IDF+SVD en fusion précoce · modèle texte seul), deux
stratégies de fusion avec le modèle tabulaire, un gain mesuré AU-DESSUS du tabulaire.

LIMITE LOGIQUE, à lire avant les chiffres : les verbatims sont générés à partir de la CLASSE
(tonalité) et des features. Le texte contient donc, par construction, une information sur le
label que les features tabulaires n'ont pas — c'est le label lui-même, bruité à 25 %. Le gain
mesuré est un plafond artificiel : il démontre que la chaîne texte → fusion fonctionne, il ne
prouve RIEN sur la valeur de vrais verbatims en production. Le plafond théorique d'un texte
parfaitement lu est ~ (1 − bruit) + bruit/3 ≈ 83 % de concordance avec la classe.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .data import RACINE, charger_config
from .evaluate import metriques
from .target import ORDRE


def main():
    cfg = charger_config(); seed = cfg["seed"]
    art = joblib.load(RACINE / "models" / "modele_final.joblib")
    df = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet")
    v = pd.read_parquet(RACINE / "data" / "processed" / "verbatims.parquet").set_index("Customer ID")
    df["verbatim"] = df["Customer ID"].map(v["verbatim"])
    sources = v["source"].value_counts().to_dict()
    source = " + ".join(f"{k} ({n})" for k, n in sources.items())
    modele_llm = next((str(x) for x in v["modele"].dropna().unique()), None) if "modele" in v else None

    rep = df["repondant_S3"].values; sil = ~rep
    y = df["cible_M3"].astype(str).values
    num, cat, classes = art["num"], art["cat"], art["classes"]
    X_tab = df[num + cat]

    # --- modèle tabulaire (référence) --------------------------------------------
    p_tab = art["pipeline"].predict_proba(X_tab)          # calibré, pour la fusion
    pred_tab = np.asarray(art["pipeline_decision"].predict(X_tab))
    m_tab = metriques(y[sil], pred_tab[sil])

    # --- modèle texte seul : TF-IDF 1-2 grammes -----------------------------------
    texte = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0, random_state=seed))])
    texte.fit(df.loc[rep, "verbatim"], y[rep])
    p_txt = texte.predict_proba(df["verbatim"])
    ordre_txt = [list(texte.classes_).index(c) for c in classes]; p_txt = p_txt[:, ordre_txt]
    m_txt = metriques(y[sil], np.array(classes)[p_txt[sil].argmax(1)])

    # --- fusion tardive : poids choisi par validation croisée sur les répondants ----
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p_txt_cv = cross_val_predict(texte, df.loc[rep, "verbatim"], y[rep], cv=skf, method="predict_proba")
    p_txt_cv = p_txt_cv[:, [list(sorted(set(y[rep]))).index(c) for c in classes]]
    p_tab_cv = p_tab[rep]
    meilleur_w, meilleur_k = 0.0, -1
    grille_w = {}
    for w in np.linspace(0, 1, 11):
        pf = (1 - w) * p_tab_cv + w * p_txt_cv
        k = metriques(y[rep], np.array(classes)[pf.argmax(1)])["kappa_quadratique"]
        grille_w[round(float(w), 1)] = round(k, 4)
        if k > meilleur_k:
            meilleur_k, meilleur_w = k, float(w)
    p_fus = (1 - meilleur_w) * p_tab + meilleur_w * p_txt
    m_fus = metriques(y[sil], np.array(classes)[p_fus[sil].argmax(1)])

    # --- fusion précoce : features tabulaires préparées + SVD(20) du TF-IDF ----------
    prep = art["pipeline_brut"].named_steps["preparation"]
    Xt = prep.transform(X_tab)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit(df.loc[rep, "verbatim"])
    svd = TruncatedSVD(20, random_state=seed).fit(tfidf.transform(df.loc[rep, "verbatim"]))
    Z = svd.transform(tfidf.transform(df["verbatim"]))
    Xe = np.hstack([Xt, Z])
    precoce = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed).fit(Xe[rep], y[rep])
    m_pre = metriques(y[sil], precoce.predict(Xe[sil]))

    # --- concordance réelle tonalité / classe dans les verbatims --------------------
    TON = {"Detracteur": "frustré", "Passif": "neutre", "Promoteur": "enthousiaste"}
    conc = float((df["Customer ID"].map(v["tonalite_visee"]).values == df["cible_M3"].map(TON).values).mean())

    def resume(m):
        return {"kappa": round(m["kappa_quadratique"], 4), "macro_f1": round(m["macro_f1"], 4),
                "rappel_detracteur": round(m["par_classe"]["Detracteur"]["rappel"], 4),
                "rappel_passif": round(m["par_classe"]["Passif"]["rappel"], 4)}

    res = {
        "source_verbatims": source, "sources_detail": sources, "modele_llm": modele_llm, "n_verbatims": int(df["verbatim"].notna().sum()),
        "concordance_tonalite_classe": round(conc, 4), "plafond_theorique": round(0.75 + 0.25 / 3, 4),
        "tabulaire_seul": resume(m_tab), "texte_seul": resume(m_txt),
        "fusion_tardive": {"poids_texte": meilleur_w, "grille_poids_cv": grille_w, **resume(m_fus)},
        "fusion_precoce_svd20": resume(m_pre),
        "gain_kappa_fusion_tardive": round(m_fus["kappa_quadratique"] - m_tab["kappa_quadratique"], 4),
        "lecture": ("Le texte encode la classe par construction (tonalité tirée de la classe, bruit 25 %). "
                    "Tout gain est un artefact du protocole de génération : il valide la chaîne, pas la valeur "
                    "de vrais verbatims."),
    }
    with open(RACINE / "reports" / "texte.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    joblib.dump({"pipeline_texte": texte, "classes": classes, "poids_texte": meilleur_w},
                RACINE / "models" / "modele_texte.joblib")
    print(f"tabulaire kappa={m_tab['kappa_quadratique']:.3f} · texte seul={m_txt['kappa_quadratique']:.3f} · "
          f"fusion tardive (w={meilleur_w:.1f})={m_fus['kappa_quadratique']:.3f} · précoce={m_pre['kappa_quadratique']:.3f}")


if __name__ == "__main__":
    main()
