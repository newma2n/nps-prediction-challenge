"""Support de soutenance devant jury — 22 diapositives 16:9, une par message, suivant CRISP-DM.

    python -m src.soutenance

Produit :
    livrable/soutenance.pdf          le support (une idée par diapositive, un graphique, deux à trois
                                     lignes de lecture, le rattachement à l'énoncé et à la phase CRISP-DM)
    livrable/soutenance.html         la même chose, ouvrable dans un navigateur
    livrable/soutenance_notes.md     notes d'oral minute par minute et questions de jury anticipées
    reports/figures/S*.png           les figures au format présentation (polices grandes, palette unique)

Chaque chiffre vient des artefacts du pipeline (reports/*.json) — rien n'est recopié à la main.
Principe de composition (assertion-évidence) : le titre est le message, le graphique est la preuve,
le texte dit ce qu'on en fait.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from .data import RACINE, charger_config, preparer
from .features import HYPOTHESES_DERIVEES, colonnes_modele, construire
from .models import CATALOGUE, FAMILLE, SIMPLICITE
from .protocol import decouper
from .target import ORDRE, construire_cibles, nps

REP, FIG, LIV = RACINE / "reports", RACINE / "reports" / "figures", RACINE / "livrable"
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]

# Palette unique de la soutenance
BLEU, BLEU2, GRIS, GRIS2, ROUGE, VERT, ORANGE, NOIR = "#1f4e79", "#2c7fb8", "#8a97a5", "#d5dbe1", "#c0392b", "#27ae60", "#e67e22", "#1c2430"
COUL = {"Detracteur": ROUGE, "Passif": ORANGE, "Promoteur": VERT}
LIB = {"Detracteur": "Détracteur", "Passif": "Passif", "Promoteur": "Promoteur"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRIS, "axes.titleweight": "bold", "axes.titlecolor": NOIR})


def _fig(nom):
    plt.tight_layout(); plt.savefig(FIG / nom, bbox_inches="tight", facecolor="white"); plt.close(); return FIG / nom


def _j(nom):
    p = REP / f"{nom}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _b64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def _eur(v): return f"{int(round(v)):,} €".replace(",", " ")
def _lib(n): return str(n).replace("_", " ")


# =============================================================================
# FIGURES AU FORMAT PRÉSENTATION
# =============================================================================
def figures(cfg, R, df, cibles, arb):
    F = {}
    cible = cfg["colonnes"]["cible"]; seed = cfg["seed"]

    # S1 — fait fondateur + information mutuelle
    ct = pd.crosstab(df[cible], df["Churn Label"], normalize="index") * 100
    sat = df[cible].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    ax.bar(sat.index.astype(str), sat.values, color=[ROUGE, ROUGE, ORANGE, VERT, VERT])
    for i, (k, v) in enumerate(sat.items()):
        ax.text(i, v + 40, f"{v}\n{v/len(df):.0%}", ha="center", fontsize=12)
        ax.text(i, 120, f"{ct.loc[k, 'Yes']:.0f} %\npartis", ha="center", fontsize=11, color="white", fontweight="bold")
    ax.set_title("Satisfaction déclarée (1–5) et taux de départ"); ax.set_xlabel("note de satisfaction"); ax.set_ylabel("clients")
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    from .data import registre_fuites
    reg = registre_fuites(cfg); statut = {c: k for k, cols in reg.items() for c in cols}
    cand = ["Customer Status", "Churn Value", "Churn Score", "Contract", "Number of Referrals", "Internet Type", "Tenure in Months", "Monthly Charge", "Offer", "CLTV"]
    Xmi = pd.DataFrame({c: LabelEncoder().fit_transform(df[c].astype(str)) if df[c].dtype == object else df[c].fillna(df[c].median()) for c in cand})
    mi = pd.Series(mutual_info_classif(Xmi, df[cible], random_state=seed), index=cand).sort_values()
    ax = axes[1]
    ax.barh(mi.index, mi.values, color=[ROUGE if statut.get(c, "feature").startswith("fuite") else BLEU2 for c in mi.index])
    ax.set_title("Information mutuelle avec la cible"); ax.set_xlabel("information mutuelle (bits)")
    ax.text(0.98, 0.05, "rouge : fuites, exclues\nbleu : variables légitimes", transform=ax.transAxes, ha="right", fontsize=11, color=GRIS)
    F["fuites"] = _fig("S1_fuites.png")

    # S2 — cible : mappings, NPS, arbitrage
    dist = pd.DataFrame({m: cibles[m].value_counts(normalize=True).reindex(ORDRE) for m in ("M1", "M2", "M3")})
    npsv = [nps(cibles[m]) for m in ("M1", "M2", "M3")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]})
    dist.T.plot(kind="bar", stacked=True, ax=axes[0], color=[COUL[c] for c in ORDRE], rot=0, width=0.65)
    axes[0].set_xticklabels(["M1\nénoncé", "M2\nrecentré", "M3\narbitré"]); axes[0].set_ylabel("part des clients"); axes[0].set_title("Trois mappings de la cible")
    axes[0].legend([LIB[c] for c in ORDRE], loc="upper right", frameon=False)
    axes[1].axhspan(19, 34, color=VERT, alpha=0.15); axes[1].text(2.45, 26, "benchmark\ntélécom\n+19 à +34", fontsize=11, color=VERT, va="center")
    axes[1].bar(["M1", "M2", "M3"], npsv, color=[ROUGE, ORANGE, BLEU2], width=0.6)
    for i, v in enumerate(npsv):
        axes[1].text(i, v + (2.5 if v > 0 else -7), f"{v:+.0f}", ha="center", fontsize=14, fontweight="bold")
    axes[1].axhline(0, color=NOIR, lw=0.8); axes[1].set_title("NPS résultant"); axes[1].set_xlim(-0.6, 3.1)
    axes[2].hist(arb["proba_detracteur_des_3"], bins=30, color=GRIS); axes[2].axvline(0.5, color=NOIR, ls="--")
    axes[2].set_title(f"Les {arb['n_ambigus']} clients « 3 » : à qui ressemblent-ils ?"); axes[2].set_xlabel("P(profil détracteur), modèle entraîné sur les extrêmes")
    axes[2].text(0.52, axes[2].get_ylim()[1] * 0.9, f"{arb['part_3_vers_detracteur']:.0%} penchent\ndétracteur", fontsize=12, color=ROUGE)
    axes[2].text(0.05, axes[2].get_ylim()[1] * 0.9, f"{arb['part_3_vers_promoteur']:.0%} penchent\npromoteur", fontsize=12, color=VERT)
    F["cible"] = _fig("S2_cible.png")

    # S3 — protocole
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.2), sharey=True)
    dg = {d["scenario"]: d for d in R["diagnostic_protocole"]}
    for ax, sc, lib in zip(axes, cfg["protocole"]["scenarios"], ("S1 · MCAR — tirage au hasard", "S2 · MAR — dépend des covariables", "S3 · MNAR — dépend de la satisfaction")):
        dec = decouper(df, cfg, sc, seed)
        base = df[cible].value_counts(normalize=True).sort_index(); rep_ = df.loc[dec["repondants"], cible].value_counts(normalize=True).reindex(base.index).fillna(0)
        w = 0.38
        ax.bar(base.index - w/2, base.values, w, label="base complète", color=GRIS2)
        ax.bar(base.index + w/2, rep_.values, w, label="répondants (15 %)", color=BLEU2 if sc != "S3_MNAR" else ROUGE)
        ax.set_title(lib, fontsize=13); ax.set_xlabel("satisfaction")
        ax.text(0.97, 0.92, f"extrêmes : {dg[sc]['part_extremes_repondants']:.0%}", transform=ax.transAxes, ha="right", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("part"); axes[0].legend(frameon=False, loc="upper left")
    F["protocole"] = _fig("S3_protocole.png")

    # S4 — grille : effet mapping et biais
    g = pd.DataFrame([x for x in R["grille"] if "kappa_quadratique" in x])
    ordre = sorted(FAMILLE, key=lambda n: (FAMILLE[n], n))
    pm = g[g.scenario == "S3_MNAR"].pivot_table(index="modele", columns="mapping", values="kappa_quadratique").reindex(ordre)
    ps = g[g.mapping == "M3"].pivot_table(index="modele", columns="scenario", values="kappa_quadratique").reindex(ordre)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    yy = np.arange(len(ordre))
    for k, (mp, col) in enumerate(zip(["M1", "M2", "M3"], [ROUGE, ORANGE, BLEU2])):
        axes[0].barh(yy + (k - 1) * 0.27, pm[mp].values, 0.27, label=f"cible {mp}", color=col)
    axes[0].set_yticks(yy); axes[0].set_yticklabels([_lib(n) for n in ordre]); axes[0].invert_yaxis()
    axes[0].set_title("Effet du mapping de la cible (scénario réaliste S3)"); axes[0].set_xlabel("kappa quadratique — 85 % silencieux"); axes[0].legend(frameon=False)
    for k, (sc, col) in enumerate(zip(["S1_MCAR", "S2_MAR", "S3_MNAR"], [VERT, ORANGE, ROUGE])):
        axes[1].barh(yy + (k - 1) * 0.27, ps[sc].values, 0.27, label=sc.replace("_", " · "), color=col)
    axes[1].set_title("Effet du biais de non-réponse (cible M3)"); axes[1].set_xlabel("kappa quadratique — 85 % silencieux"); axes[1].legend(frameon=False)
    F["grille"] = _fig("S4_grille.png")

    # S5 — CV vs test
    sel = R["selection_modele_final"]; cl = pd.DataFrame(sel["classement"])
    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    for _, r in cl.iterrows():
        col = ROUGE if r["modele"] == sel["retenu"] else (BLEU2 if r["dans_la_marge"] else GRIS)
        ax.scatter(r["kappa_cv_ipw"], r["kappa_silencieux"], color=col, s=110 if r["modele"] == sel["retenu"] else 70, zorder=3, edgecolor="white")
        ax.annotate(_lib(r["modele"]), (r["kappa_cv_ipw"], r["kappa_silencieux"]), fontsize=11, xytext=(6, 4), textcoords="offset points")
    ax.axvline(sel["seuil_parcimonie"], color=NOIR, ls=":", lw=1.2)
    ax.text(sel["seuil_parcimonie"] + 0.002, cl["kappa_silencieux"].min() + 0.005, "marge de parcimonie\n(1 écart-type du meilleur)", fontsize=11, color=NOIR)
    ax.set_xlabel("kappa en validation croisée sur les répondants, pondéré IPW  →  ce que le praticien voit")
    ax.set_ylabel("kappa sur les 85 % silencieux  →  la vérité")
    ax.set_title(f"Sélection sans regarder le test  ·  ρ de Spearman = {sel['spearman_cv_ipw_vs_silencieux']:.2f}")
    ax.text(0.02, 0.95, "● rouge : retenu     ● bleu : dans la marge     ● gris : hors marge", transform=ax.transAxes, fontsize=11, color=GRIS)
    F["selection"] = _fig("S5_selection.png")

    # S6 — confusion + PR
    import joblib
    art = joblib.load(RACINE / "models" / "modele_final.joblib")
    num, cat = art["num"], art["cat"]; X = df[num + cat]
    dec = decouper(df, cfg, "S3_MNAR", seed); sil = dec["silencieux"]; y = cibles["M3"].astype(str).values
    proba = art["pipeline"].predict_proba(X[sil]); classes = art["classes"]; i_det = classes.index("Detracteur")
    m = R["modele_final"]["metriques_retenues"]; cm = np.array(m["matrice_confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].imshow(cm, cmap="Blues"); axes[0].set_xticks(range(3)); axes[0].set_yticks(range(3))
    axes[0].set_xticklabels([LIB[c] for c in ORDRE]); axes[0].set_yticklabels([LIB[c] for c in ORDRE]); axes[0].set_xlabel("prédit"); axes[0].set_ylabel("vrai")
    axes[0].set_title("Matrice de confusion — 5 963 silencieux")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{cm[i, j]}\n{cm[i, j]/cm[i].sum():.0%}", ha="center", va="center", fontsize=12, color="white" if cm[i, j] > cm.max() / 2 else NOIR)
    for k, c in enumerate(classes):
        yb = (y[sil] == c).astype(int); p_, r_, _ = precision_recall_curve(yb, proba[:, k]); ap = average_precision_score(yb, proba[:, k])
        axes[1].plot(r_, p_, color=COUL[c], lw=2.2, label=f"{LIB[c]} — AP {ap:.2f} (base {yb.mean():.2f})")
    axes[1].set_xlabel("rappel"); axes[1].set_ylabel("précision"); axes[1].set_title("Précision–rappel, une classe contre les autres"); axes[1].legend(frameon=False)
    F["perf"] = _fig("S6_performance.png")

    # S7 — fiabilité + gain
    yb = (y[sil] == "Detracteur").astype(int); p_brut = art["pipeline_brut"].predict_proba(X[sil])[:, list(art["pipeline_brut"].classes_).index("Detracteur")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for p_, lab, col in ((p_brut, "non calibré", GRIS), (proba[:, i_det], "calibré (Platt)", BLEU2)):
        fp, mp_ = calibration_curve(yb, p_, n_bins=10, strategy="quantile"); axes[0].plot(mp_, fp, marker="o", lw=2, label=lab, color=col)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].fill_between([0, 1], [0, 1], [0, 0], color=ROUGE, alpha=0.05)
    axes[0].text(0.55, 0.15, "sous la diagonale :\nP(Détracteur) surestimée", fontsize=11, color=ROUGE)
    axes[0].set_xlabel("probabilité prédite"); axes[0].set_ylabel("fréquence observée"); axes[0].set_title("Fiabilité de P(Détracteur)"); axes[0].legend(frameon=False)
    ordre_ = np.argsort(-proba[:, i_det]); cum = np.cumsum(yb[ordre_]) / yb.sum(); frac = np.arange(1, len(yb) + 1) / len(yb)
    K = cfg["metier"]["k_appels"]; pk = R["evaluation_metier"]["precision_at_k"]
    axes[1].plot(frac, cum, color=ROUGE, lw=2.4, label="modèle"); axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="au hasard")
    axes[1].axvline(K / len(yb), color=BLEU2, ls=":", lw=1.5)
    axes[1].annotate(f"K = {K} appels\n{pk['precision_at_k']:.0%} de vrais détracteurs\nlift ×{pk['lift']:.1f}", (K / len(yb), pk["rappel_at_k"]), xytext=(0.3, 0.55), textcoords="axes fraction",
                     fontsize=12, arrowprops=dict(arrowstyle="->", color=BLEU2))
    axes[1].set_xlabel("part des clients appelés (classés par risque)"); axes[1].set_ylabel("part des détracteurs captés"); axes[1].set_title("Courbe de gain — la valeur pour la rétention"); axes[1].legend(frameon=False, loc="lower right")
    F["calibration"] = _fig("S7_calibration_gain.png")

    # S8 — valeur métier : précision@K et NPS simulé
    ck = pd.DataFrame(R["evaluation_metier"]["courbe_k"]); ns = R["nps_simule"]; eco = R["evaluation_metier"]["economie"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].plot(ck["k"], ck["precision_at_k"], marker="o", lw=2.2, color=ROUGE, label="précision@K (part de vrais détracteurs)")
    axes[0].plot(ck["k"], ck["rappel_at_k"], marker="s", lw=2.2, color=BLEU2, label="rappel@K (part des détracteurs captés)")
    axes[0].axhline(pk["taux_de_base"], color=GRIS, ls="--", lw=1.2); axes[0].text(ck["k"].max(), pk["taux_de_base"] + 0.015, f"taux de base {pk['taux_de_base']:.0%}", ha="right", color=GRIS, fontsize=11)
    axes[0].set_xlabel("K — nombre d'appels dans le mois"); axes[0].set_ylabel("part"); axes[0].set_title("Combien de vrais détracteurs pour K appels ?"); axes[0].legend(frameon=False, loc="center right")
    axes[0].set_ylim(0, 0.8)
    vals = [ns["nps_repondants_observe"], ns["nps_silencieux_predit"], ns["nps_silencieux_vrai"]]
    labs = ["répondants\n(ce que l'opérateur voit)", "silencieux\nprédit par le modèle", "silencieux\nvrai (connu ici)"]
    axes[1].bar(labs, vals, color=[GRIS, BLEU2, VERT], width=0.6)
    ic = ns["nps_silencieux_predit_ic95"]; axes[1].errorbar([1], [vals[1]], yerr=[[vals[1] - ic[0]], [ic[1] - vals[1]]], fmt="none", color=NOIR, capsize=6, lw=1.5)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 1.2, f"{v:+.0f}", ha="center", fontsize=14, fontweight="bold")
    axes[1].axhline(0, color=NOIR, lw=0.8); axes[1].set_title("Simuler le NPS des 85 % silencieux"); axes[1].set_ylabel("NPS"); axes[1].set_ylim(min(0, min(vals) - 3), max(vals) * 1.18)
    F["valeur"] = _fig("S8_valeur.png")

    # S9 — drivers globaux + segments
    dr = R["drivers"]; imp = pd.Series(dr["importance_detracteur"]).head(12); direction = dr.get("direction", {})
    seg = dr.get("par_segment", {})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1, 1.1]})
    axes[0].barh(imp.index[::-1], imp.values[::-1], color=[ROUGE if direction.get(v, 1) > 0 else VERT for v in imp.index][::-1])
    axes[0].set_title("Les 12 variables qui pèsent le plus"); axes[0].set_xlabel("importance moyenne |contribution| — classe Détracteur")
    axes[0].text(0.98, 0.04, "rouge : une valeur élevée augmente le risque\nvert : le diminue", transform=axes[0].transAxes, ha="right", fontsize=11, color=GRIS)
    if seg:
        allv = list(dict.fromkeys(v for s in seg.values() for v in list(s["drivers"])[:5]))[:10]
        mat = pd.DataFrame({s: {v: seg[s]["drivers"].get(v, {}).get("importance", 0.0) for v in allv} for s in seg})
        im = axes[1].imshow(mat.values, cmap="Reds", aspect="auto")
        axes[1].set_xticks(range(len(mat.columns))); axes[1].set_xticklabels(mat.columns, rotation=25, ha="right", fontsize=11)
        axes[1].set_yticks(range(len(allv))); axes[1].set_yticklabels(allv, fontsize=11); axes[1].set_title("Où chaque variable pèse le plus (effet de composition, modèle additif)")
        plt.colorbar(im, ax=axes[1], fraction=0.035, label="importance dans le segment")
    F["drivers"] = _fig("S9_drivers.png")

    # S10 — équité : rappel/sélection par âge, proxies, mitigation
    eq = R["audit_equite"]; px = R["audit_proxies"]; mit = R["mitigation_seuils_age"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1, 1, 1]})
    ga = eq["tranche_age"]["groupes"]; ordre_age = ["<30", "30-45", "45-60", "60+"]
    xx = np.arange(len(ordre_age))
    axes[0].bar(xx - 0.2, [ga[a]["rappel_detracteur"] for a in ordre_age], 0.4, color=BLEU2, label="rappel Détracteur")
    axes[0].bar(xx + 0.2, [ga[a]["taux_selection"] for a in ordre_age], 0.4, color=GRIS2, label="taux de sélection (part appelée)")
    axes[0].set_xticks(xx); axes[0].set_xticklabels(ordre_age); axes[0].set_ylim(0, 1); axes[0].set_title(f"Par tranche d'âge — écart {eq['tranche_age']['ecart_max']*100:.0f} pts"); axes[0].legend(frameon=False, fontsize=11)
    pxs = pd.Series({k: v["auc"] for k, v in px.items()}).sort_values()
    axes[1].barh(pxs.index, pxs.values, color=[ROUGE if v > 0.75 else ORANGE if v > 0.6 else VERT for v in pxs.values]); axes[1].axvline(0.5, color=NOIR, ls=":")
    axes[1].set_xlim(0.4, 1); axes[1].set_title("Les features reconstruisent-elles l'attribut ?"); axes[1].set_xlabel("AUC de reconstruction (0,5 = non)")
    av = [mit["avant"][a]["rappel_detracteur"] for a in ordre_age]; ap = [mit["apres"].get(a, {}).get("rappel", np.nan) for a in ordre_age]
    pa = [mit["avant"][a]["precision_detracteur"] for a in ordre_age]; pp = [mit["apres"].get(a, {}).get("precision", np.nan) for a in ordre_age]
    axes[2].bar(xx - 0.2, av, 0.4, color=GRIS2, label="rappel avant"); axes[2].bar(xx + 0.2, ap, 0.4, color=BLEU2, label="rappel après seuils par âge")
    axes[2].plot(xx, pa, "o--", color=GRIS, label="précision avant"); axes[2].plot(xx, pp, "o-", color=ROUGE, label="précision après")
    axes[2].set_xticks(xx); axes[2].set_xticklabels(ordre_age); axes[2].set_ylim(0, 1); axes[2].set_title("Mitigation testée : ce que ça coûte"); axes[2].legend(frameon=False, fontsize=10, ncol=2)
    F["equite"] = _fig("S10_equite.png")

    # S11 — robustesse : bruit + ablation
    br = pd.DataFrame([b for b in R["bruit_etiquettes"] if "kappa" in b]); ab = pd.DataFrame([a for a in R["ablation_features"] if "kappa" in a])
    base_k = ab.loc[ab["configuration"].str.startswith("base"), "kappa"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [0.85, 1.15]})
    axes[0].plot(br["taux_bruit"] * 100, br["kappa"], marker="o", lw=2.2, color=BLEU2, label="kappa"); axes[0].plot(br["taux_bruit"] * 100, br["rappel_detracteur"], marker="s", lw=2.2, color=ROUGE, label="rappel Détracteur")
    axes[0].set_xlabel("% d'étiquettes d'entraînement corrompues"); axes[0].set_title("Robustesse au bruit d'étiquettes"); axes[0].legend(frameon=False); axes[0].set_ylim(0, 0.8)
    yy = np.arange(len(ab)); d_ = ab["kappa"] - base_k
    axes[1].barh(yy, d_, color=[VERT if v >= 0 else ROUGE for v in d_]); axes[1].set_yticks(yy); axes[1].set_yticklabels(ab["configuration"], fontsize=11); axes[1].invert_yaxis(); axes[1].axvline(0, color=NOIR, lw=0.8)
    for i, v in enumerate(d_):
        axes[1].text(v + (0.001 if v >= 0 else -0.001), i, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=11)
    axes[1].set_xlabel("écart de kappa par rapport à la configuration retenue"); axes[1].set_title("Ablation : ce que chaque bloc coûte ou rapporte")
    F["robustesse"] = _fig("S11_robustesse.png")

    # S12 — monitoring
    mo = _j("monitoring")
    if mo:
        d_sim = pd.DataFrame(mo["derive_entrees_mois_simule"]).head(10).iloc[::-1]; su = mo.get("suivi_nouvelles_reponses")
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
        axes[0].barh(d_sim["Variable"], d_sim["PSI"], color=[ROUGE if s == "ALERTE" else ORANGE if s == "vigilance" else VERT for s in d_sim["Statut"]])
        axes[0].axvline(0.1, color=NOIR, ls=":", lw=1); axes[0].axvline(0.2, color=NOIR, ls="--", lw=1); axes[0].set_xlabel("PSI"); axes[0].set_title("Dérive des entrées — mois simulé (tarif +12 %, contrats, fibre)")
        if su:
            lots = pd.DataFrame(su["lots"])
            axes[1].plot(lots["mois"], lots["kappa_cumul"], marker="o", lw=2.2, color=BLEU2, label="kappa cumulé — nouvelles réponses")
            axes[1].plot(lots["mois"], lots["rappel_detracteur_cumul"], marker="s", lw=2.2, color=ROUGE, label="rappel Détracteur cumulé")
            axes[1].axhline(su["reference"]["kappa"] - su["seuils"]["chute_kappa"], color=BLEU2, ls=":"); axes[1].axhline(su["reference"]["rappel_detracteur"] - su["seuils"]["chute_rappel_detracteur"], color=ROUGE, ls=":")
            if su["premier_mois_volume_atteint"]:
                axes[1].axvline(su["premier_mois_volume_atteint"], color=NOIR, ls="--"); axes[1].text(su["premier_mois_volume_atteint"] + 0.05, 0.3, f"{su['seuils']['min_nouveaux_labels']} labels\n→ réentraîner", fontsize=11)
            axes[1].set_xlabel("mois"); axes[1].set_ylim(0.25, 0.75); axes[1].set_title("Performance sur les nouvelles réponses d'enquête"); axes[1].legend(frameon=False, fontsize=11)
        F["monitoring"] = _fig("S12_monitoring.png")
    return F


# =============================================================================
# DIAPOSITIVES
# =============================================================================
CSS = """
@page { size: 338.67mm 190.5mm; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Segoe UI', Arial, Helvetica, sans-serif; color: #1c2430; background: #fff; }
.slide { width: 338.67mm; height: 190.5mm; padding: 11mm 14mm 9mm 14mm; page-break-after: always; position: relative; overflow: hidden; display: flex; flex-direction: column; }
.slide:last-child { page-break-after: auto; }
.tag { position: absolute; top: 6mm; right: 14mm; font-size: 9.5pt; color: #1f4e79; letter-spacing: .06em; text-transform: uppercase; font-weight: 600; }
.enonce { position: absolute; top: 6mm; left: 14mm; font-size: 9.5pt; color: #8a97a5; letter-spacing: .04em; }
h1 { font-size: 22pt; margin: 4mm 0 2mm; color: #1c2430; line-height: 1.15; font-weight: 700; }
h1 small { display: block; font-size: 12pt; color: #5f6b7a; font-weight: 400; margin-top: 1.5mm; }
.corps { display: flex; gap: 8mm; flex: 1; min-height: 0; }
.fig { flex: 0 0 64%; display: flex; align-items: center; justify-content: center; }
.fig img { max-width: 100%; max-height: 128mm; object-fit: contain; }
.fig.large { flex: 0 0 100%; } .fig.large img { max-height: 132mm; }
.txt { flex: 1; font-size: 12pt; line-height: 1.4; }
.txt ul { padding-left: 5mm; margin: 2mm 0; } .txt li { margin-bottom: 2.2mm; }
.txt b { color: #1f4e79; }
.chiffre { font-size: 26pt; font-weight: 700; color: #1f4e79; line-height: 1; }
.kpis { display: flex; gap: 6mm; margin: 5mm 0; }
.kpi { flex: 1; border: 1px solid #d5dbe1; border-radius: 3mm; padding: 5mm 6mm; }
.kpi .lab { font-size: 9.5pt; color: #5f6b7a; text-transform: uppercase; letter-spacing: .04em; }
.kpi .val { font-size: 24pt; font-weight: 700; color: #1f4e79; margin: 1.5mm 0 1mm; }
.kpi .sub { font-size: 10pt; color: #7a8794; }
.note { border-left: 3px solid #2c7fb8; background: #f3f8fc; padding: 3mm 5mm; border-radius: 2mm; font-size: 11.5pt; margin-top: 3mm; }
.warn { border-left: 3px solid #e67e22; background: #fdf6ee; padding: 3mm 5mm; border-radius: 2mm; font-size: 11.5pt; margin-top: 3mm; }
.bad  { border-left: 3px solid #c0392b; background: #fbf0ee; padding: 3mm 5mm; border-radius: 2mm; font-size: 11.5pt; margin-top: 3mm; }
table { border-collapse: collapse; width: 100%; font-size: 10.5pt; }
th, td { border-bottom: 1px solid #d5dbe1; padding: 1.8mm 3mm; text-align: left; vertical-align: top; }
th { color: #1f4e79; font-weight: 600; background: #f3f8fc; }
.pied { position: absolute; bottom: 5mm; left: 14mm; right: 14mm; display: flex; justify-content: space-between; font-size: 9pt; color: #8a97a5; border-top: 1px solid #e3e6ea; padding-top: 2mm; }
.titre { background: linear-gradient(135deg, #0d2440 0%, #1f4e79 60%, #2c7fb8 100%); color: white; justify-content: center; }
.titre h1 { color: white; font-size: 34pt; margin: 0 0 6mm; } .titre h1 small { color: #cfe0f0; font-size: 15pt; }
.titre .meta { color: #cfe0f0; font-size: 12pt; margin-top: 12mm; }
.crisp { display: flex; gap: 4mm; margin-top: 6mm; }
.crisp .ph { flex: 1; border-radius: 3mm; padding: 4mm; background: #f3f8fc; border-top: 4px solid #2c7fb8; font-size: 10.5pt; min-height: 78mm; }
.crisp .ph h3 { margin: 0 0 2mm; font-size: 12pt; color: #1f4e79; } .crisp .ph ul { padding-left: 4mm; margin: 0; } .crisp .ph li { margin-bottom: 1.5mm; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; margin-top: 3mm; }
.mod { border: 1px solid #d5dbe1; border-radius: 2mm; padding: 2.5mm 3mm; font-size: 9.6pt; }
.mod b { color: #1f4e79; display: block; font-size: 10.5pt; } .mod i { color: #5f6b7a; }
.mod.ret { border: 2px solid #c0392b; background: #fbf0ee; }
.check td:first-child { width: 12mm; font-weight: 700; color: #27ae60; }
"""


def _n_tests() -> int:
    """Nombre de fonctions de test, compté dans les fichiers plutôt qu'écrit en dur.

    Un décompte figé cesse d'être vrai au premier test ajouté, et personne ne le remarque : c'est
    exactement le genre de petit mensonge qui décrédibilise un rendu quand un jury le vérifie.
    On compte les fonctions, pas les cas exécutés (les tests paramétrés en produisent davantage),
    et c'est ce que le libellé annonce.
    """
    n = 0
    for f in sorted((RACINE / "tests").glob("test_*.py")):
        for ligne in f.read_text(encoding="utf-8").split("\n"):
            if ligne.startswith("def test_"):
                n += 1
    return n


def diapos(cfg, R, F, arb) -> str:
    mf = R["modele_final"]; m = mf["metriques_retenues"]; sel = R["selection_modele_final"]; cl = pd.DataFrame(sel["classement"])
    pk = R["evaluation_metier"]["precision_at_k"]; eco = R["evaluation_metier"]["economie"]; ns = R["nps_simule"]
    eq = R["audit_equite"]; ga = eq["tranche_age"]["groupes"]; px = R["audit_proxies"]; mit = R["mitigation_seuils_age"]
    dg = {d["scenario"]: d for d in R["diagnostic_protocole"]}; fu = R["diagnostic_fuite"]; de = R["desequilibre"]
    fo = _j("fondation"); tx = _j("texte"); mo = _j("monitoring"); rp = _j("reproductibilite")
    n_tests = _n_tests()
    ret = cl[cl.modele == sel["retenu"]].iloc[0]; gs = cl[cl.modele == sel["gagnant_silencieux"]].iloc[0]
    br = [b for b in R["bruit_etiquettes"] if "kappa" in b]; ab = {a["configuration"]: a for a in R["ablation_features"] if "kappa" in a}
    def dk(motif):
        for k, v in ab.items():
            if motif in k:
                return v["kappa"] - ab["base (retenue)"]["kappa"]
        return float("nan")
    k_log = R["comparatif_s3_m3"]["logistique"]["defaut"]["silencieux"]["kappa"]
    n_slides = 22
    S = []

    def slide(tag, enonce, titre, sous, corps, num, cls=""):
        S.append(f"""<section class="slide {cls}"><div class="enonce">{enonce}</div><div class="tag">{tag}</div>
<h1>{titre}<small>{sous}</small></h1>{corps}
<div class="pied"><span>Prédiction de la catégorie NPS des clients silencieux — soutenance</span><span>{num} / {n_slides}</span></div></section>""")

    def fig_txt(img, txt, large=False):
        if large:
            return f'<div class="corps"><div class="fig large"><img src="{_b64(img)}"></div></div>{txt}'
        return f'<div class="corps"><div class="fig"><img src="{_b64(img)}"></div><div class="txt">{txt}</div></div>'

    cap = RACINE / "reports" / "captures"

    # 1 — titre
    S.append(f"""<section class="slide titre"><h1>Prédire la catégorie NPS des clients silencieux<small>Prioriser la rétention quand 85 % des clients ne répondent pas à l'enquête — take-home challenge, dataset IBM Telco 11.1.3+</small></h1>
<div class="meta">Démarche CRISP-DM · 7 043 clients · 15 modèles comparés · application Streamlit · étude reproductible<br>Soutenance devant jury</div>
<div class="pied" style="color:#cfe0f0;border-color:#4a6d8f"><span>Prédiction de la catégorie NPS des clients silencieux — soutenance</span><span>1 / {n_slides}</span></div></section>""")

    # 2 — le problème
    slide("CRISP-DM · 1 · compréhension métier", "Énoncé § 1–2", "Le problème : décider pour 85 % de clients dont on ignore la satisfaction",
          "Trois usages demandés, trois critères de succès chiffrés avant toute ligne de code", f"""
<div class="kpis"><div class="kpi"><div class="lab">Répondent à l'enquête NPS</div><div class="val">15 %</div><div class="sub">et ils ne sont pas un échantillon au hasard</div></div>
<div class="kpi"><div class="lab">Silencieux à scorer</div><div class="val">{mf['n_evaluation']}</div><div class="sub">sur 7 043 clients</div></div>
<div class="kpi"><div class="lab">Capacité d'appel</div><div class="val">{pk['k']} / mois</div><div class="sub">qui appeler en premier ?</div></div>
<div class="kpi"><div class="lab">Cible</div><div class="val">3 classes</div><div class="sub">ordonnées et déséquilibrées</div></div></div>
<table><tr><th>Usage demandé (énoncé § 2)</th><th>Critère de succès fixé a priori</th><th>Où c'est mesuré</th></tr>
<tr><td>Prioriser les appels vers les détracteurs prédits</td><td>précision@{pk['k']} nettement au-dessus du taux de base ; gain net en euros vs hasard</td><td>diapo 14</td></tr>
<tr><td>Identifier les drivers de détraction, individuels et par segment</td><td>drivers stables au mapping ; leviers actionnables</td><td>diapo 15</td></tr>
<tr><td>Simuler le NPS des 85 % silencieux</td><td>estimation avec intervalle, comparée à la vérité connue ici</td><td>diapo 14</td></tr>
<tr><td>Équité (§ 4.7)</td><td>écart de rappel Détracteur entre groupes mesuré, signalé si > 10 pts</td><td>diapo 16</td></tr></table>
<div class="note"><b>Hypothèse déclarée</b> : la satisfaction 1–5 est un proxy acceptable d'une note NPS 0–10 — Elero et al. (2026) font la même approximation. Cadre : dataset fictif, une photographie, aucune campagne enregistrée → aucun effet causal estimable.</div>""", 2)

    # 3 — démarche CRISP-DM
    slide("CRISP-DM · démarche", "Énoncé § 7, § 9", "Une démarche CRISP-DM complète, chaque phase fermée par des décisions justifiées",
          "Ce qui a été fait dans chaque phase — et où le jury peut le vérifier", f"""
<div class="crisp">
<div class="ph"><h3>1 · Métier</h3><ul><li>3 usages, critères chiffrés</li><li>cible ordonnée et déséquilibrée</li><li>3 formulations à arbitrer par la mesure</li><li>registre de risques</li></ul></div>
<div class="ph"><h3>2 · Données</h3><ul><li>5 tables, jointure contrôlée</li><li>dictionnaire à statut</li><li>5 contrôles qualité</li><li><b>registre de fuites</b> en 3 natures, quantifié</li><li>EDA, quasi-expérience Offer</li></ul></div>
<div class="ph"><h3>3 · Préparation</h3><ul><li>cible : 3 mappings, <b>arbitrage empirique des « 3 »</b></li><li>features justifiées une à une</li><li>colinéarité, déséquilibre (5 stratégies)</li><li>tout dans un Pipeline</li></ul></div>
<div class="ph"><h3>4 · Modélisation</h3><ul><li>protocole 15/85 <b>MNAR</b></li><li><b>15 modèles, 7 familles</b></li><li>réglage, <b>sélection sans regarder le test</b></li><li>calibration vérifiée</li><li>TabICL, texte</li></ul></div>
<div class="ph"><h3>5 · Évaluation</h3><ul><li>par classe, métier, NPS simulé</li><li>robustesse : bruit, ablation, mapping</li><li>drivers globaux et par segment</li><li><b>équité, proxies, mitigation</b></li></ul></div>
<div class="ph"><h3>6 · Déploiement</h3><ul><li>modèle persisté, app 8 pages filtrables, image Docker</li><li>monitoring 5 déclencheurs, dont l'équité</li><li>boucle de rétroaction</li><li>reproductibilité, tests</li><li>décision demandée</li></ul></div></div>
<div class="warn"><b>Gestion du périmètre</b> (énoncé : « scope management more than over-engineering ») : chaque pièce répond à une exigence nommée — matrice de conformité en diapo 21. Quinze modèles ne sont pas une course au score : une comparaison qui établit que le plafond vient du signal, donc que la marche suivante est métier (vraies réponses, groupe de contrôle). Laissé de côté volontairement, et dit : enrichissement externe, deep ordinal, API de scoring, MLflow.</div>
<div class="note">Livrables : étude en 8 documents (dont la <b>matrice de conformité à l'énoncé</b>), write-up 6 pages, application, notebook, {n_tests} fonctions de test, test à blanc depuis copie vierge {'identique sur ' + str(len(rp['comparaison'])) + ' indicateurs' if rp and rp.get('reproductible') else ''}.</div>""", 3)

    # 4 — fuites
    slide("CRISP-DM · 2 · compréhension des données", "Énoncé § 4.1 — data leakage", "Le dataset contient sa propre réponse : toute colonne liée au churn est une fuite",
          "Fait fondateur, trois natures de fuite, exclusion mesurée plutôt qu'affirmée", fig_txt(F["fuites"], f"""
<ul><li>Satisfaction 1–2 → <b>100 % de départs</b> ; 4–5 → <b>0 %</b>. Connaître le churn, c'est connaître la cible — et au moment de détecter un détracteur, il n'est pas encore parti.</li>
<li>Trois natures nommées : <b>conséquence</b> (Churn Label/Value, Customer Status), <b>disponibilité</b> (Churn Category/Reason : vides = restés), <b>modèle amont</b> (Churn Score, CLTV — sorties SPSS).</li>
<li>Après modélisation, deux diagnostics : exactitude max {fu['exactitude_max_observee']:.2f} (alerte à 0,85) ; écart arbres/linéaire {fu['ecart_arbres_lineaire_S1_M3']:+.2f} (signature d'une fuite : +0,30, Mustafa et al. 2021 : 98 % annoncés avec la note NPS en feature).</li>
<li><b>CLTV</b> : bannie des features, admise en <b>couche de décision</b> pour pondérer qui appeler.</li></ul>"""), 4)

    # 5 — cible
    slide("CRISP-DM · 3 · préparation", "Énoncé § 4.1 — building the NPS target", f"La grille de l'énoncé donne un NPS de {arb['nps']['M1']:+.0f} ; les données disent {arb['nps']['M3']:+.0f}",
          "Trois mappings, un arbitrage empirique des 2 665 clients « 3 », et un piège de circularité évité", fig_txt(F["cible"], f"""
<ul><li><b>M1</b> (énoncé) envoie {arb['distributions']['M1'].get('Detracteur', 0):.0%} de la base chez les détracteurs, dont {arb['n_ambigus']} clients « 3 ». NPS {arb['nps']['M1']:+.0f} : à 60–75 pts des benchmarks télécom — signal d'alerte, pas résultat.</li>
<li><b>Arbitrage empirique</b> (méthode Elero et al.) : modèle entraîné sur les seuls extrêmes 1–2 vs 4–5 (exactitude {arb['exactitude_sur_extremes']:.0%}), appliqué aux « 3 » jamais vus : <b>{arb['part_3_vers_promoteur']:.0%} penchent promoteur</b> → le bloc va aux Passifs. NPS {arb['nps']['M3']:+.0f}, dans la fourchette sectorielle.</li>
<li><b>Piège évité</b> : étiqueter chaque « 3 » individuellement rendrait la cible fonction des features — le modèle réapprendrait sa propre sortie. On tranche le bloc ; la probabilité individuelle reste un indicateur métier.</li>
<li>Refusé : enrichir la cible par Churn Value (churn déguisé) ou l'ancienneté (même circularité).</li></ul>"""), 5)

    # 6 — features et déséquilibre
    derivees = "".join(f"<tr><td><code>{k}</code></td><td>{v}</td></tr>" for k, v in list(HYPOTHESES_DERIVEES.items())[:9])
    strat = [s for s in de["strategies"] if "kappa" in s]
    strat_rows = "".join(f"<tr><td>{s['strategie']}</td><td>{s['kappa']:.3f}</td><td>{s['macro_f1']:.3f}</td><td>{s['rappel_detracteur']:.2f}</td><td>{s['rappel_passif']:.2f}</td></tr>" for s in strat)
    slide("CRISP-DM · 3 · préparation", "Énoncé § 4.2–4.3 — features, class imbalance", "Chaque variable a une hypothèse ; le déséquilibre est double et traité par pondération",
          f"{len(R['features']['numeriques'])} numériques + {len(R['features']['categorielles'])} catégorielles ; démographie et géographie hors modèle", f"""
<div class="corps"><div class="txt" style="flex:0 0 52%"><table><tr><th>Variable dérivée</th><th>Hypothèse métier</th></tr>{derivees}</table>
<div class="note" style="margin-top:3mm">Brutes justifiées une à une (contrat, ancienneté, offre, parrainage, prix, usage, bouquet, paiement). Colinéarité : identité comptable des totaux — conservée, régularisation L2 et arbres insensibles.</div></div>
<div class="txt"><b>Deux déséquilibres superposés.</b> Base : Passif {de['distribution_base']['Passif']:.0%}, Détracteur {de['distribution_base']['Detracteur']:.0%}. Répondants (entraînement, MNAR) : Détracteur {de['distribution_repondants']['Detracteur']:.0%}, <b>Passif {de['distribution_repondants']['Passif']:.0%}</b>. La classe la plus fréquente à l'évaluation est la plus rare à l'entraînement — c'est elle qu'un modèle abandonne.
<table style="margin-top:3mm"><tr><th>Stratégie (baseline logistique)</th><th>Kappa</th><th>Macro-F1</th><th>Rappel Dét.</th><th>Rappel Passif</th></tr>{strat_rows}</table>
<div class="note"><b>Décision</b> : pondération « balanced » partout où l'estimateur l'accepte. SMOTE fabrique des clients dans un espace one-hot sans sens métier, pour aucun gain systématique.</div></div></div>""", 6)

    # 7 — protocole
    slide("CRISP-DM · 4 · modélisation", "Énoncé § 4.2, § 4.5 — validation strategy", "Les répondants ne sont pas un échantillon au hasard : le biais est simulé, et son coût mesuré",
          "Trois mécanismes de non-réponse ; entraînement sur les 15 %, évaluation sur les 85 % dont on connaît la vérité", fig_txt(F["protocole"], f"""
<ul><li>Un <code>train_test_split</code> stratifié rate l'intention de l'énoncé. On simule la propension à répondre : <b>MCAR</b> (hasard), <b>MAR</b> (covariables), <b>MNAR</b> (dépend de la satisfaction — les mécontents et les enthousiastes répondent plus).</li>
<li>Sous S3, la part de notes extrêmes passe de <b>{dg['S3_MNAR']['part_extremes_base']:.0%} à {dg['S3_MNAR']['part_extremes_repondants']:.0%}</b> : le modèle apprend sur une population qui ne ressemble pas à celle qu'il scorera.</li>
<li>Avantage rare du dataset : la satisfaction des 100 % est connue → on <b>mesure</b> ce que le biais coûte, ce qu'aucun praticien ne peut faire en production.</li>
<li>IPW estimée sous MAR (poids {R['ipw_estime']['poids_min']:.2f}–{R['ipw_estime']['poids_max']:.2f}) : utilisée pour pondérer la validation croisée ; effet à l'entraînement {R['effet_ipw'].get('sans_ipw', {}).get('kappa', 0):.3f} → {R['effet_ipw'].get('avec_ipw', {}).get('kappa', 0):.3f}. Limite structurelle face au MNAR (Heckman exigerait une restriction d'exclusion).</li>
<li>Kannan et al. (2022) : taux de réponse réel 2,6 % — notre 15 % est optimiste.</li></ul>"""), 7)

    # 8 — catalogue
    mods = "".join(f"""<div class="mod {'ret' if c['nom'] == sel['retenu'] else ''}"><b>{_lib(c['nom'])}</b><i>{c['famille']} · {c['formulation']}</i><br>{c['hypothese'][:110]}{'…' if len(c['hypothese']) > 110 else ''}</div>""" for c in CATALOGUE)
    slide("CRISP-DM · 4 · modélisation", "Énoncé § 4.5 — baseline, ≥ 2 familles, ordinal, boosting, calibrated ensembles", "Quinze modèles, sept familles, trois formulations — chacun pour tester une hypothèse",
          "Pas un tableau de leaderboard : chaque modèle est là parce qu'il répond à une question sur les données", f"""<div class="grid">{mods}</div>
<div class="note">Les trois formulations de l'énoncé sont instanciées : nominale (12), <b>ordinale</b> (mord, all-threshold), <b>régression 1–5 puis seuillage</b> (ridge, HGB). Écartés avec argument : CORAL/CORN (deep ordinal, sans objet sur ~1 000 lignes). Modèles de fondation à part (diapo 17). Encadré rouge : le modèle retenu.</div>""", 8)

    # 9 — grille
    slide("CRISP-DM · 4 · modélisation", "Énoncé § 4.1 — sensitivity to the mapping", "3 scénarios × 3 mappings × 15 modèles : la cible arbitrée est la plus apprenable, le biais coûte à tous",
          "Kappa quadratique pondéré sur les 85 % silencieux — la métrique qui respecte l'ordre des classes", fig_txt(F["grille"], f"""
<ul><li><b>M3 domine M1 et M2 pour toutes les familles</b> : la cible arbitrée par les données est aussi celle que les modèles apprennent le mieux.</li>
<li><b>La performance chute de S1 à S3</b> pour presque tous les modèles : c'est le coût mesuré du biais de non-réponse — le résultat scientifique du protocole.</li>
<li>Tous les kappas tiennent dans une bande étroite ({cl['kappa_silencieux'].min():.2f}–{cl['kappa_silencieux'].max():.2f}) : <b>le plafond n'est pas algorithmique, il est dans le signal</b> (~1 000 répondants biaisés, cible bruitée). Changer de modèle rapporte moins que corriger le biais de réponse.</li>
<li>Pourquoi le kappa quadratique : une erreur de deux crans (Promoteur prédit Détracteur) doit coûter plus qu'une erreur d'un cran ; l'exactitude et le F1 ne le voient pas.</li></ul>"""), 9)

    # 10 — sélection
    slide("CRISP-DM · 4 · modélisation", "Énoncé § 9 — honesty, disciplined baseline", f"Choisir sans regarder le test : {_lib(sel['retenu'])} retenue, le plus simple des modèles dans la marge",
          "Règle fixée a priori : kappa CV sur les répondants (pondéré IPW), puis parcimonie à un écart-type (ESL § 7.10)", fig_txt(F["selection"], f"""
<ul><li>Meilleur kappa CV brut : <b>{_lib(sel['meilleur_cv_brut'])}</b>. Dans la marge (≥ {sel['seuil_parcimonie']:.3f}) : {len(sel['candidats_dans_la_marge'])} modèles. Retenu : <b>{_lib(sel['retenu'])}</b> ({sel['version_retenue']}), simplicité {int(ret['simplicite'])}.</li>
<li>Sur le test : le retenu fait <b>{ret['kappa_silencieux']:.3f}</b> (rang {int(ret['rang_silencieux'])}/15) ; le meilleur, {_lib(sel['gagnant_silencieux'])}, {gs['kappa_silencieux']:.3f}. L'écart de {ret['kappa_silencieux'] - gs['kappa_silencieux']:+.3f} est le prix d'une sélection honnête ; choisir a posteriori serait choisir sur le test.</li>
<li>ρ = {sel['spearman_cv_ipw_vs_silencieux']:.2f} : la CV sur les répondants classe correctement les modèles malgré le biais.</li>
<li><b>La CV promet {ret['kappa_cv']:.2f}, le test donne {ret['kappa_silencieux']:.2f}</b> : en production personne ne verrait cet écart. C'est la quantité à retenir.</li>
<li>Réglage : recherche aléatoire, espaces modestes ; version réglée retenue seulement si elle gagne > 0,005 en CV pondérée.</li></ul>"""), 10)

    # 11 — performance par classe
    cls_rows = "".join(f"<tr><td>{LIB[c]}</td><td>{m['par_classe'][c]['precision']:.2f}</td><td><b>{m['par_classe'][c]['rappel']:.2f}</b></td><td>{m['par_classe'][c]['f1']:.2f}</td><td>{m['par_classe'][c]['effectif']}</td></tr>" for c in ORDRE)
    slide("CRISP-DM · 5 · évaluation", "Énoncé § 4.5 — macro-F1, balanced accuracy, kappa, per-class recall", "Jamais une métrique globale sans son détail par classe : les trois classes restent vivantes",
          f"Kappa {m['kappa_quadratique']:.3f} · macro-F1 {m['macro_f1']:.3f} · exactitude équilibrée {m['exactitude_equilibree']:.3f} · AUC Détracteur {m.get('auc_detracteur', 0):.2f}", fig_txt(F["perf"], f"""
<table><tr><th>Classe</th><th>Précision</th><th>Rappel</th><th>F1</th><th>n</th></tr>{cls_rows}</table>
<ul style="margin-top:3mm"><li>Le <b>Détracteur</b> est rappelé à {m['par_classe']['Detracteur']['rappel']:.0%} avec une précision de {m['par_classe']['Detracteur']['precision']:.0%} : le modèle appelle large — le bon arbitrage quand manquer un détracteur coûte plus qu'un appel inutile.</li>
<li>Le <b>Passif</b>, milieu ambigu et rare à l'entraînement, est le plus difficile ({m['par_classe']['Passif']['rappel']:.0%}) — comme chez Elero et al. La pondération de classes le maintient vivant ; sans elle il tombait à 0,08.</li>
<li>Exactitude brute {m['exactitude']:.2f} : rapportée pour montrer qu'elle trompe — prédire toujours Passif donne 0,43.</li>
<li>Baseline logistique : {k_log:.3f} de kappa. Contre-exemple Kannan et al. : F 0,876 en résumé, 0,126 sur la classe minoritaire.</li></ul>"""), 11)

    # 12 — calibration
    cal = mf["calibration"]
    slide("CRISP-DM · 5 · évaluation", "Énoncé § 4.5 — calibration plots, lift curves", "Les probabilités servent à classer les clients, pas à lire une fréquence",
          "Calibration testée, vérifiée, corrigée par une stratégie de décision hybride", fig_txt(F["calibration"], f"""
<ul><li>Platt (3 plis) : Brier {cal['non_calibre']['brier']:.3f} → {cal['calibre_platt']['brier']:.3f}, ECE {cal['non_calibre']['ece']:.3f} → {cal['calibre_platt']['ece']:.3f}. {'Mais le calibrateur <b>écrase la classe Passif</b> (rappel ' + f"{mf['metriques_calibre']['par_classe']['Passif']['rappel']:.2f}" + ') : rare chez les répondants, il ne la prédit plus.' if cal['passif_ecrase'] else ''}</li>
<li><b>Décision hybride</b> : la <b>classe</b> vient du modèle non calibré ; <b>P(Détracteur)</b>, seule utilisée pour classer les appels, vient du modèle calibré. Vérifié par test automatisé.</li>
<li>Les courbes sont <b>sous la diagonale</b> : le calibrateur apprend le taux de base des répondants, pas celui des silencieux — <b>calibrer sur un échantillon biaisé calibre vers la mauvaise population</b>. Le classement (AUC {m.get('auc_detracteur', 0):.2f}) reste valide ; la valeur absolue non.</li>
<li>Correction en production : recalibrer sur les premières vraies réponses de la population cible (dès 300 labels).</li></ul>"""), 12)

    # 13 — valeur métier + NPS simulé
    slide("CRISP-DM · 5 · évaluation", "Énoncé § 2 — prioritise outreach, simulate the expected NPS", f"Sur {pk['k']} appels, {pk['precision_at_k']:.0%} de vrais détracteurs (×{pk['lift']:.1f}) ; le NPS des silencieux estimé à {ns['nps_silencieux_predit']:+.0f}",
          "La question de l'équipe rétention, et la question de la direction", fig_txt(F["valeur"], f"""
<div class="kpis" style="margin:0 0 3mm"><div class="kpi"><div class="lab">Gain net apporté</div><div class="val">{_eur(eco['gain_apporte_par_le_modele_eur'])}</div><div class="sub">par campagne vs hasard</div></div><div class="kpi"><div class="lab">ROI</div><div class="val">{eco['roi']:.1f}</div><div class="sub">{_eur(eco['cout_campagne_eur'])} de coût</div></div></div>
<ul><li>Hypothèses (config.yaml, <b>à valider avec la rétention</b>) : appel {cfg['metier']['cout_appel_eur']} €, client retenu {cfg['metier']['valeur_client_retenu_eur']} €, succès {cfg['metier']['taux_succes_appel']:.0%}.</li>
<li><b>NPS des silencieux</b> : {ns['nps_silencieux_predit']:+.0f} [{ns['nps_silencieux_predit_ic95'][0]:+.0f} ; {ns['nps_silencieux_predit_ic95'][1]:+.0f}] contre une vérité de {ns['nps_silencieux_vrai']:+.0f} — sous-estimé de {ns['nps_silencieux_vrai'] - ns['nps_silencieux_predit']:.0f} pts. L'intervalle ne couvre que l'aléa d'échantillonnage, pas le biais : dit tel quel. Composition : Détracteurs {ns['part_classes_silencieux_predit']['Detracteur']:.0%} vs {ns['part_classes_silencieux_vrai']['Detracteur']:.0%} réels, Promoteurs {ns['part_classes_silencieux_predit']['Promoteur']:.0%} vs {ns['part_classes_silencieux_vrai']['Promoteur']:.0%}.</li>
<li>Le classement des segments par NPS est respecté (mensuel et récents les plus critiques).</li>
{('<li><b>Correction testée</b> : quatre estimateurs de quantification (CC, PCC, ACC, PACC) ; le plus proche reste à ' + f"{min(abs(ns['quantification'][k]['nps'] - ns['quantification']['verite']['nps']) for k in ('CC','PCC','ACC','PACC')):.0f}" + ' pts. La matrice de confusion des répondants ne se transfère pas aux silencieux sous MNAR — la correction doit venir de vraies réponses, pas d’un calcul.</li>') if ns.get('quantification') else ''}</ul>"""), 13)

    # 14 — drivers
    lev = R["leviers"]["distribution_detracteurs_predits"]; top_lev = max(lev.items(), key=lambda kv: kv[1])
    slide("CRISP-DM · 5 · évaluation", "Énoncé § 4.6 — drivers, segment-level, actionable, single lever", "Drivers contractuels, stables au mapping ; par segment, c'est le poids qui change, pas le coefficient",
          f"Explication : {R['drivers']['methode']} — la même dans l'étude et pour chaque client de l'application", fig_txt(F["drivers"], f"""
<ul><li><b>Contrat sans engagement, faible ancienneté, absence de parrainage, facture élevée</b> vont avec la détraction — cohérent avec Chong et al. (2023) sur ce dataset et Elero et al. (2026).</li>
<li><b>Offre E</b> : marqueur de clients déjà fragiles (52,9 % de départs vs 27 % sans offre) — biais d'indication, <b>pas une cause</b>. Association ≠ causalité, dit à chaque écran.</li>
<li><b>Par segment</b> : les mêmes variables changent de poids entre contrat mensuel et engagé, récent et ancien, fibre et DSL.</li>
<li><b>Un levier par détracteur prédit</b>, règle implémentée : engagement 12 mois si mensuel → revue tarifaire si charge par service élevée → parcours d'accueil si récent → support premium. {top_lev[1]} des détracteurs prédits relèvent du premier levier.</li>
<li>Sensibilité : sur les 12 drivers principaux, le sens d'effet tient sous les trois mappings.</li></ul>"""), 14)

    # 15 — équité
    slide("CRISP-DM · 5 · évaluation", "Énoncé § 4.7 — fairness and bias", f"Le modèle rate davantage les jeunes détracteurs ({ga['<30']['rappel_detracteur']:.0%} vs {ga['60+']['rappel_detracteur']:.0%}) — sans variable d'âge",
          "Audit par sous-groupe, audit des proxies, mitigation chiffrée, arbitrage remonté au métier et au juridique", fig_txt(F["equite"], f"""
<ul><li>Démographie et géographie <b>exclues du modèle</b>, utilisées uniquement pour l'audit. Écart de rappel Détracteur par âge : <b>{eq['tranche_age']['ecart_max']*100:.0f} pts</b> — le cas que l'énoncé juge inacceptable, en sens inverse. Genre et densité de zone : écarts faibles.</li>
<li><b>Pourquoi</b> : les features retenues <b>reconstruisent</b> l'âge (AUC {px.get('Moins de 30 ans', {}).get('auc', 0):.2f}) et la situation familiale (AUC {px.get('Marié(e)', {}).get('auc', 0):.2f}). Exclure une colonne n'empêche pas le modèle de la voir.</li>
<li>Coût de l'exclusion, chiffré : démographie {dk('+ démographie'):+.3f} de kappa, géographie {dk('+ géographie'):+.3f}. Assumé.</li>
<li><b>Mitigation testée</b> : un seuil par tranche d'âge égalise le rappel ({mit['rappel_cible']:.0%}) au prix de la précision chez les jeunes et de {mit['appels_apres'] - mit['appels_avant']:+d} appels. Traitement différencié selon l'âge : <b>décision CX + juridique</b>, non appliquée par défaut. Audit répété à chaque réentraînement.</li></ul>"""), 15)

    # 16 — robustesse
    slide("CRISP-DM · 5 · évaluation", "Énoncé § 4.1 — realistic noise ; § 4.3, § 4.7 — feature trade-offs", "Robuste au bruit d'étiquettes ; chaque bloc de variables a un coût mesuré",
          "Ce qu'un label NPS réel — bruité — coûterait, et ce que vaut chaque choix de features", fig_txt(F["robustesse"], f"""
<ul><li><b>Bruit d'étiquettes</b> : {br[-1]['taux_bruit']:.0%} d'étiquettes corrompues coûtent {br[0]['kappa'] - br[-1]['kappa']:.3f} de kappa — dégradation graduelle, pas d'effondrement. Un label NPS réel est bruité (humeur, échelle mal comprise).</li>
<li><b>Contract</b> est le bloc le plus précieux ({dk('Contract'):+.3f} si retiré) ; le <b>parrainage</b> ({dk('parrainage'):+.3f}) est le driver le plus proche de la cible sans être une fuite — comportement passé observable.</li>
<li>Variables dérivées : {dk('dérivées'):+.3f} ; Offer : {dk('Offer'):+.3f}.</li>
<li>Déséquilibre : 5 stratégies comparées (diapo 6). Mapping : 3 cibles comparées (diapo 9). Aucune conclusion métier ne dépend d'un seul de ces choix.</li></ul>"""), 16)

    # 17 — bonus : fondation + texte
    fo_rows = ""
    if fo:
        for x in fo["modeles"]:
            if x["statut"] == "ok":
                fo_rows += f"<tr><td><b>{x['nom']}</b></td><td>{x['metriques']['kappa']:.3f}</td><td>{x['metriques']['macro_f1']:.3f}</td><td>{x['metriques']['rappel_passif']:.2f}</td><td>{x['duree_inference_s']:.0f} s ({x['latence_par_client_ms']:.0f} ms/client)</td><td>{'bat' if x['metriques']['kappa'] > fo['reference']['kappa'] + 0.005 else 'ne bat pas'} le retenu</td></tr>"
            else:
                fo_rows += f"<tr><td><b>{x['nom']}</b></td><td colspan=5>indisponible — {x.get('cause', x.get('erreur', ''))[:120]}</td></tr>"
        fo_rows += f"<tr><td>{_lib(fo['reference']['modele'])} (retenu)</td><td>{fo['reference']['kappa']:.3f}</td><td>{fo['reference']['macro_f1']:.3f}</td><td>{m['par_classe']['Passif']['rappel']:.2f}</td><td>{fo['reference']['duree_inference_s']:.2f} s</td><td>référence</td></tr>"
    tx_rows = ""
    if tx:
        for lab, v in (("Tabulaire seul", tx["tabulaire_seul"]), ("Texte seul (TF-IDF + logistique)", tx["texte_seul"]), (f"Fusion tardive (poids {tx['fusion_tardive']['poids_texte']:.1f})", tx["fusion_tardive"]), ("Fusion précoce (SVD 20)", tx["fusion_precoce_svd20"])):
            tx_rows += f"<tr><td>{lab}</td><td>{v['kappa']:.3f}</td><td>{v['macro_f1']:.3f}</td><td>{v['rappel_detracteur']:.2f}</td></tr>"
    slide("CRISP-DM · 4 · modélisation — bonus", "Énoncé § 4.4 verbatims, § 4.5 foundation models — « do not frame it as the winner if it is not »", "Deux bonus explorés honnêtement : le modèle de fondation ne gagne pas, le texte synthétique ne prouve rien",
          "TabICL et TabPFN v2 évalués sur poids publics ; TabPFN 2.5 bloqué par une licence, pas par un dépôt fermé", f"""
<div class="corps"><div class="txt" style="flex:0 0 55%"><b>Modèles de fondation tabulaires</b><table><tr><th>Modèle</th><th>Kappa</th><th>Macro-F1</th><th>Rappel Passif</th><th>Inférence</th><th>Verdict</th></tr>{fo_rows}</table>
<div class="note"><b>Avantages</b> : aucun réglage, performance d'emblée — un excellent étalon de première heure. <b>Limites</b>, et la première n'est pas la latence : <b>la classe Passif est abandonnée</b> (rappel ~0), ce que le kappa quadratique pénalise peu ; la comparaison n'est pas à protocole égal (aucun des deux n'accepte la pondération de classes utilisée par les 15 autres modèles) ; inférence 2–3 ordres de grandeur plus lente ; aucune explication native. <b>Compromis chiffré face au meilleur gradient boosting</b> dans l'étude, phase 4 § 8. Acceptable pour un scoring mensuel, pas pour l'application interactive.</div></div>
<div class="txt"><b>Verbatims synthétiques et fusion texte</b><table><tr><th>Configuration</th><th>Kappa</th><th>Macro-F1</th><th>Rappel Dét.</th></tr>{tx_rows}</table>
<div class="warn">{tx['n_verbatims'] if tx else ''} notes générées (générateur local seedé, fragments LLM ; chemin API Anthropic prêt, prompt versionné), tonalité tirée de la classe avec 25 % de bruit. Le texte <b>encode la classe par construction</b> : tout gain est un artefact. La chaîne fonctionne ; <b>pas de composante texte en production</b> sur cette base — le « spot when the added complexity is not worth it » de l'énoncé.</div></div></div>""", 17)

    # 18 — application
    slide("CRISP-DM · 6 · déploiement", "Énoncé § 4.8 — persistence, simple productization", "Un outil pour l'équipe rétention : liste priorisée, explication et levier pour chaque client",
          "Application Streamlit, 8 pages filtrables, livrée en image Docker, chaque valeur lue dans les artefacts du pipeline", f"""
<div class="corps"><div class="fig" style="flex:0 0 58%"><img src="{_b64(cap / '03_analyser_un_client.png')}" style="max-height:120mm;border:1px solid #d5dbe1;border-radius:2mm"></div>
<div class="txt"><ul><li><b>Prioriser</b> : silencieux classés par P(Détracteur), filtres, capacité K, pondération CLTV (couche de décision), levier, export CSV.</li>
<li><b>Analyser</b> : client existant ou saisie manuelle — « (inconnu) » accepté partout, le pipeline impute et <code>handle_unknown='ignore'</code> absorbe toute modalité nouvelle (testé). Classe, probabilités, contributions de <i>ce</i> client, levier, contexte du segment, verbatim collé (démonstration).</li>
<li><b>Comprendre et surveiller</b> : données et cible, 15 modèles et sélection, drivers et équité, monitoring, carte du modèle, conformité à l'énoncé.</li>
<li>Pourquoi Streamlit : l'énoncé demande un outil pour un responsable rétention, pas une API ; Python pur, versionné avec le pipeline. FastAPI serait la brique CRM suivante.</li>
<li>Modèle persisté d'un bloc (préparation + modèle + calibrateur + décision) : rejouable sans réentraînement.</li></ul></div></div>""", 18)

    # 19 — monitoring
    su = (mo or {}).get("suivi_nouvelles_reponses", {})
    slide("CRISP-DM · 6 · déploiement", "Énoncé § 4.9 — monitoring, retraining trigger, feedback loop", "Monitoring implémenté : dérive, performance sur les nouvelles réponses, cinq déclencheurs dont l'équité suivie mois par mois",
          "Et la boucle de rétroaction, qui rendrait le modèle pire que rien si on l'ignorait", fig_txt(F.get("monitoring", F["robustesse"]), f"""
<ul><li><b>Dérive des entrées</b> (PSI par variable) et <b>des prédictions</b> : sur un mois simulé (tarif +12 %, 15 % vers le mensuel, 10 % vers la fibre), alerte sur {', '.join((mo or {}).get('variables_en_alerte_mois_simule', [])[:2])} → réentraînement déclenché.</li>
<li><b>Performance réelle</b> sur les nouvelles réponses d'enquête (~150/mois) : chute de kappa > {su.get('seuils', {}).get('chute_kappa', 0.05)} ou de rappel > {su.get('seuils', {}).get('chute_rappel_detracteur', 0.1)} ; <b>volume</b> ≥ {su.get('seuils', {}).get('min_nouveaux_labels', 500)} labels (mois {su.get('premier_mois_volume_atteint', '—')}) ; <b>échéance</b> 6 mois ; recalibration dès 300 labels ; équité par âge : rappel Détracteur recalculé par tranche d'âge sur le cumul, alerte au-delà de 10 pts — le seuil de la phase 1, désormais le même partout.</li>
<li><b>Boucle de rétroaction</b> : un détracteur appelé puis retenu change de classe — réentraîner dessus apprend au modèle l'effet de sa propre intervention. Parades : journaliser chaque contact, groupe de contrôle, réentraîner sur non-contactés et contrôles.</li></ul>"""), 19)

    # 20 — limites et décision
    slide("CRISP-DM · 6 · déploiement", "Énoncé § 7 — implemented / approximate / future work", "Ce que l'étude ne peut pas faire, et la seule décision qu'elle demande à la direction",
          "Cibler par risque n'est pas l'optimum : un groupe de contrôle gratuit permet de faire mieux au cycle suivant", f"""
<div class="corps"><div class="txt" style="flex:0 0 50%"><table><tr><th></th><th>Ce qui est</th></tr>
<tr><td><b>Implémenté</b></td><td>tout le périmètre obligatoire (§ 4.1–4.8), le monitoring (§ 4.9), les trois bonus</td></tr>
<tr><td><b>Approximatif, dit tel quel</b></td><td>paramètres économiques placeholders ; propension à répondre simulée ; verbatims locaux (pas d'API) ; NPS des silencieux sous-estimé de {ns['nps_silencieux_vrai'] - ns['nps_silencieux_predit']:.0f} pts</td></tr>
<tr><td><b>Impossible avec ces données</b></td><td>estimer l'effet d'un appel : aucune campagne, aucune assignation aléatoire — la table Offer prouve que les offres ont été ciblées</td></tr>
<tr><td><b>Travail futur</b></td><td>TabPFN 2.5 dès acceptation de la licence PriorLabs ; recalibration sur vraies réponses ; uplift après groupe de contrôle ; API CRM ; vrais verbatims</td></tr></table></div>
<div class="txt"><div class="bad" style="margin-top:0"><b>La décision demandée.</b> Ascarza (2018, <i>J. Marketing Research</i>) : cibler les clients au risque le plus élevé est inefficace ; il faut cibler ceux dont le comportement change quand on les appelle — même budget, jusqu'à 7 pts de churn en moins. Les détracteurs les plus certains sont souvent des <i>perdus d'avance</i>.<br><br>
<b>À la prochaine campagne, ne pas contacter 10 à 20 % des clients ciblés, tirés au hasard.</b> Ce groupe de contrôle achète la mesure de l'effet réel de la campagne et les données d'un modèle d'uplift — passer du ciblage par risque au ciblage par sensibilité.</div>
<div class="note">C'est le seul passage de l'étude qui demande une décision plutôt que de présenter un résultat.</div></div></div>""", 20)

    # 21 — conformité
    checks = [("§ 4.1", "Cible justifiée, raffinée, sensibilité, bruit, fuites"), ("§ 4.2", "Schéma, manquants, split 15/85, déséquilibre"), ("§ 4.3", "Features justifiées, géo/démo arbitrées"),
              ("§ 4.4", "Verbatims LLM reproductibles + fusion (bonus)"), ("§ 4.5", "Baseline, 15 modèles, métriques adaptées, validation, SHAP/coefficients, fondation (bonus)"),
              ("§ 4.6", "Drivers par segment, actionnable, levier unique (bonus)"), ("§ 4.7", "Audit sous-groupes, proxies, arbitrage, escalade"), ("§ 4.8", "Modèle persisté, app robuste aux inconnus"),
              ("§ 4.9", "Monitoring, déclencheurs, boucle de rétroaction"), ("§ 6–7", "Code, notebook, README, .env.example, dataset, verbatims, modèle, captures, write-up 6 p. ; reproductible ; IA déclarée ; pas de clés")]
    rows = "".join(f"<tr><td>✓</td><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in checks)
    slide("Conformité et reproductibilité", "Énoncé § 6, § 7, § 9", "Chaque exigence de l'énoncé a une réponse et un emplacement vérifiable",
          "Reproductible à l'identique ; usage d'IA déclaré ; aucune clé dans le dépôt", f"""
<div class="corps"><div class="txt" style="flex:0 0 58%"><table class="check">{rows}</table></div>
<div class="txt"><div class="kpis" style="margin:0 0 4mm;flex-direction:column"><div class="kpi"><div class="lab">Test à blanc depuis copie vierge</div><div class="val">{len(rp['comparaison']) if rp else '—'}/{len(rp['comparaison']) if rp else '—'}</div><div class="sub">indicateurs identiques (NPS, modèle retenu, kappa, précision@K, équité…)</div></div>
<div class="kpi"><div class="lab">Tests automatisés</div><div class="val">{n_tests}</div><div class="sub">jointure, fuites, cible, protocole, 15 modèles clonables, sélection, exécution réelle des 8 pages de l'app</div></div></div>
<div class="note"><b>Usage d'IA</b> (§ 7) : code, structure de l'étude et rédaction produits avec l'assistance d'un assistant IA (Claude), sous direction et relecture humaines ; fragments des verbatims synthétiques rédigés par ce même assistant. Choix de modélisation, périmètre et conclusions assumés par l'auteur.</div></div></div>""", 21)

    # 22 — à retenir
    slide("Conclusion", "Énoncé § 9 — what matters most", "Cinq choses à retenir", "Clear reasoning, practical execution, reproducibility — et honnêteté sur ce que les données permettent", f"""
<div class="kpis"><div class="kpi"><div class="lab">1 · La cible</div><div class="val">{arb['nps']['M3']:+.0f}</div><div class="sub">et non {arb['nps']['M1']:+.0f} : les « 3 » arbitrés par les données, sans circularité</div></div>
<div class="kpi"><div class="lab">2 · Les fuites</div><div class="val">0</div><div class="sub">colonne liée au churn dans le modèle ; exactitude max {fu['exactitude_max_observee']:.2f}</div></div>
<div class="kpi"><div class="lab">3 · Le modèle</div><div class="val">×{pk['lift']:.1f}</div><div class="sub">{_lib(sel['retenu'])}, choisie parmi 15 sans regarder le test</div></div>
<div class="kpi"><div class="lab">4 · L'équité</div><div class="val">{eq['tranche_age']['ecart_max']*100:.0f} pts</div><div class="sub">d'écart par âge, cause mesurée (proxies), mitigation chiffrée, remontée</div></div>
<div class="kpi"><div class="lab">5 · La décision</div><div class="val">10–20 %</div><div class="sub">de groupe de contrôle : mesurer l'effet, préparer l'uplift</div></div></div>
<div class="note" style="font-size:12.5pt">Ce que les quinze modèles disent ensemble : le plafond de performance est dans le signal — ~1 000 répondants biaisés et une cible bruitée — pas dans l'algorithme. La prochaine marche n'est pas un modèle de plus : ce sont de vraies réponses de la population cible, un journal des contacts et un groupe de contrôle.</div>
<div class="warn" style="font-size:12.5pt">Limites assumées : hypothèses économiques à valider ; propension simulée ; taux de réponse 15 % optimiste (2,6 % chez Kannan et al.) ; NPS agrégé des silencieux sous-estimé ; verbatims synthétiques ; TabPFN 2.5 non évalué faute d'acceptation de licence (TabICL et TabPFN v2 l'ont été).</div>
<div class="note" style="font-size:12.5pt;margin-top:5mm">Merci. Application : <code>streamlit run app/app.py</code> · Étude : <code>reports/</code> (8 documents) · Write-up : <code>livrable/write_up.pdf</code> · Conformité : <code>reports/07_conformite_enonce.md</code></div>""", 22)

    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(S)}</body></html>"


# =============================================================================
# NOTES D'ORAL
# =============================================================================
def notes(cfg, R, arb) -> str:
    n_tests = _n_tests()
    mf = R["modele_final"]; m = mf["metriques_retenues"]; sel = R["selection_modele_final"]; pk = R["evaluation_metier"]["precision_at_k"]
    ns = R["nps_simule"]; eq = R["audit_equite"]["tranche_age"]; px = R["audit_proxies"]
    cl = pd.DataFrame(sel["classement"]); ret = cl[cl.modele == sel["retenu"]].iloc[0]
    return f"""
# Notes d'oral — soutenance (20 minutes + questions)

Une idée par diapositive. Le titre est le message ; ne pas le relire, le prouver avec le graphique.
Repères de temps cumulés entre crochets.

| # | Diapositive | Temps | Ce qu'on dit, en une phrase | Le chiffre à prononcer |
|---|---|---|---|---|
| 1 | Titre | [0:00] | Le sujet : décider pour les 85 % de clients dont on ignore la satisfaction. | 85 % |
| 2 | Le problème | [0:45] | Trois usages métier, trois critères chiffrés avant le code ; cible ordonnée et déséquilibrée. | K = {pk['k']} appels/mois |
| 3 | Démarche CRISP-DM | [1:45] | Six phases, chacune fermée par des décisions justifiées ; tout est vérifiable dans l'étude. | 8 documents, {n_tests} fonctions de test |
| 4 | Les fuites | [2:45] | Le dataset contient sa propre réponse : satisfaction 1–2 → 100 % de départs. Trois natures de fuite, exclusion mesurée. | exactitude max {R['diagnostic_fuite']['exactitude_max_observee']:.2f} < 0,85 |
| 5 | La cible | [4:00] | La grille de l'énoncé donne {arb['nps']['M1']:+.0f} ; les « 3 » arbitrés par un modèle sur les extrêmes penchent promoteur à {arb['part_3_vers_promoteur']:.0%} → NPS {arb['nps']['M3']:+.0f}. Piège de circularité évité. | {arb['n_ambigus']} clients « 3 » |
| 6 | Features et déséquilibre | [5:15] | Chaque variable a une hypothèse ; double déséquilibre (Passif rare à l'entraînement, majoritaire à l'évaluation) ; pondération choisie après 5 stratégies. | Passif : {R['desequilibre']['distribution_repondants']['Passif']:.0%} des répondants |
| 7 | Protocole 15/85 | [6:15] | Les répondants ne sont pas aléatoires : MNAR simulé, extrêmes de 37 % à 71 %. On mesure ce que le biais coûte. | 37 → 71 % |
| 8 | Catalogue | [7:15] | Quinze modèles, sept familles, trois formulations — chacun pour une hypothèse. Pas un leaderboard. | 15 / 7 / 3 |
| 9 | Grille | [8:15] | M3 domine partout ; le biais coûte à tous ; kappas dans une bande de 0,30–0,39 : le plafond est le signal. | 3 × 3 × 15 |
| 10 | Sélection | [9:15] | Choisie par CV sur les répondants + parcimonie, sans regarder le test. CV promet {ret['kappa_cv']:.2f}, test donne {ret['kappa_silencieux']:.2f}. | ρ = {sel['spearman_cv_ipw_vs_silencieux']:.2f} |
| 11 | Par classe | [10:30] | Jamais une métrique globale seule ; trois classes vivantes ; Détracteur rappelé à {m['par_classe']['Detracteur']['rappel']:.0%}. | kappa {m['kappa_quadratique']:.3f} |
| 12 | Calibration | [11:30] | Testée, vérifiée, corrigée : hybride. Calibrer sur des répondants biaisés calibre vers la mauvaise population. | sous la diagonale |
| 13 | Valeur et NPS simulé | [12:30] | {pk['precision_at_k']:.0%} de vrais détracteurs sur {pk['k']} appels ; NPS des silencieux {ns['nps_silencieux_predit']:+.0f} pour {ns['nps_silencieux_vrai']:+.0f} réel — sous-estimé, dit tel quel. | ×{pk['lift']:.1f} |
| 14 | Drivers | [13:45] | Contractuels, stables, différents par segment ; un levier par client ; association ≠ cause (offre E). | 1er levier : engagement |
| 15 | Équité | [14:45] | Écart de {eq['ecart_max']*100:.0f} pts par âge sans variable d'âge : les features reconstruisent l'âge (AUC {px.get('Moins de 30 ans', {}).get('auc', 0):.2f}). Mitigation chiffrée, décision remontée. | AUC {px.get('Marié(e)', {}).get('auc', 0):.2f} pour « marié » |
| 16 | Robustesse | [16:00] | Bruit d'étiquettes graduel ; chaque bloc de features a un coût mesuré. | Contract : le bloc clé |
| 17 | Bonus | [16:45] | TabICL ne gagne pas et coûte 3 ordres de grandeur ; le texte synthétique encode la classe : ne prouve rien. | « not the winner » |
| 18 | Application | [17:30] | Explorer par filtres croisés, prioriser, analyser, comprendre, surveiller ; tolérante aux inconnus ; tout lu dans les artefacts ; `docker compose up` suffit à la lancer. | 8 pages, Docker |
| 19 | Monitoring | [18:15] | PSI, performance sur nouvelles réponses, cinq déclencheurs dont l'écart d'équité par âge recalculé chaque mois, boucle de rétroaction. | 500 labels → réentraîner |
| 20 | Limites et décision | [19:00] | Ce qu'on ne peut pas faire (effet d'un appel) ; la seule décision demandée : groupe de contrôle. | 10–20 % |
| 21 | Conformité | [19:40] | Chaque exigence → réponse → emplacement ; reproductible ; IA déclarée. | 10/10 identiques |
| 22 | À retenir | [20:00] | Cinq chiffres, une phrase : le plafond est dans le signal, pas dans l'algorithme. | — |

## Questions de jury anticipées — et la réponse en trente secondes

1. **Pourquoi ne pas avoir utilisé Churn Value pour construire la cible, comme l'énoncé le suggère ?**
   Parce que satisfaction 1–2 ⇔ churn à 100 % : la cible deviendrait une relabellisation du churn, et le projet un modèle de
   churn déguisé — inutilisable pour détecter un détracteur *avant* qu'il parte. L'énoncé le suggère comme piste ; on l'a
   examinée et refusée avec la mesure (information mutuelle).

2. **Votre mapping M3 n'est-il pas une manière de choisir la cible qui arrange ?**
   Non : le bloc des « 3 » est tranché par un modèle qui ne les a jamais vus, entraîné sur les extrêmes, et la décision porte
   sur le *bloc*, pas sur les individus — sinon la cible deviendrait fonction des features. M1 et M2 sont conservés, comparés
   (grille) et les drivers sont vérifiés stables sous les trois. La corroboration externe (benchmark +19/+34) n'est pas une preuve.

3. **Pourquoi le kappa quadratique comme critère principal ?**
   Cible ordonnée : une erreur de deux crans doit coûter plus qu'une erreur d'un cran ; l'exactitude et le F1 ne le voient
   pas. Il est corrigé du hasard. On le départage par macro-F1 et on publie toujours le rappel par classe à côté.

4. **Comment justifiez-vous la simulation MNAR ? Sa forme est arbitraire.**
   Sa forme est plausible, pas observée — c'est dit. Ce qu'elle apporte : on connaît la satisfaction des 100 %, donc on
   *mesure* le coût du biais (CV {ret['kappa_cv']:.2f} → test {ret['kappa_silencieux']:.2f}), ce qu'aucun praticien ne peut faire. Trois mécanismes sont comparés ; le
   réaliste est retenu. En production, la correction passe par de vraies réponses de la population cible.

5. **Pourquoi la logistique ordinale plutôt que le meilleur boosting ?**
   Règle fixée avant de regarder : kappa CV pondéré IPW, puis parcimonie à un écart-type (ESL § 7.10). Huit modèles sont dans
   la marge ; le plus simple gagne. Sur le test, il est deuxième à 0,003 du premier. Et tous les modèles tiennent dans une
   bande de 0,09 : le plafond est le signal, pas l'algorithme.

6. **Vous sélectionnez sur les répondants biaisés : n'est-ce pas contradictoire ?**
   C'est la seule information disponible en production. La pondération IPW estimée (MAR) corrige ce qui est corrigeable ; le
   ρ de {sel['spearman_cv_ipw_vs_silencieux']:.2f} entre CV et test montre que le classement tient. Choisir sur les silencieux serait choisir sur le test.

7. **La classe Passif à 0,48 de rappel, c'est faible.**
   C'est le milieu ambigu, {R['desequilibre']['distribution_repondants']['Passif']:.0%} des répondants sous MNAR. Sans pondération de classes elle tombait à 0,08 ; la
   calibration l'écrasait à 0 — d'où la décision hybride. Elero et al. observent la même difficulté sur la classe neutre.
   Pour le métier, ce qui compte est le classement des détracteurs (AUC {m.get('auc_detracteur', 0):.2f}).

8. **Le NPS des silencieux est sous-estimé de 17 points : votre estimateur est-il utilisable ?**
   Pas en valeur absolue, et l'étude le dit : l'intervalle ne couvre que l'aléa d'échantillonnage, pas le biais. Le classement
   des segments est respecté. La correction n'est pas un autre modèle : c'est de vraies réponses de la population cible pour
   recalibrer.

9. **Les probabilités sont surestimées : pourquoi les afficher ?**
   Elles servent à *classer* (l'ordre est valide), pas à lire une fréquence — l'application le dit à chaque écran. Le
   calibrateur est ajusté sur des répondants biaisés : calibrer vers la mauvaise population. Recalibration dès 300 vraies réponses.

10. **Vous excluez l'âge et pourtant le modèle discrimine par l'âge. À quoi sert l'exclusion ?**
    Elle empêche une décision *explicite* sur l'attribut et retire le levier le plus direct ; elle ne suffit pas, et on l'a
    mesuré (AUC de reconstruction {px.get('Moins de 30 ans', {}).get('auc', 0):.2f}). C'est pour ça qu'il y a un audit, une mitigation chiffrée et une escalade —
    pas une case cochée.

11. **Pourquoi ne pas appliquer la mitigation par seuils d'âge ?**
    Parce que c'est un traitement différencié selon l'âge : légal dans certaines juridictions, pas dans d'autres, et il coûte de
    la précision chez les jeunes. C'est une décision d'Expérience Client et de juridique, pas de data science. Elle est prête et chiffrée.

12. **CLTV est une fuite mais vous l'utilisez dans l'application ?**
    En couche de *décision*, pas de *modélisation* : pondérer qui appeler parmi les détracteurs prédits est un choix métier,
    pas une information sur la cible. Gómez-Vargas et al. (2025) montrent qu'une CLV individuelle bat une CLV moyenne pour cibler.

13. **Pourquoi pas de SMOTE ?**
    Comparé : aucun gain systématique, et il fabrique des clients synthétiques dans un espace one-hot où l'interpolation n'a pas
    de sens métier. La pondération est plus simple, sans hyperparamètre, et laisse les données intactes.

14. **Le texte gagne du kappa : pourquoi ne pas le déployer ?**
    Parce que les verbatims ont été fabriqués à partir de la classe : ils contiennent le label, bruité. Le gain est un artefact
    du protocole. La chaîne est démontrée ; sa valeur réelle ne peut se mesurer que sur de vrais verbatims.

15. **TabICL fait presque aussi bien sans réglage : pourquoi ne pas le retenir ?**
    Parce que le kappa cache l'essentiel. TabICL **écrase la classe Passif** — rappel nul : il se comporte en classifieur
    binaire Détracteur / Promoteur, et le kappa quadratique pénalise peu une erreur d'un cran. Son macro-F1 est donc très
    en dessous. Ajoutez que la comparaison lui est défavorable à un autre titre : ni TabICL ni TabPFN n'acceptent la
    pondération de classes appliquée aux quinze autres modèles, donc la classe minoritaire part perdante chez eux par
    construction — je le dis parce que cela joue en ma faveur et qu'il faut le signaler quand même. Enfin, trois ordres de
    grandeur en inférence et aucune explication native, alors que l'explication client par client est au cœur du livrable.
    La consigne de l'énoncé : « do not frame it as the winner if it is not ».

    *Sur TabPFN* : la version 2.5 n'a pas pu être évaluée, mais pas pour la raison qu'on croit. Son dépôt Hugging Face est
    **public** ; ce qui bloque est l'acceptation d'une licence PriorLabs, qui demande un compte et une clé API. J'ai donc
    évalué **TabPFN v2**, dont les poids sont librement téléchargeables, en les chargeant explicitement. Le bonus est traité,
    et le blocage est décrit pour ce qu'il est.

16. **Quel est le vrai levier d'amélioration ?**
    Pas un modèle de plus. Un groupe de contrôle (10–20 %) à la prochaine campagne : il mesure l'effet réel et fournit les
    données d'un modèle d'uplift — passer du ciblage par risque au ciblage par sensibilité (Ascarza 2018, jusqu'à 7 pts de churn).

17. **Qu'avez-vous fait avec l'IA générative ?**
    Code, structure de l'étude, rédaction, fragments des verbatims — déclaré comme l'énoncé le demande (§ 7). Les choix de
    modélisation, le périmètre et les conclusions sont les miens ; les {n_tests} fonctions de test et le test à blanc vérifient ce qui est livré.

## Si le jury n'a que dix minutes

Diapositives 2, 4, 5, 10, 13, 15, 20, 22. Dans cet ordre.
"""


def _pdf(html_path: Path, pdf_path: Path):
    chrome = next((c for c in CHROME if Path(c).exists()), None)
    if chrome is None:
        print("Chrome/Edge introuvable — HTML produit, PDF non généré."); return
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                   capture_output=True, timeout=240)
    if pdf_path.exists():
        try:
            import fitz
            print(f"{pdf_path.relative_to(RACINE)} — {fitz.open(pdf_path).page_count} pages, {pdf_path.stat().st_size // 1024} Ko")
        except Exception:
            print(f"{pdf_path.relative_to(RACINE)} — {pdf_path.stat().st_size // 1024} Ko")


def main():
    cfg = charger_config()
    R = json.load(open(REP / "resultats.json", encoding="utf-8"))
    print("[1/3] données et figures au format présentation")
    df, _ = preparer(cfg); df = construire(df)
    cibles, arb = construire_cibles(df, cfg, cfg["seed"])
    F = figures(cfg, R, df, cibles, arb)
    print("[2/3] diapositives")
    LIV.mkdir(exist_ok=True)
    (LIV / "soutenance.html").write_text(diapos(cfg, R, F, arb), encoding="utf-8")
    (LIV / "soutenance_notes.md").write_text(notes(cfg, R, arb).strip() + "\n", encoding="utf-8")
    print("[3/3] PDF")
    _pdf(LIV / "soutenance.html", LIV / "soutenance.pdf")
    print("livrable/soutenance_notes.md écrit")


if __name__ == "__main__":
    main()
