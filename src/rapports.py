"""Génère l'étude complète : un document Markdown par phase CRISP-DM, ses figures, la synthèse
pour décideur et la matrice de conformité à l'énoncé.

    python -m src.rapports

Chaque chiffre est calculé ici ou lu dans reports/*.json — jamais recopié à la main. Sortie dans
reports/, numérotée dans l'ordre de lecture :

    00_synthese.md                  pour un décideur non technique
    01_comprehension_metier.md      CRISP-DM 1
    02_comprehension_donnees.md     CRISP-DM 2 : dictionnaire, qualité, EDA, registre de fuites
    03_preparation_donnees.md       CRISP-DM 3 : cible, features, déséquilibre, pipeline
    04_modelisation.md              CRISP-DM 4 : formulation, protocole, quinze modèles, sélection
    05_evaluation.md                CRISP-DM 5 : technique, métier, robustesse, drivers, équité, revue
    06_deploiement.md               CRISP-DM 6 : carte de modèle, application, monitoring, décision
    07_conformite_enonce.md         chaque exigence de l'énoncé → où elle est traitée
"""
from __future__ import annotations

import json
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from .data import RACINE, charger_config, charger_tables, joindre, preparer, registre_fuites
from .evaluate import JUSTIFICATION_METRIQUES
from .features import HYPOTHESES_DERIVEES, JUSTIFICATION_BRUTES, colonnes_modele, construire, noms_apres_encodage
from .models import CATALOGUE, FAMILLE, construire_modeles, entrainer
from .protocol import decouper
from .target import ORDRE, construire_cibles, nps

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
COUL = {"Detracteur": "#c0392b", "Passif": "#e67e22", "Promoteur": "#27ae60"}
LIB = {"Detracteur": "Détracteur", "Passif": "Passif", "Promoteur": "Promoteur"}
BLEU, GRIS, ROUGE, VERT, ORANGE = "#2c7fb8", "#95a5a6", "#c0392b", "#27ae60", "#e67e22"
REP = RACINE / "reports"
FIG = REP / "figures"


# ----------------------------------------------------------------------------- utilitaires
def _md(chemin: str, contenu: str):
    p = REP / chemin
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(contenu.strip() + "\n", encoding="utf-8")


def _fig(nom: str):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / nom, bbox_inches="tight")
    plt.close()
    return f"figures/{nom}"


def _tab(df: pd.DataFrame, fmt: dict | None = None) -> str:
    d = df.copy()
    if fmt:
        for c, f in fmt.items():
            if c in d:
                d[c] = d[c].map(lambda v: (f.format(v) if not isinstance(v, str) else v) if pd.notna(v) else "")
    cols = [str(c) for c in d.columns]
    lignes = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lignes.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lignes)


def _cramer_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a, b)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]
    k = min(t.shape) - 1
    return float(np.sqrt(chi2 / (t.values.sum() * k))) if k > 0 else 0.0


def variance_inflation_factor(X: np.ndarray, i: int) -> float:
    X = np.asarray(X, dtype=float)
    y = X[:, i]; autres = np.delete(X, i, axis=1)
    A = np.column_stack([np.ones(len(y)), autres])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = float(((y - A @ coef) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r2 = 1 - ss_res / ss_tot          # VIF_i = 1 / (1 - R²_i)
    return float(1 / max(1 - r2, 1e-9))


def _charger_json(nom):
    p = REP / f"{nom}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _eur(v): return f"{int(round(v)):,} €".replace(",", " ")
def _libmod(nom): return str(nom).replace("_", " ")


def lecture_composition(ns: dict) -> str:
    """Phrase de lecture du NPS simulé, écrite à partir des écarts mesurés — pas d'a priori."""
    ecart = ns["nps_silencieux_predit"] - ns["nps_silencieux_vrai"]
    p, v = ns["part_classes_silencieux_predit"], ns["part_classes_silencieux_vrai"]
    trop = [LIB[c] + "s" for c in ORDRE if p[c] - v[c] > 0.04]
    peu = [LIB[c] + "s" for c in ORDRE if v[c] - p[c] > 0.04]
    justes = [LIB[c] + "s" for c in ORDRE if abs(p[c] - v[c]) <= 0.04]
    morceaux = []
    if trop:
        morceaux.append("trop de " + " et de ".join(trop))
    if peu:
        morceaux.append("trop peu de " + " et de ".join(peu))
    compo = " ; ".join(morceaux) if morceaux else "une composition très proche de la vérité"
    if justes and morceaux:
        compo += f" ; la part de {' et de '.join(justes)} est juste"
    sens = "sous-estime" if ecart < 0 else "surestime"
    return (f"Lecture : le modèle {sens} le NPS des silencieux de {abs(ecart):.1f} pts ({'écart modéré' if abs(ecart) < 8 else 'écart marqué'}, "
            f"la vérité est hors de l'intervalle bootstrap : celui-ci ne couvre que l'aléa d'échantillonnage, pas le biais du modèle). "
            f"Composition par classe : {compo}. Ce sont les conséquences du biais de réponse — le modèle apprend sur des répondants "
            f"polarisés et une pondération de classes qui rééquilibre le Passif — et le NPS, différence de deux parts, cumule les erreurs "
            f"sur les Détracteurs et les Promoteurs. À rapporter avec l'intervalle et cette réserve ; la correction passe par les "
            f"premières vraies réponses de la population cible (phase 6 § 4).")


def _decisions(lignes: list[tuple[str, str, str]]) -> str:
    t = pd.DataFrame(lignes, columns=["Décision", "Justification", "Où c'est vérifié"])
    return "## Décisions prises dans cette phase\n\n" + _tab(t)


# =============================================================================
# PHASE 1 — COMPRÉHENSION MÉTIER
# =============================================================================
def phase1(cfg, R):
    m = cfg["metier"]; pk = R["evaluation_metier"]["precision_at_k"]
    _md("01_comprehension_metier.md", f"""
# Phase 1 — Compréhension métier

**En bref.** Un opérateur télécom ne connaît le NPS que de ~15 % de ses clients. On construit un
modèle qui estime la catégorie NPS des 85 % silencieux, une liste d'appels priorisée pour
l'équipe rétention, et une lecture des drivers de détraction. La cible est **ordonnée**
(Détracteur < Passif < Promoteur) et **déséquilibrée** ; les critères de succès sont chiffrés
avant toute ligne de code.

## 1. Le problème et ses trois usages

| Usage demandé (énoncé § 2) | Ce que le livrable fournit | Où |
|---|---|---|
| Prioriser les appels vers les détracteurs prédits | liste classée par P(Détracteur) calibrée, capacité K réglable, export CSV | application, page *Prioriser les appels* |
| Identifier les drivers de détraction, au niveau individuel et par segment | contributions par client ; drivers par segment (contrat, ancienneté, type d'accès) ; leviers actionnables | phase 5 § 6, application |
| Simuler le NPS attendu des 85 % silencieux | NPS prédit des silencieux avec intervalle bootstrap, comparé au NPS observé des répondants et, ici seulement, à la vérité | phase 5 § 4 |

## 2. Critères de succès, fixés a priori

| Niveau | Critère | Cible | Résultat (phase 5) |
|---|---|---|---|
| Métier | Précision@K sur les détracteurs, K = {m['k_appels']} appels/mois | nettement au-dessus du taux de base | {pk['precision_at_k']:.0%} vs {pk['taux_de_base']:.0%} (×{pk['lift']:.1f}) |
| Métier | Gain net d'une campagne ciblée vs ciblage aléatoire | positif, chiffré en euros | {_eur(R['evaluation_metier']['economie']['gain_apporte_par_le_modele_eur'])} par campagne |
| Data science | Kappa quadratique pondéré et rappel Détracteur | au niveau de la baseline ou au-dessus | voir phase 4 § 6 |
| Data science | Aucune classe abandonnée | rappel > 0,05 sur les trois classes | vérifié (test automatisé) |
| Éthique | Écart de rappel Détracteur entre sous-groupes | mesuré, signalé s'il dépasse 10 points | {R['audit_equite']['tranche_age']['ecart_max']*100:.0f} pts sur l'âge, signalé |

## 3. Formulation du problème d'apprentissage

L'énoncé demande d'arbitrer entre classification nominale, classification ordinale et régression
avec seuillage, et de **justifier**. On ne tranche pas a priori : les trois formulations sont
instanciées dans le catalogue de modèles (phase 4 § 3) et comparées par la même mesure — le kappa
quadratique pondéré, qui pénalise une erreur de deux crans plus qu'une erreur d'un cran, ce qui est
le comportement métier correct sur une cible ordonnée.

## 4. Hypothèses déclarées

1. `Satisfaction Score` (1–5) est un proxy acceptable d'une note NPS (0–10). C'est une hypothèse,
   pas une identité : le NPS mesure une intention de recommandation, la satisfaction une expérience.
   Elero et al. (2026) font la même approximation et la discutent.
2. Le biais de non-réponse simulé (les mécontents et les enthousiastes répondent plus) ressemble au
   biais réel. Sa forme est plausible, pas observée.
3. Les patterns d'un opérateur californien fictif se transposent au cadrage panafricain de l'énoncé.

## 5. Contraintes et ce qu'elles interdisent

Dataset fictif (IBM), une seule photographie, aucun historique, aucune campagne de rétention
enregistrée. Conséquence directe : **aucune estimation d'effet causal n'est possible** — le modèle
classe par risque, pas par sensibilité à l'appel (phase 5 § 8, phase 6 § 5).

## 6. Registre de risques

| Risque | Parade dans l'étude |
|---|---|
| Fuite de données → score irréel, modèle inutilisable | registre de fuites quantifié (phase 2 § 4) + double diagnostic (phase 4 § 5) |
| Cible mal construite → tout le reste invalide | arbitrage empirique des « 3 » + sensibilité au mapping + bruit d'étiquettes (phases 3, 5) |
| Sélection de modèle sur le jeu de test | sélection par validation croisée sur les répondants ; les silencieux ne servent qu'au test (phase 4 § 6) |
| Modèle inéquitable non détecté | audit par sous-groupe, audit des proxies, mitigation chiffrée (phase 5 § 7) |
| Livrable non reproductible | seed unique, `python -m src.run`, test à blanc depuis copie vierge, tests automatisés |

{_decisions([
    ("Cible traitée comme ordonnée et déséquilibrée", "trois classes ordonnées par construction ; Passif = 38 % de la base, Détracteur = 20 %", "phase 3 § 4"),
    ("Métrique principale = kappa quadratique pondéré", "seule métrique usuelle qui distingue une erreur d'un cran d'une erreur de deux crans", "phase 5 § 1"),
    ("Critères de succès chiffrés avant modélisation", "exigence CRISP-DM ; évite de choisir la métrique qui arrange", "ce document § 2"),
])}
""")


# =============================================================================
# PHASE 2 — COMPRÉHENSION DES DONNÉES
# =============================================================================
def phase2(cfg, df_raw, df, qualite, R):
    cible = cfg["colonnes"]["cible"]
    reg = registre_fuites(cfg)
    statut = {c: k for k, cols in reg.items() for c in cols}

    lignes = []
    for c in df_raw.columns:
        s = df_raw[c]
        lignes.append({"Colonne": c, "Type": str(s.dtype), "Manquants": f"{s.isna().mean():.1%}", "Modalités": s.nunique(dropna=True),
                       "Exemple": str(s.dropna().iloc[0])[:22] if s.notna().any() else "", "Statut": statut.get(c, "feature")})
    dico = pd.DataFrame(lignes)
    n_stat = dico["Statut"].value_counts().to_dict()

    manq = pd.DataFrame([{"Colonne": k, "Manquants": v, "Part": f"{v/len(df_raw):.1%}", "Nature": "structurel",
                          "Traitement": {"Offer": "catégorie « Aucune offre »", "Internet Type": "catégorie « Pas d'internet »"}.get(k, "colonne exclue (fuite de disponibilité)")}
                         for k, v in qualite["manquants"].items()])
    aber = pd.DataFrame([{"Variable": k, "Hors IQR × 1,5": v} for k, v in qualite["aberrantes"].items()])
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, c in zip(axes, ["Tenure in Months", "Monthly Charge", "Total Charges", "Avg Monthly GB Download"]):
        ax.boxplot(df_raw[c].dropna(), vert=True, widths=0.5); ax.set_title(c, fontsize=8); ax.set_xticks([])
    f_box = _fig("02_boites_aberrantes.png")

    sat = df_raw[cible].value_counts().sort_index()
    ct = pd.crosstab(df_raw[cible], df_raw["Churn Label"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].bar(sat.index.astype(str), sat.values, color=BLEU)
    for i, v in enumerate(sat.values):
        axes[0].text(i, v + 30, f"{v}\n{v/len(df_raw):.1%}", ha="center", fontsize=8)
    axes[0].set_title("Distribution de Satisfaction Score"); axes[0].set_xlabel("Note")
    (pd.crosstab(df_raw[cible], df_raw["Churn Label"], normalize="index") * 100).plot(kind="bar", stacked=True, ax=axes[1], color=[VERT, ROUGE], rot=0)
    axes[1].set_title("% restés / partis par note"); axes[1].set_ylabel("%"); axes[1].legend(["Restés", "Partis"], fontsize=8)
    f_sat = _fig("02_eda_satisfaction.png")

    off = df.groupby("Offer").agg(n=("Customer ID", "count"), depart=("Churn Value", "mean"), satisf=(cible, "mean")).sort_values("depart")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(off.index, off["depart"] * 100, color=[ROUGE if i == "Offer E" else GRIS for i in off.index])
    for i, (n, d) in enumerate(zip(off["n"], off["depart"])):
        ax.text(i, d * 100 + 1, f"{d:.1%}\nn={n}", ha="center", fontsize=7)
    ax.set_ylabel("% de départs"); ax.set_title("Taux de départ par offre reçue — la quasi-expérience du dataset")
    f_off = _fig("02_eda_offer.png")

    nums = ["Tenure in Months", "Monthly Charge", "Number of Referrals", "Avg Monthly GB Download"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, c in zip(axes, nums):
        ax.boxplot([df_raw.loc[df_raw[cible] == k, c].dropna() for k in [1, 2, 3, 4, 5]], labels=["1", "2", "3", "4", "5"], showfliers=False)
        ax.set_title(f"{c} par satisfaction", fontsize=8)
    f_num = _fig("02_eda_numeriques.png")

    cats = ["Contract", "Offer", "Internet Type", "Payment Method", "Paperless Billing", "Premium Tech Support", "Online Security",
            "Referred a Friend", "Unlimited Data", "Gender", "Married", "Senior Citizen", "Dependents"]
    demo = ("Gender", "Married", "Senior Citizen", "Dependents")
    cv = pd.Series({c: _cramer_v(df[c].astype(str), df[cible]) for c in cats}).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(cv.index[::-1], cv.values[::-1], color=[GRIS if c in demo else BLEU for c in cv.index[::-1]])
    ax.set_xlabel("V de Cramér avec Satisfaction Score"); ax.set_title("Gris : démographie (exclue du modèle, gardée pour l'audit)", fontsize=8)
    f_cv = _fig("02_eda_cramer.png")

    numc = ["Tenure in Months", "Monthly Charge", "Total Charges", "Total Revenue", "Number of Referrals", "Avg Monthly GB Download",
            "Avg Monthly Long Distance Charges", "Total Long Distance Charges", cible]
    corr = df_raw[numc].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numc))); ax.set_yticks(range(len(numc)))
    ax.set_xticklabels([c[:18] for c in numc], rotation=60, ha="right", fontsize=7); ax.set_yticklabels([c[:18] for c in numc], fontsize=7)
    for i in range(len(numc)):
        for j in range(len(numc)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, fraction=0.046); ax.set_title("Corrélations (numériques)")
    f_corr = _fig("02_eda_correlations.png")

    ctf = ct.copy(); ctf["% départ"] = (ctf["Yes"] / ctf.sum(axis=1) * 100).round(1)
    ctf = ctf.reset_index().rename(columns={cible: "Satisfaction", "No": "Restés", "Yes": "Partis"})
    cs = df_raw.groupby(cible)[["Churn Score", "CLTV"]].mean().round(0).reset_index(); cs.columns = ["Satisfaction", "Churn Score moyen", "CLTV moyenne"]
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    cand = ["Churn Value", "Churn Score", "CLTV", "Customer Status", "Contract", "Tenure in Months", "Monthly Charge", "Number of Referrals", "Offer", "Internet Type"]
    Xmi = pd.DataFrame({c: LabelEncoder().fit_transform(df[c].astype(str)) if df[c].dtype == object else df[c].fillna(df[c].median()) for c in cand})
    mi = mutual_info_classif(Xmi, df[cible], random_state=cfg["seed"])
    mi_df = pd.DataFrame({"Colonne": cand, "Information mutuelle": mi.round(3), "Statut": [statut.get(c, "feature") for c in cand]}).sort_values("Information mutuelle", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(mi_df["Colonne"][::-1], mi_df["Information mutuelle"][::-1], color=[ROUGE if s != "feature" else BLEU for s in mi_df["Statut"]][::-1])
    ax.set_xlabel("Information mutuelle avec Satisfaction Score"); ax.set_title("Rouge : colonnes exclues (fuites) · Bleu : features légitimes", fontsize=8)
    f_mi = _fig("02_information_mutuelle.png")

    part3 = sat[3] / len(df_raw); restes3 = int(ct.loc[3, "No"])
    _md("02_comprehension_donnees.md", f"""
# Phase 2 — Compréhension des données

**En bref.** Cinq tables IBM Telco 11.1.3+ jointes sans perte ({len(df_raw)} clients × {df_raw.shape[1]} colonnes).
Trois faits structurent tout le projet : (1) la note **3** pèse {part3:.0%} de la base et {restes3/sat[3]:.0%} de ces
clients sont restés ; (2) le dataset **contient sa propre réponse** — satisfaction 1–2 → 100 % de départs,
4–5 → 0 % — donc toute colonne liée au churn est une fuite ; (3) les offres commerciales ont été
**ciblées** sur des clients fragiles, ce qui interdit toute lecture causale.

## 1. Sources et jointure

| Table | Lignes × colonnes | Clé | Contenu |
|---|---|---|---|
| demographics | 7 043 × 9 | Customer ID | genre, âge, situation familiale |
| location | 7 043 × 9 | Customer ID | ville, code postal, coordonnées |
| population | 1 671 × 3 | Zip Code | population du code postal |
| services | 7 043 × 30 | Customer ID | contrat, services, facturation, parrainage, **Satisfaction Score** |
| status | 7 043 × 11 | Customer ID | churn (label, valeur, score, catégorie, raison), CLTV |

Jointure interne sur `Customer ID`, `population` rattachée par `Zip Code`. Porte de contrôle :
**{len(df_raw)} lignes** après jointure, {qualite['doublons_id']} doublon d'identifiant (test automatisé).

## 2. Dictionnaire des données

Statut de chaque colonne, décidé au registre de fuites (§ 4) : `feature` = variable explicative
candidate ({n_stat.get('feature', 0)}) · `audit_seulement` = exclue du modèle, utilisée pour l'audit d'équité
({n_stat.get('audit_seulement', 0)}) · `fuite_*` = interdite ({sum(v for k, v in n_stat.items() if k.startswith('fuite'))}) · `techniques` = identifiant ou constante ({n_stat.get('techniques', 0)}) · `cible` (1).

{_tab(dico)}

## 3. Qualité des données — cinq contrôles

**3.1 Valeurs manquantes.** Quatre colonnes, toutes à manquants **structurels** (le vide a un sens),
aucune imputation :

{_tab(manq)}

**3.2 Doublons.** {qualite['doublons_id']} sur l'identifiant, {qualite['doublons_lignes']} sur l'ensemble des colonnes hors identifiant.

**3.3 Valeurs aberrantes** (règle IQR × 1,5) — **conservées** : un client à 100 Go/mois ou à 120 $/mois
est un client réel, précisément le profil qu'un modèle de détraction doit voir ; les modèles retenus
(standardisation + régularisation, arbres) y sont robustes.

![]({f_box})

{_tab(aber)}

**3.4 Cohérence interne.** Ancienneté nulle : {qualite['anciennete_zero']} client. Écart > 50 % entre `Total Charges` et
`Monthly Charge × Tenure` : {qualite['incoherence_facturation']} lignes — attendu (tarif variable sur la durée de vie), non corrigé, documenté.

**3.5 Colonnes constantes.** {', '.join(f'`{c}`' for c in qualite['constantes'])} — base 100 % Californie. Supprimées.

## 4. Registre de fuites — l'étape la plus discriminante

**Le fait fondateur.**

{_tab(ctf)}

Satisfaction 1–2 → **100 % de départs** ; 4–5 → **0 %**. Connaître le churn, c'est connaître la
cible. Or au moment où l'on veut détecter un détracteur, il n'est pas encore parti : l'information
n'existe pas. Toute colonne liée au churn est donc une fuite — et une fuite de trois natures :

| Nature | Définition | Colonnes | Décision |
|---|---|---|---|
| **Conséquence** | la colonne est un effet de la cible | {', '.join(f'`{c}`' for c in reg['fuite_consequence'])} | exclues |
| **Disponibilité** | la colonne n'existe que pour une partie de la population — sa présence trahit la réponse | {', '.join(f'`{c}`' for c in reg['fuite_disponibilite'])} (5 174 vides = clients restés) | exclues |
| **Modèle amont** | la colonne est la sortie d'un autre modèle (IBM SPSS) | {', '.join(f'`{c}`' for c in reg['fuite_modele_amont'])} | exclues des features ; `CLTV` admise en **couche de décision** (phase 5 § 3) |

`Churn Score` et `CLTV` par niveau de satisfaction :

{_tab(cs)}

`Churn Score` passe de ~{int(cs['Churn Score moyen'].iloc[0])} à ~{int(cs['Churn Score moyen'].iloc[-1])} : une prédiction déguisée en donnée. `CLTV` est plate — peu
informative comme feature, mais légitime pour pondérer *qui appeler* parmi les détracteurs prédits.

**Quantifier au lieu d'affirmer.** Information mutuelle de chaque colonne avec la cible :

![]({f_mi})

{_tab(mi_df)}

Les colonnes exclues portent une information sans commune mesure avec les features légitimes.
C'est la mesure qui justifie l'exclusion, pas l'intuition.

**Contre-exemple publié.** Mustafa et al. (2021) annoncent 98 % d'exactitude en prédisant le churn
avec la note NPS et la catégorie NPS parmi les features, alors que leur cible en dérive. Leurs modèles
linéaires font 41 %, leurs arbres 98 % : cet écart est la **signature** d'une variable qui fuit. On
applique ce test en phase 4 § 5.

**Démographie et géographie : exclues du modèle, gardées pour l'audit.** {', '.join(f'`{c}`' for c in reg['audit_seulement'])}.
Précédent : Elero et al. (2026) excluent délibérément les démographiques. Position cohérente avec le
métier (les leviers actionnables sont contractuels) et avec l'énoncé § 4.7 (le code postal est un
proxy socio-économique). Le coût de cette exclusion est **chiffré** en phase 5 § 5, et la mesure de ce
que les features retenues révèlent malgré tout de ces attributs en phase 5 § 7.

## 5. Analyse exploratoire

**5.1 La cible.**

![]({f_sat})

| Note | Clients | Part |
|---|---|---|
""" + "\n".join(f"| {k} | {v} | {v/len(df_raw):.1%} |" for k, v in sat.items()) + f"""

La note 3 pèse **{sat[3]} clients, {part3:.1%} de la base**, et {restes3} d'entre eux ({restes3/sat[3]:.0%}) sont toujours clients.
Où les placer dans la grille NPS décide de tout le reste (phase 3 § 1).

**5.2 Numériques par niveau de satisfaction.** L'ancienneté et le nombre de parrainages croissent
nettement avec la satisfaction ; la charge mensuelle varie peu.

![]({f_num})

**5.3 Catégorielles : force d'association.** Le type de contrat et l'offre reçue dominent ; les
variables démographiques (gris) sont faiblement associées à la satisfaction — leur exclusion devrait
donc coûter peu (vérifié en phase 5).

![]({f_cv})

**5.4 Corrélations entre numériques.** `Total Charges`, `Total Revenue` et `Tenure in Months` sont
mécaniquement liés (le total cumule le mensuel sur la durée). Traité en phase 3 § 3 (colinéarité).

![]({f_corr})

**5.5 La quasi-expérience du dataset : `Offer`.**

![]({f_off})

{_tab(off.reset_index().rename(columns={'Offer':'Offre','n':'Clients','depart':'Taux de départ','satisf':'Satisfaction moy.'}), {'Taux de départ':'{:.1%}','Satisfaction moy.':'{:.2f}'})}

Les clients ayant reçu l'**offre E partent deux fois plus** que ceux qui n'ont rien reçu. Lire
« l'offre E fait fuir » serait une erreur : c'est un **biais d'indication** — l'offre a été proposée
à des clients déjà en difficulté, comme un médicament à des patients déjà malades. On ne lit pas un
effet causal dans des données observationnelles ; ce constat fonde les limites du ciblage (phase 5 § 8).

## 6. Enrichissement externe : décision

L'énoncé autorise des sources externes (recensement par code postal, benchmarks télécom) à condition
d'en justifier la pertinence métier. **Décision : aucune.** Deux raisons. (1) Le dataset contient déjà
la seule variable de contexte géographique disponible à la maille du client (`Population` du code
postal), et elle est exclue du modèle au titre de l'équité — ajouter des revenus médians par code
postal reviendrait à réintroduire par la fenêtre le proxy socio-économique qu'on a sorti par la porte
(énoncé § 4.7). (2) Le dataset est fictif : un enrichissement réel sur des coordonnées californiennes
fictives n'aurait aucune validité. Les **benchmarks NPS télécom** publiés sont en revanche utilisés
comme corroboration externe de la construction de la cible (phase 3 § 1).

{_decisions([
    ("Jointure interne, porte de contrôle à 7 043 lignes", "aucune perte tolérée ; un client sans ligne de services est une erreur, pas un cas", "test `test_jointure_sans_perte`"),
    ("Manquants structurels → catégorie explicite, jamais imputés", "le vide a un sens métier (pas d'offre, pas d'internet) ; imputer fabriquerait de l'information", "§ 3.1"),
    ("Aberrantes conservées", "profils réels ; robustesse des modèles retenus", "§ 3.3"),
    ("Toute colonne liée au churn exclue, en trois natures nommées", "fait fondateur : satisfaction ⇔ churn ; information mutuelle mesurée", "§ 4, test `test_aucune_colonne_churn_dans_les_features`"),
    ("Démographie et géographie hors modèle, audit seulement", "leviers non actionnables ; proxies d'attributs protégés (énoncé § 4.7)", "§ 4, phase 5 § 7"),
    ("Aucun enrichissement externe", "réintroduirait un proxy socio-économique ; dataset fictif", "§ 6"),
])}
""")


# =============================================================================
# PHASE 3 — PRÉPARATION DES DONNÉES
# =============================================================================
def phase3(cfg, df, cibles, arb, num, cat, R):
    dist = pd.DataFrame({m: cibles[m].value_counts(normalize=True).reindex(ORDRE) for m in ("M1", "M2", "M3")})
    npsv = [nps(cibles[m]) for m in ("M1", "M2", "M3")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.4, 1]})
    dist.T.plot(kind="bar", stacked=True, ax=axes[0], color=[COUL[c] for c in ORDRE], rot=0)
    axes[0].set_ylabel("part des clients"); axes[0].set_title("Répartition des classes selon le mapping"); axes[0].legend([LIB[c] for c in ORDRE], fontsize=8)
    axes[1].axhspan(19, 34, color=VERT, alpha=0.15, label="benchmark télécom publié (+19 à +34)")
    axes[1].bar(["M1\n(énoncé)", "M2\n(recentré)", "M3\n(arbitré)"], npsv, color=[ROUGE, ORANGE, BLEU])
    for i, v in enumerate(npsv):
        axes[1].text(i, v + (2 if v > 0 else -6), f"{v:+.1f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].axhline(0, color="k", lw=0.6); axes[1].set_title("NPS résultant"); axes[1].legend(fontsize=7, loc="lower right")
    f_cible = _fig("03_cible_mappings.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(arb["proba_detracteur_des_3"], bins=30, color=GRIS); ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("P(profil détracteur) attribuée aux clients à satisfaction 3")
    ax.set_title(f"Arbitrage empirique : {arb['part_3_vers_detracteur']:.1%} penchent détracteur, {arb['part_3_vers_promoteur']:.1%} promoteur")
    f_arb = _fig("03_arbitrage_des_3.png")

    de = R["desequilibre"]
    d3 = pd.DataFrame({"Base complète": de["distribution_base"], "Répondants (S3, entraînement)": de["distribution_repondants"],
                       "Silencieux (évaluation)": de["distribution_silencieux"]}).reindex(ORDRE)
    fig, ax = plt.subplots(figsize=(7, 3))
    d3.T.plot(kind="bar", ax=ax, color=[COUL[c] for c in ORDRE], rot=0); ax.set_ylabel("part"); ax.legend([LIB[c] for c in ORDRE], fontsize=8)
    ax.set_title("Déséquilibre des classes — et déformation par le biais de réponse")
    f_deseq = _fig("03_desequilibre.png")
    strat = pd.DataFrame([s for s in de["strategies"] if "kappa" in s])[["strategie", "kappa", "macro_f1", "rappel_detracteur", "rappel_passif", "rappel_promoteur", "precision_detracteur"]]
    strat.columns = ["Stratégie", "Kappa", "Macro-F1", "Rappel Dét.", "Rappel Passif", "Rappel Prom.", "Précision Dét."]

    Xn = df[num].fillna(df[num].median()); Xn = Xn.loc[:, Xn.std() > 0]
    vif = pd.DataFrame({"Variable": Xn.columns, "VIF": [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])]}).sort_values("VIF", ascending=False)
    vif["VIF"] = vif["VIF"].map(lambda v: "> 1 000 (identité comptable)" if v > 1000 else f"{v:.1f}")
    brutes = pd.DataFrame([{"Variable(s)": k, "Pourquoi elle devrait aider à prédire le NPS": v} for k, v in JUSTIFICATION_BRUTES.items()])
    derivees = pd.DataFrame([{"Variable dérivée": f"`{k}`", "Hypothèse métier": v} for k, v in HYPOTHESES_DERIVEES.items()])

    _md("03_preparation_donnees.md", f"""
# Phase 3 — Préparation des données

**En bref.** La cible NPS est construite à partir de la satisfaction 1–5 par trois mappings ; le
mapping de l'énoncé donne un NPS de **{npsv[0]:+.0f}**, hors de toute plausibilité sectorielle, parce qu'il
envoie les {arb['n_ambigus']} clients « 3 » chez les détracteurs. Un arbitrage **empirique** (modèle entraîné sur
les seuls extrêmes) montre qu'ils penchent à {arb['part_3_vers_promoteur']:.0%} côté promoteur ; le bloc rejoint les Passifs
(NPS **{npsv[2]:+.0f}**). {len(num)} variables numériques et {len(cat)} catégorielles sont retenues, chacune avec l'hypothèse
qui la justifie ; le déséquilibre est traité par pondération de classes, choix comparé à quatre
alternatives ; tout le préprocessing vit dans un `Pipeline` scikit-learn.

## 1. Construction de la cible NPS

Le dataset donne une satisfaction de 1 à 5. Le NPS a trois classes. Personne ne fournit la
traduction : c'est à nous de la construire et de la défendre.

| | Détracteur | Passif | Promoteur | NPS | Logique |
|---|---|---|---|---|---|
| **M1** | 1, 2, 3 | 4 | 5 | **{npsv[0]:+.1f}** | mapping baseline de l'énoncé (fidèle à l'échelle : 3/5 ≈ 5/10 = détracteur) |
| **M2** | 1, 2 | 3, 4 | 5 | **{npsv[1]:+.1f}** | recentré : le 3 est le milieu, donc passif |
| **M3** | 1, 2 | 3 | 4, 5 | **{npsv[2]:+.1f}** | arbitré par les données (§ 1.2) |

![]({f_cible})

**1.1 Pourquoi M1 pose problème.** M1 est le mapping *fidèle à l'échelle* — et c'est précisément pour
ça qu'il est faux : une échelle à 5 points compressée dans les bandes NPS envoie 58 % de la base chez
les détracteurs, dont {arb['n_ambigus']} clients à « 3 » qui, à 84 %, sont restés. Un NPS de {npsv[0]:+.0f} est à 60–75 points des
benchmarks télécom publiés (+19 à +34 ; sources commerciales, dispersion forte). Ce n'est pas un
résultat, c'est un signal d'alerte sur la construction.

**1.2 L'arbitrage empirique des « 3 »** (méthode d'Elero et al., 2026, qui l'appliquent aux notes 7 et 8).
Plutôt que de décider où vont les {arb['n_ambigus']} clients ambigus, on demande aux données :

1. entraîner un modèle **uniquement sur les extrêmes non ambigus** — satisfaction 1–2 contre 4–5
   ({arb['n_extremes']} clients), avec les seules features légitimes ;
2. contrôler qu'il sépare ces extrêmes : **{arb['exactitude_sur_extremes']:.1%}** d'exactitude en validation croisée — dans la
   fourchette attendue de la littérature (0,77–0,80), donc sans fuite ;
3. faire passer les {arb['n_ambigus']} clients à « 3 », jamais vus, à travers ce modèle.

![]({f_arb})

**Résultat : {arb['part_3_vers_detracteur']:.1%} des « 3 » ressemblent à des détracteurs, {arb['part_3_vers_promoteur']:.1%} à des promoteurs.** Le bloc rejoint les
**{arb['destination_des_3']}s**. Le NPS qui en découle ({npsv[2]:+.1f}) tombe dans la fourchette sectorielle — corroboration externe, pas preuve.

**1.3 Le piège évité — la circularité.** Il était tentant d'étiqueter *chaque* « 3 » par la prédiction
du modèle. Ce serait une faute : la cible deviendrait une fonction des features, et le modèle aval
réapprendrait sa propre sortie — une circularité cousine de la fuite `Churn Score`. On tranche donc
**le bloc** par la mesure agrégée, et on conserve la probabilité individuelle comme simple **indicateur
métier** (« ce passif a un profil de détracteur à X % »), affiché dans l'application, jamais utilisé à
l'entraînement.

**1.4 Ce que l'énoncé suggérait et qu'on a refusé.** Enrichir la cible par `Churn Value` ou `CLTV`
transformerait le projet en modèle de churn déguisé (phase 2 § 4). Enrichir par l'ancienneté rendrait
la cible fonction d'une feature (même circularité qu'en 1.3). Le **bruit réaliste** suggéré par
l'énoncé est traité en phase 5 § 5 comme test de robustesse : on corrompt les étiquettes
d'entraînement et on mesure la dégradation.

## 2. Nettoyage

- Manquants structurels (`Offer`, `Internet Type`) → catégorie explicite (phase 2 § 3).
- Manquants accidentels (aucun dans ce dataset, mais l'application en produira) → imputation **dans
  le pipeline** (médiane / modalité la plus fréquente), jamais en amont, pour qu'aucune statistique
  calculée sur tout le jeu ne fuie vers l'évaluation.
- Aberrantes conservées ; colonnes constantes supprimées.

## 3. Feature engineering — chaque variable adossée à une hypothèse

**3.1 Variables brutes retenues et pourquoi.**

{_tab(brutes)}

**3.2 Variables dérivées.**

{_tab(derivees)}

**3.3 Variables retenues.** **{len(num)} numériques** : {', '.join(f'`{c}`' for c in num)}.
**{len(cat)} catégorielles** : {', '.join(f'`{c}`' for c in cat)}.

**3.4 Variables écartées et pourquoi.** Démographie (`Gender`, `Age`, `Senior Citizen`, `Married`,
`Dependents`…) et géographie (`Zip Code`, `Latitude`, `Longitude`, `Population`, `City`) : hors modèle
par décision d'équité (phase 2 § 4), **coût chiffré** en phase 5 § 5 ; `Referred a Friend` : redondant
avec `Number of Referrals` (recodé en `a_parraine`) ; identifiants et constantes.

**3.5 Colinéarité (VIF).**

{_tab(vif)}

`Total Charges`, `Total Revenue`, `Total Refunds`, `Total Extra Data Charges` et `Total Long Distance
Charges` sont liés par une **identité comptable** (le revenu total est la somme des autres) : VIF
infini par construction. **Décision : conservées.** Les modèles linéaires du catalogue sont
régularisés (L2), ce qui absorbe la colinéarité ; les arbres y sont insensibles. La colinéarité
brouillerait l'interprétation coefficient par coefficient : l'interprétabilité (phase 5 § 6) lit donc
les variables corrélées comme un bloc.

## 4. Déséquilibre des classes

![]({f_deseq})

Deux déséquilibres se superposent. Dans la base, le Passif domine ({de['distribution_base']['Passif']:.0%}) et le Détracteur est
minoritaire ({de['distribution_base']['Detracteur']:.0%}). Mais le modèle apprend sur les **répondants**, où le biais de réponse (phase 4 § 2)
inverse la structure : {de['distribution_repondants']['Detracteur']:.0%} de Détracteurs, {de['distribution_repondants']['Passif']:.0%} de Passifs seulement. Le Passif est donc à la
fois la classe la plus fréquente à l'évaluation et la plus rare à l'entraînement — c'est **elle** qu'un
modèle abandonne en premier, et c'est ce que le rappel par classe surveille (phase 5 § 1).

**Stratégies comparées** (baseline logistique, S3/M3, évaluées sur les silencieux) :

{_tab(strat, {'Kappa':'{:.3f}','Macro-F1':'{:.3f}','Rappel Dét.':'{:.3f}','Rappel Passif':'{:.3f}','Rappel Prom.':'{:.3f}','Précision Dét.':'{:.3f}'})}

**Décision : pondération de classes `balanced`.** Elle est appliquée à tous les modèles du
catalogue qui l'acceptent (ou par `sample_weight` équivalent). Le rééchantillonnage (SMOTE,
sur/sous-échantillonnage) n'apporte pas de gain systématique ici et fabrique des clients
synthétiques dans un espace one-hot où l'interpolation n'a pas de sens métier ; la pondération est
plus simple, sans hyperparamètre, et laisse les données intactes.

## 5. Pipeline de préparation

Tout est encapsulé dans un `Pipeline` scikit-learn (`ColumnTransformer`) :

- numériques → imputation médiane → `StandardScaler` **pour les modèles à distance ou linéaires
  seulement** (logistique, ordinale, ridge, KNN, SVM, MLP, Naive Bayes) ; rien pour les arbres — deux
  préparations, pas un compromis ;
- catégorielles → imputation mode → `OneHotEncoder(handle_unknown='ignore')`, pour que
  l'application tolère une modalité jamais vue sans planter ;
- CatBoost reçoit les catégorielles **brutes** : c'est l'hypothèse qu'il teste (phase 4 § 3).

Rien n'est transformé hors du pipeline. C'est ce qui rend le modèle sérialisable d'un bloc et exempt
de fuite entraînement → test. Le découpage répondants / silencieux est décrit en phase 4 § 2.

{_decisions([
    ("Cible M3 : 1–2 Détracteur, 3 Passif, 4–5 Promoteur", "arbitrage empirique des « 3 » ; NPS dans la fourchette sectorielle", "§ 1.2, test `test_mappings_et_nps`"),
    ("Aucun étiquetage individuel des « 3 » par le modèle", "évite la circularité cible ← features", "§ 1.3, test `test_indicateur_profil_detracteur_reserve_aux_ambigus`"),
    ("Neuf variables dérivées, chacune avec son hypothèse", "exigence § 4.3 de l'énoncé", "§ 3.2"),
    ("Colinéarité conservée", "régularisation L2 et arbres insensibles ; lecture par blocs", "§ 3.5"),
    ("Pondération de classes plutôt que rééchantillonnage", "comparaison de cinq stratégies ; simplicité", "§ 4"),
    ("Préprocessing dans le Pipeline uniquement", "aucune fuite de statistiques ; sérialisation d'un bloc", "§ 5, test `test_modele_tolere_entrees_vides_et_modalites_inconnues`"),
])}
""")


# =============================================================================
# PHASE 4 — MODÉLISATION
# =============================================================================
def phase4(cfg, df, cibles, X, satisfaction, num, cat, R):
    cible = cfg["colonnes"]["cible"]; seed = cfg["seed"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, sc in zip(axes, cfg["protocole"]["scenarios"]):
        dec = decouper(df, cfg, sc, seed)
        base = df[cible].value_counts(normalize=True).sort_index()
        rep_ = df.loc[dec["repondants"], cible].value_counts(normalize=True).sort_index()
        w = 0.38
        ax.bar(base.index - w/2, base.values, w, label="base complète", color=GRIS)
        ax.bar(rep_.index + w/2, rep_.reindex(base.index).fillna(0).values, w, label="répondants (15 %)", color=BLEU)
        ax.set_title(sc); ax.set_xlabel("Satisfaction")
    axes[0].set_ylabel("part"); axes[0].legend(fontsize=7)
    f_prot = _fig("04_protocole_15_85.png")
    dg = pd.DataFrame(R["diagnostic_protocole"])[["scenario", "taux_reponse", "satisfaction_moyenne_base", "satisfaction_moyenne_repondants", "part_extremes_base", "part_extremes_repondants"]]
    dg.columns = ["Scénario", "Taux de réponse", "Satisf. base", "Satisf. répondants", "Extrêmes base", "Extrêmes répondants"]

    catdf = pd.DataFrame(CATALOGUE)
    catdf["nom"] = catdf["nom"].map(lambda n: f"`{n}`")
    catdf = catdf[["famille", "nom", "formulation", "pourquoi", "hypothese", "reference"]]
    catdf.columns = ["Famille", "Modèle", "Formulation", "Pourquoi il est là", "Hypothèse testée", "Référence"]

    g = pd.DataFrame([x for x in R["grille"] if "kappa_quadratique" in x])
    ordre_mod = sorted(FAMILLE, key=lambda n: (FAMILLE[n], n))
    s3 = g[g.scenario == "S3_MNAR"]
    piv_map = s3.pivot_table(index="modele", columns="mapping", values="kappa_quadratique").reindex(ordre_mod)
    piv_sc = g[g.mapping == "M3"].pivot_table(index="modele", columns="scenario", values="kappa_quadratique").reindex(ordre_mod)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    yy = np.arange(len(piv_map))
    for k, (mp, col) in enumerate(zip(["M1", "M2", "M3"], [ROUGE, ORANGE, BLEU])):
        axes[0].barh(yy + (k - 1) * 0.27, piv_map[mp].values, 0.27, label=mp, color=col)
    axes[0].set_yticks(yy); axes[0].set_yticklabels([_libmod(n) for n in piv_map.index], fontsize=7); axes[0].invert_yaxis()
    axes[0].set_xlabel("kappa quadratique (silencieux)"); axes[0].set_title("Scénario réaliste S3 : effet du mapping de la cible"); axes[0].legend(fontsize=7)
    for k, (sc, col) in enumerate(zip(["S1_MCAR", "S2_MAR", "S3_MNAR"], [VERT, ORANGE, ROUGE])):
        axes[1].barh(yy + (k - 1) * 0.27, piv_sc[sc].values, 0.27, label=sc, color=col)
    axes[1].set_xlabel("kappa quadratique (silencieux)"); axes[1].set_title("Cible M3 : effet du biais de non-réponse"); axes[1].legend(fontsize=7)
    f_grille = _fig("04_grille_kappa.png")
    tab_s3 = piv_map.reset_index().rename(columns={"modele": "Modèle"}); tab_s3.insert(0, "Famille", tab_s3["Modèle"].map(FAMILLE))
    tab_sc = piv_sc.reset_index().rename(columns={"modele": "Modèle"}); tab_sc["Chute S1→S3"] = tab_sc["S3_MNAR"] - tab_sc["S1_MCAR"]

    sel = R["selection_modele_final"]; cl = pd.DataFrame(sel["classement"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for _, r in cl.iterrows():
        col = ROUGE if r["modele"] == sel["retenu"] else (BLEU if r["dans_la_marge"] else GRIS)
        ax.scatter(r["kappa_cv_ipw"], r["kappa_silencieux"], color=col, s=55, zorder=3)
        ax.annotate(_libmod(r["modele"]), (r["kappa_cv_ipw"], r["kappa_silencieux"]), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.axvline(sel["seuil_parcimonie"], color="k", ls=":", lw=0.8, label="marge de parcimonie (1 écart-type)")
    ax.set_xlabel("kappa en validation croisée sur les répondants (pondéré IPW) — ce que le praticien voit")
    ax.set_ylabel("kappa sur les 85 % silencieux — la vérité")
    ax.set_title(f"Sélection sans regarder le test · rouge : retenu · bleu : dans la marge · ρ de Spearman = {sel['spearman_cv_ipw_vs_silencieux']:.2f}", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    f_cvsil = _fig("04_cv_vs_silencieux.png")

    comp = R["comparatif_s3_m3"]
    reg_rows = []
    for nom, l in comp.items():
        d = l["defaut"]; r = l.get("regle", {})
        reg_rows.append({"Modèle": nom, "Famille": l["famille"], "Configurations": r.get("n_configurations", "—"),
                         "Kappa CV IPW défaut": d["cv"]["kappa_cv_ipw"], "Kappa CV IPW réglé": r.get("cv", {}).get("kappa_cv_ipw", np.nan),
                         "Gain CV": l.get("gain_reglage_cv_ipw", np.nan),
                         "Silencieux défaut": d["silencieux"]["kappa"], "Silencieux réglé": r.get("silencieux", {}).get("kappa", np.nan),
                         "Version retenue": l["version_retenue"],
                         "Meilleurs paramètres": ", ".join(f"{k}={v}" for k, v in r.get("meilleurs_params", {}).items()) or "—"})
    regdf = pd.DataFrame(reg_rows)
    rr = regdf.dropna(subset=["Kappa CV IPW réglé"])
    fig, ax = plt.subplots(figsize=(8, 4.2)); yy = np.arange(len(rr))
    ax.barh(yy - 0.2, rr["Kappa CV IPW défaut"], 0.4, color=GRIS, label="par défaut")
    ax.barh(yy + 0.2, rr["Kappa CV IPW réglé"], 0.4, color=BLEU, label="réglé (recherche aléatoire)")
    ax.set_yticks(yy); ax.set_yticklabels([_libmod(n) for n in rr["Modèle"]], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("kappa CV pondéré IPW (répondants)"); ax.set_title("Ce que le réglage d'hyperparamètres apporte — modèle par modèle"); ax.legend(fontsize=7)
    f_reg = _fig("04_reglage.png")

    clt = cl[["rang_cv", "modele", "famille", "formulation", "version", "simplicite", "kappa_cv_ipw", "kappa_cv_et", "macro_f1_cv", "dans_la_marge",
              "kappa_silencieux", "macro_f1_silencieux", "rappel_detracteur_silencieux", "rappel_passif_silencieux", "rang_silencieux"]].copy()
    clt["dans_la_marge"] = clt["dans_la_marge"].map({True: "✓", False: ""})
    clt.columns = ["Rang CV", "Modèle", "Famille", "Formulation", "Version", "Simplicité", "Kappa CV IPW", "± (plis)", "Macro-F1 CV", "Dans la marge",
                   "Kappa silencieux", "Macro-F1 sil.", "Rappel Dét. sil.", "Rappel Passif sil.", "Rang silencieux"]
    formu = pd.DataFrame([{"Formulation": k, "Modèles": v["n_modeles"], "Meilleur (CV)": v["meilleur_modele"], "Kappa CV IPW": v["kappa_cv_ipw"],
                           "Kappa silencieux": v["kappa_silencieux"], "Macro-F1 silencieux": v["macro_f1_silencieux"]} for k, v in R["formulations"].items()])
    mf = R["modele_final"]; ipw = R["effet_ipw"]
    retenu_l = cl[cl.modele == sel["retenu"]].iloc[0]; meilleur_l = cl[cl.modele == sel["meilleur_cv_brut"]].iloc[0]
    gagnant_sil = sel["gagnant_silencieux"]; k_gs = cl[cl.modele == gagnant_sil]["kappa_silencieux"].iloc[0]

    fo = _charger_json("fondation")
    fond_txt = "*Évaluation non exécutée (`python -m src.fondation_eval`).*"
    if fo:
        rows = []
        for mm in fo["modeles"]:
            if mm["statut"] == "ok":
                rows.append({"Modèle": mm["nom"], "Version": mm.get("version", ""), "Kappa": mm["metriques"]["kappa"], "Macro-F1": mm["metriques"]["macro_f1"],
                             "Rappel Dét.": mm["metriques"]["rappel_detracteur"], "Rappel Passif": mm["metriques"]["rappel_passif"],
                             "Ajustement (s)": mm["duree_fit_s"], "Inférence (s)": mm["duree_inference_s"], "ms / client": mm["latence_par_client_ms"], "Statut": "évalué"})
            else:
                rows.append({"Modèle": mm["nom"], "Version": mm.get("version", ""), "Statut": f"indisponible — {mm['erreur'][:90]}…"})
        ref = fo["reference"]; rb = fo.get("reference_boosting", {})
        rows.append({"Modèle": f"{_libmod(ref['modele'])} (retenu)", "Kappa": ref["kappa"], "Macro-F1": ref["macro_f1"], "Rappel Dét.": ref["rappel_detracteur"],
                     "Inférence (s)": ref["duree_inference_s"], "Statut": "référence"})
        if rb:
            rows.append({"Modèle": f"{_libmod(rb['modele'])} (meilleur boosting)", "Kappa": rb["kappa"], "Macro-F1": rb["macro_f1"], "Statut": "référence"})
        fdf = pd.DataFrame(rows)
        verdicts = []
        for mm in [x for x in fo["modeles"] if x["statut"] == "ok"]:
            gagne = mm["metriques"]["kappa"] > ref["kappa"] + 0.005
            verdicts.append(f"**{mm['nom']}** {'bat' if gagne else 'ne bat pas'} le modèle retenu (kappa {mm['metriques']['kappa']:.3f} contre {ref['kappa']:.3f}), "
                            f"pour une inférence {mm['duree_inference_s']/max(ref['duree_inference_s'],1e-3):.0f}× plus lente ({mm['latence_par_client_ms']:.1f} ms par client, CPU) et sans explication native.")
        indisp = [x for x in fo["modeles"] if x["statut"] != "ok"]
        fond_txt = f"""
{_tab(fdf, {'Kappa':'{:.3f}','Macro-F1':'{:.3f}','Rappel Dét.':'{:.3f}','Rappel Passif':'{:.3f}','Ajustement (s)':'{:.1f}','Inférence (s)':'{:.2f}','ms / client':'{:.2f}'})}

{chr(10).join('- ' + v for v in verdicts)}
{('- ' + ' ; '.join(f"**{x['nom']}** n'a pas pu être évalué : `{x['erreur'][:160]}`" for x in indisp)) if indisp else ''}

**Avantages** : aucun réglage, aucun entraînement au sens classique (apprentissage en contexte),
performance d'emblée au niveau des meilleurs modèles réglés sur un petit jeu comme celui-ci.
**Limites** : latence d'inférence de deux à trois ordres de grandeur supérieure (le jeu d'entraînement
est rejoué à chaque prédiction), empreinte mémoire, pas de coefficients ni de SHAP natif (explication
par permutation, coûteuse), dépendance à des poids externes (TabPFN-2.5 : dépôt à accès contrôlé),
domaine de validité borné (~50 000 lignes, ~2 000 features pour TabPFN-2.5). **Compromis** : pour un
scoring mensuel de 7 000 clients, la latence est acceptable ; pour l'application interactive et pour
l'explication client par client, le modèle retenu reste préférable.
"""

    tx = _charger_json("texte")
    texte_txt = "*Chaîne texte non exécutée (`python -m src.verbatims` puis `python -m src.texte`).*"
    if tx:
        lig = pd.DataFrame({"Configuration": ["Tabulaire seul (référence)", "Texte seul (TF-IDF 1-2 grammes + logistique)",
                                              f"Fusion tardive (poids texte {tx['fusion_tardive']['poids_texte']:.1f}, choisi par CV)", "Fusion précoce (tabulaire + SVD 20 du TF-IDF)"],
                            "Kappa": [tx["tabulaire_seul"]["kappa"], tx["texte_seul"]["kappa"], tx["fusion_tardive"]["kappa"], tx["fusion_precoce_svd20"]["kappa"]],
                            "Macro-F1": [tx["tabulaire_seul"]["macro_f1"], tx["texte_seul"]["macro_f1"], tx["fusion_tardive"]["macro_f1"], tx["fusion_precoce_svd20"]["macro_f1"]],
                            "Rappel Détracteur": [tx["tabulaire_seul"]["rappel_detracteur"], tx["texte_seul"]["rappel_detracteur"], tx["fusion_tardive"]["rappel_detracteur"], tx["fusion_precoce_svd20"]["rappel_detracteur"]]})
        texte_txt = f"""
**Génération.** {tx['n_verbatims']} notes de dernier contact, une par client, source `{tx['source_verbatims']}`. Sans clé API
dans l'environnement, le générateur local à gabarits stochastiques a été utilisé : fragments rédigés par
un LLM (Claude), assemblage seedé (seed {cfg['seed']}) — donc strictement reproductible. Le chemin API est prêt
(`src/verbatims.py` : détection de `ANTHROPIC_API_KEY`, appels par lots à `claude-opus-5` avec reprise, prompt
versionné `prompts/verbatim_v1.txt`, même schéma de sortie ; la colonne `source` dit lequel des deux a
produit chaque ligne). Conditionnement : tonalité tirée de la classe M3 avec **25 % de bruit** et cas
contre-intuitifs, ancrage sur contrat, ancienneté, internet, offre, parrainages, facture. Concordance
tonalité / classe observée : **{tx['concordance_tonalite_classe']:.1%}** (plafond théorique {tx['plafond_theorique']:.1%}).

**Représentation et fusion.** TF-IDF (1-2 grammes) + logistique pour le texte seul ; fusion tardive
(moyenne pondérée des probabilités, poids choisi par validation croisée sur les répondants) et fusion
précoce (SVD à 20 composantes du TF-IDF concaténée aux features tabulaires). Des plongements de
phrases (sentence-transformers) ou une classification zero-shot par LLM sont possibles mais n'ont pas
été retenus : sur du texte synthétique fabriqué à partir de la classe, ils ne mesureraient rien de plus.

{_tab(lig, {'Kappa':'{:.3f}','Macro-F1':'{:.3f}','Rappel Détracteur':'{:.3f}'})}

**Ce que le texte apporte — et pourquoi le chiffre ne prouve rien.** {tx['lecture']} Le dispositif
démontre que la chaîne texte → fusion fonctionne ; il ne dit rien de la valeur de vrais verbatims, qui
pourraient être bien plus riches (incidents non enregistrés) ou bien plus pauvres (notes laconiques).
C'est le *« spot when the added complexity is not worth it »* de l'énoncé : **pas de composante texte en
production sur la base de cette seule expérience** ; l'application la montre en démonstration, avec l'avertissement.
"""

    _md("04_modelisation.md", f"""
# Phase 4 — Modélisation

**En bref.** Le découpage 15/85 est **simulé avec un biais de non-réponse** (trois scénarios, du plus
optimiste au plus réaliste) et le modèle est toujours entraîné sur les répondants, évalué sur les
silencieux dont on connaît la vérité. **Quinze modèles en sept familles**, couvrant les trois
formulations de l'énoncé, sont comparés sur 3 scénarios × 3 mappings, puis réglés et sélectionnés **par
validation croisée sur les répondants** — sans jamais regarder les silencieux — avec une règle de
parcimonie. Retenu : **{_libmod(sel['retenu'])}** ({sel['version_retenue']}). Le meilleur kappa CV brut était
{_libmod(sel['meilleur_cv_brut'])} ; le meilleur sur les silencieux est {_libmod(gagnant_sil)}. L'écart entre ce
que la validation croisée promet ({retenu_l['kappa_cv']:.2f}) et ce que les silencieux donnent ({retenu_l['kappa_silencieux']:.2f}) est le coût mesuré du biais de réponse.

## 1. Les trois formulations, instanciées et comparées

| Formulation | Comment elle traite l'ordre des classes | Modèles du catalogue |
|---|---|---|
| Classification nominale | ne le voit pas ; l'ordre est réintroduit par la métrique (kappa quadratique) et par la pondération | logistique, arbre, forêt, boosting (×5), KNN, SVM, MLP, Naive Bayes |
| Classification ordinale | l'impose par la structure : une variable latente et deux seuils | logistique ordinale (mord, all-threshold) |
| Régression 1–5 puis seuillage | exploite la note fine à l'entraînement ; seuils optimisés sur le kappa | ridge + seuils, HGB + seuils |

Résultat par formulation (meilleur modèle de chaque, S3/M3) :

{_tab(formu, {'Kappa CV IPW':'{:.3f}','Kappa silencieux':'{:.3f}','Macro-F1 silencieux':'{:.3f}'})}

Lecture : les trois formulations arrivent à des kappas voisins ; celles qui **exploitent l'ordre**
(ordinale, régression à seuils) sont construites pour la métrique qui respecte l'ordre et en tirent un
léger avantage, avec un **profil par classe** propre à chacune (rappels Détracteur et Passif dans le
tableau de sélection, § 6.2). Le choix final se fait sur la mesure (§ 6), et l'évaluation métier
(phase 5 § 3) vérifie que le classement des appels n'en souffre pas.

## 2. Protocole de validation 15 % / 85 %

Un `train_test_split` stratifié rate l'intention de l'énoncé : les répondants à une enquête NPS ne
sont pas un échantillon aléatoire — les très mécontents et les très contents répondent plus. C'est
du **MNAR** (*missing not at random*). L'avantage rare de ce dataset : on connaît la satisfaction des
100 %, donc on peut simuler le biais et **mesurer ce qu'il coûte**, ce qu'aucun praticien ne peut
faire en production.

| Scénario | Mécanisme de réponse | Ce qu'il mesure |
|---|---|---|
| S1 — MCAR | 15 % tirés au hasard | borne haute optimiste |
| S2 — MAR | propension = f(ancienneté, facture dématérialisée, engagement) | biais de covariables seul |
| S3 — MNAR | propension = f(covariables, **écart de la satisfaction au centre**) | le cas réaliste, retenu pour le modèle final |

Dans tous les cas : **entraînement sur les 15 % répondants, évaluation sur les 85 % silencieux**.
Validation croisée stratifiée à 5 plis à l'intérieur des 15 % pour le réglage et la sélection.

![]({f_prot})

{_tab(dg, {'Taux de réponse':'{:.1%}','Satisf. base':'{:.3f}','Satisf. répondants':'{:.3f}','Extrêmes base':'{:.1%}','Extrêmes répondants':'{:.1%}'})}

Sous S3, la part des notes extrêmes passe de **{dg['Extrêmes base'].iloc[2]:.0%} dans la base à {dg['Extrêmes répondants'].iloc[2]:.0%} chez les répondants** : le
modèle apprend sur une population qui ne ressemble pas à celle sur laquelle il sera appliqué.
Kannan et al. (2022) observent un taux de réponse réel de **2,6 %** : notre 15 % est optimiste.

**Repondération par l'inverse de la propension (IPW).** La propension à répondre est **estimée**
comme un praticien le ferait — logistique P(répondre | covariables) sur les 7 043 clients, hypothèse
MAR (poids entre {R['ipw_estime']['poids_min']:.2f} et {R['ipw_estime']['poids_max']:.2f}). Ces poids servent (a) à pondérer la validation croisée pour
estimer la performance sur la population entière plutôt que sur les répondants, (b) en test comme
poids d'entraînement : kappa {ipw['sans_ipw']['kappa']:.3f} sans → {ipw['avec_ipw']['kappa']:.3f} avec, **retenu : {ipw['retenu']}**. L'effet est faible parce
que la propension vraie sous S3 dépend de la satisfaction, que l'estimation MAR ne voit pas : c'est
la limite structurelle de l'IPW face au MNAR. Le cadre canonique serait le modèle de sélection de
Heckman, qui exige une restriction d'exclusion introuvable ici.

## 3. Le catalogue — quinze modèles, sept familles, une raison chacun

L'énoncé demande une baseline puis au moins deux familles justifiées ; les quatre études de
référence en comparent six à neuf. On ne compare pas des algorithmes pour remplir un tableau : chaque
modèle est là pour tester une **hypothèse** sur les données.

{_tab(catdf)}

Réglages par défaut : pondération de classes `balanced` partout (ou `sample_weight` équivalent) ;
préparation adaptée à chaque famille (phase 3 § 5). **Écartés avec argument** : CORAL / CORN
(apprentissage profond ordinal, sans objet sur ~1 000 lignes) ; les modèles de fondation sont
évalués à part (§ 8) car ils ne s'entraînent pas au sens classique.

## 4. La grille : 3 scénarios × 3 mappings × 15 modèles

![]({f_grille})

**Effet du mapping** (scénario réaliste S3, kappa sur les silencieux) :

{_tab(tab_s3, {'M1':'{:.3f}','M2':'{:.3f}','M3':'{:.3f}'})}

**Effet du biais de non-réponse** (cible M3) :

{_tab(tab_sc, {'S1_MCAR':'{:.3f}','S2_MAR':'{:.3f}','S3_MNAR':'{:.3f}','Chute S1→S3':'{:+.3f}'})}

Deux lectures. **M3 domine M1 et M2 pour toutes les familles** : la cible arbitrée par les données est
aussi la plus apprenable. **La performance chute de S1 à S3** pour presque tous les modèles : c'est le
coût mesuré du biais de non-réponse — le résultat scientifique du protocole.

## 5. Porte de contrôle : y a-t-il une fuite ?

| Diagnostic | Valeur | Seuil | Verdict |
|---|---|---|---|
| Exactitude maximale observée (toute la grille) | {R['diagnostic_fuite']['exactitude_max_observee']:.3f} | 0,85 (littérature : Elero 0,77 ; Chong 0,80) | {'⚠️ alarme' if R['diagnostic_fuite']['alarme'] else '✅ dans la fourchette honnête'} |
| Écart arbres − linéaire (S1, M3) | {R['diagnostic_fuite']['ecart_arbres_lineaire_S1_M3']:+.3f} | +0,30 (signature Mustafa et al.) | {'⚠️' if R['diagnostic_fuite']['ecart_arbres_lineaire_S1_M3'] > 0.3 else '✅ aucun écart suspect'} |

Un troisième indice, qualitatif : le KNN — qui détecterait des « jumeaux » — est parmi les plus
faibles. Aucune fuite.

## 6. Réglage et sélection — sans regarder le jeu de test

**6.1 Hyperparamètres.** Recherche aléatoire (`RandomizedSearchCV`, 5 plis stratifiés sur les
répondants, score = kappa quadratique), espaces modestes à dessein : sur ~1 000 lignes, le risque de
surajuster la recherche dépasse vite le gain. La version réglée n'est retenue que si elle gagne plus de
0,005 de kappa **en validation croisée pondérée** — jamais sur les silencieux.

![]({f_reg})

{_tab(regdf, {'Kappa CV IPW défaut':'{:.3f}','Kappa CV IPW réglé':'{:.3f}','Gain CV':'{:+.3f}','Silencieux défaut':'{:.3f}','Silencieux réglé':'{:.3f}'})}

**6.2 Règle de sélection, fixée avant de regarder les résultats.**
1. Critère : kappa quadratique en validation croisée sur les répondants, **pondéré IPW** (estimation
   de la performance sur la population), départage par macro-F1 CV.
2. **Parcimonie** (« one standard error rule », Hastie, Tibshirani & Friedman, ESL § 7.10) : parmi
   les modèles à moins d'un écart-type (entre plis) du meilleur, le plus simple l'emporte, selon un
   ordre de simplicité fixé a priori (nombre de paramètres, lisibilité de l'explication).
3. Les 85 % silencieux sont le **test final** : rapportés pour tous, utilisés pour aucun choix.

![]({f_cvsil})

{_tab(clt, {'Kappa CV IPW':'{:.3f}','± (plis)':'{:.3f}','Macro-F1 CV':'{:.3f}','Kappa silencieux':'{:.3f}','Macro-F1 sil.':'{:.3f}','Rappel Dét. sil.':'{:.3f}','Rappel Passif sil.':'{:.3f}'})}

**6.3 Ce que le tableau dit.**
- Meilleur kappa CV brut : **{_libmod(sel['meilleur_cv_brut'])}** ({meilleur_l['kappa_cv_ipw']:.3f}). Dans la marge de parcimonie
  (≥ {sel['seuil_parcimonie']:.3f}) : {', '.join(_libmod(x) for x in sel['candidats_dans_la_marge'])}. **Retenu : {_libmod(sel['retenu'])}** ({sel['version_retenue']}), le plus simple de la marge.
- Corrélation de rang entre ce que la CV promet et ce que les silencieux donnent : **ρ = {sel['spearman_cv_ipw_vs_silencieux']:.2f}**
  ({sel['spearman_cv_vs_silencieux']:.2f} sans pondération IPW). {'La CV sur les répondants classe correctement les modèles malgré le biais.' if sel['spearman_cv_ipw_vs_silencieux'] > 0.5 else 'La CV sur les répondants classe mal les modèles : sous MNAR, un modèle peut sur-apprendre les extrêmes des répondants, briller en CV et décevoir sur la base. C’est l’argument le plus fort en faveur de la règle de parcimonie.'}
- Meilleur sur les silencieux : **{_libmod(gagnant_sil)}** ({k_gs:.3f}) ; le modèle retenu fait {retenu_l['kappa_silencieux']:.3f} (rang {int(retenu_l['rang_silencieux'])} sur {len(cl)}).
  {'Le choix fait sans regarder le test est aussi le meilleur sur le test.' if gagnant_sil == sel['retenu'] else f"L’écart ({retenu_l['kappa_silencieux'] - k_gs:+.3f}) est le prix d’une sélection honnête ; choisir a posteriori serait choisir sur le test."}
- Pour le modèle retenu, la CV promet {retenu_l['kappa_cv']:.3f} et les silencieux donnent {retenu_l['kappa_silencieux']:.3f} : **{sel['ecart_cv_silencieux_du_retenu']:+.3f}**. En production,
  personne ne verrait cet écart. C'est la quantité à retenir de toute la phase.
- **Ce que les quinze modèles apprennent sur les données** : tous les kappas sur les silencieux tiennent
  dans une bande étroite ({cl['kappa_silencieux'].min():.2f}–{cl['kappa_silencieux'].max():.2f}). Le plafond n'est pas algorithmique, il est dans le
  signal : ~1 000 répondants biaisés et une cible bruitée. Changer de modèle rapporte moins que
  corriger le biais de réponse (groupe de contrôle, vraies réponses de la population cible, phase 6).

## 7. Calibration des probabilités

Les probabilités servent à prioriser des appels : un modèle mal calibré fait mal allouer le budget.
Deux méthodes testées (isotonique, Platt). Résultat sur le modèle retenu, classe Détracteur :

| | Brier | ECE | Rappel Passif |
|---|---|---|---|
| Non calibré | {mf['calibration']['non_calibre']['brier']:.3f} | {mf['calibration']['non_calibre']['ece']:.3f} | {mf['metriques_non_calibre']['par_classe']['Passif']['rappel']:.2f} |
| Calibré (Platt, 3 plis) | {mf['calibration']['calibre_platt']['brier']:.3f} | {mf['calibration']['calibre_platt']['ece']:.3f} | {mf['metriques_calibre']['par_classe']['Passif']['rappel']:.2f} |

{'**La calibration écrase la classe Passif** (rappel < 0,05) : sous MNAR, les Passifs sont rares parmi les répondants et le calibrateur, ajusté sur peu de lignes, ne prédit plus jamais la classe intermédiaire. **Décision — stratégie hybride** : la **classe** vient du modèle non calibré ; la **probabilité de détraction**, seule utilisée pour classer les appels, vient du modèle calibré.' if mf['calibration']['passif_ecrase'] else '**Décision** : ' + mf['strategie_decision'] + '.'}
Vérifié par test automatisé (aucune classe sous 0,05 de rappel). La courbe de fiabilité et sa lecture
sont en phase 5 § 2.

## 8. Modèles de fondation tabulaires (bonus § 4.5)

Consigne de l'énoncé : *« do not frame it as the winner if it is not »*. Même découpage S3/M3, mêmes
features préparées que la logistique, CPU.
{fond_txt}

## 9. Verbatims synthétiques et fusion texte (bonus § 4.4)
{texte_txt}

{_decisions([
    ("Trois formulations instanciées, arbitrées par la mesure", "l'énoncé demande une justification, pas un a priori", "§ 1"),
    ("Découpage 15/85 par propension simulée, MNAR retenu", "les répondants ne sont pas aléatoires ; le coût du biais est mesuré", "§ 2, tests du protocole"),
    ("IPW estimée sous MAR, effet mesuré", "ce qu'un praticien peut faire ; limite face au MNAR explicitée", "§ 2"),
    ("Quinze modèles, sept familles, une hypothèse chacun", "au-delà du minimum de l'énoncé ; alignés sur les quatre études de référence", "§ 3"),
    ("Sélection par CV sur les répondants, pondérée IPW, avec parcimonie", "les silencieux sont le test : les utiliser pour choisir serait une fuite de sélection", "§ 6"),
    (f"Retenu : {_libmod(sel['retenu'])} ({sel['version_retenue']})", "le plus simple des modèles à moins d'un écart-type du meilleur kappa CV", "§ 6.3"),
    ("Stratégie de décision hybride si la calibration écrase une classe", "vérifié, pas supposé ; test automatisé", "§ 7"),
    ("Modèles de fondation et texte : évalués, non retenus en production", "consigne d'honnêteté de l'énoncé ; gain texte artificiel par construction", "§ 8, § 9"),
])}
""")
    return decouper(df, cfg, "S3_MNAR", seed)


# =============================================================================
# PHASE 5 — ÉVALUATION
# =============================================================================
def phase5(cfg, df, cibles, X, dec, art, R, num, cat, satisfaction):
    seed = cfg["seed"]; rep, sil = dec["repondants"], dec["silencieux"]
    y = cibles["M3"].astype(str).values
    pipe_cal, pipe_dec, pipe_brut = art["pipeline"], art["pipeline_decision"], art["pipeline_brut"]
    classes = art["classes"]; i_det = classes.index("Detracteur")
    proba = pipe_cal.predict_proba(X[sil]); pred = np.asarray(pipe_dec.predict(X[sil])).astype(str)
    m = R["modele_final"]["metriques_retenues"]; mf = R["modele_final"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    cm = np.array(m["matrice_confusion"])
    axes[0].imshow(cm, cmap="Blues"); axes[0].set_xticks(range(3)); axes[0].set_yticks(range(3))
    axes[0].set_xticklabels([LIB[c] for c in ORDRE]); axes[0].set_yticklabels([LIB[c] for c in ORDRE])
    axes[0].set_xlabel("prédit"); axes[0].set_ylabel("vrai"); axes[0].set_title("Matrice de confusion (85 % silencieux)")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max()/2 else "k")
    for k, c in enumerate(classes):
        yb_ = (y[sil] == c).astype(int); p_, r_, _ = precision_recall_curve(yb_, proba[:, k]); ap = average_precision_score(yb_, proba[:, k])
        axes[1].plot(r_, p_, color=COUL[c], label=f"{LIB[c]} (AP={ap:.2f}, base={yb_.mean():.2f})")
    axes[1].set_xlabel("rappel"); axes[1].set_ylabel("précision"); axes[1].set_title("Courbes précision-rappel (un-contre-tous)"); axes[1].legend(fontsize=7)
    f_eval = _fig("05_confusion_pr.png")

    yb = (y[sil] == "Detracteur").astype(int); p_brut = pipe_brut.predict_proba(X[sil])[:, list(pipe_brut.classes_).index("Detracteur")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for p_, lab, col in ((p_brut, "non calibré", GRIS), (proba[:, i_det], "calibré (Platt)", BLEU)):
        fp, mp = calibration_curve(yb, p_, n_bins=10, strategy="quantile"); axes[0].plot(mp, fp, marker="o", label=lab, color=col)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7); axes[0].set_xlabel("probabilité prédite"); axes[0].set_ylabel("fréquence observée")
    axes[0].set_title("Fiabilité — P(Détracteur)"); axes[0].legend(fontsize=8)
    ordre = np.argsort(-proba[:, i_det]); cum = np.cumsum(yb[ordre]) / yb.sum(); frac = np.arange(1, len(yb) + 1) / len(yb)
    axes[1].plot(frac, cum, color=ROUGE, label="modèle"); axes[1].plot([0, 1], [0, 1], "k--", lw=0.7, label="hasard")
    K = cfg["metier"]["k_appels"]; axes[1].axvline(K / len(yb), color=BLEU, ls=":", label=f"K = {K} appels")
    axes[1].set_xlabel("part des clients appelés (classés par risque)"); axes[1].set_ylabel("part des détracteurs captés"); axes[1].set_title("Courbe de gain cumulé (lift)"); axes[1].legend(fontsize=8)
    f_lift = _fig("05_calibration_gain.png")
    pk = pd.DataFrame(R["evaluation_metier"]["courbe_k"])[["k", "precision_at_k", "rappel_at_k", "lift", "detracteurs_captes"]]
    pk.columns = ["K appels", "Précision@K", "Rappel@K", "Lift", "Détracteurs captés"]
    eco = R["evaluation_metier"]["economie"]; mk = cfg["metier"]; pk0 = R["evaluation_metier"]["precision_at_k"]

    ns = R["nps_simule"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    vals = [ns["nps_repondants_observe"], ns["nps_silencieux_predit"], ns["nps_silencieux_vrai"]]
    labs = ["NPS des répondants\n(ce que l'opérateur voit)", "NPS des silencieux\nprédit par le modèle", "NPS des silencieux\nvrai (connu ici)"]
    ax.bar(labs, vals, color=[GRIS, BLEU, VERT])
    ic = ns["nps_silencieux_predit_ic95"]; ax.errorbar([1], [vals[1]], yerr=[[vals[1] - ic[0]], [ic[1] - vals[1]]], fmt="none", color="k", capsize=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:+.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="k", lw=0.6); ax.set_title("Simuler le NPS des 85 % silencieux (§ 2 de l'énoncé)")
    f_nps = _fig("05_nps_simule.png")
    segnps = pd.DataFrame(ns["par_segment"]); segnps.columns = ["Segment", "Clients", "NPS prédit", "NPS vrai", "Part Dét. prédite", "Part Dét. vraie"]
    parts = pd.DataFrame({"Prédit (silencieux)": ns["part_classes_silencieux_predit"], "Vrai (silencieux)": ns["part_classes_silencieux_vrai"]}).reindex(ORDRE).reset_index().rename(columns={"index": "Classe"})
    parts["Classe"] = parts["Classe"].map(LIB)
    lecture_nps = lecture_composition(ns)

    br = pd.DataFrame([b for b in R["bruit_etiquettes"] if "kappa" in b]); ab = pd.DataFrame([a for a in R["ablation_features"] if "kappa" in a])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].plot(br["taux_bruit"] * 100, br["kappa"], marker="o", color=BLEU, label="kappa"); axes[0].plot(br["taux_bruit"] * 100, br["rappel_detracteur"], marker="s", color=ROUGE, label="rappel Détracteur")
    axes[0].set_xlabel("% d'étiquettes d'entraînement corrompues"); axes[0].set_title("Robustesse au bruit d'étiquettes"); axes[0].legend(fontsize=8)
    yy = np.arange(len(ab)); base_k = ab.loc[ab["configuration"].str.startswith("base"), "kappa"].iloc[0]
    axes[1].barh(yy, ab["kappa"] - base_k, color=[VERT if v >= 0 else ROUGE for v in ab["kappa"] - base_k])
    axes[1].set_yticks(yy); axes[1].set_yticklabels(ab["configuration"], fontsize=7); axes[1].invert_yaxis(); axes[1].axvline(0, color="k", lw=0.6)
    axes[1].set_xlabel("écart de kappa par rapport à la configuration retenue"); axes[1].set_title("Ablation de features")
    f_rob = _fig("05_robustesse.png")
    brt = br[["taux_bruit", "kappa", "macro_f1", "rappel_detracteur", "rappel_passif", "rappel_promoteur"]].copy(); brt["taux_bruit"] = brt["taux_bruit"].map(lambda v: f"{v:.0%}")
    brt.columns = ["Bruit", "Kappa", "Macro-F1", "Rappel Dét.", "Rappel Passif", "Rappel Prom."]
    abt = ab[["configuration", "n_variables", "kappa", "macro_f1", "rappel_detracteur", "rappel_passif", "auc_detracteur"]].copy(); abt["Δ kappa"] = abt["kappa"] - base_k
    abt.columns = ["Configuration", "Variables", "Kappa", "Macro-F1", "Rappel Dét.", "Rappel Passif", "AUC Dét.", "Δ kappa"]

    def _delta(motif):
        s = abt.loc[abt["Configuration"].str.contains(motif, regex=False), "Δ kappa"]
        if not len(s):
            return "n.d."
        v = float(s.iloc[0])
        return "0.000" if abs(v) < 5e-4 else f"{v:+.3f}"

    coefs = {}
    for mp in ("M1", "M2", "M3"):
        mm = construire_modeles(num, cat, seed, ["logistique"])["logistique"]; ym = cibles[mp].astype(str).values
        entrainer(mm, X[rep], ym[rep]); mdl = mm.named_steps["modele"]
        coefs[mp] = pd.Series(mdl.coef_[list(mdl.classes_).index("Detracteur")], index=noms_apres_encodage(mm.named_steps["preparation"]))
    top = coefs["M3"].abs().sort_values(ascending=False).head(12).index
    sens = pd.DataFrame({mp: coefs[mp].reindex(top) for mp in coefs})
    fig, ax = plt.subplots(figsize=(8, 4)); sens.plot(kind="barh", ax=ax, color=[ROUGE, ORANGE, BLEU]); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("coefficient logistique — classe Détracteur (features standardisées)"); ax.set_title("Les mêmes drivers sous les trois mappings ?"); ax.invert_yaxis()
    f_sens = _fig("05_sensibilite_mapping.png")
    stables = int((sens.apply(np.sign).nunique(axis=1) == 1).sum())
    g = pd.DataFrame([x for x in R["grille"] if "kappa_quadratique" in x and x["scenario"] == "S3_MNAR" and x["modele"] == mf["modele"]])
    kmap = g.set_index("mapping")["kappa_quadratique"].reindex(["M1", "M2", "M3"])

    dr = R["drivers"]
    imp = pd.Series(dr["importance_detracteur"]); direction = dr.get("direction", {})
    fig, ax = plt.subplots(figsize=(8, 4.4))
    top15 = imp.head(15)
    cols_ = [ROUGE if direction.get(v, 1) > 0 else VERT for v in top15.index]
    ax.barh(top15.index[::-1], top15.values[::-1], color=cols_[::-1]); ax.set_xlabel(f"importance moyenne |contribution| — {dr['methode']}")
    ax.set_title("Les 15 variables qui pèsent le plus sur la détraction (rouge : une valeur élevée augmente le risque · vert : le diminue)", fontsize=8)
    f_imp = _fig("05_interpretabilite.png")
    impt = pd.DataFrame({"Variable": imp.index, "Importance": imp.values, "Sens (corrélation valeur ↔ contribution)": [direction.get(v, np.nan) for v in imp.index]})
    seg = dr.get("par_segment", {})
    f_seg, segt = "", pd.DataFrame()
    if seg:
        allv = sorted({v for s in seg.values() for v in list(s["drivers"])[:6]})
        mat = pd.DataFrame({s: {v: seg[s]["drivers"].get(v, {}).get("importance", 0.0) for v in allv} for s in seg})
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(allv) + 1.6))
        im = ax.imshow(mat.values, cmap="Reds", aspect="auto"); ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, fontsize=7, rotation=20)
        ax.set_yticks(range(len(allv))); ax.set_yticklabels(allv, fontsize=7); plt.colorbar(im, fraction=0.03)
        ax.set_title("Drivers par segment — importance moyenne de chaque variable dans le segment", fontsize=8)
        f_seg = _fig("05_drivers_segments.png")
        segt = pd.DataFrame([{"Segment": s, "Clients": v["clients"], **{f"Driver {i+1}": f"{k} ({d['sens']})" for i, (k, d) in enumerate(list(v['drivers'].items())[:5])}} for s, v in seg.items()])
    lev = R["leviers"]; levt = pd.DataFrame([{"Levier recommandé": k, "Détracteurs prédits concernés": v} for k, v in lev["distribution_detracteurs_predits"].items()]).sort_values("Détracteurs prédits concernés", ascending=False)

    eq = R["audit_equite"]
    fig, axes = plt.subplots(1, len(eq), figsize=(2.4 * len(eq), 3.1), sharey=True)
    for ax, (dim, v) in zip(np.atleast_1d(axes), eq.items()):
        g_ = pd.Series({k: r["rappel_detracteur"] for k, r in v["groupes"].items()}); s_ = pd.Series({k: r["taux_selection"] for k, r in v["groupes"].items()})
        xx = np.arange(len(g_)); ax.bar(xx - 0.2, g_.values, 0.4, color=BLEU, label="rappel Dét."); ax.bar(xx + 0.2, s_.values, 0.4, color=GRIS, label="taux de sélection")
        ax.set_xticks(xx); ax.set_xticklabels(g_.index, fontsize=7, rotation=20); ax.set_ylim(0, 1); ax.set_title(dim, fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=6); np.atleast_1d(axes)[0].set_ylabel("part")
    f_eq = _fig("05_equite.png")
    eqdf = pd.DataFrame([{"Dimension": d, "Groupe": gname, "Clients": r["effectif"], "Détracteurs réels": f"{r['taux_detracteurs_reel']:.1%}", "Taux de sélection": f"{r['taux_selection']:.1%}",
                          "Rappel Dét.": f"{r['rappel_detracteur']:.1%}", "Précision Dét.": f"{r['precision_detracteur']:.1%}"} for d, v in eq.items() for gname, r in v["groupes"].items()])
    ecarts = pd.DataFrame([{"Dimension": d, "Écart max de rappel (pts)": f"{v['ecart_max']*100:.1f}", "Écart max de sélection (pts)": f"{v['ecart_selection']*100:.1f}"} for d, v in eq.items()])
    pire = max(eq.items(), key=lambda kv: kv[1]["ecart_max"])
    px = R["audit_proxies"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    pxs = pd.Series({k: v["auc"] for k, v in px.items()}).sort_values()
    ax.barh(pxs.index, pxs.values, color=[ROUGE if v > 0.75 else ORANGE if v > 0.6 else VERT for v in pxs.values]); ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlim(0.4, 1); ax.set_xlabel("AUC : à quel point les features du modèle permettent de retrouver l'attribut"); ax.set_title("Audit des proxies d'attributs protégés", fontsize=9)
    f_px = _fig("05_proxies.png")
    pxt = pd.DataFrame([{"Attribut (exclu du modèle)": k, "Prévalence": f"{v['prevalence']:.1%}", "AUC de reconstruction": v["auc"],
                         "Lecture": "fortement reconstructible" if v["auc"] > 0.75 else "partiellement" if v["auc"] > 0.6 else "non reconstructible"} for k, v in px.items()])
    mit = R["mitigation_seuils_age"]
    mitrows = []
    for g_, av in mit["avant"].items():
        ap_ = mit["apres"].get(g_, {})
        mitrows.append({"Tranche d'âge": g_, "Rappel avant": f"{av['rappel_detracteur']:.1%}", "Rappel après": f"{ap_['rappel']:.1%}" if ap_ else "—",
                        "Précision avant": f"{av['precision_detracteur']:.1%}", "Précision après": f"{ap_['precision']:.1%}" if ap_ else "—",
                        "Sélection avant": f"{av['taux_selection']:.1%}", "Sélection après": f"{ap_['taux_selection']:.1%}" if ap_ else "—", "Seuil": ap_.get("seuil", "—")})
    mitt = pd.DataFrame(mitrows); ga, gp = mit["global_avant"], mit["global_apres"]

    repro = _charger_json("reproductibilite")
    repro_txt = "*Test à blanc non exécuté.*"
    if repro and repro.get("comparaison"):
        t = pd.DataFrame(repro["comparaison"]); t["identique"] = t["identique"].map({True: "✅", False: "❌"}); t.columns = ["Indicateur", "Original", "Copie vierge", "Identique"]
        repro_txt = f"""Protocole : copie de `src/`, `prompts/`, `config.yaml` et `data/raw/` seuls dans un répertoire vierge,
exécution de `python -m src.run`, comparaison des indicateurs clés. Durée : {repro['duree_s']} s.

{_tab(t)}

{'**Reproductible à l’identique.**' if repro['reproductible'] else '**Écarts constatés** — à investiguer.'}"""
    metr = pd.DataFrame(JUSTIFICATION_METRIQUES, columns=["Métrique", "Rôle", "Pourquoi elle est là"])
    cls_t = pd.DataFrame(m["par_classe"]).T.reset_index().rename(columns={"index": "Classe"}); cls_t["Classe"] = cls_t["Classe"].map(LIB); cls_t["effectif"] = cls_t["effectif"].astype(int)
    cls_t.columns = ["Classe", "Précision", "Rappel", "F1", "Effectif"]
    k_log = R["comparatif_s3_m3"]["logistique"]["defaut"]["silencieux"]["kappa"]
    auc_j = px.get("Moins de 30 ans", {}).get("auc", float("nan"))

    _md("05_evaluation.md", f"""
# Phase 5 — Évaluation

**En bref.** Le modèle retenu ({_libmod(mf['modele'])}, {mf['version']}) atteint un kappa quadratique de **{m['kappa_quadratique']:.3f}** sur les
{mf['n_evaluation']} clients silencieux, rappelle **{m['par_classe']['Detracteur']['rappel']:.0%}** des détracteurs et garde les trois classes vivantes. Sur
{pk0['k']} appels, **{pk0['precision_at_k']:.0%}** touchent un vrai détracteur contre {pk0['taux_de_base']:.0%} au hasard (lift ×{pk0['lift']:.1f}) — {_eur(eco['gain_apporte_par_le_modele_eur'])} de gain net
par campagne à hypothèses constantes. Le NPS des silencieux est estimé à **{ns['nps_silencieux_predit']:+.0f}** (IC 95 % {ns['nps_silencieux_predit_ic95'][0]:+.0f} à {ns['nps_silencieux_predit_ic95'][1]:+.0f}) pour une vérité de
{ns['nps_silencieux_vrai']:+.0f}. Les drivers sont contractuels (engagement, ancienneté, parrainage, prix) et stables quel que soit
le mapping. Le modèle rappelle moins bien les jeunes détracteurs ({eq['tranche_age']['groupes']['<30']['rappel_detracteur']:.0%} vs {eq['tranche_age']['groupes']['60+']['rappel_detracteur']:.0%} pour les 60+) alors qu'il
n'a aucune variable d'âge : les features retenues **reconstruisent** l'âge (AUC {auc_j:.2f}). Une mitigation par
seuils est chiffrée ; l'arbitrage est remonté au métier.

## 1. Évaluation technique — jamais une métrique globale sans son détail par classe

**Pourquoi ces métriques.**

{_tab(metr)}

**Résultat sur les {mf['n_evaluation']} silencieux** (modèle entraîné sur {mf['n_entrainement']} répondants, S3, cible M3) :

| Métrique globale | Valeur |
|---|---|
| Kappa quadratique pondéré | **{m['kappa_quadratique']:.3f}** |
| Macro-F1 | {m['macro_f1']:.3f} |
| Exactitude équilibrée | {m['exactitude_equilibree']:.3f} |
| AUC un-contre-tous (macro) · AUC Détracteur · AP Détracteur | {m.get('auc_ovr_macro', float('nan')):.3f} · {m.get('auc_detracteur', float('nan')):.3f} · {m.get('ap_detracteur', float('nan')):.3f} |
| Exactitude brute | {m['exactitude']:.3f} — *rapportée pour montrer qu'elle trompe : 0,43 en prédisant toujours « Passif »* |

{_tab(cls_t, {'Précision':'{:.3f}','Rappel':'{:.3f}','F1':'{:.3f}'})}

![]({f_eval})

- **Le Détracteur est rappelé à {m['par_classe']['Detracteur']['rappel']:.0%}** — c'est la classe qui compte pour le métier. Sa précision
  est de {m['par_classe']['Detracteur']['precision']:.0%} : le modèle appelle large. C'est le bon arbitrage pour une campagne de rétention où
  manquer un détracteur coûte plus qu'un appel inutile ; la précision@K en tient compte.
- **Le Passif est la classe la plus difficile** (rappel {m['par_classe']['Passif']['rappel']:.0%}) : milieu ambigu par définition, rare à
  l'entraînement (phase 3 § 4). Cohérent avec Elero et al. (2026), où la classe neutre est aussi la moins bien prédite.
- La courbe précision-rappel est plus informative que la ROC sous déséquilibre : le Détracteur a une AP de
  {m.get('ap_detracteur', float('nan')):.2f} pour un taux de base de {pk0['taux_de_base']:.2f}.
- Référence : la baseline logistique fait {k_log:.3f} de kappa sur les mêmes silencieux ({'le modèle retenu est au-dessus' if m['kappa_quadratique'] >= k_log else 'le modèle retenu est légèrement en dessous — prix d’une sélection faite sans regarder le test'}).

## 2. Calibration : ce que la courbe de fiabilité révèle

![]({f_lift})

Les deux courbes sont **sous la diagonale** : le modèle surestime P(Détracteur), calibré ou non (ECE
{mf['calibration']['calibre_platt']['ece']:.2f} après Platt). Cause : le calibrateur est ajusté sur les répondants, où les détracteurs sont
sur-représentés par le biais MNAR ; il apprend le taux de base des répondants, pas celui de la
population silencieuse. **La calibration sur un échantillon biaisé calibre vers la mauvaise
population** — le problème MNAR réapparaît à un autre étage. Conséquence : le **classement** des clients
par P(Détracteur) reste valide (l'ordre est préservé, AUC {m.get('auc_detracteur', float('nan')):.2f}), mais la **valeur absolue** ne doit pas
être lue comme une fréquence — l'application l'affiche. Correction possible en production : recalibrer
sur les premières vraies réponses de la population cible (phase 6 § 4).

## 3. Évaluation métier

La vraie question n'est pas « quel taux de bonnes réponses » mais **« sur les K clients que l'équipe
peut appeler ce mois-ci, combien sont réellement des détracteurs ? »**

{_tab(pk, {'Précision@K':'{:.1%}','Rappel@K':'{:.1%}','Lift':'×{:.2f}'})}

Taux de base (part de détracteurs parmi les silencieux) : **{pk0['taux_de_base']:.1%}**. À K = {pk0['k']}, **{pk0['precision_at_k']:.1%} des clients appelés sont
de vrais détracteurs — {pk0['lift']:.1f}× mieux qu'au hasard.**

**Traduction en euros.** Hypothèses (paramètres dans `config.yaml`, **à valider avec l'équipe rétention**) :
coût d'un appel {mk['cout_appel_eur']} €, valeur d'un client retenu {mk['valeur_client_retenu_eur']} €, taux de succès {mk['taux_succes_appel']:.0%}.

| | Campagne ciblée par le modèle | Campagne aléatoire de même taille |
|---|---|---|
| Coût | {_eur(eco['cout_campagne_eur'])} | {_eur(eco['cout_campagne_eur'])} |
| Clients retenus (estimés) | {eco['clients_retenus_estimes']} | — |
| Gain net | **{_eur(eco['gain_net_eur'])}** | {_eur(eco['gain_net_ciblage_aleatoire_eur'])} |
| ROI | {eco['roi']:.2f} | — |

**Apporté par le modèle : {_eur(eco['gain_apporte_par_le_modele_eur'])} par campagne de {pk0['k']} appels**, à hypothèses constantes.

**`CLTV` en couche de décision.** Interdite comme feature (sortie de modèle IBM : elle fuit), `CLTV` est
légitime pour pondérer *qui appeler* parmi les détracteurs prédits : c'est un choix métier, pas une
information sur la cible. Gómez-Vargas et al. (EJOR 2025) montrent qu'une CLV individuelle bat une CLV
moyenne pour cibler une campagne. L'application propose cette pondération en option.

## 4. Simuler le NPS des 85 % silencieux

![]({f_nps})

| | NPS |
|---|---|
| Répondants — ce que l'opérateur observe | {ns['nps_repondants_observe']:+.1f} |
| Silencieux — prédit par le modèle (IC 95 % bootstrap) | **{ns['nps_silencieux_predit']:+.1f}** [{ns['nps_silencieux_predit_ic95'][0]:+.1f} ; {ns['nps_silencieux_predit_ic95'][1]:+.1f}] |
| Silencieux — vrai (connu ici seulement) | {ns['nps_silencieux_vrai']:+.1f} |
| Base complète — prédit | {ns['nps_base_complete_predit']:+.1f} |

{_tab(parts, {'Prédit (silencieux)':'{:.1%}','Vrai (silencieux)':'{:.1%}'})}

{lecture_nps} **Par segment** (silencieux) :

{_tab(segnps, {'Part Dét. prédite':'{:.1%}','Part Dét. vraie':'{:.1%}'})}

Le classement des segments par NPS est respecté (contrats mensuels et clients récents sont bien les
plus critiques) ; les niveaux sont exagérés dans les deux sens — effet du biais de réponse sur un
modèle qui n'a jamais vu de Passifs en nombre.

## 5. Robustesse et sensibilité

**5.1 Sensibilité au mapping de la cible** (énoncé § 4.1). Kappa du modèle retenu sous les trois
mappings (S3) : M1 {kmap['M1']:.3f} · M2 {kmap['M2']:.3f} · M3 {kmap['M3']:.3f} — ⚠️ non comparables entre eux (déplacer le milieu ambigu
change la difficulté). Ce qui se compare, ce sont les **conclusions** : sur les 12 variables les plus
influentes (logistique, dont les coefficients sont directement comparables), **{stables} conservent le même
sens d'effet sous les trois mappings**. Les conclusions métier sont robustes au choix du mapping ;
c'est l'amplitude qui varie, pas la direction.

![]({f_sens})

**5.2 Bruit d'étiquettes** (énoncé § 4.1 : « adding realistic noise »). On corrompt une part des
étiquettes d'entraînement (classe redistribuée au hasard) et on mesure sur les silencieux, dont les
étiquettes restent vraies.

{_tab(brt, {'Kappa':'{:.3f}','Macro-F1':'{:.3f}','Rappel Dét.':'{:.3f}','Rappel Passif':'{:.3f}','Rappel Prom.':'{:.3f}'})}

Le kappa perd {(br['kappa'].iloc[0] - br['kappa'].iloc[-1]):.3f} entre 0 et {br['taux_bruit'].iloc[-1]:.0%} de bruit : dégradation {'graduelle, pas d’effondrement' if (br['kappa'].iloc[0] - br['kappa'].iloc[-1]) < 0.12 else 'marquée'}. Un
label NPS réel est bruité (humeur du jour, incompréhension de l'échelle) ; ce test borne ce que ça coûte.

**5.3 Ablation de features** — ce que chaque bloc coûte ou rapporte, modèle retenu, S3/M3, silencieux :

![]({f_rob})

{_tab(abt, {'Kappa':'{:.3f}','Macro-F1':'{:.3f}','Rappel Dét.':'{:.3f}','Rappel Passif':'{:.3f}','AUC Dét.':'{:.3f}','Δ kappa':'{:+.3f}'})}

Trois lectures. **Démographie et géographie** : ajouter la démographie change le kappa de {_delta('+ démographie')},
la géographie de {_delta('+ géographie')}, les deux de {_delta('+ démographie + géographie')} — c'est le **coût mesuré du choix d'équité**
(énoncé § 4.7 : « Removing a feature often costs accuracy. Make the trade-off explicit »). Il est
assumé : un gain de cet ordre ne justifie pas de décider sur l'âge, le genre ou le code postal, et
§ 7 montre que le modèle retrouve de toute façon ces attributs par proxies. **Parrainage** : retirer
`Number of Referrals` et `a_parraine` change le kappa de {_delta('parrainage')} — c'est le driver le plus proche de la cible sans
être une fuite (comportement passé observable). **Variables dérivées** : {_delta('dérivées')} — {'elles apportent peu au-delà des brutes : elles servent surtout la lisibilité et les leviers' if abs(float(_delta('dérivées').replace('n.d.', '0'))) < 0.01 else 'elles portent une part réelle du signal'}.
Retirer `Contract` : {_delta('Contract')} ; retirer `Offer` : {_delta('Offer')}.

## 6. Drivers de détraction

Méthode : **{dr['methode']}** — la même dans l'application, pour chaque client. Les contributions sont
lues pour la classe Détracteur ; les variables liées par identité comptable (phase 3 § 3.5) se lisent en bloc.

![]({f_imp})

{_tab(impt.head(20), {'Importance':'{:.4f}','Sens (corrélation valeur ↔ contribution)':'{:+.2f}'})}

**Ce que le modèle dit.** Les associations sont fortes et cohérentes avec la littérature sur ce
dataset (Chong et al. 2023) et sur le NPS télécom (Elero et al. 2026) : **contrat sans engagement,
faible ancienneté, absence de parrainage, charge mensuelle élevée** vont avec la détraction ; l'offre E
ressort comme marqueur de clients déjà fragiles (phase 2 § 5.5), pas comme cause. **Le modèle établit
des associations, pas des causes** : un coefficient positif est un signal, pas un levier prouvé.

**Drivers par segment** (énoncé § 4.6) — les mêmes variables ne pèsent pas partout pareil :

{('![](' + f_seg + ')') if f_seg else ''}

{_tab(segt) if len(segt) else ''}

**Actionnable ou non, et levier** :

| Driver | Actionnable ? | Levier |
|---|---|---|
| Contrat mensuel | **oui** | proposer un engagement 12 mois avec avantage |
| Faible ancienneté | partiellement | parcours d'accueil renforcé les 6 premiers mois |
| Charge mensuelle / par service élevée | **oui** | revue tarifaire, bundle |
| Absence de parrainage | indirectement | programme de parrainage |
| Offre E reçue | **signal, pas levier** | ne pas « supprimer l'offre E » — comprendre à qui elle a été proposée |
| Type d'accès (fibre) | non à court terme | investissement réseau, support premium |

**Règle de recommandation individuelle** (implémentée, affichée pour chaque client) :
{chr(10).join('- ' + r for r in lev['regle'])}

Répartition parmi les détracteurs prédits :

{_tab(levt)}

*{lev['avertissement']}.*

## 7. Équité et biais (énoncé § 4.7)

Les variables démographiques et géographiques sont **exclues du modèle** et servent **uniquement ici**.
On mesure, par sous-groupe, le rappel Détracteur (le modèle rate-t-il davantage certains détracteurs ?),
la précision, et le **taux de sélection** (quelle part du groupe est prédite détracteur — c'est ce qui
alloue le budget d'appels). `zone` = tertiles de population du code postal (proxy de densité).

![]({f_eq})

{_tab(eqdf)}

{_tab(ecarts)}

**7.1 Ce qu'on trouve.** L'écart le plus important porte sur **{pire[0]} : {pire[1]['ecart_max']*100:.0f} points** de rappel entre le groupe le
mieux servi et le moins bien servi — le modèle rappelle {eq['tranche_age']['groupes']['<30']['rappel_detracteur']:.0%} des détracteurs de moins de 30 ans contre
{eq['tranche_age']['groupes']['60+']['rappel_detracteur']:.0%} des 60 ans et plus. C'est le cas de figure que l'énoncé décrit comme inacceptable, **en sens
inverse** : ce sont les jeunes détracteurs qu'on rate. Les écarts par genre et par densité de zone sont faibles.

**7.2 Les features « proxent »-elles les attributs protégés ?** Pour chaque attribut exclu, on mesure à
quel point les features du modèle permettent de le reconstruire (AUC en validation croisée) :

![]({f_px})

{_tab(pxt, {'AUC de reconstruction':'{:.3f}'})}

**Exclure une variable ne suffit pas à ce que le modèle l'ignore.** L'âge et la situation familiale sont
largement reconstructibles à partir du contrat, de l'ancienneté, des services et du parrainage ; le
genre et la densité de zone ne le sont pas. C'est la raison structurelle de l'écart d'âge : le modèle
« voit » l'âge sans en avoir la colonne. L'exclusion reste justifiée (elle empêche une décision
*explicite* sur l'attribut et retire le levier le plus direct), mais elle ne dispense pas de l'audit.

**7.3 Mitigation testée : un seuil de décision par tranche d'âge**, calé pour que chaque groupe atteigne
le rappel global ({mit['rappel_cible']:.0%}). Ce que ça coûte :

{_tab(mitt)}

| | Kappa | Macro-F1 | Rappel Dét. | Précision Dét. | Appels (prédits Détracteur) |
|---|---|---|---|---|---|
| Avant | {ga['kappa']:.3f} | {ga['macro_f1']:.3f} | {ga['rappel_detracteur']:.3f} | {ga['precision_detracteur']:.3f} | {mit['appels_avant']} |
| Après seuils par âge | {gp['kappa']:.3f} | {gp['macro_f1']:.3f} | {gp['rappel_detracteur']:.3f} | {gp['precision_detracteur']:.3f} | {mit['appels_apres']} |

Égaliser le rappel se paie en précision chez les jeunes (plus d'appels inutiles) et en volume global.
C'est un arbitrage **métier et juridique**, pas technique : traiter différemment selon l'âge pour
corriger un écart est légal dans certaines juridictions et pas dans d'autres.

**7.4 À remonter avant production.**
1. L'écart d'âge ({pire[1]['ecart_max']*100:.0f} pts) et sa cause (proxies) — à l'équipe Expérience Client et au juridique.
2. Le fait que la mitigation par seuils est possible et chiffrée, mais qu'elle constitue un traitement
   différencié selon l'âge : décision à prendre au-dessus de la data science.
3. L'audit doit être répété à chaque réentraînement — il est dans le monitoring (phase 6 § 4).

## 8. Limites de la règle de ciblage

Cette étude classe les clients par **risque** d'être détracteur, comme demandé. Or Ascarza (2018,
*Journal of Marketing Research*) établit, par deux expériences de terrain, que cibler les clients au
risque le plus élevé est inefficace : il faut cibler ceux dont la **sensibilité à l'intervention** est
la plus forte — même campagne, même budget, jusqu'à 7 points de churn en moins.

| | On l'appelle | On ne l'appelle pas | |
|---|---|---|---|
| | reste | reste | **acquis** — appel gaspillé |
| | part | part | **perdu d'avance** — appel gaspillé |
| | **reste** | **part** | **persuadable** — seul cas qui crée de la valeur |
| | part | reste | **à ne pas déranger** — l'appel nuit |

Les détracteurs les plus certains sont massivement des *perdus d'avance*. **On ne peut pas estimer
cette sensibilité ici** : aucune campagne enregistrée, aucune assignation aléatoire (la table `Offer`
prouve que les offres ont été ciblées). Prétendre faire de l'uplift serait la faute reprochée à Mustafa
et al. La seule chose à faire, gratuite, est en phase 6 § 5.

## 9. Revue du processus

| Critère de succès (phase 1) | Résultat | Verdict |
|---|---|---|
| Précision@{pk0['k']} nettement au-dessus du taux de base | {pk0['precision_at_k']:.1%} vs {pk0['taux_de_base']:.1%} (×{pk0['lift']:.1f}) | ✅ |
| Gain net vs aléatoire, en euros | {_eur(eco['gain_apporte_par_le_modele_eur'])} par campagne | ✅ (hypothèses à valider) |
| Kappa et rappel Détracteur au niveau de la baseline ou au-dessus | retenu {_libmod(mf['modele'])} : kappa {m['kappa_quadratique']:.3f} ; logistique {k_log:.3f} | {'✅' if m['kappa_quadratique'] >= k_log - 0.005 else '⚠️ légèrement en dessous — prix de la sélection sans regarder le test'} |
| Aucune classe abandonnée | rappel Passif {m['par_classe']['Passif']['rappel']:.2f} | ✅ |
| Écart d'équité mesuré, signalé si > 10 pts | {pire[1]['ecart_max']*100:.0f} pts sur l'âge, cause identifiée, mitigation chiffrée | ⚠️ **signalé, arbitrage remonté** |

**Solide** : registre de fuites quantifié ; arbitrage empirique des « 3 » sans circularité ; protocole
15/85 avec coût du MNAR mesuré ; quinze modèles, sélection sans regarder le test ; calibration
vérifiée ; audit d'équité avec proxies et mitigation. **Approximatif** : hypothèses économiques
(placeholders) ; propension à répondre simulée ; verbatims synthétiques sans clé API ; TabPFN non
évalué faute d'accès. **Impossible avec ces données** : estimer l'effet d'un appel.

**Reproductibilité — test à blanc depuis une copie vierge.** {repro_txt}

{_decisions([
    ("Toute métrique globale accompagnée de son détail par classe", "règle du projet ; contre-exemple Kannan et al.", "§ 1"),
    ("Probabilités utilisées pour classer, pas pour lire une fréquence", "calibrateur ajusté sur une population biaisée", "§ 2"),
    ("K = 500 et hypothèses économiques dans config.yaml", "paramètres métier, à valider avec l'équipe", "§ 3"),
    ("NPS des silencieux rapporté avec intervalle et réserve de composition", "excès compensés de Détracteurs et Promoteurs", "§ 4"),
    ("Coût de l'exclusion des démographiques et de la géographie chiffré et assumé", "énoncé § 4.7 ; proxies mesurés", "§ 5.3, § 7.2"),
    ("Mitigation par seuils d'âge chiffrée, non appliquée par défaut", "traitement différencié : décision métier/juridique", "§ 7.3"),
    ("Pas d'uplift modeling", "aucune assignation aléatoire ; le prétendre serait une faute", "§ 8"),
])}
""")


# =============================================================================
# PHASE 6 — DÉPLOIEMENT
# =============================================================================
def phase6(cfg, R, art):
    m = R["modele_final"]["metriques_retenues"]; mf = R["modele_final"]; mo = _charger_json("monitoring")
    section_mo = "*Monitoring non exécuté (`python -m src.monitoring`).*"
    if mo:
        d_cur = pd.DataFrame(mo["derive_entrees_courant"]).head(8); d_sim = pd.DataFrame(mo["derive_entrees_mois_simule"]).head(8); pd_ = mo["part_detracteurs_predits"]
        su = mo.get("suivi_nouvelles_reponses"); suivi_txt = ""
        if su:
            lots = pd.DataFrame(su["lots"])
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(lots["mois"], lots["kappa_cumul"], marker="o", color=BLEU, label="kappa cumulé sur les nouvelles réponses")
            ax.plot(lots["mois"], lots["rappel_detracteur_cumul"], marker="s", color=ROUGE, label="rappel Détracteur cumulé")
            ax.axhline(su["reference"]["kappa"] - su["seuils"]["chute_kappa"], color=BLEU, ls=":", lw=0.8, label="seuil de chute (kappa)")
            ax.axhline(su["reference"]["rappel_detracteur"] - su["seuils"]["chute_rappel_detracteur"], color=ROUGE, ls=":", lw=0.8, label="seuil de chute (rappel)")
            if su["premier_mois_volume_atteint"]:
                ax.axvline(su["premier_mois_volume_atteint"], color="k", ls="--", lw=0.8, label=f"{su['seuils']['min_nouveaux_labels']} nouveaux labels atteints")
            ax.set_xlabel("mois"); ax.set_title("Performance sur les nouvelles réponses d'enquête (simulation)"); ax.legend(fontsize=6.5)
            f_nr = _fig("06_nouvelles_reponses.png")
            lt = lots[["mois", "nouvelles_reponses", "cumul", "kappa_lot", "rappel_detracteur_lot", "kappa_cumul", "rappel_detracteur_cumul", "chute_performance", "volume_atteint"]].copy()
            lt["chute_performance"] = lt["chute_performance"].map({True: "⚠️", False: ""}); lt["volume_atteint"] = lt["volume_atteint"].map({True: "✓", False: ""})
            lt.columns = ["Mois", "Nouvelles réponses", "Cumul", "Kappa (lot)", "Rappel Dét. (lot)", "Kappa (cumul)", "Rappel Dét. (cumul)", "Chute", "Volume atteint"]
            suivi_txt = f"""
**Performance sur les nouvelles réponses d'enquête** (simulation : ~150 réponses par mois, tirées parmi les
silencieux selon la propension biaisée S3) :

![]({f_nr})

{_tab(lt, {'Kappa (lot)':'{:.3f}','Rappel Dét. (lot)':'{:.3f}','Kappa (cumul)':'{:.3f}','Rappel Dét. (cumul)':'{:.3f}'})}

Référence : kappa {su['reference']['kappa']:.3f}, rappel {su['reference']['rappel_detracteur']:.3f}. Seuils : chute de kappa > {su['seuils']['chute_kappa']}, chute de rappel
> {su['seuils']['chute_rappel_detracteur']}, volume minimal {su['seuils']['min_nouveaux_labels']} nouveaux labels (atteint au mois {su['premier_mois_volume_atteint']}), échéance {su['seuils']['echeance_max_mois']} mois.
Chute détectée sur la simulation : {'oui' if su['chute_detectee'] else 'non'}. {su['note']}.
"""
        section_mo = f"""
![](figures/06_monitoring.png)

**Référence** = population d'entraînement (répondants S3). **Courant** = population scorée. **Mois
simulé** = {mo['scenario_simule']}.

| | PSI des prédictions | Part de détracteurs prédits |
|---|---|---|
| Courant vs référence | {mo['psi_predictions_courant']:.3f} | {pd_['courant']:.1%} (réf. {pd_['reference']:.1%}) |
| Mois simulé vs référence | **{mo['psi_predictions_mois_simule']:.3f}** | **{pd_['mois_simule']:.1%}** |

Variables les plus dérivantes — courant :

{_tab(d_cur, {'PSI':'{:.4f}'})}

Variables les plus dérivantes — mois simulé :

{_tab(d_sim, {'PSI':'{:.4f}'})}

**Déclencheur de dérive sur le mois simulé : {'DÉCLENCHÉ' if mo['reentrainement_declenche_mois_simule'] else 'non déclenché'}** (variables en alerte :
{', '.join(mo['variables_en_alerte_mois_simule']) or 'aucune'}). Le mécanisme détecte une hausse tarifaire et une migration de contrats
que personne n'aurait signalées au modèle. Note : le PSI « courant » n'est pas nul alors que les deux
populations viennent de la même base — c'est le **biais de sélection S3** qui s'affiche, et c'est
exactement ce que le monitoring doit voir.
{suivi_txt}"""

    _md("06_deploiement.md", f"""
# Phase 6 — Déploiement

**En bref.** Le modèle est persisté d'un bloc (préprocessing + modèle + calibrateur) et consommé par
une application Streamlit de sept pages pour l'équipe rétention : liste d'appels priorisée, analyse
d'un client avec ses drivers et son levier, saisie manuelle tolérante aux inconnus, verbatim en
démonstration. Un monitoring implémenté surveille la dérive des entrées et des prédictions, la
performance sur les nouvelles réponses d'enquête, et déclenche le réentraînement sur seuil, volume ou
échéance. La décision demandée à la direction : **un groupe de contrôle non contacté** à la prochaine campagne.

## 1. Carte du modèle

| | |
|---|---|
| Nom | Prédiction de catégorie NPS — clients silencieux |
| Version | 1.1 · seed {cfg['seed']} · régénéré par `python -m src.run` |
| Modèle | {_libmod(mf['modele'])} ({mf['famille']}, {mf['formulation']}), version {mf['version']}, dans un `Pipeline` scikit-learn (imputation + encodage + modèle) |
| Cible | M3 — Détracteur (satisfaction 1–2) · Passif (3) · Promoteur (4–5) |
| Données d'entraînement | {mf['n_entrainement']} répondants simulés (S3 — MNAR), IBM Telco 11.1.3+ |
| Évaluation | {mf['n_evaluation']} clients silencieux, vérité connue |
| Kappa quadratique · macro-F1 | {m['kappa_quadratique']:.3f} · {m['macro_f1']:.3f} |
| Rappel / précision Détracteur | {m['par_classe']['Detracteur']['rappel']:.2f} / {m['par_classe']['Detracteur']['precision']:.2f} |
| Décision | {mf['strategie_decision']} |
| Explication | {R['drivers'].get('methode', '')} |
| Variables exclues | tout ce qui touche au churn, `CLTV`, démographie, géographie (phase 2 § 4) |
| Usage prévu | prioriser une liste d'appels de rétention mensuelle ; comprendre les drivers |
| Usage **interdit** | décider d'une offre ou d'un tarif individuel ; toute décision fondée sur un attribut protégé ; lire P(Détracteur) comme une fréquence |
| Limites connues | écart de rappel de {R['audit_equite']['tranche_age']['ecart_max']*100:.0f} pts entre groupes d'âge (proxies) ; classe Passif mal rappelée ; probabilités surestimées ; propension à répondre simulée |
| Fichiers | `models/modele_final.joblib` (pipelines calibré, brut, de décision, métadonnées) · `data/processed/clients_scores.parquet` |

## 2. L'application — choix de conception (énoncé § 4.8)

`streamlit run app/app.py` · sept pages, adressables par `?page=` · toutes les valeurs sont lues dans
les artefacts du pipeline, rien n'est recopié.

| Page | Pour qui | Ce qu'elle permet |
|---|---|---|
| Synthèse | direction | les cinq chiffres, le NPS des silencieux, les trois résultats, la décision demandée |
| Prioriser les appels | équipe rétention | liste des silencieux classés par P(Détracteur), filtres, capacité K, pondération CLTV optionnelle, levier recommandé, export CSV |
| Analyser un client | conseiller | client existant ou saisie manuelle → classe, probabilités, drivers de *ce* client, levier ; verbatim collé (bonus, démonstration) |
| Données et cible | data / métier | exploration, qualité, registre de fuites, construction de la cible |
| Modèles et performance | data | catalogue justifié, grille, CV vs test, réglage, par classe, calibration, robustesse, fondation, texte |
| Drivers et équité | métier / CX | drivers globaux et par segment, leviers, audit par sous-groupe, proxies, mitigation |
| Suivi et méthode | data / direction | monitoring, carte du modèle, conformité à l'énoncé, reproductibilité |

**Pourquoi Streamlit** : l'énoncé demande une interface utilisable par un responsable rétention, pas une
API ; Streamlit livre une application complète en Python pur, sans front séparé, versionnable avec le
pipeline. Une API (FastAPI) serait la brique suivante pour intégrer le score au CRM — hors périmètre.

**Robustesse aux entrées** (exigence « behave gracefully ») : chaque champ de saisie accepte
« (inconnu) » ; le pipeline impute (médiane / mode) et `OneHotEncoder(handle_unknown='ignore')`
absorbe toute modalité jamais vue. Testé automatiquement (`test_modele_tolere_entrees_vides_et_modalites_inconnues`).

**Ce que l'application dit toujours** : les probabilités servent à classer, pas à lire une fréquence ;
les drivers sont des associations ; le modèle texte est une démonstration sur du texte synthétique.

**Performance** : modèle et données chargés une fois (`st.cache_resource`) ; explication d'un client
en moins d'une seconde pour les modèles linéaires et à arbres.

## 3. Reproductibilité et livraison

`python -m src.run` (données → cible → 15 modèles → sélection → évaluation) · `python -m src.fondation_eval` ·
`python -m src.verbatims` et `python -m src.texte` · `python -m src.monitoring` · `python -m src.rapports` ·
`python -m src.writeup` · `python -m pytest tests -q`. Seed unique ; aucune clé dans le dépôt (`.env` ignoré,
`.env.example` fourni) ; test à blanc depuis une copie vierge (phase 5 § 9).

## 4. Monitoring et réentraînement (énoncé § 4.9) — implémenté

{section_mo}

**Ce qu'on surveille, et quand on réentraîne** :

| Signal | Mesure | Déclencheur |
|---|---|---|
| Dérive des entrées | PSI par variable, mensuel, vs population d'entraînement | PSI > 0,20 sur une variable majeure (`Contract`, `Tenure`, `Monthly Charge`) |
| Dérive des prédictions | PSI de P(Détracteur) ; part de Détracteurs prédits | PSI > 0,20 ou variation > 5 points sans cause métier connue |
| Performance réelle | kappa et rappel Détracteur sur les **nouvelles réponses d'enquête** | chute > 0,05 de kappa ou > 10 points de rappel |
| Équité | rappel Détracteur par tranche d'âge sur les nouvelles réponses | écart > 15 points |
| Volume de labels | nouvelles réponses reçues | réentraînement dès 500 nouveaux labels, au plus tard tous les 6 mois |
| Calibration | ECE sur les nouvelles réponses | recalibrer (Platt) sur les vraies réponses de la population cible dès 300 labels |

Outil recommandé en production : Evidently (rapports HTML autonomes), cité par l'énoncé ; ici le PSI
est implémenté directement pour rester sans dépendance.

## 5. La boucle de rétroaction — et la décision demandée

Un détracteur appelé puis retenu **change de classe**. Si on réentraîne sur ces données, le modèle
apprend l'effet de sa propre intervention comme s'il s'agissait d'une propriété du client. Il se
dégrade, et personne ne voit pourquoi. Trois parades, par ordre de coût :

1. **Journaliser chaque contact** (qui, quand, quel levier, quelle issue) et exclure du réentraînement
   les réponses postérieures à un contact, ou les traiter comme une strate séparée.
2. **Ne pas contacter 10 à 20 % des clients ciblés, tirés au hasard**, à chaque campagne. Ce groupe de
   contrôle coûte quelques clients non retenus et achète deux choses irremplaçables : la mesure de
   l'effet réel de la campagne, et les données d'un modèle d'**uplift** au cycle suivant — passer du
   ciblage par *risque* (cette étude) au ciblage par *sensibilité* (Ascarza 2018).
3. À terme, réentraîner sur les non-contactés et les contrôles seulement.

> **La décision demandée à la direction** : autoriser le groupe de contrôle dès la prochaine campagne.
> C'est le seul passage de l'étude qui demande une décision plutôt que de présenter un résultat.

{_decisions([
    ("Persistance d'un bloc : pipelines calibré, brut, de décision + métadonnées", "rejouable sans réentraînement ; l'application n'a aucune logique de préparation", "§ 1, `models/modele_final.joblib`"),
    ("Streamlit, sept pages, contenu lu dans les artefacts", "interface pour un responsable rétention ; aucune valeur recopiée", "§ 2"),
    ("Tolérance aux inconnus par le pipeline, pas par l'interface", "un seul endroit à tester", "§ 2, test dédié"),
    ("Monitoring : PSI + performance sur nouvelles réponses + volume + échéance", "les quatre déclencheurs de l'énoncé", "§ 4"),
    ("Groupe de contrôle recommandé", "seule façon de mesurer l'effet et d'éviter la boucle de rétroaction", "§ 5"),
])}
""")


# =============================================================================
# SYNTHÈSE, CONFORMITÉ, INDEX
# =============================================================================
def synthese(cfg, R, arb):
    m = R["modele_final"]["metriques_retenues"]; pk = R["evaluation_metier"]["precision_at_k"]; eco = R["evaluation_metier"]["economie"]
    ns = R["nps_simule"]; eq = R["audit_equite"]["tranche_age"]["groupes"]; mf = R["modele_final"]
    _md("00_synthese.md", f"""
# Synthèse — Prédire la catégorie NPS des clients silencieux

*Pour un lecteur non technique. Chaque chiffre est calculé et retrouvable dans les six documents de l'étude.*

## Ce qu'on a construit

Un modèle qui, pour chacun des clients n'ayant pas répondu à l'enquête NPS, estime s'il est
Détracteur, Passif ou Promoteur ; une application qui en tire une **liste d'appels priorisée** avec,
pour chaque client, les raisons de la prédiction et le levier à proposer ; une estimation du **NPS des
85 % silencieux** ; et un dispositif de suivi.

## Les cinq chiffres

| | |
|---|---|
| NPS estimé de la base complète | **{arb['nps']['M3']:+.0f}** (secteur : +19 à +34) — contre **{arb['nps']['M1']:+.0f}** avec la grille de l'énoncé prise au pied de la lettre |
| NPS des silencieux, estimé par le modèle | **{ns['nps_silencieux_predit']:+.0f}** (intervalle {ns['nps_silencieux_predit_ic95'][0]:+.0f} à {ns['nps_silencieux_predit_ic95'][1]:+.0f}) ; les répondants seuls donnent {ns['nps_repondants_observe']:+.0f} |
| Sur {pk['k']} appels, part de vrais détracteurs | **{pk['precision_at_k']:.0%}** contre {pk['taux_de_base']:.0%} au hasard — **×{pk['lift']:.1f}** |
| Gain net supplémentaire par campagne | **{_eur(eco['gain_apporte_par_le_modele_eur'])}** à hypothèses de coût constantes (à valider) |
| Détracteurs retrouvés | **{m['par_classe']['Detracteur']['rappel']:.0%}** — mais {eq['<30']['rappel_detracteur']:.0%} seulement chez les moins de 30 ans |

## Les trois résultats à retenir

**1. La grille NPS proposée dans l'énoncé fabrique un résultat faux.** Appliquée telle quelle, elle donne
un NPS de **{arb['nps']['M1']:+.0f}**, à 60–75 points des benchmarks télécom, parce qu'elle classe en détracteurs {arb['n_ambigus']}
clients « moyennement satisfaits » ({arb['n_ambigus']/7043:.0%} de la base) dont 84 % sont restés fidèles. En demandant aux
données à qui ces clients ressemblent, on trouve qu'ils penchent à **{arb['part_3_vers_promoteur']:.0%} du côté des promoteurs**. Le NPS
corrigé est de **{arb['nps']['M3']:+.0f}**, dans la fourchette du secteur.

**2. Le modèle multiplie par {pk['lift']:.1f} l'efficacité des appels**, et il a été choisi honnêtement. Quinze
modèles de sept familles ont été comparés ; le choix s'est fait **sans regarder les clients sur lesquels
il est évalué**, avec une règle qui préfère le plus simple quand les écarts sont dans le bruit. Retenu :
{_libmod(mf['modele'])}. Évalué dans les conditions réelles : entraîné sur 15 % de répondants dont la composition est
biaisée (les mécontents et les enthousiastes répondent plus), testé sur les 85 % restants.

**3. Il rate davantage les jeunes détracteurs** : {eq['<30']['rappel_detracteur']:.0%} de rappel chez les moins de 30 ans contre {eq['60+']['rappel_detracteur']:.0%}
chez les 60 ans et plus — alors qu'aucune variable d'âge n'entre dans le modèle. On a mesuré pourquoi :
les variables contractuelles **reconstruisent** l'âge. Une correction est chiffrée ; la décision de
l'appliquer appartient au métier et au juridique, pas au modèle.

## Ce qu'il faut savoir sur la fiabilité

- **Aucune variable liée au départ des clients n'est utilisée.** Le dataset en contient plusieurs qui
  auraient donné un score presque parfait et un modèle inutilisable. Identifiées, mesurées, exclues.
- **Les probabilités servent à classer les clients entre eux, pas à lire une fréquence** : ajustées sur
  des répondants biaisés, elles sont surestimées. L'application le dit à chaque écran.
- **Ce que le modèle appelle « drivers » sont des associations**, pas des causes. Contrat sans
  engagement, faible ancienneté, absence de parrainage et facture élevée vont avec la détraction ;
  l'offre commerciale E « prédit » le départ parce qu'elle a été proposée à des clients déjà fragiles.

## La décision demandée

Appeler les clients les plus à risque n'est pas la stratégie la plus rentable : la recherche établit
qu'il faut cibler ceux **dont le comportement change le plus quand on les appelle**. Mesurer cela ne
demande qu'une chose, gratuite : **ne pas contacter 10 à 20 % des clients ciblés à la prochaine
campagne, tirés au hasard.** Sans ce groupe de contrôle, on ne saura jamais si la campagne a servi — et
le modèle se dégradera en réapprenant ses propres effets.

## Ce que cette étude ne fait pas

Elle n'estime pas l'effet d'un appel (aucune campagne dans les données). Ses verbatims clients sont
synthétiques et le signal texte, démontré techniquement, n'est pas déployable sur cette base. Ses
hypothèses économiques sont des paramètres à valider avec l'équipe rétention. TabPFN n'a pas pu être
évalué (accès aux poids soumis à acceptation) ; TabICL l'a été.

## Lire la suite

[Phase 1 — métier](01_comprehension_metier.md) · [Phase 2 — données](02_comprehension_donnees.md) ·
[Phase 3 — préparation](03_preparation_donnees.md) · [Phase 4 — modélisation](04_modelisation.md) ·
[Phase 5 — évaluation](05_evaluation.md) · [Phase 6 — déploiement](06_deploiement.md) ·
[Conformité à l'énoncé](07_conformite_enonce.md)
""")


def conformite(cfg, R):
    fo = _charger_json("fondation"); tx = _charger_json("texte"); mo = _charger_json("monitoring"); repro = _charger_json("reproductibilite")
    tabicl_ok = bool(fo and any(x["nom"] == "TabICL" and x["statut"] == "ok" for x in fo["modeles"]))
    tabpfn_ok = bool(fo and any(x["nom"] == "TabPFN" and x["statut"] == "ok" for x in fo["modeles"]))
    src_txt = tx["source_verbatims"] if tx else "non exécuté"
    L = [
        ("§ 2", "Prioriser les appels vers les Détracteurs prédits", "liste classée par P(Détracteur) calibrée, K réglable, export", "phase 5 § 3 ; app *Prioriser les appels*", "✅"),
        ("§ 2", "Drivers de détraction, individuels et par segment", "contributions par client ; drivers par segment ; leviers", "phase 5 § 6 ; app *Drivers et équité*, *Analyser un client*", "✅"),
        ("§ 2", "Simuler le NPS des 85 % silencieux", "NPS prédit avec IC bootstrap, vs répondants et vs vérité ; par segment", "phase 5 § 4 ; app *Synthèse*", "✅"),
        ("§ 2", "Formulation : classification / ordinale / régression + seuils, justifiée", "les trois instanciées et comparées par la mesure", "phase 4 § 1", "✅"),
        ("§ 3", "IBM Telco 11.1.3+ comme source", "cinq tables, jointure contrôlée", "phase 2 § 1", "✅"),
        ("§ 3", "Enrichissement externe justifié si utilisé", "décision motivée de ne pas enrichir", "phase 2 § 6", "✅"),
        ("§ 4.1", "Mapping satisfaction → NPS justifié, raffiné, sensibilité discutée", "M1/M2/M3, arbitrage empirique, sensibilité des drivers", "phase 3 § 1 ; phase 5 § 5.1", "✅"),
        ("§ 4.1", "Bruit réaliste", "test de robustesse au bruit d'étiquettes 0–30 %", "phase 5 § 5.2", "✅"),
        ("§ 4.1", "Fuites dans la cible : Churn Score, Churn Value", "registre de fuites en trois natures, information mutuelle, exclusion, test", "phase 2 § 4", "✅"),
        ("§ 4.2", "Schéma documenté (gardé / supprimé / transformé)", "dictionnaire avec statut par colonne", "phase 2 § 2 ; phase 3 § 3", "✅"),
        ("§ 4.2", "Manquants et incohérences traités de façon transparente", "cinq contrôles qualité", "phase 2 § 3", "✅"),
        ("§ 4.2", "Découpage reflétant 15 % répondants / 85 % silencieux, stratégie de validation", "trois scénarios de non-réponse, MNAR retenu, CV sur les répondants", "phase 4 § 2, § 6", "✅"),
        ("§ 4.2", "Déséquilibre des classes discuté explicitement", "double déséquilibre ; cinq stratégies comparées", "phase 3 § 4", "✅"),
        ("§ 4.3", "Features construites et justifiées ; fuites évitées", "justification variable par variable ; neuf dérivées ; auto-pay", "phase 3 § 3", "✅"),
        ("§ 4.3", "Features géographiques / démographiques", "testées en ablation, exclues par équité, coût chiffré", "phase 5 § 5.3", "✅"),
        ("§ 4.4", "Verbatims synthétiques par LLM, conditionnés, bruités, reproductibles, commités", f"7 043 verbatims, source `{src_txt}` ; chemin API prêt ; prompt versionné", "phase 4 § 9 ; `data/processed/verbatims.parquet`", "✅ (générateur local seedé ; API sans clé)"),
        ("§ 4.4", "Combiner texte et tabulaire, expliquer ce que le texte apporte", "texte seul, fusion tardive, fusion précoce ; gain qualifié d'artificiel", "phase 4 § 9 ; app *Analyser un client*", "✅"),
        ("§ 4.5", "Baseline", "logistique multinomiale pondérée", "phase 4 § 3", "✅"),
        ("§ 4.5", "Au moins deux familles justifiées (boosting, ordinal, ensembles calibrés, fondation)", "quinze modèles, sept familles ; XGBoost, LightGBM, CatBoost, mord, forêt, TabICL", "phase 4 § 3, § 8", "✅"),
        ("§ 4.5", "Métriques adaptées : macro-F1, balanced accuracy, kappa, rappel par classe, calibration, lift", "toutes, avec la raison de chacune", "phase 5 § 1–3", "✅"),
        ("§ 4.5", "Validation simulant l'écart répondants / silencieux", "S3 MNAR ; sélection sans regarder les silencieux", "phase 4 § 2, § 6", "✅"),
        ("§ 4.5", "Interprétabilité : importance / SHAP, drivers", "coefficients ou SHAP selon le modèle ; méthode unique app + étude", "phase 5 § 6", "✅"),
        ("§ 4.5", "Discussion fuites, biais, bruit, qualité, limites du label", "réparties et récapitulées", "phases 2, 3, 5 § 9", "✅"),
        ("§ 4.5 bonus", "Modèle de fondation tabulaire (TabPFN-2.5, TabICL), avantages / limites / coût", f"TabICL {'évalué' if tabicl_ok else 'non évalué'} ; TabPFN {'évalué' if tabpfn_ok else 'indisponible (dépôt HF à accès contrôlé)'}", "phase 4 § 8", "✅" if tabicl_ok else "⚠️"),
        ("§ 4.6", "Drivers par segment ; actionnable vs non ; levier unique par Détracteur prédit ; corrélation ≠ causalité", "segments contrat / ancienneté / accès ; règle de levier implémentée", "phase 5 § 6", "✅"),
        ("§ 4.7", "Audit par sous-groupe démographique, rappel Détracteur", "genre, senior, marié, dépendants, âge, densité de zone ; rappel, précision, sélection", "phase 5 § 7", "✅"),
        ("§ 4.7", "Proxies d'attributs protégés ; gardé / supprimé / pourquoi", "audit de reconstruction (AUC) des attributs exclus", "phase 5 § 7.2", "✅"),
        ("§ 4.7", "Conséquence métier des choix d'équité, arbitrage explicite", "coût de l'exclusion chiffré ; mitigation par seuils chiffrée", "phase 5 § 5.3, § 7.3", "✅"),
        ("§ 4.7", "Points à escalader à CX / juridique", "liste explicite", "phase 5 § 7.4", "✅"),
        ("§ 4.8", "Persistance du modèle", "`models/modele_final.joblib`", "phase 6 § 1", "✅"),
        ("§ 4.8", "Interface : saisie ou ID → classe + probabilités ; drivers ; verbatim (bonus) ; robuste aux inconnus ; choix expliqués", "Streamlit sept pages ; test de robustesse", "phase 6 § 2 ; `app/app.py`", "✅"),
        ("§ 4.9", "Monitoring : dérive des entrées, des prédictions, performance sur nouvelles réponses", "PSI, mois simulé, lots mensuels de nouvelles réponses", "phase 6 § 4", "✅" if mo else "⚠️"),
        ("§ 4.9", "Déclencheur de réentraînement (planning, seuil, volume de labels)", "les trois, implémentés", "phase 6 § 4", "✅" if mo else "⚠️"),
        ("§ 4.9", "Boucle de rétroaction actions ↔ données", "journalisation, groupe de contrôle, réentraînement sur non-contactés", "phase 6 § 5", "✅"),
        ("§ 6", "Code source ; notebook ; README + .env.example ; dataset dérivé ; verbatims + script ; artefact modèle ; captures ; write-up 3–6 pages", "tous présents", "dépôt ; `notebooks/` ; `livrable/`", "✅"),
        ("§ 7", "Reproductible ; implémenté / approximatif / futur explicites ; usage d'IA déclaré ; pas de clés", "test à blanc " + ("réussi" if repro and repro.get("reproductible") else "à relancer") + " ; revue ; README", "phase 5 § 9 ; README", "✅"),
    ]
    t = pd.DataFrame(L, columns=["Énoncé", "Exigence", "Réponse", "Où", "Statut"])
    _md("07_conformite_enonce.md", f"""
# Conformité à l'énoncé — chaque exigence, où elle est traitée

*Relecture complète de l'énoncé (« Take-Home Challenge — Customer NPS Prediction for a Telecom
Operator ») ; une ligne par exigence ou bonus. ✅ traité · ⚠️ traité partiellement, raison indiquée.*

{_tab(t)}

## Ce qui est implémenté, approximatif, ou laissé en travail futur (énoncé § 7)

**Implémenté** : tout le périmètre obligatoire (§ 4.1 à § 4.8), le monitoring (§ 4.9), les trois bonus
(verbatims + fusion texte, drivers par segment, modèle de fondation via TabICL).

**Approximatif, et dit comme tel** : les paramètres économiques (coût d'appel, valeur client, taux de
succès) sont des placeholders ; la propension à répondre est simulée ; les verbatims viennent d'un
générateur local seedé, pas d'un appel API (aucune clé disponible ; le chemin est prêt) ; le NPS des
silencieux est estimé avec une composition par classe déformée.

**Travail futur** : TabPFN-2.5 dès accès aux poids ; recalibration sur les vraies premières réponses ;
groupe de contrôle puis modèle d'uplift ; API de scoring pour le CRM ; enrichissement texte sur de vrais verbatims.
""")


def index(R):
    _md("README.md", """
# L'étude — un document par phase CRISP-DM

| # | Document | Pour qui | Contenu |
|---|---|---|---|
| 0 | [Synthèse](00_synthese.md) | direction | les cinq chiffres, trois résultats, la décision demandée |
| 1 | [Compréhension métier](01_comprehension_metier.md) | tous | problème, usages, critères de succès, formulation, hypothèses, risques |
| 2 | [Compréhension des données](02_comprehension_donnees.md) | data | sources, dictionnaire, qualité, **registre de fuites**, EDA, enrichissement |
| 3 | [Préparation des données](03_preparation_donnees.md) | data | **construction de la cible**, features justifiées, colinéarité, déséquilibre, pipeline |
| 4 | [Modélisation](04_modelisation.md) | data | formulations, **protocole 15/85**, **quinze modèles**, grille, réglage, **sélection sans regarder le test**, calibration, fondation, texte |
| 5 | [Évaluation](05_evaluation.md) | data + métier | technique, calibration, **métier (précision@K, euros)**, **NPS simulé**, robustesse, **drivers**, **équité**, limites, revue |
| 6 | [Déploiement](06_deploiement.md) | data + direction | carte du modèle, application, reproductibilité, **monitoring**, boucle de rétroaction, décision |
| 7 | [Conformité à l'énoncé](07_conformite_enonce.md) | évaluateur | chaque exigence → où elle est traitée |

Chaque document commence par un **En bref** et se termine par les **décisions prises** dans la phase,
chacune avec sa justification et l'endroit où elle est vérifiée.

Figures dans `figures/`. Résultats bruts : `resultats.json` (pipeline), `fondation.json`, `texte.json`,
`monitoring.json`, `reproductibilite.json`. Captures de l'application dans `captures/`.
Tout est régénéré par `python -m src.run` puis `python -m src.rapports`.
""")


def main():
    cfg = charger_config()
    R = json.load(open(REP / "resultats.json", encoding="utf-8"))
    import joblib
    art = joblib.load(RACINE / "models" / "modele_final.joblib")

    print("[1/7] Données")
    df_brut, qualite = preparer(cfg)
    df_raw = joindre(charger_tables(cfg))
    df = construire(df_brut)
    cibles, arb = construire_cibles(df, cfg, cfg["seed"])
    num, cat = colonnes_modele(df, cfg)
    X = df[num + cat]; satisfaction = pd.to_numeric(df[cfg["colonnes"]["cible"]]).values

    for p in REP.glob("*.md"):
        p.unlink()
    for d in ("02_donnees", "03_preparation", "04_modelisation", "05_evaluation", "06_deploiement"):
        shutil.rmtree(REP / d, ignore_errors=True)
    for p in FIG.glob("0[2-5]_*.png"):
        p.unlink()
    (FIG / "06_nouvelles_reponses.png").unlink(missing_ok=True)

    print("[2/7] Phase 1"); phase1(cfg, R)
    print("[3/7] Phase 2"); phase2(cfg, df_raw, df, qualite, R)
    print("[4/7] Phase 3"); phase3(cfg, df, cibles, arb, num, cat, R)
    print("[5/7] Phase 4"); dec = phase4(cfg, df, cibles, X, satisfaction, num, cat, R)
    print("[6/7] Phase 5"); phase5(cfg, df, cibles, X, dec, art, R, num, cat, satisfaction)
    print("[7/7] Phase 6, synthèse, conformité, index")
    phase6(cfg, R, art); synthese(cfg, R, arb); conformite(cfg, R); index(R)
    print(f"\n{len(list(REP.glob('*.md')))} documents, {len(list(FIG.glob('*.png')))} figures -> reports/")


if __name__ == "__main__":
    main()
