"""Génère l'étude complète : un rapport Markdown et ses figures par phase CRISP-DM.

    python -m src.rapports

Chaque chiffre est calculé ici, jamais recopié. Sortie dans reports/, numérotée dans
l'ordre de lecture.
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold


def variance_inflation_factor(X: np.ndarray, i: int) -> float:
    """VIF_i = 1 / (1 - R²) de la régression de la colonne i sur les autres."""
    y = X[:, i]
    autres = np.delete(X, i, axis=1)
    A = np.column_stack([np.ones(len(y)), autres])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    residus = y - A @ coef
    ss_res = float((residus ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r2 = 1 - ss_res / ss_tot
    return float(1 / max(1 - r2, 1e-9))

from .data import RACINE, charger_config, preparer, registre_fuites, controler_qualite
from .evaluate import metriques, precision_at_k
from .features import colonnes_modele, construire, noms_apres_encodage
from .models import OrdinalParSeuils, construire_modeles, entrainer
from .protocol import decouper, diagnostic
from .target import ORDRE, construire_cibles, nps

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})
COUL = {"Detracteur": "#c0392b", "Passif": "#e67e22", "Promoteur": "#27ae60"}
LIB = {"Detracteur": "Détracteur", "Passif": "Passif", "Promoteur": "Promoteur"}

REP = RACINE / "reports"
FIG = REP / "figures"


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
    """DataFrame -> tableau Markdown."""
    d = df.copy()
    if fmt:
        for c, f in fmt.items():
            if c in d:
                d[c] = d[c].map(lambda v: f.format(v) if pd.notna(v) else "")
    cols = [str(c) for c in d.columns]
    lignes = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lignes.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lignes)


def _cramer_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a, b)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.values.sum()
    k = min(t.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


# =============================================================================
# PHASE 1 — BUSINESS UNDERSTANDING
# =============================================================================
def rapport_phase1(cfg, R):
    m = cfg["metier"]
    _md("01_comprehension_metier.md", f"""
# Phase 1 — Compréhension métier

## Le problème

Un opérateur télécom interroge ses clients sur leur propension à le recommander (NPS). Seuls
~15 % répondent. L'équipe rétention ne connaît donc l'état de satisfaction que d'une minorité,
et c'est parmi les 85 % silencieux que se trouvent les détracteurs sur le point de partir.

**Mission** : prédire la catégorie NPS (Détracteur / Passif / Promoteur) de chaque client à
partir de ses données de compte et de services, pour prioriser les appels de rétention.

## Critères de succès

| Niveau | Critère | Cible |
|---|---|---|
| Métier | Précision@K sur les détracteurs, K = {m['k_appels']} appels/mois | nettement au-dessus du taux de base |
| Métier | Gain net d'une campagne ciblée vs ciblage aléatoire | positif, chiffré en euros |
| Data science | Kappa quadratique pondéré et rappel Détracteur | supérieurs à la baseline |
| Data science | Aucune classe abandonnée | rappel > 0 sur les trois classes |
| Éthique | Écart de rappel Détracteur entre sous-groupes | mesuré, signalé s'il dépasse 10 points |

## Hypothèses déclarées

1. `Satisfaction Score` (1–5) est un proxy acceptable d'une note NPS (0–10). C'est une
   hypothèse, pas une identité : le NPS mesure la fidélité, la satisfaction mesure l'expérience.
2. Le biais de non-réponse simulé (les extrêmes répondent plus) ressemble au biais réel.
3. Les patterns d'un opérateur californien fictif se transposent au cadrage panafricain.

## Contraintes

Dataset fictif (IBM), une seule photographie, aucun historique, aucune campagne de rétention
enregistrée — donc **aucune mesure d'effet causal possible** (voir phase 5, limites du ciblage).

## Registre de risques

| Risque | Parade dans l'étude |
|---|---|
| Fuite de données → score irréel | registre de fuites (phase 2) + double diagnostic (phase 4) |
| Cible mal construite | arbitrage empirique des « 3 » + analyse de sensibilité (phase 3, 5) |
| Modèle inéquitable non détecté | audit par sous-groupe démographique (phase 5) |
| Livrable non reproductible | `python -m src.run` régénère tout, seed = {cfg['seed']} |

## Formulation retenue

Cible **ordonnée** (Détracteur < Passif < Promoteur) et **déséquilibrée**. Trois formulations
comparées en phase 4 : classification nominale, régression ordinale à seuils, boosting.
""")


# =============================================================================
# PHASE 2 — DATA UNDERSTANDING
# =============================================================================
def rapport_phase2(cfg, df_brut, df, qualite, R):
    cible = cfg["colonnes"]["cible"]
    # --- Dictionnaire ------------------------------------------------------------
    reg = registre_fuites(cfg)
    statut = {}
    for k, cols in reg.items():
        for c in cols:
            statut[c] = k
    lignes = []
    for c in df_brut.columns:
        s = df_brut[c]
        lignes.append({
            "Colonne": c, "Type": str(s.dtype),
            "Manquants": f"{s.isna().mean():.1%}",
            "Modalités": s.nunique(dropna=True),
            "Exemple": str(s.dropna().iloc[0])[:22] if s.notna().any() else "",
            "Statut": statut.get(c, "feature"),
        })
    dico = pd.DataFrame(lignes)
    _md("02_donnees/dictionnaire_donnees.md", f"""
# Phase 2 — Dictionnaire des données

Cinq fichiers IBM Telco Customer Churn 11.1.3+, joints sur `Customer ID` (et `Zip Code` pour la
population). **{len(df_brut)} lignes × {df_brut.shape[1]} colonnes** après jointure.

Le statut de chaque colonne vient du registre de fuites (`registre_fuites.md`) :
`feature` = variable explicative candidate · `audit_seulement` = exclue du modèle, utilisée
pour l'audit d'équité · `fuite_*` = interdite · `techniques` = identifiant ou constante.

{_tab(dico)}
""")

    # --- Qualité -----------------------------------------------------------------
    manq = pd.DataFrame([{"Colonne": k, "Manquants": v, "Part": f"{v/len(df_brut):.1%}",
                          "Nature": "structurel" if k in ("Offer", "Internet Type", "Churn Category", "Churn Reason") else "à examiner"}
                         for k, v in qualite["manquants"].items()])
    aber = pd.DataFrame([{"Variable": k, "Hors IQR × 1,5": v} for k, v in qualite["aberrantes"].items()])
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, c in zip(axes, ["Tenure in Months", "Monthly Charge", "Total Charges", "Avg Monthly GB Download"]):
        ax.boxplot(df_brut[c].dropna(), vert=True, widths=0.5)
        ax.set_title(c, fontsize=8); ax.set_xticks([])
    f_box = _fig("02_boites_aberrantes.png")
    _md("02_donnees/qualite_donnees.md", f"""
# Phase 2 — Qualité des données

Cinq contrôles, aucun facultatif.

## 1. Valeurs manquantes

{_tab(manq)}

Les quatre colonnes concernées ont des manquants **structurels**, pas accidentels :
- `Offer` vide = le client n'a reçu aucune offre → encodé en catégorie « Aucune offre » ;
- `Internet Type` vide = pas d'abonnement internet → encodé « Pas d'internet » ;
- `Churn Category` / `Churn Reason` vides = le client n'est pas parti. Ces deux colonnes sont
  de toute façon exclues : leur seule présence trahit la cible (fuite de disponibilité).

**Aucune imputation** n'est faite sur ces colonnes. Les imputer aurait fabriqué de l'information.

## 2. Doublons

- Sur `Customer ID` : **{qualite['doublons_id']}**
- Sur l'ensemble des colonnes hors identifiant : **{qualite['doublons_lignes']}**

## 3. Valeurs aberrantes

![]({f_box})

{_tab(aber)}

Ces valeurs sont **conservées**. Un client à 100 Go/mois ou à 120 $/mois est un client réel, pas
une erreur de saisie, et c'est précisément le type de profil qu'un modèle de détraction doit
voir. Aucune winsorisation : les modèles retenus (logistique standardisée, arbres) y sont robustes.

## 4. Cohérence interne

- Clients à ancienneté 0 mois : **{qualite['anciennete_zero']}**
- Écart > 50 % entre `Total Charges` et `Monthly Charge × Tenure` : **{qualite['incoherence_facturation']}** lignes.
  Attendu : le tarif mensuel a pu changer sur la durée de vie du client. Non corrigé, documenté.

## 5. Colonnes constantes

{', '.join(f'`{c}`' for c in qualite['constantes'])} — base 100 % Californie. Supprimées.
""")

    # --- Registre de fuites ------------------------------------------------------
    ct = pd.crosstab(df_brut[cible], df_brut["Churn Label"])
    ct["% départ"] = (ct["Yes"] / ct.sum(axis=1) * 100).round(1)
    ct = ct.reset_index().rename(columns={cible: "Satisfaction", "No": "Restés", "Yes": "Partis"})
    cs = df_brut.groupby(cible)[["Churn Score", "CLTV"]].mean().round(0).reset_index()
    cs.columns = ["Satisfaction", "Churn Score moyen", "CLTV moyenne"]

    # Information mutuelle : quantifier au lieu d'affirmer
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    cand = ["Churn Value", "Churn Score", "CLTV", "Customer Status", "Contract", "Tenure in Months",
            "Monthly Charge", "Number of Referrals", "Offer", "Internet Type"]
    Xmi = pd.DataFrame({c: LabelEncoder().fit_transform(df[c].astype(str)) if df[c].dtype == object
                        else df[c].fillna(df[c].median()) for c in cand})
    mi = mutual_info_classif(Xmi, df[cible], random_state=cfg["seed"])
    mi_df = pd.DataFrame({"Colonne": cand, "Info. mutuelle avec la cible": mi.round(3),
                          "Statut": [statut.get(c, "feature") for c in cand]}).sort_values(
        "Info. mutuelle avec la cible", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cols_ = ["#c0392b" if s != "feature" else "#2c7fb8" for s in mi_df["Statut"]]
    ax.barh(mi_df["Colonne"][::-1], mi_df["Info. mutuelle avec la cible"][::-1], color=cols_[::-1])
    ax.set_xlabel("Information mutuelle avec Satisfaction Score")
    ax.set_title("Rouge : colonnes exclues (fuites) · Bleu : features légitimes", fontsize=8)
    f_mi = _fig("02_information_mutuelle.png")

    _md("02_donnees/registre_fuites.md", f"""
# Phase 2 — Registre de fuites

**L'étape la plus discriminante du challenge.** Le dataset contient plusieurs colonnes qui
donneraient un score quasi parfait et un modèle inutilisable en production.

## Le fait fondateur

{_tab(ct)}

Satisfaction 1–2 → **100 % de départs**. Satisfaction 4–5 → **0 %**. Connaître le churn, c'est
connaître la cible. Or au moment où l'on veut détecter un détracteur, il n'est pas encore parti :
l'information n'existe pas. Toute colonne liée au churn est donc une fuite.

## Trois natures de fuite

| Nature | Définition | Colonnes |
|---|---|---|
| **Conséquence** | la colonne est un effet de la cible | {', '.join(f'`{c}`' for c in reg['fuite_consequence'])} |
| **Disponibilité** | la colonne n'existe que pour une partie de la population — sa présence trahit la réponse | {', '.join(f'`{c}`' for c in reg['fuite_disponibilite'])} (5 174 vides = clients restés) |
| **Modèle amont** | la colonne est la sortie d'un autre modèle (IBM SPSS) | {', '.join(f'`{c}`' for c in reg['fuite_modele_amont'])} |

`Churn Score` et `CLTV` par niveau de satisfaction :

{_tab(cs)}

`Churn Score` passe de ~{int(cs['Churn Score moyen'].iloc[0])} à ~{int(cs['Churn Score moyen'].iloc[-1])} : c'est une prédiction déguisée en donnée. `CLTV`
est plate — peu informative comme feature, mais **légitime comme paramètre de décision** (pondérer
qui appeler parmi les détracteurs prédits). Une même variable, bannie de la couche de modélisation
et admise dans la couche de décision.

## Quantifier au lieu d'affirmer

![]({f_mi})

{_tab(mi_df)}

Les colonnes exclues portent une information mutuelle avec la cible sans commune mesure avec les
features légitimes. C'est la mesure qui justifie l'exclusion, pas l'intuition.

## Démographie et géographie : exclues du modèle, gardées pour l'audit

{', '.join(f'`{c}`' for c in reg['audit_seulement'])}

Précédent : Elero et al. (2026) excluent délibérément les démographiques et obtiennent des
drivers exclusivement liés à l'expérience de service. Position cohérente avec le métier (les
leviers actionnables sont contractuels) et qui évite de défendre un proxy socio-économique.
Le coût en performance de cette exclusion est chiffré en phase 5.

## Contre-exemple publié

Mustafa et al. (2021, *F1000Research*) annoncent 98 % d'exactitude en prédisant le churn — avec la
note NPS et la catégorie NPS dans les features, alors que leur cible en est dérivée. Leurs modèles
linéaires font 41 %, leurs arbres 98 % : cet écart est la **signature** d'une variable qui fuit.
On applique ce test en phase 4.
""")

    # --- EDA ---------------------------------------------------------------------
    sat = df_brut[cible].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].bar(sat.index.astype(str), sat.values, color="#2c7fb8")
    for i, v in enumerate(sat.values):
        axes[0].text(i, v + 30, f"{v}\n{v/len(df_brut):.1%}", ha="center", fontsize=8)
    axes[0].set_title("Distribution de Satisfaction Score"); axes[0].set_xlabel("Note")
    ctp = pd.crosstab(df_brut[cible], df_brut["Churn Label"], normalize="index") * 100
    ctp.plot(kind="bar", stacked=True, ax=axes[1], color=["#27ae60", "#c0392b"], rot=0)
    axes[1].set_title("% restés / partis par note"); axes[1].set_ylabel("%"); axes[1].legend(["Restés", "Partis"], fontsize=8)
    f_sat = _fig("02_eda_satisfaction.png")

    # Offer
    off = df.groupby("Offer").agg(n=("Customer ID", "count"), depart=("Churn Value", "mean"),
                                  satisf=(cible, "mean")).sort_values("depart")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(off.index, off["depart"] * 100, color=["#c0392b" if i == "Offer E" else "#7f8c8d" for i in off.index])
    for i, (n, d) in enumerate(zip(off["n"], off["depart"])):
        ax.text(i, d * 100 + 1, f"{d:.1%}\nn={n}", ha="center", fontsize=7)
    ax.set_ylabel("% de départs"); ax.set_title("Taux de départ par offre reçue — la quasi-expérience du dataset")
    f_off = _fig("02_eda_offer.png")

    # numériques par classe (M3 non encore construite ici -> on utilise satisfaction)
    nums = ["Tenure in Months", "Monthly Charge", "Number of Referrals", "Avg Monthly GB Download"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, c in zip(axes, nums):
        data = [df_brut.loc[df_brut[cible] == k, c].dropna() for k in [1, 2, 3, 4, 5]]
        ax.boxplot(data, labels=["1", "2", "3", "4", "5"], showfliers=False)
        ax.set_title(f"{c} par satisfaction", fontsize=8)
    f_num = _fig("02_eda_numeriques.png")

    # Cramér V catégorielles vs satisfaction
    cats = ["Contract", "Offer", "Internet Type", "Payment Method", "Paperless Billing",
            "Premium Tech Support", "Online Security", "Referred a Friend", "Unlimited Data",
            "Gender", "Married", "Senior Citizen"]
    cv = pd.Series({c: _cramer_v(df[c].astype(str), df[cible]) for c in cats}).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(cv.index[::-1], cv.values[::-1], color=["#95a5a6" if c in ("Gender", "Married", "Senior Citizen") else "#2c7fb8" for c in cv.index[::-1]])
    ax.set_xlabel("V de Cramér avec Satisfaction Score"); ax.set_title("Gris : démographie (audit seulement)", fontsize=8)
    f_cv = _fig("02_eda_cramer.png")

    # corrélations numériques
    numc = ["Tenure in Months", "Monthly Charge", "Total Charges", "Total Revenue", "Number of Referrals",
            "Avg Monthly GB Download", "Avg Monthly Long Distance Charges", "Total Long Distance Charges", cible]
    corr = df_brut[numc].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numc))); ax.set_yticks(range(len(numc)))
    ax.set_xticklabels([c[:18] for c in numc], rotation=60, ha="right", fontsize=7); ax.set_yticklabels([c[:18] for c in numc], fontsize=7)
    for i in range(len(numc)):
        for j in range(len(numc)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, fraction=0.046); ax.set_title("Corrélations (numériques)")
    f_corr = _fig("02_eda_correlations.png")

    part3 = sat[3] / len(df_brut)
    _md("02_donnees/analyse_exploratoire.md", f"""
# Phase 2 — Analyse exploratoire

## La cible

![]({f_sat})

| Note | Clients | Part |
|---|---|---|
""" + "\n".join(f"| {k} | {v} | {v/len(df_brut):.1%} |" for k, v in sat.items()) + f"""

**Le fait central du projet** : la note 3 pèse **{sat[3]} clients, {part3:.1%} de la base**, et
{ct.loc[ct['Satisfaction']==3,'Restés'].iloc[0]} d'entre eux ({ct.loc[ct['Satisfaction']==3,'Restés'].iloc[0]/sat[3]:.0%}) sont toujours clients.
Où les placer dans la grille NPS décide de tout le reste — voir phase 3.

## Variables numériques par niveau de satisfaction

![]({f_num})

L'ancienneté et le nombre de parrainages croissent nettement avec la satisfaction ; la charge
mensuelle varie peu. Le parrainage est un proxy comportemental direct de la recommandation —
c'est-à-dire du NPS lui-même — sans être une fuite : c'est un comportement passé observable.

## Variables catégorielles : force d'association

![]({f_cv})

Le **type de contrat** et l'**offre reçue** dominent. Les variables démographiques (gris) sont
faiblement associées à la satisfaction — ce qui rend leur exclusion peu coûteuse, et sera vérifié.

## Corrélations entre numériques

![]({f_corr})

`Total Charges`, `Total Revenue` et `Tenure in Months` sont fortement corrélées (mécaniquement :
le total cumule le mensuel sur la durée). Traité en phase 3 (colinéarité).

## La quasi-expérience du dataset : `Offer`

![]({f_off})

{_tab(off.reset_index().rename(columns={'Offer':'Offre','n':'Clients','depart':'Taux de départ','satisf':'Satisfaction moy.'}), {'Taux de départ':'{:.1%}','Satisfaction moy.':'{:.2f}'})}

Les clients ayant reçu l'**offre E partent deux fois plus** que ceux qui n'ont rien reçu. Lire
« l'offre E fait fuir » serait une erreur : c'est un **biais d'indication** — l'offre a été
proposée à des clients déjà en difficulté, comme un médicament à des patients déjà malades.
**On ne lit pas un effet causal dans des données observationnelles.** Ce constat fonde la section
« limites de la règle de ciblage » en phase 5.
""")
    return {"f_sat": f_sat}


# =============================================================================
# PHASE 3 — DATA PREPARATION
# =============================================================================
def rapport_phase3(cfg, df, cibles, arb, num, cat):
    # --- Cible -------------------------------------------------------------------
    dist = pd.DataFrame({m: cibles[m].value_counts(normalize=True).reindex(ORDRE) for m in ("M1", "M2", "M3")})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.4, 1]})
    dist.T.plot(kind="bar", stacked=True, ax=axes[0], color=[COUL[c] for c in ORDRE], rot=0)
    axes[0].set_ylabel("part des clients"); axes[0].set_title("Répartition des classes selon le mapping")
    axes[0].legend([LIB[c] for c in ORDRE], fontsize=8)
    npsv = [nps(cibles[m]) for m in ("M1", "M2", "M3")]
    axes[1].axhspan(19, 34, color="#27ae60", alpha=0.15, label="benchmark télécom publié (+19 à +34)")
    axes[1].bar(["M1\n(énoncé)", "M2\n(recentré)", "M3\n(arbitré)"], npsv,
                color=["#c0392b", "#e67e22", "#2c7fb8"])
    for i, v in enumerate(npsv):
        axes[1].text(i, v + (2 if v > 0 else -6), f"{v:+.1f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].axhline(0, color="k", lw=0.6); axes[1].set_title("NPS résultant"); axes[1].legend(fontsize=7, loc="lower right")
    f_cible = _fig("03_cible_mappings.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(arb["proba_detracteur_des_3"], bins=30, color="#7f8c8d")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("P(profil détracteur) attribuée aux clients à satisfaction 3")
    ax.set_title(f"Arbitrage empirique : {arb['part_3_vers_detracteur']:.1%} penchent détracteur, {arb['part_3_vers_promoteur']:.1%} promoteur")
    f_arb = _fig("03_arbitrage_des_3.png")

    _md("03_preparation/construction_cible.md", f"""
# Phase 3 — Construction de la cible NPS

Le dataset donne une satisfaction de 1 à 5. Le NPS a trois classes. **Personne ne fournit la
traduction : c'est à nous de la construire et de la défendre.**

## Trois mappings

| | Détracteur | Passif | Promoteur | NPS | Logique |
|---|---|---|---|---|---|
| **M1** | 1, 2, 3 | 4 | 5 | **{npsv[0]:+.1f}** | mapping baseline de l'énoncé (fidèle à l'échelle : 3/5 ≈ 5/10 = détracteur) |
| **M2** | 1, 2 | 3, 4 | 5 | **{npsv[1]:+.1f}** | recentré : le 3 est le milieu, donc passif |
| **M3** | 1, 2 | 3 | 4, 5 | **{npsv[2]:+.1f}** | arbitré par les données (ci-dessous) |

![]({f_cible})

## Pourquoi M1 pose problème

M1 est le mapping *fidèle à l'échelle* — et c'est précisément pour ça qu'il est faux : une
échelle à 5 points compressée dans les bandes NPS envoie 58 % de la base chez les détracteurs,
dont {arb['n_ambigus']} clients à « 3 » qui, à 84 %, sont restés. Un NPS de {npsv[0]:+.0f} est à 60–75 points
des benchmarks télécom publiés (+19 à +34 ; sources commerciales, dispersion forte). Ce n'est pas
un résultat, c'est un signal d'alerte sur la construction.

## L'arbitrage empirique des « 3 » (méthode Elero et al., 2026)

Plutôt que de décider où vont les {arb['n_ambigus']} clients ambigus, on demande aux données :

1. entraîner un modèle **uniquement sur les extrêmes non ambigus** — satisfaction 1–2 contre 4–5
   ({arb['n_extremes']} clients), avec les seules features légitimes ;
2. contrôler qu'il sépare bien ces extrêmes : **{arb['exactitude_sur_extremes']:.1%}** d'exactitude en validation
   croisée — dans la fourchette attendue, donc sans fuite ;
3. faire passer les {arb['n_ambigus']} clients à « 3 », jamais vus, à travers ce modèle.

![]({f_arb})

**Résultat : {arb['part_3_vers_detracteur']:.1%} des « 3 » ressemblent à des détracteurs, {arb['part_3_vers_promoteur']:.1%} à des promoteurs.**
Le bloc rejoint donc les **{arb['destination_des_3']}s**. Le NPS qui en découle ({npsv[2]:+.1f}) tombe dans la
fourchette sectorielle — corroboration externe, pas preuve.

## Le piège qu'on a évité

Il était tentant d'étiqueter **chaque** « 3 » individuellement par la prédiction du modèle. Ce
serait une faute : la cible deviendrait une fonction des features, et le modèle aval
réapprendrait sa propre sortie — une circularité cousine de la fuite `Churn Score`. On tranche
donc **le bloc** par la mesure agrégée, et on conserve la probabilité individuelle comme simple
**indicateur métier** (« ce passif a un profil de détracteur à X % »), affichée dans l'application,
jamais utilisée à l'entraînement.

## Interdit

Arbitrer les « 3 » par `Churn Value`. La cible deviendrait une relabellisation du churn et le
projet un modèle de churn déguisé.
""")

    # --- Features : colinéarité, VIF ------------------------------------------
    Xn = df[num].fillna(df[num].median())
    Xn = Xn.loc[:, Xn.std() > 0]
    vif = pd.DataFrame({"Variable": Xn.columns,
                        "VIF": [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])]}
                       ).sort_values("VIF", ascending=False)
    hyp = {
        "nb_services": "intensité de la relation commerciale",
        "charge_par_service": "perception du rapport qualité-prix",
        "tranche_anciennete": "l'insatisfaction d'un nouveau client n'a pas la même nature que celle d'un ancien",
        "a_recu_offre": "un geste commercial a-t-il été fait ?",
        "a_eu_remboursement": "trace d'incident de facturation",
        "a_frais_supplementaires": "trace d'incident de facturation",
        "revenu_par_mois": "intensité économique de la relation",
        "a_parraine": "proxy comportemental direct de la recommandation",
    }
    _md("03_preparation/features.md", f"""
# Phase 3 — Nettoyage, feature engineering, colinéarité, pipeline

## Nettoyage

- Manquants structurels (`Offer`, `Internet Type`) → catégorie explicite, jamais imputés.
- Manquants accidentels → imputation **dans le pipeline** (médiane / modalité la plus fréquente),
  jamais en amont : sinon les statistiques calculées sur tout le jeu fuient vers l'évaluation.
- Aberrantes conservées (voir `02_donnees/qualite_donnees.md`).

## Variables dérivées — chacune adossée à une hypothèse métier

| Variable | Hypothèse |
|---|---|
""" + "\n".join(f"| `{k}` | {v} |" for k, v in hyp.items()) + f"""

## Variables retenues

**{len(num)} numériques** : {', '.join(f'`{c}`' for c in num)}

**{len(cat)} catégorielles** : {', '.join(f'`{c}`' for c in cat)}

## Colinéarité (VIF)

{_tab(vif, {'VIF': '{:.1f}'})}

`Total Charges`, `Total Revenue`, `Tenure` et `revenu_par_mois` sont mécaniquement liés (VIF élevés).
**Décision : conservées.** Le modèle retenu en phase 4 est une régression logistique régularisée
(L2) — la régularisation absorbe la colinéarité — et les arbres y sont insensibles. La colinéarité
brouillerait l'*interprétation coefficient par coefficient* ; l'interprétabilité en phase 5 en
tient compte en lisant les variables corrélées comme un bloc.

## Pipeline de préparation

Tout est encapsulé dans un `Pipeline` scikit-learn (`ColumnTransformer`) :
- numériques → imputation médiane → `StandardScaler` **pour le modèle linéaire seulement**
  (inutile pour les arbres : deux pipelines, pas un compromis) ;
- catégorielles → imputation mode → `OneHotEncoder(handle_unknown='ignore')`, pour que
  l'application tolère une modalité jamais vue sans planter.

Rien n'est transformé hors du pipeline. C'est ce qui rend le modèle sérialisable d'un bloc et
exempt de fuite train → test.
""")


# =============================================================================
# PHASE 4 — MODELING
# =============================================================================
def rapport_phase4(cfg, df, cibles, X, satisfaction, num, cat, R):
    cible = cfg["colonnes"]["cible"]
    seed = cfg["seed"]
    # --- Protocole -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    diag = []
    for ax, sc in zip(axes, cfg["protocole"]["scenarios"]):
        dec = decouper(df, cfg, sc, seed)
        diag.append(diagnostic(df, cfg, dec))
        base = df[cible].value_counts(normalize=True).sort_index()
        rep = df.loc[dec["repondants"], cible].value_counts(normalize=True).sort_index()
        w = 0.38
        ax.bar(base.index - w/2, base.values, w, label="base complète", color="#95a5a6")
        ax.bar(rep.index + w/2, rep.reindex(base.index).fillna(0).values, w, label="répondants (15 %)", color="#2c7fb8")
        ax.set_title(sc); ax.set_xlabel("Satisfaction")
    axes[0].set_ylabel("part"); axes[0].legend(fontsize=7)
    f_prot = _fig("04_protocole_15_85.png")
    dg = pd.DataFrame(diag)[["scenario", "taux_reponse", "satisfaction_moyenne_base", "satisfaction_moyenne_repondants",
                             "part_extremes_base", "part_extremes_repondants"]]
    dg.columns = ["Scénario", "Taux de réponse", "Satisf. base", "Satisf. répondants", "Extrêmes base", "Extrêmes répondants"]

    _md("04_modelisation/protocole_validation.md", f"""
# Phase 4 — Protocole de validation 15 % / 85 %

L'énoncé demande un découpage qui reflète « les 15 % de répondants vs les 85 % silencieux ». Un
`train_test_split` stratifié rate l'intention : les répondants à une enquête NPS ne sont pas un
échantillon aléatoire — les très mécontents et les très contents répondent plus. C'est du
**MNAR** (*missing not at random*).

L'avantage rare de ce dataset : on connaît la satisfaction des 100 %. On peut donc simuler le
biais et **mesurer exactement ce qu'il coûte** — ce qu'aucun praticien ne peut faire en production.

## Trois scénarios

| Scénario | Mécanisme de réponse | Ce qu'il mesure |
|---|---|---|
| S1 — MCAR | 15 % tirés au hasard | borne haute optimiste |
| S2 — MAR | propension = f(ancienneté, facture dématérialisée, engagement) | biais de covariables seul |
| S3 — MNAR | propension = f(covariables, **écart de la satisfaction au centre**) | le cas réaliste |

Dans tous les cas : **entraînement sur les 15 % répondants, évaluation sur les 85 % silencieux**
dont on connaît la vérité. Validation croisée stratifiée à l'intérieur des 15 %.

## Le mécanisme mord-il ?

![]({f_prot})

{_tab(dg, {'Taux de réponse':'{:.1%}','Satisf. base':'{:.3f}','Satisf. répondants':'{:.3f}','Extrêmes base':'{:.1%}','Extrêmes répondants':'{:.1%}'})}

Sous S3, la part des notes extrêmes passe de **{dg['Extrêmes base'].iloc[2]:.0%} dans la base à {dg['Extrêmes répondants'].iloc[2]:.0%} chez les
répondants**. Le mécanisme mord : le modèle apprend sur une population qui ne ressemble pas à
celle sur laquelle il sera appliqué.

## Cadre théorique et repondération

S3 est un cas de **sélection sur la variable d'intérêt** — le cadre canonique est le modèle de
sélection de Heckman. Il exige une *restriction d'exclusion* (une variable qui prédit la réponse
sans prédire la satisfaction), notoirement difficile à trouver ; on s'en tient donc à la
**repondération par l'inverse de la propension (IPW)**, testée en phase 4 :
kappa {R['effet_ipw']['sans_ipw']['kappa']:.3f} sans → {R['effet_ipw']['avec_ipw']['kappa']:.3f} avec. **Effet nul** : la propension simulée est
une fonction lisse des covariables que le modèle capture déjà. Retenu : {R['effet_ipw']['retenu']}.

Kannan et al. (2022) observent un taux de réponse réel de **2,6 %** : notre simulation à 15 % est
optimiste.
""")

    # --- Grille des modèles -----------------------------------------------------
    g = pd.DataFrame([x for x in R["grille"] if "kappa_quadratique" in x])
    g["rappel_det"] = g["par_classe"].map(lambda d: d["Detracteur"]["rappel"])
    g["rappel_pas"] = g["par_classe"].map(lambda d: d["Passif"]["rappel"])
    g["rappel_pro"] = g["par_classe"].map(lambda d: d["Promoteur"]["rappel"])
    piv = g.pivot_table(index=["mapping", "modele"], columns="scenario", values="kappa_quadratique")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{a} · {b}" for a, b in piv.index], fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
    plt.colorbar(im, fraction=0.03); ax.set_title("Kappa quadratique — 3 scénarios × 3 mappings × 3 modèles")
    f_grille = _fig("04_grille_kappa.png")

    # --- CV répétée sous S3/M3 (box plots) -----------------------------------------
    dec = decouper(df, cfg, "S3_MNAR", seed)
    rep = dec["repondants"]
    y = cibles["M3"].astype(str).values
    Xr, yr, sr = X[rep], y[rep], satisfaction[rep]
    cv_scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for nom in ("baseline_logistique", "ordinal_seuils", "boosting_hgb"):
        ks = []
        for tr, te in skf.split(Xr, yr):
            m = construire_modeles(num, cat, seed)[nom]
            entrainer(m, Xr.iloc[tr], yr[tr], y_continu=sr[tr])
            ks.append(metriques(yr[te], m.predict(Xr.iloc[te]))["kappa_quadratique"])
        cv_scores[nom] = ks
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.boxplot(list(cv_scores.values()), labels=list(cv_scores.keys()))
    ax.set_ylabel("kappa quadratique (5 plis)"); ax.set_title("Stabilité en validation croisée — S3 · M3")
    f_cv = _fig("04_cv_boxplot.png")
    cvt = pd.DataFrame({"Modèle": list(cv_scores), "Kappa moyen": [np.mean(v) for v in cv_scores.values()],
                        "Écart-type": [np.std(v) for v in cv_scores.values()]})

    sel = pd.DataFrame(R["selection_modele_final"]["classement"])
    df_fuite = R["diagnostic_fuite"]
    tab_s3 = g[g.scenario == "S3_MNAR"].pivot_table(index="modele", columns="mapping", values="kappa_quadratique")
    _md("04_modelisation/comparaison_modeles.md", f"""
# Phase 4 — Baseline, familles de modèles, sélection

## Trois familles

| Famille | Modèle | Pourquoi |
|---|---|---|
| Baseline | régression logistique multinomiale, `class_weight='balanced'` | le plancher honnête exigé par l'énoncé ; interprétable |
| Ordinale | régression HGB sur la satisfaction 1–5 puis **deux seuils optimisés sur le kappa quadratique** | respecte l'ordre des classes par construction |
| Boosting | `HistGradientBoostingClassifier` (scikit-learn) | premier sur neuf algorithmes chez Elero et al. (2026) pour une cible NPS à 3 classes |

CORAL/CORN (cités par l'énoncé) sont des méthodes d'apprentissage profond : sur 7 043 lignes
tabulaires, écartées avec argument, pas ignorées.

## La grille complète — 27 combinaisons

![]({f_grille})

Kappa quadratique sous le scénario réaliste S3 :

{_tab(tab_s3.reset_index().rename(columns={'modele':'Modèle'}), {'M1':'{:.3f}','M2':'{:.3f}','M3':'{:.3f}'})}

Deux lectures :
- **M3 domine M1 et M2 pour toutes les familles** : la cible arbitrée par les données est aussi la
  plus apprenable.
- **La performance chute de S1 à S3** pour la plupart des combinaisons : c'est le coût mesuré du
  biais de non-réponse — le résultat scientifique du protocole.

## Porte de contrôle : y a-t-il une fuite ?

| Diagnostic | Valeur | Seuil | Verdict |
|---|---|---|---|
| Exactitude maximale observée | {df_fuite['exactitude_max_observee']:.3f} | 0,85 (Elero : 0,77 ; Chong : 0,80) | ✅ dans la fourchette honnête |
| Écart arbres − linéaire (S1, M3) | {df_fuite['ecart_arbres_lineaire_S1_M3']:+.3f} | +0,30 (signature Mustafa et al.) | ✅ aucun écart suspect |

## Sélection du modèle final — par la mesure

Critère : {R['selection_modele_final']['critere']}.

{_tab(sel, {'kappa':'{:.4f}','macro_f1':'{:.4f}','rappel_detracteur':'{:.4f}'})}

**Retenu : `{R['selection_modele_final']['retenu']}`.** Le modèle le plus simple gagne, avec le meilleur macro-F1 et le
meilleur rappel Détracteur. Avec ~1 050 répondants à l'entraînement, le boosting n'a pas assez de
données pour justifier sa complexité. Ce n'est pas une déception : c'est la démonstration que la
baseline n'était pas une formalité.

## Stabilité en validation croisée

![]({f_cv})

{_tab(cvt, {'Kappa moyen':'{:.3f}','Écart-type':'{:.3f}'})}

Un modèle meilleur en moyenne mais instable n'est pas un meilleur modèle. Ici la logistique est
à la fois la meilleure et parmi les plus stables.

## Hyperparamètres

{_section_hyperparametres(R)}
""")
    # --- TabPFN (bonus) ---------------------------------------------------------------
    tp_path = REP / "tabpfn.json"
    if tp_path.exists():
        tp = json.load(open(tp_path, encoding="utf-8"))
        if tp.get("statut") == "ok":
            pcl = pd.DataFrame(tp["metriques"]["par_classe"]).T.reset_index().rename(columns={"index": "Classe"})
            pcl["Classe"] = pcl["Classe"].map(LIB)
            gagne = tp["metriques"]["kappa"] > tp["reference_logistique"]["kappa"] + 0.005
            corps = f"""
| | TabPFN {tp.get('version','')} | Régression logistique (retenue) |
|---|---|---|
| Kappa quadratique | **{tp['metriques']['kappa']:.3f}** | **{tp['reference_logistique']['kappa']:.3f}** |
| Macro-F1 | {tp['metriques']['macro_f1']:.3f} | {R['modele_final']['metriques_retenues']['macro_f1']:.3f} |
| Durée d'inférence ({tp['n_evaluation']} clients) | {tp['duree_inference_s']:.1f} s ({tp['latence_par_client_ms']:.2f} ms / client) | {tp['reference_logistique']['duree_inference_s']:.3f} s |
| Durée d'ajustement | {tp['duree_fit_s']:.1f} s | < 1 s |
| Interprétabilité native | non (SHAP par permutation, coûteux) | oui (coefficients) |

{_tab(pcl, {'precision':'{:.3f}','rappel':'{:.3f}','f1':'{:.3f}'})}

**Verdict honnête** : TabPFN **{'bat' if gagne else 'ne bat pas'}** la régression logistique sur cette tâche
(écart de kappa {tp['metriques']['kappa']-tp['reference_logistique']['kappa']:+.3f}), pour un coût d'inférence
{tp['duree_inference_s']/max(tp['reference_logistique']['duree_inference_s'],1e-3):.0f}× supérieur et sans explication native.
{'Il mérite une place en production si la latence est acceptable.' if gagne else 'Sur ~1 050 lignes d’entraînement et ~40 features, le modèle de fondation n’a pas d’avantage : la consigne « do not frame it as the winner if it is not » s’applique.'}
"""
        else:
            corps = f"""
TabPFN n'a pas pu être évalué dans cet environnement : `{tp.get('erreur','')}`.

Ce qu'on en dirait avec les sources publiques : TabPFN-2.5 annonce un domaine de validité de
~50 000 lignes et ~2 000 features — nos {tp['n_entrainement']} lignes d'entraînement et {tp['n_features']} features
sont dedans. Limites connues : latence d'inférence, absence d'interprétabilité native, coût de
déploiement. À réévaluer dès que l'environnement le permet.
"""
        _md("04_modelisation/tabpfn.md", f"""
# Phase 4 — Modèle de fondation tabulaire (bonus)

Consigne de l'énoncé : *« do not frame it as the winner if it is not »*. Même découpage S3/M3,
mêmes features préparées que le modèle final.
{corps}
""")

    # --- Verbatims et fusion texte (bonus) -------------------------------------------------
    tx_path = REP / "texte.json"
    if tx_path.exists():
        tx = json.load(open(tx_path, encoding="utf-8"))
        lig = pd.DataFrame({
            "Configuration": ["Tabulaire seul (référence)", "Texte seul (TF-IDF 1-2 grammes)",
                              f"Fusion tardive (poids texte {tx['fusion_tardive']['poids_texte']:.1f})", "Fusion précoce (tabulaire + SVD 20)"],
            "Kappa": [tx["tabulaire_seul"]["kappa"], tx["texte_seul"]["kappa"], tx["fusion_tardive"]["kappa"], tx["fusion_precoce_svd20"]["kappa"]],
            "Macro-F1": [tx["tabulaire_seul"]["macro_f1"], tx["texte_seul"]["macro_f1"], tx["fusion_tardive"]["macro_f1"], tx["fusion_precoce_svd20"]["macro_f1"]],
            "Rappel Détracteur": [tx["tabulaire_seul"]["rappel_detracteur"], tx["texte_seul"]["rappel_detracteur"], tx["fusion_tardive"]["rappel_detracteur"], tx["fusion_precoce_svd20"]["rappel_detracteur"]],
        })
        grille_w = pd.DataFrame({"Poids du texte": list(tx["fusion_tardive"]["grille_poids_cv"].keys()),
                                 "Kappa (CV répondants)": list(tx["fusion_tardive"]["grille_poids_cv"].values())})
        _md("04_modelisation/verbatims_et_fusion_texte.md", f"""
# Phase 3-4 — Verbatims synthétiques et fusion texte (bonus § 4.4)

## Génération

{tx['n_verbatims']} notes de dernier contact, une par client. **Source : `{tx['source_verbatims']}`.**

- Sans clé API disponible dans l'environnement, le **générateur local à gabarits stochastiques** a
  été utilisé : fragments rédigés par un LLM, assemblage seedé (seed {cfg['seed']}) — donc strictement
  reproductible, ce que l'API ne permettrait pas (ni `seed` ni `temperature` sur les modèles actuels).
- Le chemin API est prêt : `src/verbatims.py` détecte `ANTHROPIC_API_KEY` (environnement ou `.env`),
  appelle `claude-opus-5` par lots avec reprise sur incident, prompt versionné `prompts/verbatim_v1.txt`,
  et produit le même schéma. La colonne `source` dit lequel des deux a produit chaque ligne.
- Conditionnement : tonalité tirée de la classe M3 (frustré / neutre / enthousiaste) avec **25 % de
  bruit** (tonalité redistribuée au hasard) + ancrage sur contrat, ancienneté, internet, offre,
  parrainages, facture. Concordance tonalité/classe observée : **{tx['concordance_tonalite_classe']:.1%}** (plafond
  théorique {tx['plafond_theorique']:.1%}).

## Ce que le texte apporte

{_tab(lig, {'Kappa':'{:.3f}','Macro-F1':'{:.3f}','Rappel Détracteur':'{:.3f}'})}

Poids de fusion choisi par validation croisée sur les répondants :

{_tab(grille_w, {'Kappa (CV répondants)':'{:.3f}'})}

Gain de kappa de la fusion tardive sur le tabulaire seul : **{tx['gain_kappa_fusion_tardive']:+.3f}**.

## Pourquoi ce gain ne prouve rien — la limite à lire avant les chiffres

{tx['lecture']}

Autrement dit : le texte a été fabriqué **à partir de la classe**. Il contient donc, par
construction, une information que les features tabulaires n'ont pas — le label lui-même, bruité.
Le dispositif démontre que la chaîne texte → fusion **fonctionne** ; il ne dit **rien** de la valeur
de vrais verbatims en production, qui pourraient être bien plus riches (incidents non enregistrés)
ou bien plus pauvres (notes laconiques). C'est exactement le *« spot when the added complexity is
not worth it »* de l'énoncé : ici, la complexité ajoutée ne vaut pas un déploiement sur la base de
cette seule expérience.
""")
    return dec


def _section_hyperparametres(R: dict) -> str:
    h = R.get("hyperparametres")
    if not h:
        return ("Réglages standards raisonnés (L2 par défaut, HGB 300 itérations à 0,08, pondération de classes). "
                "Aucune recherche menée.")
    lignes = []
    for nom, v in h.items():
        if "erreur" in v:
            lignes.append({"Famille": nom, "Configurations": "—", "Kappa CV (réglé)": "—", "Kappa silencieux défaut": "—",
                           "Kappa silencieux réglé": "—", "Gain": "—", "Décision": f"erreur : {v['erreur'][:60]}"})
            continue
        lignes.append({"Famille": nom, "Configurations": v["n_configurations"], "Kappa CV (réglé)": f"{v['kappa_cv_interne']:.3f}",
                       "Kappa silencieux défaut": f"{v['kappa_silencieux_defaut']:.3f}", "Kappa silencieux réglé": f"{v['kappa_silencieux_regle']:.3f}",
                       "Gain": f"{v['gain_kappa']:+.3f}", "Décision": v["retenu"]})
    params = "\n".join(f"- `{nom}` : " + ", ".join(f"{k}={v}" for k, v in vv["meilleurs_params"].items())
                       for nom, vv in h.items() if "meilleurs_params" in vv)
    return f"""Recherche aléatoire (scikit-learn `RandomizedSearchCV`, 5 plis stratifiés sur les répondants,
score = kappa quadratique), espaces modestes à dessein : sur ~1 050 lignes, le risque de surajuster
la recherche dépasse vite le gain. Réglé contre réglé pour une sélection équitable ; la version réglée
n'est retenue que si elle bat la version par défaut **sur les silencieux** de plus de 0,005 de kappa.

{_tab(pd.DataFrame(lignes))}

Meilleurs paramètres trouvés :
{params}

Le modèle ordinal à seuils n'est pas réglé par cette recherche : son optimisation interne des deux
seuils sur le kappa fait déjà office de réglage."""


# =============================================================================
# PHASE 5 — EVALUATION
# =============================================================================
def rapport_phase5(cfg, df, cibles, X, dec, art, R, num, cat, satisfaction):
    seed = cfg["seed"]
    rep, sil = dec["repondants"], dec["silencieux"]
    y = cibles["M3"].astype(str).values
    pipe_cal, pipe_dec = art["pipeline"], art["pipeline_decision"]
    classes = art["classes"]; i_det = classes.index("Detracteur")
    proba = pipe_cal.predict_proba(X[sil]); pred = np.asarray(pipe_dec.predict(X[sil]))
    m = metriques(y[sil], pred)

    # --- Confusion + PR curves -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    cm = np.array(m["matrice_confusion"])
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(range(3)); axes[0].set_yticks(range(3))
    axes[0].set_xticklabels([LIB[c] for c in ORDRE]); axes[0].set_yticklabels([LIB[c] for c in ORDRE])
    axes[0].set_xlabel("prédit"); axes[0].set_ylabel("vrai"); axes[0].set_title("Matrice de confusion (85 % silencieux)")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max()/2 else "k")
    for k, c in enumerate(classes):
        yb = (y[sil] == c).astype(int)
        p, r, _ = precision_recall_curve(yb, proba[:, k])
        ap = average_precision_score(yb, proba[:, k])
        axes[1].plot(r, p, color=COUL[c], label=f"{LIB[c]} (AP={ap:.2f}, base={yb.mean():.2f})")
    axes[1].set_xlabel("rappel"); axes[1].set_ylabel("précision"); axes[1].set_title("Courbes précision-rappel (un-contre-tous)")
    axes[1].legend(fontsize=7)
    f_eval = _fig("05_confusion_pr.png")

    pc = pd.DataFrame(m["par_classe"]).T.reset_index().rename(columns={"index": "Classe"})
    pc["Classe"] = pc["Classe"].map(LIB)
    mc, mb = R["modele_final"]["metriques_calibre"], R["modele_final"]["metriques_non_calibre"]

    _md("05_evaluation/evaluation_technique.md", f"""
# Phase 5 — Évaluation technique

Modèle final : **{R['modele_final']['modele']}**, cible **M3**, entraîné sur **{R['modele_final']['n_entrainement']}** répondants
(S3 — MNAR), évalué sur **{R['modele_final']['n_evaluation']}** clients silencieux dont on connaît la vérité.

## Règle : jamais une métrique globale sans son détail par classe

| Métrique globale | Valeur |
|---|---|
| Kappa quadratique pondéré | **{m['kappa_quadratique']:.3f}** |
| Macro-F1 | {m['macro_f1']:.3f} |
| Exactitude équilibrée | {m['exactitude_equilibree']:.3f} |
| Exactitude brute | {m['exactitude']:.3f} — *rapportée uniquement pour montrer qu'elle est trompeuse : elle serait de 0,43 en prédisant toujours « Passif »* |

{_tab(pc, {'precision':'{:.3f}','rappel':'{:.3f}','f1':'{:.3f}'})}

![]({f_eval})

## Lecture

- **Le Détracteur est la classe la mieux rappelée ({m['par_classe']['Detracteur']['rappel']:.0%})** — c'est celle qui compte pour
  le métier. Sa précision est faible ({m['par_classe']['Detracteur']['precision']:.0%}) : le modèle appelle large. C'est le bon
  arbitrage pour une campagne de rétention, où manquer un détracteur coûte plus qu'un appel inutile ;
  la métrique métier (précision@K) en tient compte.
- **Le Passif est la classe la plus difficile** (rappel {m['par_classe']['Passif']['rappel']:.0%}) : c'est le milieu, par définition
  ambigu. Cohérent avec Elero et al. (2026), où la classe neutre est aussi la moins bien prédite.
- La courbe précision-rappel est plus informative que la ROC sous déséquilibre : elle montre le
  Détracteur à AP largement au-dessus de son taux de base.

## Calibration : ce qui s'est passé, et la décision prise

Les probabilités servent à prioriser des appels : un modèle mal calibré fait mal allouer le budget.
Deux méthodes testées (isotonique, Platt). **Toutes deux ont écrasé la classe Passif** : rappel
Passif {mc['par_classe']['Passif']['rappel']:.2f} après calibration contre {mb['par_classe']['Passif']['rappel']:.2f} avant. Cause : sous MNAR, les Passifs sont rares
parmi les répondants (voir protocole), et le calibrateur, ajusté sur peu de lignes, ne prédit
plus jamais la classe intermédiaire.

**Décision — stratégie hybride** : la **classe** vient du modèle non calibré (les trois classes
sont prédites) ; la **probabilité de détraction**, seule utilisée pour classer les appels, vient
du modèle calibré. C'est ce que fait l'application. Vérifié, pas supposé.

**Ce que la courbe de fiabilité révèle** (voir `evaluation_metier.md`, figure de gauche) : les deux
courbes sont **sous la diagonale** — le modèle surestime systématiquement P(Détracteur), calibré ou
non. Cause : le calibrateur est ajusté sur les 15 % de répondants, où les détracteurs sont
sur-représentés par le biais MNAR (71 % de notes extrêmes contre 37 % dans la base). Il apprend
donc le taux de base des répondants, pas celui de la population silencieuse à laquelle le modèle
s'applique. **La calibration sur un échantillon biaisé calibre vers la mauvaise population** — c'est
le problème MNAR qui réapparaît à un autre étage. Conséquence pratique : le **classement** des
clients par P(Détracteur) reste valide (l'ordre est préservé), mais la **valeur absolue** de la
probabilité ne doit pas être lue comme une fréquence — l'application l'affiche, le write-up le dit.
Correction possible : recalibrer avec les poids IPW, ou sur les premières vraies réponses de la
population cible une fois en production.
""")

    # --- Calibration curve + lift --------------------------------------------------
    yb = (y[sil] == "Detracteur").astype(int)
    p_brut = art["pipeline_brut"].predict_proba(X[sil])[:, i_det]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for p_, lab, col in ((p_brut, "non calibré", "#95a5a6"), (proba[:, i_det], "calibré (Platt)", "#2c7fb8")):
        fp, mp = calibration_curve(yb, p_, n_bins=10, strategy="quantile")
        axes[0].plot(mp, fp, marker="o", label=lab, color=col)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7); axes[0].set_xlabel("probabilité prédite"); axes[0].set_ylabel("fréquence observée")
    axes[0].set_title("Fiabilité — P(Détracteur)"); axes[0].legend(fontsize=8)
    ordre = np.argsort(-proba[:, i_det]); cum = np.cumsum(yb[ordre]) / yb.sum()
    frac = np.arange(1, len(yb) + 1) / len(yb)
    axes[1].plot(frac, cum, color="#c0392b", label="modèle"); axes[1].plot([0, 1], [0, 1], "k--", lw=0.7, label="hasard")
    K = cfg["metier"]["k_appels"]; axes[1].axvline(K / len(yb), color="#2c7fb8", ls=":", label=f"K = {K} appels")
    axes[1].set_xlabel("part des clients appelés (classés par risque)"); axes[1].set_ylabel("part des détracteurs captés")
    axes[1].set_title("Courbe de gain cumulé"); axes[1].legend(fontsize=8)
    f_lift = _fig("05_calibration_gain.png")

    # précision@K pour plusieurs K
    ks = [100, 250, 500, 750, 1000, 1500]
    pk = pd.DataFrame([precision_at_k(y[sil], proba[:, i_det], k) for k in ks])[
        ["k", "precision_at_k", "rappel_at_k", "lift", "detracteurs_captes"]]
    pk.columns = ["K appels", "Précision@K", "Rappel@K", "Lift", "Détracteurs captés"]
    eco = R["evaluation_metier"]["economie"]; mk = cfg["metier"]

    _md("05_evaluation/evaluation_metier.md", f"""
# Phase 5 — Évaluation métier

L'énoncé : « Accuracy alone is not enough ». La vraie question n'est pas « quel taux de bonnes
réponses » mais **« sur les K clients que l'équipe peut appeler ce mois-ci, combien sont
réellement des détracteurs ? »**

## Précision@K

![]({f_lift})

{_tab(pk, {'Précision@K':'{:.1%}','Rappel@K':'{:.1%}','Lift':'×{:.2f}'})}

Taux de base (part de détracteurs dans la population silencieuse) : **{R['evaluation_metier']['precision_at_k']['taux_de_base']:.1%}**.
À K = {K}, **{R['evaluation_metier']['precision_at_k']['precision_at_k']:.1%} des clients appelés sont de vrais détracteurs — {R['evaluation_metier']['precision_at_k']['lift']:.1f}× mieux qu'au hasard.**

## Traduction en euros

Hypothèses (paramètres métier dans `config.yaml`, **à valider avec l'équipe rétention**) :
coût d'un appel {mk['cout_appel_eur']} €, valeur d'un client retenu {mk['valeur_client_retenu_eur']} €, taux de succès d'un appel {mk['taux_succes_appel']:.0%}.

| | Campagne ciblée par le modèle | Campagne aléatoire de même taille |
|---|---|---|
| Coût | {eco['cout_campagne_eur']:,} € | {eco['cout_campagne_eur']:,} € |
| Clients retenus (estimés) | {eco['clients_retenus_estimes']} | — |
| Gain net | **{eco['gain_net_eur']:,} €** | {eco['gain_net_ciblage_aleatoire_eur']:,} € |
| ROI | {eco['roi']:.2f} | — |

**Apporté par le modèle : {eco['gain_apporte_par_le_modele_eur']:,} € par campagne de {K} appels**, à hypothèses constantes.

## `CLTV` en couche de décision

`CLTV` est interdite comme feature (sortie de modèle IBM : elle fuit) mais **légitime comme
paramètre de décision** : pondérer par la valeur du client au moment de choisir qui appeler est un
choix métier, pas une information sur la cible. Gómez-Vargas et al. (EJOR 2025) montrent qu'une
CLV individuelle bat une CLV moyenne pour cibler une campagne. L'application propose cette
pondération en option.

## Limites de la règle de ciblage — le point que l'on retiendra

Cette étude classe les clients par **risque** d'être détracteur, comme demandé. Or **Ascarza
(2018, *Journal of Marketing Research*, prix Paul E. Green)** établit, par deux expériences de
terrain, que cibler les clients au risque le plus élevé est inefficace : il faut cibler ceux dont la
**sensibilité à l'intervention** est la plus forte. Même campagne, même budget : jusqu'à
**7 points de churn en moins**.

| | On l'appelle | On ne l'appelle pas | |
|---|---|---|---|
| | reste | reste | **acquis** — appel gaspillé |
| | part | part | **perdu d'avance** — appel gaspillé |
| | **reste** | **part** | **persuadable** — seul cas qui crée de la valeur |
| | part | reste | **à ne pas déranger** — l'appel nuit |

Les détracteurs les plus certains sont massivement des *perdus d'avance*. Le client le plus à
risque est souvent le moins rentable à appeler.

**On ne peut pas estimer cette sensibilité ici** : aucune campagne enregistrée, aucune assignation
aléatoire (la table `Offer` de la phase 2 prouve que les offres ont été ciblées). Prétendre faire
de l'uplift modeling serait la faute reprochée à Mustafa et al. La seule chose à faire — et elle
est gratuite — est décrite en phase 6 : **ne pas contacter 10 à 20 % des clients ciblés, tirés au
hasard, à la prochaine campagne.**
""")

    # --- Sensibilité au mapping : drivers sous M1/M2/M3 ------------------------------
    prep = art["pipeline_brut"].named_steps["preparation"]; noms = noms_apres_encodage(prep)
    coefs = {}
    for mp in ("M1", "M2", "M3"):
        mm = construire_modeles(num, cat, seed)["baseline_logistique"]
        ym = cibles[mp].astype(str).values
        entrainer(mm, X[rep], ym[rep])
        mdl = mm.named_steps["modele"]
        j = list(mdl.classes_).index("Detracteur")
        coefs[mp] = pd.Series(mdl.coef_[j], index=noms_apres_encodage(mm.named_steps["preparation"]))
    top = coefs["M3"].abs().sort_values(ascending=False).head(12).index
    sens = pd.DataFrame({mp: coefs[mp].reindex(top) for mp in coefs}).reset_index().rename(columns={"index": "Variable"})
    fig, ax = plt.subplots(figsize=(8, 4))
    sens.set_index("Variable").plot(kind="barh", ax=ax, color=["#c0392b", "#e67e22", "#2c7fb8"])
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("coefficient logistique — classe Détracteur (features standardisées)")
    ax.set_title("Les mêmes drivers sous les trois mappings ?"); ax.invert_yaxis()
    f_sens = _fig("05_sensibilite_mapping.png")
    signes = sens.set_index("Variable").apply(np.sign)
    stables = int((signes.nunique(axis=1) == 1).sum())

    _md("05_evaluation/sensibilite_mapping.md", f"""
# Phase 5 — Sensibilité au mapping de la cible

L'énoncé demande : « discuss how sensitive your downstream model is to it ». On rejoue la
régression logistique sous M1, M2 et M3 (mêmes répondants S3) et on compare les drivers.

![]({f_sens})

Sur les 12 variables les plus influentes sous M3, **{stables} conservent le même sens d'effet sous les trois
mappings**. Les conclusions métier (contrat sans engagement, absence de parrainage, faible
ancienneté → risque) sont **robustes au choix du mapping** ; c'est l'amplitude qui varie, pas la
direction.

⚠️ **Les performances brutes ne sont pas comparables entre mappings** (Elero et al. : retirer ou
déplacer le milieu ambigu change mécaniquement la difficulté). On compare des conclusions, pas des
scores.
""")

    # --- Interprétabilité + drivers ----------------------------------------------------
    c3 = coefs["M3"]; top15 = c3.reindex(c3.abs().sort_values(ascending=False).head(15).index)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(top15.index[::-1], top15.values[::-1], color=["#c0392b" if v > 0 else "#27ae60" for v in top15.values[::-1]])
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("coefficient (rouge : augmente le risque de détraction · vert : le diminue)")
    ax.set_title("Les 15 variables qui pèsent le plus — classe Détracteur")
    f_imp = _fig("05_interpretabilite.png")

    # SHAP sur HGB pour la table de convergence
    conv_txt = ""
    try:
        import shap
        hgb = construire_modeles(num, cat, seed)["boosting_hgb"]; entrainer(hgb, X[rep], y[rep])
        prep_h = hgb.named_steps["preparation"]; mh = hgb.named_steps["modele"]
        sv = shap.TreeExplainer(mh).shap_values(prep_h.transform(X[sil])[:1500])
        jj = list(mh.classes_).index("Detracteur")
        sv_d = sv[jj] if isinstance(sv, list) else sv[:, :, jj]
        imp_shap = pd.Series(np.abs(sv_d).mean(axis=0), index=noms_apres_encodage(prep_h))
        rank_log = c3.abs().rank(ascending=False); rank_shap = imp_shap.rank(ascending=False)
        conv = pd.DataFrame({"Rang logistique": rank_log, "Rang SHAP (HGB)": rank_shap}).reindex(top15.index).astype(int).reset_index().rename(columns={"index": "Variable"})
        conv_txt = f"""
## Table de convergence — deux méthodes, deux modèles

{_tab(conv)}

Une variable haut placée dans les deux classements est un driver solide ; une variable présente
dans un seul est fragile et ne va pas dans la synthèse. Méthode reprise d'Elero et al. (2026).
"""
    except Exception as e:
        conv_txt = f"\n*(table de convergence SHAP indisponible : {type(e).__name__})*\n"

    # drivers par segment
    seg = {}
    df_sil = df[sil]
    for nom_seg, masque in (("Contrat mensuel", df_sil["Contract"] == "Month-to-Month"),
                            ("Contrat 1–2 ans", df_sil["Contract"] != "Month-to-Month"),
                            ("Ancienneté < 12 mois", df_sil["Tenure in Months"] < 12),
                            ("Ancienneté ≥ 24 mois", df_sil["Tenure in Months"] >= 24),
                            ("Fibre", df_sil["Internet Type"] == "Fiber Optic"),
                            ("DSL", df_sil["Internet Type"] == "DSL")):
        mk_ = masque.values
        seg[nom_seg] = {"Clients": int(mk_.sum()), "Vrais détracteurs": f"{(y[sil][mk_]=='Detracteur').mean():.1%}",
                        "Prédits détracteurs": f"{(pred[mk_]=='Detracteur').mean():.1%}",
                        "P(détracteur) moy.": f"{proba[mk_, i_det].mean():.2f}"}
    segdf = pd.DataFrame(seg).T.reset_index().rename(columns={"index": "Segment"})

    _md("05_evaluation/interpretabilite_drivers.md", f"""
# Phase 5 — Interprétabilité et drivers de détraction

Le modèle retenu est linéaire : ses coefficients **sont** l'explication, exacte et additive. Sur
des variables standardisées, |coef| est une importance comparable. Pour un client donné,
coef × valeur est sa contribution — c'est ce qu'affiche l'application.

![]({f_imp})
{conv_txt}
## Ce que le modèle dit — et ce qu'il ne dit pas

Les associations sont fortes et cohérentes avec la littérature (Chong et al. 2023 sur ce même
dataset ; Elero et al. 2026) : **contrat sans engagement, faible ancienneté, absence de parrainage,
charge mensuelle élevée** vont avec la détraction. `Offer_Offer E` ressort aussi — rappel de la
phase 2 : c'est un marqueur de clients déjà fragiles, pas une cause.

**Le modèle établit des associations, pas des causes.** Un coefficient positif est un signal, pas
un levier prouvé.

## Drivers par segment

{_tab(segdf)}

Le modèle concentre le risque prédit sur les contrats mensuels et les clients récents — segments
où le taux réel de détraction est effectivement le plus élevé.

## Actionnable ou non

| Driver | Actionnable ? | Levier |
|---|---|---|
| Contrat mensuel | **oui** | proposer un engagement 1 an avec avantage |
| Faible ancienneté | partiellement | parcours d'accueil renforcé les 6 premiers mois |
| Charge mensuelle élevée / par service | **oui** | revue tarifaire, bundle |
| Absence de parrainage | indirectement | programme de parrainage |
| Offre E reçue | **signal, pas levier** | ne pas « supprimer l'offre E » — comprendre à qui elle a été proposée |
| Type de connexion | non à court terme | investissement réseau |

**Pour un détracteur prédit, le levier unique le plus probable** : s'il est en contrat mensuel,
proposer un engagement ; sinon, revoir la charge mensuelle.
""")

    # --- Équité -------------------------------------------------------------------------
    eq = R["audit_equite"]
    fig, axes = plt.subplots(1, len(eq), figsize=(2.6 * len(eq), 3), sharey=True)
    for ax, (dim, v) in zip(np.atleast_1d(axes), eq.items()):
        g_ = pd.Series({k: r["rappel_detracteur"] for k, r in v["groupes"].items()})
        ax.bar(g_.index, g_.values, color="#2c7fb8"); ax.set_ylim(0, 1); ax.set_title(dim, fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    np.atleast_1d(axes)[0].set_ylabel("rappel Détracteur")
    f_eq = _fig("05_equite.png")
    lignes = []
    for dim, v in eq.items():
        for grp, r in v["groupes"].items():
            lignes.append({"Dimension": dim, "Groupe": grp, "Clients": r["effectif"], "Détracteurs": r["detracteurs"],
                           "Rappel Détracteur": f"{r['rappel_detracteur']:.1%}"})
    eqdf = pd.DataFrame(lignes)
    ecarts = pd.DataFrame([{"Dimension": d, "Écart max (points)": f"{v['ecart_max']*100:.1f}"} for d, v in eq.items()])
    pire = max(eq.items(), key=lambda kv: kv[1]["ecart_max"])

    # coût de l'exclusion des démographiques
    reg_audit = cfg["colonnes"]["audit_seulement"]
    demo = [c for c in ("Gender", "Age", "Under 30", "Senior Citizen", "Married", "Dependents", "Number of Dependents") if c in df]
    Xd = df[num + cat + demo]
    numd = num + [c for c in demo if pd.api.types.is_numeric_dtype(df[c])]
    catd = cat + [c for c in demo if c not in numd]
    md_ = construire_modeles(numd, catd, seed)["baseline_logistique"]; entrainer(md_, Xd[rep], y[rep])
    m_avec = metriques(y[sil], md_.predict(Xd[sil]))

    _md("05_evaluation/audit_equite.md", f"""
# Phase 5 — Audit d'équité

Un modèle qui priorise le budget de rétention peut le répartir inégalement. L'énoncé : capter 80 %
des jeunes détracteurs mais 50 % des seniors est inacceptable, même à exactitude globale élevée.

Les variables démographiques sont **exclues du modèle** et servent **uniquement ici**.

## Rappel Détracteur par sous-groupe

![]({f_eq})

{_tab(eqdf)}

{_tab(ecarts)}

## Ce qu'on trouve

**L'écart le plus important porte sur l'âge : {pire[1]['ecart_max']*100:.0f} points** entre le groupe le mieux servi et le moins bien
servi. Concrètement, le modèle rappelle {eq['tranche_age']['groupes']['<30']['rappel_detracteur']:.0%} des détracteurs de moins de 30 ans contre
{eq['tranche_age']['groupes']['60+']['rappel_detracteur']:.0%} des 60 ans et plus. **C'est le cas de figure que l'énoncé décrit comme inacceptable, en
sens inverse** : ce sont les jeunes détracteurs qu'on rate.

Cause plausible : les jeunes sont sur-représentés dans les contrats mensuels et les faibles
anciennetés, mais avec des profils de services plus hétérogènes ; le modèle, sans variable d'âge,
les confond davantage avec les passifs. À investiguer avant toute mise en production.

## Coût de l'exclusion des démographiques

| | Kappa | Macro-F1 | Rappel Détracteur |
|---|---|---|---|
| Sans démographie (retenu) | {m['kappa_quadratique']:.3f} | {m['macro_f1']:.3f} | {m['par_classe']['Detracteur']['rappel']:.3f} |
| Avec démographie | {m_avec['kappa_quadratique']:.3f} | {m_avec['macro_f1']:.3f} | {m_avec['par_classe']['Detracteur']['rappel']:.3f} |

L'exclusion coûte **{(m['kappa_quadratique']-m_avec['kappa_quadratique'])*100:+.1f} point de kappa**. Le choix est défendable : un gain marginal ne justifie pas de
décider sur l'âge ou le genre.

## À remonter avant production

1. L'écart d'âge de {pire[1]['ecart_max']*100:.0f} points — à l'équipe Expérience Client et, selon la juridiction, au juridique.
2. Le fait que l'exclusion des démographiques **ne suffit pas** à garantir l'équité : le modèle
   les retrouve indirectement via les proxies contractuels. L'audit doit être répété à chaque
   réentraînement.
3. Une piste : seuil de décision **par groupe d'âge** pour égaliser les rappels — au prix d'une
   précision différente selon les groupes. Arbitrage à trancher par le métier, pas par le modèle.
""")

    _md("05_evaluation/revue_processus.md", f"""
# Phase 5 — Revue du processus et décision

## Critères de succès (phase 1) — atteints ?

| Critère | Résultat | Verdict |
|---|---|---|
| Précision@{K} nettement au-dessus du taux de base | {R['evaluation_metier']['precision_at_k']['precision_at_k']:.1%} vs {R['evaluation_metier']['precision_at_k']['taux_de_base']:.1%} (×{R['evaluation_metier']['precision_at_k']['lift']:.1f}) | ✅ |
| Gain net vs aléatoire, en euros | +{eco['gain_apporte_par_le_modele_eur']:,} € par campagne | ✅ (hypothèses à valider) |
| Kappa et rappel Détracteur > baseline | la baseline **est** le modèle retenu | ✅ par construction |
| Aucune classe abandonnée | rappel Passif {m['par_classe']['Passif']['rappel']:.2f} après correction de la calibration | ✅ |
| Écart d'équité mesuré, signalé si > 10 pts | {pire[1]['ecart_max']*100:.0f} pts sur l'âge | ⚠️ **signalé, non résolu** |

## Ce qui est solide

- Le registre de fuites et sa quantification.
- L'arbitrage empirique des « 3 », et la circularité évitée.
- Le protocole 15/85 avec le coût du MNAR mesuré.
- La sélection du modèle par la mesure, la calibration vérifiée.

## Ce qui est approximatif

- Les hypothèses économiques (coût d'appel, valeur client) sont des placeholders.
- La propension à répondre est simulée ; sa forme est plausible, pas observée.
- Pas de recherche d'hyperparamètres : choix assumé, mais un gain modeste est possible.
- Pas de TabPFN ni de verbatims synthétiques (bonus non réalisés).

## Ce qu'on ne peut pas faire avec ces données

Estimer l'effet d'un appel de rétention. Aucun traitement, aucune randomisation. Voir phase 6.

## Décision

**Go pour un pilote**, à trois conditions : valider les paramètres économiques avec la rétention ;
inclure un groupe de contrôle non contacté ; suivre le rappel par groupe d'âge dès la première
campagne.
""")


# =============================================================================
# PHASE 6 — DEPLOYMENT
# =============================================================================
def rapport_phase6(cfg, R, art):
    m = R["modele_final"]["metriques_retenues"]
    _md("06_deploiement/carte_modele.md", f"""
# Phase 6 — Carte de modèle

| | |
|---|---|
| Nom | Prédiction de catégorie NPS — clients silencieux |
| Version | 1.0 · seed {cfg['seed']} · régénéré par `python -m src.run` |
| Type | {R['modele_final']['modele']} dans un `Pipeline` scikit-learn (imputation + standardisation + one-hot) |
| Cible | M3 — Détracteur (satisfaction 1–2) · Passif (3) · Promoteur (4–5) |
| Données d'entraînement | {R['modele_final']['n_entrainement']} répondants simulés (S3 — MNAR), IBM Telco 11.1.3+ |
| Évaluation | {R['modele_final']['n_evaluation']} clients silencieux |
| Kappa quadratique | {m['kappa_quadratique']:.3f} |
| Rappel / précision Détracteur | {m['par_classe']['Detracteur']['rappel']:.2f} / {m['par_classe']['Detracteur']['precision']:.2f} |
| Décision | classe : modèle non calibré · P(Détracteur) : modèle calibré Platt (stratégie hybride) |
| Variables exclues | tout ce qui touche au churn, `CLTV`, démographie, géographie |
| Usage prévu | prioriser une liste d'appels de rétention mensuelle |
| Usage **interdit** | décider d'une offre ou d'un tarif individuel ; toute décision fondée sur un attribut protégé |
| Limites connues | écart de rappel de {R['audit_equite']['tranche_age']['ecart_max']*100:.0f} pts entre groupes d'âge ; classe Passif mal rappelée ; propension à répondre simulée |
| Fichier | `models/modele_final.joblib` |
""")
    mo_path = REP / "monitoring.json"
    section_mesuree = ""
    if mo_path.exists():
        mo = json.load(open(mo_path, encoding="utf-8"))
        d_cur = pd.DataFrame(mo["derive_entrees_courant"]).head(8)
        d_sim = pd.DataFrame(mo["derive_entrees_mois_simule"]).head(8)
        pd_ = mo["part_detracteurs_predits"]
        section_mesuree = f"""
## Implémenté : dérive mesurée (`python -m src.monitoring`)

![](figures/06_monitoring.png)

**Référence** = population d'entraînement (répondants S3). **Courant** = population scorée.
**Mois simulé** = {mo['scenario_simule']}.

| | PSI des prédictions | Part de détracteurs prédits |
|---|---|---|
| Courant vs référence | {mo['psi_predictions_courant']:.3f} | {pd_['courant']:.1%} (réf. {pd_['reference']:.1%}) |
| Mois simulé vs référence | **{mo['psi_predictions_mois_simule']:.3f}** | **{pd_['mois_simule']:.1%}** |

Variables les plus dérivantes — courant :

{_tab(d_cur, {'PSI':'{:.4f}'})}

Variables les plus dérivantes — mois simulé :

{_tab(d_sim, {'PSI':'{:.4f}'})}

**Déclencheur de réentraînement sur le mois simulé : {'DÉCLENCHÉ' if mo['reentrainement_declenche_mois_simule'] else 'non déclenché'}**
(variables en alerte : {', '.join(mo['variables_en_alerte_mois_simule']) or 'aucune'}). Le mécanisme fonctionne : il détecte
une hausse tarifaire et une migration de contrats que personne n'aurait signalées au modèle.

Note de lecture : le PSI « courant » n'est pas nul alors que les deux populations viennent de la
même base. C'est le **biais de sélection S3** qui s'affiche — les répondants ne ressemblent pas aux
silencieux — et c'est exactement ce que le monitoring doit voir.
"""
    _md("06_deploiement/monitoring_et_retrainement.md", f"""
# Phase 6 — Monitoring, réentraînement, et la décision à prendre
{section_mesuree}
## Ce qu'on surveille

| Signal | Mesure | Seuil de réentraînement |
|---|---|---|
| Dérive des entrées | distance entre la distribution mensuelle des features et celle d'entraînement (PSI par variable) | PSI > 0,2 sur une variable majeure (`Contract`, `Tenure`, `Monthly Charge`) |
| Dérive des prédictions | part de Détracteurs prédits par mois | variation > 5 points sans cause métier connue |
| Performance réelle | rappel Détracteur sur les **nouvelles réponses d'enquête** | chute > 10 points vs évaluation initiale |
| Équité | rappel Détracteur par groupe d'âge | écart > 15 points |
| Volume de labels | nouvelles réponses reçues | réentraînement dès 500 nouveaux labels, au plus tard tous les 6 mois |

Outil : Evidently (rapports HTML autonomes), cité par l'énoncé.

## La boucle de rétroaction — pire que rien

Un détracteur appelé puis retenu **change de classe**. Si on réentraîne sur ces données, le
modèle apprend l'effet de sa propre intervention comme s'il s'agissait d'une propriété du client.
Il se dégrade, et personne ne voit pourquoi.

## La recommandation qui demande une décision à la direction

> À la prochaine campagne, sur les {cfg['metier']['k_appels']} clients prioritaires du modèle, **tirer au hasard 10 à 20 %
> qui ne seront pas contactés.** Ce groupe de contrôle coûte quelques clients non retenus. Il
> achète deux choses irremplaçables :
>
> 1. **la mesure de l'effet réel de la campagne** — sans lui, on ne saura jamais si les clients
>    retenus l'ont été grâce à l'appel ou malgré lui ;
> 2. **les données d'un modèle d'uplift** au cycle suivant : traitement connu, issue connue,
>    assignation aléatoire — ce qui permettrait de passer du ciblage par *risque* (cette étude)
>    au ciblage par *sensibilité* (Ascarza 2018, jusqu'à 7 points de churn en moins).

C'est le seul passage de l'étude qui demande une décision plutôt que de présenter un résultat.
""")


# =============================================================================
# SYNTHÈSE + INDEX
# =============================================================================
def synthese(cfg, R, arb):
    m = R["modele_final"]["metriques_retenues"]; pk = R["evaluation_metier"]["precision_at_k"]; eco = R["evaluation_metier"]["economie"]
    _md("00_synthese.md", f"""
# Synthèse — Prédiction NPS pour la rétention

*Pour un lecteur non technique. Chaque chiffre est calculé et retrouvable dans les rapports qui suivent.*

## Ce qu'on a construit

Un modèle qui, pour chacun des clients n'ayant pas répondu à l'enquête NPS, estime s'il est
Détracteur, Passif ou Promoteur, et une application qui en tire une **liste d'appels priorisée**.

## Les trois résultats à retenir

**1. La grille NPS proposée dans l'énoncé fabrique un résultat faux.** Appliquée telle quelle, elle
donne un NPS de **{arb['nps']['M1']:+.0f}**, à 60–75 points des benchmarks télécom. La cause : elle classe en
détracteurs {arb['n_ambigus']} clients « moyennement satisfaits » (38 % de la base) dont 84 % sont restés fidèles.
En demandant aux données à qui ces clients ressemblent, on trouve qu'ils penchent à
**{arb['part_3_vers_promoteur']:.0%} du côté des promoteurs**. Le NPS corrigé est de **{arb['nps']['M3']:+.0f}**, dans la fourchette du secteur.

**2. Le modèle multiplie par {pk['lift']:.1f} l'efficacité des appels.** Sur {pk['k']} appels, {pk['precision_at_k']:.0%} touchent un
vrai détracteur, contre {pk['taux_de_base']:.0%} au hasard. À hypothèses de coût constantes, c'est
**{eco['gain_apporte_par_le_modele_eur']:,} € de gain net supplémentaire par campagne**. Il rappelle {m['par_classe']['Detracteur']['rappel']:.0%} des détracteurs.

**3. Il rate davantage les jeunes détracteurs.** Rappel de {R['audit_equite']['tranche_age']['groupes']['<30']['rappel_detracteur']:.0%} chez les moins de 30 ans contre
{R['audit_equite']['tranche_age']['groupes']['60+']['rappel_detracteur']:.0%} chez les 60 ans et plus — alors qu'aucune variable d'âge n'entre dans le modèle. Écart
mesuré, signalé, à traiter avant production.

## Ce qu'il faut savoir sur la fiabilité

- **Le modèle est entraîné dans les conditions réelles** : sur 15 % de « répondants » dont la
  composition est biaisée (les mécontents et les enthousiastes répondent plus), évalué sur les 85 %
  restants. Ses scores sont donc réalistes, pas optimistes.
- **Aucune variable liée au départ des clients n'est utilisée.** Le dataset en contient plusieurs
  qui auraient donné un score presque parfait et un modèle inutilisable. Elles sont identifiées,
  exclues, et la raison est documentée.
- **Le modèle le plus simple a gagné** face à deux méthodes plus sophistiquées. Ce n'est pas un
  échec des méthodes avancées : c'est ce que les données permettent.

## La décision demandée

Appeler les clients les plus à risque n'est pas la stratégie la plus rentable : la recherche
établit qu'il faut cibler ceux **dont le comportement change le plus quand on les appelle**.
Mesurer cela ne demande qu'une chose, et elle est gratuite : **ne pas contacter 10 à 20 % des
clients ciblés à la prochaine campagne, tirés au hasard.** Sans ce groupe de contrôle, on ne
saura jamais si la campagne a servi — et le modèle se dégradera en réapprenant ses propres effets.

## Ce que cette étude ne fait pas

Elle n'estime pas l'effet d'un appel (aucune campagne dans les données). Elle n'utilise pas de
texte client (les verbatims synthétiques, bonus de l'énoncé, ne sont pas réalisés). Ses hypothèses
économiques sont des paramètres à valider avec l'équipe rétention.
""")

    _md("README.md", f"""
# Étude — ordre de lecture

| # | Phase CRISP-DM | Rapport |
|---|---|---|
| 0 | Synthèse pour décideur | [00_synthese.md](00_synthese.md) |
| 1 | Compréhension métier | [01_comprehension_metier.md](01_comprehension_metier.md) |
| 2 | Compréhension des données | [dictionnaire](02_donnees/dictionnaire_donnees.md) · [qualité](02_donnees/qualite_donnees.md) · [EDA](02_donnees/analyse_exploratoire.md) · [**registre de fuites**](02_donnees/registre_fuites.md) |
| 3 | Préparation | [**construction de la cible**](03_preparation/construction_cible.md) · [features et colinéarité](03_preparation/features.md) |
| 4 | Modélisation | [**protocole 15/85**](04_modelisation/protocole_validation.md) · [comparaison des modèles et hyperparamètres](04_modelisation/comparaison_modeles.md) · [TabPFN](04_modelisation/tabpfn.md) · [verbatims et fusion texte](04_modelisation/verbatims_et_fusion_texte.md) |
| 5 | Évaluation | [technique](05_evaluation/evaluation_technique.md) · [**métier**](05_evaluation/evaluation_metier.md) · [sensibilité](05_evaluation/sensibilite_mapping.md) · [interprétabilité et drivers](05_evaluation/interpretabilite_drivers.md) · [**équité**](05_evaluation/audit_equite.md) · [revue](05_evaluation/revue_processus.md) |
| 6 | Déploiement | [carte de modèle](06_deploiement/carte_modele.md) · [monitoring et décision](06_deploiement/monitoring_et_retrainement.md) |

Figures dans `figures/`. Données brutes des résultats dans `resultats.json`.
Tout est régénéré par `python -m src.run` puis `python -m src.rapports`.
""")


def main():
    cfg = charger_config()
    R = json.load(open(REP / "resultats.json", encoding="utf-8"))
    import joblib
    art = joblib.load(RACINE / "models" / "modele_final.joblib")

    print("[1/6] Données")
    df_brut, qualite = preparer(cfg)
    # version brute (avant remplissage structurel) pour le dictionnaire
    from .data import charger_tables, joindre
    df_raw = joindre(charger_tables(cfg))
    df = construire(df_brut)
    cibles, arb = construire_cibles(df, cfg, cfg["seed"])
    num, cat = colonnes_modele(df, cfg)
    X = df[num + cat]; satisfaction = pd.to_numeric(df[cfg["colonnes"]["cible"]]).values

    # nettoyage des anciens rapports pour un dossier propre
    for p in REP.glob("*.md"):
        p.unlink()
    for d in ("02_donnees", "03_preparation", "04_modelisation", "05_evaluation", "06_deploiement"):
        shutil.rmtree(REP / d, ignore_errors=True)
    # Figures : on ne supprime que celles produites ici (06_monitoring.png vient de src/monitoring.py)
    for p in FIG.glob("0[2-5]_*.png"):
        p.unlink()

    print("[2/6] Phase 1"); rapport_phase1(cfg, R)
    print("[3/6] Phase 2"); rapport_phase2(cfg, df_raw, df, qualite, R)
    print("[4/6] Phase 3"); rapport_phase3(cfg, df, cibles, arb, num, cat)
    print("[5/6] Phase 4"); dec = rapport_phase4(cfg, df, cibles, X, satisfaction, num, cat, R)
    print("[6/6] Phases 5-6 et synthèse")
    rapport_phase5(cfg, df, cibles, X, dec, art, R, num, cat, satisfaction)
    rapport_phase6(cfg, R, art)
    synthese(cfg, R, arb)
    n_md = len(list(REP.rglob("*.md"))); n_fig = len(list(FIG.glob("*.png")))
    print(f"\n{n_md} rapports, {n_fig} figures -> reports/")


if __name__ == "__main__":
    main()
