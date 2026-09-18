"""Construit et exécute le notebook d'exploration (livrable § 6 de l'énoncé : « Notebooks, if used »).

    python scripts/notebook_eda.py

Le notebook rejoue, cellule par cellule, la compréhension des données et la construction de la
cible à partir des fonctions de `src/` — mêmes chiffres que l'étude, exécutés et sauvegardés avec
leurs sorties dans notebooks/01_exploration_et_cible.ipynb.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "notebooks" / "01_exploration_et_cible.ipynb"

CELLULES = [
    ("md", """# Exploration des données et construction de la cible NPS

Notebook d'accompagnement de l'étude (phases 2 et 3 de CRISP-DM). Il rejoue les analyses à partir des
fonctions de `src/` : les chiffres sont ceux des rapports `reports/02_comprehension_donnees.md` et
`reports/03_preparation_donnees.md`. Lecture recommandée : d'abord ces rapports, puis ce notebook pour
manipuler les données.

**Convention** : aucune constante recopiée — tout vient de `config.yaml` et des données brutes."""),
    ("code", """import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
RACINE = Path.cwd().resolve()
if not (RACINE / "src").exists():
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 160)
from src.data import charger_config, charger_tables, joindre, controler_qualite, preparer, registre_fuites
from src.features import construire, colonnes_modele, HYPOTHESES_DERIVEES
from src.target import construire_cibles, nps, ORDRE
from src.protocol import decouper, diagnostic
cfg = charger_config(); print("seed :", cfg["seed"])"""),
    ("md", "## 1. Sources et jointure\n\nCinq tables, jointure interne sur `Customer ID`, population rattachée par `Zip Code`. Porte de contrôle : 7 043 lignes."),
    ("code", """tables = charger_tables(cfg)
for k, t in tables.items():
    print(f"{k:14s} {t.shape}")
df_raw = joindre(tables)
print("\\nAprès jointure :", df_raw.shape)
df_raw.head(3).T"""),
    ("md", "## 2. Qualité — cinq contrôles\n\nManquants structurels (le vide a un sens), doublons, aberrantes (conservées), cohérence interne, colonnes constantes."),
    ("code", """q = controler_qualite(df_raw)
print("Manquants :", q["manquants"])
print("Doublons id :", q["doublons_id"], "| doublons lignes :", q["doublons_lignes"])
print("Aberrantes (IQR x 1,5) :", q["aberrantes"])
print("Ancienneté nulle :", q["anciennete_zero"], "| incohérences facturation :", q["incoherence_facturation"])
print("Constantes :", q["constantes"])"""),
    ("md", "## 3. La cible brute : `Satisfaction Score`\n\nLe fait fondateur du projet : satisfaction 1–2 → 100 % de départs, 4–5 → 0 %. La note 3 est la seule zone d'incertitude."),
    ("code", """cible = cfg["colonnes"]["cible"]
dist = df_raw[cible].value_counts().sort_index()
ct = pd.crosstab(df_raw[cible], df_raw["Churn Label"])
ct["% départ"] = (ct["Yes"] / ct.sum(axis=1) * 100).round(1)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].bar(dist.index.astype(str), dist.values, color="#2c7fb8"); axes[0].set_title("Distribution de Satisfaction Score")
for i, v in enumerate(dist.values):
    axes[0].text(i, v + 30, f"{v}\\n{v/len(df_raw):.1%}", ha="center", fontsize=8)
(pd.crosstab(df_raw[cible], df_raw["Churn Label"], normalize="index") * 100).plot(kind="bar", stacked=True, ax=axes[1], color=["#27ae60", "#c0392b"], rot=0)
axes[1].set_title("% restés / partis par note"); plt.tight_layout(); plt.show()
ct"""),
    ("md", """## 4. Registre de fuites

Toute colonne liée au churn est une fuite (conséquence, disponibilité, modèle amont). On le **mesure** par
information mutuelle plutôt que de l'affirmer."""),
    ("code", """from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
reg = registre_fuites(cfg)
for k, v in reg.items():
    print(f"{k:22s} {v}")
df, _ = preparer(cfg)
cand = ["Churn Value", "Churn Score", "CLTV", "Customer Status", "Contract", "Tenure in Months", "Monthly Charge", "Number of Referrals", "Offer", "Internet Type"]
Xmi = pd.DataFrame({c: LabelEncoder().fit_transform(df[c].astype(str)) if df[c].dtype == object else df[c].fillna(df[c].median()) for c in cand})
mi = pd.Series(mutual_info_classif(Xmi, df[cible], random_state=cfg["seed"]), index=cand).sort_values(ascending=False)
mi.round(3)"""),
    ("md", "## 5. La quasi-expérience `Offer`\n\nLes clients ayant reçu l'offre E partent deux fois plus. Biais d'indication : l'offre a été proposée à des clients déjà fragiles. Aucune lecture causale."),
    ("code", """df.groupby("Offer").agg(clients=("Customer ID", "count"), taux_depart=("Churn Value", "mean"), satisfaction=(cible, "mean")).round(3).sort_values("taux_depart")"""),
    ("md", "## 6. Variables explicatives et dérivées\n\nChaque variable dérivée est adossée à une hypothèse métier (`src/features.py`)."),
    ("code", """df = construire(df)
num, cat = colonnes_modele(df, cfg)
print(len(num), "numériques :", num)
print(len(cat), "catégorielles :", cat)
pd.DataFrame([{"variable": k, "hypothèse": v} for k, v in HYPOTHESES_DERIVEES.items()])"""),
    ("md", """## 7. Construction de la cible : trois mappings et arbitrage empirique des « 3 »

Le mapping de l'énoncé (M1) donne un NPS très négatif parce qu'il envoie les 2 665 clients à « 3 » chez les
détracteurs. Un modèle entraîné sur les seuls extrêmes (1–2 vs 4–5) dit à qui ils ressemblent."""),
    ("code", """cibles, arb = construire_cibles(df, cfg, cfg["seed"])
print({m: nps(cibles[m]) for m in ("M1", "M2", "M3")})
print("Extrêmes :", arb["n_extremes"], "| ambigus :", arb["n_ambigus"], "| exactitude sur extrêmes (CV) :", arb["exactitude_sur_extremes"])
print("Part des 3 penchant détracteur :", arb["part_3_vers_detracteur"], "-> le bloc rejoint :", arb["destination_des_3"])
pd.DataFrame({m: cibles[m].value_counts(normalize=True).reindex(ORDRE).round(3) for m in ("M1", "M2", "M3")})"""),
    ("md", "## 8. Le protocole 15 / 85 : trois mécanismes de non-réponse\n\nSous S3 (MNAR), les extrêmes répondent plus : le modèle apprend sur une population qui ne ressemble pas à celle qu'il scorera."),
    ("code", """pd.DataFrame([diagnostic(df, cfg, decouper(df, cfg, sc, cfg["seed"])) for sc in cfg["protocole"]["scenarios"]])"""),
    ("md", """## Suite

La modélisation (quinze modèles, sélection sans regarder le test), l'évaluation et le déploiement sont
exécutés par `python -m src.run` et documentés dans `reports/04_modelisation.md` et suivants. Ce notebook
s'arrête volontairement là : le pipeline est la source de vérité, le notebook une fenêtre dessus."""),
]


def main():
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for genre, src in CELLULES:
        nb.cells.append(new_markdown_cell(src) if genre == "md" else new_code_cell(src))
    SORTIE.parent.mkdir(exist_ok=True)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(RACINE)}})
    client.execute()
    nbformat.write(nb, SORTIE)
    n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"{SORTIE.relative_to(RACINE)} — {len(nb.cells)} cellules, {n_out} cellules de code avec sorties")


if __name__ == "__main__":
    main()
