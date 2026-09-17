# Prédiction de la catégorie NPS — clients silencieux d'un opérateur télécom

Take-home challenge Data Science (Artefact). Prédire, pour les 85 % de clients qui ne répondent
pas à l'enquête NPS, s'ils sont Détracteur, Passif ou Promoteur — et en tirer une liste d'appels
de rétention priorisée.

## Lire l'étude

Commencer par **[reports/00_synthese.md](reports/00_synthese.md)** (2 pages, pour décideur), puis
suivre l'ordre CRISP-DM dans **[reports/README.md](reports/README.md)**.

Les documents de cadrage sont dans `docs/` : compréhension du problème, plan de travail en 38
étapes, état de l'art, angle différenciant.

## Reproduire

```bash
pip install -r requirements.txt
python -m src.run        # pipeline complet : données → cible → modèles → évaluation (~3 min)
python -m src.rapports   # génère les 19 rapports et 17 figures dans reports/
python -m streamlit run app/app.py --server.port 8501
```

Vérifier :

```bash
python -m pytest tests -q         # 17 tests : jointure, fuites, cible, protocole, métriques, robustesse de l'app
python -m src.monitoring         # dérive PSI et déclencheur de réentraînement
python -m src.writeup            # livrable/write_up.pdf
python scripts/captures.py       # captures des 10 pages (application lancée)
```

Puis ouvrir http://localhost:8501.

Tout est déterministe (seed dans `config.yaml`). Aucune clé d'API n'est nécessaire.

## Arborescence

```
config.yaml            paramètres, registre de fuites, mappings de cible, hypothèses métier
data/raw/              5 fichiers IBM Telco Customer Churn 11.1.3+ (non versionnés)
src/
  data.py              ingestion, jointure, qualité, registre de fuites
  features.py          variables dérivées, pipeline de préparation
  target.py            cible M1/M2/M3, arbitrage empirique des « 3 »
  protocol.py          protocole 15/85 — MCAR / MAR / MNAR, poids IPW
  models.py            baseline, ordinal à seuils, HistGradientBoosting, calibration
  evaluate.py          métriques par classe, précision@K, économie, équité
  run.py               orchestrateur
  rapports.py          génération de l'étude (Markdown + figures)
app/app.py             application Streamlit pour l'équipe rétention
models/                modèle final sérialisé (pipeline complet)
reports/               l'étude, ordonnée par phase CRISP-DM
docs/                  cadrage, plan, état de l'art
```

## Ce que ce projet fait de particulier

- **Registre de fuites quantifié** : le dataset contient des colonnes (`Churn Score`, `Churn Value`,
  `CLTV`…) qui donneraient un score presque parfait. Identifiées, mesurées, exclues.
- **Cible arbitrée par les données** : les 2 665 clients à satisfaction « 3 » sont classés par un
  modèle entraîné sur les seuls extrêmes — 72 % ressemblent à des promoteurs.
- **Validation dans les conditions réelles** : entraînement sur 15 % de répondants au biais de
  réponse simulé (MNAR), évaluation sur les 85 % restants.
- **Sélection par la mesure** : la régression logistique bat le boosting ; le modèle le plus
  simple est retenu.
- **Équité mesurée** : écart de rappel de 25 points entre groupes d'âge, signalé.
- **Limite de la règle de ciblage** : cibler par risque n'est pas optimal (Ascarza 2018) ; le
  protocole pour faire mieux est décrit.

## Usage d'outils d'IA

Le code, la structure de l'étude et la rédaction ont été produits avec l'assistance d'un
assistant IA (Claude), sous direction et relecture humaines. Aucun verbatim client synthétique
n'a été généré. Les choix de modélisation, les décisions de périmètre et les conclusions sont
assumés par l'auteur.
