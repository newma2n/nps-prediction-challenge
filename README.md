# Prédiction de la catégorie NPS des clients silencieux

Seuls 15 % des clients répondent à l'enquête NPS. Ce projet estime la catégorie NPS des 85 % qui ne
répondent pas, en tire une liste d'appels priorisée, identifie les facteurs de détraction et met en
place un suivi.

## Application

<https://nps-prediction-challenge.streamlit.app/>

En local, avec Docker :

```bash
docker compose up          # puis http://localhost:8501
```

Ou avec Python 3.11 ou 3.12 :

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Documents

- [livrable/write_up.pdf](livrable/write_up.pdf) — 6 pages : approche, construction de la cible,
  modélisation, évaluation, drivers, limites, suites.
- [livrable/etude_complete.pdf](livrable/etude_complete.pdf) — le détail par phase CRISP-DM et la
  correspondance avec l'énoncé.

Versions Markdown, figures et résultats bruts dans [reports/](reports/). Captures de l'application
dans [reports/captures/](reports/captures/).

## Pages de l'application

Filtres croisés valables partout : contrat, accès, offre, classe prédite, âge, levier, ancienneté,
facture, probabilité, recherche par identifiant.

| Page | Usage |
|---|---|
| Synthèse | les chiffres clés |
| Explorer les clients | croiser des critères, lire le NPS et la composition de la sélection, exporter |
| Prioriser les appels | la liste du mois, classée par risque, avec le levier et l'économie |
| Analyser un client | choisir un client par critères, voir sa prédiction et ses leviers |
| Données et cible · Modèles et performance · Drivers et équité · Suivi et méthode | le raisonnement derrière les chiffres |

## Rejouer le calcul

Déposer les cinq classeurs IBM dans `data/raw/` (voir ci-dessous), puis :

```bash
pip install -r requirements.txt -r requirements-pipeline.txt
python -m src.run          # données, cible, 15 modèles, sélection, évaluation  (~35 min)
python -m src.verbatims
python -m src.texte
python -m src.monitoring
python -m src.rapports     # l'étude et les figures
python -m src.writeup      # les deux PDF
python -m pytest tests -q  # 53 tests
```

Une seule graine, dans `config.yaml`. Aucune clé nécessaire. Avec Docker :
`docker compose --profile pipeline run --rm pipeline`.

## Données sources

`data/raw/` n'est pas versionné. Y déposer les cinq classeurs du jeu *IBM Telco Customer Churn*,
version 11.1.3+ :

```
Telco_customer_churn_demographics.xlsx
Telco_customer_churn_location.xlsx
Telco_customer_churn_population.xlsx
Telco_customer_churn_services.xlsx
Telco_customer_churn_status.xlsx
```

Source : [IBM Accelerator Catalog](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

## Arborescence

```
app/app.py             application Streamlit
src/                   le pipeline, un module par étape
  data.py              ingestion, jointure, qualité, registre de fuites
  features.py          variables retenues et leurs hypothèses
  target.py            construction de la cible, arbitrage des notes « 3 »
  protocol.py          protocole 15/85, scénarios de non-réponse
  models.py            catalogue de 15 modèles, réglage, calibration
  evaluate.py          métriques, précision@K, économie, équité, quantification
  explication.py       contributions individuelles à la détraction
  run.py               orchestrateur
  verbatims.py         verbatims synthétiques
  texte.py             modèle texte et fusion
  monitoring.py        dérive, nouvelles réponses, déclencheurs
  rapports.py          génération de l'étude
  writeup.py           génération des PDF
data/processed/        clients_scores.parquet, verbatims.parquet
models/                modèle entraîné
reports/               étude, figures, captures, résultats bruts
livrable/              write_up.pdf, etude_complete.pdf
notebooks/             notebook d'exploration, exécuté
tests/                 53 tests
```

## Usage d'outils d'IA

Le code et la rédaction ont été produits avec l'assistance de Claude, sous direction et relecture
humaines. Les verbatims synthétiques ont été rédigés par ce même assistant, et la source de chaque
texte est tracée dans le fichier livré. Les choix de modélisation, le périmètre et les conclusions
sont assumés par l'auteur.
