# Prédiction de la catégorie NPS des clients silencieux

Un opérateur télécom n'obtient de réponse à son enquête NPS que de 15 % de ses clients. Ce projet
estime, pour les 85 % restants, s'ils sont Détracteur, Passif ou Promoteur, en tire une liste
d'appels priorisée pour l'équipe rétention, identifie les facteurs de détraction et met en place un
suivi.

## Lancer l'application

```bash
docker compose up
```

Puis ouvrir <http://localhost:8501>. Rien d'autre à installer, aucune donnée à télécharger, aucune
clé. L'image contient le modèle entraîné, les 7 043 clients scorés et les rapports.

Sans Docker :

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Les deux documents à lire

| Document | Contenu |
|---|---|
| [livrable/write_up.pdf](livrable/write_up.pdf) | 6 pages. L'approche, la construction de la cible, les décisions de modélisation, l'évaluation, les facteurs de détraction, les limites et la suite. C'est le document à lire en premier. |
| [livrable/etude_complete.pdf](livrable/etude_complete.pdf) | Le détail complet, une partie par phase CRISP-DM : données, cible, modèles, évaluation, déploiement, et la correspondance point par point avec l'énoncé. |

Les mêmes contenus au format Markdown sont dans [reports/](reports/), avec les figures et les
résultats bruts en JSON. Les captures de l'application sont dans [reports/captures/](reports/captures/).

## Ce que fait l'application

Huit pages, avec des filtres croisés valables partout : contrat, type d'accès, offre, classe
prédite, tranche d'âge, levier, ancienneté, facture, probabilité de détraction, recherche par
identifiant.

- **Synthèse** — les chiffres clés et les trois résultats.
- **Explorer les clients** — croiser des critères et lire aussitôt le NPS, le taux de détracteurs et
  la composition de la sélection ; export CSV.
- **Prioriser les appels** — la liste d'appels du mois, classée par risque, avec le levier et
  l'économie de la campagne.
- **Analyser un client** — choisir un client par critères, voir sa prédiction, ce qui pèse pour lui
  et les deux leviers à activer.
- **Données et cible**, **Modèles et performance**, **Drivers et équité**, **Suivi et méthode** — le
  raisonnement derrière les chiffres.

## Rejouer le calcul

Il faut d'abord les cinq classeurs IBM dans `data/raw/` (voir plus bas), puis :

```bash
pip install -r requirements.txt -r requirements-pipeline.txt
python -m src.run          # données, cible, 15 modèles, sélection, évaluation  (~35 min)
python -m src.verbatims    # verbatims synthétiques
python -m src.texte        # modèle texte et fusion
python -m src.monitoring   # dérive, nouvelles réponses, équité
python -m src.rapports     # l'étude et les figures
python -m src.writeup      # les deux PDF
python -m pytest tests -q  # 53 tests
```

Tout est déterministe : une seule graine, dans `config.yaml`. Aucune clé n'est nécessaire.

Avec Docker, `docker compose --profile pipeline run --rm pipeline` fait la même chose dans un
environnement figé.

## Données sources

`data/raw/` n'est pas versionné. Il faut y déposer les cinq classeurs du jeu *IBM Telco Customer
Churn*, version **11.1.3+** :

```
Telco_customer_churn_demographics.xlsx
Telco_customer_churn_location.xlsx
Telco_customer_churn_population.xlsx
Telco_customer_churn_services.xlsx
Telco_customer_churn_status.xlsx
```

Source : [IBM Accelerator Catalog](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

⚠️ Le fichier Kaggle le plus diffusé, `WA_Fn-UseC_-Telco-Customer-Churn.csv` du compte *blastchar*,
est la version 2018 : il n'a pas la colonne `Satisfaction Score`, qui est la base de la cible. Il ne
convient pas.

## Déployer en ligne

L'application est prête pour Streamlit Community Cloud : fichier principal `app/app.py`, Python 3.11,
aucun secret. `requirements.txt` ne contient que les dépendances de l'interface, pour rester sous la
limite de ressources de l'hébergement ; le reste est dans `requirements-pipeline.txt`.

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
reports/               l'étude, les figures, les captures, les résultats bruts
livrable/              write_up.pdf, etude_complete.pdf
notebooks/             notebook d'exploration, exécuté
tests/                 53 tests
```

## Usage d'outils d'IA

Le code, la structure de l'étude et la rédaction ont été produits avec l'assistance de Claude, sous
direction et relecture humaines. Les verbatims synthétiques ont été rédigés par ce même assistant,
et la source de chaque texte est tracée dans le fichier livré. Les choix de modélisation, les
décisions de périmètre et les conclusions sont assumés par l'auteur.
