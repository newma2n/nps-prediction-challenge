# Prédiction de la catégorie NPS — clients silencieux d'un opérateur télécom

Take-home challenge Data Science (Artefact). Prédire, pour les 85 % de clients qui ne répondent pas à
l'enquête NPS, s'ils sont Détracteur, Passif ou Promoteur ; en tirer une liste d'appels de rétention
priorisée, les drivers de détraction et une estimation du NPS des silencieux.

## Lire

| Pour | Document |
|---|---|
| Présenter devant un jury | **[livrable/soutenance.pdf](livrable/soutenance.pdf)** (22 diapositives, CRISP-DM) + [notes d'oral et questions anticipées](livrable/soutenance_notes.md) |
| Décider en 10 minutes | **[livrable/write_up.pdf](livrable/write_up.pdf)** (6 pages) ou [reports/00_synthese.md](reports/00_synthese.md) |
| Vérifier que rien de l'énoncé n'a été oublié | [reports/07_conformite_enonce.md](reports/07_conformite_enonce.md) |
| Suivre la méthode, phase par phase | [reports/README.md](reports/README.md) → un document par phase CRISP-DM |
| Tout avoir en un PDF | [livrable/etude_complete.pdf](livrable/etude_complete.pdf) |
| Manipuler les données | [notebooks/01_exploration_et_cible.ipynb](notebooks/01_exploration_et_cible.ipynb) (exécuté) |
| Tester l'outil | **`docker compose up`** → http://localhost:8501 (8 pages, filtres croisés) ; sans Docker : `streamlit run app/app.py` ; captures dans `reports/captures/` |

Cadrage : `docs/` (compréhension du problème, plan en 38 étapes avec état d'avancement, état de l'art).

## Lancer en une commande (Docker)

C'est le chemin recommandé : aucune installation Python, aucune donnée à télécharger, aucune clé.

```bash
docker compose up --build        # première fois : ~5 min de construction
```

Puis ouvrir **<http://localhost:8501>**. Pour arrêter : `docker compose down`.

L'image embarque le modèle entraîné, le jeu de données dérivé (7 043 clients scorés), les rapports,
les figures et les PDF du livrable. Elle pèse environ 800 Mo et ne contient **ni** les fichiers
bruts **ni** aucun secret. Le conteneur tourne sans privilèges et expose une sonde de santé
(`/_stcore/health`), que `docker compose ps` affiche comme `healthy` lorsque l'application est prête.

**Rejouer tout le calcul depuis les fichiers bruts** — à réserver à qui veut vérifier la
reproductibilité, car il faut d'abord déposer les cinq classeurs IBM dans `data/raw/` (voir la
section suivante) et compter une trentaine de minutes :

```bash
docker compose --profile pipeline run --rm pipeline          # données → cible → 15 modèles → évaluation
docker compose --profile pipeline run --rm pipeline python -m src.rapports    # régénère l'étude
docker compose --profile pipeline run --rm pytest            # les tests d'invariants
docker compose up -d --build app                             # recharger l'application
```

Les sorties reviennent sur la machine hôte (`models/`, `data/processed/`, `reports/`, `livrable/`)
grâce aux montages déclarés dans `docker-compose.yml`.

⚠️ Ces trois commandes-là **exigent les fichiers bruts** dans `data/raw/`, y compris les tests : ils
vérifient la chaîne depuis la source (jointure sans perte, exclusion des fuites, construction de la
cible), ce qui n'a aucun sens sans elle. Sans les fichiers bruts, `docker compose up` reste
disponible et montre l'intégralité du travail.

> **Pourquoi deux images.** L'étage `app` ne contient que ce dont l'interface a besoin ; l'étage
> `pipeline` ajoute le catalogue de modèles (XGBoost, LightGBM, CatBoost, SHAP, notebook, tests).
> Mettre les modèles de fondation — PyTorch, TabICL, TabPFN — dans l'image de livraison la ferait
> passer de 0,8 à plus de 3 Go sans rien apporter à ce qui s'affiche : leurs résultats sont déjà
> calculés et lus depuis `reports/fondation.json`. Pour les réexécuter, c'est `pip install -r
> requirements.txt` hors conteneur, puis `python -m src.fondation_eval`.

## Déployer l'application en ligne

L'application est prête pour **Streamlit Community Cloud** : dépôt GitHub, fichier principal
`app/app.py`, Python 3.11. Aucun secret n'est nécessaire.

`requirements.txt` ne contient que les dépendances de l'interface, volontairement : c'est ce fichier
qu'installe l'hébergeur, et y laisser PyTorch ou les modèles de fondation ferait dépasser la limite
de ressources pour des paquets dont l'application n'a aucun besoin. Le modèle entraîné, le jeu de
données dérivé et les rapports sont versionnés, donc l'application fonctionne sans recalcul et sans
fichier brut. Pour rejouer le calcul, ajouter `requirements-pipeline.txt`.

## Données sources

Le dépôt ne contient aucune donnée : `data/raw/` est ignoré par git. Pour rejouer le pipeline, il
faut y déposer les **cinq classeurs** du jeu *IBM Telco Customer Churn*, version **11.1.3+** :

```
data/raw/Telco_customer_churn_demographics.xlsx     7 043 lignes — âge, genre, situation familiale
data/raw/Telco_customer_churn_location.xlsx         7 043 lignes — code postal, latitude, longitude
data/raw/Telco_customer_churn_population.xlsx       1 671 lignes — population par code postal
data/raw/Telco_customer_churn_services.xlsx         7 043 lignes — contrat, services, facturation
data/raw/Telco_customer_churn_status.xlsx           7 043 lignes — Satisfaction Score, statut, CLTV
```

Source : IBM Accelerator Catalog, *Telco customer churn (11.1.3+)*
<https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113>.

⚠️ **Le fichier Kaggle le plus courant ne convient pas.** Le `WA_Fn-UseC_-Telco-Customer-Churn.csv`
diffusé sous le compte *blastchar* est la version 2018 : elle n'a ni `Satisfaction Score`, ni
`Customer Status`, ni les tables de démographie et de localisation. Or `Satisfaction Score` **est**
la variable cible de ce travail : sans elle, rien ne se reconstruit.

Contrôles automatiques à l'ingestion (`src/data.py`) : 7 043 lignes après jointure, aucun client
sans ligne de services, de statut ou de localisation, et aucune colonne témoin manquante. Une
version différente fait échouer le pipeline au lieu de produire des chiffres faux.

## Reproduire

```bash
pip install -r requirements.txt -r requirements-pipeline.txt   # numpy < 2 impératif
python -m src.run                        # données → cible → 15 modèles → sélection → évaluation (~20 min)
python -m src.fondation_eval             # TabICL, TabPFN v2 (et TabPFN 2.5 si TABPFN_TOKEN)
python -m src.verbatims                  # verbatims synthétiques (API Anthropic si clé dans .env, sinon générateur local seedé)
python -m src.texte                      # modèle texte et fusion
python -m src.monitoring                 # dérive PSI, nouvelles réponses, déclencheurs
python -m src.rapports                   # l'étude : 8 documents, ~22 figures dans reports/
python -m src.writeup                    # livrable/write_up.pdf et livrable/etude_complete.pdf
python -m src.soutenance                 # livrable/soutenance.pdf (22 diapos) + notes d'oral
python scripts/notebook_eda.py           # notebook exécuté
python -m pytest tests -q                # tests d'invariants (données, fuites, cible, protocole, 15 modèles, sélection, app)
python -m streamlit run app/app.py       # puis python scripts/captures.py pour les captures
```

Tout est déterministe (seed dans `config.yaml`). Aucune clé d'API n'est nécessaire (`.env.example`).

**Modèles de fondation.** TabICL et **TabPFN v2** s'exécutent sans condition : leurs poids sont
publics sur Hugging Face, et TabPFN v2 est chargé par `model_path` explicite. **TabPFN 2.5** (dépôt
`Prior-Labs/tabpfn_3_5` — la numérotation du dépôt suit la génération interne, c'est bien la 2.5)
est public lui aussi, mais son téléchargement passe par une acceptation de licence PriorLabs :
compte sur ux.priorlabs.ai, puis `TABPFN_TOKEN` dans `.env` (jamais dans le dépôt). Sans ce jeton,
la 2.5 est déclarée indisponible avec la cause exacte, et les deux autres sont évalués.

## Ce que ce projet fait de particulier

- **Registre de fuites quantifié** : `Churn Score`, `Churn Value`, `Customer Status`, `CLTV`… donneraient
  un score presque parfait. Trois natures de fuite nommées, information mutuelle mesurée, exclusion
  testée, double diagnostic après modélisation.
- **Cible arbitrée par les données** : la grille de l'énoncé donne un NPS de −42 ; les 2 665 clients à
  satisfaction « 3 » sont classés par un modèle entraîné sur les seuls extrêmes (72 % penchent
  promoteur) — sans étiquetage individuel, pour éviter la circularité.
- **Validation dans les conditions réelles** : 15 % de répondants au biais de non-réponse simulé
  (MNAR), évaluation sur les 85 % silencieux dont on connaît la vérité ; le coût du biais est mesuré.
- **Quinze modèles, sept familles, trois formulations**, chacun avec la raison de sa présence et
  l'hypothèse qu'il teste ; **sélection sans regarder le test** (CV sur les répondants, pondérée IPW,
  règle de parcimonie), puis test final.
- **NPS des silencieux simulé** avec intervalle, comparé au NPS des répondants et à la vérité.
- **Robustesse** : bruit d'étiquettes, ablation de features (coût chiffré de l'exclusion des
  démographiques), sensibilité au mapping, cinq stratégies de déséquilibre.
- **Équité** : rappel et taux de sélection par sous-groupe, **audit des proxies** (les features
  reconstruisent l'âge, AUC > 0,9), mitigation par seuils chiffrée et remontée au métier.
- **Drivers par segment et levier recommandé** pour chaque client ; **monitoring** implémenté (PSI,
  nouvelles réponses, équité suivie mois par mois, cinq déclencheurs) ; **groupe de contrôle** recommandé.

## Arborescence

```
Dockerfile             image de livraison (étage app) et image de calcul (étage pipeline)
docker-compose.yml     `docker compose up` → l'application ; profil `pipeline` → tout recalculer
docker/                requirements séparés : application seule / pipeline complet
config.yaml            paramètres, registre de fuites, mappings de cible, hypothèses métier
data/raw/              5 fichiers IBM Telco Customer Churn 11.1.3+ (non versionnés)
data/processed/        clients_scores.parquet (prédictions, leviers), verbatims.parquet
src/
  data.py              ingestion, jointure, qualité, registre de fuites
  features.py          variables dérivées et leurs hypothèses, justification des brutes, pipeline
  target.py            cible M1/M2/M3, arbitrage empirique des « 3 »
  protocol.py          protocole 15/85 — MCAR / MAR / MNAR, IPW vraie et estimée
  models.py            catalogue des 15 modèles (7 familles), espaces de réglage, calibration
  explication.py       contributions à la détraction, quel que soit le modèle (app + étude)
  evaluate.py          métriques justifiées, précision@K, économie, NPS simulé, équité, proxies, mitigation
  run.py               orchestrateur : grille, CV, réglage, sélection, robustesse, évaluation, persistance
  fondation_eval.py    TabICL / TabPFN (bonus)
  verbatims.py         verbatims synthétiques — API Anthropic + générateur local seedé
  texte.py             TF-IDF, fusion tardive et précoce
  monitoring.py        PSI, mois simulé, nouvelles réponses, déclencheurs
  rapports.py          l'étude : un document par phase + synthèse + conformité
  writeup.py           write-up 3–6 pages et étude complète → PDF
app/app.py             application Streamlit, 8 pages avec filtres croisés, adressables par ?page=
notebooks/             notebook d'exploration exécuté
scripts/               captures d'écran (Selenium), construction du notebook
tests/                 tests d'invariants (pytest)
prompts/               prompt versionné des verbatims
models/                modèle final (pipelines calibré / brut / décision) et modèle texte
reports/               l'étude (00 → 07), figures/, captures/, *.json des résultats
livrable/              write_up.pdf (3–6 pages), etude_complete.pdf, soutenance.pdf, notes d'oral
docs/                  cadrage, plan en 38 étapes, état de l'art, angle différenciant
```

## Usage d'outils d'IA (énoncé § 7)

Le code  et la rédaction ont été produits avec l'assistance d'un assistant IA(Claude), sous direction et relecture humaines. Les fragments des verbatims synthétiques ont été rédigés
par ce même assistant ; l'assemblage est local et seedé. Les choix de modélisation, les décisions de
périmètre et les conclusions sont assumés par l'auteur.
