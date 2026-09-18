# Prédiction de la catégorie NPS — clients silencieux d'un opérateur télécom

Take-home challenge Data Science (Artefact). Prédire, pour les 85 % de clients qui ne répondent pas à
l'enquête NPS, s'ils sont Détracteur, Passif ou Promoteur ; en tirer une liste d'appels de rétention
priorisée, les drivers de détraction et une estimation du NPS des silencieux.

## Lire

| Pour | Document |
|---|---|
| Décider en 10 minutes | **[livrable/write_up.pdf](livrable/write_up.pdf)** (6 pages) ou [reports/00_synthese.md](reports/00_synthese.md) |
| Vérifier que rien de l'énoncé n'a été oublié | [reports/07_conformite_enonce.md](reports/07_conformite_enonce.md) |
| Suivre la méthode, phase par phase | [reports/README.md](reports/README.md) → un document par phase CRISP-DM |
| Tout avoir en un PDF | [livrable/etude_complete.pdf](livrable/etude_complete.pdf) |
| Manipuler les données | [notebooks/01_exploration_et_cible.ipynb](notebooks/01_exploration_et_cible.ipynb) (exécuté) |
| Tester l'outil | `streamlit run app/app.py` → http://localhost:8501 (7 pages) ; captures dans `reports/captures/` |

Cadrage : `docs/` (compréhension du problème, plan en 38 étapes avec état d'avancement, état de l'art).

## Reproduire

```bash
pip install -r requirements.txt          # numpy < 2 impératif
python -m src.run                        # données → cible → 15 modèles → sélection → évaluation (~20 min)
python -m src.fondation_eval             # TabICL (et TabPFN si accès aux poids)
python -m src.verbatims                  # verbatims synthétiques (API Anthropic si clé dans .env, sinon générateur local seedé)
python -m src.texte                      # modèle texte et fusion
python -m src.monitoring                 # dérive PSI, nouvelles réponses, déclencheurs
python -m src.rapports                   # l'étude : 8 documents, ~22 figures dans reports/
python -m src.writeup                    # livrable/write_up.pdf et livrable/etude_complete.pdf
python scripts/notebook_eda.py           # notebook exécuté
python -m pytest tests -q                # tests d'invariants (données, fuites, cible, protocole, 15 modèles, sélection, app)
python -m streamlit run app/app.py       # puis python scripts/captures.py pour les captures
```

Tout est déterministe (seed dans `config.yaml`). Aucune clé d'API n'est nécessaire (`.env.example`).
TabPFN requiert l'acceptation des conditions du dépôt Hugging Face `Prior-Labs/tabpfn_3_5` et
`huggingface-cli login` ; TabICL n'a pas cette contrainte et est évalué.

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
  nouvelles réponses, quatre déclencheurs) ; **groupe de contrôle** recommandé.

## Arborescence

```
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
app/app.py             application Streamlit, 7 pages, adressables par ?page=
notebooks/             notebook d'exploration exécuté
scripts/               captures d'écran (Selenium), construction du notebook
tests/                 tests d'invariants (pytest)
prompts/               prompt versionné des verbatims
models/                modèle final (pipelines calibré / brut / décision) et modèle texte
reports/               l'étude (00 → 07), figures/, captures/, *.json des résultats
livrable/              write_up.pdf, etude_complete.pdf (+ sources .md / .html)
docs/                  cadrage, plan en 38 étapes, état de l'art, angle différenciant
```

## Usage d'outils d'IA (énoncé § 7)

Le code, la structure de l'étude et la rédaction ont été produits avec l'assistance d'un assistant IA
(Claude), sous direction et relecture humaines. Les fragments des verbatims synthétiques ont été rédigés
par ce même assistant ; l'assemblage est local et seedé. Les choix de modélisation, les décisions de
périmètre et les conclusions sont assumés par l'auteur.
