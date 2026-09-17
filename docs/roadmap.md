# Plan de travail — Challenge NPS (Artefact)

Document de référence unique. **38 étapes, structurées selon CRISP-DM**, aucune optionnelle.
Pour chaque étape : l'objectif, ce qu'on fait, les outils et pourquoi ceux-là, ce qui sort, et le
critère qui dit que c'est fini.

Documents liés : `comprehension_du_probleme.md` (le métier), `etat_de_l_art.md` (les quatre
études et ce qu'elles imposent), `brief/enonce_challenge_nps.pdf` (l'énoncé).

---

## État d'avancement au 17/09

| Étapes | Phase | Statut | Où |
|---|---|---|---|
| 1 – 3 | Business understanding | ✅ fait | `reports/01_comprehension_metier.md` |
| 4 – 8 | Data understanding | ✅ fait | `reports/02_donnees/` (dictionnaire, qualité, EDA, registre de fuites) |
| 9 – 12, 14 – 15 | Data preparation | ✅ fait | `reports/03_preparation/` |
| 13 | Verbatims synthétiques | ⚠️ fait **sans API** — générateur local seedé, chemin API prêt | `src/verbatims.py`, `prompts/verbatim_v1.txt` |
| 16 – 19, 21 – 24 | Modeling | ✅ fait | `reports/04_modelisation/` |
| 20 | TabPFN | ⛔ **bloqué** — dépôt Hugging Face soumis à acceptation ; script prêt | `src/tabpfn_eval.py`, `reports/04_modelisation/tabpfn.md` |
| 25 – 32 | Evaluation | ✅ fait | `reports/05_evaluation/` |
| 33 – 34 | Persistance, Streamlit | ✅ fait — 10 pages | `models/`, `app/app.py` |
| 35 | Captures | ✅ fait — 10 captures Selenium | `reports/captures/` |
| 36 | Monitoring | ✅ **implémenté** (PSI, mois simulé, déclencheur) | `src/monitoring.py` |
| 37 | Write-up | ✅ fait — 4 pages | `livrable/write_up.pdf` |
| 38 | Reproductibilité | ✅ tests (`pytest`) + test à blanc depuis copie vierge | `tests/`, `reports/reproductibilite.json` |

Écarts avec le plan initial, assumés : pas de notebook EDA (les figures et l'analyse sont dans
`reports/02_donnees/`, régénérables par `src/rapports.py`) ; la comparaison LightGBM/CatBoost de
second rang n'a pas été menée (HGB réglé reste derrière la logistique, la hiérarchie est nette).

---

## Convention de confiance

Trois natures d'affirmations, qui ne se valent pas :

- **Faits mesurés** — calculés sur le dataset. Vérifiables en rejouant le code. Seuls chiffres
  utilisables tels quels.
- **Faits sourcés** — capacités d'outils, résultats publiés, comportements d'API. Vérifiés contre
  une source, ou marqués ⚠️ quand ils ne le sont pas.
- **Jugements** — choix d'outils, priorités, hypothèses. Discutables, à trancher par toi.

**Tout passage marqué ⚠️ est une réserve : ne pas le reprendre dans le write-up sans
vérification.**

## Règle de publication des métriques

**Aucune métrique globale n'est publiée sans son détail par classe, dans le même tableau. Sans
exception.** Kannan et al. (2022) annoncent 0,876 de F-score en résumé ; leur classe minoritaire
est à 0,126, et une de leurs configurations n'a prédit aucun cas minoritaire. C'est exactement ce
que l'énoncé sanctionne (« Accuracy alone is not enough »).

---

## Le pipeline en une image

```
PHASE 1  BUSINESS UNDERSTANDING     étapes 1 → 3     cadrage, critères de succès, risques
PHASE 2  DATA UNDERSTANDING         étapes 4 → 8     ingestion, description, EDA, qualité, fuites
PHASE 3  DATA PREPARATION           étapes 9 → 15    cible, nettoyage, features, encodage
PHASE 4  MODELING                   étapes 16 → 24   protocole, baseline, modèles, calibration
PHASE 5  EVALUATION                 étapes 25 → 32   technique, métier, SHAP, équité, revue
PHASE 6  DEPLOYMENT                 étapes 33 → 38   persistance, Streamlit, monitoring, remise
```

---

# PHASE 1 — BUSINESS UNDERSTANDING

## Étape 1 — Contexte, objectifs métier et critères de succès

**Objectif** — écrire ce qu'on cherche à obtenir, et à quoi on saura qu'on a réussi. CRISP-DM
l'exige avant toute ligne de code, et l'énoncé est jugé là-dessus.

**Ce qu'on fait**
1. Objectif métier : permettre à l'équipe rétention de prioriser ses appels sur les détracteurs
   prédits parmi les 85 % de clients silencieux.
2. **Critère de succès métier** : sur les K clients appelables dans le mois, la proportion de
   vrais détracteurs dépasse nettement le taux de base. K est fixé à l'étape 27.
3. **Critère de succès data science** : rappel Détracteur et kappa quadratique pondéré supérieurs
   à la baseline, avec des probabilités calibrées.
4. Objectif secondaire explicite : expliquer **pourquoi** un client est détracteur, pas seulement
   **qui**.

**Sortie** — `docs/comprehension_du_probleme.md`, section « critères de succès ».

**Fin d'étape** — les critères sont chiffrables, pas rhétoriques.

---

## Étape 2 — Ressources, hypothèses, contraintes et risques

**Objectif** — la case CRISP-DM que tout le monde saute, et qui distingue un plan d'une intention.

**Ce qu'on fait**
1. Ressources : dataset IBM Telco 11.1.3+ (5 fichiers, présents), Python 3.11, 6 jours.
2. **Hypothèses déclarées** : que `Satisfaction Score` est un proxy acceptable du NPS ; que le
   biais de non-réponse simulé ressemble au biais réel ; que les patterns d'un opérateur
   californien fictif se transposent au cadrage panafricain de l'énoncé.
3. **Contraintes** : dataset fictif, une seule photographie temporelle, aucun historique.
4. **Registre de risques**, avec parade pour chacun :

| Risque | Parade |
|---|---|
| Fuite non détectée → score irréel, candidature disqualifiée | étapes 8 et 17, double diagnostic |
| Cible mal construite → tout le reste invalide | étape 10, arbitrage empirique + sensibilité (28) |
| Modèle inéquitable non vu | étape 31, audit obligatoire |
| Livrable non reproductible | étape 38, test à blanc depuis un clone vierge |

**Sortie** — `docs/cadrage_projet.md`.

**Fin d'étape** — chaque risque a une étape qui le traite.

---

## Étape 3 — Traduction en problème de data science et plan de projet

**Objectif** — passer de la demande métier à une formulation technique, et **justifier la
formulation** comme l'énoncé l'exige au § 2.

**Ce qu'on fait**
1. Poser les trois formulations candidates : classification nominale, **classification ordinale**,
   régression sur 1–5 puis seuillage. Argumenter *a priori*, trancher par les chiffres à l'étape 18.
2. Acter que la cible est **ordonnée** (Détracteur < Passif < Promoteur) et **déséquilibrée**.
3. Écrire le plan de projet : ce document.

**Sortie** — ce fichier, plus la section « cadrage » du write-up.

**Fin d'étape** — la question « classification, ordinal ou régression ? » est posée proprement,
avec les critères qui la trancheront.

---

# PHASE 2 — DATA UNDERSTANDING

## Étape 4 — Collecte initiale des données

**Objectif** — charger les cinq tables et les réunir en une table client unique.

**Ce qu'on fait**
1. Lecture : `demographics` (7 043 × 9), `location` (7 043 × 9), `population` (1 671 × 3),
   `services` (7 043 × 30), `status` (7 043 × 11).
2. Jointure des quatre tables sur `Customer ID`, rattachement de `population` via `Zip Code`.
3. Normalisation des noms en `snake_case`, typage explicite.
4. Écriture en Parquet.

**Outils**

| Outil | Rôle | Pourquoi celui-là |
|---|---|---|
| pandas | manipulation | 7 043 lignes : polars n'apporte rien à cette taille, pandas est le standard attendu |
| openpyxl | lecture xlsx | seul moteur pandas pour le `.xlsx` |
| pyarrow / Parquet | stockage intermédiaire | **préserve les types** ; un CSV réintroduit l'ambiguïté à chaque relecture et provoque des bugs silencieux |

**Sortie** — `data/interim/clients.parquet`.

**Fin d'étape** — exactement **7 043 lignes** après jointure. Toute perte est un bug de jointure,
jamais une donnée manquante.

---

## Étape 5 — Description des données

**Objectif** — le rapport de description exigé par CRISP-DM : savoir ce qu'on a avant de le
regarder.

**Ce qu'on fait**
1. Pour chaque colonne : nom, type, source, description métier, cardinalité, taux de manquants,
   plage de valeurs.
2. Identification des colonnes constantes à supprimer : `Count`, `Quarter`, et vérification de
   `Country` / `State` (base 100 % Californie).
3. Volumétrie et empreinte mémoire.

**Sortie** — `docs/dictionnaire_donnees.md`.

**Fin d'étape** — aucune colonne du dataset n'est sans description.

---

## Étape 6 — Analyse exploratoire (EDA)

**Objectif** — comprendre la structure des données et produire les constats visuels qui
nourriront le write-up.

**Ce qu'on fait**
1. **Univarié** : distribution de chaque variable, histogrammes, boîtes à moustaches, comptages.
2. **Distribution de la cible** : satisfaction 1 (922), 2 (518), 3 (2 665), 4 (1 789), 5 (1 149).
   Le poids des « 3 » — **37,8 %** — est le fait central du projet.
3. **Bivarié** : croisement de chaque variable candidate avec la satisfaction. Quelles variables
   séparent réellement les classes ?
4. **Corrélations** : matrice sur les numériques, V de Cramér sur les catégorielles.
5. **Géographique** : distribution par code postal, densité de population.
6. **Segments** : profils par type de contrat, ancienneté, type de connexion.
7. **Analyse de `Offer` — la quasi-expérience du dataset.** Croisement offre × taux de départ ×
   satisfaction. **Déjà mesuré** : Offer A → 6,7 % de départs, aucune offre → 27,1 %,
   **Offer E → 52,9 %**. Les clients ayant reçu l'offre E partent **deux fois plus** que ceux qui
   n'ont rien reçu. Lecture naïve (« l'offre E fait fuir ») presque certainement fausse : c'est un
   **biais d'indication** — l'offre a été proposée aux clients déjà en difficulté. Démonstration
   en une ligne de tableau qu'on ne lit pas un effet causal dans des données observationnelles.
   Socle de l'étape 27 — voir `angle_differenciant.md`.

**Outils** — pandas, matplotlib, seaborn. Rendu statique et reproductible, intégrable au PDF ;
Plotly serait interactif mais inexploitable dans un write-up imprimé.

**Sortie** — `notebooks/01_eda.ipynb`, figures dans `reports/figures/`.

**Fin d'étape** — chaque variable a été regardée au moins une fois, et les cinq figures qui iront
dans le write-up sont identifiées.

---

## Étape 7 — Vérification de la qualité des données

**Objectif** — le contrôle qualité formel de CRISP-DM. Cinq contrôles distincts, aucun facultatif.

**Ce qu'on fait**
1. **Valeurs manquantes** — et surtout distinguer manquant *accidentel* et manquant
   **structurel** : `Offer` à 3 877 NaN signifie « aucune offre promotionnelle », `Internet Type`
   à 1 526 NaN signifie « pas d'internet ». Ce sont des **catégories**, pas des trous. Les imputer
   serait une faute.
2. **Doublons** — sur `Customer ID`, puis sur l'ensemble des colonnes hors identifiant. Attendu :
   zéro. À vérifier, pas à supposer.
3. **Valeurs aberrantes** — sur `Tenure in Months`, `Monthly Charge`, `Total Charges`,
   `Total Revenue`, `Avg Monthly GB Download`. Méthode : écart interquartile + inspection
   visuelle. **Une aberrante n'est pas automatiquement une erreur** : un client à très forte
   consommation est réel. On documente, on ne supprime qu'avec justification.
4. **Cohérence interne** — `Total Charges` doit être cohérent avec `Monthly Charge × Tenure` ;
   une ancienneté de 0 mois avec un total facturé non nul est une incohérence à tracer.
5. **Cardinalités suspectes** — catégorielles à modalité unique, ou à cardinalité proche du nombre
   de lignes.

**Sortie** — `reports/qualite_donnees.md`.

**Fin d'étape** — les cinq contrôles sont passés et chaque anomalie a une décision écrite.

---

## Étape 8 — Registre de fuites

**Objectif** — **l'étape la plus discriminante du challenge.** Décider, colonne par colonne, ce
qui a le droit d'entrer dans le modèle. Elle appartient au *Data Understanding* : c'est un constat
sur les données, pas une transformation.

**Ce qu'on fait**
1. Classer chaque colonne en `cible` / `fuite` / `feature` / `suspect`, avec justification écrite.
2. **Quantifier au lieu d'affirmer** : information mutuelle entre chaque colonne candidate et la
   cible. Une valeur anormalement élevée révèle une fuite qu'on n'aurait pas devinée.
3. Distinguer **trois natures de fuite** — ce que la plupart des candidats ne feront pas :

| Nature | Définition | Colonnes concernées |
|---|---|---|
| **de conséquence** | la colonne est un effet de la cible | `Churn Label`, `Churn Value`, `Customer Status` |
| **de disponibilité** | la colonne n'existe que pour une partie de la population | `Churn Category`, `Churn Reason` (5 174 NaN — leur seule présence trahit la réponse) |
| **de modèle amont** | la colonne est produite par un autre modèle | `Churn Score` (IBM SPSS), `CLTV` |

Le fait fondateur : satisfaction 1–2 → **100 % de départs**, satisfaction 4–5 → **0 %**.
Connaître le churn, c'est presque connaître la cible.

**Outils**

| Outil | Rôle | Pourquoi |
|---|---|---|
| `sklearn.feature_selection.mutual_info_classif` | détecter les dépendances suspectes | capte le non-linéaire qu'une corrélation de Pearson manque |
| `pandas.crosstab` | dépendances catégorielles | c'est ce qui a révélé les 100 % / 0 % |

**Sortie** — `docs/registre_fuites.md` + liste d'exclusion dans `config.yaml`.

**Fin d'étape** — aucune colonne non classée.

**En entretien** — « Comment avez-vous traité les fuites ? » se répond avec une typologie en trois
natures et des chiffres, pas avec « j'ai enlevé les colonnes de churn ».

---

# PHASE 3 — DATA PREPARATION

## Étape 9 — Sélection des données

**Objectif** — acter le périmètre : quelles lignes, quelles colonnes, et pourquoi.

**Ce qu'on fait**
1. Lignes : les 7 043 clients sont tous conservés. Toute exclusion serait justifiée ici.
2. Colonnes retenues, à partir du registre de fuites.
3. **Exclusion des attributs démographiques et géographiques des variables explicatives**,
   conservés uniquement comme **dimensions d'audit** pour l'étape 31. Précédent : Elero et al.
   (2026) excluent délibérément les démographiques, et leurs drivers dominants sont tous des
   variables d'expérience de service. Cohérent avec le métier — les leviers actionnables sont
   contractuels, pas démographiques — et ça évite d'avoir à défendre un proxy socio-économique.

**Sortie** — section « périmètre » de `docs/registre_fuites.md`.

**Fin d'étape** — le périmètre est figé et justifié ligne à ligne.

---

## Étape 10 — Construction de la cible NPS

**Objectif** — transformer `Satisfaction Score` (1–5) en label NPS à trois classes ordonnées, et
**justifier le mapping** (§ 4.1 de l'énoncé).

**Ce qu'on fait**

1. **M1 — le mapping baseline de l'énoncé** (5 → Promoteur, 4 → Passif, ≤ 3 → Détracteur).
   Conséquence : 58,3 % de détracteurs, 25,4 % de passifs, 16,3 % de promoteurs → **NPS = −42**.
   **Benchmark sectoriel, désormais sourcé** : les télécoms affichent les NPS les plus bas de
   tous les secteurs, mais **positifs** — de l'ordre de **+19 à +34** selon les éditeurs
   (CustomerGauge, QuestionPro, Survicate). ⚠️ Sources **commerciales**, pas de littérature
   relue, et dispersion énorme (19, 31, 34, 54 selon l'année et l'éditeur) : à citer comme ordre
   de grandeur, jamais comme référence exacte. **Correction de ma première estimation** : j'avais
   avancé « −10 à +30 » de mémoire, c'était faux et dans le mauvais sens. L'écart de notre −42
   au secteur n'est donc pas de 30 points mais de **60 à 75**. L'anomalie est plus forte, pas plus
   faible.

2. **M2 — recentré** : 5 → Promoteur, 3 et 4 → Passif, ≤ 2 → Détracteur.

3. **M3 — arbitrage empirique des « 3 »**, méthode reprise d'Elero et al. (2026) et **apport
   principal de la revue de littérature** :
   - entraîner un modèle **uniquement sur les extrêmes non ambigus** : satisfaction 1–2 contre
     4–5, les 2 665 clients à satisfaction 3 étant exclus de l'entraînement ;
   - faire passer ces 2 665 clients, jamais vus, à travers le modèle ;
   - mesurer la proportion qui bascule de chaque côté ;
   - construire M3 à partir de ce résultat.

   Sur leurs données la méthode est stable sur neuf algorithmes : les « 7 » tombent côté
   insatisfait dans 69 à 81 % des cas, les « 8 » côté satisfait dans 59 à 67 % — avec assez de cas
   inverses pour prouver que le milieu contient réellement les deux profils.

   **Pourquoi ça change tout** : on passe de « j'ai décidé que les 3 seraient des passifs » à
   « les 3 se répartissent à X % du côté détracteur, voici la mesure ». C'est la différence entre
   choisir et démontrer.

4. **Interdiction absolue** d'arbitrer par `Churn Value` : la cible deviendrait une relabellisation
   du churn, et le projet un modèle de churn déguisé.

5. Bruit contrôlé, comme l'énoncé y invite, avec mesure de son effet.

**Outils** — pandas, numpy (générateur seedé). Étape de raisonnement métier, pas de technique.

**Sortie** — `data/processed/cible.parquet` (trois colonnes de labels),
`docs/construction_cible.md`.

**Fin d'étape** — les trois mappings tournent, leurs NPS respectifs sont chiffrés, et la
répartition empirique des « 3 » est mesurée.

---

## Étape 11 — Nettoyage des données

**Objectif** — appliquer les décisions de l'étape 7.

**Ce qu'on fait**
1. Manquants structurels → **encodés en catégorie explicite** (`"Pas d'offre"`, `"Pas
   d'internet"`), jamais imputés.
2. Manquants accidentels → imputation documentée (médiane pour les numériques, modalité la plus
   fréquente pour les catégorielles), **à l'intérieur du pipeline**.
3. Aberrantes → décision par variable : conservée, plafonnée (*winsorisation*) ou retirée, avec
   justification.
4. Incohérences → correction ou marquage par un indicateur.

**Outils** — `sklearn.impute.SimpleImputer` **dans le pipeline**, jamais en amont : sinon les
statistiques d'imputation calculées sur tout le jeu fuient vers l'évaluation.

**Sortie** — `src/nettoyage.py`.

**Fin d'étape** — le nombre de lignes est inchangé, ou toute perte est justifiée nominalement.

---

## Étape 12 — Feature engineering

**Objectif** — construire les variables explicatives. Critère annoncé par l'énoncé : *« the
quality of the features you build, the leakage you avoid, and the clarity with which you explain
why each feature should help »*.

**Ce qu'on fait** — chaque variable dérivée est adossée à une hypothèse métier explicite :

| Variable construite | Hypothèse métier |
|---|---|
| charge mensuelle / nombre de services | perception du rapport qualité-prix |
| nombre total de services souscrits | intensité de la relation commerciale |
| ancienneté en tranches | l'insatisfaction d'un nouveau client n'a pas la même nature que celle d'un ancien |
| parrainage (`Referred a Friend`, `Number of Referrals`) | **proxy comportemental direct de la recommandation**, donc du NPS lui-même |
| présence d'une offre promotionnelle | effet d'un geste commercial |
| type de connexion, données incluses | qualité de service perçue |
| remboursements, frais supplémentaires | traces d'incidents de facturation |
| densité de population du code postal | urbain / rural, qualité de couverture |

**Sortie** — `src/features.py`, `data/processed/features.parquet`.

**Fin d'étape** — chaque feature a une phrase expliquant *pourquoi elle devrait prédire la
satisfaction*. Une feature sans justification est retirée.

---

## Étape 13 — Génération des verbatims synthétiques *(bonus § 4.4)*

**Objectif** — produire une note de dernier contact par client, pour tester l'apport du texte.

**Ce qu'on fait**
1. Prompt conditionné sur un sous-ensemble de features (ancienneté, contrat, services, risque).
2. Tonalité corrélée à la classe **mais imparfaitement** : l'énoncé demande explicitement du bruit
   et des cas contre-intuitifs. Un détracteur poli, un promoteur laconique.
3. Génération par lots, avec reprise sur incident.
4. ⚠️ **Reproductibilité — à traiter honnêtement.** L'énoncé dit « fix your seed », or la Messages
   API d'Anthropic **n'expose aucun paramètre de graine**, et `temperature` a été **supprimé** des
   modèles actuels (Opus 5, Sonnet 5, Opus 4.7/4.8) : l'envoyer renvoie une erreur 400. Aucun
   réglage ne rend la génération déterministe. La seule reproductibilité atteignable : **épingler
   l'identifiant exact du modèle, versionner le prompt, et commiter les textes générés** — ce sont
   eux, et non le script, qui constituent l'artefact reproductible. À écrire tel quel : c'est une
   consigne qu'on ne peut pas satisfaire littéralement, et le dire vaut mieux que le simuler.

**Outils** — SDK Anthropic ; `.env` + `.env.example`, aucune clé dans le dépôt (§ 7).

**Sortie** — `data/processed/verbatims.parquet`, `src/generation_verbatims.py`,
`prompts/verbatim_v1.txt`.

**Fin d'étape** — un verbatim par client, commité, avec son prompt versionné.

---

## Étape 14 — Colinéarité et sélection de variables

**Objectif** — réduire le bruit et la redondance.

**Ce qu'on fait**
1. **Colinéarité** : matrice de corrélation, facteur d'inflation de la variance (VIF) sur les
   numériques. `Monthly Charge`, `Total Charges` et `Tenure` sont mécaniquement liés — décider
   lesquels garder.
2. **Sélection par importance** : forêt aléatoire avec `class_weight='balanced'`, classement des
   variables par importance, coupure sous un seuil d'importance cumulée. Méthode reprise de Chong
   et al. (2023), qui retiennent 15 variables sur 19 représentant ~94 % de l'importance totale.
3. **Deux jeux de features** conservés : complet et réduit, comparés à l'étape 26.

**Outils** — `statsmodels` (VIF), `RandomForestClassifier` (`feature_importances_`).

**Sortie** — `reports/selection_variables.md`.

**Fin d'étape** — aucune paire de variables à corrélation extrême ne subsiste sans décision écrite.

---

## Étape 15 — Encodage, normalisation et pipeline de préparation

**Objectif** — rendre les données consommables par les algorithmes, **sans jamais laisser fuir
l'information du train vers le test**.

**Ce qu'on fait**
1. **Encodage** : `OneHotEncoder(handle_unknown='ignore')` pour les catégorielles — l'application
   Streamlit devra tolérer une modalité inconnue sans planter. Catégorielles natives pour les
   modèles à arbres, qui évitent l'explosion dimensionnelle.
2. **Normalisation** : `StandardScaler` **indispensable** pour la régression logistique, les
   modèles ordinaux et tout modèle à distance ; **inutile** pour les arbres. Deux pipelines
   distincts plutôt qu'un compromis.
3. **Assemblage** : tout dans un `Pipeline` scikit-learn avec `ColumnTransformer`. Jamais de
   transformation en amont du pipeline.

**Outils**

| Outil | Rôle | Pourquoi |
|---|---|---|
| `Pipeline` + `ColumnTransformer` | encapsulation preprocessing + modèle | garantit l'absence de fuite et rend le modèle sérialisable d'un bloc à l'étape 33 |
| `OneHotEncoder(handle_unknown='ignore')` | catégorielles | tolérance aux modalités inconnues, exigée par l'interface |
| `StandardScaler` | mise à l'échelle | requis par les modèles linéaires et ordinaux |

**Sortie** — `src/pipeline.py`.

**Fin d'étape** — le pipeline s'exécute de bout en bout et produit une matrice sans NaN.

---

# PHASE 4 — MODELING

## Étape 16 — Conception du protocole de test

**Objectif** — le *« generate test design »* de CRISP-DM, et **le différenciateur technique du
projet**.

L'énoncé demande un découpage reflétant « the 15 % of respondents vs the 85 % silent base ». Un
`train_test_split` stratifié rate l'intention : les répondants à une enquête NPS ne sont pas un
échantillon aléatoire. C'est du **MNAR** (*missing not at random*).

L'avantage rare de ce dataset : on connaît la satisfaction des 100 %. On peut donc mesurer ce
qu'aucun praticien ne peut mesurer en production — la dégradation réelle due au biais de réponse.

**Ce qu'on fait**
1. Modéliser une propension à répondre, en trois scénarios :

| Scénario | Mécanisme | Ce qu'il mesure |
|---|---|---|
| S1 — MCAR | 15 % tirés au hasard | borne haute optimiste |
| S2 — MAR | propension = f(ancienneté, contrat, facture dématérialisée) | biais de covariables seul |
| S3 — MNAR | propension = f(covariables, **satisfaction**) | le cas réaliste |

2. Sous chaque scénario : entraîner sur les 15 % « répondants », évaluer sur les 85 % silencieux
   dont on connaît la vérité. **L'écart S1 → S3 est le résultat scientifique du projet.**
3. Poids de repondération par l'inverse de la propension (IPW), et test de leur effet correctif.
4. Validation croisée stratifiée à l'intérieur des 15 %.

**Ancrage littérature** — Kannan et al. (2022) posent le problème en toutes lettres : *« only
about 15–20 % of customers respond […] poor survey response rates and the possible loss of
valuable insights from non-respondents »*. Chez eux le taux réel est de **2,6 %** : notre
simulation à 15 % est donc **optimiste**. Référence centrale à citer : Zihayat et al. (2021),
*Leveraging non-respondent data in customer satisfaction modeling*, J. Business Research 135.

**Cadre théorique de S3 — le modèle de sélection de Heckman.** Le mécanisme MNAR porte un nom et
a un traitement canonique, qui a valu un Nobel d'économie à son auteur : deux équations estimées
conjointement, une de **sélection** (qui répond à l'enquête ?) et une de **résultat** (quelle
satisfaction ?), en modélisant la corrélation de leurs termes d'erreur.

⚠️ **Limite sérieuse à énoncer** : Heckman exige une **restriction d'exclusion** — une variable
qui prédit la réponse **sans** prédire la satisfaction. De telles variables sont notoirement
difficiles à trouver, et le modèle impose en général une normalité bivariée. Candidat plausible
ici : la **facturation dématérialisée** (liée à la propension à répondre en ligne, peu liée à la
satisfaction du service). **Si aucune restriction ne tient, on le documente et on s'en tient à
l'IPW.** Nommer le cadre et expliquer pourquoi on ne peut pas l'appliquer pleinement vaut mieux
que de l'ignorer.

**Outils** — `LogisticRegression` (propension, interprétable — on doit pouvoir énoncer le
mécanisme simulé), `statsmodels` (Heckman), `numpy.random.default_rng(seed)`, `StratifiedKFold`.

**Sortie** — `data/processed/splits/`, `docs/protocole_validation.md`.

**Fin d'étape** — sous S3, la distribution de satisfaction des répondants **diffère visiblement**
de celle de la base complète. Si elle est identique, le mécanisme ne mord pas et il faut le durcir.

---

## Étape 17 — Baseline

**Objectif** — la référence contre laquelle tout le reste sera jugé. L'énoncé est catégorique :
*« Do not skip this step — it disciplines the rest of your work. »*

**Ce qu'on fait** — régression logistique multinomiale, pipeline complet, évaluée sur la grille
scénario × mapping.

**Outils** — `LogisticRegression(multi_class='multinomial')` : interprétable, rapide, plancher
honnête.

**Sortie** — `reports/baseline.md`.

**Fin d'étape — porte de contrôle critique du projet.** Deux diagnostics :

1. **Seuil absolu.** Si la baseline dépasse **~0,85** sur une métrique quelconque, il reste une
   fuite → retour immédiat à l'étape 8. Calibrage : Elero et al. (2026) obtiennent **0,77**
   d'exactitude sur une cible NPS à trois classes avec 64 647 clients ; Chong et al. (2023)
   **0,80** sur l'ancienne version de notre propre dataset. La fourchette honnête est
   **0,70 – 0,80**. Au-delà de 0,85, on est hors de tout précédent publié.
2. **Écart arbres / linéaires.** Un écart supérieur à ~30 points signale une variable qui fuit,
   qu'un arbre isole d'un seul découpage là où un modèle linéaire ne l'exploite pas. C'est
   exactement le motif de Mustafa et al. (2021) — 98 % avec les arbres, 41 % avec la régression
   logistique — dans une étude publiée dont la cible était dérivée d'une feature restée dans le
   modèle.

---

## Étape 18 — Modèles ordinaux

**Objectif** — exploiter l'ordre des classes. § 2 de l'énoncé : *« NPS is fundamentally an ordinal
target […] justify your choice »*. Un classifieur nominal traite une confusion
Promoteur/Détracteur comme équivalente à Promoteur/Passif — ce qui est faux métier.

**Ce qu'on fait**
1. Régression logistique ordinale à seuils ordonnés.
2. Alternative : régression sur la satisfaction 1–5, puis optimisation des deux seuils de découpage.
3. Comparaison explicite ordinal vs nominal, sur les mêmes données.

**Outils**

| Outil | Rôle | Pourquoi |
|---|---|---|
| `mord` | régression ordinale | cité par l'énoncé au § 10. ⚠️ Peu maintenu et mal aligné sur scikit-learn récent — **vérifier qu'il s'installe et tourne avec scikit-learn 1.3 avant de bâtir dessus** |
| `scipy.optimize` | recherche des seuils | optimisation directe du kappa quadratique ; **suffit à couvrir l'exigence ordinale** si `mord` résiste |

**CORAL et CORN, cités par l'énoncé au § 10** — vérification faite, ce sont des méthodes
d'**apprentissage profond** (PyTorch), conçues pour garantir la cohérence de rang entre les K−1
tâches binaires d'un réseau de neurones. Sur 7 043 lignes de données tabulaires, un réseau profond
n'a aucune chance de battre un gradient boosting. **On les cite, on explique pourquoi on ne les
retient pas** — volume insuffisant, complexité injustifiée — et on couvre l'exigence ordinale par
la régression à seuils optimisés. Écarter une piste avec un argument est plus fort que l'ignorer,
et plus fort encore que de l'utiliser sans raison.

**Sortie** — `reports/ordinal.md`.

**Fin d'étape** — la question de l'étape 3 est tranchée par des chiffres.

---

## Étape 19 — Gradient boosting

**Objectif** — l'état de l'art sur données tabulaires. § 4.5 exige au moins deux familles.

**Ce qu'on fait**
1. `HistGradientBoostingClassifier` multiclasse, avec pondération de classes.
2. Variante ordinale : cascade de deux classifieurs binaires (Détracteur vs reste, puis Passif vs
   Promoteur), qui respecte l'ordre.
3. Traçage systématique des expériences.

**Outils**

| Outil | Rôle | Pourquoi |
|---|---|---|
| **`HistGradientBoostingClassifier`** | **boosting principal** | livré dans scikit-learn, **déjà installé**. Elero et al. (2026) le classent **premier sur neuf algorithmes** pour une cible NPS à trois classes (exactitude 0,774, meilleur F1 Détracteur à 0,842) avec la plus faible variance en validation croisée |
| LightGBM, CatBoost | comparaisons de second rang | ⚠️ **les trois grands se valent ici** : XGBoost gère aussi les catégorielles nativement depuis la 1.5/1.6. Le choix se tranche **en testant**, pas en argumentant |
| MLflow | suivi d'expériences | cité par l'énoncé ; permet de reconstituer *pourquoi* un modèle a été retenu |

**Sortie** — `models/hgb/`, `reports/comparaison_modeles.md`.

**Fin d'étape** — au moins deux familles comparées, écart au baseline chiffré.

---

## Étape 20 — Modèle de fondation tabulaire *(bonus)*

**Objectif** — bonus explicite, avec consigne d'honnêteté : *« do not frame it as the winner if it
is not »*.

**Ce qu'on fait** — TabPFN-2.5 sur le même découpage. Domaine de validité annoncé par PriorLabs :
**~50 000 lignes et ~2 000 features** (TabPFN v2 plafonnait vers 10 000 — ne pas confondre les
générations). Nos 7 043 lignes sont très largement dans le domaine : ce n'est pas un cas limite.
Mesure du temps d'inférence et de l'empreinte mémoire.

**Limites à documenter sans complaisance** — latence à l'inférence, pas d'interprétabilité native
(donc SHAP par permutation, coûteux), coût en production.

**Sortie** — `reports/tabpfn.md`.

**Fin d'étape** — avantages **et** limites documentés, conclusion assumée même si négative.

---

## Étape 21 — Modélisation du signal textuel et fusion *(bonus)*

**Objectif** — mesurer ce que le texte apporte, et **savoir conclure qu'il n'apporte presque
rien**.

**Ce qu'on fait**
1. Trois représentations : TF-IDF, embeddings de phrases, classification zéro-shot par LLM.
2. Deux stratégies de fusion : tardive (moyenne pondérée des probabilités) et précoce
   (concaténation aux features tabulaires).
3. Mesure du gain **au-dessus** de la baseline tabulaire, pas dans l'absolu.
4. **Limite logique à énoncer explicitement** : un texte généré à partir des features tabulaires
   ne peut contenir aucune information absente de ces features. Tout gain mesuré relève donc de la
   **circularité**. Le dispositif teste la capacité à extraire du signal textuel ; il ne prouve
   rien sur l'apport de vrais verbatims en production.

C'est précisément le *« spot when the added complexity is not worth it »* de l'énoncé.

**Outils** — `TfidfVectorizer` (référence indispensable : si les embeddings ne la battent pas, ils
ne servent à rien), `sentence-transformers` (`all-MiniLM-L6-v2`, local, sans coût d'API), SDK
Anthropic (zéro-shot).

**Sortie** — `reports/apport_texte.md`.

**Fin d'étape** — la limite de circularité figure noir sur blanc.

---

## Étape 22 — Optimisation des hyperparamètres

**Objectif** — tirer le maximum de chaque famille, à budget de calcul contraint.

**Ce qu'on fait**
1. Espace de recherche défini par modèle.
2. Recherche **bayésienne** plutôt qu'exhaustive.
3. Optimisation **sur le rappel Détracteur ou le kappa quadratique**, jamais sur l'accuracy.
4. Validation croisée interne au jeu d'entraînement, pour ne pas surajuster sur un tirage.

**Outils** — Optuna (échantillonneur TPE, bien plus efficace qu'une grille à budget égal) ;
`GridSearchCV` en secours.

**Sortie** — `reports/hyperparametres.md`, études MLflow.

**Fin d'étape** — chaque modèle final a ses hyperparamètres justifiés par une recherche tracée.

---

## Étape 23 — Traitement du déséquilibre

**Objectif** — l'énoncé demande de discuter le déséquilibre explicitement (§ 4.2).

**Ce qu'on fait**
1. Quantifier le déséquilibre sous chaque mapping.
2. Comparer trois traitements : **pondération de classes**, **ajustement des seuils de décision**,
   **suréchantillonnage SMOTE**.
3. **Position argumentée** : privilégier pondération et seuils. ⚠️ SMOTE dégrade la calibration —
   or on a besoin de probabilités fiables à l'étape 24 pour prioriser les appels. Le tester quand
   même, et **montrer** la dégradation plutôt que l'affirmer.

**Outils** — `class_weight='balanced'`, `imbalanced-learn` (SMOTE, à titre de comparaison).

**Sortie** — `reports/desequilibre.md`.

**Fin d'étape** — le choix est adossé à une mesure de calibration, pas à une habitude.

---

## Étape 24 — Calibration des probabilités

**Objectif** — l'équipe rétention priorise par probabilité. Un modèle qui annonce 80 % de risque
là où le taux réel est 40 % conduit à mal allouer le budget.

**Ce qu'on fait**
1. Courbes de fiabilité par classe, calcul de l'*Expected Calibration Error*.
2. Recalibration isotonique et Platt, comparées.
3. Optimisation des seuils **sur le rappel Détracteur**.

**Outils** — `CalibratedClassifierCV`, `calibration_curve` (produit la figure attendue par
l'énoncé).

**Sortie** — `reports/calibration.md` + figures.

**Fin d'étape** — le modèle final est calibré, ses seuils justifiés par un objectif métier.

---

# PHASE 5 — EVALUATION

## Étape 25 — Évaluation technique par classe

**Objectif** — la photographie complète de la performance. Rien de global sans détail par classe.

**Ce qu'on fait**
1. **Matrice de confusion** pour chaque modèle × scénario × mapping. C'est l'objet qui montre
   *quelles* erreurs sont commises — une confusion Promoteur/Détracteur n'a pas le même coût
   qu'une confusion voisine.
2. **Métriques par classe** : précision, rappel, F1 pour Détracteur, Passif, Promoteur séparément.
3. **Métriques globales adaptées** : macro-F1, exactitude équilibrée, **kappa quadratique pondéré**
   (respecte l'ordre : pénalise davantage une erreur de deux crans).
4. **Courbes ROC et précision-rappel** en un-contre-tous. ⚠️ Sous déséquilibre, la **courbe
   précision-rappel est plus informative que la ROC** — la ROC reste flatteuse quand la classe
   négative domine.
5. L'accuracy figure au rapport **uniquement pour démontrer qu'elle est trompeuse ici**.

**Outils** — `confusion_matrix`, `classification_report`,
`cohen_kappa_score(weights='quadratic')`, `precision_recall_curve`, `roc_auc_score`.

**Sortie** — `reports/evaluation_technique.md`.

**Fin d'étape** — la table respecte la règle de publication : aucune métrique globale sans son
détail par classe.

---

## Étape 26 — Comparaison statistique des modèles

**Objectif** — établir qu'un modèle est meilleur, pas qu'il a eu de la chance sur un tirage.

**Ce qu'on fait**
1. **Validation croisée répétée** (10 plis) pour chaque modèle final.
2. **Boîtes à moustaches** des scores par pli — méthode d'Elero et al. (2026), qui comparent ainsi
   neuf algorithmes et retiennent HGB pour sa **faible variance** autant que pour sa moyenne.
3. Moyenne **et écart-type** rapportés : un modèle meilleur en moyenne mais très instable n'est
   pas un meilleur modèle.
4. Comparaison également des deux jeux de features de l'étape 14.

**Outils** — `cross_val_score`, `RepeatedStratifiedKFold`, matplotlib.

**Sortie** — `reports/comparaison_statistique.md` + figure.

**Fin d'étape** — le modèle retenu l'est pour une raison mesurée, écart-type inclus.

---

## Étape 27 — Évaluation métier

**Objectif** — l'énoncé écrit *« Accuracy alone is not enough »*. La vraie question n'est pas
« quel taux de bonnes réponses » mais « sur les K clients appelables ce mois-ci, combien sont
réellement des détracteurs ».

**Ce qu'on fait**
1. **Fixer K explicitement** : la capacité d'appel mensuelle de l'équipe rétention.
2. **Précision@K et rappel@K sur la classe Détracteur**, courbe de lift, courbe de gain cumulé.
3. **Traduction en euros** : coût d'un appel, valeur d'un client retenu, seuil de rentabilité.
4. Comparaison à la stratégie naïve : appeler au hasard, ou appeler les plus gros ARPU.
5. **Pondérer la priorisation par la valeur client individuelle.** Gómez-Vargas, Maldonado et
   Vairetti (EJOR 2025) montrent qu'une CLV **individuelle** bat une CLV moyenne pour cibler une
   campagne : l'agrégation détruit l'information d'arbitrage.
   **Distinction à faire valoir** : `CLTV` est **interdite comme feature** (sortie de modèle IBM,
   elle fuit) mais **légitime comme paramètre de décision** — pondérer par la valeur du client au
   moment de choisir qui appeler est un choix métier, pas une information sur la cible. Une même
   variable bannie de la couche de modélisation et admise dans la couche de décision : savoir
   séparer ces deux couches est une réponse d'entretien qui se retient.
6. **Section « limites de la règle de ciblage »** — le passage différenciant du livrable,
   détaillé dans `angle_differenciant.md` :
   - matrice des quatre types de clients (acquis / perdus d'avance / **persuadables** / à ne pas
     déranger) ;
   - la table `Offer` de l'étape 6 comme preuve qu'un effet observationnel n'est pas causal ;
   - **Ascarza (2018, Journal of Marketing Research, prix Paul E. Green)** : cibler les clients au
     risque le plus élevé est inefficace ; il faut cibler ceux dont la **sensibilité à
     l'intervention** est la plus forte. Sur ses deux expériences de terrain, la même campagne
     aurait réduit le churn de **7 points de plus** avec la bonne règle ;
   - conclusion assumée : notre modèle classe par **risque**, ce que l'énoncé demande et qu'on
     livre intégralement — mais l'optimum économique se classerait par **sensibilité**, et voici
     ce qu'il faudrait pour y accéder (étape 36) ;
   - ⚠️ **Ne pas prétendre faire de l'uplift modeling** : sans traitement randomisé, les données
     ne le permettent pas. L'annoncer serait exactement la faute reprochée à Mustafa et al. (2021).

**Outils** — fonctions maison (aucune implémentation standard ne correspond à un budget d'appels
contraint), matplotlib.

**Sortie** — `reports/evaluation_metier.md`, **modèle final figé**.

**Fin d'étape** — le modèle retenu est justifié par la métrique métier, pas par l'accuracy.

---

## Étape 28 — Analyse de sensibilité au mapping

**Objectif** — l'énoncé demande explicitement *« discuss how sensitive your downstream model is to
it »*. C'est la validation de l'étape 10.

**Ce qu'on fait** — rejouer l'intégralité du pipeline sous M1, M2 et M3, puis vérifier si les
**conclusions métier** (drivers, segments à risque, leviers) restent stables. Une conclusion qui
s'inverse selon le mapping est fragile, et il faut le dire.

⚠️ **Piège de comparaison** : Elero et al. obtiennent 0,85 sur Model 1, **0,95 sur Model 2** et
0,77 sur Model 3 — et l'expliquent : retirer le milieu ambigu gonfle mécaniquement la performance.
**Nos M1/M2/M3 ne sont pas comparables entre eux en valeur absolue.** Le dire, sinon on commet
l'erreur qu'on prétend dénoncer.

**Outils** — MLflow, pour comparer les trois campagnes sans confusion.

**Sortie** — `reports/sensibilite_mapping.md`.

**Fin d'étape** — on sait lesquelles des conclusions sont robustes et lesquelles ne le sont pas.

---

## Étape 29 — Interprétabilité

**Objectif** — § 4.5 : *« At minimum, produce feature importance or SHAP values. »*

**Ce qu'on fait**
1. **SHAP global** : importance et sens de l'effet.
2. **SHAP local** pour un client donné — alimentera directement l'interface de l'étape 34.
3. **Table de convergence**, méthode d'Elero et al. (2026) : importances natives des modèles à
   arbres **×** valeurs SHAP **×** les trois mappings. Chez eux, les cinq mêmes variables
   ressortent dans toutes les méthodes et toutes les formulations — seul l'ordre bouge. **Une
   variable qui domine partout est un driver solide ; une variable qui n'apparaît que dans une
   méthode ne va pas dans le write-up.**
4. Vérification de cohérence avec les coefficients de la baseline logistique.

**Outils**

| Outil | Rôle | Pourquoi |
|---|---|---|
| `shap.TreeExplainer` | attribution | exact et rapide sur les arbres, contrairement à KernelSHAP, approché et lent |
| importances natives | contre-épreuve | méthode indépendante ; la convergence des deux renforce la conclusion |
| LIME | écarté | approximation locale instable d'une exécution à l'autre ; SHAP a des garanties issues des valeurs de Shapley |

**Sortie** — `reports/interpretabilite.md` + figures.

**Fin d'étape** — la table de convergence existe et distingue drivers solides et drivers fragiles.

---

## Étape 30 — Drivers de détraction

**Objectif** — § 4.6. Le passage du « qui » au « pourquoi », qui transforme un modèle en outil de
pilotage.

**Ce qu'on fait**
1. Drivers **par segment** : nouveaux vs anciens clients, mensuel vs annuel, fibre vs DSL.
2. Classement **actionnable / non actionnable** — l'entreprise peut renégocier un contrat, elle ne
   peut pas changer l'âge d'un client.
3. Pour chaque détracteur prédit, identifier **le levier unique le plus probable**.
4. Vocabulaire rigoureux : le modèle établit des associations, pas des causes. L'énoncé n'exige pas
   de rigueur causale mais exige l'honnêteté sur la distinction.

**Point de comparaison** — chez Elero et al., les drivers dominants sont tous des variables
d'expérience de service : vitesse de navigation, respect des engagements annoncés, exactitude de la
facturation, stabilité de la connexion, clarté de l'offre. Si nos drivers ressemblent à ça, c'est
bon signe.

**Sortie** — `reports/drivers.md`.

**Fin d'étape** — chaque driver est classé actionnable ou non, et associé à un levier concret.

---

## Étape 31 — Audit d'équité

**Objectif** — § 4.7. Un modèle qui priorise le budget de rétention peut le répartir inégalement.
L'énoncé donne l'exemple chiffré : capter 80 % des jeunes détracteurs mais 50 % des seniors est
inacceptable, même à accuracy globale élevée.

**Ce qu'on fait**
1. **Rappel Détracteur par sous-groupe** : genre, senior, situation maritale, présence d'enfants,
   zone géographique — sur les dimensions **exclues des features à l'étape 9** et conservées
   précisément pour cet audit.
2. Chiffrer le **coût en performance** de cette exclusion : entraîner une variante *avec* les
   démographiques, comparer. L'énoncé exige que l'arbitrage soit explicite, pas seulement évoqué.
3. Formuler la conséquence métier en langage non technique.
4. Identifier ce qui devrait **remonter à une équipe Juridique ou Expérience Client** avant mise en
   production.

**Outils** — `fairlearn.metrics.MetricFrame` : calcule n'importe quelle métrique ventilée par
groupe avec les écarts min/max ; c'est l'outil que l'énoncé cite.

**Sortie** — `docs/audit_equite.md`.

**Fin d'étape** — l'arbitrage équité / performance est **chiffré**.

---

## Étape 32 — Revue du processus et décision

**Objectif** — la case *« review process »* de CRISP-DM, systématiquement sautée. Prendre du recul
avant de déployer.

**Ce qu'on fait**
1. Relire les 31 étapes : qu'est-ce qui a été bâclé, qu'est-ce qui reste approximatif ?
2. Confronter les résultats aux **critères de succès de l'étape 1**. Sont-ils atteints ? Sinon, le
   dire.
3. Vérifier qu'aucune hypothèse de l'étape 2 n'a été invalidée en route.
4. **Décision go / no-go** sur le déploiement, argumentée.

**Sortie** — `reports/revue_processus.md`, qui devient la section « limites » du write-up.

**Fin d'étape** — la liste de ce qui est solide et de ce qui ne l'est pas est écrite **avant** de
rédiger le livrable.

---

# PHASE 6 — DEPLOYMENT

## Étape 33 — Persistance du modèle et carte de modèle

**Objectif** — § 4.8 : *« Persist the trained model so it can be reused without retraining. »*

**Ce qu'on fait**
1. Sérialiser le **pipeline complet** (preprocessing + modèle), jamais l'estimateur seul : sinon
   l'application devrait réimplémenter le preprocessing, avec risque de divergence entre
   entraînement et service.
2. Rédiger une **carte de modèle** : version, date, données d'entraînement, périmètre d'usage,
   métriques par classe, limites connues, populations sur lesquelles il ne doit pas être utilisé.

**Outils**

| Outil | Rôle | Pourquoi |
|---|---|---|
| `joblib` | sérialisation | standard scikit-learn, efficace sur les tableaux numpy |
| `skops` | alternative citée | format plus sûr que le pickle ; à mentionner comme le choix correct en production réelle |

**Sortie** — `models/modele_final.joblib`, `models/carte_modele.md`.

**Fin d'étape** — le modèle se recharge et prédit dans un processus neuf, sans réentraînement.

---

## Étape 34 — Application Streamlit

**Objectif** — § 4.8. *« It needs to be usable, clear, and to behave gracefully when some inputs
are missing or unknown. »* Le critère est l'utilisabilité, pas l'esthétique.

**Ce qu'on fait**
1. **Deux modes d'entrée** : sélection d'un client existant par identifiant, ou saisie manuelle.
2. **Sortie** : classe NPS prédite, probabilités des trois classes, niveau de confiance.
3. **Top drivers SHAP locaux** pour ce client précis, formulés en langage métier — pas
   « `contract_month_to_month = 1, SHAP +0.18` » mais « contrat sans engagement : principal facteur
   de risque pour ce client ».
4. **Zone de saisie d'un verbatim**, avec affichage du déplacement de la prédiction (bonus § 4.8).
5. **Vue liste** : les K clients à appeler en priorité, exportable — c'est le livrable que l'équipe
   rétention utilisera réellement.
6. Gestion explicite des champs vides ou modalités inconnues, d'où le `handle_unknown='ignore'`
   décidé à l'étape 15.

**Outils**

| Outil | Rôle | Pourquoi |
|---|---|---|
| Streamlit | interface | déjà installé ; suffisant pour un formulaire et des graphiques, sans code front. **La seule vraie alternative est Gradio** — même catégorie d'outil, le choix relève de la préférence |
| FastAPI | *pas une alternative* | ⚠️ FastAPI est une couche de **service** (exposer le modèle à un autre système), Streamlit une couche d'**interface** (un humain clique). En production on a les deux. L'énoncé les énumère ensemble au § 4.8, ce qui prête à confusion — savoir les distinguer est un point à marquer en entretien |

**Sortie** — `app/`.

**Fin d'étape** — l'application démarre dans un environnement neuf, sans réentraînement, et ne
plante sur aucune saisie incomplète.

---

## Étape 35 — Captures et démonstration

**Objectif** — livrable explicitement listé au § 6 : *« Interface screenshots and/or a short demo
video »*.

**Ce qu'on fait** — captures des trois écrans clés (saisie, prédiction avec drivers, liste
priorisée) et courte vidéo de démonstration commentée.

**Sortie** — `reports/captures/`, `reports/demo.mp4`.

**Fin d'étape** — la démonstration est compréhensible sans commentaire oral.

---

## Étape 36 — Monitoring et réentraînement

**Objectif** — § 4.9. Montrer qu'on sait ce qui casse **après** la mise en production.

**Ce qu'on fait**
1. Dérive des données d'entrée et dérive des prédictions.
2. Suivi de la performance sur les nouvelles réponses d'enquête au fil de l'eau.
3. Déclencheur de réentraînement : seuil de dérive, ou volume minimal de nouveaux labels.
4. **Le point le plus intéressant — la boucle de rétroaction.** Un détracteur contacté puis retenu
   change de classe. Les données d'entraînement futures sont donc contaminées par l'action
   commerciale elle-même : le modèle apprend l'effet de sa propre intervention. Seule parade
   sérieuse — conserver un **groupe de contrôle non contacté**, ce qui a un coût commercial et
   s'arbitre au niveau de la direction.

**L'encadré à placer dans le write-up** — il répond au § 4.9 **et** ouvre la voie au ciblage par
sensibilité de l'étape 27, en une seule recommandation :

> Pour passer du ciblage par risque au ciblage par sensibilité, il faut une seule chose :
> **randomiser**. À la prochaine campagne, sur les K clients prioritaires du modèle, tirer au
> hasard 10 à 20 % qui **ne seront pas contactés**. Ce groupe de contrôle coûte quelques clients
> non retenus. Il achète deux choses irremplaçables : **la mesure de l'effet réel de la
> campagne**, et **les données nécessaires à un modèle d'uplift** au cycle suivant.
>
> Sans lui, il se produit pire que rien : un **empoisonnement des données d'entraînement
> futures**. Un détracteur appelé puis retenu change de classe, et le modèle réapprend l'effet de
> sa propre intervention comme s'il s'agissait d'une propriété du client.

C'est le seul passage du livrable qui **demande une décision à la direction** plutôt que de lui
présenter un résultat. C'est aussi celui qu'un directeur Expérience Client retiendra.

**Outils** — Evidently : open source, rapports HTML autonomes, cité par l'énoncé au § 10.

**Sortie** — `docs/monitoring.md` + rapport Evidently d'exemple.

**Fin d'étape** — le déclencheur de réentraînement est chiffré, et la boucle de rétroaction est
expliquée.

---

## Étape 37 — Write-up final

**Objectif** — **le livrable le plus lu.** 3 à 6 pages qu'un directeur Expérience Client puisse
comprendre et actionner. Artefact vend du conseil : écrire pour un décideur est le cœur du métier.

**Ce qu'on fait** — plan imposé par le § 6, nommé avec le vocabulaire **CRISP-DM** que tout
consultant data reconnaît :

1. Synthèse exécutive (lisible seule)
2. Compréhension métier et critères de succès
3. Compréhension des données — dont les **fuites identifiées**
4. Préparation — dont la **construction de la cible** et l'arbitrage empirique des « 3 »
5. Modélisation — baseline, familles comparées, formulation ordinale
6. Évaluation — technique **et** métier, drivers, équité
7. Déploiement — interface, monitoring
8. Limites et prochaines étapes
9. Déclaration d'usage des outils IA (exigée au § 7)

**Règles de rédaction** — chaque affirmation adossée à un chiffre ; les limites présentées comme
des choix maîtrisés, pas comme des excuses.

**Points forts à mettre en avant, parce qu'ils sont rares** : l'incohérence du lien Kaggle de
l'énoncé, la typologie des trois natures de fuite, le NPS de −42 comme signal d'alerte,
l'arbitrage empirique des « 3 », le protocole MNAR à trois scénarios, la limite de circularité des
verbatims, et le contre-exemple publié de Mustafa et al. (2021).

**Outils** — rédaction en Markdown, conversion en PDF via Chrome en mode headless
(`--headless --print-to-pdf`), méthode déjà validée sur ce projet.

**Sortie** — `livrable/write_up.pdf`.

**Fin d'étape** — un lecteur non technique comprend la recommandation sans lire le code.

---

## Étape 38 — Reproductibilité et remise

**Objectif** — § 7 : *« Someone else should be able to re-run your pipeline from raw data to
prediction. »* Un projet brillant non reproductible est un projet refusé.

**Ce qu'on fait**
1. README : contexte, installation, exécution, arborescence, résultats principaux.
2. `.env.example` sans aucune valeur réelle ; vérification qu'aucune clé n'a jamais été commitée,
   **y compris dans l'historique Git**.
3. **Déclaration explicite de l'usage des outils IA** (§ 7) — ce qui a été généré, assisté ou
   rédigé avec un LLM.
4. **Test à blanc** : cloner dans un répertoire vierge, installer, exécuter le pipeline complet,
   vérifier que les chiffres du write-up sont reproduits à l'identique.
5. Relecture finale : cohérence entre code, rapports et write-up.

**Outils** — Git, pip, l'orchestrateur `python -m src.run`.

**Fin d'étape** — le pipeline complet se rejoue depuis zéro et reproduit les chiffres publiés.

---

# Socle technique

À monter avant l'étape 4.

| Élément | Choix | Pourquoi |
|---|---|---|
| Langage | Python 3.11 | déjà installé, compatible avec toute la chaîne |
| Isolation | `venv` + `pip`, versions figées | suffisant ; conda ajouterait du poids sans bénéfice |
| Configuration | `config.yaml` unique (chemins, seed, paramètres) | aucune constante codée en dur dans les scripts |
| Orchestration | `python -m src.run <etape>` | `make` n'est pas garanti sous Windows ; un point d'entrée Python reste portable |
| Versionnement | Git, `.gitignore` excluant `data/`, `models/`, `.env` | exigé implicitement par le § 7 |

**Arborescence** — `src/` (code), `scripts/` (points d'entrée numérotés),
`data/{raw,interim,processed}/`, `models/`, `reports/`, `notebooks/`, `docs/`, `app/`, `prompts/`,
`livrable/`.

---

# Calendrier

⚠️ **Révisé le 17/09.** Brief récupéré le 09/09, échéance à J+14 vers le **23/09** : il reste
**6 jours**. Le périmètre ne change pas, la cadence double. Si la date de réception réelle diffère,
ce tableau est à refaire.

| Jour | Phases | Étapes |
|---|---|---|
| **Mer 17/09** | 1 et 2 | 1 → 8 — cadrage, ingestion, EDA, qualité, **registre de fuites** |
| **Jeu 18/09** | 3 | 9 → 15 — sélection, **cible + arbitrage des « 3 »**, nettoyage, features, encodage |
| **Ven 19/09** | 4 (début) | 16 → 19 — **protocole 15/85**, baseline, ordinal, boosting |
| **Sam 20/09** | 4 (fin) | 20 → 24 — TabPFN, texte, hyperparamètres, déséquilibre, calibration |
| **Dim 21/09** | 5 (début) | 25 → 28 — évaluation technique, comparaison statistique, métier, sensibilité |
| **Lun 22/09** | 5 (fin) et 6 | 29 → 35 — SHAP, drivers, équité, revue, persistance, **Streamlit** |
| **Mar 23/09** | 6 (fin) | 36 → 38 — monitoring, write-up, reproductibilité, remise |

**Règle qui rend ce calendrier tenable** : le write-up ne s'écrit pas le 23. Chaque étape produit
son rapport dans `reports/` au moment où elle est faite ; le 23 est une journée d'**assemblage et
de relecture**. Une étape dont le rapport n'est pas écrit n'est pas terminée.

---

# Méthode de travail

Pour chaque étape : la décision et ses alternatives sont posées → **tu tranches** → on code → tu
réexpliques la décision avec tes mots avant de passer à la suivante. Si tu ne peux pas l'expliquer,
on ne passe pas.

## Soutenance blanche

Les questions qui tomberont :

1. Pourquoi ce mapping de la satisfaction vers le NPS, et qu'est-ce qui change si on prend l'autre ?
2. Qu'avez-vous fait des clients à satisfaction 3, et sur quelle base ?
3. Pourquoi un traitement ordinal plutôt que nominal, et qu'est-ce que ça a apporté en chiffres ?
4. Comment avez-vous simulé le biais de non-réponse, et de combien la performance chute-t-elle ?
5. Quelles fuites avez-vous trouvées, et comment les avez-vous **détectées** plutôt que devinées ?
6. Votre modèle traite-t-il tous les groupes de clients équitablement ? Comment le savez-vous ?
7. Le texte apporte-t-il quelque chose, et pourquoi votre expérience ne peut pas le prouver ?
8. Qu'est-ce qui casserait ce modèle six mois après sa mise en production ?
