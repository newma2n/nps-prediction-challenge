# Phase 6 — Déploiement

**En bref.** Le modèle est persisté d'un bloc (préprocessing + modèle + calibrateur) et consommé par
une application Streamlit de huit pages, filtrable, pour l'équipe rétention : liste d'appels priorisée, analyse
d'un client avec ses drivers et son levier, saisie manuelle tolérante aux inconnus, verbatim en
démonstration. Un monitoring implémenté surveille la dérive des entrées et des prédictions, la
performance sur les nouvelles réponses d'enquête, et déclenche le réentraînement sur seuil, volume ou
échéance. La décision demandée à la direction : **un groupe de contrôle non contacté** à la prochaine campagne.

## 1. Carte du modèle

| | |
|---|---|
| Nom | Prédiction de catégorie NPS — clients silencieux |
| Version | 1.1 · seed 42 · régénéré par `python -m src.run` |
| Modèle | logistique ordinale (Linéaire, classification ordinale), version réglée, dans un `Pipeline` scikit-learn (imputation + encodage + modèle) |
| Cible | M3 — Détracteur (satisfaction 1–2) · Passif (3) · Promoteur (4–5) |
| Données d'entraînement | 1080 répondants simulés (S3 — MNAR), IBM Telco 11.1.3+ |
| Évaluation | 5963 clients silencieux, vérité connue |
| Kappa quadratique · macro-F1 | 0.384 · 0.490 |
| Rappel / précision Détracteur | 0.63 / 0.44 |
| Décision | hybride — classe : modèle non calibré (la calibration écrasait la classe Passif) ; P(Détracteur) : modèle calibré (Platt), pour le classement des appels |
| Explication | coefficients × valeur (exact, additif) |
| Variables exclues | tout ce qui touche au churn, `CLTV`, démographie, géographie (phase 2 § 4) |
| Usage prévu | prioriser une liste d'appels de rétention mensuelle ; comprendre les drivers |
| Usage **interdit** | décider d'une offre ou d'un tarif individuel ; toute décision fondée sur un attribut protégé ; lire P(Détracteur) comme une fréquence |
| Limites connues | écart de rappel de 26 pts entre groupes d'âge (proxies) ; classe Passif mal rappelée ; probabilités surestimées ; propension à répondre simulée |
| Fichiers | `models/modele_final.joblib` (pipelines calibré, brut, de décision, métadonnées) · `data/processed/clients_scores.parquet` |

## 2. L'application — choix de conception (énoncé § 4.8)

`docker compose up` ou `streamlit run app/app.py` · huit pages, adressables par `?page=` · toutes les valeurs sont lues dans
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

**Livraison par conteneur.** « Someone else should be able to re-run your pipeline » (énoncé § 7) suppose que cette
personne parvienne d'abord à installer l'environnement — ce qui, avec quinze modèles et une contrainte `numpy < 2`,
n'est pas acquis. Le dépôt embarque donc une image Docker :

```bash
docker compose up --build          # l'application sur http://localhost:8501
```

L'image de livraison contient le modèle entraîné, le jeu de données dérivé, les rapports et les figures : **aucune
donnée brute ni clé n'est nécessaire pour voir l'outil**. Elle tourne sans privilèges et expose une sonde de santé.
Un second étage, construit à la demande (`docker compose --profile pipeline run --rm pipeline`), ajoute le catalogue
de modèles et rejoue tout le calcul depuis les cinq classeurs IBM, avec les mêmes versions épinglées — c'est la
réponse la plus forte à l'exigence de reproductibilité, puisqu'elle fige l'environnement en même temps que le code.
Les modèles de fondation (PyTorch, TabICL, TabPFN) restent hors image : ils la feraient passer de 0,8 à plus de 3 Go
pour des résultats déjà calculés et lus dans `reports/fondation.json`.

## 4. Monitoring et réentraînement (énoncé § 4.9) — implémenté


![](figures/06_monitoring.png)

**Référence** = population d'entraînement (répondants S3). **Courant** = population scorée. **Mois
simulé** = tarif +12 %, 15 % des contrats basculés en mensuel, 10 % migrés vers la fibre.

| | PSI des prédictions | Part de détracteurs prédits |
|---|---|---|
| Courant vs référence | 0.025 | 31.4% (réf. 32.9%) |
| Mois simulé vs référence | **0.054** | **36.4%** |

Variables les plus dérivantes — courant :

| Variable | PSI | Statut |
|---|---|---|
| Total Charges | 0.0482 | stable |
| Total Revenue | 0.0443 | stable |
| Monthly Charge | 0.0414 | stable |
| Tenure in Months | 0.0396 | stable |
| Total Long Distance Charges | 0.0385 | stable |
| nb_services | 0.0357 | stable |
| Internet Type | 0.0313 | stable |
| tranche_anciennete | 0.0301 | stable |

Variables les plus dérivantes — mois simulé :

| Variable | PSI | Statut |
|---|---|---|
| Monthly Charge | 1.3813 | ALERTE |
| charge_par_service | 0.6490 | ALERTE |
| Total Charges | 0.0482 | stable |
| Contract | 0.0475 | stable |
| Total Revenue | 0.0443 | stable |
| Tenure in Months | 0.0396 | stable |
| Total Long Distance Charges | 0.0385 | stable |
| nb_services | 0.0357 | stable |

**Déclencheur de dérive sur le mois simulé : DÉCLENCHÉ** (variables en alerte :
Monthly Charge, charge_par_service). Le mécanisme détecte une hausse tarifaire et une migration de contrats
que personne n'aurait signalées au modèle. Note : le PSI « courant » n'est pas nul alors que les deux
populations viennent de la même base — c'est le **biais de sélection S3** qui s'affiche, et c'est
exactement ce que le monitoring doit voir.

**Performance sur les nouvelles réponses d'enquête** (simulation : ~150 réponses par mois, tirées parmi les
silencieux selon la propension biaisée S3) :

![](figures/06_nouvelles_reponses.png)

| Mois | Nouvelles réponses | Cumul | Kappa (lot) | Rappel Dét. (lot) | Kappa (cumul) | Rappel Dét. (cumul) | Écart d'équité (âge, cumul) | Alerte équité | Chute | Volume atteint |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 150 | 150 | 0.510 | 0.583 | 0.510 | 0.583 | 13.6% | ⚠️ |  |  |
| 2 | 150 | 300 | 0.556 | 0.622 | 0.534 | 0.602 | 20.8% | ⚠️ |  |  |
| 3 | 150 | 450 | 0.523 | 0.595 | 0.531 | 0.600 | 12.4% | ⚠️ |  |  |
| 4 | 150 | 600 | 0.600 | 0.641 | 0.549 | 0.612 | 18.2% | ⚠️ |  | ✓ |
| 5 | 150 | 750 | 0.451 | 0.476 | 0.530 | 0.587 | 17.8% | ⚠️ |  | ✓ |
| 6 | 150 | 900 | 0.420 | 0.609 | 0.513 | 0.591 | 13.4% | ⚠️ |  | ✓ |

Référence : kappa 0.384, rappel 0.629. Seuils : chute de kappa > 0.05, chute de rappel
> 0.1, volume minimal 500 nouveaux labels (atteint au mois 4), échéance 6 mois.
Chute détectée sur la simulation : non. réponses tirées parmi les silencieux selon la propension S3 (les extrêmes répondent plus) : la performance sur les nouveaux répondants est donc optimiste par rapport à la base — exactement le biais que le monitoring réel subira.

**L'équité est surveillée, pas seulement auditée une fois.** le rappel Détracteur est recalculé par tranche d'âge sur les réponses cumulées ; une alerte est levée au-delà de 10% d'écart entre le groupe le mieux et le moins bien servi. Un groupe comptant moins de 10 détracteurs n'est pas évalué : le rappel y serait trop bruité pour décider.
Sur cette simulation, l'alerte d'équité se déclenche dès le mois 1 —
ce qui est attendu : l’écart mesuré en phase 5 dépasse le seuil de signalement, et le monitoring le retrouve sur les nouvelles réponses.

**Les déclencheurs de réentraînement, et leur état sur le mois simulé** — on montre qu'ils se déclenchent, on ne se contente pas de les annoncer :

| Déclencheur | Règle | Déclenché sur la simulation |
|---|---|---|
| dérive des entrées | PSI > 0.2 sur au moins une variable du modèle | ⚠️ oui |
| chute de performance | kappa cumulé < référence − 0,05 ou rappel Détracteur < référence − 0,10 | non |
| volume de nouveaux labels | 500 nouvelles réponses d'enquête cumulées | ⚠️ oui |
| équité | écart de rappel Détracteur entre tranches d'âge > 10% | ⚠️ oui |
| échéance | réentraînement au plus tard tous les 6 mois | ⚠️ oui |


**Ce qu'on surveille, et quand on réentraîne** :

| Signal | Mesure | Déclencheur |
|---|---|---|
| Dérive des entrées | PSI par variable, mensuel, vs population d'entraînement | PSI > 0,20 sur une variable majeure (`Contract`, `Tenure`, `Monthly Charge`) |
| Dérive des prédictions | PSI de P(Détracteur) ; part de Détracteurs prédits | PSI > 0,20 ou variation > 5 points sans cause métier connue |
| Performance réelle | kappa et rappel Détracteur sur les **nouvelles réponses d'enquête** | chute > 0,05 de kappa ou > 10 points de rappel |
| Équité | rappel Détracteur par tranche d'âge sur le **cumul** des nouvelles réponses — **implémenté**, drapeau `alerte_equite` | écart > 10 points, le seuil de la phase 1 |
| Volume de labels | nouvelles réponses reçues | réentraînement dès 500 nouveaux labels, au plus tard tous les 6 mois |
| Calibration | ECE sur les nouvelles réponses | recalibrer (Platt) sur les vraies réponses de la population cible dès 300 labels |

*Le seuil d'équité est **le même partout dans l'étude** — celui fixé à la phase 1. Une version
antérieure de cette carte en annonçait 15 quand le critère de succès en annonçait 10 : deux seuils
pour une même règle, donc aucune règle. L'écart n'est lisible qu'en **cumul** : sur 150 réponses par
mois, un groupe d'âge compte trop peu de détracteurs pour qu'un rappel y soit interprétable, et le
monitoring n'évalue donc un groupe qu'à partir de dix détracteurs observés.*

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

## Décisions prises dans cette phase

| Décision | Justification | Où c'est vérifié |
|---|---|---|
| Persistance d'un bloc : pipelines calibré, brut, de décision + métadonnées | rejouable sans réentraînement ; l'application n'a aucune logique de préparation | § 1, `models/modele_final.joblib` |
| Streamlit, huit pages filtrables, contenu lu dans les artefacts | interface pour un responsable rétention ; aucune valeur recopiée ; les filtres croisés évitent de chercher dans 5 963 lignes | § 2 |
| Tolérance aux inconnus par le pipeline, pas par l'interface | un seul endroit à tester | § 2, test dédié |
| Monitoring : PSI + performance sur nouvelles réponses + volume + équité + échéance | cinq déclencheurs, chacun avec sa règle et son état sur le mois simulé | § 4 |
| Groupe de contrôle recommandé | seule façon de mesurer l'effet et d'éviter la boucle de rétroaction | § 5 |
