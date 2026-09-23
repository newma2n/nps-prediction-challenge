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

Les seuils ci-dessous sont écrits dans `config.yaml` (section `criteres_succes`), vérifiés par le
pipeline (clé `criteres_succes` de `resultats.json`) et verrouillés par un test automatisé. Un
critère qu'on ne peut pas rater n'est pas un critère : chacun a donc une valeur numérique décidée
avant la modélisation, et le tableau affiche le résultat obtenu à côté.

| Niveau | Critère | Cible chiffrée a priori | Résultat (phase 5) | |
|---|---|---|---|---|
| Métier | Précision@K sur les détracteurs, K = 500 appels/mois | ≥ 35%, et lift ≥ 2 | 51% vs 18% de base (×2.8) | ✅ |
| Métier | Gain net d'une campagne ciblée vs ciblage aléatoire | > 0 € | 18 585 € apportés par le modèle | ✅ |
| Data science | Kappa quadratique pondéré, modèle retenu vs baseline logistique | au niveau de la baseline ou au-dessus | 0.384 contre 0.371 | ✅ |
| Data science | Rappel Détracteur, modèle retenu vs baseline logistique | au niveau de la baseline ou au-dessus | 0.629 contre 0.805 | ⚠️ **non atteint** — voir ci-dessous |
| Data science | Aucune classe abandonnée | rappel ≥ 5% sur les trois classes | minimum 42% | ✅ |
| Éthique | Écart de rappel Détracteur entre sous-groupes | mesuré, **signalé** au-delà de 10% | 26 pts sur l'âge | ⚠️ signalé (phase 5 § 7) |

**Le critère de rappel Détracteur n'est pas atteint, et ce n'est pas un oubli.** La baseline logistique capte 81% des détracteurs contre 63% pour le modèle retenu. Elle y parvient en prédisant « Détracteur » beaucoup plus souvent : sa précision est plus faible, et surtout elle écrase la classe Passif. Le critère métier — qui décide réellement de l'usage — départage dans l'autre sens : à capacité d'appel égale, c'est la précision@K qui compte, et elle vaut 51%. Le rappel brut d'un classifieur qui sur-prédit une classe n'est pas une performance ; on le publie en défaveur du modèle retenu, avec la raison de ne pas le suivre. Détail en phase 4 § 6.


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

### 5.1 Contraintes de données

Dataset fictif (IBM), une seule photographie, aucun historique, aucune campagne de rétention
enregistrée. Conséquence directe : **aucune estimation d'effet causal n'est possible** — le modèle
classe par risque, pas par sensibilité à l'appel (phase 5 § 8, phase 6 § 5).

### 5.2 Contraintes de projet : deux semaines, et la gestion du périmètre

L'énoncé fixe une limite de deux semaines et annonce valoriser *« thoughtful scope management more
than over-engineering »* (§ 1). Le périmètre a donc été arbitré explicitement, et les refus sont
documentés au même titre que les ajouts.

| Décision | Sens | Raison |
|---|---|---|
| Quinze modèles, sept familles | **dans** le périmètre | l'exigence minimale est « au moins deux familles ». La grille sert à **mesurer le plafond de signal** des données, pas à chercher un meilleur score : quand quinze modèles très différents plafonnent au même endroit, la limite vient des données, et c'est cette conclusion qui est utile au métier. Coût réel : 35 minutes d'exécution, une seule fois. |
| CORAL / CORN (réseaux ordinaux) | **hors** périmètre | environ 1 000 lignes d'entraînement : un réseau profond n'a aucune chance de battre un modèle linéaire régularisé. Cités et écartés avec l'argument (phase 4 § 3). |
| Enrichissement Census par code postal | **hors** périmètre | voir phase 2 § 6 : l'affectation des clients aux codes postaux est produite par le générateur d'IBM. |
| MLflow | **hors** périmètre | une seule exécution déterministe, tracée par `resultats.json` et le seed : un serveur de suivi n'ajouterait rien ici. |
| TabPFN 2.5 | **partiellement hors** périmètre | bloqué par une acceptation de licence externe ; TabICL et TabPFN v2 sont évalués à sa place (phase 4 § 8). |
| Appels d'API payants pour les verbatims | **hors** périmètre | le bonus texte est traité avec des textes rédigés hors ligne et assemblés localement, de façon reproductible (phase 4 § 9). |

Ce que cette contrainte **n'a pas** permis : entraîner un modèle par segment, tester l'enrichissement
externe, et mener une véritable campagne avec groupe de contrôle — la seule façon de passer du
risque à la sensibilité (phase 6 § 5).

## 6. Registre de risques

| Risque | Parade dans l'étude |
|---|---|
| Fuite de données → score irréel, modèle inutilisable | registre de fuites quantifié (phase 2 § 4) + double diagnostic (phase 4 § 5) |
| Cible mal construite → tout le reste invalide | arbitrage empirique des « 3 » + sensibilité au mapping + bruit d'étiquettes (phases 3, 5) |
| Sélection de modèle sur le jeu de test | sélection par validation croisée sur les répondants ; les silencieux ne servent qu'au test (phase 4 § 6) |
| Modèle inéquitable non détecté | audit par sous-groupe, audit des proxies, mitigation chiffrée (phase 5 § 7) |
| Livrable non reproductible | seed unique, `python -m src.run`, test à blanc depuis copie vierge, tests automatisés |

## Décisions prises dans cette phase

| Décision | Justification | Où c'est vérifié |
|---|---|---|
| Cible traitée comme ordonnée et déséquilibrée | trois classes ordonnées par construction ; Passif = 38% de la base, Détracteur = 20% | phase 3 § 4 |
| Métrique principale = kappa quadratique pondéré | seule métrique usuelle qui distingue une erreur d'un cran d'une erreur de deux crans | phase 5 § 1 |
| Critères de succès chiffrés avant modélisation | exigence CRISP-DM ; évite de choisir la métrique qui arrange | ce document § 2 |
