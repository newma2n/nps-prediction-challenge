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

| Niveau | Critère | Cible | Résultat (phase 5) |
|---|---|---|---|
| Métier | Précision@K sur les détracteurs, K = 500 appels/mois | nettement au-dessus du taux de base | 51% vs 18% (×2.8) |
| Métier | Gain net d'une campagne ciblée vs ciblage aléatoire | positif, chiffré en euros | 18 585 € par campagne |
| Data science | Kappa quadratique pondéré et rappel Détracteur | au niveau de la baseline ou au-dessus | voir phase 4 § 6 |
| Data science | Aucune classe abandonnée | rappel > 0,05 sur les trois classes | vérifié (test automatisé) |
| Éthique | Écart de rappel Détracteur entre sous-groupes | mesuré, signalé s'il dépasse 10 points | 26 pts sur l'âge, signalé |

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

Dataset fictif (IBM), une seule photographie, aucun historique, aucune campagne de rétention
enregistrée. Conséquence directe : **aucune estimation d'effet causal n'est possible** — le modèle
classe par risque, pas par sensibilité à l'appel (phase 5 § 8, phase 6 § 5).

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
| Cible traitée comme ordonnée et déséquilibrée | trois classes ordonnées par construction ; Passif = 38 % de la base, Détracteur = 20 % | phase 3 § 4 |
| Métrique principale = kappa quadratique pondéré | seule métrique usuelle qui distingue une erreur d'un cran d'une erreur de deux crans | phase 5 § 1 |
| Critères de succès chiffrés avant modélisation | exigence CRISP-DM ; évite de choisir la métrique qui arrange | ce document § 2 |
