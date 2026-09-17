# Phase 5 — Interprétabilité et drivers de détraction

Le modèle retenu est linéaire : ses coefficients **sont** l'explication, exacte et additive. Sur
des variables standardisées, |coef| est une importance comparable. Pour un client donné,
coef × valeur est sa contribution — c'est ce qu'affiche l'application.

![](figures/05_interpretabilite.png)

## Table de convergence — deux méthodes, deux modèles

| Variable | Rang logistique | Rang SHAP (HGB) |
|---|---|---|
| Number of Referrals | 1 | 1 |
| Contract_Two Year | 2 | 2 |
| Tenure in Months | 3 | 7 |
| Contract_Month-to-Month | 4 | 3 |
| Monthly Charge | 5 | 4 |
| tranche_anciennete_6-12m | 6 | 29 |
| a_parraine | 7 | 58 |
| Offer_Offer D | 8 | 40 |
| tranche_anciennete_0-6m | 9 | 58 |
| Offer_Offer E | 10 | 39 |
| tranche_anciennete_4a+ | 11 | 58 |
| Online Security_No | 12 | 13 |
| Phone Service_Yes | 13 | 58 |
| Online Security_Yes | 14 | 19 |
| Phone Service_No | 15 | 58 |

Une variable haut placée dans les deux classements est un driver solide ; une variable présente
dans un seul est fragile et ne va pas dans la synthèse. Méthode reprise d'Elero et al. (2026).

## Ce que le modèle dit — et ce qu'il ne dit pas

Les associations sont fortes et cohérentes avec la littérature (Chong et al. 2023 sur ce même
dataset ; Elero et al. 2026) : **contrat sans engagement, faible ancienneté, absence de parrainage,
charge mensuelle élevée** vont avec la détraction. `Offer_Offer E` ressort aussi — rappel de la
phase 2 : c'est un marqueur de clients déjà fragiles, pas une cause.

**Le modèle établit des associations, pas des causes.** Un coefficient positif est un signal, pas
un levier prouvé.

## Drivers par segment

| Segment | Clients | Vrais détracteurs | Prédits détracteurs | P(détracteur) moy. |
|---|---|---|---|---|
| Contrat mensuel | 3087 | 32.1% | 63.2% | 0.50 |
| Contrat 1–2 ans | 2876 | 3.2% | 7.0% | 0.16 |
| Ancienneté < 12 mois | 1815 | 33.8% | 59.2% | 0.49 |
| Ancienneté ≥ 24 mois | 3266 | 9.1% | 20.3% | 0.24 |
| Fibre | 2510 | 27.8% | 56.5% | 0.46 |
| DSL | 1454 | 13.3% | 28.6% | 0.30 |

Le modèle concentre le risque prédit sur les contrats mensuels et les clients récents — segments
où le taux réel de détraction est effectivement le plus élevé.

## Actionnable ou non

| Driver | Actionnable ? | Levier |
|---|---|---|
| Contrat mensuel | **oui** | proposer un engagement 1 an avec avantage |
| Faible ancienneté | partiellement | parcours d'accueil renforcé les 6 premiers mois |
| Charge mensuelle élevée / par service | **oui** | revue tarifaire, bundle |
| Absence de parrainage | indirectement | programme de parrainage |
| Offre E reçue | **signal, pas levier** | ne pas « supprimer l'offre E » — comprendre à qui elle a été proposée |
| Type de connexion | non à court terme | investissement réseau |

**Pour un détracteur prédit, le levier unique le plus probable** : s'il est en contrat mensuel,
proposer un engagement ; sinon, revoir la charge mensuelle.
