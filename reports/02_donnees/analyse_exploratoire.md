# Phase 2 — Analyse exploratoire

## La cible

![](figures/02_eda_satisfaction.png)

| Note | Clients | Part |
|---|---|---|
| 1 | 922 | 13.1% |
| 2 | 518 | 7.4% |
| 3 | 2665 | 37.8% |
| 4 | 1789 | 25.4% |
| 5 | 1149 | 16.3% |

**Le fait central du projet** : la note 3 pèse **2665 clients, 37.8% de la base**, et
2236 d'entre eux (84%) sont toujours clients.
Où les placer dans la grille NPS décide de tout le reste — voir phase 3.

## Variables numériques par niveau de satisfaction

![](figures/02_eda_numeriques.png)

L'ancienneté et le nombre de parrainages croissent nettement avec la satisfaction ; la charge
mensuelle varie peu. Le parrainage est un proxy comportemental direct de la recommandation —
c'est-à-dire du NPS lui-même — sans être une fuite : c'est un comportement passé observable.

## Variables catégorielles : force d'association

![](figures/02_eda_cramer.png)

Le **type de contrat** et l'**offre reçue** dominent. Les variables démographiques (gris) sont
faiblement associées à la satisfaction — ce qui rend leur exclusion peu coûteuse, et sera vérifié.

## Corrélations entre numériques

![](figures/02_eda_correlations.png)

`Total Charges`, `Total Revenue` et `Tenure in Months` sont fortement corrélées (mécaniquement :
le total cumule le mensuel sur la durée). Traité en phase 3 (colinéarité).

## La quasi-expérience du dataset : `Offer`

![](figures/02_eda_offer.png)

| Offre | Clients | Taux de départ | Satisfaction moy. |
|---|---|---|---|
| Offer A | 520 | 6.7% | 3.55 |
| Offer B | 824 | 12.3% | 3.48 |
| Offer C | 415 | 22.9% | 3.32 |
| Offer D | 602 | 26.7% | 3.25 |
| Aucune offre | 3877 | 27.1% | 3.24 |
| Offer E | 805 | 52.9% | 2.81 |

Les clients ayant reçu l'**offre E partent deux fois plus** que ceux qui n'ont rien reçu. Lire
« l'offre E fait fuir » serait une erreur : c'est un **biais d'indication** — l'offre a été
proposée à des clients déjà en difficulté, comme un médicament à des patients déjà malades.
**On ne lit pas un effet causal dans des données observationnelles.** Ce constat fonde la section
« limites de la règle de ciblage » en phase 5.
