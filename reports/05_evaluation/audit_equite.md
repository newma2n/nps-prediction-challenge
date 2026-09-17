# Phase 5 — Audit d'équité

Un modèle qui priorise le budget de rétention peut le répartir inégalement. L'énoncé : capter 80 %
des jeunes détracteurs mais 50 % des seniors est inacceptable, même à exactitude globale élevée.

Les variables démographiques sont **exclues du modèle** et servent **uniquement ici**.

## Rappel Détracteur par sous-groupe

![](figures/05_equite.png)

| Dimension | Groupe | Clients | Détracteurs | Rappel Détracteur |
|---|---|---|---|---|
| Gender | Female | 2939 | 540 | 80.2% |
| Gender | Male | 3024 | 543 | 76.8% |
| Senior Citizen | No | 5019 | 816 | 75.0% |
| Senior Citizen | Yes | 944 | 267 | 89.1% |
| Married | No | 3110 | 697 | 79.5% |
| Married | Yes | 2853 | 386 | 76.7% |
| Dependents | No | 4568 | 1019 | 78.8% |
| Dependents | Yes | 1395 | 64 | 73.4% |
| tranche_age | 30-45 | 1655 | 271 | 75.3% |
| tranche_age | 45-60 | 1617 | 276 | 83.7% |
| tranche_age | 60+ | 1397 | 350 | 86.0% |
| tranche_age | <30 | 1294 | 186 | 61.3% |

| Dimension | Écart max (points) |
|---|---|
| Gender | 3.4 |
| Senior Citizen | 14.1 |
| Married | 2.8 |
| Dependents | 5.4 |
| tranche_age | 24.7 |

## Ce qu'on trouve

**L'écart le plus important porte sur l'âge : 25 points** entre le groupe le mieux servi et le moins bien
servi. Concrètement, le modèle rappelle 61% des détracteurs de moins de 30 ans contre
86% des 60 ans et plus. **C'est le cas de figure que l'énoncé décrit comme inacceptable, en
sens inverse** : ce sont les jeunes détracteurs qu'on rate.

Cause plausible : les jeunes sont sur-représentés dans les contrats mensuels et les faibles
anciennetés, mais avec des profils de services plus hétérogènes ; le modèle, sans variable d'âge,
les confond davantage avec les passifs. À investiguer avant toute mise en production.

## Coût de l'exclusion des démographiques

| | Kappa | Macro-F1 | Rappel Détracteur |
|---|---|---|---|
| Sans démographie (retenu) | 0.370 | 0.497 | 0.785 |
| Avec démographie | 0.372 | 0.501 | 0.790 |

L'exclusion coûte **-0.3 point de kappa**. Le choix est défendable : un gain marginal ne justifie pas de
décider sur l'âge ou le genre.

## À remonter avant production

1. L'écart d'âge de 25 points — à l'équipe Expérience Client et, selon la juridiction, au juridique.
2. Le fait que l'exclusion des démographiques **ne suffit pas** à garantir l'équité : le modèle
   les retrouve indirectement via les proxies contractuels. L'audit doit être répété à chaque
   réentraînement.
3. Une piste : seuil de décision **par groupe d'âge** pour égaliser les rappels — au prix d'une
   précision différente selon les groupes. Arbitrage à trancher par le métier, pas par le modèle.
