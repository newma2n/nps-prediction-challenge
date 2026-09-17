# Phase 2 — Registre de fuites

**L'étape la plus discriminante du challenge.** Le dataset contient plusieurs colonnes qui
donneraient un score quasi parfait et un modèle inutilisable en production.

## Le fait fondateur

| Satisfaction | Restés | Partis | % départ |
|---|---|---|---|
| 1.0 | 0.0 | 922.0 | 100.0 |
| 2.0 | 0.0 | 518.0 | 100.0 |
| 3.0 | 2236.0 | 429.0 | 16.1 |
| 4.0 | 1789.0 | 0.0 | 0.0 |
| 5.0 | 1149.0 | 0.0 | 0.0 |

Satisfaction 1–2 → **100 % de départs**. Satisfaction 4–5 → **0 %**. Connaître le churn, c'est
connaître la cible. Or au moment où l'on veut détecter un détracteur, il n'est pas encore parti :
l'information n'existe pas. Toute colonne liée au churn est donc une fuite.

## Trois natures de fuite

| Nature | Définition | Colonnes |
|---|---|---|
| **Conséquence** | la colonne est un effet de la cible | `Churn Label`, `Churn Value`, `Customer Status` |
| **Disponibilité** | la colonne n'existe que pour une partie de la population — sa présence trahit la réponse | `Churn Category`, `Churn Reason` (5 174 vides = clients restés) |
| **Modèle amont** | la colonne est la sortie d'un autre modèle (IBM SPSS) | `Churn Score`, `CLTV` |

`Churn Score` et `CLTV` par niveau de satisfaction :

| Satisfaction | Churn Score moyen | CLTV moyenne |
|---|---|---|
| 1.0 | 82.0 | 4158.0 |
| 2.0 | 82.0 | 4104.0 |
| 3.0 | 55.0 | 4473.0 |
| 4.0 | 50.0 | 4514.0 |
| 5.0 | 50.0 | 4381.0 |

`Churn Score` passe de ~82 à ~50 : c'est une prédiction déguisée en donnée. `CLTV`
est plate — peu informative comme feature, mais **légitime comme paramètre de décision** (pondérer
qui appeler parmi les détracteurs prédits). Une même variable, bannie de la couche de modélisation
et admise dans la couche de décision.

## Quantifier au lieu d'affirmer

![](figures/02_information_mutuelle.png)

| Colonne | Info. mutuelle avec la cible | Statut |
|---|---|---|
| Customer Status | 0.421 | fuite_consequence |
| Churn Value | 0.416 | fuite_consequence |
| Churn Score | 0.281 | fuite_modele_amont |
| Contract | 0.092 | feature |
| Number of Referrals | 0.058 | feature |
| Internet Type | 0.052 | feature |
| Tenure in Months | 0.046 | feature |
| Monthly Charge | 0.045 | feature |
| Offer | 0.033 | feature |
| CLTV | 0.016 | fuite_modele_amont |

Les colonnes exclues portent une information mutuelle avec la cible sans commune mesure avec les
features légitimes. C'est la mesure qui justifie l'exclusion, pas l'intuition.

## Démographie et géographie : exclues du modèle, gardées pour l'audit

`Gender`, `Age`, `Under 30`, `Senior Citizen`, `Married`, `Dependents`, `Number of Dependents`, `City`, `Zip Code`, `Latitude`, `Longitude`, `Population`

Précédent : Elero et al. (2026) excluent délibérément les démographiques et obtiennent des
drivers exclusivement liés à l'expérience de service. Position cohérente avec le métier (les
leviers actionnables sont contractuels) et qui évite de défendre un proxy socio-économique.
Le coût en performance de cette exclusion est chiffré en phase 5.

## Contre-exemple publié

Mustafa et al. (2021, *F1000Research*) annoncent 98 % d'exactitude en prédisant le churn — avec la
note NPS et la catégorie NPS dans les features, alors que leur cible en est dérivée. Leurs modèles
linéaires font 41 %, leurs arbres 98 % : cet écart est la **signature** d'une variable qui fuit.
On applique ce test en phase 4.
