# Phase 3 — Nettoyage, feature engineering, colinéarité, pipeline

## Nettoyage

- Manquants structurels (`Offer`, `Internet Type`) → catégorie explicite, jamais imputés.
- Manquants accidentels → imputation **dans le pipeline** (médiane / modalité la plus fréquente),
  jamais en amont : sinon les statistiques calculées sur tout le jeu fuient vers l'évaluation.
- Aberrantes conservées (voir `02_donnees/qualite_donnees.md`).

## Variables dérivées — chacune adossée à une hypothèse métier

| Variable | Hypothèse |
|---|---|
| `nb_services` | intensité de la relation commerciale |
| `charge_par_service` | perception du rapport qualité-prix |
| `tranche_anciennete` | l'insatisfaction d'un nouveau client n'a pas la même nature que celle d'un ancien |
| `a_recu_offre` | un geste commercial a-t-il été fait ? |
| `a_eu_remboursement` | trace d'incident de facturation |
| `a_frais_supplementaires` | trace d'incident de facturation |
| `revenu_par_mois` | intensité économique de la relation |
| `a_parraine` | proxy comportemental direct de la recommandation |

## Variables retenues

**17 numériques** : `Number of Referrals`, `Tenure in Months`, `Avg Monthly Long Distance Charges`, `Avg Monthly GB Download`, `Monthly Charge`, `Total Charges`, `Total Refunds`, `Total Extra Data Charges`, `Total Long Distance Charges`, `Total Revenue`, `nb_services`, `charge_par_service`, `a_recu_offre`, `a_eu_remboursement`, `a_frais_supplementaires`, `revenu_par_mois`, `a_parraine`

**17 catégorielles** : `Offer`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Internet Type`, `Online Security`, `Online Backup`, `Device Protection Plan`, `Premium Tech Support`, `Streaming TV`, `Streaming Movies`, `Streaming Music`, `Unlimited Data`, `Contract`, `Paperless Billing`, `Payment Method`, `tranche_anciennete`

## Colinéarité (VIF)

| Variable | VIF |
|---|---|
| Total Long Distance Charges | 1000000000.0 |
| Total Charges | 1000000000.0 |
| Total Refunds | 1000000000.0 |
| Total Extra Data Charges | 1000000000.0 |
| Total Revenue | 1000000000.0 |
| revenu_par_mois | 86.5 |
| Monthly Charge | 68.1 |
| nb_services | 19.4 |
| Avg Monthly Long Distance Charges | 18.7 |
| Tenure in Months | 7.6 |
| charge_par_service | 4.9 |
| a_eu_remboursement | 4.3 |
| a_frais_supplementaires | 3.1 |
| a_parraine | 2.1 |
| Number of Referrals | 2.1 |
| Avg Monthly GB Download | 1.3 |
| a_recu_offre | 1.0 |

`Total Charges`, `Total Revenue`, `Tenure` et `revenu_par_mois` sont mécaniquement liés (VIF élevés).
**Décision : conservées.** Le modèle retenu en phase 4 est une régression logistique régularisée
(L2) — la régularisation absorbe la colinéarité — et les arbres y sont insensibles. La colinéarité
brouillerait l'*interprétation coefficient par coefficient* ; l'interprétabilité en phase 5 en
tient compte en lisant les variables corrélées comme un bloc.

## Pipeline de préparation

Tout est encapsulé dans un `Pipeline` scikit-learn (`ColumnTransformer`) :
- numériques → imputation médiane → `StandardScaler` **pour le modèle linéaire seulement**
  (inutile pour les arbres : deux pipelines, pas un compromis) ;
- catégorielles → imputation mode → `OneHotEncoder(handle_unknown='ignore')`, pour que
  l'application tolère une modalité jamais vue sans planter.

Rien n'est transformé hors du pipeline. C'est ce qui rend le modèle sérialisable d'un bloc et
exempt de fuite train → test.
