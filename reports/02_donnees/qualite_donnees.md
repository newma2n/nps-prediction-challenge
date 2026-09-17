# Phase 2 — Qualité des données

Cinq contrôles, aucun facultatif.

## 1. Valeurs manquantes

| Colonne | Manquants | Part | Nature |
|---|---|---|---|
| Churn Category | 5174 | 73.5% | structurel |
| Churn Reason | 5174 | 73.5% | structurel |
| Offer | 3877 | 55.0% | structurel |
| Internet Type | 1526 | 21.7% | structurel |

Les quatre colonnes concernées ont des manquants **structurels**, pas accidentels :
- `Offer` vide = le client n'a reçu aucune offre → encodé en catégorie « Aucune offre » ;
- `Internet Type` vide = pas d'abonnement internet → encodé « Pas d'internet » ;
- `Churn Category` / `Churn Reason` vides = le client n'est pas parti. Ces deux colonnes sont
  de toute façon exclues : leur seule présence trahit la cible (fuite de disponibilité).

**Aucune imputation** n'est faite sur ces colonnes. Les imputer aurait fabriqué de l'information.

## 2. Doublons

- Sur `Customer ID` : **0**
- Sur l'ensemble des colonnes hors identifiant : **0**

## 3. Valeurs aberrantes

![](figures/02_boites_aberrantes.png)

| Variable | Hors IQR × 1,5 |
|---|---|
| Tenure in Months | 0 |
| Monthly Charge | 0 |
| Total Charges | 0 |
| Total Revenue | 21 |
| Avg Monthly GB Download | 362 |

Ces valeurs sont **conservées**. Un client à 100 Go/mois ou à 120 $/mois est un client réel, pas
une erreur de saisie, et c'est précisément le type de profil qu'un modèle de détraction doit
voir. Aucune winsorisation : les modèles retenus (logistique standardisée, arbres) y sont robustes.

## 4. Cohérence interne

- Clients à ancienneté 0 mois : **0**
- Écart > 50 % entre `Total Charges` et `Monthly Charge × Tenure` : **2** lignes.
  Attendu : le tarif mensuel a pu changer sur la durée de vie du client. Non corrigé, documenté.

## 5. Colonnes constantes

`Country`, `State` — base 100 % Californie. Supprimées.
