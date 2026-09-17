# Phase 2 — Dictionnaire des données

Cinq fichiers IBM Telco Customer Churn 11.1.3+, joints sur `Customer ID` (et `Zip Code` pour la
population). **7043 lignes × 51 colonnes** après jointure.

Le statut de chaque colonne vient du registre de fuites (`registre_fuites.md`) :
`feature` = variable explicative candidate · `audit_seulement` = exclue du modèle, utilisée
pour l'audit d'équité · `fuite_*` = interdite · `techniques` = identifiant ou constante.

| Colonne | Type | Manquants | Modalités | Exemple | Statut |
|---|---|---|---|---|---|
| Customer ID | object | 0.0% | 7043 | 8779-QRDMV | techniques |
| Gender | object | 0.0% | 2 | Male | audit_seulement |
| Age | int64 | 0.0% | 62 | 78 | audit_seulement |
| Under 30 | object | 0.0% | 2 | No | audit_seulement |
| Senior Citizen | object | 0.0% | 2 | Yes | audit_seulement |
| Married | object | 0.0% | 2 | No | audit_seulement |
| Dependents | object | 0.0% | 2 | No | audit_seulement |
| Number of Dependents | int64 | 0.0% | 10 | 0 | audit_seulement |
| Country | object | 0.0% | 1 | United States | techniques |
| State | object | 0.0% | 1 | California | techniques |
| City | object | 0.0% | 1106 | Los Angeles | audit_seulement |
| Zip Code | int64 | 0.0% | 1626 | 90022 | audit_seulement |
| Lat Long | object | 0.0% | 1679 | 34.02381, -118.156582 | techniques |
| Latitude | float64 | 0.0% | 1626 | 34.02381 | audit_seulement |
| Longitude | float64 | 0.0% | 1625 | -118.156582 | audit_seulement |
| Referred a Friend | object | 0.0% | 2 | No | feature |
| Number of Referrals | int64 | 0.0% | 12 | 0 | feature |
| Tenure in Months | int64 | 0.0% | 72 | 1 | feature |
| Offer | object | 55.0% | 5 | Offer E | feature |
| Phone Service | object | 0.0% | 2 | No | feature |
| Avg Monthly Long Distance Charges | float64 | 0.0% | 3584 | 0.0 | feature |
| Multiple Lines | object | 0.0% | 2 | No | feature |
| Internet Service | object | 0.0% | 2 | Yes | feature |
| Internet Type | object | 21.7% | 3 | DSL | feature |
| Avg Monthly GB Download | int64 | 0.0% | 50 | 8 | feature |
| Online Security | object | 0.0% | 2 | No | feature |
| Online Backup | object | 0.0% | 2 | No | feature |
| Device Protection Plan | object | 0.0% | 2 | Yes | feature |
| Premium Tech Support | object | 0.0% | 2 | No | feature |
| Streaming TV | object | 0.0% | 2 | No | feature |
| Streaming Movies | object | 0.0% | 2 | Yes | feature |
| Streaming Music | object | 0.0% | 2 | No | feature |
| Unlimited Data | object | 0.0% | 2 | No | feature |
| Contract | object | 0.0% | 3 | Month-to-Month | feature |
| Paperless Billing | object | 0.0% | 2 | Yes | feature |
| Payment Method | object | 0.0% | 3 | Bank Withdrawal | feature |
| Monthly Charge | float64 | 0.0% | 1585 | 39.65 | feature |
| Total Charges | float64 | 0.0% | 6540 | 39.65 | feature |
| Total Refunds | float64 | 0.0% | 500 | 0.0 | feature |
| Total Extra Data Charges | int64 | 0.0% | 16 | 20 | feature |
| Total Long Distance Charges | float64 | 0.0% | 6110 | 0.0 | feature |
| Total Revenue | float64 | 0.0% | 6996 | 59.65 | feature |
| Satisfaction Score | int64 | 0.0% | 5 | 3 | cible |
| Customer Status | object | 0.0% | 3 | Churned | fuite_consequence |
| Churn Label | object | 0.0% | 2 | Yes | fuite_consequence |
| Churn Value | int64 | 0.0% | 2 | 1 | fuite_consequence |
| Churn Score | int64 | 0.0% | 81 | 91 | fuite_modele_amont |
| CLTV | int64 | 0.0% | 3438 | 5433 | fuite_modele_amont |
| Churn Category | object | 73.5% | 5 | Competitor | fuite_disponibilite |
| Churn Reason | object | 73.5% | 20 | Competitor offered mor | fuite_disponibilite |
| Population | int64 | 0.0% | 1569 | 68701 | audit_seulement |
