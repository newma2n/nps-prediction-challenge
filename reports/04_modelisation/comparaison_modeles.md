# Phase 4 — Baseline, familles de modèles, sélection

## Trois familles

| Famille | Modèle | Pourquoi |
|---|---|---|
| Baseline | régression logistique multinomiale, `class_weight='balanced'` | le plancher honnête exigé par l'énoncé ; interprétable |
| Ordinale | régression HGB sur la satisfaction 1–5 puis **deux seuils optimisés sur le kappa quadratique** | respecte l'ordre des classes par construction |
| Boosting | `HistGradientBoostingClassifier` (scikit-learn) | premier sur neuf algorithmes chez Elero et al. (2026) pour une cible NPS à 3 classes |

CORAL/CORN (cités par l'énoncé) sont des méthodes d'apprentissage profond : sur 7 043 lignes
tabulaires, écartées avec argument, pas ignorées.

## La grille complète — 27 combinaisons

![](figures/04_grille_kappa.png)

Kappa quadratique sous le scénario réaliste S3 :

| Modèle | M1 | M2 | M3 |
|---|---|---|---|
| baseline_logistique | 0.231 | 0.342 | 0.370 |
| boosting_hgb | 0.219 | 0.302 | 0.343 |
| ordinal_seuils | 0.244 | 0.346 | 0.368 |

Deux lectures :
- **M3 domine M1 et M2 pour toutes les familles** : la cible arbitrée par les données est aussi la
  plus apprenable.
- **La performance chute de S1 à S3** pour la plupart des combinaisons : c'est le coût mesuré du
  biais de non-réponse — le résultat scientifique du protocole.

## Porte de contrôle : y a-t-il une fuite ?

| Diagnostic | Valeur | Seuil | Verdict |
|---|---|---|---|
| Exactitude maximale observée | 0.611 | 0,85 (Elero : 0,77 ; Chong : 0,80) | ✅ dans la fourchette honnête |
| Écart arbres − linéaire (S1, M3) | -0.055 | +0,30 (signature Mustafa et al.) | ✅ aucun écart suspect |

## Sélection du modèle final — par la mesure

Critère : kappa quadratique puis macro-F1, scénario S3_MNAR, mapping M3.

| modele | kappa | macro_f1 | rappel_detracteur |
|---|---|---|---|
| baseline_logistique | 0.3696 | 0.4970 | 0.7849 |
| ordinal_seuils | 0.3677 | 0.4495 | 0.6288 |
| boosting_hgb | 0.3428 | 0.3872 | 0.7405 |

**Retenu : `baseline_logistique`.** Le modèle le plus simple gagne, avec le meilleur macro-F1 et le
meilleur rappel Détracteur. Avec ~1 050 répondants à l'entraînement, le boosting n'a pas assez de
données pour justifier sa complexité. Ce n'est pas une déception : c'est la démonstration que la
baseline n'était pas une formalité.

## Stabilité en validation croisée

![](figures/04_cv_boxplot.png)

| Modèle | Kappa moyen | Écart-type |
|---|---|---|
| baseline_logistique | 0.554 | 0.046 |
| ordinal_seuils | 0.586 | 0.055 |
| boosting_hgb | 0.575 | 0.061 |

Un modèle meilleur en moyenne mais instable n'est pas un meilleur modèle. Ici la logistique est
à la fois la meilleure et parmi les plus stables.

## Hyperparamètres

Recherche aléatoire (scikit-learn `RandomizedSearchCV`, 5 plis stratifiés sur les répondants,
score = kappa quadratique), espaces modestes à dessein : sur ~1 050 lignes, le risque de surajuster
la recherche dépasse vite le gain. Réglé contre réglé pour une sélection équitable ; la version réglée
n'est retenue que si elle bat la version par défaut **sur les silencieux** de plus de 0,005 de kappa.

| Famille | Configurations | Kappa CV (réglé) | Kappa silencieux défaut | Kappa silencieux réglé | Gain | Décision |
|---|---|---|---|---|---|---|
| baseline_logistique | 12 | 0.581 | 0.370 | 0.369 | -0.000 | version par défaut |
| boosting_hgb | 20 | 0.625 | 0.343 | 0.365 | +0.022 | non retenu (famille non finale) |

Meilleurs paramètres trouvés :
- `baseline_logistique` : C=0.038986037025490715
- `boosting_hgb` : min_samples_leaf=10.0, max_leaf_nodes=15.0, max_iter=100.0, learning_rate=0.05, l2_regularization=0.0

Le modèle ordinal à seuils n'est pas réglé par cette recherche : son optimisation interne des deux
seuils sur le kappa fait déjà office de réglage.
