# Phase 1 — Compréhension métier

## Le problème

Un opérateur télécom interroge ses clients sur leur propension à le recommander (NPS). Seuls
~15 % répondent. L'équipe rétention ne connaît donc l'état de satisfaction que d'une minorité,
et c'est parmi les 85 % silencieux que se trouvent les détracteurs sur le point de partir.

**Mission** : prédire la catégorie NPS (Détracteur / Passif / Promoteur) de chaque client à
partir de ses données de compte et de services, pour prioriser les appels de rétention.

## Critères de succès

| Niveau | Critère | Cible |
|---|---|---|
| Métier | Précision@K sur les détracteurs, K = 500 appels/mois | nettement au-dessus du taux de base |
| Métier | Gain net d'une campagne ciblée vs ciblage aléatoire | positif, chiffré en euros |
| Data science | Kappa quadratique pondéré et rappel Détracteur | supérieurs à la baseline |
| Data science | Aucune classe abandonnée | rappel > 0 sur les trois classes |
| Éthique | Écart de rappel Détracteur entre sous-groupes | mesuré, signalé s'il dépasse 10 points |

## Hypothèses déclarées

1. `Satisfaction Score` (1–5) est un proxy acceptable d'une note NPS (0–10). C'est une
   hypothèse, pas une identité : le NPS mesure la fidélité, la satisfaction mesure l'expérience.
2. Le biais de non-réponse simulé (les extrêmes répondent plus) ressemble au biais réel.
3. Les patterns d'un opérateur californien fictif se transposent au cadrage panafricain.

## Contraintes

Dataset fictif (IBM), une seule photographie, aucun historique, aucune campagne de rétention
enregistrée — donc **aucune mesure d'effet causal possible** (voir phase 5, limites du ciblage).

## Registre de risques

| Risque | Parade dans l'étude |
|---|---|
| Fuite de données → score irréel | registre de fuites (phase 2) + double diagnostic (phase 4) |
| Cible mal construite | arbitrage empirique des « 3 » + analyse de sensibilité (phase 3, 5) |
| Modèle inéquitable non détecté | audit par sous-groupe démographique (phase 5) |
| Livrable non reproductible | `python -m src.run` régénère tout, seed = 42 |

## Formulation retenue

Cible **ordonnée** (Détracteur < Passif < Promoteur) et **déséquilibrée**. Trois formulations
comparées en phase 4 : classification nominale, régression ordinale à seuils, boosting.
