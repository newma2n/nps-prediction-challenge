# Phase 5 — Revue du processus et décision

## Critères de succès (phase 1) — atteints ?

| Critère | Résultat | Verdict |
|---|---|---|
| Précision@500 nettement au-dessus du taux de base | 48.8% vs 18.2% (×2.7) | ✅ |
| Gain net vs aléatoire, en euros | +17,235 € par campagne | ✅ (hypothèses à valider) |
| Kappa et rappel Détracteur > baseline | la baseline **est** le modèle retenu | ✅ par construction |
| Aucune classe abandonnée | rappel Passif 0.32 après correction de la calibration | ✅ |
| Écart d'équité mesuré, signalé si > 10 pts | 25 pts sur l'âge | ⚠️ **signalé, non résolu** |

## Ce qui est solide

- Le registre de fuites et sa quantification.
- L'arbitrage empirique des « 3 », et la circularité évitée.
- Le protocole 15/85 avec le coût du MNAR mesuré.
- La sélection du modèle par la mesure, la calibration vérifiée.

## Ce qui est approximatif

- Les hypothèses économiques (coût d'appel, valeur client) sont des placeholders.
- La propension à répondre est simulée ; sa forme est plausible, pas observée.
- Pas de recherche d'hyperparamètres : choix assumé, mais un gain modeste est possible.
- Pas de TabPFN ni de verbatims synthétiques (bonus non réalisés).

## Ce qu'on ne peut pas faire avec ces données

Estimer l'effet d'un appel de rétention. Aucun traitement, aucune randomisation. Voir phase 6.

## Décision

**Go pour un pilote**, à trois conditions : valider les paramètres économiques avec la rétention ;
inclure un groupe de contrôle non contacté ; suivre le rappel par groupe d'âge dès la première
campagne.
