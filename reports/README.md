# L'étude — un document par phase CRISP-DM

| # | Document | Pour qui | Contenu |
|---|---|---|---|
| 0 | [Synthèse](00_synthese.md) | direction | les cinq chiffres, trois résultats, la décision demandée |
| 1 | [Compréhension métier](01_comprehension_metier.md) | tous | problème, usages, critères de succès, formulation, hypothèses, risques |
| 2 | [Compréhension des données](02_comprehension_donnees.md) | data | sources, dictionnaire, qualité, **registre de fuites**, EDA, enrichissement |
| 3 | [Préparation des données](03_preparation_donnees.md) | data | **construction de la cible**, features justifiées, colinéarité, déséquilibre, pipeline |
| 4 | [Modélisation](04_modelisation.md) | data | formulations, **protocole 15/85**, **quinze modèles**, grille, réglage, **sélection sans regarder le test**, calibration, fondation, texte |
| 5 | [Évaluation](05_evaluation.md) | data + métier | technique, calibration, **métier (précision@K, euros)**, **NPS simulé**, robustesse, **drivers**, **équité**, limites, revue |
| 6 | [Déploiement](06_deploiement.md) | data + direction | carte du modèle, application, reproductibilité, **monitoring**, boucle de rétroaction, décision |
| 7 | [Conformité à l'énoncé](07_conformite_enonce.md) | évaluateur | chaque exigence → où elle est traitée |

Chaque document commence par un **En bref** et se termine par les **décisions prises** dans la phase,
chacune avec sa justification et l'endroit où elle est vérifiée.

Figures dans `figures/`. Résultats bruts : `resultats.json` (pipeline), `fondation.json`, `texte.json`,
`monitoring.json`, `reproductibilite.json`. Captures de l'application dans `captures/`.
Tout est régénéré par `python -m src.run` puis `python -m src.rapports`.
