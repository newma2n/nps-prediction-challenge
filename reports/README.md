# Étude — ordre de lecture

| # | Phase CRISP-DM | Rapport |
|---|---|---|
| 0 | Synthèse pour décideur | [00_synthese.md](00_synthese.md) |
| 1 | Compréhension métier | [01_comprehension_metier.md](01_comprehension_metier.md) |
| 2 | Compréhension des données | [dictionnaire](02_donnees/dictionnaire_donnees.md) · [qualité](02_donnees/qualite_donnees.md) · [EDA](02_donnees/analyse_exploratoire.md) · [**registre de fuites**](02_donnees/registre_fuites.md) |
| 3 | Préparation | [**construction de la cible**](03_preparation/construction_cible.md) · [features et colinéarité](03_preparation/features.md) |
| 4 | Modélisation | [**protocole 15/85**](04_modelisation/protocole_validation.md) · [comparaison des modèles et hyperparamètres](04_modelisation/comparaison_modeles.md) · [TabPFN](04_modelisation/tabpfn.md) · [verbatims et fusion texte](04_modelisation/verbatims_et_fusion_texte.md) |
| 5 | Évaluation | [technique](05_evaluation/evaluation_technique.md) · [**métier**](05_evaluation/evaluation_metier.md) · [sensibilité](05_evaluation/sensibilite_mapping.md) · [interprétabilité et drivers](05_evaluation/interpretabilite_drivers.md) · [**équité**](05_evaluation/audit_equite.md) · [revue](05_evaluation/revue_processus.md) |
| 6 | Déploiement | [carte de modèle](06_deploiement/carte_modele.md) · [monitoring et décision](06_deploiement/monitoring_et_retrainement.md) |

Figures dans `figures/`. Données brutes des résultats dans `resultats.json`.
Tout est régénéré par `python -m src.run` puis `python -m src.rapports`.
