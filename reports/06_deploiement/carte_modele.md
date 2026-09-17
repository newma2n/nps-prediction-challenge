# Phase 6 — Carte de modèle

| | |
|---|---|
| Nom | Prédiction de catégorie NPS — clients silencieux |
| Version | 1.0 · seed 42 · régénéré par `python -m src.run` |
| Type | baseline_logistique dans un `Pipeline` scikit-learn (imputation + standardisation + one-hot) |
| Cible | M3 — Détracteur (satisfaction 1–2) · Passif (3) · Promoteur (4–5) |
| Données d'entraînement | 1080 répondants simulés (S3 — MNAR), IBM Telco 11.1.3+ |
| Évaluation | 5963 clients silencieux |
| Kappa quadratique | 0.370 |
| Rappel / précision Détracteur | 0.78 / 0.39 |
| Décision | classe : modèle non calibré · P(Détracteur) : modèle calibré Platt (stratégie hybride) |
| Variables exclues | tout ce qui touche au churn, `CLTV`, démographie, géographie |
| Usage prévu | prioriser une liste d'appels de rétention mensuelle |
| Usage **interdit** | décider d'une offre ou d'un tarif individuel ; toute décision fondée sur un attribut protégé |
| Limites connues | écart de rappel de 25 pts entre groupes d'âge ; classe Passif mal rappelée ; propension à répondre simulée |
| Fichier | `models/modele_final.joblib` |
