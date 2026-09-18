# Conformité à l'énoncé — chaque exigence, où elle est traitée

*Relecture complète de l'énoncé (« Take-Home Challenge — Customer NPS Prediction for a Telecom
Operator ») ; une ligne par exigence ou bonus. ✅ traité · ⚠️ traité partiellement, raison indiquée.*

| Énoncé | Exigence | Réponse | Où | Statut |
|---|---|---|---|---|
| § 2 | Prioriser les appels vers les Détracteurs prédits | liste classée par P(Détracteur) calibrée, K réglable, export | phase 5 § 3 ; app *Prioriser les appels* | ✅ |
| § 2 | Drivers de détraction, individuels et par segment | contributions par client ; drivers par segment ; leviers | phase 5 § 6 ; app *Drivers et équité*, *Analyser un client* | ✅ |
| § 2 | Simuler le NPS des 85 % silencieux | NPS prédit avec IC bootstrap, vs répondants et vs vérité ; par segment | phase 5 § 4 ; app *Synthèse* | ✅ |
| § 2 | Formulation : classification / ordinale / régression + seuils, justifiée | les trois instanciées et comparées par la mesure | phase 4 § 1 | ✅ |
| § 3 | IBM Telco 11.1.3+ comme source | cinq tables, jointure contrôlée | phase 2 § 1 | ✅ |
| § 3 | Enrichissement externe justifié si utilisé | décision motivée de ne pas enrichir | phase 2 § 6 | ✅ |
| § 4.1 | Mapping satisfaction → NPS justifié, raffiné, sensibilité discutée | M1/M2/M3, arbitrage empirique, sensibilité des drivers | phase 3 § 1 ; phase 5 § 5.1 | ✅ |
| § 4.1 | Bruit réaliste | test de robustesse au bruit d'étiquettes 0–30 % | phase 5 § 5.2 | ✅ |
| § 4.1 | Fuites dans la cible : Churn Score, Churn Value | registre de fuites en trois natures, information mutuelle, exclusion, test | phase 2 § 4 | ✅ |
| § 4.2 | Schéma documenté (gardé / supprimé / transformé) | dictionnaire avec statut par colonne | phase 2 § 2 ; phase 3 § 3 | ✅ |
| § 4.2 | Manquants et incohérences traités de façon transparente | cinq contrôles qualité | phase 2 § 3 | ✅ |
| § 4.2 | Découpage reflétant 15 % répondants / 85 % silencieux, stratégie de validation | trois scénarios de non-réponse, MNAR retenu, CV sur les répondants | phase 4 § 2, § 6 | ✅ |
| § 4.2 | Déséquilibre des classes discuté explicitement | double déséquilibre ; cinq stratégies comparées | phase 3 § 4 | ✅ |
| § 4.3 | Features construites et justifiées ; fuites évitées | justification variable par variable ; neuf dérivées ; auto-pay | phase 3 § 3 | ✅ |
| § 4.3 | Features géographiques / démographiques | testées en ablation, exclues par équité, coût chiffré | phase 5 § 5.3 | ✅ |
| § 4.4 | Verbatims synthétiques par LLM, conditionnés, bruités, reproductibles, commités | 7 043 verbatims, source `gabarit_stochastique_seed` ; chemin API prêt ; prompt versionné | phase 4 § 9 ; `data/processed/verbatims.parquet` | ✅ (générateur local seedé ; API sans clé) |
| § 4.4 | Combiner texte et tabulaire, expliquer ce que le texte apporte | texte seul, fusion tardive, fusion précoce ; gain qualifié d'artificiel | phase 4 § 9 ; app *Analyser un client* | ✅ |
| § 4.5 | Baseline | logistique multinomiale pondérée | phase 4 § 3 | ✅ |
| § 4.5 | Au moins deux familles justifiées (boosting, ordinal, ensembles calibrés, fondation) | quinze modèles, sept familles ; XGBoost, LightGBM, CatBoost, mord, forêt, TabICL | phase 4 § 3, § 8 | ✅ |
| § 4.5 | Métriques adaptées : macro-F1, balanced accuracy, kappa, rappel par classe, calibration, lift | toutes, avec la raison de chacune | phase 5 § 1–3 | ✅ |
| § 4.5 | Validation simulant l'écart répondants / silencieux | S3 MNAR ; sélection sans regarder les silencieux | phase 4 § 2, § 6 | ✅ |
| § 4.5 | Interprétabilité : importance / SHAP, drivers | coefficients ou SHAP selon le modèle ; méthode unique app + étude | phase 5 § 6 | ✅ |
| § 4.5 | Discussion fuites, biais, bruit, qualité, limites du label | réparties et récapitulées | phases 2, 3, 5 § 9 | ✅ |
| § 4.5 bonus | Modèle de fondation tabulaire (TabPFN-2.5, TabICL), avantages / limites / coût | TabICL évalué ; TabPFN indisponible (dépôt HF à accès contrôlé) | phase 4 § 8 | ✅ |
| § 4.6 | Drivers par segment ; actionnable vs non ; levier unique par Détracteur prédit ; corrélation ≠ causalité | segments contrat / ancienneté / accès ; règle de levier implémentée | phase 5 § 6 | ✅ |
| § 4.7 | Audit par sous-groupe démographique, rappel Détracteur | genre, senior, marié, dépendants, âge, densité de zone ; rappel, précision, sélection | phase 5 § 7 | ✅ |
| § 4.7 | Proxies d'attributs protégés ; gardé / supprimé / pourquoi | audit de reconstruction (AUC) des attributs exclus | phase 5 § 7.2 | ✅ |
| § 4.7 | Conséquence métier des choix d'équité, arbitrage explicite | coût de l'exclusion chiffré ; mitigation par seuils chiffrée | phase 5 § 5.3, § 7.3 | ✅ |
| § 4.7 | Points à escalader à CX / juridique | liste explicite | phase 5 § 7.4 | ✅ |
| § 4.8 | Persistance du modèle | `models/modele_final.joblib` | phase 6 § 1 | ✅ |
| § 4.8 | Interface : saisie ou ID → classe + probabilités ; drivers ; verbatim (bonus) ; robuste aux inconnus ; choix expliqués | Streamlit sept pages ; test de robustesse | phase 6 § 2 ; `app/app.py` | ✅ |
| § 4.9 | Monitoring : dérive des entrées, des prédictions, performance sur nouvelles réponses | PSI, mois simulé, lots mensuels de nouvelles réponses | phase 6 § 4 | ✅ |
| § 4.9 | Déclencheur de réentraînement (planning, seuil, volume de labels) | les trois, implémentés | phase 6 § 4 | ✅ |
| § 4.9 | Boucle de rétroaction actions ↔ données | journalisation, groupe de contrôle, réentraînement sur non-contactés | phase 6 § 5 | ✅ |
| § 6 | Code source ; notebook ; README + .env.example ; dataset dérivé ; verbatims + script ; artefact modèle ; captures ; write-up 3–6 pages | tous présents | dépôt ; `notebooks/` ; `livrable/` | ✅ |
| § 7 | Reproductible ; implémenté / approximatif / futur explicites ; usage d'IA déclaré ; pas de clés | test à blanc réussi ; revue ; README | phase 5 § 9 ; README | ✅ |

## Ce qui est implémenté, approximatif, ou laissé en travail futur (énoncé § 7)

**Implémenté** : tout le périmètre obligatoire (§ 4.1 à § 4.8), le monitoring (§ 4.9), les trois bonus
(verbatims + fusion texte, drivers par segment, modèle de fondation via TabICL).

**Approximatif, et dit comme tel** : les paramètres économiques (coût d'appel, valeur client, taux de
succès) sont des placeholders ; la propension à répondre est simulée ; les verbatims viennent d'un
générateur local seedé, pas d'un appel API (aucune clé disponible ; le chemin est prêt) ; le NPS des
silencieux est estimé avec une composition par classe déformée.

**Travail futur** : TabPFN-2.5 dès accès aux poids ; recalibration sur les vraies premières réponses ;
groupe de contrôle puis modèle d'uplift ; API de scoring pour le CRM ; enrichissement texte sur de vrais verbatims.
