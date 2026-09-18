# Phase 5 — Évaluation

**En bref.** Le modèle retenu (logistique ordinale, réglée) atteint un kappa quadratique de **0.384** sur les
5963 clients silencieux, rappelle **63%** des détracteurs et garde les trois classes vivantes. Sur
500 appels, **51%** touchent un vrai détracteur contre 18% au hasard (lift ×2.8) — 18 585 € de gain net
par campagne à hypothèses constantes. Le NPS des silencieux est estimé à **+4** (IC 95 % +2 à +5) pour une vérité de
+20. Les drivers sont contractuels (engagement, ancienneté, parrainage, prix) et stables quel que soit
le mapping. Le modèle rappelle moins bien les jeunes détracteurs (48% vs 73% pour les 60+) alors qu'il
n'a aucune variable d'âge : les features retenues **reconstruisent** l'âge (AUC 0.92). Une mitigation par
seuils est chiffrée ; l'arbitrage est remonté au métier.

## 1. Évaluation technique — jamais une métrique globale sans son détail par classe

**Pourquoi ces métriques.**

| Métrique | Rôle | Pourquoi elle est là |
|---|---|---|
| Kappa quadratique pondéré | métrique principale de sélection | cible ordonnée : une erreur de deux crans (Promoteur prédit Détracteur) doit coûter plus qu'une erreur d'un cran ; le kappa quadratique le fait, l'exactitude et le F1 ne le font pas ; corrigé du hasard |
| Macro-F1 | départage | moyenne non pondérée des F1 des trois classes : une classe minoritaire abandonnée fait chuter le score, ce que l'exactitude masque |
| Exactitude équilibrée | lecture | moyenne des rappels par classe ; même logique que le macro-F1 côté rappel |
| Rappel / précision / F1 par classe | obligatoire à chaque publication | la règle du projet : le Détracteur est la classe qui compte pour le métier, le Passif celle qu'un modèle abandonne en premier |
| AUC un-contre-tous et AP (Détracteur) | qualité du classement | l'application classe les clients par P(Détracteur) : l'AUC et l'aire précision-rappel mesurent ce classement indépendamment du seuil ; l'AP est préférée sous déséquilibre |
| Brier et ECE (Détracteur) | qualité des probabilités | les probabilités servent à allouer un budget d'appels : Brier mesure leur erreur quadratique, l'ECE l'écart entre probabilité annoncée et fréquence observée |
| Précision@K, rappel@K, lift | métrique métier | la question de l'équipe rétention : « sur les K clients qu'on peut appeler ce mois-ci, combien sont réellement des détracteurs ? » |
| Exactitude brute | rapportée pour montrer qu'elle trompe | prédire toujours Passif donne ~0,43 ; l'énoncé la disqualifie comme critère |

**Résultat sur les 5963 silencieux** (modèle entraîné sur 1080 répondants, S3, cible M3) :

| Métrique globale | Valeur |
|---|---|
| Kappa quadratique pondéré | **0.384** |
| Macro-F1 | 0.490 |
| Exactitude équilibrée | 0.511 |
| AUC un-contre-tous (macro) · AUC Détracteur · AP Détracteur | 0.677 · 0.820 · 0.462 |
| Exactitude brute | 0.486 — *rapportée pour montrer qu'elle trompe : 0,43 en prédisant toujours « Passif »* |

| Classe | Précision | Rappel | F1 | Effectif |
|---|---|---|---|---|
| Détracteur | 0.438 | 0.629 | 0.516 | 1083 |
| Passif | 0.473 | 0.481 | 0.477 | 2590 |
| Promoteur | 0.547 | 0.423 | 0.477 | 2290 |

![](figures/05_confusion_pr.png)

- **Le Détracteur est rappelé à 63%** — c'est la classe qui compte pour le métier. Sa précision
  est de 44% : le modèle appelle large. C'est le bon arbitrage pour une campagne de rétention où
  manquer un détracteur coûte plus qu'un appel inutile ; la précision@K en tient compte.
- **Le Passif est la classe la plus difficile** (rappel 48%) : milieu ambigu par définition, rare à
  l'entraînement (phase 3 § 4). Cohérent avec Elero et al. (2026), où la classe neutre est aussi la moins bien prédite.
- La courbe précision-rappel est plus informative que la ROC sous déséquilibre : le Détracteur a une AP de
  0.46 pour un taux de base de 0.18.
- Référence : la baseline logistique fait 0.370 de kappa sur les mêmes silencieux (le modèle retenu est au-dessus).

## 2. Calibration : ce que la courbe de fiabilité révèle

![](figures/05_calibration_gain.png)

Les deux courbes sont **sous la diagonale** : le modèle surestime P(Détracteur), calibré ou non (ECE
0.15 après Platt). Cause : le calibrateur est ajusté sur les répondants, où les détracteurs sont
sur-représentés par le biais MNAR ; il apprend le taux de base des répondants, pas celui de la
population silencieuse. **La calibration sur un échantillon biaisé calibre vers la mauvaise
population** — le problème MNAR réapparaît à un autre étage. Conséquence : le **classement** des clients
par P(Détracteur) reste valide (l'ordre est préservé, AUC 0.82), mais la **valeur absolue** ne doit pas
être lue comme une fréquence — l'application l'affiche. Correction possible en production : recalibrer
sur les premières vraies réponses de la population cible (phase 6 § 4).

## 3. Évaluation métier

La vraie question n'est pas « quel taux de bonnes réponses » mais **« sur les K clients que l'équipe
peut appeler ce mois-ci, combien sont réellement des détracteurs ? »**

| K appels | Précision@K | Rappel@K | Lift | Détracteurs captés |
|---|---|---|---|---|
| 100 | 59.0% | 5.5% | ×3.25 | 59 |
| 250 | 54.8% | 12.7% | ×3.02 | 137 |
| 500 | 51.2% | 23.6% | ×2.82 | 256 |
| 750 | 50.0% | 34.6% | ×2.75 | 375 |
| 1000 | 48.0% | 44.3% | ×2.64 | 480 |
| 1500 | 43.9% | 60.9% | ×2.42 | 659 |
| 2000 | 40.6% | 74.9% | ×2.23 | 811 |

Taux de base (part de détracteurs parmi les silencieux) : **18.2%**. À K = 500, **51.2% des clients appelés sont
de vrais détracteurs — 2.8× mieux qu'au hasard.**

**Traduction en euros.** Hypothèses (paramètres dans `config.yaml`, **à valider avec l'équipe rétention**) :
coût d'un appel 12 €, valeur d'un client retenu 450 €, taux de succès 25%.

| | Campagne ciblée par le modèle | Campagne aléatoire de même taille |
|---|---|---|
| Coût | 6 000 € | 6 000 € |
| Clients retenus (estimés) | 64.0 | — |
| Gain net | **22 800 €** | 4 215 € |
| ROI | 3.80 | — |

**Apporté par le modèle : 18 585 € par campagne de 500 appels**, à hypothèses constantes.

**`CLTV` en couche de décision.** Interdite comme feature (sortie de modèle IBM : elle fuit), `CLTV` est
légitime pour pondérer *qui appeler* parmi les détracteurs prédits : c'est un choix métier, pas une
information sur la cible. Gómez-Vargas et al. (EJOR 2025) montrent qu'une CLV individuelle bat une CLV
moyenne pour cibler une campagne. L'application propose cette pondération en option.

## 4. Simuler le NPS des 85 % silencieux

![](figures/05_nps_simule.png)

| | NPS |
|---|---|
| Répondants — ce que l'opérateur observe | +26.9 |
| Silencieux — prédit par le modèle (IC 95 % bootstrap) | **+3.6** [+1.6 ; +5.3] |
| Silencieux — vrai (connu ici seulement) | +20.2 |
| Base complète — prédit | +4.2 |

| Classe | Prédit (silencieux) | Vrai (silencieux) |
|---|---|---|
| Détracteur | 26.1% | 18.2% |
| Passif | 44.2% | 43.4% |
| Promoteur | 29.7% | 38.4% |

Lecture : le modèle sous-estime le NPS des silencieux de 16.6 pts (écart marqué, la vérité est hors de l'intervalle bootstrap : celui-ci ne couvre que l'aléa d'échantillonnage, pas le biais du modèle). Composition par classe : trop de Détracteurs ; trop peu de Promoteurs ; la part de Passifs est juste. Ce sont les conséquences du biais de réponse — le modèle apprend sur des répondants polarisés et une pondération de classes qui rééquilibre le Passif — et le NPS, différence de deux parts, cumule les erreurs sur les Détracteurs et les Promoteurs. À rapporter avec l'intervalle et cette réserve ; la correction passe par les premières vraies réponses de la population cible (phase 6 § 4). **Par segment** (silencieux) :

| Segment | Clients | NPS prédit | NPS vrai | Part Dét. prédite | Part Dét. vraie |
|---|---|---|---|---|---|
| Contrat mensuel | 3087 | -40.9 | -1.3 | 49.0% | 32.1% |
| Contrat 1–2 ans | 2876 | 51.5 | 43.4 | 1.5% | 3.2% |
| Ancienneté < 12 mois | 1815 | -36.1 | -2.6 | 49.5% | 33.8% |
| Ancienneté ≥ 24 mois | 3266 | 28.1 | 32.7 | 11.3% | 9.1% |
| Fibre | 2510 | -31.4 | 2.0 | 42.7% | 27.8% |
| DSL | 1454 | 4.7 | 26.1 | 18.4% | 13.3% |
| Sans internet | 1278 | 80.8 | 50.9 | 0.0% | 4.5% |

Le classement des segments par NPS est respecté (contrats mensuels et clients récents sont bien les
plus critiques) ; les niveaux sont exagérés dans les deux sens — effet du biais de réponse sur un
modèle qui n'a jamais vu de Passifs en nombre.

## 5. Robustesse et sensibilité

**5.1 Sensibilité au mapping de la cible** (énoncé § 4.1). Kappa du modèle retenu sous les trois
mappings (S3) : M1 0.247 · M2 0.351 · M3 0.376 — ⚠️ non comparables entre eux (déplacer le milieu ambigu
change la difficulté). Ce qui se compare, ce sont les **conclusions** : sur les 12 variables les plus
influentes (logistique, dont les coefficients sont directement comparables), **10 conservent le même
sens d'effet sous les trois mappings**. Les conclusions métier sont robustes au choix du mapping ;
c'est l'amplitude qui varie, pas la direction.

![](figures/05_sensibilite_mapping.png)

**5.2 Bruit d'étiquettes** (énoncé § 4.1 : « adding realistic noise »). On corrompt une part des
étiquettes d'entraînement (classe redistribuée au hasard) et on mesure sur les silencieux, dont les
étiquettes restent vraies.

| Bruit | Kappa | Macro-F1 | Rappel Dét. | Rappel Passif | Rappel Prom. |
|---|---|---|---|---|---|
| 0% | 0.384 | 0.490 | 0.629 | 0.481 | 0.423 |
| 5% | 0.377 | 0.483 | 0.611 | 0.525 | 0.371 |
| 10% | 0.367 | 0.477 | 0.548 | 0.540 | 0.375 |
| 20% | 0.353 | 0.466 | 0.497 | 0.637 | 0.292 |
| 30% | 0.277 | 0.418 | 0.334 | 0.747 | 0.212 |

Le kappa perd 0.108 entre 0 et 30% de bruit : dégradation graduelle, pas d’effondrement. Un
label NPS réel est bruité (humeur du jour, incompréhension de l'échelle) ; ce test borne ce que ça coûte.

**5.3 Ablation de features** — ce que chaque bloc coûte ou rapporte, modèle retenu, S3/M3, silencieux :

![](figures/05_robustesse.png)

| Configuration | Variables | Kappa | Macro-F1 | Rappel Dét. | Rappel Passif | AUC Dét. | Δ kappa |
|---|---|---|---|---|---|---|---|
| base (retenue) | 35 | 0.384 | 0.490 | 0.629 | 0.481 | 0.818 | +0.000 |
| + démographie | 42 | 0.393 | 0.491 | 0.642 | 0.471 | 0.825 | +0.009 |
| + géographie (lat, lon, population) | 38 | 0.382 | 0.490 | 0.621 | 0.479 | 0.818 | -0.002 |
| + démographie + géographie | 45 | 0.386 | 0.490 | 0.639 | 0.471 | 0.825 | +0.001 |
| − variables dérivées | 26 | 0.367 | 0.481 | 0.608 | 0.481 | 0.806 | -0.017 |
| − parrainage (Number of Referrals, a_parraine) | 33 | 0.357 | 0.481 | 0.597 | 0.501 | 0.796 | -0.027 |
| − Offer et a_recu_offre | 33 | 0.383 | 0.488 | 0.629 | 0.478 | 0.819 | -0.001 |
| − Contract | 34 | 0.357 | 0.481 | 0.560 | 0.501 | 0.796 | -0.027 |

Trois lectures. **Démographie et géographie** : ajouter la démographie change le kappa de +0.009,
la géographie de -0.002, les deux de +0.001 — c'est le **coût mesuré du choix d'équité**
(énoncé § 4.7 : « Removing a feature often costs accuracy. Make the trade-off explicit »). Il est
assumé : un gain de cet ordre ne justifie pas de décider sur l'âge, le genre ou le code postal, et
§ 7 montre que le modèle retrouve de toute façon ces attributs par proxies. **Parrainage** : retirer
`Number of Referrals` et `a_parraine` change le kappa de -0.027 — c'est le driver le plus proche de la cible sans
être une fuite (comportement passé observable). **Variables dérivées** : -0.017 — elles portent une part réelle du signal.
Retirer `Contract` : -0.027 ; retirer `Offer` : -0.001.

## 6. Drivers de détraction

Méthode : **coefficients × valeur (exact, additif)** — la même dans l'application, pour chaque client. Les contributions sont
lues pour la classe Détracteur ; les variables liées par identité comptable (phase 3 § 3.5) se lisent en bloc.

![](figures/05_interpretabilite.png)

| Variable | Importance | Sens (corrélation valeur ↔ contribution) |
|---|---|---|
| Number of Referrals | 0.5627 | -1.00 |
| Monthly Charge | 0.5022 | +1.00 |
| Contract_Month-to-Month | 0.4820 | +1.00 |
| a_parraine | 0.4668 | +1.00 |
| Total Charges | 0.4013 | -1.00 |
| Online Security_No | 0.3932 | -1.00 |
| Total Long Distance Charges | 0.2833 | +1.00 |
| Phone Service_Yes | 0.2617 | -1.00 |
| Total Revenue | 0.2284 | -1.00 |
| revenu_par_mois | 0.2115 | +1.00 |
| Avg Monthly Long Distance Charges | 0.2061 | -1.00 |
| Internet Service_Yes | 0.1904 | +1.00 |
| Online Security_Yes | 0.1671 | +1.00 |
| Contract_Two Year | 0.1501 | -1.00 |
| Avg Monthly GB Download | 0.1485 | -1.00 |
| tranche_anciennete_0-6m | 0.1436 | +1.00 |
| Paperless Billing_Yes | 0.1299 | +1.00 |
| Streaming Music_No | 0.1122 | +1.00 |
| Device Protection Plan_No | 0.0945 | +1.00 |
| Paperless Billing_No | 0.0931 | -1.00 |

**Ce que le modèle dit.** Les associations sont fortes et cohérentes avec la littérature sur ce
dataset (Chong et al. 2023) et sur le NPS télécom (Elero et al. 2026) : **contrat sans engagement,
faible ancienneté, absence de parrainage, charge mensuelle élevée** vont avec la détraction ; l'offre E
ressort comme marqueur de clients déjà fragiles (phase 2 § 5.5), pas comme cause. **Le modèle établit
des associations, pas des causes** : un coefficient positif est un signal, pas un levier prouvé.

**Drivers par segment** (énoncé § 4.6) — les mêmes variables ne pèsent pas partout pareil :

![](figures/05_drivers_segments.png)

| Segment | Clients | Driver 1 | Driver 2 | Driver 3 | Driver 4 | Driver 5 |
|---|---|---|---|---|---|---|
| Contrat mensuel | 3087 | Contract_Month-to-Month (▲ risque) | Number of Referrals (▲ risque) | a_parraine (▼ risque) | Online Security_No (▼ risque) | Monthly Charge (▼ risque) |
| Contrat 1–2 ans | 2876 | Number of Referrals (▼ risque) | Monthly Charge (▼ risque) | a_parraine (▲ risque) | Total Charges (▼ risque) | Online Security_No (▼ risque) |
| Ancienneté < 12 mois | 1815 | Contract_Month-to-Month (▲ risque) | Online Security_No (▼ risque) | Number of Referrals (▲ risque) | tranche_anciennete_0-6m (▲ risque) | Total Charges (▲ risque) |
| Ancienneté ≥ 24 mois | 3266 | Number of Referrals (▼ risque) | Monthly Charge (▲ risque) | a_parraine (▲ risque) | Total Charges (▼ risque) | Online Security_No (▼ risque) |
| Fibre | 2510 | Contract_Month-to-Month (▲ risque) | Number of Referrals (▲ risque) | Monthly Charge (▲ risque) | a_parraine (▼ risque) | Total Charges (▼ risque) |
| DSL | 1454 | Number of Referrals (▼ risque) | a_parraine (▼ risque) | Contract_Month-to-Month (▲ risque) | Total Charges (▲ risque) | Monthly Charge (▼ risque) |

**Actionnable ou non, et levier** :

| Driver | Actionnable ? | Levier |
|---|---|---|
| Contrat mensuel | **oui** | proposer un engagement 12 mois avec avantage |
| Faible ancienneté | partiellement | parcours d'accueil renforcé les 6 premiers mois |
| Charge mensuelle / par service élevée | **oui** | revue tarifaire, bundle |
| Absence de parrainage | indirectement | programme de parrainage |
| Offre E reçue | **signal, pas levier** | ne pas « supprimer l'offre E » — comprendre à qui elle a été proposée |
| Type d'accès (fibre) | non à court terme | investissement réseau, support premium |

**Règle de recommandation individuelle** (implémentée, affichée pour chaque client) :
- 1. contrat mensuel → proposer un engagement 12 mois avec avantage
- 2. sinon charge par service > médiane → revue tarifaire / bundle
- 3. sinon ancienneté < 12 mois → parcours d'accueil renforcé
- 4. sinon internet sans support premium → proposer le support technique premium
- 5. sinon → appel de courtoisie et diagnostic

Répartition parmi les détracteurs prédits :

| Levier recommandé | Détracteurs prédits concernés |
|---|---|
| Proposer un engagement 12 mois avec avantage | 1513 |
| Revue tarifaire / bundle (charge par service élevée) | 17 |
| Proposer le support technique premium | 13 |
| Appel de courtoisie et diagnostic | 12 |
| Parcours d'accueil renforcé (client récent) | 1 |

*heuristique dérivée des drivers (associations), pas d'un effet causal mesuré.*

## 7. Équité et biais (énoncé § 4.7)

Les variables démographiques et géographiques sont **exclues du modèle** et servent **uniquement ici**.
On mesure, par sous-groupe, le rappel Détracteur (le modèle rate-t-il davantage certains détracteurs ?),
la précision, et le **taux de sélection** (quelle part du groupe est prédite détracteur — c'est ce qui
alloue le budget d'appels). `zone` = tertiles de population du code postal (proxy de densité).

![](figures/05_equite.png)

| Dimension | Groupe | Clients | Détracteurs réels | Taux de sélection | Rappel Dét. | Précision Dét. |
|---|---|---|---|---|---|---|
| Gender | Female | 2939 | 18.4% | 27.1% | 64.1% | 43.5% |
| Gender | Male | 3024 | 18.0% | 25.2% | 61.7% | 44.0% |
| Senior Citizen | No | 5019 | 16.3% | 23.9% | 58.5% | 39.7% |
| Senior Citizen | Yes | 944 | 28.3% | 37.6% | 76.4% | 57.5% |
| Married | No | 3110 | 22.4% | 31.9% | 60.7% | 42.7% |
| Married | Yes | 2853 | 13.5% | 19.8% | 66.8% | 45.7% |
| Dependents | No | 4568 | 22.3% | 30.0% | 62.9% | 46.7% |
| Dependents | Yes | 1395 | 4.6% | 13.2% | 62.5% | 21.7% |
| tranche_age | 30-45 | 1655 | 16.4% | 26.4% | 60.5% | 37.5% |
| tranche_age | 45-60 | 1617 | 17.1% | 26.5% | 62.0% | 40.0% |
| tranche_age | 60+ | 1397 | 25.1% | 34.1% | 73.4% | 54.0% |
| tranche_age | <30 | 1294 | 14.4% | 16.6% | 47.9% | 41.4% |
| zone | densité moyenne | 1998 | 16.1% | 25.0% | 62.6% | 40.3% |
| zone | faible densité | 1982 | 17.4% | 25.0% | 63.8% | 44.4% |
| zone | forte densité | 1983 | 21.0% | 28.3% | 62.4% | 46.3% |

| Dimension | Écart max de rappel (pts) | Écart max de sélection (pts) |
|---|---|---|
| Gender | 2.4 | 1.9 |
| Senior Citizen | 17.9 | 13.7 |
| Married | 6.2 | 12.1 |
| Dependents | 0.4 | 16.9 |
| tranche_age | 25.6 | 17.4 |
| zone | 1.4 | 3.4 |

**7.1 Ce qu'on trouve.** L'écart le plus important porte sur **tranche_age : 26 points** de rappel entre le groupe le
mieux servi et le moins bien servi — le modèle rappelle 48% des détracteurs de moins de 30 ans contre
73% des 60 ans et plus. C'est le cas de figure que l'énoncé décrit comme inacceptable, **en sens
inverse** : ce sont les jeunes détracteurs qu'on rate. Les écarts par genre et par densité de zone sont faibles.

**7.2 Les features « proxent »-elles les attributs protégés ?** Pour chaque attribut exclu, on mesure à
quel point les features du modèle permettent de le reconstruire (AUC en validation croisée) :

![](figures/05_proxies.png)

| Attribut (exclu du modèle) | Prévalence | AUC de reconstruction | Lecture |
|---|---|---|---|
| Moins de 30 ans | 19.9% | 0.920 | fortement reconstructible |
| Senior (65+) | 15.8% | 0.874 | fortement reconstructible |
| Femme | 49.3% | 0.486 | non reconstructible |
| Marié(e) | 47.9% | 0.982 | fortement reconstructible |
| A des dépendants | 23.4% | 0.793 | fortement reconstructible |
| Zone de forte densité | 33.3% | 0.505 | non reconstructible |

**Exclure une variable ne suffit pas à ce que le modèle l'ignore.** L'âge et la situation familiale sont
largement reconstructibles à partir du contrat, de l'ancienneté, des services et du parrainage ; le
genre et la densité de zone ne le sont pas. C'est la raison structurelle de l'écart d'âge : le modèle
« voit » l'âge sans en avoir la colonne. L'exclusion reste justifiée (elle empêche une décision
*explicite* sur l'attribut et retire le levier le plus direct), mais elle ne dispense pas de l'audit.

**7.3 Mitigation testée : un seuil de décision par tranche d'âge**, calé pour que chaque groupe atteigne
le rappel global (63%). Ce que ça coûte :

| Tranche d'âge | Rappel avant | Rappel après | Précision avant | Précision après | Sélection avant | Sélection après | Seuil |
|---|---|---|---|---|---|---|---|
| 30-45 | 60.5% | 63.1% | 37.5% | 38.3% | 26.4% | 27.0% | 0.5692 |
| 45-60 | 62.0% | 63.0% | 40.0% | 40.6% | 26.5% | 26.5% | 0.5726 |
| 60+ | 73.4% | 63.1% | 54.0% | 54.8% | 34.1% | 28.8% | 0.6132 |
| <30 | 47.9% | 63.4% | 41.4% | 33.6% | 16.6% | 27.1% | 0.4304 |

| | Kappa | Macro-F1 | Rappel Dét. | Précision Dét. | Appels (prédits Détracteur) |
|---|---|---|---|---|---|
| Avant | 0.384 | 0.490 | 0.629 | 0.438 | 1556 |
| Après seuils par âge | 0.373 | 0.483 | 0.629 | 0.419 | 1626 |

Égaliser le rappel se paie en précision chez les jeunes (plus d'appels inutiles) et en volume global.
C'est un arbitrage **métier et juridique**, pas technique : traiter différemment selon l'âge pour
corriger un écart est légal dans certaines juridictions et pas dans d'autres.

**7.4 À remonter avant production.**
1. L'écart d'âge (26 pts) et sa cause (proxies) — à l'équipe Expérience Client et au juridique.
2. Le fait que la mitigation par seuils est possible et chiffrée, mais qu'elle constitue un traitement
   différencié selon l'âge : décision à prendre au-dessus de la data science.
3. L'audit doit être répété à chaque réentraînement — il est dans le monitoring (phase 6 § 4).

## 8. Limites de la règle de ciblage

Cette étude classe les clients par **risque** d'être détracteur, comme demandé. Or Ascarza (2018,
*Journal of Marketing Research*) établit, par deux expériences de terrain, que cibler les clients au
risque le plus élevé est inefficace : il faut cibler ceux dont la **sensibilité à l'intervention** est
la plus forte — même campagne, même budget, jusqu'à 7 points de churn en moins.

| | On l'appelle | On ne l'appelle pas | |
|---|---|---|---|
| | reste | reste | **acquis** — appel gaspillé |
| | part | part | **perdu d'avance** — appel gaspillé |
| | **reste** | **part** | **persuadable** — seul cas qui crée de la valeur |
| | part | reste | **à ne pas déranger** — l'appel nuit |

Les détracteurs les plus certains sont massivement des *perdus d'avance*. **On ne peut pas estimer
cette sensibilité ici** : aucune campagne enregistrée, aucune assignation aléatoire (la table `Offer`
prouve que les offres ont été ciblées). Prétendre faire de l'uplift serait la faute reprochée à Mustafa
et al. La seule chose à faire, gratuite, est en phase 6 § 5.

## 9. Revue du processus

| Critère de succès (phase 1) | Résultat | Verdict |
|---|---|---|
| Précision@500 nettement au-dessus du taux de base | 51.2% vs 18.2% (×2.8) | ✅ |
| Gain net vs aléatoire, en euros | 18 585 € par campagne | ✅ (hypothèses à valider) |
| Kappa et rappel Détracteur au niveau de la baseline ou au-dessus | retenu logistique ordinale : kappa 0.384 ; logistique 0.370 | ✅ |
| Aucune classe abandonnée | rappel Passif 0.48 | ✅ |
| Écart d'équité mesuré, signalé si > 10 pts | 26 pts sur l'âge, cause identifiée, mitigation chiffrée | ⚠️ **signalé, arbitrage remonté** |

**Solide** : registre de fuites quantifié ; arbitrage empirique des « 3 » sans circularité ; protocole
15/85 avec coût du MNAR mesuré ; quinze modèles, sélection sans regarder le test ; calibration
vérifiée ; audit d'équité avec proxies et mitigation. **Approximatif** : hypothèses économiques
(placeholders) ; propension à répondre simulée ; verbatims synthétiques sans clé API ; TabPFN non
évalué faute d'accès. **Impossible avec ces données** : estimer l'effet d'un appel.

**Reproductibilité — test à blanc depuis une copie vierge.** Protocole : copie de `src/`, `prompts/`, `config.yaml` et `data/raw/` seuls dans un répertoire vierge,
exécution de `python -m src.run`, comparaison des indicateurs clés. Durée : 1336.3 s.

| Indicateur | Original | Copie vierge | Identique |
|---|---|---|---|
| NPS M1 | -42.0 | -42.0 | ✅ |
| NPS M3 | 21.3 | 21.3 | ✅ |
| part des 3 -> détracteur | 0.2788 | 0.2788 | ✅ |
| modèle retenu | logistique_ordinale | logistique_ordinale | ✅ |
| version retenue | réglée | réglée | ✅ |
| kappa final (silencieux) | 0.384446 | 0.384446 | ✅ |
| macro-F1 final | 0.490177 | 0.490177 | ✅ |
| précision@K | 0.512 | 0.512 | ✅ |
| NPS silencieux prédit | 3.6 | 3.6 | ✅ |
| écart d'équité (âge) | 0.2558 | 0.2558 | ✅ |

**Reproductible à l’identique.**

## Décisions prises dans cette phase

| Décision | Justification | Où c'est vérifié |
|---|---|---|
| Toute métrique globale accompagnée de son détail par classe | règle du projet ; contre-exemple Kannan et al. | § 1 |
| Probabilités utilisées pour classer, pas pour lire une fréquence | calibrateur ajusté sur une population biaisée | § 2 |
| K = 500 et hypothèses économiques dans config.yaml | paramètres métier, à valider avec l'équipe | § 3 |
| NPS des silencieux rapporté avec intervalle et réserve de composition | excès compensés de Détracteurs et Promoteurs | § 4 |
| Coût de l'exclusion des démographiques et de la géographie chiffré et assumé | énoncé § 4.7 ; proxies mesurés | § 5.3, § 7.2 |
| Mitigation par seuils d'âge chiffrée, non appliquée par défaut | traitement différencié : décision métier/juridique | § 7.3 |
| Pas d'uplift modeling | aucune assignation aléatoire ; le prétendre serait une faute | § 8 |
