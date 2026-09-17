# État de l'art — ce que la littérature impose à nos choix

Quatre études dépouillées. Ce document ne résume pas : il convertit chaque publication en
**décision contraignante** pour le projet. Format : ce que la littérature établit → ce qu'on fait.

## Les sources

| Réf | Étude | Pourquoi elle compte ici |
|---|---|---|
| **[E]** | Elero, Lima, dos Santos, Leal (2026), *Exploring NPS with ML and XAI: Evidence from Brazilian Broadband Services*, **Computers** 15(2):96 | **La plus proche de notre problème.** Likert 0–10 → NPS, trois formulations de cible comparées, SHAP, 64 647 clients télécom |
| **[C]** | Chong, Khaw, Yeong, Chuah (2023), *Customer Churn Prediction of Telecom Company Using ML*, **J. Soft Computing & Data Mining** 4(2) | Tourne sur **l'ancienne version de notre dataset** (blastchar, 7 043 lignes) — donne l'ordre de grandeur attendu |
| **[M]** | Mustafa, Sook Ling, Abdul Razak (2021), *Customer churn prediction for telecom: A Malaysian Case Study*, **F1000Research** 10:1274 | **Cas d'école de fuite de données**, publié et relu par les pairs. À citer comme contre-exemple |
| **[K]** | Kannan, Yan, Ramakrishnan, Wijaya (2022), *Prediction of Customer tNPS Using ML*, **ICTIM 2022**, AEBMR 228:166–179 | Pose explicitement le problème des **non-répondants** — notre partie 4.2 |

---

# 1. La structure M1 / M2 / M3 est validée par la littérature

**Ce qu'établit [E]** — confrontés au même problème que nous (une note Likert à convertir en
catégories NPS, avec un milieu ambigu), les auteurs ne tranchent pas : ils construisent
**trois cibles en parallèle** et comparent.

| Leur modèle | Construction | Notre équivalent |
|---|---|---|
| Model 1 | binaire, le milieu est **réparti** (8 → satisfait, 7 → insatisfait) | **M1** — la satisfaction 3 est répartie |
| Model 2 | binaire, le milieu est **retiré** (7 et 8 exclus de l'entraînement) | **M2** — on entraîne sur 1–2 vs 4–5 |
| Model 3 | multiclasse, les **trois classes NPS** telles quelles | **M3** — Détracteur / Passif / Promoteur |

**Décision.** On garde les trois mappings, mais **on les renomme selon cette logique** (répartir /
retirer / conserver) plutôt que selon des seuils arbitraires. Ce n'est plus notre invention :
c'est une méthodologie publiée, et on peut la citer.

---

# 2. La meilleure idée à reprendre : faire trancher les « 3 » par les données

C'est **l'apport le plus important** de toute cette lecture, et il résout directement notre
problème des 2 665 clients à satisfaction 3 (37,8 % de la base).

**Ce que fait [E]** (leur § 4.6 et § 5.5) : ils entraînent Model 2 **uniquement sur les extrêmes
non ambigus** (détracteurs ≤ 6 vs promoteurs ≥ 9), puis ils font passer les clients neutres
(7 et 8), jamais vus à l'entraînement, à travers le modèle — et ils regardent de quel côté
chacun tombe.

Leur résultat, stable sur les neuf algorithmes testés :

| Note neutre | Classée « insatisfait » | Classée « satisfait » |
|---|---|---|
| **7** | **69,2 % à 80,8 %** | 19,2 % à 30,8 % |
| **8** | 33,1 % à 40,7 % | **59,3 % à 66,9 %** |

Conclusion des auteurs : *« scores of "7" tend toward dissatisfaction, while scores of "8" tend
toward satisfaction »* — mais avec une part non négligeable de cas inverses, ce qui prouve que
le milieu contient réellement les deux profils.

**Décision — on transpose intégralement.** Étape 3 du plan devient :

1. entraîner sur les extrêmes non ambigus seuls : satisfaction **1–2** (détracteurs) vs **4–5** (promoteurs) ;
2. faire passer les **2 665 clients à satisfaction 3**, absents de l'entraînement, à travers ce modèle ;
3. mesurer la proportion qui bascule de chaque côté ;
4. **construire M3 à partir de ce résultat**, et non d'un choix arbitraire.

On passe de « j'ai décidé que les 3 seraient des passifs » à « les 3 se répartissent à X % du
côté détracteur, voici la pièce ». **C'est la différence entre un candidat qui choisit et un
candidat qui démontre.**

---

# 3. On sait maintenant à quoi ressemble un score honnête

Le problème posé à l'étape 7 était : à partir de quel score faut-il soupçonner une fuite ?
La littérature donne la réponse, sur des données comparables.

**[E], cible multiclasse à 3 classes NPS** (64 647 clients, 25 variables) :

| | Exactitude | F1 Détracteur | F1 Neutre | F1 Promoteur |
|---|---|---|---|---|
| Meilleurs (HGB, RF, GB) | **0,771 – 0,774** | 0,84 | 0,73 | 0,71 |
| Pire (KNN) | 0,691 | 0,78 | 0,66 | 0,57 |

**[C], churn binaire sur l'ancienne version de notre dataset** (7 043 clients) :

| | Exactitude | Précision (churn) | **Rappel (churn)** | F1 (churn) |
|---|---|---|---|---|
| XGBoost | 0,797 | 0,647 | **0,519** | 0,576 |
| Random Forest | 0,791 | 0,672 | **0,415** | 0,513 |
| KNN | 0,770 | 0,600 | **0,408** | 0,486 |

**Décisions.**

- **Fourchette attendue pour notre M3 : exactitude 0,70 – 0,80.** On vise cette zone, pas plus.
- **Le seuil d'alarme de l'étape 7 passe de 0,95 à ~0,85.** J'avais fixé 0,95 au jugé ; la
  littérature montre qu'au-delà de 0,85 sur trois classes NPS, on est déjà hors de tout
  précédent publié. Sonner l'alarme plus tôt.
- **Le Détracteur est la classe la plus facile à détecter** (F1 ≈ 0,84 contre 0,71 pour le
  Promoteur). C'est une bonne nouvelle métier : c'est précisément la classe qui nous intéresse.
  À souligner dans le write-up.
- **Attention aux comparaisons entre formulations.** [E] obtiennent 0,85 sur Model 1, **0,95 sur
  Model 2**, 0,77 sur Model 3 — et l'expliquent : retirer le milieu ambigu gonfle mécaniquement
  la performance. Nos M1/M2/M3 **ne sont pas comparables entre eux** en valeur absolue. Le dire
  explicitement, sinon on commet l'erreur qu'on prétend dénoncer.

---

# 4. [M] est un cas de fuite publié — notre meilleur argument de section 4.1

Cette étude prédit `POTENTIAL CHURNER` chez un opérateur malaisien et annonce **98 %
d'exactitude** (CART, KNN, SVM). Elle est relue par deux pairs et publiée.

Or, en lisant leur tableau de variables (Table 12), leurs features incluent :

- `S_NPS_FEEDBACK` — la note NPS brute de 0 à 10 ;
- `S_NPS_FEEDBACK_TYPE_FK` — la catégorie NPS (Promoteur / Passif / Détracteur).

Et leur cible ? Leur propre tableau de découverte l'avoue : *« Detractor: 312 […] potential
churner 247 = **79.2 % equivalence to given rating between 0-5** »*. **La cible est construite à
partir de la note NPS, qui reste dans les features.** Ils rapportent même une corrélation de
−0,85 / −0,91 entre les deux, et la présentent comme un *résultat* : *« Customers with lower NPS
ratings are more likely to churn »*.

**Le détail qui achève la démonstration** : leurs modèles linéaires (LR 41 %, LDA 42 %, NB 41 %)
s'effondrent pendant que les modèles à arbres atteignent 98 %. Cet écart n'est pas une
supériorité des arbres — c'est **la signature d'une variable qui fuit**, que les arbres isolent
d'un seul découpage et que les modèles linéaires, sur des catégories encodées en entiers,
n'exploitent pas.

**Décisions.**

1. **Nouveau diagnostic de fuite, à ajouter à l'étape 2** : si l'écart entre modèles à arbres et
   modèles linéaires est énorme (> 30 points), chercher la variable qui fuit avant de se
   réjouir. C'est un test gratuit et il aurait sauvé cette publication.
2. **Citer [M] dans le write-up**, section fuites. Montrer qu'on sait repérer le piège jusque
   dans la littérature relue vaut mieux que de dire qu'on l'a évité chez soi.

---

# 5. Les non-répondants : notre angle est documenté, pas inventé

C'était notre pari différenciant. Il est adossé à des sources.

**[K] pose le problème en toutes lettres** : *« about only 15–20 % of customers respond to the NPS
survey after their interactions with customer support service. Organizations have two major
challenges: **poor survey response rates and the possible loss of valuable insights from
non-respondents** »*. Les 15 % de notre énoncé ne sont donc pas un chiffre décoratif.

Pire — dans leurs données réelles, seuls **7 707 retours sur 287 729 interactions**, soit **2,6 %**.
Le taux réel peut être bien plus bas que 15 %.

**La citation à ne pas manquer** : [K] renvoie à **Zihayat, Ayanso, Davoudi et al. (2021),
*Leveraging non-respondent data in customer satisfaction modeling*, Journal of Business Research
135:112–126.** C'est le traitement académique exact de notre partie 4.2. À citer dans le write-up.

Autre référence utile : **Vélez, Ayuso, Perales-González, Rodríguez (2020), *Churn and NPS
forecasting for business decision-making*, Knowledge-Based Systems 196:105762** — ils ne
dépassent pas **54 % d'exactitude** en prédisant le NPS. Utile pour calibrer les attentes.

**Décision.** Le protocole à trois scénarios (MCAR / MAR / MNAR) de l'étape 4 est conservé tel
quel, mais désormais **justifié par citation**, et on ajoute une remarque : notre simulation à
15 % est une hypothèse **optimiste** au regard des 2,6 % observés par [K].

---

# 6. Quatre corrections concrètes à nos choix techniques

### a) `HistGradientBoostingClassifier` suffit — pas besoin de LightGBM

[E] testent neuf algorithmes ; **l'Histogram Gradient Boosting arrive en tête** (exactitude
0,774 en multiclasse, meilleur F1 Détracteur à 0,842), avec la variance la plus faible en
validation croisée. Or il est **livré dans scikit-learn** (`sklearn.ensemble.HistGradientBoostingClassifier`),
déjà installé.

**Décision.** HGB devient le modèle de boosting principal. LightGBM et CatBoost restent des
comparaisons de second rang si le temps le permet — plus une dépendance bloquante à installer
à l'étape 0.

### b) Ne jamais publier une métrique globale sans le détail par classe

[K] annoncent en résumé un **F-score de 0,876**. Le tableau détaillé donne, pour la classe
minoritaire : **0,126**. Et leur régression logistique en partition 90:10 affiche un F-measure
de la classe basse marqué **« NA »** — elle n'a essentiellement prédit aucun cas minoritaire.
Le résumé n'en dit rien.

**Décision — règle absolue du projet.** Toute métrique globale (exactitude, F1 pondéré) est
publiée **accompagnée du détail par classe**, dans le même tableau. Aucune exception. C'est
exactement la « honesty » que l'énoncé dit évaluer.

### c) Exclure les données démographiques des features a un précédent

[E] écrivent : *« As the objective of the project was to explore the most important features
influencing service satisfaction, **demographic data were excluded** »*. Et leurs drivers les
plus importants sont tous des variables d'expérience de service — vitesse, respect des
engagements, exactitude de la facturation, stabilité, clarté de l'offre — jamais des attributs
de personne.

**Décision, qui simplifie notre partie équité (§ 4.7).** On passe de « entraîner avec et sans le
code postal » à une position plus nette et défendable :

> Les attributs démographiques et géographiques sont **exclus des variables explicatives** et
> conservés **uniquement comme dimensions d'audit**, pour mesurer le rappel Détracteur par
> sous-groupe.

Avantages : position citable, cohérente avec l'objectif métier (les leviers actionnables sont
contractuels, pas démographiques), et elle évite d'avoir à défendre l'usage d'un proxy
socio-économique. On chiffre quand même le coût en performance de cette exclusion — l'énoncé
demande que l'arbitrage soit explicite.

### d) Croiser importance native et SHAP

[E] ne se contentent pas de SHAP : ils calculent les importances natives de RF, DT et GB, **puis**
les valeurs SHAP, et comparent les classements (leur Table 14). Les cinq mêmes variables
ressortent dans les quatre méthodes et les trois formulations de cible — seul l'ordre bouge.

**Décision.** À l'étape 14, produire la même table de convergence : importances natives × SHAP ×
M1/M2/M3. Une variable qui domine partout est un driver solide ; une variable qui n'apparaît
que dans une méthode ne va pas dans le write-up.

---

# 7. Deux nuances conceptuelles à intégrer au write-up

**Le NPS mesure la fidélité, pas la satisfaction.** [E] le disent nettement, en citant Baquero :
*« NPS is not a direct measure of satisfaction, but rather of customer loyalty »*, et traiter le
NPS comme proxy de satisfaction est *« a practical approach »*, pas une identité.

Cela nous concerne directement, et **en sens inverse** : nous partons d'une mesure de
*satisfaction* (le `Satisfaction Score` d'IBM) pour fabriquer une mesure de *fidélité* (le NPS).
C'est une hypothèse, et il faut l'écrire comme telle — pas la laisser passer pour une conversion
d'unités.

**Le NPS est critiqué dans la littérature.** [E] citent Kristensen & Eskildsen : le découpage en
trois groupes rend l'indicateur **imprécis** et peut conduire à de mauvaises décisions de gestion.
Leur propre analyse le confirme empiriquement — les neutres ont des profils des deux bords, donc
*« treating neutral customers as those who do not affect the business can lead to erroneous
conclusions »*.

**Décision.** La section « limites » du write-up ne dit pas « mon modèle est perfectible » mais
**« la métrique elle-même est contestée, voici ce que ça implique pour l'usage qu'en fera
l'équipe rétention »**. C'est le registre d'un consultant, pas d'un étudiant.

---

# 8. Structure méthodologique : adopter CRISP-DM

[K] structurent tout leur travail selon **CRISP-DM** (Business Understanding → Data
Understanding → Data Preparation → Modelling → Evaluation → Deployment), en citant Schröer et
al. (2021).

**Décision.** Nos 24 étapes se mappent naturellement dessus. On **nomme les parties du write-up
avec le vocabulaire CRISP-DM** : c'est un cadre que tout consultant data reconnaît, et Artefact
est un cabinet de conseil. Le fond ne change pas, le vocabulaire devient celui de l'évaluateur.

---

# 9. Une mise en garde de lecture

Le tableau 4 de [E] annonce, pour Model 1, « Satisfied 40 % — 38 890 » et « Dissatisfied 60 % —
26 057 ». Les effectifs contredisent les pourcentages : 38 890 sur un total de 64 647 fait 60 %,
pas 40 %. Les étiquettes sont inversées, et la somme des deux effectifs dépasse le total annoncé.

Ça ne retire rien à la valeur de l'étude, et ça rappelle une chose utile : **un chiffre publié
n'est pas un chiffre vérifié.** Chaque valeur reprise de ces papiers dans notre write-up doit
être recoupée avec son texte, et présentée comme provenant de l'étude — jamais comme un fait
établi que nous aurions mesuré.

Même règle que la convention de confiance de `roadmap.md` : faits mesurés, faits sourcés,
jugements. Ces quatre études alimentent la colonne du milieu, pas celle de gauche.
