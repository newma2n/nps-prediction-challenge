# L'angle différenciant — de la prédiction à la décision

Ce document contient la seule chose qui puisse vraiment distinguer ce livrable des autres.
Tout le reste du plan, un bon candidat le fera. Ceci, presque personne ne le fera.

---

# 1. La thèse

L'énoncé demande, au § 2 :

> *« prioritise outreach towards predicted Detractors before they churn »*

**Cette règle de ciblage est démontrée sous-optimale par la littérature académique.** Pas
discutable, pas « selon les cas » : démontrée, par deux expériences de terrain, dans une revue
de premier rang, avec un prix à la clé.

**Eva Ascarza (2018), *Retention Futility: Targeting High-Risk Customers Might Be Ineffective*,
Journal of Marketing Research** — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2759170) ·
[SAGE](https://journals.sagepub.com/doi/abs/10.1509/jmr.16.0163) ·
[prix Paul E. Green 2018](https://www.ama.org/2019/04/24/escarzas-retention-futility-targeting-high-risk-customers-might-be-ineffective-wins-2018-award/)

Sa conclusion, en une phrase : **une entreprise ne doit pas cibler les clients au risque le plus
élevé, mais ceux dont la sensibilité à l'intervention est la plus forte.** Sur ses données, la
même campagne de rétention aurait réduit le churn de **7 points de pourcentage supplémentaires**
avec la bonne règle de ciblage. Son diagnostic est brutal : *« retention programs are sometimes
futile not because firms offer the wrong incentives but because they do not apply the right
targeting rules »*.

## Pourquoi c'est vrai — l'intuition en quatre cases

Face à une action de rétention, un client appartient à l'un de quatre types :

| | **On l'appelle** | **On ne l'appelle pas** | Type |
|---|---|---|---|
| | reste | reste | **Acquis** — l'appel est gaspillé |
| | part | part | **Perdu d'avance** — l'appel est gaspillé |
| | **reste** | **part** | **Persuadable** — le seul qui crée de la valeur |
| | part | reste | **À ne pas déranger** — l'appel nuit |

Un modèle de prédiction du NPS classe les clients par **probabilité d'être détracteur**. Or les
détracteurs les plus certains sont massivement des **perdus d'avance** : ils sont déjà décidés.
L'appel leur est offert, il ne change rien, et le budget est consommé.

C'est le paradoxe central : **le client le plus à risque est souvent le moins rentable à
appeler.**

Littérature convergente : Devriendt, Berrevoets et Verbeke, *Why you should stop predicting
customer churn and start using uplift models*, **Information Sciences** —
[KU Leuven](https://lirias.kuleuven.be/server/api/core/bitstreams/52a5adf6-6451-4f5c-b36e-c0ad69163bae/content) ·
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0020025519312022). Ils
introduisent le **Maximum Profit Uplift (MPU)** et montrent que classer les clients par
probabilité de churn est moins efficace que les classer par sensibilité à l'intervention.

---

# 2. Ce que le dataset nous offre sans qu'on l'ait demandé

La colonne `Offer` contient une **quasi-expérience**. Mesuré sur nos données :

| Offre | Clients | Taux de départ | Satisfaction moyenne |
|---|---:|---:|---:|
| Offer A | 520 | **6,7 %** | 3,55 |
| Offer B | 824 | 12,3 % | 3,48 |
| Offer C | 415 | 22,9 % | 3,32 |
| **Aucune offre** | **3 877** | **27,1 %** | **3,24** |
| Offer D | 602 | 26,7 % | 3,25 |
| Offer E | 805 | **52,9 %** | 2,81 |

Regarde la dernière ligne. **Les clients ayant reçu l'offre E partent deux fois plus que ceux
qui n'ont rien reçu.**

La lecture naïve — *« l'offre E fait fuir les clients, supprimons-la »* — est presque
certainement fausse. L'explication réaliste est un **biais d'indication** : l'offre E a été
proposée **aux clients déjà en difficulté**. C'est le même phénomène qu'en médecine, où les
patients qui prennent un médicament sont plus malades que ceux qui n'en prennent pas — et où
conclure que le médicament rend malade serait absurde.

**C'est une démonstration en une ligne de tableau que l'on ne peut pas lire un effet causal dans
des données observationnelles.** Et c'est mesuré sur les données du challenge, pas emprunté à
un article.

---

# 3. Le piège dans lequel il ne faut pas tomber

Tentation évidente : annoncer « je fais de l'uplift modeling ».

**Ce serait exactement la faute qu'on reproche à Mustafa et al. (2021)** — annoncer une méthode
que les données ne permettent pas de soutenir.

L'uplift modeling exige de connaître, pour chaque client, **le traitement reçu et l'issue
observée, avec une assignation aléatoire**. Notre dataset n'a :

- aucune campagne de rétention enregistrée ;
- aucune assignation aléatoire — `Offer` est manifestement ciblée, la ligne « Offer E » le prouve ;
- aucune dimension temporelle permettant un avant/après.

**On ne peut donc pas estimer d'uplift ici. Et le dire est précisément ce qui a de la valeur.**

---

# 4. Ce qu'on livre alors — trois niveaux

## Niveau 1 — Ce qui est demandé, exécuté proprement

Le modèle de prédiction NPS à trois classes, calibré, avec précision@K sur les détracteurs.
C'est le livrable. Il est intégralement fait. **On ne substitue rien.**

## Niveau 2 — La démonstration que la règle de ciblage est perfectible

Une section courte du write-up, chiffrée sur nos données :

1. Rappeler la matrice des quatre types.
2. Montrer, avec la table `Offer` ci-dessus, qu'un effet observationnel n'est pas un effet causal.
3. Quantifier l'enjeu : reprendre le chiffre d'Ascarza — **jusqu'à 7 points de churn en plus**
   avec la bonne règle de ciblage, sur la même campagne et le même budget.
4. Conclure : notre modèle classe par **risque**, ce qui est ce qui était demandé, mais
   l'optimum économique se classerait par **sensibilité**.

## Niveau 3 — Le protocole qui permettrait d'y arriver

Un encadré opérationnel, qui **répond en même temps au § 4.9 de l'énoncé** (boucle de
rétroaction) :

> **Pour passer du ciblage par risque au ciblage par sensibilité, il faut une seule chose :
> randomiser.**
>
> À la prochaine campagne, sur les K clients prioritaires du modèle, en tirer au hasard une
> fraction — 10 à 20 % — qui **ne sera pas contactée**. Ce groupe de contrôle coûte quelques
> clients non retenus. Il achète deux choses irremplaçables :
>
> 1. **la mesure de l'effet réel de la campagne** — sans lui, on ne saura jamais si les clients
>    retenus l'ont été grâce à l'appel ou malgré lui ;
> 2. **les données nécessaires à un modèle d'uplift** au cycle suivant : traitement connu, issue
>    connue, assignation aléatoire.
>
> Sans ce groupe de contrôle, il se produit pire que rien : un **empoisonnement des données
> d'entraînement futures**. Un détracteur appelé puis retenu change de classe, et le modèle
> réapprend l'effet de sa propre intervention comme s'il s'agissait d'une propriété du client.
> Le modèle se dégrade, et personne ne voit pourquoi.

C'est le seul passage du livrable qui **demande une décision à la direction** plutôt que de lui
présenter un résultat. C'est aussi celui qu'un directeur Expérience Client retiendra.

## Démonstration méthodologique (optionnelle, si le temps le permet)

Appliquer une méthode d'uplift sur `Offer` **en affichant la limite** : comparer l'estimation
naïve (différence brute de taux) et une estimation ajustée sur les covariables (T-learner ou
S-learner), montrer l'écart, et conclure que même l'estimation ajustée reste biaisée par les
variables non observées ayant motivé l'attribution de l'offre.

Objectif : démontrer qu'on maîtrise la technique **et** qu'on connaît ses conditions de validité.
Le second point compte davantage que le premier.

---

# 5. Trois renforcements techniques issus de la même recherche

## a) Donner une colonne vertébrale statistique au scénario MNAR

Notre scénario S3 simule un biais de non-réponse dépendant de la satisfaction elle-même. Ce
mécanisme porte un nom et a un traitement : le **modèle de sélection de Heckman**, qui vaut à son
auteur un prix Nobel d'économie —
[Wikipédia](https://en.wikipedia.org/wiki/Heckman_correction) ·
[primer méthodologique, Int. J. Epidemiology](https://academic.oup.com/ije/article/52/1/5/6969005).

Deux équations : une de **sélection** (qui répond à l'enquête ?) et une de **résultat** (quelle
satisfaction ?), estimées conjointement en modélisant la corrélation de leurs termes d'erreur.
C'est l'approche canonique du MNAR, par opposition aux *pattern mixture models*.

⚠️ **Limite à énoncer, elle est sérieuse.** Heckman exige une **restriction d'exclusion** : une
variable qui prédit la réponse à l'enquête **sans** prédire la satisfaction. Une telle variable
est *« elusive »* en pratique, et le modèle impose en général une hypothèse de normalité
bivariée. Travaux récents sur ces limites :
[A Robust Classifier Under MNAR Sample Selection Bias](https://arxiv.org/pdf/2305.15641).

**Décision.** On nomme Heckman comme le cadre théorique de S3, on l'implémente si une restriction
d'exclusion plausible existe (candidat : facturation dématérialisée — plausiblement liée à la
propension à répondre en ligne, peu liée à la satisfaction du service), **et on documente
l'échec si aucune ne tient**. L'IPW reste la solution de repli. Nommer le cadre et expliquer
pourquoi on ne peut pas l'appliquer pleinement vaut mieux que de l'ignorer.

## b) Trancher la question ordinale sans se perdre

L'énoncé cite **CORAL** et **CORN** au § 10. Vérification faite : ce sont des méthodes
d'**apprentissage profond**, implémentées en PyTorch
([coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch)), conçues pour garantir
la *cohérence de rang* entre les K−1 tâches binaires d'un réseau de neurones.

**Décision.** Sur 7 043 lignes de données tabulaires, un réseau profond n'a aucune chance de
battre un gradient boosting. On **cite CORAL/CORN, on explique pourquoi on ne les retient pas**
— volume insuffisant, complexité injustifiée — et on couvre l'exigence ordinale par la
régression à seuils optimisés sur le kappa quadratique. Écarter une piste avec un argument est
plus fort que l'ignorer, et plus fort encore que de l'utiliser sans raison.

## c) Le benchmark NPS, enfin sourcé

J'avais avancé « −10 à +30 » de mémoire. **C'était faux, et dans le mauvais sens.** Les
benchmarks publiés donnent pour les télécoms des moyennes **positives** mais les plus basses de
tous les secteurs — de l'ordre de **+19 à +34** selon les sources
([CustomerGauge](https://customergauge.com/benchmarks/blog/telecommunications-nps-benchmarks-and-cx-trends) ·
[QuestionPro](https://www.questionpro.com/blog/nps-benchmarks/) ·
[Survicate](https://survicate.com/nps-benchmarks/)).

⚠️ Ce sont des **sources commerciales**, pas de la littérature relue, et leur dispersion est
énorme (19, 31, 34, 54 selon l'éditeur et l'année). À citer comme un ordre de grandeur du secteur,
jamais comme une référence exacte.

**Ce que ça change** : notre **NPS de −42 sous M1 est encore plus anormal** que je ne le disais.
L'écart au secteur n'est pas de 30 points mais de **60 à 75 points**. L'argument s'en trouve
renforcé, pas affaibli.

## d) Une distinction subtile qui vaut le détour : `CLTV`

Gómez-Vargas, Maldonado et Vairetti (2025), *A predict-and-optimize approach to profit-driven
churn prevention*, **European Journal of Operational Research** 324(2):555–566 —
[arXiv](https://arxiv.org/abs/2310.07047) — montrent qu'il faut utiliser la **valeur vie client
individuelle**, et non une CLV moyenne, pour cibler une campagne : l'agrégation détruit
l'information qui permet d'arbitrer.

Or `CLTV` figure sur notre liste de colonnes **suspectes** : c'est une sortie de modèle IBM, donc
interdite comme variable explicative.

**Les deux positions sont compatibles, et c'est tout l'intérêt :**

> `CLTV` est **interdite comme feature** — elle fuit, c'est une prédiction déguisée en donnée.
> `CLTV` est **légitime comme paramètre de décision** — au moment de choisir qui appeler parmi
> les détracteurs prédits, pondérer par la valeur du client est un choix métier, pas une
> information sur la cible.

Une même variable, bannie de la couche de modélisation et admise dans la couche de décision.
Savoir séparer ces deux couches est exactement ce qu'on attend d'un data scientist en conseil —
et c'est une réponse d'entretien qui se retient.

---

# 6. Pourquoi ceci fait gagner le poste

Relis les critères de l'énoncé au § 9. Ils ne demandent pas un bon modèle. Ils demandent :

> *« evidence that you can define and execute a useful ML problem under imperfect,
> business-realistic conditions »* et *« A write-up that a non-data-scientist leader
> (e.g. a Customer Experience director) could read and act on »*.

Un candidat qui livre le modèle demandé a exécuté une commande.

Un candidat qui livre le modèle demandé, **puis** démontre avec ses propres chiffres que la règle
de ciblage de l'énoncé laisse de la valeur sur la table, **puis** propose le protocole exact pour
la récupérer — **a fait le travail d'un consultant**. C'est le métier d'Artefact, littéralement.

Et la posture est juste : on ne conteste pas la commande, on l'exécute intégralement **et** on
signale ce qu'on ferait à l'étape suivante. C'est ce que dit un bon consultant à son client. Ce
n'est ni de l'arrogance ni du hors-sujet — c'est la valeur ajoutée.

**La phrase à placer dans la synthèse exécutive :**

> *Le modèle livré identifie les clients les plus susceptibles d'être détracteurs, comme demandé.
> Il reste qu'appeler les clients les plus à risque n'est pas la stratégie la plus rentable : la
> littérature établit qu'il faut cibler ceux dont le comportement change le plus quand on les
> appelle. Mesurer cette sensibilité ne demande qu'une chose, et elle est gratuite : ne pas
> contacter 10 % des clients ciblés à la prochaine campagne.*

---

# 7. Ce que ça ajoute au plan

Trois insertions, pas une refonte.

| Où | Ajout | Coût |
|---|---|---|
| **Étape 6** (EDA) | analyse de `Offer` × churn × satisfaction — déjà calculée ci-dessus | 15 min |
| **Étape 27** (évaluation métier) | section « limites de la règle de ciblage » : matrice des quatre types, chiffre d'Ascarza, pondération par CLTV en couche de décision | 1 h |
| **Étape 36** (monitoring) | l'encadré « groupe de contrôle » — il sert à la fois la boucle de rétroaction du § 4.9 et l'uplift futur | 30 min |

Plus, en option si le temps le permet, la démonstration T-learner sur `Offer` avec ses réserves.

**Moins de deux heures pour la seule partie du livrable qu'on retiendra.**

---

# Références

- Ascarza, E. (2018). *Retention Futility: Targeting High-Risk Customers Might Be Ineffective*.
  **Journal of Marketing Research** 55(1). Prix Paul E. Green 2018.
- Devriendt, F., Berrevoets, J., Verbeke, W. *Why you should stop predicting customer churn and
  start using uplift models*. **Information Sciences**.
- Gómez-Vargas, N., Maldonado, S., Vairetti, C. (2025). *A predict-and-optimize approach to
  profit-driven churn prevention*. **European Journal of Operational Research** 324(2):555–566.
- Heckman, J. *Sample selection bias as a specification error*. Voir le primer méthodologique,
  **International Journal of Epidemiology** 52(1).
- Cao, W., Mirjalili, V., Raschka, S. (2019). CORAL. — Shi, X., Cao, W., Raschka, S. (2021). CORN.
- Zihayat, M., Ayanso, A., Davoudi, H. et al. (2021). *Leveraging non-respondent data in customer
  satisfaction modeling*. **Journal of Business Research** 135:112–126.
