# Comprendre le problème

Document de cadrage — de quoi parle ce challenge, avant toute considération technique.
Il servira de base à la première section du write-up final.

---

## 1. Le NPS, c'est quoi

Une seule question posée au client :

> « Sur une échelle de 0 à 10, quelle est la probabilité que vous recommandiez notre service
> à un proche ? »

Selon la note, le client tombe dans une des trois catégories :

| Note | Catégorie | Ce que ça veut dire |
|---|---|---|
| 9 – 10 | **Promoteur** | il est content et il le dit autour de lui |
| 7 – 8 | **Passif** | il est satisfait mais sans attachement ; il partira si mieux ailleurs |
| 0 – 6 | **Détracteur** | il est mécontent, et il peut nuire à la réputation |

Le **score NPS** de l'entreprise = `% Promoteurs − % Détracteurs`. Il va de −100 à +100.
Les passifs ne comptent pas dans le calcul, mais ils existent et ils comptent pour le business.

Le détail contre-intuitif à retenir : **une note de 6/10 est un détracteur.** Ce qui ressemble
à « moyen » est compté comme négatif. C'est la définition standard, pas une erreur.

Deux propriétés importantes pour la suite :

- Les trois catégories sont **ordonnées** : Détracteur < Passif < Promoteur. Se tromper de
  deux crans (prédire Promoteur pour un Détracteur) est bien plus grave que de se tromper d'un cran.
- La distribution est presque toujours **déséquilibrée** : une catégorie écrase les autres.

---

## 2. Le problème de l'entreprise

Un opérateur télécom envoie régulièrement cette enquête à ses clients.
**Seuls 15 % répondent.**

Conséquence directe : l'entreprise connaît l'état de satisfaction de 15 % de sa base et ignore
tout des 85 % autres. Or c'est dans ces 85 % silencieux que se cachent des détracteurs sur le
point de partir.

L'équipe rétention — celle qui appelle les clients à risque pour les retenir — se retrouve
aveugle. Elle ne peut agir que sur les gens qui ont pris la peine de répondre, c'est-à-dire
rarement ceux qu'il faudrait appeler.

**La mission : deviner la catégorie NPS des 85 % silencieux** à partir de ce que l'entreprise
sait déjà d'eux — leur contrat, leurs factures, leurs services, leur ancienneté, leur historique.

Le bénéfice attendu est concret :
- appeler en priorité les détracteurs prédits, **avant** qu'ils ne résilient ;
- comprendre **pourquoi** les gens deviennent détracteurs, pour corriger la cause ;
- estimer le vrai NPS de toute la base, pas seulement celui des répondants.

---

## 3. Les données dont on dispose

Le dataset **IBM Telco Customer Churn**, 7 043 clients d'un opérateur fictif de Californie,
réparti en 5 fichiers :

| Fichier | Contenu |
|---|---|
| `demographics` | âge, genre, situation familiale, personnes à charge |
| `location` | ville, code postal, coordonnées |
| `services` | forfait, internet, options, montants facturés |
| `status` | satisfaction, statut du client, valeur estimée |
| `population` | population par code postal |

La colonne qui porte tout le projet est **`Satisfaction Score`**, une note de 1 à 5 donnée par
le client. C'est un vrai signal humain présent dans les données — pas une invention.

Répartition réelle :

| Note | Nombre de clients | Part |
|---|---:|---:|
| 1 | 922 | 13,1 % |
| 2 | 518 | 7,4 % |
| 3 | 2 665 | 37,8 % |
| 4 | 1 789 | 25,4 % |
| 5 | 1 149 | 16,3 % |

---

## 4. Ce qu'il faut construire

Concrètement, le livrable est une chaîne complète :

```
données du client  →  modèle  →  Détracteur / Passif / Promoteur
                                 + probabilité de chaque classe
                                 + les raisons de ce classement
                                          ↓
                          liste priorisée pour l'équipe rétention
                                          ↓
                          application où un manager consulte un client
```

Plus un rapport de 3 à 6 pages expliquant les choix, lisible par un directeur non technique.

---

## 5. Pourquoi ce n'est pas simple

Trois difficultés réelles, et c'est sur elles que se joue l'évaluation.

### a) La cible n'existe pas — il faut la fabriquer

Les données donnent une satisfaction de 1 à 5. Le NPS, lui, a trois catégories.
**Personne ne fournit la traduction : c'est à toi de la construire et de la défendre.**

La traduction évidente — 5 = Promoteur, 4 = Passif, 3 et moins = Détracteur — donne
58,3 % de détracteurs, et donc un NPS de **−42**.

Ce chiffre pose problème. La cause est visible : la note 3 représente à elle seule 2 665 clients,
soit 37,8 % de la base, et **84 % d'entre eux sont toujours clients**. Les classer en bloc comme
détracteurs fabrique un résultat que le métier ne reconnaîtra pas.

D'où la vraie question du projet : **que fait-on des « 3 » ?** Il n'y a pas de bonne réponse,
seulement une réponse argumentée. C'est la première décision que tu auras à prendre.

### b) Le piège caché dans les données

Le dataset contient aussi l'information de résiliation (« churn »). Croisée avec la satisfaction :

| Satisfaction | Restés | Partis | % de départs |
|---|---:|---:|---:|
| 1 | 0 | 922 | **100 %** |
| 2 | 0 | 518 | **100 %** |
| 3 | 2 236 | 429 | 16 % |
| 4 | 1 789 | 0 | **0 %** |
| 5 | 1 149 | 0 | **0 %** |

Autrement dit : **connaître le churn, c'est presque connaître la satisfaction.**

Si tu laisses ces colonnes dans le modèle, tu obtiens un score quasi parfait — et un modèle
totalement inutile. Parce qu'en situation réelle, au moment où tu veux prédire qu'un client est
détracteur, **il n'est pas encore parti**. L'information dont ton modèle dépend n'existe pas
encore.

C'est ce qu'on appelle une **fuite de données**. Le dataset en contient plusieurs :

- des colonnes qui sont une **conséquence** de ce qu'on veut prédire (`Churn Label`, `Churn Value`) ;
- des colonnes qui **n'existent que pour certains clients** (`Churn Reason`, remplie uniquement
  pour ceux qui sont partis — sa seule présence trahit la réponse) ;
- des colonnes **produites par un autre modèle** (`Churn Score`, `CLTV`, calculées par IBM SPSS).

Un candidat qui ne voit pas ça rend un travail avec d'excellents chiffres et se disqualifie sans
comprendre pourquoi. **C'est le principal test de ce challenge.**

### c) Ceux qui répondent ne ressemblent pas à ceux qui se taisent

Tu vas entraîner ton modèle sur les 15 % qui ont répondu, puis l'appliquer aux 85 % qui n'ont
rien dit. Mais répondre à une enquête n'est pas un hasard : les très mécontents et les très
contents répondent davantage que les indifférents.

Les deux populations sont donc différentes, et un modèle appris sur l'une se dégrade sur l'autre.

Ce dataset offre une occasion rare : on connaît la satisfaction des **100 %**. On peut donc
simuler le biais, puis **mesurer exactement combien on perd** — ce qu'aucun praticien ne peut
faire en production. C'est l'angle qui distinguera ton travail.

---

## 6. Ce qui est réellement évalué

L'énoncé le dit noir sur blanc : *« We are not evaluating raw predictive performance in isolation. »*

Ce qui compte :

1. Avoir compris que la cible est **ordonnée et déséquilibrée**.
2. Être **honnête** sur la façon dont le label a été construit, et sur les fuites.
3. Commencer par un **modèle simple** avant tout modèle sophistiqué.
4. Mesurer avec des métriques **qui parlent au métier**, pas l'accuracy.
5. Vérifier que le modèle ne **désavantage pas un groupe** de clients.
6. Livrer une interface **utilisable** par une équipe terrain.
7. Écrire un rapport qu'un **directeur non technique** peut lire et actionner.

Et le contexte : le challenge vient d'**Artefact**, un cabinet de conseil en data. Leur métier
est de rendre un travail technique compréhensible et actionnable pour un client. La qualité de
l'explication compte donc autant que la qualité du modèle.

---

## 7. En une phrase

> Transformer une note de satisfaction incomplète et piégée en un outil qui dit à une équipe
> rétention **qui appeler en priorité, et pourquoi** — en étant capable d'expliquer chaque choix
> et d'admettre les limites du résultat.
