# Phase 6 — Monitoring, réentraînement, et la décision à prendre

## Implémenté : dérive mesurée (`python -m src.monitoring`)

![](figures/06_monitoring.png)

**Référence** = population d'entraînement (répondants S3). **Courant** = population scorée.
**Mois simulé** = tarif +12 %, 15 % des contrats basculés en mensuel, 10 % migrés vers la fibre.

| | PSI des prédictions | Part de détracteurs prédits |
|---|---|---|
| Courant vs référence | 0.031 | 31.0% (réf. 33.3%) |
| Mois simulé vs référence | **0.060** | **36.6%** |

Variables les plus dérivantes — courant :

| Variable | PSI | Statut |
|---|---|---|
| Total Charges | 0.0482 | stable |
| Total Revenue | 0.0443 | stable |
| Monthly Charge | 0.0414 | stable |
| Tenure in Months | 0.0396 | stable |
| Total Long Distance Charges | 0.0385 | stable |
| nb_services | 0.0357 | stable |
| Internet Type | 0.0313 | stable |
| tranche_anciennete | 0.0301 | stable |

Variables les plus dérivantes — mois simulé :

| Variable | PSI | Statut |
|---|---|---|
| Monthly Charge | 1.3813 | ALERTE |
| charge_par_service | 0.6490 | ALERTE |
| Total Charges | 0.0482 | stable |
| Contract | 0.0475 | stable |
| Total Revenue | 0.0443 | stable |
| Tenure in Months | 0.0396 | stable |
| Total Long Distance Charges | 0.0385 | stable |
| nb_services | 0.0357 | stable |

**Déclencheur de réentraînement sur le mois simulé : DÉCLENCHÉ**
(variables en alerte : Monthly Charge, charge_par_service). Le mécanisme fonctionne : il détecte
une hausse tarifaire et une migration de contrats que personne n'aurait signalées au modèle.

Note de lecture : le PSI « courant » n'est pas nul alors que les deux populations viennent de la
même base. C'est le **biais de sélection S3** qui s'affiche — les répondants ne ressemblent pas aux
silencieux — et c'est exactement ce que le monitoring doit voir.

## Ce qu'on surveille

| Signal | Mesure | Seuil de réentraînement |
|---|---|---|
| Dérive des entrées | distance entre la distribution mensuelle des features et celle d'entraînement (PSI par variable) | PSI > 0,2 sur une variable majeure (`Contract`, `Tenure`, `Monthly Charge`) |
| Dérive des prédictions | part de Détracteurs prédits par mois | variation > 5 points sans cause métier connue |
| Performance réelle | rappel Détracteur sur les **nouvelles réponses d'enquête** | chute > 10 points vs évaluation initiale |
| Équité | rappel Détracteur par groupe d'âge | écart > 15 points |
| Volume de labels | nouvelles réponses reçues | réentraînement dès 500 nouveaux labels, au plus tard tous les 6 mois |

Outil : Evidently (rapports HTML autonomes), cité par l'énoncé.

## La boucle de rétroaction — pire que rien

Un détracteur appelé puis retenu **change de classe**. Si on réentraîne sur ces données, le
modèle apprend l'effet de sa propre intervention comme s'il s'agissait d'une propriété du client.
Il se dégrade, et personne ne voit pourquoi.

## La recommandation qui demande une décision à la direction

> À la prochaine campagne, sur les 500 clients prioritaires du modèle, **tirer au hasard 10 à 20 %
> qui ne seront pas contactés.** Ce groupe de contrôle coûte quelques clients non retenus. Il
> achète deux choses irremplaçables :
>
> 1. **la mesure de l'effet réel de la campagne** — sans lui, on ne saura jamais si les clients
>    retenus l'ont été grâce à l'appel ou malgré lui ;
> 2. **les données d'un modèle d'uplift** au cycle suivant : traitement connu, issue connue,
>    assignation aléatoire — ce qui permettrait de passer du ciblage par *risque* (cette étude)
>    au ciblage par *sensibilité* (Ascarza 2018, jusqu'à 7 points de churn en moins).

C'est le seul passage de l'étude qui demande une décision plutôt que de présenter un résultat.
