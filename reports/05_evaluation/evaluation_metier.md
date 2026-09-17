# Phase 5 — Évaluation métier

L'énoncé : « Accuracy alone is not enough ». La vraie question n'est pas « quel taux de bonnes
réponses » mais **« sur les K clients que l'équipe peut appeler ce mois-ci, combien sont
réellement des détracteurs ? »**

## Précision@K

![](figures/05_calibration_gain.png)

| K appels | Précision@K | Rappel@K | Lift | Détracteurs captés |
|---|---|---|---|---|
| 100 | 55.0% | 5.1% | ×3.03 | 55 |
| 250 | 50.4% | 11.6% | ×2.78 | 126 |
| 500 | 48.8% | 22.5% | ×2.69 | 244 |
| 750 | 47.7% | 33.1% | ×2.63 | 358 |
| 1000 | 46.5% | 42.9% | ×2.56 | 465 |
| 1500 | 43.7% | 60.5% | ×2.40 | 655 |

Taux de base (part de détracteurs dans la population silencieuse) : **18.2%**.
À K = 500, **48.8% des clients appelés sont de vrais détracteurs — 2.7× mieux qu'au hasard.**

## Traduction en euros

Hypothèses (paramètres métier dans `config.yaml`, **à valider avec l'équipe rétention**) :
coût d'un appel 12 €, valeur d'un client retenu 450 €, taux de succès d'un appel 25%.

| | Campagne ciblée par le modèle | Campagne aléatoire de même taille |
|---|---|---|
| Coût | 6,000 € | 6,000 € |
| Clients retenus (estimés) | 61.0 | — |
| Gain net | **21,450 €** | 4,215 € |
| ROI | 3.58 | — |

**Apporté par le modèle : 17,235 € par campagne de 500 appels**, à hypothèses constantes.

## `CLTV` en couche de décision

`CLTV` est interdite comme feature (sortie de modèle IBM : elle fuit) mais **légitime comme
paramètre de décision** : pondérer par la valeur du client au moment de choisir qui appeler est un
choix métier, pas une information sur la cible. Gómez-Vargas et al. (EJOR 2025) montrent qu'une
CLV individuelle bat une CLV moyenne pour cibler une campagne. L'application propose cette
pondération en option.

## Limites de la règle de ciblage — le point que l'on retiendra

Cette étude classe les clients par **risque** d'être détracteur, comme demandé. Or **Ascarza
(2018, *Journal of Marketing Research*, prix Paul E. Green)** établit, par deux expériences de
terrain, que cibler les clients au risque le plus élevé est inefficace : il faut cibler ceux dont la
**sensibilité à l'intervention** est la plus forte. Même campagne, même budget : jusqu'à
**7 points de churn en moins**.

| | On l'appelle | On ne l'appelle pas | |
|---|---|---|---|
| | reste | reste | **acquis** — appel gaspillé |
| | part | part | **perdu d'avance** — appel gaspillé |
| | **reste** | **part** | **persuadable** — seul cas qui crée de la valeur |
| | part | reste | **à ne pas déranger** — l'appel nuit |

Les détracteurs les plus certains sont massivement des *perdus d'avance*. Le client le plus à
risque est souvent le moins rentable à appeler.

**On ne peut pas estimer cette sensibilité ici** : aucune campagne enregistrée, aucune assignation
aléatoire (la table `Offer` de la phase 2 prouve que les offres ont été ciblées). Prétendre faire
de l'uplift modeling serait la faute reprochée à Mustafa et al. La seule chose à faire — et elle
est gratuite — est décrite en phase 6 : **ne pas contacter 10 à 20 % des clients ciblés, tirés au
hasard, à la prochaine campagne.**
