# Synthèse — Prédire la catégorie NPS des clients silencieux

*Pour un lecteur non technique. Chaque chiffre est calculé et retrouvable dans les six documents de l'étude.*

## Ce qu'on a construit

Un modèle qui, pour chacun des clients n'ayant pas répondu à l'enquête NPS, estime s'il est
Détracteur, Passif ou Promoteur ; une application qui en tire une **liste d'appels priorisée** avec,
pour chaque client, les raisons de la prédiction et le levier à proposer ; une estimation du **NPS des
85 % silencieux** ; et un dispositif de suivi.

## Les cinq chiffres

| | |
|---|---|
| NPS **réel** de la base sous la grille retenue | **+21** — connu ici parce que la satisfaction des 7 043 clients est dans le jeu de données ; ce n'est **pas** une sortie du modèle. Avec la grille de l'énoncé prise au pied de la lettre : **-42** |
| NPS des silencieux, **estimé par le modèle** | **+4** (intervalle +2 à +5). La vérité, connue ici seulement, est **+20** : l'estimateur **sous-estime de 17 points** et l'intervalle ne couvre pas ce biais. Les répondants seuls donneraient +27. Ce chiffre se lit **en relatif** — classement des segments — pas en absolu |
| Sur 500 appels, part de vrais détracteurs | **51%** contre 18% au hasard — **×2.8** |
| Gain net supplémentaire par campagne | **18 585 €** à hypothèses de coût constantes (à valider) |
| Détracteurs retrouvés | **63%** — mais 47% seulement chez les moins de 30 ans |

## Les trois résultats à retenir

**1. La grille NPS proposée dans l'énoncé fabrique un résultat faux.** Appliquée telle quelle, elle donne
un NPS de **-42**, à 60–75 points des benchmarks télécom, parce qu'elle classe en détracteurs 2665
clients « moyennement satisfaits » (38% de la base) dont 84 % sont restés fidèles. En demandant aux
données à qui ces clients ressemblent, on trouve qu'ils penchent à **72% du côté des promoteurs**. Le NPS
corrigé est de **+21**, dans la fourchette du secteur.

**2. Le modèle multiplie par 2.8 l'efficacité des appels**, et il a été choisi honnêtement. Quinze
modèles de sept familles ont été comparés ; le choix s'est fait **sans regarder les clients sur lesquels
il est évalué**, avec une règle qui préfère le plus simple quand les écarts sont dans le bruit. Retenu :
logistique ordinale. Évalué dans les conditions réelles : entraîné sur 15 % de répondants dont la composition est
biaisée (les mécontents et les enthousiastes répondent plus), testé sur les 85 % restants.

**3. Il rate davantage les jeunes détracteurs** : 47% de rappel chez les moins de 30 ans contre 73%
chez les 60 ans et plus — alors qu'aucune variable d'âge n'entre dans le modèle. On a mesuré pourquoi :
les variables contractuelles **reconstruisent** l'âge. Une correction est chiffrée ; la décision de
l'appliquer appartient au métier et au juridique, pas au modèle.

## Ce qu'il faut savoir sur la fiabilité

- **Aucune variable liée au départ des clients n'est utilisée.** Le dataset en contient plusieurs qui
  auraient donné un score presque parfait et un modèle inutilisable. Identifiées, mesurées, exclues.
- **Les probabilités servent à classer les clients entre eux, pas à lire une fréquence** : ajustées sur
  des répondants biaisés, elles sont surestimées. L'application le dit à chaque écran.
- **Ce que le modèle appelle « drivers » sont des associations**, pas des causes. Contrat sans
  engagement, faible ancienneté, absence de parrainage et facture élevée vont avec la détraction ;
  l'offre commerciale E « prédit » le départ parce qu'elle a été proposée à des clients déjà fragiles.

## La décision demandée

Appeler les clients les plus à risque n'est pas la stratégie la plus rentable : la recherche établit
qu'il faut cibler ceux **dont le comportement change le plus quand on les appelle**. Mesurer cela ne
demande qu'une chose, gratuite : **ne pas contacter 10 à 20 % des clients ciblés à la prochaine
campagne, tirés au hasard.** Sans ce groupe de contrôle, on ne saura jamais si la campagne a servi — et
le modèle se dégradera en réapprenant ses propres effets.

## Ce que cette étude ne fait pas

Elle n'estime pas l'effet d'un appel (aucune campagne dans les données). Ses verbatims clients sont
synthétiques et le signal texte, démontré techniquement, n'est pas déployable sur cette base. Ses
hypothèses économiques sont des paramètres à valider avec l'équipe rétention. TabPFN 2.5 n'a pas pu
être évalué — sa licence PriorLabs demande un compte et une clé API — mais TabICL et TabPFN v2 l'ont
été, sur des poids publics.

## Lire la suite

[Phase 1 — métier](01_comprehension_metier.md) · [Phase 2 — données](02_comprehension_donnees.md) ·
[Phase 3 — préparation](03_preparation_donnees.md) · [Phase 4 — modélisation](04_modelisation.md) ·
[Phase 5 — évaluation](05_evaluation.md) · [Phase 6 — déploiement](06_deploiement.md) ·
[Conformité à l'énoncé](07_conformite_enonce.md)
