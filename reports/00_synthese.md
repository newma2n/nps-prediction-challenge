# Synthèse — Prédiction NPS pour la rétention

*Pour un lecteur non technique. Chaque chiffre est calculé et retrouvable dans les rapports qui suivent.*

## Ce qu'on a construit

Un modèle qui, pour chacun des clients n'ayant pas répondu à l'enquête NPS, estime s'il est
Détracteur, Passif ou Promoteur, et une application qui en tire une **liste d'appels priorisée**.

## Les trois résultats à retenir

**1. La grille NPS proposée dans l'énoncé fabrique un résultat faux.** Appliquée telle quelle, elle
donne un NPS de **-42**, à 60–75 points des benchmarks télécom. La cause : elle classe en
détracteurs 2665 clients « moyennement satisfaits » (38 % de la base) dont 84 % sont restés fidèles.
En demandant aux données à qui ces clients ressemblent, on trouve qu'ils penchent à
**72% du côté des promoteurs**. Le NPS corrigé est de **+21**, dans la fourchette du secteur.

**2. Le modèle multiplie par 2.7 l'efficacité des appels.** Sur 500 appels, 49% touchent un
vrai détracteur, contre 18% au hasard. À hypothèses de coût constantes, c'est
**17,235 € de gain net supplémentaire par campagne**. Il rappelle 78% des détracteurs.

**3. Il rate davantage les jeunes détracteurs.** Rappel de 61% chez les moins de 30 ans contre
86% chez les 60 ans et plus — alors qu'aucune variable d'âge n'entre dans le modèle. Écart
mesuré, signalé, à traiter avant production.

## Ce qu'il faut savoir sur la fiabilité

- **Le modèle est entraîné dans les conditions réelles** : sur 15 % de « répondants » dont la
  composition est biaisée (les mécontents et les enthousiastes répondent plus), évalué sur les 85 %
  restants. Ses scores sont donc réalistes, pas optimistes.
- **Aucune variable liée au départ des clients n'est utilisée.** Le dataset en contient plusieurs
  qui auraient donné un score presque parfait et un modèle inutilisable. Elles sont identifiées,
  exclues, et la raison est documentée.
- **Le modèle le plus simple a gagné** face à deux méthodes plus sophistiquées. Ce n'est pas un
  échec des méthodes avancées : c'est ce que les données permettent.

## La décision demandée

Appeler les clients les plus à risque n'est pas la stratégie la plus rentable : la recherche
établit qu'il faut cibler ceux **dont le comportement change le plus quand on les appelle**.
Mesurer cela ne demande qu'une chose, et elle est gratuite : **ne pas contacter 10 à 20 % des
clients ciblés à la prochaine campagne, tirés au hasard.** Sans ce groupe de contrôle, on ne
saura jamais si la campagne a servi — et le modèle se dégradera en réapprenant ses propres effets.

## Ce que cette étude ne fait pas

Elle n'estime pas l'effet d'un appel (aucune campagne dans les données). Elle n'utilise pas de
texte client (les verbatims synthétiques, bonus de l'énoncé, ne sont pas réalisés). Ses hypothèses
économiques sont des paramètres à valider avec l'équipe rétention.
