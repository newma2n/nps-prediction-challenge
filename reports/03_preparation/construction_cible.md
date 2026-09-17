# Phase 3 — Construction de la cible NPS

Le dataset donne une satisfaction de 1 à 5. Le NPS a trois classes. **Personne ne fournit la
traduction : c'est à nous de la construire et de la défendre.**

## Trois mappings

| | Détracteur | Passif | Promoteur | NPS | Logique |
|---|---|---|---|---|---|
| **M1** | 1, 2, 3 | 4 | 5 | **-42.0** | mapping baseline de l'énoncé (fidèle à l'échelle : 3/5 ≈ 5/10 = détracteur) |
| **M2** | 1, 2 | 3, 4 | 5 | **-4.1** | recentré : le 3 est le milieu, donc passif |
| **M3** | 1, 2 | 3 | 4, 5 | **+21.3** | arbitré par les données (ci-dessous) |

![](figures/03_cible_mappings.png)

## Pourquoi M1 pose problème

M1 est le mapping *fidèle à l'échelle* — et c'est précisément pour ça qu'il est faux : une
échelle à 5 points compressée dans les bandes NPS envoie 58 % de la base chez les détracteurs,
dont 2665 clients à « 3 » qui, à 84 %, sont restés. Un NPS de -42 est à 60–75 points
des benchmarks télécom publiés (+19 à +34 ; sources commerciales, dispersion forte). Ce n'est pas
un résultat, c'est un signal d'alerte sur la construction.

## L'arbitrage empirique des « 3 » (méthode Elero et al., 2026)

Plutôt que de décider où vont les 2665 clients ambigus, on demande aux données :

1. entraîner un modèle **uniquement sur les extrêmes non ambigus** — satisfaction 1–2 contre 4–5
   (4378 clients), avec les seules features légitimes ;
2. contrôler qu'il sépare bien ces extrêmes : **80.1%** d'exactitude en validation
   croisée — dans la fourchette attendue, donc sans fuite ;
3. faire passer les 2665 clients à « 3 », jamais vus, à travers ce modèle.

![](figures/03_arbitrage_des_3.png)

**Résultat : 27.9% des « 3 » ressemblent à des détracteurs, 72.1% à des promoteurs.**
Le bloc rejoint donc les **Passifs**. Le NPS qui en découle (+21.3) tombe dans la
fourchette sectorielle — corroboration externe, pas preuve.

## Le piège qu'on a évité

Il était tentant d'étiqueter **chaque** « 3 » individuellement par la prédiction du modèle. Ce
serait une faute : la cible deviendrait une fonction des features, et le modèle aval
réapprendrait sa propre sortie — une circularité cousine de la fuite `Churn Score`. On tranche
donc **le bloc** par la mesure agrégée, et on conserve la probabilité individuelle comme simple
**indicateur métier** (« ce passif a un profil de détracteur à X % »), affichée dans l'application,
jamais utilisée à l'entraînement.

## Interdit

Arbitrer les « 3 » par `Churn Value`. La cible deviendrait une relabellisation du churn et le
projet un modèle de churn déguisé.
