# Notes d'oral — soutenance (20 minutes + questions)

Une idée par diapositive. Le titre est le message ; ne pas le relire, le prouver avec le graphique.
Repères de temps cumulés entre crochets.

| # | Diapositive | Temps | Ce qu'on dit, en une phrase | Le chiffre à prononcer |
|---|---|---|---|---|
| 1 | Titre | [0:00] | Le sujet : décider pour les 85 % de clients dont on ignore la satisfaction. | 85 % |
| 2 | Le problème | [0:45] | Trois usages métier, trois critères chiffrés avant le code ; cible ordonnée et déséquilibrée. | K = 500 appels/mois |
| 3 | Démarche CRISP-DM | [1:45] | Six phases, chacune fermée par des décisions justifiées ; tout est vérifiable dans l'étude. | 8 documents, 39 tests |
| 4 | Les fuites | [2:45] | Le dataset contient sa propre réponse : satisfaction 1–2 → 100 % de départs. Trois natures de fuite, exclusion mesurée. | exactitude max 0.65 < 0,85 |
| 5 | La cible | [4:00] | La grille de l'énoncé donne -42 ; les « 3 » arbitrés par un modèle sur les extrêmes penchent promoteur à 72% → NPS +21. Piège de circularité évité. | 2665 clients « 3 » |
| 6 | Features et déséquilibre | [5:15] | Chaque variable a une hypothèse ; double déséquilibre (Passif rare à l'entraînement, majoritaire à l'évaluation) ; pondération choisie après 5 stratégies. | Passif : 7% des répondants |
| 7 | Protocole 15/85 | [6:15] | Les répondants ne sont pas aléatoires : MNAR simulé, extrêmes de 37 % à 71 %. On mesure ce que le biais coûte. | 37 → 71 % |
| 8 | Catalogue | [7:15] | Quinze modèles, sept familles, trois formulations — chacun pour une hypothèse. Pas un leaderboard. | 15 / 7 / 3 |
| 9 | Grille | [8:15] | M3 domine partout ; le biais coûte à tous ; kappas dans une bande de 0,30–0,39 : le plafond est le signal. | 3 × 3 × 15 |
| 10 | Sélection | [9:15] | Choisie par CV sur les répondants + parcimonie, sans regarder le test. CV promet 0.61, test donne 0.38. | ρ = 0.79 |
| 11 | Par classe | [10:30] | Jamais une métrique globale seule ; trois classes vivantes ; Détracteur rappelé à 63%. | kappa 0.384 |
| 12 | Calibration | [11:30] | Testée, vérifiée, corrigée : hybride. Calibrer sur des répondants biaisés calibre vers la mauvaise population. | sous la diagonale |
| 13 | Valeur et NPS simulé | [12:30] | 51% de vrais détracteurs sur 500 appels ; NPS des silencieux +4 pour +20 réel — sous-estimé, dit tel quel. | ×2.8 |
| 14 | Drivers | [13:45] | Contractuels, stables, différents par segment ; un levier par client ; association ≠ cause (offre E). | 1er levier : engagement |
| 15 | Équité | [14:45] | Écart de 26 pts par âge sans variable d'âge : les features reconstruisent l'âge (AUC 0.92). Mitigation chiffrée, décision remontée. | AUC 0.98 pour « marié » |
| 16 | Robustesse | [16:00] | Bruit d'étiquettes graduel ; chaque bloc de features a un coût mesuré. | Contract : le bloc clé |
| 17 | Bonus | [16:45] | TabICL ne gagne pas et coûte 3 ordres de grandeur ; le texte synthétique encode la classe : ne prouve rien. | « not the winner » |
| 18 | Application | [17:30] | Prioriser, analyser, comprendre, surveiller ; tolérante aux inconnus ; tout lu dans les artefacts. | 7 pages |
| 19 | Monitoring | [18:15] | PSI, performance sur nouvelles réponses, quatre déclencheurs, boucle de rétroaction. | 500 labels → réentraîner |
| 20 | Limites et décision | [19:00] | Ce qu'on ne peut pas faire (effet d'un appel) ; la seule décision demandée : groupe de contrôle. | 10–20 % |
| 21 | Conformité | [19:40] | Chaque exigence → réponse → emplacement ; reproductible ; IA déclarée. | 10/10 identiques |
| 22 | À retenir | [20:00] | Cinq chiffres, une phrase : le plafond est dans le signal, pas dans l'algorithme. | — |

## Questions de jury anticipées — et la réponse en trente secondes

1. **Pourquoi ne pas avoir utilisé Churn Value pour construire la cible, comme l'énoncé le suggère ?**
   Parce que satisfaction 1–2 ⇔ churn à 100 % : la cible deviendrait une relabellisation du churn, et le projet un modèle de
   churn déguisé — inutilisable pour détecter un détracteur *avant* qu'il parte. L'énoncé le suggère comme piste ; on l'a
   examinée et refusée avec la mesure (information mutuelle).

2. **Votre mapping M3 n'est-il pas une manière de choisir la cible qui arrange ?**
   Non : le bloc des « 3 » est tranché par un modèle qui ne les a jamais vus, entraîné sur les extrêmes, et la décision porte
   sur le *bloc*, pas sur les individus — sinon la cible deviendrait fonction des features. M1 et M2 sont conservés, comparés
   (grille) et les drivers sont vérifiés stables sous les trois. La corroboration externe (benchmark +19/+34) n'est pas une preuve.

3. **Pourquoi le kappa quadratique comme critère principal ?**
   Cible ordonnée : une erreur de deux crans doit coûter plus qu'une erreur d'un cran ; l'exactitude et le F1 ne le voient
   pas. Il est corrigé du hasard. On le départage par macro-F1 et on publie toujours le rappel par classe à côté.

4. **Comment justifiez-vous la simulation MNAR ? Sa forme est arbitraire.**
   Sa forme est plausible, pas observée — c'est dit. Ce qu'elle apporte : on connaît la satisfaction des 100 %, donc on
   *mesure* le coût du biais (CV 0.61 → test 0.38), ce qu'aucun praticien ne peut faire. Trois mécanismes sont comparés ; le
   réaliste est retenu. En production, la correction passe par de vraies réponses de la population cible.

5. **Pourquoi la logistique ordinale plutôt que le meilleur boosting ?**
   Règle fixée avant de regarder : kappa CV pondéré IPW, puis parcimonie à un écart-type (ESL § 7.10). Huit modèles sont dans
   la marge ; le plus simple gagne. Sur le test, il est deuxième à 0,003 du premier. Et tous les modèles tiennent dans une
   bande de 0,09 : le plafond est le signal, pas l'algorithme.

6. **Vous sélectionnez sur les répondants biaisés : n'est-ce pas contradictoire ?**
   C'est la seule information disponible en production. La pondération IPW estimée (MAR) corrige ce qui est corrigeable ; le
   ρ de 0.79 entre CV et test montre que le classement tient. Choisir sur les silencieux serait choisir sur le test.

7. **La classe Passif à 0,48 de rappel, c'est faible.**
   C'est le milieu ambigu, 7% des répondants sous MNAR. Sans pondération de classes elle tombait à 0,08 ; la
   calibration l'écrasait à 0 — d'où la décision hybride. Elero et al. observent la même difficulté sur la classe neutre.
   Pour le métier, ce qui compte est le classement des détracteurs (AUC 0.82).

8. **Le NPS des silencieux est sous-estimé de 17 points : votre estimateur est-il utilisable ?**
   Pas en valeur absolue, et l'étude le dit : l'intervalle ne couvre que l'aléa d'échantillonnage, pas le biais. Le classement
   des segments est respecté. La correction n'est pas un autre modèle : c'est de vraies réponses de la population cible pour
   recalibrer.

9. **Les probabilités sont surestimées : pourquoi les afficher ?**
   Elles servent à *classer* (l'ordre est valide), pas à lire une fréquence — l'application le dit à chaque écran. Le
   calibrateur est ajusté sur des répondants biaisés : calibrer vers la mauvaise population. Recalibration dès 300 vraies réponses.

10. **Vous excluez l'âge et pourtant le modèle discrimine par l'âge. À quoi sert l'exclusion ?**
    Elle empêche une décision *explicite* sur l'attribut et retire le levier le plus direct ; elle ne suffit pas, et on l'a
    mesuré (AUC de reconstruction 0.92). C'est pour ça qu'il y a un audit, une mitigation chiffrée et une escalade —
    pas une case cochée.

11. **Pourquoi ne pas appliquer la mitigation par seuils d'âge ?**
    Parce que c'est un traitement différencié selon l'âge : légal dans certaines juridictions, pas dans d'autres, et il coûte de
    la précision chez les jeunes. C'est une décision d'Expérience Client et de juridique, pas de data science. Elle est prête et chiffrée.

12. **CLTV est une fuite mais vous l'utilisez dans l'application ?**
    En couche de *décision*, pas de *modélisation* : pondérer qui appeler parmi les détracteurs prédits est un choix métier,
    pas une information sur la cible. Gómez-Vargas et al. (2025) montrent qu'une CLV individuelle bat une CLV moyenne pour cibler.

13. **Pourquoi pas de SMOTE ?**
    Comparé : aucun gain systématique, et il fabrique des clients synthétiques dans un espace one-hot où l'interpolation n'a pas
    de sens métier. La pondération est plus simple, sans hyperparamètre, et laisse les données intactes.

14. **Le texte gagne du kappa : pourquoi ne pas le déployer ?**
    Parce que les verbatims ont été fabriqués à partir de la classe : ils contiennent le label, bruité. Le gain est un artefact
    du protocole. La chaîne est démontrée ; sa valeur réelle ne peut se mesurer que sur de vrais verbatims.

15. **TabICL fait presque aussi bien sans réglage : pourquoi ne pas le retenir ?**
    Il ne bat pas le modèle retenu, il écrase la classe Passif, il coûte trois ordres de grandeur en inférence et n'a pas
    d'explication native. La consigne de l'énoncé : « do not frame it as the winner if it is not ». TabPFN n'a pas pu être
    évalué (accès aux poids).

16. **Quel est le vrai levier d'amélioration ?**
    Pas un modèle de plus. Un groupe de contrôle (10–20 %) à la prochaine campagne : il mesure l'effet réel et fournit les
    données d'un modèle d'uplift — passer du ciblage par risque au ciblage par sensibilité (Ascarza 2018, jusqu'à 7 pts de churn).

17. **Qu'avez-vous fait avec l'IA générative ?**
    Code, structure de l'étude, rédaction, fragments des verbatims — déclaré comme l'énoncé le demande (§ 7). Les choix de
    modélisation, le périmètre et les conclusions sont les miens ; les 39 tests et le test à blanc vérifient ce qui est livré.

## Si le jury n'a que dix minutes

Diapositives 2, 4, 5, 10, 13, 15, 20, 22. Dans cet ordre.
