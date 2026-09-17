# Phase 4 — Protocole de validation 15 % / 85 %

L'énoncé demande un découpage qui reflète « les 15 % de répondants vs les 85 % silencieux ». Un
`train_test_split` stratifié rate l'intention : les répondants à une enquête NPS ne sont pas un
échantillon aléatoire — les très mécontents et les très contents répondent plus. C'est du
**MNAR** (*missing not at random*).

L'avantage rare de ce dataset : on connaît la satisfaction des 100 %. On peut donc simuler le
biais et **mesurer exactement ce qu'il coûte** — ce qu'aucun praticien ne peut faire en production.

## Trois scénarios

| Scénario | Mécanisme de réponse | Ce qu'il mesure |
|---|---|---|
| S1 — MCAR | 15 % tirés au hasard | borne haute optimiste |
| S2 — MAR | propension = f(ancienneté, facture dématérialisée, engagement) | biais de covariables seul |
| S3 — MNAR | propension = f(covariables, **écart de la satisfaction au centre**) | le cas réaliste |

Dans tous les cas : **entraînement sur les 15 % répondants, évaluation sur les 85 % silencieux**
dont on connaît la vérité. Validation croisée stratifiée à l'intérieur des 15 %.

## Le mécanisme mord-il ?

![](figures/04_protocole_15_85.png)

| Scénario | Taux de réponse | Satisf. base | Satisf. répondants | Extrêmes base | Extrêmes répondants |
|---|---|---|---|---|---|
| S1_MCAR | 15.2% | 3.245 | 3.230 | 36.8% | 39.7% |
| S2_MAR | 15.4% | 3.245 | 3.316 | 36.8% | 33.0% |
| S3_MNAR | 15.3% | 3.245 | 3.373 | 36.8% | 71.4% |

Sous S3, la part des notes extrêmes passe de **37% dans la base à 71% chez les
répondants**. Le mécanisme mord : le modèle apprend sur une population qui ne ressemble pas à
celle sur laquelle il sera appliqué.

## Cadre théorique et repondération

S3 est un cas de **sélection sur la variable d'intérêt** — le cadre canonique est le modèle de
sélection de Heckman. Il exige une *restriction d'exclusion* (une variable qui prédit la réponse
sans prédire la satisfaction), notoirement difficile à trouver ; on s'en tient donc à la
**repondération par l'inverse de la propension (IPW)**, testée en phase 4 :
kappa 0.370 sans → 0.369 avec. **Effet nul** : la propension simulée est
une fonction lisse des covariables que le modèle capture déjà. Retenu : sans IPW.

Kannan et al. (2022) observent un taux de réponse réel de **2,6 %** : notre simulation à 15 % est
optimiste.
