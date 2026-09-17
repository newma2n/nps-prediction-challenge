# Phase 5 — Évaluation technique

Modèle final : **baseline_logistique**, cible **M3**, entraîné sur **1080** répondants
(S3 — MNAR), évalué sur **5963** clients silencieux dont on connaît la vérité.

## Règle : jamais une métrique globale sans son détail par classe

| Métrique globale | Valeur |
|---|---|
| Kappa quadratique pondéré | **0.370** |
| Macro-F1 | 0.497 |
| Exactitude équilibrée | 0.556 |
| Exactitude brute | 0.498 — *rapportée uniquement pour montrer qu'elle est trompeuse : elle serait de 0,43 en prédisant toujours « Passif »* |

| Classe | precision | rappel | f1 | effectif |
|---|---|---|---|---|
| Détracteur | 0.395 | 0.785 | 0.525 | 1083.0 |
| Passif | 0.583 | 0.324 | 0.416 | 2590.0 |
| Promoteur | 0.540 | 0.559 | 0.549 | 2290.0 |

![](figures/05_confusion_pr.png)

## Lecture

- **Le Détracteur est la classe la mieux rappelée (78%)** — c'est celle qui compte pour
  le métier. Sa précision est faible (39%) : le modèle appelle large. C'est le bon
  arbitrage pour une campagne de rétention, où manquer un détracteur coûte plus qu'un appel inutile ;
  la métrique métier (précision@K) en tient compte.
- **Le Passif est la classe la plus difficile** (rappel 32%) : c'est le milieu, par définition
  ambigu. Cohérent avec Elero et al. (2026), où la classe neutre est aussi la moins bien prédite.
- La courbe précision-rappel est plus informative que la ROC sous déséquilibre : elle montre le
  Détracteur à AP largement au-dessus de son taux de base.

## Calibration : ce qui s'est passé, et la décision prise

Les probabilités servent à prioriser des appels : un modèle mal calibré fait mal allouer le budget.
Deux méthodes testées (isotonique, Platt). **Toutes deux ont écrasé la classe Passif** : rappel
Passif 0.00 après calibration contre 0.32 avant. Cause : sous MNAR, les Passifs sont rares
parmi les répondants (voir protocole), et le calibrateur, ajusté sur peu de lignes, ne prédit
plus jamais la classe intermédiaire.

**Décision — stratégie hybride** : la **classe** vient du modèle non calibré (les trois classes
sont prédites) ; la **probabilité de détraction**, seule utilisée pour classer les appels, vient
du modèle calibré. C'est ce que fait l'application. Vérifié, pas supposé.

**Ce que la courbe de fiabilité révèle** (voir `evaluation_metier.md`, figure de gauche) : les deux
courbes sont **sous la diagonale** — le modèle surestime systématiquement P(Détracteur), calibré ou
non. Cause : le calibrateur est ajusté sur les 15 % de répondants, où les détracteurs sont
sur-représentés par le biais MNAR (71 % de notes extrêmes contre 37 % dans la base). Il apprend
donc le taux de base des répondants, pas celui de la population silencieuse à laquelle le modèle
s'applique. **La calibration sur un échantillon biaisé calibre vers la mauvaise population** — c'est
le problème MNAR qui réapparaît à un autre étage. Conséquence pratique : le **classement** des
clients par P(Détracteur) reste valide (l'ordre est préservé), mais la **valeur absolue** de la
probabilité ne doit pas être lue comme une fréquence — l'application l'affiche, le write-up le dit.
Correction possible : recalibrer avec les poids IPW, ou sur les premières vraies réponses de la
population cible une fois en production.
