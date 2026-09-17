# Phase 3-4 — Verbatims synthétiques et fusion texte (bonus § 4.4)

## Génération

7043 notes de dernier contact, une par client. **Source : `gabarit_stochastique_seed`.**

- Sans clé API disponible dans l'environnement, le **générateur local à gabarits stochastiques** a
  été utilisé : fragments rédigés par un LLM, assemblage seedé (seed 42) — donc strictement
  reproductible, ce que l'API ne permettrait pas (ni `seed` ni `temperature` sur les modèles actuels).
- Le chemin API est prêt : `src/verbatims.py` détecte `ANTHROPIC_API_KEY` (environnement ou `.env`),
  appelle `claude-opus-5` par lots avec reprise sur incident, prompt versionné `prompts/verbatim_v1.txt`,
  et produit le même schéma. La colonne `source` dit lequel des deux a produit chaque ligne.
- Conditionnement : tonalité tirée de la classe M3 (frustré / neutre / enthousiaste) avec **25 % de
  bruit** (tonalité redistribuée au hasard) + ancrage sur contrat, ancienneté, internet, offre,
  parrainages, facture. Concordance tonalité/classe observée : **83.2%** (plafond
  théorique 83.3%).

## Ce que le texte apporte

| Configuration | Kappa | Macro-F1 | Rappel Détracteur |
|---|---|---|---|
| Tabulaire seul (référence) | 0.370 | 0.497 | 0.785 |
| Texte seul (TF-IDF 1-2 grammes) | 0.381 | 0.481 | 0.622 |
| Fusion tardive (poids texte 0.2) | 0.392 | 0.381 | 0.786 |
| Fusion précoce (tabulaire + SVD 20) | 0.418 | 0.514 | 0.800 |

Poids de fusion choisi par validation croisée sur les répondants :

| Poids du texte | Kappa (CV répondants) |
|---|---|
| 0.0 | 0.622 |
| 0.1 | 0.622 |
| 0.2 | 0.627 |
| 0.3 | 0.620 |
| 0.4 | 0.598 |
| 0.5 | 0.570 |
| 0.6 | 0.506 |
| 0.7 | 0.415 |
| 0.8 | 0.337 |
| 0.9 | 0.226 |
| 1.0 | 0.161 |

Gain de kappa de la fusion tardive sur le tabulaire seul : **+0.022**.

## Pourquoi ce gain ne prouve rien — la limite à lire avant les chiffres

Le texte encode la classe par construction (tonalité tirée de la classe, bruit 25 %). Tout gain est un artefact du protocole de génération : il valide la chaîne, pas la valeur de vrais verbatims.

Autrement dit : le texte a été fabriqué **à partir de la classe**. Il contient donc, par
construction, une information que les features tabulaires n'ont pas — le label lui-même, bruité.
Le dispositif démontre que la chaîne texte → fusion **fonctionne** ; il ne dit **rien** de la valeur
de vrais verbatims en production, qui pourraient être bien plus riches (incidents non enregistrés)
ou bien plus pauvres (notes laconiques). C'est exactement le *« spot when the added complexity is
not worth it »* de l'énoncé : ici, la complexité ajoutée ne vaut pas un déploiement sur la base de
cette seule expérience.
