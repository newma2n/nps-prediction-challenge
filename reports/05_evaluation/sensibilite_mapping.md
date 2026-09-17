# Phase 5 — Sensibilité au mapping de la cible

L'énoncé demande : « discuss how sensitive your downstream model is to it ». On rejoue la
régression logistique sous M1, M2 et M3 (mêmes répondants S3) et on compare les drivers.

![](figures/05_sensibilite_mapping.png)

Sur les 12 variables les plus influentes sous M3, **10 conservent le même sens d'effet sous les trois
mappings**. Les conclusions métier (contrat sans engagement, absence de parrainage, faible
ancienneté → risque) sont **robustes au choix du mapping** ; c'est l'amplitude qui varie, pas la
direction.

⚠️ **Les performances brutes ne sont pas comparables entre mappings** (Elero et al. : retirer ou
déplacer le milieu ambigu change mécaniquement la difficulté). On compare des conclusions, pas des
scores.
