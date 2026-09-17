# Phase 4 — Modèle de fondation tabulaire (bonus)

Consigne de l'énoncé : *« do not frame it as the winner if it is not »*. Même découpage S3/M3,
mêmes features préparées que le modèle final.

TabPFN n'a pas pu être évalué dans cet environnement : `TabPFNHuggingFaceGatedRepoError: HuggingFace authentication error downloading from 'Prior-Labs/tabpfn_3_5'.
This model is gated and requires you to accept its terms.

Please follow these steps:
1. Visit https://huggingface.co/Prior-Labs/tabpfn_3_5 in your browser and accept the terms of use.
2. Log in to your Hugging Face account v`.

Ce qu'on en dirait avec les sources publiques : TabPFN-2.5 annonce un domaine de validité de
~50 000 lignes et ~2 000 features — nos 1080 lignes d'entraînement et 62 features
sont dedans. Limites connues : latence d'inférence, absence d'interprétabilité native, coût de
déploiement. À réévaluer dès que l'environnement le permet.
