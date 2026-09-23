# ==============================================================================
# Image de livraison — « NPS · Rétention »
#
#   docker compose up            → l'application sur http://localhost:8501
#   docker compose --profile pipeline run --rm pipeline   → rejoue toute l'étude
#
# Deux étages, et la raison compte : l'examinateur qui veut seulement VOIR le
# travail n'a pas à télécharger le catalogue de modèles. L'étage `app` ne
# contient que ce dont l'interface a besoin (~0,8 Go) ; l'étage `pipeline`
# ajoute le nécessaire pour tout recalculer depuis les fichiers bruts.
# ==============================================================================
FROM python:3.11-slim AS app

# Pas de .pyc, sortie non tamponnée (les journaux arrivent en direct dans docker logs),
# et pas de cache pip : trois réglages qui allègent l'image et rendent l'exécution lisible.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
# Les deux derniers réglages méritent une explication, car ce sont des protections qu'on désactive.
# Le conteneur est une démo servie derrière un port publié : dès que l'URL n'est pas exactement
# « localhost » — machine distante, adresse IP, reverse proxy — Streamlit refuse le websocket et la
# page reste vide, sans message. Le conteneur ne reçoit aucune donnée de l'utilisateur et n'écrit
# rien : le risque couvert par ces protections est nul ici, l'inconvénient était réel.

# libgomp1 : OpenMP, requis par scikit-learn et par le boosting. curl : sonde de santé.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Les dépendances d'abord, le code ensuite : une modification du code ne réinstalle rien.
COPY docker/requirements-app.txt ./docker/requirements-app.txt
RUN pip install --no-cache-dir -r docker/requirements-app.txt

# Code, configuration et artefacts produits par le pipeline. Les artefacts sont versionnés
# (l'énoncé § 6 les demande), donc l'image est complète sans les fichiers bruts.
COPY config.yaml ./
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY reports/ ./reports/
COPY livrable/ ./livrable/
COPY prompts/ ./prompts/
COPY README.md ./

# Exécution sans privilèges : rien dans ce conteneur n'a besoin de root.
RUN useradd --create-home --uid 10001 nps && chown -R nps:nps /app
USER nps

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/app.py"]


# ==============================================================================
# Étage `pipeline` — tout recalculer depuis les cinq classeurs IBM.
# Construit uniquement sur demande (profil compose « pipeline »).
# ==============================================================================
FROM app AS pipeline
USER root
COPY docker/requirements-pipeline.txt ./docker/requirements-pipeline.txt
RUN pip install --no-cache-dir -r docker/requirements-pipeline.txt
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/
RUN chown -R nps:nps /app
USER nps
CMD ["python", "-m", "src.run"]
