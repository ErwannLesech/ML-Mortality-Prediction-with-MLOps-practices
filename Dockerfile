# Dockerfile pour MLOps - Prédiction de Mortalité
# Image Python légère avec toutes les dépendances ML

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="MLOps Team"
LABEL description="API de prédiction de mortalité avec pipeline MLOps"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=api.py
ENV FLASK_ENV=production
ENV API_PORT=5000

# Création du répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de requirements en premier (pour le cache Docker)
COPY requirements.txt requirements-test.txt ./

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-test.txt

# Copie du code source
COPY src/ ./src/
COPY tests/ ./tests/
COPY *.py ./
COPY pytest.ini ./

# Création des répertoires nécessaires
RUN mkdir -p data/raw data/processed models logs reports

# Copie des scripts et configuration
COPY Makefile ./
COPY run_tests.sh ./
RUN chmod +x run_tests.sh

# Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash mlops
RUN chown -R mlops:mlops /app
USER mlops

# Exposition du port API
EXPOSE ${API_PORT}

# Vérification de santé
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Point d'entrée par défaut
CMD ["python3", "api.py"]
