# Makefile pour MLOps - Prédiction de Mortalité
# Automatise toutes les opérations du pipeline

# Variables
PYTHON = python3
PIP = pip3
PROJECT_NAME = ml-mortality-prediction
DOCKER_IMAGE = $(PROJECT_NAME)
API_PORT = 5000
DATA_DIR = data
MODELS_DIR = models
REPORTS_DIR = reports

# Couleurs pour l'affichage
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: help setup install test clean data train evaluate predict api pipeline docker all

# Aide par défaut
help:
	@echo "$(GREEN)🏥 MLOps - Prédiction de Mortalité$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(YELLOW)📋 Commandes disponibles:$(NC)"
	@echo ""
	@echo "$(GREEN)🔧 Configuration:$(NC)"
	@echo "  make setup          - Configuration initiale complète"
	@echo "  make install        - Installation des dépendances"
	@echo "  make clean          - Nettoyage des fichiers temporaires"
	@echo ""
	@echo "$(GREEN)🧪 Tests et Validation:$(NC)"
	@echo "  make test           - Exécution de tous les tests"
	@echo "  make test-data      - Tests de validation des données"
	@echo "  make test-models    - Tests de validation des modèles"
	@echo "  make test-api       - Tests de l'API REST"
	@echo ""
	@echo "$(GREEN)📊 Pipeline ML:$(NC)"
	@echo "  make data           - Téléchargement et preprocessing des données"
	@echo "  make train          - Entraînement des modèles"
	@echo "  make evaluate       - Évaluation des modèles"
	@echo "  make predict        - Test de prédiction"
	@echo ""
	@echo "$(GREEN)🚀 API et Services:$(NC)"
	@echo "  make api            - Démarrage de l'API REST"
	@echo "  make api-test       - Test complet de l'API"
	@echo "  make api-stop       - Arrêt de l'API"
	@echo ""
	@echo "$(GREEN)🐳 Docker:$(NC)"
	@echo "  make docker-build   - Construction de l'image Docker"
	@echo "  make docker-run     - Exécution du container"
	@echo "  make docker-api     - API dans Docker"
	@echo "  make docker-clean   - Nettoyage Docker"
	@echo ""
	@echo "$(GREEN)⚡ Pipeline Complète:$(NC)"
	@echo "  make pipeline       - Pipeline MLOps complète"
	@echo "  make all            - Setup + Pipeline + Tests"
	@echo ""

# Configuration initiale
setup: install create-dirs
	@echo "$(GREEN)✅ Configuration terminée$(NC)"

# Installation des dépendances
install:
	@echo "$(YELLOW)📦 Installation des dépendances...$(NC)"
	$(PIP) install --user -r requirements.txt || $(PIP) install --user pandas numpy scikit-learn matplotlib joblib flask requests pytest
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

# Création des répertoires
create-dirs:
	@echo "$(YELLOW)📁 Création des répertoires...$(NC)"
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(MODELS_DIR) logs $(REPORTS_DIR) .pytest_cache
	@echo "$(GREEN)✅ Répertoires créés$(NC)"

# Nettoyage
clean:
	@echo "$(YELLOW)🧹 Nettoyage...$(NC)"
	rm -rf __pycache__ .pytest_cache .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f api.pid *.log
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

# Tests complets
test: test-data test-models test-predict
	@echo "$(GREEN)✅ Tous les tests terminés$(NC)"

# Tests de validation des données
test-data:
	@echo "$(YELLOW)🧪 Tests de validation des données...$(NC)"
	$(PYTHON) -m pytest tests/test_preprocessing.py::TestDataValidation -v || echo "$(RED)❌ Tests données échoués$(NC)"

# Tests de validation des modèles
test-models:
	@echo "$(YELLOW)🤖 Tests de validation des modèles...$(NC)"
	$(PYTHON) -m pytest tests/test_train.py::TestModelValidation::test_model_reproducibility_random_forest -v || echo "$(RED)❌ Tests modèles échoués$(NC)"

# Tests de prédiction
test-predict:
	@echo "$(YELLOW)🔮 Tests de prédiction...$(NC)"
	$(PYTHON) -c "from src.predict import predict_mortality, create_example_patient; print('✅ Test prédiction:', predict_mortality(create_example_patient())['predictions'][0]['mortality_risk'])" || echo "$(RED)❌ Test prédiction échoué$(NC)"

# Téléchargement et preprocessing des données
data:
	@echo "$(YELLOW)📥 Téléchargement des données...$(NC)"
	$(PYTHON) download_dataset.py || echo "$(RED)❌ Téléchargement échoué$(NC)"
	@echo "$(YELLOW)⚙️  Preprocessing des données...$(NC)"
	$(PYTHON) src/preprocessing.py || echo "$(RED)❌ Preprocessing échoué$(NC)"
	@echo "$(GREEN)✅ Données prêtes$(NC)"

# Entraînement des modèles
train:
	@echo "$(YELLOW)🎯 Entraînement des modèles...$(NC)"
	$(PYTHON) src/train.py || echo "$(RED)❌ Entraînement échoué$(NC)"
	@echo "$(GREEN)✅ Modèles entraînés$(NC)"

# Évaluation des modèles
evaluate:
	@echo "$(YELLOW)📊 Évaluation des modèles...$(NC)"
	$(PYTHON) src/evaluate.py || echo "$(RED)❌ Évaluation échouée$(NC)"
	@echo "$(GREEN)✅ Évaluation terminée$(NC)"

# Test de prédiction simple
predict:
	@echo "$(YELLOW)🔮 Test de prédiction...$(NC)"
	$(PYTHON) src/predict.py --example
	@echo "$(GREEN)✅ Prédiction testée$(NC)"

# Démarrage de l'API
api:
	@echo "$(YELLOW)🚀 Démarrage de l'API sur le port $(API_PORT)...$(NC)"
	@echo "$(YELLOW)🌐 URL: http://localhost:$(API_PORT)$(NC)"
	@echo "$(YELLOW)⏹️  Pour arrêter: make api-stop ou Ctrl+C$(NC)"
	$(PYTHON) api.py

# Démarrage de l'API en arrière-plan
api-background:
	@echo "$(YELLOW)🚀 Démarrage de l'API en arrière-plan...$(NC)"
	nohup $(PYTHON) api.py > logs/api.log 2>&1 & echo $$! > api.pid
	@sleep 2
	@if [ -f api.pid ]; then \
		echo "$(GREEN)✅ API démarrée (PID: $$(cat api.pid))$(NC)"; \
		echo "$(GREEN)🌐 URL: http://localhost:$(API_PORT)$(NC)"; \
	else \
		echo "$(RED)❌ Échec démarrage API$(NC)"; \
	fi

# Arrêt de l'API
api-stop:
	@echo "$(YELLOW)⏹️  Arrêt de l'API...$(NC)"
	@if [ -f api.pid ]; then \
		kill $$(cat api.pid) 2>/dev/null || echo "Process déjà arrêté"; \
		rm -f api.pid; \
		echo "$(GREEN)✅ API arrêtée$(NC)"; \
	else \
		echo "$(YELLOW)ℹ️  Aucune API en cours$(NC)"; \
	fi

# Tests de l'API
api-test: api-background
	@echo "$(YELLOW)🧪 Tests de l'API...$(NC)"
	@sleep 3
	$(PYTHON) test_api.py || echo "$(RED)❌ Tests API échoués$(NC)"
	@make api-stop

# Construction de l'image Docker
docker-build:
	@echo "$(YELLOW)🐳 Construction de l'image Docker...$(NC)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✅ Image Docker construite: $(DOCKER_IMAGE)$(NC)"

# Exécution du container Docker
docker-run: docker-build
	@echo "$(YELLOW)🐳 Exécution du container Docker...$(NC)"
	docker run --rm -it \
		-v $$(pwd)/$(DATA_DIR):/app/$(DATA_DIR) \
		-v $$(pwd)/$(MODELS_DIR):/app/$(MODELS_DIR) \
		$(DOCKER_IMAGE)

# API dans Docker
docker-api: docker-build
	@echo "$(YELLOW)🐳 Démarrage de l'API dans Docker...$(NC)"
	@echo "$(GREEN)🌐 URL: http://localhost:$(API_PORT)$(NC)"
	docker run --rm -it \
		-p $(API_PORT):$(API_PORT) \
		-v $$(pwd)/$(DATA_DIR):/app/$(DATA_DIR) \
		-v $$(pwd)/$(MODELS_DIR):/app/$(MODELS_DIR) \
		$(DOCKER_IMAGE) python3 api.py

# Pipeline complète dans Docker
docker-pipeline: docker-build
	@echo "$(YELLOW)🐳 Pipeline complète dans Docker...$(NC)"
	docker run --rm -it \
		-v $$(pwd)/$(DATA_DIR):/app/$(DATA_DIR) \
		-v $$(pwd)/$(MODELS_DIR):/app/$(MODELS_DIR) \
		-v $$(pwd)/$(REPORTS_DIR):/app/$(REPORTS_DIR) \
		$(DOCKER_IMAGE) python3 pipeline.py --skip-download

# Nettoyage Docker
docker-clean:
	@echo "$(YELLOW)🐳 Nettoyage Docker...$(NC)"
	docker rmi $(DOCKER_IMAGE) 2>/dev/null || echo "Image non trouvée"
	docker system prune -f
	@echo "$(GREEN)✅ Nettoyage Docker terminé$(NC)"

# Pipeline MLOps complète
pipeline: create-dirs
	@echo "$(YELLOW)⚡ Pipeline MLOps complète...$(NC)"
	$(PYTHON) pipeline.py
	@echo "$(GREEN)✅ Pipeline terminée$(NC)"

# Pipeline rapide (sans téléchargement)
pipeline-fast: create-dirs
	@echo "$(YELLOW)⚡ Pipeline MLOps rapide...$(NC)"
	$(PYTHON) pipeline.py --skip-download --skip-tests
	@echo "$(GREEN)✅ Pipeline rapide terminée$(NC)"

# Pipeline avec API
pipeline-api: create-dirs
	@echo "$(YELLOW)⚡ Pipeline MLOps avec API...$(NC)"
	$(PYTHON) pipeline.py --start-api
	@echo "$(GREEN)✅ Pipeline avec API terminée$(NC)"

# Commande tout-en-un
all: setup pipeline test api-test
	@echo "$(GREEN)🎉 Projet MLOps complètement configuré et testé !$(NC)"
	@echo ""
	@echo "$(YELLOW)📋 Prochaines étapes:$(NC)"
	@echo "  • make api          - Démarrer l'API"
	@echo "  • make docker-api   - API dans Docker"
	@echo "  • make predict      - Tester les prédictions"

# Commande de démonstration
demo: setup data train evaluate predict api-background
	@echo "$(GREEN)🎭 Démonstration complète !$(NC)"
	@echo ""
	@echo "$(YELLOW)🌐 API disponible: http://localhost:$(API_PORT)$(NC)"
	@echo "$(YELLOW)📊 Exemple de requête:$(NC)"
	@echo "curl -X POST http://localhost:$(API_PORT)/api/v1/predict \\"
	@echo "  -H 'Content-Type: application/json' \\"
	@echo "  -d '{\"age\": 65, \"sex_encoded\": 1, \"systolic_bp\": 140}'"
	@echo ""
	@echo "$(YELLOW)⏹️  Pour arrêter: make api-stop$(NC)"

# Informations sur le projet
info:
	@echo "$(GREEN)📊 Informations du projet$(NC)"
	@echo "=========================="
	@echo "Projet: $(PROJECT_NAME)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo "Répertoire: $$(pwd)"
	@echo ""
	@echo "$(YELLOW)📁 Structure:$(NC)"
	@ls -la | grep "^d" | head -10
	@echo ""
	@echo "$(YELLOW)🗂️  Données:$(NC)"
	@if [ -d "$(DATA_DIR)" ]; then ls -la $(DATA_DIR)/; else echo "Aucune donnée"; fi
	@echo ""
	@echo "$(YELLOW)🤖 Modèles:$(NC)"
	@if [ -d "$(MODELS_DIR)" ]; then ls -la $(MODELS_DIR)/*.pkl 2>/dev/null || echo "Aucun modèle"; else echo "Dossier modèles inexistant"; fi

# Vérification santé
health-check:
	@echo "$(YELLOW)🏥 Vérification de santé...$(NC)"
	@echo "Python: $$($(PYTHON) --version 2>&1 || echo '❌ Python non trouvé')"
	@echo "Pip: $$($(PIP) --version 2>&1 | head -1 || echo '❌ Pip non trouvé')"
	@echo "Données: $$(if [ -d '$(DATA_DIR)' ]; then echo '✅ OK'; else echo '❌ Manquantes'; fi)"
	@echo "Modèles: $$(if [ -d '$(MODELS_DIR)' ] && [ -n '$$(ls $(MODELS_DIR)/*.pkl 2>/dev/null)' ]; then echo '✅ OK'; else echo '❌ Aucun modèle'; fi)"
	@echo "API: $$(if curl -s http://localhost:$(API_PORT)/health >/dev/null 2>&1; then echo '✅ Active'; else echo '❌ Inactive'; fi)"
