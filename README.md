# 🏥 ML Mortality Prediction with MLOps Practices

Un projet complet de prédiction de mortalité sur un dataset clinique implémentant les bonnes pratiques MLOps avec API REST, Docker, et pipelines CI/CD.

## 🎯 Objectif

Développer un système de prédiction de mortalité pour les patients en soins intensifs en utilisant des données cliniques, avec une approche MLOps complète incluant :
- Pipeline ML automatisée
- API REST pour les prédictions
- Tests de validation des données et modèles
- Containerisation Docker
- CI/CD avec GitHub Actions
- Monitoring et observabilité

## 🏗️ Architecture

```
📦 ML-Mortality-Prediction-with-MLOps-practices/
├── 🔧 src/                    # Code source ML
│   ├── preprocessing.py       # Nettoyage et transformation des données
│   ├── train.py              # Entraînement du modèle
│   ├── evaluate.py           # Évaluation et métriques
│   └── predict.py            # Inférence
├── 🌐 api.py                 # API REST Flask
├── 🔄 pipeline.py            # Orchestrateur MLOps
├── 🧪 tests/                 # Tests de validation
│   ├── test_data_validation.py    # Validation des données
│   ├── test_model_validation.py   # Validation du modèle
│   └── test_api.py               # Tests de l'API
├── 📊 notebooks/             # Analyses exploratoires
│   └── eda.ipynb            # Analyse exploratoire des données
├── 🐳 Docker/               # Configuration Docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── 🚀 .github/workflows/    # CI/CD GitHub Actions
├── 📈 data/                 # Données
│   ├── raw/                 # Données brutes
│   └── processed/           # Données préprocessées
├── 🤖 models/               # Modèles entraînés
├── 📋 reports/              # Rapports d'évaluation
└── 🔧 Makefile              # Automation complète
```

## 🚀 Démarrage Rapide

### Option 1 : Docker (Recommandé)
```bash
# Clone du projet
git clone https://github.com/ErwannLesech/ML-Mortality-Prediction-with-MLOps-practices.git
cd ML-Mortality-Prediction-with-MLOps-practices

# Démarrage complet avec Docker
make docker-build
make docker-up

# L'API sera disponible sur http://localhost:5000
```

### Option 2 : Installation locale
```bash
# Configuration de l'environnement
make setup
make install

# Pipeline ML complète
make pipeline

# Démarrage de l'API
make api
```

## 🔧 Utilisation

### API REST

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Prédiction pour un patient
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "gender": "M",
    "admission_type": "EMERGENCY",
    "diagnosis": "SEPSIS",
    "icu_stay_days": 5,
    "mechanical_ventilation": 1,
    "vasopressor_use": 1,
    "creatinine": 2.1,
    "bilirubin": 3.5,
    "platelets": 150000,
    "lactate": 4.2
  }'
```

#### Prédiction par lot
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      { /* données patient 1 */ },
      { /* données patient 2 */ }
    ]
  }'
```

### Pipeline MLOps

```bash
# Pipeline complète automatisée
make pipeline

# Étapes individuelles
make download-data    # Téléchargement des données
make preprocess      # Préprocessing
make train          # Entraînement
make evaluate       # Évaluation
make test-model     # Validation du modèle
```

## 🧪 Tests

### Tests complets
```bash
make test-all
```

### Tests par composant
```bash
make test           # Tests unitaires
make test-data      # Validation des données
make test-model     # Validation du modèle
make test-api       # Tests de l'API
```

## 📊 Modèle ML

### Algorithme
- **Random Forest Classifier** avec optimisation des hyperparamètres
- **Features** : 11 variables cliniques (âge, genre, biomarqueurs, traitements)
- **Performance cible** : Accuracy > 85%, AUC-ROC > 0.85

### Variables d'Entrée
| Variable | Type | Description |
|----------|------|-------------|
| `age` | Numérique | Âge du patient (18-100 ans) |
| `gender` | Catégorique | Genre (M/F) |
| `admission_type` | Catégorique | Type d'admission (EMERGENCY/ELECTIVE/URGENT) |
| `diagnosis` | Catégorique | Diagnostic principal |
| `icu_stay_days` | Numérique | Durée en soins intensifs |
| `mechanical_ventilation` | Binaire | Ventilation mécanique (0/1) |
| `vasopressor_use` | Binaire | Usage de vasopresseurs (0/1) |
| `creatinine` | Numérique | Taux de créatinine (mg/dL) |
| `bilirubin` | Numérique | Taux de bilirubine (mg/dL) |
| `platelets` | Numérique | Nombre de plaquettes |
| `lactate` | Numérique | Taux de lactate (mmol/L) |

## 🐳 Docker

### Services disponibles
```bash
make docker-build     # Construire les images
make docker-up        # Démarrer tous les services
make docker-api       # API uniquement
make docker-test      # Tests dans un conteneur
make docker-logs      # Voir les logs
make docker-down      # Arrêter les services
```

### Services Docker Compose
- **api** : API Flask de prédiction
- **pipeline** : Pipeline MLOps
- **test** : Tests automatisés
- **nginx** : Reverse proxy (production)

## 🔄 CI/CD

### GitHub Actions Workflows
- **Tests** : Tests sur matrices Python 3.8-3.10, multi-OS
- **Sécurité** : Scan des vulnérabilités et audit de sécurité
- **Docker** : Build et test des images Docker
- **Déploiement** : Déploiement automatique (configurable)

### Configuration des Secrets
```bash
DOCKER_USERNAME      # Docker Hub username
DOCKER_TOKEN         # Docker Hub access token
DEPLOY_KEY          # SSH key for deployment
```

## 📈 Monitoring

### Métriques disponibles
- **API** : `/health`, `/model/info`
- **Performance** : Latence, taux de succès
- **Modèle** : Métriques de performance continue

### Logs structurés
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "api",
  "endpoint": "/predict",
  "response_time_ms": 45,
  "prediction": 1,
  "probability": 0.78
}
```

## 🔧 Commandes Make

### Setup et Installation
```bash
make help           # Aide complète
make setup          # Configuration initiale
make install        # Installation des dépendances
make clean          # Nettoyage des fichiers temporaires
```

### Développement
```bash
make lint           # Vérification du code
make format         # Formatage automatique
make test-quick     # Tests rapides
make demo           # Démonstration complète
```

### Production
```bash
make build          # Build pour production
make deploy         # Déploiement (configurable)
make backup         # Sauvegarde des modèles
```

## 📚 Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** : Guide de déploiement complet
- **[TECHNICAL_DOC.md](TECHNICAL_DOC.md)** : Documentation technique détaillée
- **[notebooks/eda.ipynb](notebooks/eda.ipynb)** : Analyse exploratoire des données

## 🛠️ Technologies

### Machine Learning
- **scikit-learn** : Modèles ML et preprocessing
- **pandas** : Manipulation des données
- **numpy** : Calculs numériques
- **matplotlib/seaborn** : Visualisations

### API et Web
- **Flask** : Framework API REST
- **Gunicorn** : Serveur WSGI production
- **nginx** : Reverse proxy

### MLOps et DevOps
- **Docker** : Containerisation
- **GitHub Actions** : CI/CD
- **pytest** : Tests automatisés
- **Makefile** : Automation

### Monitoring
- **Logging** : Logs structurés JSON
- **Health checks** : Monitoring de santé
- **Métriques** : Performance et business metrics

## 🔐 Sécurité

### Mesures implémentées
- ✅ Container non-root
- ✅ Scan de sécurité automatique
- ✅ Validation stricte des entrées
- ✅ Pas de données sensibles dans les logs
- ✅ HTTPS ready

### Conformité
- **GDPR** : Pas de stockage PII
- **HIPAA** : Configuration prête
- **Audit trails** : Logs complets

## 📊 Performance

### Benchmarks
- **Latence API** : < 100ms (p95)
- **Débit** : > 1000 req/sec
- **Accuracy** : > 85%
- **Availability** : > 99.9%

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

### Standards de code
```bash
make lint           # Vérification
make format         # Formatage
make test-all       # Tests complets
```

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- **Équipe MLOps** - *Développement initial* - [ErwannLesech](https://github.com/ErwannLesech)

## 🙏 Remerciements

- Dataset clinique pour la recherche médicale
- Communauté open-source MLOps
- Frameworks et librairies utilisés

## 📞 Support

- **Issues** : [GitHub Issues](https://github.com/ErwannLesech/ML-Mortality-Prediction-with-MLOps-practices/issues)
- **Wiki** : [GitHub Wiki](https://github.com/ErwannLesech/ML-Mortality-Prediction-with-MLOps-practices/wiki)
- **Discussions** : [GitHub Discussions](https://github.com/ErwannLesech/ML-Mortality-Prediction-with-MLOps-practices/discussions)

---

⭐ **N'hésitez pas à mettre une étoile si ce projet vous a été utile !**