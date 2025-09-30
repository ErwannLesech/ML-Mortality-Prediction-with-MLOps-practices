# 🏥 ML Mortality Prediction with MLOps Practices

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Dataiku](https://img.shields.io/badge/Dataiku-FF6F3D?style=for-the-badge&logo=dataiku&logoColor=white)](https://www.dataiku.com/)

> 🩺 **Application web de prédiction de mortalité clinique utilisant les pratiques MLOps modernes**

Une solution complète d'aide à la décision clinique qui combine machine learning et interfaces utilisateur modernes pour prédire le risque de mortalité hospitalière basé sur les données cliniques des patients.

## 📋 Table des matières

- [📖 À propos du projet](#-à-propos-du-projet)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Technologies utilisées](#️-technologies-utilisées)
- [🚀 Installation et lancement](#-installation-et-lancement)
- [📡 API Documentation](#-api-documentation)
- [🖥️ Interface utilisateur](#️-interface-utilisateur)
- [🤖 Modèle ML](#-modèle-ml)
- [🐳 Docker](#-docker)
- [🤝 Contribution](#-contribution)
- [📄 Licence](#-licence)

## 📖 À propos du projet

Cette application permet aux professionnels de santé d'évaluer le risque de mortalité hospitalière d'un patient en utilisant :

- **Données démographiques** : âge, sexe, IMC
- **Paramètres vitaux** : tension artérielle systolique et diastolique
- **Analyses biologiques** : glucose, cholestérol, créatinine
- **Antécédents médicaux** : diabète, hypertension, réadmissions
- **Diagnostic principal** : pneumonie, insuffisance cardiaque, AVC, etc.

Le modèle de machine learning, hébergé sur Dataiku, fournit une probabilité de risque avec une interface intuitive pour faciliter la prise de décision clinique.

### ✨ Fonctionnalités principales

- 🔮 **Prédiction en temps réel** du risque de mortalité
- 📊 **Visualisation intuitive** avec graphiques circulaires et métriques
- 🎨 **Interface moderne** avec design responsive
- 🔒 **API sécurisée** avec validation des données
- 🐳 **Déploiement containerisé** avec Docker
- 🚀 **Architecture microservices** (Frontend/Backend séparés)

## 🏗️ Architecture

```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐    HTTP/JSON    ┌─────────────────┐
│                 │ ──────────────► │                 │ ──────────────► │                 │
│  React Frontend │                 │  FastAPI Backend│                 │  Dataiku Model  │
│  (Port 3000)    │ ◄────────────── │  (Port 8000)    │ ◄────────────── │     (API)       │
│                 │                 │                 │                 │                 │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
        │                                    │
        │                                    │
        ▼                                    ▼
┌─────────────────┐                 ┌─────────────────┐
│  Nginx Server   │                 │   Python App    │
│   (Docker)      │                 │   (Docker)      │
└─────────────────┘                 └─────────────────┘
```

## 🛠️ Technologies utilisées

### Backend
- **🐍 Python 3.11+** - Langage principal
- **⚡ FastAPI** - Framework web moderne et rapide
- **🔧 Uvicorn** - Serveur ASGI haute performance
- **📡 HTTPX** - Client HTTP asynchrone
- **🔍 Pydantic** - Validation des données

### Frontend
- **⚛️ React 18** - Bibliothèque UI moderne
- **⚡ Vite** - Build tool ultra-rapide
- **🎨 TailwindCSS** - Framework CSS utility-first
- **📡 Axios** - Client HTTP pour les requêtes API

### DevOps & Infrastructure
- **🐳 Docker & Docker Compose** - Containerisation
- **🌐 Nginx** - Serveur web et reverse proxy
- **☁️ Dataiku** - Plateforme ML pour l'hébergement du modèle

## 🚀 Installation et lancement

### Prérequis

- 🐳 **Docker** et **Docker Compose** installés
- 🔑 **Accès API Dataiku** avec token d'authentification

### 1. Cloner le repository

```bash
git clone https://github.com/ErwannLesech/ML-Mortality-Prediction-with-MLOps-practices.git
cd ML-Mortality-Prediction-with-MLOps-practices
```

### 2. Configuration des variables d'environnement

Créez un fichier `.env` à la racine du projet :

```bash
# Configuration API Dataiku
DATAIKU_API_URL=https://your-dataiku-instance.com/public/api/v1/predict
DATAIKU_API_TOKEN=your-api-token-here
```

### 3. Lancement avec Docker Compose

```bash
# Construction et lancement des services
docker-compose up --build

# En arrière-plan
docker-compose up -d --build
```

### 4. Accès à l'application

- 🌐 **Frontend** : http://localhost:3000
- 🔧 **Backend API** : http://localhost:8000
- 📚 **Documentation API** : http://localhost:8000/docs

### 5. Arrêt des services

```bash
docker-compose down
```

## 📡 API Documentation

### Endpoints principaux

#### `GET /`
Endpoint de base de l'API

#### `GET /health`
Vérification de l'état de santé du service

#### `POST /predict`
Prédiction de mortalité

**Corps de la requête :**
```json
{
  "age": 65,
  "sex": "Male",
  "bmi": 28.5,
  "systolic_bp": 140,
  "diastolic_bp": 90,
  "glucose": 110.0,
  "cholesterol": 200.0,
  "creatinine": 1.2,
  "diabetes": 1,
  "hypertension": 1,
  "diagnosis": "Heart Failure",
  "readmission_30d": 0
}
```

**Réponse :**
```json
{
  "result": {
    "prediction": "1",
    "probas": {
      "0": 0.25,
      "1": 0.75
    }
  }
}
```

## 🖥️ Interface utilisateur

L'interface utilisateur moderne offre :

- 📱 **Design responsive** adapté à tous les écrans
- 🎨 **Thème sombre élégant** avec effets glassmorphism
- 📊 **Visualisation interactive** des résultats
- ✅ **Validation en temps réel** des formulaires
- 🔄 **Feedback utilisateur** avec états de chargement

### Formulaire de saisie

L'interface est organisée en sections logiques :
1. **Informations de base** - Âge, sexe, IMC
2. **Tension artérielle** - Systolique et diastolique
3. **Analyses biologiques** - Glucose, cholestérol, créatinine
4. **Diagnostic principal** - Liste déroulante des pathologies
5. **Antécédents médicaux** - Cases à cocher

### Affichage des résultats

- 🎯 **Graphique circulaire** montrant le pourcentage de risque
- 📊 **Métriques détaillées** avec probabilités
- 💡 **Recommandations cliniques** basées sur le résultat
- 🎨 **Codes couleur** intuitifs (vert = faible risque, rouge = risque élevé)

## 🤖 Modèle ML

Le modèle de machine learning est hébergé sur la plateforme Dataiku et utilise :

- **🎯 Algorithme** : Classification binaire
- **📊 Features** : 12 variables cliniques
- **🎲 Output** : Probabilité de mortalité (0-1)
- **📈 Métriques** : Précision, rappel, F1-score optimisés

## 🐳 Docker

### Structure des conteneurs

```yaml
services:
  backend:
    - Port: 8000
    - Environnement: Python 3.11
    - Framework: FastAPI
    
  frontend:
    - Port: 3000 → 80 (Nginx)
    - Environnement: Node.js
    - Build: Vite + React
```

### Commandes Docker utiles

```bash
# Logs des services
docker-compose logs -f

# Reconstruire un service spécifique
docker-compose build backend
docker-compose build frontend

# Accès shell aux conteneurs
docker-compose exec backend bash
docker-compose exec frontend sh
```

## 🤝 Contribution

Nous accueillons toutes les contributions ! Voici comment procéder :

### 1. Fork du repository

```bash
# Fork via l'interface GitHub, puis clone
git clone https://github.com/VOTRE-USERNAME/ML-Mortality-Prediction-with-MLOps-practices.git
cd ML-Mortality-Prediction-with-MLOps-practices
```

### 2. Configuration du repository upstream

```bash
git remote add upstream https://github.com/ErwannLesech/ML-Mortality-Prediction-with-MLOps-practices.git
git fetch upstream
```

### 3. Création d'une branche de développement

**📋 Nomenclature des branches :**
```bash
# Format: <initiales>/<type>/<description>
# Exemples pour Erwann Lesech (erle) :

git checkout upstream/develop
git checkout -b erle/feature/add-new-prediction-model
git checkout -b erle/fix/frontend-validation-issue  
git checkout -b erle/docs/update-api-documentation
git checkout -b erle/refactor/optimize-backend-performance
```

### 4. Conventional Commits

**📝 Format des commits :**
```bash
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**🏷️ Types de commits :**
```bash
# Nouvelles fonctionnalités
git commit -m "feat(frontend): add patient history visualization component"
git commit -m "feat(backend): implement caching for prediction results"
git commit -m "feat(ml): add new risk factors to prediction model"

# Corrections de bugs
git commit -m "fix(frontend): resolve form validation error on empty fields"
git commit -m "fix(backend): handle timeout errors from Dataiku API"
git commit -m "fix(docker): correct environment variable configuration"

# Documentation
git commit -m "docs(readme): add installation instructions for development"
git commit -m "docs(api): update endpoint documentation with examples"

# Style et formatage
git commit -m "style(frontend): improve responsive design for mobile devices"
git commit -m "style(backend): format code according to PEP8 standards"

# Refactoring
git commit -m "refactor(frontend): extract prediction form into reusable component"
git commit -m "refactor(backend): optimize database queries for better performance"

# Tests
git commit -m "test(backend): add unit tests for prediction endpoint"
git commit -m "test(frontend): add integration tests for form submission"

# Configuration et outils
git commit -m "chore(docker): update base images to latest versions"
git commit -m "chore(deps): update dependencies to fix security vulnerabilities"
```

### 5. Workflow de développement

```bash
# 1. Synchroniser avec develop
git fetch upstream
git merge upstream/develop

# 2. Faire vos modifications et commits
git add .
git commit -m "feat(frontend): add new risk assessment dashboard"

# 3. Push vers votre fork
git push origin erle/feature/add-new-prediction-model

# 4. Créer une Pull Request sur GitHub
# Cibler la branche 'develop' du repository upstream
```

### 6. Standards de code

**Frontend (React/JavaScript) :**
- 📏 Utiliser ESLint et Prettier
- 🧩 Composants fonctionnels avec hooks
- 📝 PropTypes pour la validation
- 🎨 Classes TailwindCSS pour le styling

**Backend (Python) :**
- 📏 Respect de PEP8
- 🔍 Type hints obligatoires
- 🧪 Tests unitaires avec pytest
- 📚 Docstrings pour les fonctions

### 7. Types de contributions

- 🆕 **Nouvelles fonctionnalités** - Ajout de features
- 🐛 **Corrections de bugs** - Résolution de problèmes
- 📚 **Documentation** - Amélioration de la doc
- 🎨 **Interface utilisateur** - Design et UX
- ⚡ **Performance** - Optimisations
- 🧪 **Tests** - Couverture de tests
- 🔒 **Sécurité** - Améliorations sécuritaires

### 8. Processus de review

1. ✅ **Tests automatisés** doivent passer
2. 👥 **Review par les mainteneurs**
3. 🔍 **Vérification de la qualité du code**
4. 📋 **Validation de la nomenclature**
5. 🚀 **Merge dans develop**

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 👥 Équipe

**Projet MLOps - EPITA 2025**

Développé avec ❤️ pour améliorer les soins de santé grâce à l'intelligence artificielle.

---

## 📞 Support

Pour toute question ou problème :

1. 🐛 **Bugs** : Ouvrir une [issue](https://github.com/ErwannLesech/ML-Mortality-Prediction-with-MLOps-practices/issues)
2. 💡 **Suggestions** : Créer une [discussion](https://github.com/ErwannLesech/ML-Mortality-Prediction-with-MLOps-practices/discussions)
3. 📧 **Contact direct** : Via les issues GitHub

---

⚠️ **Disclaimer médical :** Cette application est un outil d'aide à la décision et ne remplace en aucun cas l'avis d'un professionnel de santé qualifié.