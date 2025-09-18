# Guide de Déploiement MLOps - Prédiction de Mortalité

## 🚀 Déploiement Rapide

### 1. Démarrage Complet (Docker)
```bash
# Construire et démarrer tous les services
make docker-build
make docker-up

# L'API sera disponible sur http://localhost:5000
```

### 2. Démarrage Local
```bash
# Installer et configurer l'environnement
make setup
make install

# Exécuter la pipeline complète
make pipeline

# Démarrer l'API
make api
```

## 🔧 Configuration

### Variables d'Environnement
1. Copiez le fichier de configuration :
   ```bash
   cp .env.example .env
   ```

2. Modifiez `.env` selon vos besoins :
   - `FLASK_ENV`: `development` ou `production`
   - `SECRET_KEY`: Clé secrète unique pour la production
   - `API_TOKEN`: Token d'authentification (optionnel)

### Configuration Docker
- **Port de l'API** : 5000 (modifiable dans `docker-compose.yml`)
- **Volumes** : Les données et modèles sont persistés
- **Réseaux** : Réseau isolé pour les services

## 📊 Endpoints de l'API

### Health Check
```bash
curl http://localhost:5000/health
```

### Prédiction Simple
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

### Prédiction par Lot
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {
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
      }
    ]
  }'
```

### Informations du Modèle
```bash
curl http://localhost:5000/model/info
```

### Exemple de Données
```bash
curl http://localhost:5000/example
```

## 🧪 Tests et Validation

### Tests Complets
```bash
make test-all
```

### Tests par Composant
```bash
make test           # Tests unitaires
make test-data      # Validation des données
make test-model     # Validation du modèle
make test-api       # Tests de l'API
```

### Tests Docker
```bash
make docker-test
```

## 📈 Pipeline MLOps

### Exécution Complète
```bash
make pipeline
```

### Étapes Individuelles
```bash
make download-data    # Télécharger les données
make preprocess      # Préprocessing
make train          # Entraînement
make evaluate       # Évaluation
make test-model     # Validation du modèle
```

## 🔄 CI/CD avec GitHub Actions

### Workflows Automatisés
- **Tests** : Exécutés sur chaque push/PR
- **Sécurité** : Scan des vulnérabilités
- **Docker** : Build et test des images
- **Déploiement** : Automatique sur main (configurable)

### Configuration des Secrets GitHub
1. Allez dans Settings > Secrets and variables > Actions
2. Ajoutez :
   - `DOCKER_USERNAME` : Nom d'utilisateur Docker Hub
   - `DOCKER_TOKEN` : Token Docker Hub
   - `DEPLOY_KEY` : Clé SSH pour le serveur de production

## 🐳 Déploiement Docker

### Services Disponibles
- **api** : API Flask de prédiction
- **pipeline** : Exécution de la pipeline MLOps
- **test** : Tests automatisés
- **nginx** : Proxy inverse (production)

### Commandes Docker
```bash
# Construire les images
make docker-build

# Démarrer les services
make docker-up

# Voir les logs
make docker-logs

# Arrêter les services
make docker-down

# Nettoyer
make docker-clean
```

## 🚀 Déploiement Production

### 1. Préparation
```bash
# Variables d'environnement
cp .env.example .env
# Modifier .env avec les valeurs de production

# Construire l'image de production
docker build -t mortality-prediction:prod --target production .
```

### 2. Déploiement avec Docker Compose
```bash
# Profil production avec nginx
docker-compose --profile production up -d
```

### 3. Déploiement Kubernetes (optionnel)
```yaml
# k8s/deployment.yaml (exemple)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mortality-prediction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mortality-prediction
  template:
    metadata:
      labels:
        app: mortality-prediction
    spec:
      containers:
      - name: api
        image: mortality-prediction:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
```

## 📊 Monitoring et Observabilité

### Métriques Disponibles
- **Santé de l'API** : `/health`
- **Métriques du modèle** : `/model/info`
- **Logs structurés** : Format JSON

### Intégration avec Prometheus (optionnel)
```bash
# Activer le monitoring
export MONITORING_ENABLED=true
export PROMETHEUS_PORT=9090
```

## 🔒 Sécurité

### Bonnes Pratiques Implémentées
- Image Docker non-root
- Scan de sécurité automatique
- Validation stricte des entrées
- Logs sécurisés (pas de données sensibles)

### Configuration HTTPS (Production)
```nginx
# nginx.conf (exemple)
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://api:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🆘 Dépannage

### Problèmes Courants

#### API ne démarre pas
```bash
# Vérifier les logs
make docker-logs

# Vérifier les ports
netstat -tulnp | grep 5000
```

#### Modèle introuvable
```bash
# Re-entraîner le modèle
make train

# Vérifier les fichiers
ls -la models/
```

#### Tests en échec
```bash
# Tests détaillés
make test-verbose

# Tests spécifiques
pytest tests/test_data_validation.py -v
```

### Support
- **Issues** : Utilisez GitHub Issues
- **Documentation** : README.md
- **Logs** : `logs/` directory
- **Configuration** : `.env` file

## 📝 Maintenance

### Mise à Jour du Modèle
```bash
# Re-entraîner avec nouvelles données
make pipeline

# Tester le nouveau modèle
make test-model

# Redémarrer l'API
make api-restart
```

### Sauvegarde
```bash
# Sauvegarder les modèles et données
tar -czf backup.tar.gz models/ data/ reports/
```

### Monitoring des Performances
```bash
# Métriques du modèle
curl http://localhost:5000/model/info

# Logs de l'API
tail -f logs/api.log
```
