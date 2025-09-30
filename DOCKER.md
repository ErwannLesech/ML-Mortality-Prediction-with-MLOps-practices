# Documentation Docker

## Prérequis
- Docker
- Docker Compose

## Configuration

1. Copiez le fichier `.env.example` vers `.env` :
```bash
cp .env.example ./clinical-mortality-app/backend/.env
```

2. Modifiez le fichier `.env` avec vos valeurs Dataiku :
```
DATAIKU_API_URL=votre_url_api_dataiku
DATAIKU_API_TOKEN=votre_token_api_dataiku
```

## Lancement de l'application

### Mode production (avec build)
```bash
docker-compose up --build
```

### Mode détaché (en arrière-plan)
```bash
docker-compose up -d --build
```

### Arrêt de l'application
```bash
docker-compose down
```

### Voir les logs
```bash
# Tous les services
docker-compose logs -f

# Un service spécifique
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Accès aux services

- **Frontend** : http://localhost:3000
- **Backend API** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs

## Structure des conteneurs

### Backend (FastAPI)
- Port : 8000
- Base de données : aucune (proxy vers Dataiku)
- Variables d'environnement : DATAIKU_API_URL, DATAIKU_API_TOKEN

### Frontend (React + Nginx)
- Port : 3000 (mappé sur 80 dans le conteneur)
- Proxy vers le backend via nginx
- Build optimisé pour la production

## Développement

Pour le développement, vous pouvez utiliser les volumes montés qui permettent le rechargement automatique :

```bash
# Le backend utilise uvicorn avec reload automatique
# Le frontend est servi via nginx (pas de hot reload)
```

Si vous voulez le hot reload pour le frontend en développement, utilisez plutôt :
```bash
cd clinical-mortality-app/frontend
npm run dev
```

Et pour le backend :
```bash
cd clinical-mortality-app/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Troubleshooting

### Problème de CORS
Vérifiez que les URLs dans `allow_origins` du backend correspondent à vos URLs d'accès.

### Problème de connexion à Dataiku
Vérifiez vos variables d'environnement dans le fichier `.env`.

### Rebuild des images
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```