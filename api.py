"""
API REST pour la prédiction de mortalité.
Expose les modèles entraînés via des endpoints HTTP.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import predict_mortality, create_example_patient

# Configuration de l'application Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration globale
API_VERSION = "v1"
MODEL_STATUS = {"loaded": False, "last_check": None}

def validate_patient_data(data):
    """Valide les données d'un patient."""
    required_fields = ['age', 'sex_encoded']
    
    # Vérifier les champs requis
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Champs manquants: {missing_fields}"
    
    # Validation des types et valeurs
    try:
        age = int(data['age'])
        if age < 0 or age > 120:
            return False, "Age doit être entre 0 et 120"
        
        sex = int(data['sex_encoded'])
        if sex not in [0, 1]:
            return False, "sex_encoded doit être 0 (femme) ou 1 (homme)"
        
        # Valider les autres champs numériques s'ils sont présents
        numeric_fields = {
            'systolic_bp': (50, 300),
            'diastolic_bp': (30, 200),
            'heart_rate': (30, 200),
            'temperature': (30.0, 45.0),
            'glucose': (30, 600),
            'bmi': (10.0, 100.0),
            'creatinine': (0.1, 20.0)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in data:
                try:
                    val = float(data[field])
                    if val < min_val or val > max_val:
                        return False, f"{field} doit être entre {min_val} et {max_val}"
                except (ValueError, TypeError):
                    return False, f"{field} doit être numérique"
        
        return True, "Validation OK"
        
    except (ValueError, TypeError) as e:
        return False, f"Erreur de validation: {e}"

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil de l'API."""
    return jsonify({
        'service': 'API Prédiction de Mortalité',
        'version': API_VERSION,
        'status': 'active',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': '/health',
            'predict': '/api/v1/predict',
            'predict_batch': '/api/v1/predict/batch',
            'model_info': '/api/v1/model/info',
            'example': '/api/v1/example'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de santé de l'API."""
    try:
        # Vérifier que les modèles sont disponibles
        models_dir = Path("models")
        model_files = list(models_dir.glob("*.pkl"))
        
        status = {
            'status': 'healthy' if model_files else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'models_available': len(model_files),
            'models_directory': str(models_dir.absolute()),
            'version': API_VERSION
        }
        
        return jsonify(status), 200 if model_files else 503
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route(f'/api/{API_VERSION}/example', methods=['GET'])
def get_example_patient():
    """Retourne un exemple de patient pour tester l'API."""
    try:
        example = create_example_patient()
        
        return jsonify({
            'example_patient': example,
            'description': 'Exemple de patient pour test de prédiction',
            'usage': f'POST /api/{API_VERSION}/predict avec ce JSON en body',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur génération exemple: {e}")
        return jsonify({'error': f'Erreur génération exemple: {e}'}), 500

@app.route(f'/api/{API_VERSION}/predict', methods=['POST'])
def predict_single_patient():
    """Prédiction pour un seul patient."""
    try:
        # Vérifier le content-type
        if not request.is_json:
            return jsonify({'error': 'Content-Type doit être application/json'}), 400
        
        patient_data = request.get_json()
        if not patient_data:
            return jsonify({'error': 'Body JSON requis'}), 400
        
        # Valider les données
        is_valid, validation_msg = validate_patient_data(patient_data)
        if not is_valid:
            return jsonify({'error': validation_msg}), 400
        
        # Effectuer la prédiction
        logger.info(f"Prédiction pour patient: age={patient_data.get('age')}, sex={patient_data.get('sex_encoded')}")
        
        result = predict_mortality(patient_data, return_proba=True)
        
        # Enrichir la réponse
        response = {
            'prediction': result['predictions'][0],
            'model_info': result['model_info'],
            'api_info': {
                'version': API_VERSION,
                'timestamp': datetime.now().isoformat(),
                'patient_data_received': patient_data
            }
        }
        
        logger.info(f"Prédiction réussie: {result['predictions'][0]['mortality_risk']}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        return jsonify({'error': f'Erreur lors de la prédiction: {e}'}), 500

@app.route(f'/api/{API_VERSION}/predict/batch', methods=['POST'])
def predict_batch_patients():
    """Prédiction pour plusieurs patients."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type doit être application/json'}), 400
        
        data = request.get_json()
        
        # Accepter soit 'patients' soit directement une liste
        if 'patients' in data:
            patients_data = data['patients']
        elif isinstance(data, list):
            patients_data = data
        else:
            return jsonify({'error': 'Format attendu: {"patients": [...]} ou directement [...]'}), 400
        
        if not isinstance(patients_data, list) or len(patients_data) == 0:
            return jsonify({'error': 'Liste de patients requise et non vide'}), 400
        
        # Limiter le nombre de patients par batch
        if len(patients_data) > 100:
            return jsonify({'error': 'Maximum 100 patients par batch'}), 400
        
        # Valider chaque patient
        for i, patient in enumerate(patients_data):
            is_valid, validation_msg = validate_patient_data(patient)
            if not is_valid:
                return jsonify({'error': f'Patient {i+1}: {validation_msg}'}), 400
        
        # Créer DataFrame pour prédiction batch
        df = pd.DataFrame(patients_data)
        
        # Effectuer la prédiction
        logger.info(f"Prédiction batch pour {len(patients_data)} patients")
        result = predict_mortality(df, return_proba=True)
        
        response = {
            'predictions': result['predictions'],
            'model_info': result['model_info'],
            'batch_info': {
                'patients_count': len(patients_data),
                'version': API_VERSION,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Prédiction batch réussie: {len(result['predictions'])} prédictions")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Erreur prédiction batch: {e}")
        return jsonify({'error': f'Erreur lors de la prédiction batch: {e}'}), 500

@app.route(f'/api/{API_VERSION}/model/info', methods=['GET'])
def get_model_info():
    """Informations sur le modèle actuel."""
    try:
        from predict import find_best_model, load_model_and_metadata
        
        # Trouver le meilleur modèle
        model_path, metadata_path = find_best_model()
        model, metadata = load_model_and_metadata(model_path, metadata_path)
        
        model_info = {
            'model_details': metadata['model_info'],
            'performance': metadata['performance'],
            'training_date': metadata['training_date'],
            'feature_count': len(metadata['feature_names']),
            'feature_names': metadata['feature_names'][:10],  # Limiter l'affichage
            'api_info': {
                'version': API_VERSION,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(model_info), 200
        
    except Exception as e:
        logger.error(f"Erreur info modèle: {e}")
        return jsonify({'error': f'Erreur récupération info modèle: {e}'}), 500

@app.errorhandler(404)
def not_found(error):
    """Gestionnaire d'erreur 404."""
    return jsonify({
        'error': 'Endpoint non trouvé',
        'available_endpoints': [
            '/',
            '/health',
            f'/api/{API_VERSION}/predict',
            f'/api/{API_VERSION}/predict/batch',
            f'/api/{API_VERSION}/model/info',
            f'/api/{API_VERSION}/example'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Gestionnaire d'erreur 500."""
    logger.error(f"Erreur interne: {error}")
    return jsonify({
        'error': 'Erreur interne du serveur',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Vérifier la disponibilité des modèles au démarrage
    models_dir = Path("models")
    if not models_dir.exists():
        logger.warning("Répertoire models/ non trouvé")
    else:
        model_files = list(models_dir.glob("*.pkl"))
        logger.info(f"Modèles trouvés: {len(model_files)}")
        for model_file in model_files:
            logger.info(f"  - {model_file.name}")
    
    logger.info("🚀 Démarrage API Prédiction de Mortalité")
    logger.info(f"Version: {API_VERSION}")
    logger.info("Endpoints disponibles:")
    logger.info(f"  - GET  /health")
    logger.info(f"  - GET  /api/{API_VERSION}/example")
    logger.info(f"  - POST /api/{API_VERSION}/predict")
    logger.info(f"  - POST /api/{API_VERSION}/predict/batch")
    logger.info(f"  - GET  /api/{API_VERSION}/model/info")
    
    # Lancer le serveur
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Mettre True pour le développement
        threaded=True
    )
