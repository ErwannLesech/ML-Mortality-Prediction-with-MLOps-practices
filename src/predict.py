"""
Script de prédiction de mortalité pour nouveaux patients.
Interface simple : JSON, CSV ou paramètres en ligne de commande.
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_best_model():
    """Trouve automatiquement le meilleur modèle basé sur les évaluations."""
    models_dir = Path("models")
    
    # Chercher le dernier rapport d'évaluation
    evaluation_files = list(models_dir.glob("evaluation_report_*.json"))
    if not evaluation_files:
        # Fallback: prendre le modèle le plus récent
        model_files = list(models_dir.glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError("Aucun modèle trouvé dans models/")
        
        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        metadata_file = Path(f"{latest_model.stem}_metadata.json")
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Métadonnées non trouvées pour {latest_model}")
        
        logger.info(f"Utilisation du modèle le plus récent: {latest_model}")
        return latest_model, metadata_file
    
    # Utiliser le meilleur modèle selon l'évaluation
    latest_evaluation = max(evaluation_files, key=lambda f: f.stat().st_mtime)
    
    with open(latest_evaluation, 'r') as f:
        evaluation_data = json.load(f)
    
    # Trouver le meilleur modèle selon plusieurs critères
    best_model_info = None
    best_score = -1
    
    for evaluation in evaluation_data['evaluations']:
        metrics = evaluation['metrics']
        
        # Utiliser plusieurs métriques pour le scoring
        # Priorité: ROC-AUC > Average Precision > Balanced Accuracy > Accuracy
        if metrics['roc_auc'] > 0.5:
            score = metrics['roc_auc']
        elif metrics['average_precision'] > 0:
            score = metrics['average_precision']
        elif metrics['balanced_accuracy'] > 0.5:
            score = metrics['balanced_accuracy']
        else:
            score = metrics['accuracy']
        
        if score > best_score:
            best_score = score
            best_model_info = evaluation['model_info']
    
    # Si aucun modèle n'est "bon", prendre le premier disponible
    if not best_model_info:
        best_model_info = evaluation_data['evaluations'][0]['model_info']
        logger.warning("Aucun modèle performant trouvé, utilisation du premier modèle disponible")
    
    model_file = models_dir / best_model_info['filename']
    metadata_file = models_dir / f"{model_file.stem}_metadata.json"
    
    logger.info(f"Meilleur modèle sélectionné: {best_model_info['name']} (Score: {best_score:.4f})")
    
    return model_file, metadata_file

def load_model_and_metadata(model_path, metadata_path):
    """Charge le modèle et ses métadonnées."""
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

def prepare_input_data(input_data, feature_names):
    """Prépare les données d'entrée pour la prédiction."""
    
    if isinstance(input_data, dict):
        # Entrée JSON/dict
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        # Entrée DataFrame
        df = input_data.copy()
    else:
        raise ValueError("Format d'entrée non supporté. Utilisez dict ou DataFrame.")
    
    # Vérifier et ajuster les colonnes
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        logger.warning(f"Features manquantes: {missing_features}")
        # Ajouter les features manquantes avec des valeurs par défaut (0 ou médiane)
        for feature in missing_features:
            df[feature] = 0
    
    # Réorganiser les colonnes dans le bon ordre
    df = df.reindex(columns=feature_names, fill_value=0)
    
    return df

def predict_mortality(input_data, model_path=None, return_proba=True):
    """
    Prédit la mortalité pour un patient ou un ensemble de patients.
    
    Args:
        input_data: dict, DataFrame ou chemin vers fichier CSV
        model_path: chemin vers le modèle (optionnel, auto-détection sinon)
        return_proba: retourner les probabilités (True) ou juste la classe (False)
    
    Returns:
        dict avec prédictions et métadonnées
    """
    
    try:
        # Trouver le modèle à utiliser
        if model_path is None:
            model_file, metadata_file = find_best_model()
        else:
            model_file = Path(model_path)
            metadata_file = Path(f"{model_file.stem}_metadata.json")
        
        # Charger le modèle
        model, metadata = load_model_and_metadata(model_file, metadata_file)
        feature_names = metadata['feature_names']
        
        # Préparer les données d'entrée
        if isinstance(input_data, str):
            # Fichier CSV
            df = pd.read_csv(input_data)
        else:
            df = prepare_input_data(input_data, feature_names)
        
        # Faire les prédictions
        predictions = model.predict(df)
        
        results = {
            'model_info': {
                'name': metadata['model_info']['name'],
                'filename': metadata['model_info']['filename'],
                'performance': metadata['performance']
            },
            'prediction_info': {
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(df)
            },
            'predictions': []
        }
        
        # Calculer les probabilités si disponible
        if hasattr(model, 'predict_proba') and return_proba:
            probabilities = model.predict_proba(df)
            
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results['predictions'].append({
                    'patient_id': i + 1,
                    'mortality_prediction': int(pred),
                    'mortality_risk': 'High' if pred == 1 else 'Low',
                    'probability_death': round(prob[1], 4),
                    'probability_survival': round(prob[0], 4),
                    'confidence': round(max(prob), 4)
                })
        else:
            for i, pred in enumerate(predictions):
                results['predictions'].append({
                    'patient_id': i + 1,
                    'mortality_prediction': int(pred),
                    'mortality_risk': 'High' if pred == 1 else 'Low'
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise

def create_example_patient():
    """Crée un exemple de patient pour test."""
    return {
        'age': 65,
        'sex_encoded': 1,  # 1 pour homme, 0 pour femme
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'heart_rate': 85,
        'temperature': 37.2,
        'respiratory_rate': 20,
        'oxygen_saturation': 95,
        'hemoglobin': 11.5,
        'white_blood_cells': 12.0,
        'platelets': 200,
        'creatinine': 1.8,
        'glucose': 180,
        'sodium': 135,
        'potassium': 4.2,
        # Ajoutez d'autres features selon votre dataset
    }

def main():
    """Interface en ligne de commande."""
    
    parser = argparse.ArgumentParser(description='Prédiction de mortalité')
    parser.add_argument('--input', type=str, help='Fichier CSV d\'entrée')
    parser.add_argument('--model', type=str, help='Chemin vers le modèle')
    parser.add_argument('--example', action='store_true', help='Tester avec un patient exemple')
    parser.add_argument('--output', type=str, help='Fichier de sortie JSON')
    
    args = parser.parse_args()
    
    try:
        if args.example:
            # Test avec un patient exemple
            patient = create_example_patient()
            print("🧪 Test avec patient exemple:")
            print(json.dumps(patient, indent=2))
            
            results = predict_mortality(patient, args.model)
            
        elif args.input:
            # Prédiction depuis fichier CSV
            print(f"📄 Prédiction depuis: {args.input}")
            results = predict_mortality(args.input, args.model)
            
        else:
            print("Usage:")
            print("  python src/predict.py --example")
            print("  python src/predict.py --input patients.csv")
            print("  python src/predict.py --input patients.csv --output predictions.json")
            return
        
        # Afficher les résultats
        print(f"\n🏥 Prédictions de mortalité:")
        print(f"Modèle utilisé: {results['model_info']['name']}")
        print(f"Performance du modèle: F1={results['model_info']['performance']['f1_score']}")
        print(f"Nombre de patients: {results['prediction_info']['n_samples']}")
        
        print(f"\n📊 Résultats:")
        for pred in results['predictions']:
            risk_emoji = "🔴" if pred['mortality_risk'] == 'High' else "🟢"
            print(f"  Patient {pred['patient_id']}: {risk_emoji} {pred['mortality_risk']} Risk", end="")
            if 'probability_death' in pred:
                print(f" (Probabilité: {pred['probability_death']:.1%})")
            else:
                print()
        
        # Sauvegarder si demandé
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Résultats sauvegardés: {args.output}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
