"""
Script d'évaluation des modèles entraînés.
Métriques adaptées aux datasets déséquilibrés avec sauvegarde des résultats.
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score,
    balanced_accuracy_score
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_latest_processed_data():
    """Trouve le dernier fichier de données processed."""
    processed_dir = Path("data/processed")
    csv_files = list(processed_dir.glob("clinical_processed_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("Aucun fichier processed trouvé.")
    
    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    return latest_file

def load_test_data(data_file):
    """Charge les données et fait le même split que l'entraînement."""
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(data_file)
    target_col = 'mortality' if 'mortality' in df.columns else 'death'
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Même split que l'entraînement
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_test, y_test

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calcule toutes les métriques d'évaluation."""
    
    metrics = {}
    
    # Métriques de base
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Métriques pour datasets déséquilibrés
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Métriques dérivées
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # = recall
    metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # = precision
    metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return metrics

def evaluate_model(model_path, metadata_path, X_test, y_test):
    """Évalue un modèle spécifique."""
    
    # Charger le modèle et ses métadonnées
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata['model_info']['name']
    logger.info(f"Évaluation du modèle: {model_name}")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculer les métriques
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Rapport détaillé
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Affichage des résultats
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall/Sensitivity: {metrics['recall']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    if metrics.get('roc_auc'):
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
    
    evaluation_results = {
        'model_info': metadata['model_info'],
        'evaluation_date': datetime.now().isoformat(),
        'test_set_size': len(y_test),
        'class_distribution': y_test.value_counts().to_dict(),
        'metrics': {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        'classification_report': class_report
    }
    
    return evaluation_results

def evaluate_all_models():
    """Évalue tous les modèles entraînés."""
    logger.info("=== DÉBUT DE L'ÉVALUATION ===")
    
    try:
        # Charger les données de test
        data_file = find_latest_processed_data()
        X_test, y_test = load_test_data(data_file)
        logger.info(f"Données de test: {len(X_test)} échantillons")
        logger.info(f"Distribution des classes: {y_test.value_counts().to_dict()}")
        
        # Trouver tous les modèles
        models_dir = Path("models")
        model_files = list(models_dir.glob("*.pkl"))
        
        if not model_files:
            raise FileNotFoundError("Aucun modèle trouvé dans models/")
        
        logger.info(f"Modèles à évaluer: {len(model_files)}")
        
        all_evaluations = []
        
        for model_file in model_files:
            # Chercher le fichier de métadonnées correspondant
            metadata_file = model_file.with_suffix('')
            metadata_file = Path(f"{metadata_file}_metadata.json")
            
            if not metadata_file.exists():
                logger.warning(f"Métadonnées non trouvées pour {model_file}")
                continue
            
            # Évaluer le modèle
            try:
                evaluation = evaluate_model(model_file, metadata_file, X_test, y_test)
                all_evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation de {model_file}: {e}")
        
        # Sauvegarder les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_report_file = f"models/evaluation_report_{timestamp}.json"
        
        evaluation_summary = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'data_file': str(data_file),
                'test_set_size': len(X_test),
                'models_evaluated': len(all_evaluations)
            },
            'evaluations': all_evaluations
        }
        
        with open(evaluation_report_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Rapport d'évaluation sauvegardé: {evaluation_report_file}")
        
        # Résumé des résultats
        logger.info("\n=== RÉSUMÉ DES ÉVALUATIONS ===")
        
        if all_evaluations:
            # Créer un tableau de comparaison
            comparison_data = []
            for eval_result in all_evaluations:
                metrics = eval_result['metrics']
                comparison_data.append({
                    'Model': eval_result['model_info']['name'],
                    'Accuracy': metrics['accuracy'],
                    'Balanced_Acc': metrics['balanced_accuracy'],
                    'F1': metrics['f1_score'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'ROC_AUC': metrics.get('roc_auc', 'N/A')
                })
            
            # Afficher le tableau
            df_comparison = pd.DataFrame(comparison_data)
            print("\n📊 Comparaison des modèles:")
            print(df_comparison.to_string(index=False, float_format='{:.4f}'.format))
            
            # Meilleur modèle par métrique
            best_f1 = max(all_evaluations, key=lambda x: x['metrics']['f1_score'])
            best_balanced = max(all_evaluations, key=lambda x: x['metrics']['balanced_accuracy'])
            
            print(f"\n🏆 Meilleur F1 Score: {best_f1['model_info']['name']} ({best_f1['metrics']['f1_score']:.4f})")
            print(f"🎯 Meilleure Balanced Accuracy: {best_balanced['model_info']['name']} ({best_balanced['metrics']['balanced_accuracy']:.4f})")
        
        return True, evaluation_report_file
        
    except Exception as e:
        logger.error(f"Erreur durant l'évaluation: {e}")
        return False, None

def main():
    """Fonction principale."""
    
    # Créer le dossier logs
    Path("logs").mkdir(exist_ok=True)
    
    success, report_file = evaluate_all_models()
    
    if success:
        print(f"\n✅ Évaluation terminée!")
        print(f"   Rapport: {report_file}")
        print("   Utilisez ce rapport pour comparer les modèles.")
    else:
        print("\n❌ Échec de l'évaluation")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
