"""
Script d'entraînement des modèles de prédiction de mortalité.
Simple et traçable : chaque modèle est sauvegardé avec ses métadonnées.
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_latest_processed_data():
    """Trouve automatiquement le dernier fichier processed."""
    processed_dir = Path("data/processed")
    csv_files = list(processed_dir.glob("clinical_processed_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("Aucun fichier processed trouvé. Lancez d'abord le preprocessing.")
    
    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Données processed détectées: {latest_file}")
    return latest_file

def load_processed_data(file_path):
    """Charge les données preprocessed et sépare features/target."""
    logger.info(f"Chargement des données: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Identifier la colonne target
    target_col = 'mortality' if 'mortality' in df.columns else 'death'
    if target_col not in df.columns:
        raise ValueError("Colonne target 'mortality' ou 'death' non trouvée")
    
    # Séparer features et target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features: {len(X.columns)} colonnes")
    logger.info(f"Target '{target_col}': {y.value_counts().to_dict()}")
    
    return X, y, target_col

def get_model_configs():
    """Configuration des modèles avec hyperparamètres."""
    configs = {
        'RandomForest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'SVM': {
            'model': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'params': {
                'kernel': 'rbf',
                'C': 1.0,
                'probability': True,
                'random_state': 42
            }
        }
    }
    return configs

def calculate_loss(model, X, y):
    """Calcule la loss (log loss) pour un modèle donné."""
    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            return log_loss(y, y_pred_proba)
        else:
            # Pour les modèles sans probabilités, utiliser l'accuracy comme proxy inverse
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            return 1 - accuracy  # Loss approximative
    except Exception as e:
        logger.warning(f"Impossible de calculer la loss: {e}")
        return None

def train_single_model_with_loss_tracking(model, model_name, X_train, X_test, y_train, y_test):
    """Entraîne un seul modèle en suivant la loss par époque/itération."""
    logger.info(f"Entraînement du modèle: {model_name}")
    
    # Initialisation du suivi de loss
    training_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': []
    }
    
    start_time = datetime.now()
    
    # Entraînement avec suivi selon le type de modèle
    if model_name == 'RandomForest':
        # Pour RandomForest, simuler des "époques" avec différents nombres d'arbres
        n_estimators_steps = [10, 25, 50, 75, 100]
        
        for step, n_est in enumerate(n_estimators_steps):
            # Créer un modèle avec ce nombre d'estimateurs
            temp_model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            temp_model.fit(X_train, y_train)
            
            # Calculer les losses
            train_loss = calculate_loss(temp_model, X_train, y_train)
            val_loss = calculate_loss(temp_model, X_test, y_test)
            
            if train_loss is not None and val_loss is not None:
                training_history['epochs'].append(step + 1)
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                logger.info(f"  Étape {step + 1}/{len(n_estimators_steps)}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Entraînement final avec le nombre complet d'estimateurs
        model.fit(X_train, y_train)
    
    elif model_name == 'SVM':
        # Pour SVM, pas d'époques, juste entraînement complet
        model.fit(X_train, y_train)
        
        # Calculer les losses finales
        train_loss = calculate_loss(model, X_train, y_train)
        val_loss = calculate_loss(model, X_test, y_test)
        
        if train_loss is not None and val_loss is not None:
            training_history['epochs'].append(1)
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            logger.info(f"  Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    else:
        # Entraînement standard pour autres modèles
        model.fit(X_train, y_train)
        train_loss = calculate_loss(model, X_train, y_train)
        val_loss = calculate_loss(model, X_test, y_test)
        
        if train_loss is not None and val_loss is not None:
            training_history['epochs'].append(1)
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Prédictions finales
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métriques finales
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        'training_time_seconds': training_time,
        'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else None,
        'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None
    }
    
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "  ROC AUC: N/A")
    logger.info(f"  Training time: {training_time:.2f}s")
    
    return model, metrics, training_history

def save_model_with_metadata(model, model_name, metrics, params, data_info, timestamp, training_history):
    """Sauvegarde le modèle avec un nom intelligent et ses métadonnées incluant l'historique d'entraînement."""
    
    # Nom intelligent du modèle
    # Format: ModelName_YYYYMMDD_HHMMSS_ACC.pkl
    accuracy_str = f"{metrics['accuracy']:.3f}".replace('.', '')
    model_filename = f"{model_name}_{timestamp}_{accuracy_str}.pkl"
    model_path = Path(f"models/{model_filename}")
    
    # Sauvegarder le modèle
    joblib.dump(model, model_path)
    logger.info(f"Modèle sauvegardé: {model_path}")
    
    # Métadonnées complètes
    metadata = {
        "model_info": {
            "name": model_name,
            "filename": model_filename,
            "algorithm": str(type(model).__name__)
        },
        "training_info": {
            "timestamp": datetime.now().isoformat(),
            "data_file": data_info['data_file'],
            "train_size": data_info['train_size'],
            "test_size": data_info['test_size'],
            "features_count": data_info['features_count'],
            "target_distribution": data_info['target_distribution']
        },
        "hyperparameters": params,
        "performance": {
            "accuracy": round(metrics['accuracy'], 4),
            "f1_score": round(metrics['f1_score'], 4),
            "roc_auc": round(metrics['roc_auc'], 4) if metrics['roc_auc'] else None,
            "training_time_seconds": round(metrics['training_time_seconds'], 2),
            "final_train_loss": round(metrics['final_train_loss'], 4) if metrics['final_train_loss'] else None,
            "final_val_loss": round(metrics['final_val_loss'], 4) if metrics['final_val_loss'] else None
        },
        "training_history": training_history,
        "feature_names": data_info['feature_names']
    }
    
    # Sauvegarder les métadonnées
    metadata_filename = f"{model_name}_{timestamp}_{accuracy_str}_metadata.json"
    metadata_path = Path(f"models/{metadata_filename}")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Métadonnées sauvegardées: {metadata_path}")
    
    return model_path, metadata_path

def train_all_models():
    """Pipeline d'entraînement complet."""
    logger.info("=== DÉBUT DE L'ENTRAÎNEMENT ===")
    
    try:
        # 1. Charger les données
        processed_file = find_latest_processed_data()
        X, y, target_col = load_processed_data(processed_file)
        
        # 2. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Split des données: {len(X_train)} train, {len(X_test)} test")
        
        # Informations sur les données pour les métadonnées
        data_info = {
            'data_file': str(processed_file),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features_count': len(X.columns),
            'target_distribution': y.value_counts().to_dict(),
            'feature_names': X.columns.tolist()
        }
        
        # 3. Entraîner tous les modèles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trained_models = []
        
        configs = get_model_configs()
        
        for model_name, config in configs.items():
            logger.info(f"\n--- Entraînement {model_name} ---")
            
            # Entraîner
            trained_model, metrics, training_history = train_single_model_with_loss_tracking(
                config['model'], model_name, X_train, X_test, y_train, y_test
            )
            
            # Sauvegarder
            model_path, metadata_path = save_model_with_metadata(
                trained_model, model_name, metrics, config['params'], data_info, timestamp, training_history
            )
            
            trained_models.append({
                'name': model_name,
                'model_path': model_path,
                'metadata_path': metadata_path,
                'metrics': metrics
            })
        
        # 4. Résumé final
        logger.info(f"\n=== ENTRAÎNEMENT TERMINÉ ===")
        logger.info(f"Modèles entraînés: {len(trained_models)}")
        
        # Trouver le meilleur modèle
        best_model = max(trained_models, key=lambda x: x['metrics']['accuracy'])
        logger.info(f"Meilleur modèle: {best_model['name']} (Accuracy: {best_model['metrics']['accuracy']:.4f})")
        
        return True, trained_models
        
    except Exception as e:
        logger.error(f"Erreur durant l'entraînement: {e}")
        return False, []

def main():
    """Fonction principale."""
    
    # Créer le dossier models et logs
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    success, trained_models = train_all_models()
    
    if success:
        print(f"✅ Entraînement réussi!")
        print(f"   {len(trained_models)} modèles entraînés")
        print("\n📊 Résultats:")
        
        for model_info in trained_models:
            metrics = model_info['metrics']
            print(f"   {model_info['name']}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        best_model = max(trained_models, key=lambda x: x['metrics']['accuracy'])
        print(f"\n🏆 Meilleur modèle: {best_model['name']}")
        print(f"   Fichier: {best_model['model_path']}")
        print("   Prochaine étape: python src/evaluate.py")
    else:
        print("❌ Échec de l'entraînement")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
