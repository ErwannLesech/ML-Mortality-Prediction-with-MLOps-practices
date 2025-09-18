"""
Script de preprocessing des données cliniques.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_latest_raw_data():
    """Trouve automatiquement le dernier fichier raw."""
    raw_dir = Path("data/raw")
    csv_files = list(raw_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("Aucun fichier CSV trouvé dans data/raw/")
    
    # Prendre le plus récent par date de modification
    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Fichier raw détecté: {latest_file}")
    return latest_file

def load_and_validate_data(file_path):
    """Charge et valide les données brutes."""
    logger.info(f"Chargement des données depuis: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Validation de base
        if len(df) == 0:
            raise ValueError("Dataset vide")
        
        logger.info(f"Valeurs manquantes: {df.isnull().sum().sum()}")
        logger.info(f"Colonnes: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise

def clean_data(df):
    """Nettoie les données."""
    logger.info("Début du nettoyage des données")
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # 1. Supprimer les doublons
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Doublons supprimés: {duplicates}")
    
    # 2. Identifier les colonnes numériques et catégorielles
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Retirer la target des colonnes si présente
    target_col = 'mortality'
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    logger.info(f"Colonnes numériques: {len(numeric_cols)}")
    logger.info(f"Colonnes catégorielles: {len(categorical_cols)}")
    
    # 3. Traitement des valeurs manquantes
    # Numérique: médiane
    if numeric_cols:
        imputer_num = SimpleImputer(strategy='median')
        df_clean[numeric_cols] = imputer_num.fit_transform(df_clean[numeric_cols])
    
    # Catégorielle: plus fréquente
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_cols] = imputer_cat.fit_transform(df_clean[categorical_cols])
    
    # 4. Traitement des valeurs aberrantes (IQR method)
    outliers_removed = 0
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            # Remplacer par les bornes (winsorization)
            df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
            df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
            outliers_removed += outliers_count
    
    if outliers_removed > 0:
        logger.info(f"Valeurs aberrantes traitées: {outliers_removed}")
    
    logger.info(f"Données nettoyées: {len(df_clean)} lignes (perte: {initial_rows - len(df_clean)})")
    return df_clean

def engineer_features(df):
    """Création de nouvelles features."""
    logger.info("Feature engineering")
    
    df_features = df.copy()
    
    # Exemple de feature engineering (à adapter selon le dataset)
    # 1. Age groups
    if 'age' in df_features.columns:
        df_features['age_group'] = pd.cut(
            df_features['age'], 
            bins=[0, 30, 60, 80, 100], 
            labels=['young', 'middle', 'senior', 'elderly']
        )
    
    # 2. BMI categories
    if 'height' in df_features.columns and 'weight' in df_features.columns:
        df_features['bmi'] = df_features['weight'] / (df_features['height'] / 100) ** 2
        df_features['bmi_category'] = pd.cut(
            df_features['bmi'],
            bins=[0, 18.5, 25, 30, 50],
            labels=['underweight', 'normal', 'overweight', 'obese']
        )
    
    logger.info(f"Features créées, nouvelles colonnes: {len(df_features.columns) - len(df.columns)}")
    return df_features

def encode_categorical(df):
    """Encode les variables catégorielles."""
    logger.info("Encodage des variables catégorielles")
    
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    # Encodage simple (à améliorer selon les besoins)
    label_encoders = {}
    
    for col in categorical_cols:
        if col not in ['mortality']:  # Éviter d'encoder la target
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Colonne encodée: {col} ({len(le.classes_)} catégories)")
    
    return df_encoded, label_encoders

def preprocess_data():
    """Pipeline de preprocessing complet."""
    logger.info("=== DÉBUT DU PREPROCESSING ===")
    
    try:
        # 1. Charger les données
        raw_file = find_latest_raw_data()
        df = load_and_validate_data(raw_file)
        
        # 2. Nettoyer
        df_clean = clean_data(df)
        
        # 3. Feature engineering
        df_features = engineer_features(df_clean)
        
        # 4. Encoder les catégories
        df_final, encoders = encode_categorical(df_features)
        
        # 5. Sauvegarder
        today = datetime.now().strftime("%Y%m%d")
        output_file = f"data/processed/clinical_processed_{today}.csv"
        
        # Créer le dossier processed
        Path("data/processed").mkdir(exist_ok=True)
        
        df_final.to_csv(output_file, index=False)
        logger.info(f"Données sauvegardées: {output_file}")
        
        # 6. Sauvegarder les métadonnées
        metadata = {
            "preprocessing_date": datetime.now().isoformat(),
            "raw_file": str(raw_file),
            "processed_file": output_file,
            "original_rows": len(df),
            "final_rows": len(df_final),
            "original_columns": len(df.columns),
            "final_columns": len(df_final.columns),
            "categorical_encoders": {k: v.classes_.tolist() for k, v in encoders.items()},
            "numeric_columns": df_final.select_dtypes(include=[np.number]).columns.tolist(),
            "target_column": "death" if "death" in df_final.columns else "mortality"
        }
        
        metadata_file = f"data/processed/metadata_{today}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Métadonnées sauvegardées: {metadata_file}")
        logger.info("=== PREPROCESSING TERMINÉ ===")
        
        return True, output_file, metadata_file
        
    except Exception as e:
        logger.error(f"Erreur durant le preprocessing: {e}")
        return False, None, None

def main():
    """Fonction principale."""
    
    # Créer le dossier logs
    Path("logs").mkdir(exist_ok=True)
    
    success, processed_file, metadata_file = preprocess_data()
    
    if success:
        print(f"✅ Preprocessing réussi!")
        print(f"   Fichier: {processed_file}")
        print(f"   Métadonnées: {metadata_file}")
        print("   Prochaine étape: python src/train.py")
    else:
        print("❌ Échec du preprocessing")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
