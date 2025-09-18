"""
Script pour télécharger le dataset clinique depuis Kaggle.
"""

import kagglehub
import shutil
from datetime import datetime
from pathlib import Path

def download_clinical_dataset():
    """Télécharge le dataset depuis Kaggle avec versioning (date du jour)."""
    
    print("Téléchargement du dataset depuis Kaggle...")
    
    try:
        # Télécharger depuis Kaggle
        path = kagglehub.dataset_download("uom190346a/synthetic-clinical-tabular-dataset")
        print(f"Dataset téléchargé dans: {path}")
        
        # Créer le dossier data/raw
        raw_data_dir = Path("data/raw")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Version avec date du jour
        today = datetime.now().strftime("%Y%m%d")
        
        # Copier les fichiers avec versioning
        download_path = Path(path)
        for file in download_path.glob("*.csv"):
            # Nom avec version: fichier_20250918.csv
            versioned_name = f"{file.stem}_{today}.csv"
            destination = raw_data_dir / versioned_name
            
            shutil.copy2(file, destination)
            file_size = destination.stat().st_size / (1024 * 1024)
            print(f"Copié: {versioned_name} ({file_size:.1f}MB)")
        
        return True
        
    except Exception as e:
        print(f"Erreur: {e}")
        return False

def main():
    """Fonction principale."""
    
    # Vérifier kagglehub
    try:
        import kagglehub
    except ImportError:
        print("Installation de kagglehub...")
        import os
        os.system("pip install kagglehub")
    
    # Télécharger
    if download_clinical_dataset():
        print("\n✅ Téléchargement terminé!")
        print("Fichiers dans data/raw/ avec la date du jour.")
    else:
        print("\n❌ Échec du téléchargement")

if __name__ == "__main__":
    main()
