"""
Script simple pour tracer les courbes d'entraînement (loss) depuis les métadonnées de modèles.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_training_curve(metadata_file_path):
    """
    Trace la courbe d'entraînement (loss) depuis un fichier de métadonnées.
    
    Args:
        metadata_file_path (str): Chemin vers le fichier de métadonnées JSON
    """
    
    try:
        # Charger les métadonnées
        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)
        
        # Récupérer l'historique d'entraînement
        training_history = metadata.get('training_history', {})
        if not training_history or not training_history.get('epochs'):
            print(f"❌ Pas d'historique d'entraînement dans {metadata_file_path}")
            return
        
        model_name = metadata['model_info']['name']
        epochs = training_history['epochs']
        train_loss = training_history['train_loss']
        val_loss = training_history['val_loss']
        
        # Tracer la courbe
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'o-', color='blue', label='Loss d\'entraînement', linewidth=2, markersize=6)
        plt.plot(epochs, val_loss, 'o-', color='red', label='Loss de validation', linewidth=2, markersize=6)
        
        plt.xlabel('Époque/Itération')
        plt.ylabel('Loss')
        plt.title(f'Courbe d\'entraînement - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Afficher les métriques finales
        perf = metadata['performance']
        info_text = f"Accuracy finale: {perf['accuracy']:.3f}\n"
        if perf.get('final_train_loss'):
            info_text += f"Loss train finale: {perf['final_train_loss']:.3f}\n"
        if perf.get('final_val_loss'):
            info_text += f"Loss val finale: {perf['final_val_loss']:.3f}"
        
        plt.text(0.02, 0.98, info_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Courbe d'entraînement affichée pour {model_name}")
        print(f"   Époques: {len(epochs)}")
        print(f"   Loss finale (train): {train_loss[-1]:.4f}")
        print(f"   Loss finale (val): {val_loss[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ Erreur lors du tracé: {e}")

def main():
    """Fonction principale avec interface simple."""
    
    if len(sys.argv) != 2:
        print("Usage: python plot_training_curve.py <metadata_file.json>")
        print("\nExemples:")
        print("  python plot_training_curve.py models/RandomForest_20250918_143022_0847_metadata.json")
        
        # Lister les fichiers disponibles
        models_dir = Path("models")
        if models_dir.exists():
            metadata_files = list(models_dir.glob("*_metadata.json"))
            if metadata_files:
                print(f"\nFichiers de métadonnées disponibles dans models/:")
                for f in sorted(metadata_files):
                    print(f"  {f}")
        return
    
    metadata_file = sys.argv[1]
    
    if not Path(metadata_file).exists():
        print(f"❌ Fichier non trouvé: {metadata_file}")
        return
    
    plot_training_curve(metadata_file)

if __name__ == "__main__":
    main()
