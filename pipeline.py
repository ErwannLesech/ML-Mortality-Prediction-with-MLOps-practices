#!/usr/bin/env python3
"""
Pipeline MLOps complète pour la prédiction de mortalité.
Orchestré toutes les étapes : téléchargement → preprocessing → entraînement → évaluation → API.
"""

import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import json
import time
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MLOps-Pipeline')

class MLOpsPipeline:
    """Pipeline MLOps pour la prédiction de mortalité."""
    
    def __init__(self, config_file=None):
        self.config = self.load_config(config_file)
        self.pipeline_start = datetime.now()
        self.results = {
            'pipeline_id': f"pipeline_{self.pipeline_start.strftime('%Y%m%d_%H%M%S')}",
            'start_time': self.pipeline_start.isoformat(),
            'steps': {},
            'status': 'started'
        }
        
        # Créer les répertoires nécessaires
        self.ensure_directories()
    
    def load_config(self, config_file):
        """Charge la configuration de la pipeline."""
        default_config = {
            'data': {
                'source': 'kaggle',
                'dataset_name': 'clinical_mortality_data',
                'skip_download': False
            },
            'preprocessing': {
                'clean_outliers': True,
                'feature_engineering': True,
                'validation_checks': True
            },
            'training': {
                'models': ['RandomForest', 'SVM'],
                'cross_validation': True,
                'save_models': True
            },
            'evaluation': {
                'test_size': 0.3,
                'metrics': ['accuracy', 'roc_auc', 'f1_score', 'balanced_accuracy'],
                'generate_report': True
            },
            'api': {
                'auto_start': False,
                'port': 5000,
                'host': '0.0.0.0'
            },
            'pipeline': {
                'run_tests': True,
                'fail_on_test_error': True,
                'cleanup_temp_files': True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
                logger.info(f"Configuration chargée depuis {config_file}")
            except Exception as e:
                logger.warning(f"Erreur chargement config {config_file}: {e}. Utilisation config par défaut.")
        
        return default_config
    
    def ensure_directories(self):
        """Crée les répertoires nécessaires."""
        directories = [
            'data/raw',
            'data/processed',
            'models',
            'logs',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_command(self, command, step_name, check_success=True):
        """Exécute une commande et log le résultat."""
        logger.info(f"🔄 Exécution étape: {step_name}")
        logger.info(f"Commande: {command}")
        
        step_start = datetime.now()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 heure max par étape
            )
            
            step_duration = (datetime.now() - step_start).total_seconds()
            
            self.results['steps'][step_name] = {
                'command': command,
                'return_code': result.returncode,
                'duration_seconds': step_duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': step_start.isoformat()
            }
            
            if result.returncode == 0:
                logger.info(f"✅ {step_name} réussi ({step_duration:.1f}s)")
                if result.stdout.strip():
                    logger.info(f"Output: {result.stdout.strip()[:200]}...")
                return True, result.stdout
            else:
                logger.error(f"❌ {step_name} échoué (code {result.returncode})")
                logger.error(f"Stderr: {result.stderr}")
                if check_success:
                    raise Exception(f"Étape {step_name} échouée: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {step_name} timeout (>1h)")
            self.results['steps'][step_name] = {
                'error': 'timeout',
                'duration_seconds': 3600,
                'timestamp': step_start.isoformat()
            }
            if check_success:
                raise Exception(f"Timeout étape {step_name}")
            return False, "Timeout"
        
        except Exception as e:
            logger.error(f"💥 Erreur {step_name}: {e}")
            self.results['steps'][step_name] = {
                'error': str(e),
                'timestamp': step_start.isoformat()
            }
            if check_success:
                raise
            return False, str(e)
    
    def step_download_data(self):
        """Étape 1: Téléchargement des données."""
        if self.config['data']['skip_download']:
            logger.info("⏭️  Téléchargement ignoré (skip_download=True)")
            return True, "Skipped"
        
        return self.run_command(
            "python3 download_dataset.py",
            "download_data"
        )
    
    def step_preprocess_data(self):
        """Étape 2: Preprocessing des données."""
        return self.run_command(
            "python3 src/preprocessing.py",
            "preprocess_data"
        )
    
    def step_train_models(self):
        """Étape 3: Entraînement des modèles."""
        return self.run_command(
            "python3 src/train.py",
            "train_models"
        )
    
    def step_evaluate_models(self):
        """Étape 4: Évaluation des modèles."""
        return self.run_command(
            "python3 src/evaluate.py",
            "evaluate_models"
        )
    
    def step_run_tests(self):
        """Étape 5: Tests de validation."""
        if not self.config['pipeline']['run_tests']:
            logger.info("⏭️  Tests ignorés (run_tests=False)")
            return True, "Skipped"
        
        # Tests de validation des données
        success_data, _ = self.run_command(
            "python3 -m pytest tests/test_preprocessing.py::TestDataValidation -v",
            "test_data_validation",
            check_success=False
        )
        
        # Tests de validation des modèles
        success_models, _ = self.run_command(
            "python3 -m pytest tests/test_train.py::TestModelValidation::test_model_reproducibility_random_forest tests/test_train.py::TestModelValidation::test_cross_validation_stability -v",
            "test_model_validation",
            check_success=False
        )
        
        # Tests de l'API de prédiction
        success_predict, _ = self.run_command(
            "python3 -c \"from predict import predict_mortality, create_example_patient; print('Test prédiction:', predict_mortality(create_example_patient())['predictions'][0]['mortality_risk'])\"",
            "test_prediction",
            check_success=False
        )
        
        overall_success = success_data and success_models and success_predict
        
        if not overall_success and self.config['pipeline']['fail_on_test_error']:
            raise Exception("Tests échoués et fail_on_test_error=True")
        
        return overall_success, f"Data: {success_data}, Models: {success_models}, Predict: {success_predict}"
    
    def step_generate_reports(self):
        """Étape 6: Génération des rapports."""
        # Créer un rapport de pipeline
        pipeline_report = {
            'pipeline_info': self.results,
            'configuration': self.config,
            'summary': self.generate_summary()
        }
        
        report_file = f"reports/pipeline_report_{self.results['pipeline_id']}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(pipeline_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Rapport généré: {report_file}")
            return True, report_file
            
        except Exception as e:
            logger.error(f"Erreur génération rapport: {e}")
            return False, str(e)
    
    def step_start_api(self):
        """Étape 7: Démarrage de l'API (optionnel)."""
        if not self.config['api']['auto_start']:
            logger.info("⏭️  API non démarrée (auto_start=False)")
            return True, "Skipped"
        
        logger.info("🚀 Démarrage API en arrière-plan...")
        
        # Démarrer l'API en arrière-plan
        try:
            process = subprocess.Popen([
                'python3', 'api.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Attendre quelques secondes pour vérifier le démarrage
            time.sleep(3)
            
            if process.poll() is None:  # Process encore en cours
                logger.info(f"✅ API démarrée (PID: {process.pid})")
                logger.info(f"🌐 URL: http://{self.config['api']['host']}:{self.config['api']['port']}")
                
                # Sauvegarder le PID pour pouvoir arrêter l'API plus tard
                with open('api.pid', 'w') as f:
                    f.write(str(process.pid))
                
                return True, f"API started (PID: {process.pid})"
            else:
                stdout, stderr = process.communicate()
                return False, f"API failed to start: {stderr}"
                
        except Exception as e:
            return False, f"Error starting API: {e}"
    
    def generate_summary(self):
        """Génère un résumé de la pipeline."""
        total_duration = (datetime.now() - self.pipeline_start).total_seconds()
        
        successful_steps = sum(1 for step in self.results['steps'].values() 
                              if step.get('return_code') == 0 or step.get('error') is None)
        total_steps = len(self.results['steps'])
        
        # Trouver les modèles générés
        models_dir = Path("models")
        model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
        
        # Trouver les rapports d'évaluation
        eval_reports = list(models_dir.glob("evaluation_report_*.json")) if models_dir.exists() else []
        
        return {
            'duration_seconds': total_duration,
            'duration_formatted': f"{total_duration:.1f}s",
            'steps_successful': successful_steps,
            'steps_total': total_steps,
            'success_rate': f"{(successful_steps/total_steps*100):.1f}%" if total_steps > 0 else "0%",
            'models_generated': len(model_files),
            'model_files': [f.name for f in model_files],
            'evaluation_reports': len(eval_reports),
            'pipeline_status': 'success' if successful_steps == total_steps else 'partial_failure'
        }
    
    def run_full_pipeline(self):
        """Exécute la pipeline complète."""
        logger.info("🚀 Démarrage Pipeline MLOps Prédiction de Mortalité")
        logger.info(f"Pipeline ID: {self.results['pipeline_id']}")
        
        steps = [
            ("Téléchargement données", self.step_download_data),
            ("Preprocessing", self.step_preprocess_data),
            ("Entraînement modèles", self.step_train_models),
            ("Évaluation modèles", self.step_evaluate_models),
            ("Tests de validation", self.step_run_tests),
            ("Génération rapports", self.step_generate_reports),
            ("Démarrage API", self.step_start_api)
        ]
        
        try:
            for step_name, step_func in steps:
                logger.info(f"\n{'='*60}")
                logger.info(f"📋 ÉTAPE: {step_name}")
                logger.info(f"{'='*60}")
                
                success, message = step_func()
                
                if not success:
                    logger.warning(f"⚠️  {step_name} échoué mais pipeline continue: {message}")
            
            # Finaliser
            self.results['end_time'] = datetime.now().isoformat()
            self.results['status'] = 'completed'
            
            summary = self.generate_summary()
            self.results['summary'] = summary
            
            logger.info(f"\n{'='*60}")
            logger.info("🎉 PIPELINE TERMINÉE")
            logger.info(f"{'='*60}")
            logger.info(f"Durée totale: {summary['duration_formatted']}")
            logger.info(f"Étapes réussies: {summary['steps_successful']}/{summary['steps_total']}")
            logger.info(f"Modèles générés: {summary['models_generated']}")
            logger.info(f"Status: {summary['pipeline_status']}")
            
            if summary['models_generated'] > 0:
                logger.info(f"Modèles: {', '.join(summary['model_files'])}")
            
            logger.info(f"\n🔗 Pour tester l'API:")
            logger.info(f"   curl -X GET http://localhost:5000/health")
            logger.info(f"   curl -X GET http://localhost:5000/api/v1/example")
            
            return True, summary
            
        except Exception as e:
            self.results['end_time'] = datetime.now().isoformat()
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            
            logger.error(f"💥 PIPELINE ÉCHOUÉE: {e}")
            return False, str(e)

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description='Pipeline MLOps Prédiction de Mortalité')
    parser.add_argument('--config', type=str, help='Fichier de configuration JSON')
    parser.add_argument('--skip-download', action='store_true', help='Ignorer le téléchargement des données')
    parser.add_argument('--skip-tests', action='store_true', help='Ignorer les tests de validation')
    parser.add_argument('--start-api', action='store_true', help='Démarrer l\'API automatiquement')
    
    args = parser.parse_args()
    
    # Charger la configuration
    config_overrides = {}
    if args.skip_download:
        config_overrides['data'] = {'skip_download': True}
    if args.skip_tests:
        config_overrides['pipeline'] = {'run_tests': False}
    if args.start_api:
        config_overrides['api'] = {'auto_start': True}
    
    # Créer et exécuter la pipeline
    pipeline = MLOpsPipeline(args.config)
    
    # Appliquer les overrides de ligne de commande
    for section, values in config_overrides.items():
        pipeline.config.setdefault(section, {}).update(values)
    
    success, result = pipeline.run_full_pipeline()
    
    if success:
        print(f"\n✅ Pipeline réussie ! Rapport: reports/pipeline_report_{pipeline.results['pipeline_id']}.json")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline échouée: {result}")
        sys.exit(1)

if __name__ == "__main__":
    main()
