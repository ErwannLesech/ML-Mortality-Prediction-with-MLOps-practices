"""
Tests unitaires pour la validation des données cliniques.
Basé sur les spécifications MLOps pour la prédiction de mortalité.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataValidation:
    
    @pytest.fixture
    def sample_clinical_data(self):
        """Génère des données cliniques conformes aux spécifications."""
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'patient_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 95, n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'bmi': np.random.normal(25.0, 4.0, n_samples),
            'systolic_bp': np.random.randint(90, 180, n_samples),
            'diastolic_bp': np.random.randint(60, 120, n_samples),
            'glucose': np.random.normal(100.0, 20.0, n_samples),
            'cholesterol': np.random.normal(200.0, 30.0, n_samples),
            'creatinine': np.random.normal(1.0, 0.3, n_samples),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'readmission_30d': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'mortality': np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
        })
    
    @pytest.fixture
    def invalid_data(self, sample_clinical_data):
        """Données avec erreurs pour tester la validation."""
        data = sample_clinical_data.copy()
        
        # Ajouter des erreurs intentionnelles
        data.loc[0, 'age'] = -5  # Âge invalide
        data.loc[1, 'age'] = 150  # Âge trop élevé
        data.loc[2, 'bmi'] = 0  # BMI invalide
        data.loc[3, 'bmi'] = 500  # BMI aberrant
        data.loc[4, 'systolic_bp'] = -10  # Pression négative
        data.loc[5, 'glucose'] = np.inf  # Valeur infinie
        data.loc[6, 'cholesterol'] = np.nan  # Valeur manquante
        data.loc[7, 'diabetes'] = 2  # Valeur binaire invalide
        
        return data
    
    def test_format_validation_age(self, sample_clinical_data):
        """Test validation du format Age (int, 0-120)."""
        data = sample_clinical_data
        
        # Vérifier le type
        assert data['age'].dtype in ['int64', 'int32'], f"Age doit être int, trouvé {data['age'].dtype}"
        
        # Vérifier les limites
        assert data['age'].min() >= 0, f"Age minimum invalide: {data['age'].min()}"
        assert data['age'].max() <= 120, f"Age maximum invalide: {data['age'].max()}"
        
        # Pas de valeurs manquantes
        assert not data['age'].isna().any(), "Age ne doit pas avoir de valeurs manquantes"
    
    def test_format_validation_sex(self, sample_clinical_data):
        """Test validation du format Sex (catégorielle Male/Female)."""
        data = sample_clinical_data
        
        # Vérifier les valeurs autorisées
        valid_values = {'Male', 'Female'}
        actual_values = set(data['sex'].unique())
        assert actual_values.issubset(valid_values), f"Sex valeurs invalides: {actual_values - valid_values}"
        
        # Pas de valeurs manquantes
        assert not data['sex'].isna().any(), "Sex ne doit pas avoir de valeurs manquantes"
    
    def test_format_validation_numeric_floats(self, sample_clinical_data):
        """Test validation BMI, Glucose, Cholesterol, Creatinine (float)."""
        data = sample_clinical_data
        
        float_columns = ['bmi', 'glucose', 'cholesterol', 'creatinine']
        
        for col in float_columns:
            if col in data.columns:
                # Vérifier le type numérique
                assert pd.api.types.is_numeric_dtype(data[col]), f"{col} doit être numérique"
                
                # Vérifier les limites spécifiques
                if col == 'bmi':
                    assert data[col].min() > 0, f"BMI doit être > 0, trouvé {data[col].min()}"
                    assert data[col].max() < 100, f"BMI doit être < 100, trouvé {data[col].max()}"
                
                elif col == 'glucose':
                    assert data[col].min() > 0, f"Glucose doit être > 0"
                    assert data[col].max() < 1000, f"Glucose trop élevé: {data[col].max()}"
                
                elif col == 'cholesterol':
                    assert data[col].min() > 0, f"Cholesterol doit être > 0"
                    assert data[col].max() < 1000, f"Cholesterol trop élevé: {data[col].max()}"
                
                elif col == 'creatinine':
                    assert data[col].min() > 0, f"Creatinine doit être > 0"
                    assert data[col].max() < 20, f"Creatinine trop élevé: {data[col].max()}"
    
    def test_format_validation_blood_pressure(self, sample_clinical_data):
        """Test validation pression artérielle (positive)."""
        data = sample_clinical_data
        
        bp_columns = ['systolic_bp', 'diastolic_bp']
        
        for col in bp_columns:
            if col in data.columns:
                # Pas de valeurs négatives
                assert data[col].min() >= 0, f"{col} ne doit pas être négatif, trouvé {data[col].min()}"
                
                # Limites raisonnables
                if col == 'systolic_bp':
                    assert data[col].max() <= 300, f"Systolic BP trop élevé: {data[col].max()}"
                    assert data[col].min() >= 50, f"Systolic BP trop bas: {data[col].min()}"
                
                elif col == 'diastolic_bp':
                    assert data[col].max() <= 200, f"Diastolic BP trop élevé: {data[col].max()}"
                    assert data[col].min() >= 30, f"Diastolic BP trop bas: {data[col].min()}"
    
    def test_format_validation_binary_variables(self, sample_clinical_data):
        """Test validation variables binaires (0/1)."""
        data = sample_clinical_data
        
        binary_columns = ['diabetes', 'hypertension', 'readmission_30d', 'mortality']
        
        for col in binary_columns:
            if col in data.columns:
                # Vérifier que seules les valeurs 0 et 1 sont présentes
                unique_values = set(data[col].unique())
                assert unique_values.issubset({0, 1}), f"{col} doit être binaire (0/1), trouvé: {unique_values}"
                
                # Vérifier le type
                assert data[col].dtype in ['int64', 'int32', 'uint8'], f"{col} doit être int"
    
    def test_missing_and_infinite_values(self, sample_clinical_data):
        """Test absence de valeurs manquantes et infinies."""
        data = sample_clinical_data
        
        # Pas de NaN
        nan_columns = data.columns[data.isna().any()].tolist()
        assert len(nan_columns) == 0, f"Colonnes avec NaN: {nan_columns}"
        
        # Pas de valeurs infinies dans les colonnes numériques
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            assert not np.isinf(data[col]).any(), f"Valeurs infinies trouvées dans {col}"
    
    def test_class_balance_mortality(self, sample_clinical_data):
        """Test équilibre des classes pour Mortality."""
        data = sample_clinical_data
        
        if 'mortality' in data.columns:
            mortality_rate = data['mortality'].mean()
            
            # Vérifier que le taux n'est pas trop faible
            assert mortality_rate >= 0.02, f"Taux de mortalité trop faible: {mortality_rate:.3f} (< 2%)"
            
            # Vérifier qu'il n'est pas trop élevé non plus
            assert mortality_rate <= 0.5, f"Taux de mortalité anormalement élevé: {mortality_rate:.3f}"
            
            # Informations pour analyse
            print(f"Taux de mortalité: {mortality_rate:.3f} ({mortality_rate*100:.1f}%)")
    
    def test_statistical_consistency(self, sample_clinical_data):
        """Test cohérence statistique des distributions."""
        data = sample_clinical_data
        
        # Références attendues pour données cliniques
        expected_ranges = {
            'age': (18, 95),
            'bmi': (15, 45),
            'systolic_bp': (80, 200),
            'diastolic_bp': (50, 130),
            'glucose': (50, 400),
            'cholesterol': (100, 400),
            'creatinine': (0.5, 5.0)
        }
        
        for col, (min_exp, max_exp) in expected_ranges.items():
            if col in data.columns:
                col_mean = data[col].mean()
                col_min = data[col].min()
                col_max = data[col].max()
                
                # Vérifier que les moyennes sont dans des plages raisonnables
                if col == 'age':
                    assert 40 <= col_mean <= 80, f"Âge moyen anormal: {col_mean:.1f}"
                elif col == 'bmi':
                    assert 20 <= col_mean <= 35, f"BMI moyen anormal: {col_mean:.1f}"
                elif col == 'glucose':
                    assert 80 <= col_mean <= 150, f"Glucose moyen anormal: {col_mean:.1f}"
                
                # Vérifier les plages globales
                assert col_min >= min_exp, f"{col} min trop bas: {col_min} < {min_exp}"
                assert col_max <= max_exp, f"{col} max trop haut: {col_max} > {max_exp}"
    
    def test_data_deduplication(self, sample_clinical_data):
        """Test déduplication des patients."""
        data = sample_clinical_data
        
        if 'patient_id' in data.columns:
            # Vérifier unicité des patient_id
            duplicate_ids = data['patient_id'].duplicated().sum()
            assert duplicate_ids == 0, f"Patient IDs dupliqués: {duplicate_ids}"
        
        # Vérifier l'unicité des combinaisons critiques
        # (par exemple: âge + sexe + BMI ne devraient pas être exactement identiques)
        if all(col in data.columns for col in ['age', 'sex', 'bmi']):
            duplicate_combinations = data[['age', 'sex', 'bmi']].duplicated().sum()
            total_patients = len(data)
            duplication_rate = duplicate_combinations / total_patients
            
            # Permettre un petit taux de duplication naturelle mais pas trop
            assert duplication_rate < 0.1, f"Taux de duplication suspect: {duplication_rate:.3f} (>10%)"
    
    def test_data_validation_with_errors(self, invalid_data):
        """Test détection d'erreurs dans les données."""
        data = invalid_data
        
        # Test détection âges invalides
        invalid_ages = (data['age'] < 0) | (data['age'] > 120)
        assert invalid_ages.sum() > 0, "Doit détecter les âges invalides"
        
        # Test détection BMI invalides
        invalid_bmis = (data['bmi'] <= 0) | (data['bmi'] > 100)
        assert invalid_bmis.sum() > 0, "Doit détecter les BMI invalides"
        
        # Test détection pressions artérielles négatives
        invalid_bp = data['systolic_bp'] < 0
        assert invalid_bp.sum() > 0, "Doit détecter les pressions négatives"
        
        # Test détection valeurs infinies
        has_infinite = np.isinf(data.select_dtypes(include=[np.number])).any().any()
        assert has_infinite, "Doit détecter les valeurs infinies"
        
        # Test détection valeurs manquantes
        has_nan = data.isna().any().any()
        assert has_nan, "Doit détecter les valeurs manquantes"
    
    def test_cohorts_analysis(self, sample_clinical_data):
        """Test analyse par cohortes (sexe, âge)."""
        data = sample_clinical_data
        
        if all(col in data.columns for col in ['sex', 'mortality']):
            # Analyse par sexe
            mortality_by_sex = data.groupby('sex')['mortality'].mean()
            
            # Les taux ne doivent pas être trop différents (pas de biais extrême)
            sex_rates = mortality_by_sex.values
            if len(sex_rates) == 2:
                ratio = max(sex_rates) / min(sex_rates) if min(sex_rates) > 0 else float('inf')
                assert ratio < 5.0, f"Biais de mortalité par sexe trop important: {ratio:.2f}"
        
        if all(col in data.columns for col in ['age', 'mortality']):
            # Analyse par groupe d'âge
            data['age_group'] = pd.cut(data['age'], bins=[0, 40, 60, 80, 120], labels=['Jeune', 'Adulte', 'Senior', 'Âgé'])
            mortality_by_age = data.groupby('age_group')['mortality'].mean()
            
            # La mortalité doit généralement augmenter avec l'âge
            age_rates = mortality_by_age.values
            if len(age_rates) >= 3:
                # Vérifier tendance croissante générale (quelques exceptions tolérées)
                increasing_trend = sum(age_rates[i] <= age_rates[i+1] for i in range(len(age_rates)-1))
                trend_ratio = increasing_trend / (len(age_rates) - 1)
                assert trend_ratio >= 0.5, f"Mortalité ne suit pas la tendance d'âge attendue: {trend_ratio:.2f}"
