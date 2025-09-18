#!/usr/bin/env python3
"""
Script de test pour l'API de prédiction de mortalité.
Teste tous les endpoints avec différents scénarios.
"""

import requests
import json
import time
import sys
from datetime import datetime

API_BASE_URL = "http://localhost:5000"

def test_endpoint(method, endpoint, data=None, expected_status=200):
    """Teste un endpoint de l'API."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        print(f"\n🔄 Test {method} {endpoint}")
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        else:
            print(f"❌ Méthode {method} non supportée")
            return False
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == expected_status:
            print("✅ Status OK")
            
            # Afficher la réponse
            if response.headers.get('content-type', '').startswith('application/json'):
                json_response = response.json()
                print(f"Réponse: {json.dumps(json_response, indent=2, ensure_ascii=False)[:300]}...")
            else:
                print(f"Réponse text: {response.text[:200]}...")
            
            return True, json_response if 'json_response' in locals() else response.text
        else:
            print(f"❌ Status inattendu. Attendu: {expected_status}, Reçu: {response.status_code}")
            print(f"Réponse: {response.text}")
            return False, response.text
            
    except requests.exceptions.ConnectionError:
        print("❌ Connexion échouée - API non démarrée?")
        return False, "Connection failed"
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False, str(e)

def wait_for_api(max_wait=30):
    """Attend que l'API soit disponible."""
    print(f"⏳ Attente de l'API (max {max_wait}s)...")
    
    for i in range(max_wait):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API disponible")
                return True
        except:
            pass
        
        time.sleep(1)
        print(f"   ... {i+1}s")
    
    print("❌ Timeout - API non disponible")
    return False

def main():
    """Tests complets de l'API."""
    print("🧪 Tests API Prédiction de Mortalité")
    print("=" * 50)
    
    # Vérifier que l'API est disponible
    if not wait_for_api():
        print("\n💡 Pour démarrer l'API: python3 api.py")
        sys.exit(1)
    
    # Tests des endpoints
    tests = [
        # Test page d'accueil
        ("GET", "/", None, 200),
        
        # Test health check
        ("GET", "/health", None, 200),
        
        # Test exemple patient
        ("GET", "/api/v1/example", None, 200),
        
        # Test info modèle
        ("GET", "/api/v1/model/info", None, 200),
    ]
    
    # Exécuter les tests GET
    results = []
    for method, endpoint, data, expected_status in tests:
        success, response = test_endpoint(method, endpoint, data, expected_status)
        results.append((f"{method} {endpoint}", success))
    
    # Test prédiction simple
    print(f"\n🔄 Test prédiction patient simple")
    patient_data = {
        "age": 65,
        "sex_encoded": 1,
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "heart_rate": 85,
        "temperature": 37.2,
        "glucose": 180,
        "bmi": 28.5,
        "creatinine": 1.8
    }
    
    success, response = test_endpoint("POST", "/api/v1/predict", patient_data, 200)
    results.append(("POST /api/v1/predict (simple)", success))
    
    if success:
        prediction = response['prediction']
        print(f"📊 Résultat: {prediction['mortality_risk']} Risk")
        if 'probability_death' in prediction:
            print(f"   Probabilité décès: {prediction['probability_death']:.1%}")
    
    # Test prédiction batch
    print(f"\n🔄 Test prédiction batch")
    batch_data = {
        "patients": [
            {"age": 45, "sex_encoded": 0, "systolic_bp": 120, "diastolic_bp": 80, "glucose": 90},
            {"age": 75, "sex_encoded": 1, "systolic_bp": 160, "diastolic_bp": 95, "glucose": 200}
        ]
    }
    
    success, response = test_endpoint("POST", "/api/v1/predict/batch", batch_data, 200)
    results.append(("POST /api/v1/predict/batch", success))
    
    if success:
        print(f"📊 {len(response['predictions'])} prédictions reçues")
        for i, pred in enumerate(response['predictions']):
            print(f"   Patient {i+1}: {pred['mortality_risk']} Risk")
    
    # Test cas d'erreur
    print(f"\n🔄 Test validation erreur")
    invalid_data = {"age": -5, "sex_encoded": 3}  # Données invalides
    success, response = test_endpoint("POST", "/api/v1/predict", invalid_data, 400)
    results.append(("POST /api/v1/predict (validation)", success))
    
    # Test endpoint inexistant
    print(f"\n🔄 Test endpoint inexistant")
    success, response = test_endpoint("GET", "/nonexistent", None, 404)
    results.append(("GET /nonexistent (404)", success))
    
    # Résultats finaux
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS DES TESTS")
    print("=" * 50)
    
    successful_tests = 0
    total_tests = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            successful_tests += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Score: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    if successful_tests == total_tests:
        print("🎉 Tous les tests passent !")
        return 0
    else:
        print("⚠️  Certains tests ont échoué")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrompus")
        sys.exit(1)
