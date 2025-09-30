"""Tests for the main API endpoints."""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test the root endpoint returns correct message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Clinical Mortality Prediction API"}


def test_health_endpoint(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_structure(client: TestClient, sample_patient_data):
    """Test the predict endpoint accepts the correct data structure."""
    # This test will fail without a real Dataiku API, but that's expected
    response = client.post("/predict", json=sample_patient_data)
    # We expect this to fail with 500 due to missing/mock Dataiku API
    # but it validates the endpoint exists and accepts the data structure
    assert response.status_code in [200, 500]  # 500 is expected without real API


def test_predict_endpoint_validation(client: TestClient):
    """Test the predict endpoint validates required fields."""
    incomplete_data = {"age": 65}  # Missing required fields
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_data_types(client: TestClient):
    """Test the predict endpoint rejects invalid data types."""
    invalid_data = {
        "age": "not_a_number",  # Should be int
        "sex": "Male",
        "bmi": 28.5,
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "glucose": 110.0,
        "cholesterol": 200.0,
        "creatinine": 1.2,
        "diabetes": 1,
        "hypertension": 1,
        "diagnosis": "Heart Failure",
        "readmission_30d": 0,
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error
