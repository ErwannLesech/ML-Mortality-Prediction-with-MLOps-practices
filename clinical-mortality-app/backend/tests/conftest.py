"""Test configuration and fixtures for the API tests."""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing."""
    return {
        "age": 65,
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
