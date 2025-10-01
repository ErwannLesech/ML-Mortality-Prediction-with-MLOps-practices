import logging
import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

load_dotenv()

app = FastAPI(title="Clinical Mortality Prediction API")

# CORS pour autoriser le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Frontend Docker container
        "http://frontend:80",  # Internal Docker network
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://0.0.0.0:3000",  # Alternative localhost
        "https://ml-mortality-prediction-frontend.onrender.com",  # Production frontend
        "https://ml-mortality-prediction-with-mlops.onrender.com",  # Production backend
    ],
    allow_credentials=True,
    allow_methods=["*", "OPTIONS"],
    allow_headers=["*"],
)


class PatientData(BaseModel):
    age: int
    sex: str
    bmi: float
    systolic_bp: int
    diastolic_bp: int
    glucose: float
    cholesterol: float
    creatinine: float
    diabetes: int
    hypertension: int
    diagnosis: str
    readmission_30d: int


# In-memory metrics store (replaces MongoDB for demo/dev)
metrics_store = []


class Metric(BaseModel):
    status: str
    latency: float
    timestamp: datetime = datetime.utcnow()


@app.get("/")
async def root():
    return {"message": "Clinical Mortality Prediction API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict_mortality(patient: PatientData):
    """
    Proxy vers l'API Dataiku pour prédire la mortalité
    """
    logging.info(f"Received prediction request for patient: {patient}")
    start_time = datetime.utcnow().timestamp()
    status = "success"
    latency = None
    try:
        # Préparer les données pour Dataiku
        payload = {
            "features": {
                **patient.dict(),
                "mortality": "0",  # Champ inutile mais requis
            }
        }

        # Appeler l'API Dataiku
        async with httpx.AsyncClient() as client:
            response = await client.post(
                os.getenv("DATAIKU_API_URL"),
                json=payload,
                headers={"Authorization": f"Bearer {os.getenv('DATAIKU_API_TOKEN')}"},
                timeout=30.0,
            )

            response.raise_for_status()
            return response.json()

    except httpx.HTTPError as e:
        status = "API Error"

        message = Mail(
            from_email=os.getenv('SENDER_EMAIL'),
            to_emails=os.getenv('SENDER_EMAIL'),
            subject='Dataiku API Error Alert',
            html_content='<strong>The Dataiku API is not responding. Please check the service.</strong>')
        try:
            sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
            #sg.set_sendgrid_data_residency("eu")

            response = sg.send(message)
            print(response.status_code)
            print(response.body)
            print(response.headers)
        except Exception as e2:
            print(os.getenv('SENDGRID_API_KEY'))
            print(str(e2))
        raise HTTPException(
            status_code=500, detail=f"Error calling Dataiku API: {str(e)}"
        )
    except Exception as e:
        status = "Internal Server Error"
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        latency = datetime.utcnow().timestamp() - start_time
        metric = Metric(
            status=status, latency=latency, timestamp=datetime.utcnow().timestamp()
        )
        try:
            metrics_store.append(metric.dict())
        except Exception as e:
            print(f"Failed to log metric: {str(e)}")


@app.post("/metrics")
def create_metric(metric: Metric):
    try:
        metrics_store.append(metric.dict())
        return {"message": "Metric saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    try:
        # Return metrics in reverse chronological order (latest first)
        metrics_store.sort(key=lambda x: x['timestamp'], reverse=True)
        return metrics_store
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
