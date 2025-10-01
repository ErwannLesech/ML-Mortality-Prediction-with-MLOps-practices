import logging
import os
import sys
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

# Configuration du logging pour Render
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

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


MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
try:
    client = MongoClient(MONGO_URI)
    # Test the connection
    client.admin.command("ping")
    db = client["metricsdb"]
    metrics_collection = db["metrics"]
    mongo_available = True
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.warning(f"MongoDB connection failed: {e}. Metrics will not be stored.")
    client = None
    db = None
    metrics_collection = None
    mongo_available = False


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
    logger.info(f"Received prediction request for patient: {patient}")
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
            logger.info(f"Calling Dataiku API with payload: {payload}")
            response = await client.post(
                os.getenv("DATAIKU_API_URL"),
                json=payload,
                headers={"Authorization": f"Bearer {os.getenv('DATAIKU_API_TOKEN')}"},
                timeout=30.0,
            )

            response.raise_for_status()
            result = response.json()
            logger.info(f"Dataiku API response: {result}")
            return result

    except httpx.HTTPError as e:
        status = "API Error"
        logger.error(f"Dataiku API error: {str(e)}")

        message = Mail(
            from_email=os.getenv("SENDER_EMAIL"),
            to_emails=os.getenv("SENDER_EMAIL"),
            subject="Dataiku API Error Alert",
            html_content=(
                "<strong>The Dataiku API is not responding. "
                "Please check the service.</strong>"
            ),
        )
        try:
            sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
            # sg.set_sendgrid_data_residency("eu")

            response = sg.send(message)
            logger.info(f"Email sent successfully: {response.status_code}")
        except Exception as e2:
            logger.error(f"Failed to send email: {str(e2)}")
        raise HTTPException(
            status_code=500, detail=f"Error calling Dataiku API: {str(e)}"
        )
    except Exception as e:
        status = "Internal Server Error"
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        latency = datetime.utcnow().timestamp() - start_time
        metric = Metric(
            status=status, latency=latency, timestamp=datetime.utcnow().timestamp()
        )
        try:
            if mongo_available and metrics_collection is not None:
                metrics_collection.insert_one(metric.dict())
            else:
                logger.info(f"Metric (not stored): {metric.dict()}")
        except Exception as e:
            logger.error(f"Failed to log metric: {str(e)}")


@app.post("/metrics")
def create_metric(metric: Metric):
    logger.info(f"Creating metric: {metric}")
    if not mongo_available or metrics_collection is None:
        logger.warning("Database not available for metrics creation")
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        metrics_collection.insert_one(metric.dict())
        logger.info("Metric saved successfully")
        return {"message": "Metric saved"}
    except Exception as e:
        logger.error(f"Failed to save metric: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    logger.info("Getting metrics from database")
    if not mongo_available or metrics_collection is None:
        logger.warning("Database not available for metrics retrieval")
        return []  # Return empty list if database not available
    try:
        docs = metrics_collection.find().sort("timestamp", -1)
        result = []
        for doc in docs:
            if "_id" in doc:
                del doc["_id"]  # Remove the _id field
            result.append(doc)
        logger.info(f"Retrieved {len(result)} metrics")
        return result

    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
