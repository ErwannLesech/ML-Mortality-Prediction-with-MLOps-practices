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

# Fallback: Use in-memory storage if MongoDB is not available
USE_MEMORY_STORAGE = os.getenv("USE_MEMORY_STORAGE", "false").lower() == "true"
in_memory_metrics = []


# Configuration MongoDB spécifique pour Render et MongoDB Atlas
def create_mongo_client():
    """Create MongoDB client with proper SSL configuration for Render deployment"""
    try:
        # Check if we're using MongoDB Atlas (contains mongodb.net)
        if "mongodb.net" in MONGO_URI:
            logger.info("Configuring MongoDB Atlas connection for Render")

            # Try different SSL configurations for MongoDB Atlas on Render
            ssl_configs = [
                # Configuration 1: Relaxed TLS for Render compatibility
                {
                    "tls": True,
                    "tlsAllowInvalidCertificates": True,
                    "tlsAllowInvalidHostnames": True,
                    "retryWrites": True,
                    "maxPoolSize": 5,
                    "serverSelectionTimeoutMS": 20000,
                    "connectTimeoutMS": 20000,
                    "socketTimeoutMS": 20000,
                },
                # Configuration 2: Basic SSL without certificate validation
                {
                    "ssl": True,
                    "retryWrites": True,
                    "maxPoolSize": 3,
                    "serverSelectionTimeoutMS": 15000,
                    "connectTimeoutMS": 15000,
                    "socketTimeoutMS": 15000,
                },
                # Configuration 3: Minimal configuration with only essential params
                {
                    "retryWrites": True,
                    "serverSelectionTimeoutMS": 10000,
                    "connectTimeoutMS": 10000,
                    "socketTimeoutMS": 10000,
                    "maxPoolSize": 1,
                },
                # Configuration 4: Force TLS 1.2 with relaxed validation
                {
                    "tls": True,
                    "tlsAllowInvalidCertificates": True,
                    "tlsInsecure": True,
                    "retryWrites": True,
                    "serverSelectionTimeoutMS": 25000,
                },
            ]

            for i, config in enumerate(ssl_configs):
                try:
                    logger.info(f"Trying MongoDB connection configuration {i+1}")
                    client = MongoClient(MONGO_URI, **config)
                    # Test the connection
                    client.admin.command("ping")
                    logger.info(
                        f"MongoDB connection successful with configuration {i+1}"
                    )
                    return client, True
                except Exception as e:
                    logger.warning(f"Configuration {i+1} failed: {e}")
                    try:
                        if 'client' in locals():
                            client.close()
                    except:
                        pass
                    continue
            
            # Try alternative URI configurations if all above failed
            logger.info("Trying alternative MongoDB URI configurations...")
            alternative_uris = []
            
            # Extract base URI components
            if "?" in MONGO_URI:
                base_uri = MONGO_URI.split("?")[0]
                alternative_uris = [
                    f"{base_uri}?retryWrites=true&w=majority&ssl=true&tlsAllowInvalidCertificates=true",
                    f"{base_uri}?retryWrites=true&w=majority&tls=true&tlsInsecure=true",
                    f"{base_uri}?retryWrites=true&w=majority",
                    f"{base_uri}?ssl=false&retryWrites=false",
                ]
            
            for j, alt_uri in enumerate(alternative_uris):
                try:
                    logger.info(f"Trying alternative URI configuration {j+1}")
                    client = MongoClient(
                        alt_uri,
                        serverSelectionTimeoutMS=10000,
                        connectTimeoutMS=10000,
                        socketTimeoutMS=10000,
                        maxPoolSize=1
                    )
                    client.admin.command("ping")
                    logger.info(f"MongoDB connection successful with alternative URI {j+1}")
                    return client, True
                except Exception as e:
                    logger.warning(f"Alternative URI {j+1} failed: {e}")
                    try:
                        if 'client' in locals():
                            client.close()
                    except:
                        pass
                    continue

            # If all configurations failed, raise the last exception
            raise Exception("All MongoDB SSL configurations and alternative URIs failed")

        else:
            # Configuration locale pour développement
            client = MongoClient(MONGO_URI)
            client.admin.command("ping")
            logger.info("MongoDB connection established successfully")
            return client, True

    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}. Metrics will not be stored.")
        return None, False


# Initialize MongoDB connection
client, mongo_available = create_mongo_client()
if mongo_available and client:
    db = client["metricsdb"]
    metrics_collection = db["metrics"]
else:
    client = None
    db = None
    metrics_collection = None


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
            elif USE_MEMORY_STORAGE:
                # Store in memory as fallback
                metric_dict = metric.dict()
                in_memory_metrics.append(metric_dict)
                # Keep only last 1000 metrics to prevent memory overflow
                if len(in_memory_metrics) > 1000:
                    in_memory_metrics.pop(0)
                logger.info(f"Metric stored in memory: {metric_dict}")
            else:
                logger.info(f"Metric (not stored): {metric.dict()}")
        except Exception as e:
            logger.error(f"Failed to log metric: {str(e)}")


@app.post("/metrics")
def create_metric(metric: Metric):
    logger.info(f"Creating metric: {metric}")
    if mongo_available and metrics_collection is not None:
        try:
            metrics_collection.insert_one(metric.dict())
            logger.info("Metric saved successfully")
            return {"message": "Metric saved"}
        except Exception as e:
            logger.error(f"Failed to save metric: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    elif USE_MEMORY_STORAGE:
        try:
            metric_dict = metric.dict()
            in_memory_metrics.append(metric_dict)
            # Keep only last 1000 metrics to prevent memory overflow
            if len(in_memory_metrics) > 1000:
                in_memory_metrics.pop(0)
            logger.info("Metric saved to memory successfully")
            return {"message": "Metric saved to memory"}
        except Exception as e:
            logger.error(f"Failed to save metric to memory: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        logger.warning("Database not available for metrics creation")
        raise HTTPException(status_code=503, detail="Database not available")


@app.get("/metrics")
def get_metrics():
    logger.info("Getting metrics from database")
    if mongo_available and metrics_collection is not None:
        try:
            docs = metrics_collection.find().sort("timestamp", -1)
            result = []
            for doc in docs:
                if "_id" in doc:
                    del doc["_id"]  # Remove the _id field
                result.append(doc)
            logger.info(f"Retrieved {len(result)} metrics from database")
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    elif USE_MEMORY_STORAGE:
        try:
            # Return metrics sorted by timestamp (newest first)
            sorted_metrics = sorted(in_memory_metrics, key=lambda x: x.get('timestamp', 0), reverse=True)
            logger.info(f"Retrieved {len(sorted_metrics)} metrics from memory")
            return sorted_metrics
        except Exception as e:
            logger.error(f"Failed to retrieve metrics from memory: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        logger.warning("Database not available for metrics retrieval")
        return []  # Return empty list if database not available
