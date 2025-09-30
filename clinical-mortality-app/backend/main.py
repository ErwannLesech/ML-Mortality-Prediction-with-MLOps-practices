from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Clinical Mortality Prediction API")

# CORS pour autoriser le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
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
    try:
        # Préparer les données pour Dataiku
        payload = {
            "features": {
                **patient.dict(),
                "mortality": "0"  # Champ inutile mais requis
            }
        }
        
        # Appeler l'API Dataiku
        async with httpx.AsyncClient() as client:
            response = await client.post(
                os.getenv("DATAIKU_API_URL"),
                json=payload,
                headers={
                    "Authorization": f"Bearer {os.getenv('DATAIKU_API_TOKEN')}"
                },
                timeout=30.0
            )
            
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Error calling Dataiku API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
