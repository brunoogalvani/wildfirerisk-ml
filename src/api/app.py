from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("./models/wildfire_model.pkl")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Wildfire Risk API",
    version="1.0.0",
    description="""
    API for wildfire risk prediction using climate
    and historical fire occurrence data.
    """
)

class PredictionInput(BaseModel):
    latitude: float
    longitude: float
    mes: int
    temperatura_c: float
    umidade_relativa_percentual: float
    precipitacao_mmdia: float
    vento_velocidade_ms: float
    numero_dias_sem_chuva: float
    risco_fogo: float

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": -10.0,
                "longitude": -55.0,
                "mes": 8,
                "temperatura_c": 41,
                "umidade_relativa_percentual": 9,
                "precipitacao_mmdia": 0,
                "vento_velocidade_ms": 8,
                "numero_dias_sem_chuva": 40,
                "risco_fogo": 0.95
            }
        }

@app.post("/wildfire-risk")
def predict(data: PredictionInput):

    input_data = pd.DataFrame([{

        "latitude": data.latitude,

        "longitude": data.longitude,

        "mes": data.mes,

        "temperatura_c": data.temperatura_c,

        "umidade_relativa_percentual":
            data.umidade_relativa_percentual,

        "precipitacao_mmdia":
            data.precipitacao_mmdia,

        "vento_velocidade_ms":
            data.vento_velocidade_ms,
        
        "numero_dias_sem_chuva":
            data.numero_dias_sem_chuva,

        "risco_fogo":
            data.risco_fogo
    }])

    prediction = model.predict(input_data)[0]

    descriptions = {
        "baixo": "Low probability of wildfire occurrence",
        "moderado": "Moderate wildfire risk",
        "alto": "High wildfire risk",
        "critico": "Critical wildfire risk"
    }

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    return {
        "risk_level": prediction,
        "description": descriptions[prediction],
        "probabilites": {
            classe: round(float(prob), 4)
            for classe, prob in zip(classes, probabilities)
        }
    }