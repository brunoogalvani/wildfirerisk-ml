from pathlib import Path
from src.utils.climate_rules import estacao_seca
from src.utils.climate_rules import name_to_uf

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
    uf: str
    latitude: float
    longitude: float
    mes: int
    temperatura_c: float
    umidade_relativa_percentual: float
    precipitacao_mmdia: float
    vento_velocidade_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "uf": "MATO GROSSO",
                "latitude": -10.0,
                "longitude": -55.0,
                "mes": 8,
                "temperatura_c": 41,
                "umidade_relativa_percentual": 9,
                "precipitacao_mmdia": 0,
                "vento_velocidade_ms": 8
            }
        }

@app.post("/wildfire-risk")
def predict(data: PredictionInput):

    if data.mes < 1 or data.mes > 12:
        return {
            "error": "Mês inválido"
        }

    uf = name_to_uf(data.uf)

    if not uf:
        return {
            "error": "UF inválida"
        }

    estacao = estacao_seca(
        uf,
        data.mes
    )

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

        "estacao_seca": estacao,
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
        "probabilities": {
            classe: round(float(prob), 4)
            for classe, prob in zip(classes, probabilities)
        }
    }