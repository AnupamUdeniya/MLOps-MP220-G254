from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    panel_area_m2: float
    sunshine_hours: float
    temp_c: float = Field(ge=6)
    dust_days: float = Field(le=50)
    panel_age_yr: float
    cloud_cover_pct: float
    roof_type: str
    area: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="MP220 Solar Energy API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput, request: Request) -> dict[str, float]:
    features = pd.DataFrame([payload.model_dump()])
    energy_kwh = float(request.app.state.model.predict(features)[0])
    return {"prediction": max(0.0, energy_kwh)}
