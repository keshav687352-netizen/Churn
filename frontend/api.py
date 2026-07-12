from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    Age: int = Field(ge=0, le=100)
    Tenure: int = Field(ge=0, le=120)
    usage_frequency: int = Field(alias="Usage Frequency", ge=0, le=100)
    support_calls: int = Field(alias="Support Calls", ge=0, le=1000)
    payment_delay: int = Field(alias="Payment Delay", ge=0, le=365)
    contract_length: str = Field(alias="Contract Length")
    total_spend: int = Field(alias="Total Spend", ge=0, le=1000000)
    last_interaction: int = Field(alias="Last Interaction", ge=0, le=365)

    @field_validator("contract_length")
    @classmethod
    def validate_contract_length(cls, value: str) -> str:
        normalized = value.strip().title()
        allowed = {"Monthly", "Quarterly", "Annual"}
        if normalized not in allowed:
            raise ValueError("Contract Length must be one of: Monthly, Quarterly, Annual")
        return normalized


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_customer(payload: CustomerInput):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Prediction failed"})

    try:
        input_df = pd.DataFrame([payload.model_dump(by_alias=True)])

        input_df["Contract Length"] = pd.Categorical(
            input_df["Contract Length"],
            categories=["Monthly", "Quarterly", "Annual"],
        )

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        stay_probability = round(float(probabilities[0]) * 100, 2)
        churn_probability = round(float(probabilities[1]) * 100, 2)
        confidence = round(max(stay_probability, churn_probability), 2)
        prediction_label = "Stay" if prediction == 0 else "Churn"

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "stay_probability": stay_probability,
            "churn_probability": churn_probability,
        }
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Prediction failed"})


@app.exception_handler(Exception)
async def handle_unexpected_errors(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Prediction failed"})