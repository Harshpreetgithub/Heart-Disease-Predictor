"""
Heart Disease Prediction API
----------------------------
FastAPI backend for Vercel. Loads the Random Forest model trained in
train_model.py (same pipeline as Chapter 8 of the report) and exposes
POST /api/predict for the frontend in /public.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

# ---- Load trained artifacts (produced by train_model.py) ----
with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(BASE_DIR / "label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open(BASE_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: str = Field(..., pattern="^(M|F)$")
    chestPainType: str = Field(..., pattern="^(TA|ATA|NAP|ASY)$")
    restingBP: int = Field(..., ge=0, le=300)
    cholesterol: int = Field(..., ge=0, le=700)
    fastingBS: int = Field(..., ge=0, le=1)
    restingECG: str = Field(..., pattern="^(Normal|ST|LVH)$")
    maxHR: int = Field(..., ge=40, le=250)
    exerciseAngina: str = Field(..., pattern="^(Y|N)$")
    oldpeak: float = Field(..., ge=-5, le=10)
    stSlope: str = Field(..., pattern="^(Up|Flat|Down)$")


def build_feature_row(p: PatientInput) -> pd.DataFrame:
    row = {
        "Age": p.age,
        "Sex": label_encoders["Sex"].transform([p.sex])[0],
        "RestingBP": p.restingBP,
        "Cholesterol": p.cholesterol,
        "FastingBS": p.fastingBS,
        "MaxHR": p.maxHR,
        "ExerciseAngina": label_encoders["ExerciseAngina"].transform([p.exerciseAngina])[0],
        "Oldpeak": p.oldpeak,
        "ChestPainType_ATA": 1 if p.chestPainType == "ATA" else 0,
        "ChestPainType_NAP": 1 if p.chestPainType == "NAP" else 0,
        "ChestPainType_TA": 1 if p.chestPainType == "TA" else 0,
        "RestingECG_Normal": 1 if p.restingECG == "Normal" else 0,
        "RestingECG_ST": 1 if p.restingECG == "ST" else 0,
        "ST_Slope_Flat": 1 if p.stSlope == "Flat" else 0,
        "ST_Slope_Up": 1 if p.stSlope == "Up" else 0,
    }
    # Enforce the exact column order the scaler/model were fit on
    return pd.DataFrame([[row[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(patient: PatientInput):
    try:
        X = build_feature_row(patient)
        X_scaled = scaler.transform(X)
        pred = int(model.predict(X_scaled)[0])
        proba = float(model.predict_proba(X_scaled)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "prediction": pred,
        "label": "Likely Heart Disease" if pred == 1 else "Likely No Heart Disease",
        "probability": round(proba, 4),
        "risk_percent": round(proba * 100, 1),
    }
