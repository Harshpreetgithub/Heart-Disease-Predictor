"""
Heart Disease Prediction API
----------------------------
FastAPI backend for Vercel. Loads the Random Forest model trained in
train_model.py (same pipeline as Chapter 8 of the report) and exposes
POST /api/predict for the frontend in /public.
"""

import json
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- Determine base directory (handles Vercel serverless environment) ----
BASE_DIR = Path(__file__).resolve().parent

# Try multiple possible paths for model files (Vercel compatibility)
possible_paths = [
    BASE_DIR,  # Local development
    BASE_DIR.parent,  # Alternative
    Path("/tmp"),  # Vercel tmp directory
]

# ---- Load trained artifacts (produced by train_model.py) ----
model = None
scaler = None
label_encoders = None
FEATURE_COLUMNS = None

def load_model_files():
    """Load model files with error handling for Vercel environment"""
    global model, scaler, label_encoders, FEATURE_COLUMNS
    
    errors = []
    
    # Try to load model.pkl
    for base_path in possible_paths:
        model_path = base_path / "model.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                print(f"✓ Loaded model from {model_path}")
                break
            except Exception as e:
                errors.append(f"model.pkl at {model_path}: {str(e)}")
    
    if model is None:
        error_msg = f"Could not load model.pkl. Tried paths: {possible_paths}. Errors: {errors}"
        print(f"✗ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # Try to load scaler.pkl
    errors = []
    for base_path in possible_paths:
        scaler_path = base_path / "scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                print(f"✓ Loaded scaler from {scaler_path}")
                break
            except Exception as e:
                errors.append(f"scaler.pkl at {scaler_path}: {str(e)}")
    
    if scaler is None:
        error_msg = f"Could not load scaler.pkl. Tried paths: {possible_paths}. Errors: {errors}"
        print(f"✗ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # Try to load label_encoders.pkl
    errors = []
    for base_path in possible_paths:
        encoders_path = base_path / "label_encoders.pkl"
        if encoders_path.exists():
            try:
                with open(encoders_path, "rb") as f:
                    label_encoders = pickle.load(f)
                print(f"✓ Loaded label_encoders from {encoders_path}")
                break
            except Exception as e:
                errors.append(f"label_encoders.pkl at {encoders_path}: {str(e)}")
    
    if label_encoders is None:
        error_msg = f"Could not load label_encoders.pkl. Tried paths: {possible_paths}. Errors: {errors}"
        print(f"✗ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # Try to load feature_columns.json
    errors = []
    for base_path in possible_paths:
        features_path = base_path / "feature_columns.json"
        if features_path.exists():
            try:
                with open(features_path) as f:
                    FEATURE_COLUMNS = json.load(f)
                print(f"✓ Loaded feature_columns from {features_path}")
                break
            except Exception as e:
                errors.append(f"feature_columns.json at {features_path}: {str(e)}")
    
    if FEATURE_COLUMNS is None:
        error_msg = f"Could not load feature_columns.json. Tried paths: {possible_paths}. Errors: {errors}"
        print(f"✗ {error_msg}")
        raise FileNotFoundError(error_msg)

# Load model files on startup
try:
    load_model_files()
except Exception as e:
    print(f"WARNING: Failed to load model files: {e}")
    # Continue anyway - will fail on first prediction with better error message

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
    """Build feature row in the exact format the model expects"""
    if label_encoders is None or FEATURE_COLUMNS is None:
        raise RuntimeError("Model files not loaded. Please check server logs.")
    
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
    """Health check endpoint"""
    status = "ok" if all([model, scaler, label_encoders, FEATURE_COLUMNS]) else "error"
    return {"status": status, "model_loaded": model is not None}


@app.post("/api/predict")
def predict(patient: PatientInput):
    """Make heart disease prediction based on patient data"""
    if model is None or scaler is None or label_encoders is None or FEATURE_COLUMNS is None:
        raise HTTPException(
            status_code=500, 
            detail="Model files not loaded. Server not ready. Check /api/health for status."
        )
    
    try:
        X = build_feature_row(patient)
        X_scaled = scaler.transform(X)
        pred = int(model.predict(X_scaled)[0])
        proba = float(model.predict_proba(X_scaled)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    return {
        "prediction": pred,
        "label": "Likely Heart Disease" if pred == 1 else "Likely No Heart Disease",
        "probability": round(proba, 4),
        "risk_percent": round(proba * 100, 1),
    }
