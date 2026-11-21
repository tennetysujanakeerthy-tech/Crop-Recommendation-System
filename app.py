from pathlib import Path

import joblib
import pandas as pd
from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ------------------------------------------------
# 2. LOAD TRAINED MODEL FILES
# ------------------------------------------------

rf_model = joblib.load(
    MODEL_DIR / "random_forest_crop_model_6features.pkl"
)

outlier_detector = joblib.load(
    MODEL_DIR / "crop_outlier_detector_6features.pkl"
)

metadata = joblib.load(
    MODEL_DIR / "crop_metadata_6features.pkl"
)

FEATURES = metadata["features"]
FEATURE_RANGES = metadata["feature_ranges"]

print("All six-feature model files loaded successfully!")

# ------------------------------------------------
# 3. CREATE FASTAPI APPLICATION
# ------------------------------------------------

app = FastAPI(
    title="Crop Recommendation API",
    description="Crop recommendation system for Agricultural Extension Officers",
    version="1.0"
)

# Allow dashboard connections during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------------------------
# 4. DEFINE INPUT DATA
# ------------------------------------------------

class FarmerInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    rainfall: float

# ------------------------------------------------
# 5. HOME ROUTE
# ------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to the Crop Recommendation API!",
        "status": "API is running"
    }

# ------------------------------------------------
# 6. HEALTH CHECK
# ------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "features": FEATURES
    }

# ------------------------------------------------
# 7. CROP PREDICTION
# ------------------------------------------------

@app.post("/predict")
def predict_crop(data: FarmerInput):

    # Arrange input features in the training order
    input_data = pd.DataFrame(
        [[
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.rainfall
        ]],
        columns=FEATURES
    )

    # Check values against training dataset ranges
    out_of_range_features = {}

    for feature in FEATURES:
        minimum = FEATURE_RANGES[feature]["min"]
        maximum = FEATURE_RANGES[feature]["max"]
        value = float(input_data.iloc[0][feature])

        if value < minimum or value > maximum:
            out_of_range_features[feature] = {
                "entered_value": value,
                "minimum": minimum,
                "maximum": maximum
            }

    # Predict crop
    predicted_crop = rf_model.predict(input_data)[0]

    # Get model probability scores
    probabilities = rf_model.predict_proba(input_data)[0]

    # Get top three crop recommendations
    top_indices = probabilities.argsort()[-3:][::-1]

    top_3 = []

    for index in top_indices:
        top_3.append({
            "crop": str(rf_model.classes_[index]),
            "model_probability": round(
                float(probabilities[index]) * 100, 2
            )
        })

    # Check whether the input looks unusual
    anomaly_result = outlier_detector.predict(input_data)[0]
    anomaly_detected = bool(anomaly_result == -1)

    # Prepare warnings
    warnings = []

    if out_of_range_features:
        warnings.append(
            "One or more values are outside the training dataset ranges."
        )

    if anomaly_detected:
        warnings.append(
            "The input combination appears unusual. Please verify the farm data."
        )

    # Return result
    return {
        "status": "success",
        "predicted_crop": str(predicted_crop),
        "model_probability": round(
            float(max(probabilities)) * 100, 2
        ),
        "top_3_recommendations": top_3,
        "out_of_range_features": out_of_range_features,
        "anomaly_detected": anomaly_detected,
        "warnings": warnings
    }
@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(BASE_DIR / "index.html")