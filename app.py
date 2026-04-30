"""
FASTAPI APPLICATION FOR HOUSE PRICE PREDICTION
Exposes REST API for predictions - FIXED VERSION (No Warnings)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import uvicorn
import warnings
warnings.filterwarnings('ignore')  # Suppress benign warnings

# Load model - with error handling
model_package = None
try:
    if os.path.exists('models/best_house_price_model.pkl'):
        model_package = joblib.load('models/best_house_price_model.pkl')
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {model_package.get('model_type', 'Unknown')}")
    else:
        print("⚠️ Model file not found. Please run 'python main.py' first to train the model.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Create FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using machine learning",
    version="1.0.0"
)

# Add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request schema
class HouseFeatures(BaseModel):
    area_sqft: float
    bedrooms: int
    bathrooms: float
    age_years: int
    condition_score: int
    garage_cars: int
    parking_spots: int
    floor_number: int
    
    class Config:
        json_schema_extra = {  # ✅ FIXED: schema_extra → json_schema_extra
            "example": {
                "area_sqft": 2000,
                "bedrooms": 3,
                "bathrooms": 2,
                "age_years": 10,
                "condition_score": 4,
                "garage_cars": 2,
                "parking_spots": 1,
                "floor_number": 2
            }
        }

# Helper function to compute derived features
def compute_derived_features(data: HouseFeatures):
    """Compute derived features from input"""
    sqft_per_room = data.area_sqft / (data.bedrooms + 0.5)
    luxury_index = (data.bathrooms * 2 + data.condition_score) / 3
    age_squared = data.age_years ** 2
    needs_renovation = 1 if data.age_years > 30 else 0
    
    # Default values if model expects these features
    location_score = 3.0  # Default medium location
    furnishing_score = 1.0  # Default semi-furnished
    
    return [
        data.area_sqft,
        data.bedrooms,
        data.bathrooms,
        data.age_years,
        data.condition_score,
        data.garage_cars,
        data.parking_spots,
        data.floor_number,
        sqft_per_room,
        luxury_index,
        age_squared,
        needs_renovation,
        location_score,
        furnishing_score
    ]

@app.get("/")
def root():
    return {
        "message": "🏠 House Price Prediction API",
        "status": "running",
        "model_loaded": model_package is not None,
        "endpoints": {
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation",
            "/redoc": "GET - Alternative API documentation",
            "/predict": "POST - Predict house price"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_package is not None,
        "model_type": model_package.get('model_type', 'Unknown') if model_package else None
    }

@app.post("/predict")
def predict_price(features: HouseFeatures):
    """
    Predict house price based on input features
    """
    if model_package is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please run 'python main.py' first to train and save the model."
        )
    
    try:
        # Compute derived features
        input_features_list = compute_derived_features(features)
        
        # ✅ FIXED: Create DataFrame with feature names (fixes sklearn warning)
        if 'feature_names' in model_package:
            feature_names = model_package['feature_names']
            input_df = pd.DataFrame([input_features_list], columns=feature_names)
        else:
            # Fallback if feature_names not in model_package
            input_df = np.array(input_features_list).reshape(1, -1)
        
        # Scale features (handle both DataFrame and numpy array)
        if hasattr(input_df, 'columns'):
            input_scaled = model_package['scaler'].transform(input_df)
        else:
            input_scaled = model_package['scaler'].transform(input_df)
        
        # Predict
        prediction = model_package['model'].predict(input_scaled)[0]
        
        return {
            "predicted_price": round(prediction, 2),
            "predicted_price_formatted": f"${round(prediction, 2):,.0f}",
            "predicted_price_inr": f"₹{round(prediction * 83, 2):,.0f}",  # Approx USD to INR
            # ✅ FIXED: dict() → model_dump() (Pydantic V2 migration)
            "input_features": features.model_dump()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the app - FIXED VERSION
if __name__ == "__main__":
    print("🚀 Starting House Price Prediction API...")
    print("📍 API will be available at: http://127.0.0.1:8000")
    print("📚 API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
    )