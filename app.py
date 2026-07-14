from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
encoders = None
feature_order = None

def load_model_files():
    """Load model and preprocessing files"""
    global model, encoders, feature_order
    
    try:
        logger.info("Loading customer_churn_model.joblib...")
        model = joblib.load("customer_churn_model.joblib")
        logger.info("✅ Model loaded")
        
        logger.info("Loading label_encoders.pkl...")
        encoders = joblib.load("label_encoders.pkl")
        logger.info(f"✅ Encoders loaded ({len(encoders)} encoders)")
        
        logger.info("Loading feature_order.pkl...")
        feature_order = joblib.load("feature_order.pkl")
        logger.info(f"✅ Feature order loaded ({len(feature_order)} features)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load files: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

# Try to load model at startup
if load_model_files():
    logger.info("🟢 All model files loaded successfully")
else:
    logger.warning("🔴 Failed to load model files - API will not work until files are available")

@app.get("/")
def home():
    """Health check endpoint"""
    if model is not None:
        return {"message": "Customer Churn Prediction API is Running 🚀", "status": "ready"}
    else:
        return {"message": "Customer Churn Prediction API", "status": "model_not_loaded"}

@app.get("/health")
def health_check():
    """Render health check"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    """Predict customer churn"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical columns
        for col, encoder in encoders.items():
            if col in df.columns:
                value = str(df.loc[0, col])
                
                if value not in encoder.classes_:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value '{value}' for '{col}'"
                    )
                
                df[col] = encoder.transform([value])
        
        # Check for missing columns
        missing = [c for c in feature_order if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )
        
        # Arrange columns exactly like training
        df = df[feature_order]
        
        # Predict
        prediction = model.predict(df)[0]
        
        return {
            "prediction": int(prediction),
            "result": "Churn" if prediction == 1 else "No Churn"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
