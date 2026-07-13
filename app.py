from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

# Load model and preprocessing files
try:
    model = joblib.load("customer_churn_model.joblib")
    encoders = joblib.load("label_encoders.pkl")
    feature_order = joblib.load("feature_order.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load files: {e}")


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is Running 🚀"}


@app.post("/predict")
def predict(data: dict):
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
        raise HTTPException(status_code=500, detail=str(e))
