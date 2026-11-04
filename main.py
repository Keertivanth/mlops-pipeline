from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Model API")

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "iris_model.joblib"

saved_data = joblib.load(model_path)

model = saved_data["model"]
target_names = saved_data["target_names"]

# Request data validation
class IrisData(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# Home endpoint
@app.get("/")
def home():
    return {"message": "Iris Model API is running"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
def predict_iris(data: IrisData):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "predicted_class": target_names[prediction],
        "class_id": prediction
    }