from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# -----------------------------
# Load ML Model
# -----------------------------
# Make sure your file is named: model.pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# -----------------------------
# Input Schema
# -----------------------------
class Features(BaseModel):
    data: list


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"status": "API is running 🎉", "predict_url": "/predict"}


@app.post("/predict")
def predict(features: Features):
    try:
        # Convert list to numpy array
        arr = np.array(features.data).reshape(1, -1)

        # Run prediction
        pred = model.predict(arr)[0]

        return {"prediction": str(pred)}

    except Exception as e:
        return {"error": str(e)}
