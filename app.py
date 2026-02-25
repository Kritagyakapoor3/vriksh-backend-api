from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load label encoder (optional)
try:
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
except:
    label_encoder = None

@app.get("/")
def home():
    return {"status": "API is running 🎉"}

@app.post("/predict")
def predict(features: list):

    data = np.array(features).reshape(1, -1)
    pred = model.predict(data)[0]

    if label_encoder:
        pred = label_encoder.inverse_transform([pred])[0]

    return {"prediction": str(pred)}
