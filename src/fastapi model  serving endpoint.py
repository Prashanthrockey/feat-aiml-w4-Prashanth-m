import time
import threading
import requests
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. train and serialize the model
dataset = load_iris()
features = dataset.data
targets = dataset.target
targetLabels = list(dataset.target_names)

model = RandomForestClassifier(n_estimators=30, random_state=42)
model.fit(features, targets)

modelFileName = "irisModel.joblib"
joblib.dump(model, modelFileName)
print("Step 1: Model trained and saved successfully")

# 2. define fastapi application and schema
class IrisFeatures(BaseModel):
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float

app = FastAPI(title="Iris Serving API")
loadedModel = joblib.load(modelFileName)

@app.get("/")
def home():
    return {"status": "online", "message": "ML Model FastAPI is running"}

@app.post("/predict")
def predict(data: IrisFeatures):
    try:
        sample = np.array([[
            data.sepalLength,
            data.sepalWidth,
            data.petalLength,
            data.petalWidth
        ]])
        predictionId = int(loadedModel.predict(sample)[0])
        probabilities = loadedModel.predict_proba(sample)[0]
        return {
            "predictionClass": targetLabels[predictionId],
            "classId": predictionId,
            "confidence": round(float(probabilities[predictionId]), 4)
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

# 3. start server in background thread and test endpoint
def runServer():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

serverThread = threading.Thread(target=runServer, daemon=True)
serverThread.start()

print("Step 2: Server running in background. Waiting for startup...")
time.sleep(2)

print("Step 3: Sending test request to http://127.0.0.1:8000/predict")
samplePayload = {
    "sepalLength": 5.1,
    "sepalWidth": 3.5,
    "petalLength": 1.4,
    "petalWidth": 0.2
}

response = requests.post("http://127.0.0.1:8000/predict", json=samplePayload)
print("Status Code:", response.status_code)
print("API Response:", response.json())