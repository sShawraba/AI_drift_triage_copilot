from fastapi import APIRouter
from backend.ml_system.services.drift_service import run_drift_check
from backend.ml_system.schemas import PredictionRequest
from backend.ml_system.services.model_service import model
from backend.ml_system.services.threshold_service import threshold
from backend.ml_system.services.prediction_store import save_prediction
import pandas as pd
from services.prediction_store import get_recent_predictions
from services.model_service import train_data


router = APIRouter()
model_version = "bank_marketing_model"

@router.post("/predict")
def predict(request: PredictionRequest):

    df = pd.DataFrame([request.dict()])

    # predict probability
    prob = model.predict(df)[0]

    # apply threshold
    prediction = 1 if prob >= threshold else 0

    save_prediction(
        features=request.dict(),
        probability=float(prob),
        prediction=int(prediction)
    )

    preds, _ = get_recent_predictions(limit=1000)
    count = len(preds)

    if count % 20 == 0:
        run_drift_check(train_data, model_version)

    return {
        "probability": float(prob),
        "prediction": int(prediction),
        "threshold": float(threshold)
    }

