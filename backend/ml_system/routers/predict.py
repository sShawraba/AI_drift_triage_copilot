# backend/ml_system/routers/predict.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.ml_system.schemas import PredictionRequest
from backend.ml_system.services.model_service import model_service
from backend.ml_system.services.prediction_store import prediction_store
from backend.ml_system.services.drift_service import DriftService
import pandas as pd

router = APIRouter()

# Global drift service (initialize in main.py lifespan)
drift_service: DriftService = None

def set_drift_service(service: DriftService):
    """Called from main.py to inject drift service."""
    global drift_service
    drift_service = service



@router.post("/predict")
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        # Convert request to dict
        request_dict = request.model_dump()
        
        # Create DataFrame with column names the model expects
        # This maps clean API names → model column names
        mapped_data = {
            "age": request_dict["age"],
            "job": request_dict["job"],
            "marital": request_dict["marital"],
            "education": request_dict["education"],
            "default": request_dict["default"],
            "housing": request_dict["housing"],
            "loan": request_dict["loan"],
            "contact": request_dict["contact"],
            "month": request_dict["month"],
            "day_of_week": request_dict["day_of_week"],
            "campaign": request_dict["campaign"],
            "pdays": request_dict["pdays"],
            "previous": request_dict["previous"],
            "poutcome": request_dict["poutcome"],
            "emp.var.rate": request_dict["emp_var_rate"],      # map
            "cons.price.idx": request_dict["cons_price_idx"],  # map
            "cons.conf.idx": request_dict["cons_conf_idx"],    # map
            "euribor3m": request_dict["euribor3m"],
            "nr.employed": request_dict["nr_employed"],        # map
        }
        
        # Calculate pdays_missing (same logic as training)
        mapped_data["pdays_missing"] = 1 if mapped_data["pdays"] == 999 else 0
        if mapped_data["pdays"] == 999:
            mapped_data["pdays"] = -1
        
        df = pd.DataFrame([mapped_data])
        
        # Get prediction
        probability, prediction = model_service.predict(df)
        
        # Store prediction (store original request dict, not mapped)
        prediction_store.save(
            features=request_dict,
            probability=probability,
            prediction=prediction
        )
        
                # Trigger drift check every 20 predictions
        count = prediction_store.get_count()
        print(f"📊 Prediction count: {count}")  

        if count % 20 == 0 and drift_service is not None:
            print(f"🔍 Triggering drift check at count={count}")  # ← ADD THIS
            background_tasks.add_task(check_drift_background)

        return {
            "probability": probability,
            "prediction": prediction,
            "threshold": model_service.threshold
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def check_drift_background():
    """Background task to check drift and alert agent."""
    global drift_service
    
    if drift_service is None:
        return
    
    # Get recent predictions as DataFrame
    live_df = prediction_store.get_recent_as_dataframe(limit=200)
    
    if len(live_df) < 50:
        return
    
    # Check drift and alert agent if needed
    await drift_service.check_and_alert(live_df)