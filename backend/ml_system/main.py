# backend/ml_system/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import pandas as pd
from backend.ml_system.routers import predict, action, health
from backend.ml_system.services.prediction_store import prediction_store
from backend.ml_system.services.drift_service import DriftService
from backend.ml_system.services.model_service import model_service

# Global variable for other modules to access
drift_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load reference data and initialize drift service."""
    global drift_service
    
    # Get agent base URL from environment or use default
    agent_base_url = os.getenv("AGENT_BASE_URL", "http://agent:8000")
    print(f"Agent base URL: {agent_base_url}")
    
    # Load reference data (validation set with predictions)
    try:
        ref_df = pd.read_csv("data/processed/reference_with_predictions.csv")
        print(f"✅ Loaded reference data with {len(ref_df)} samples")
    except FileNotFoundError:
        print("⚠️ Reference file not found, creating from validation set...")
        # Fallback: use validation set without predictions
        X_val = pd.read_csv("data/processed/X_val.csv")
        ref_df = X_val.copy()
        # Add predictions from your model
        _, preds = model_service.predict(pd.DataFrame(X_val))
        ref_df["prediction"] = preds
        ref_df.to_csv("data/processed/reference_with_predictions.csv", index=False)
        print(f"✅ Created reference file with {len(ref_df)} samples")
    
    # Get current model version
    current_version = model_service.get_current_version() if hasattr(model_service, 'get_current_version') else "1"
    
    # Initialize drift service
    drift_service = DriftService(
        reference_data=ref_df,
        agent_base_url=agent_base_url,
        model_version=current_version
    )
    
    # Inject into predict router
    predict.set_drift_service(drift_service)
    
    print("✅ Drift service initialized and ready")
    
    yield
    
    # Cleanup if needed
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Include all routers
app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
app.include_router(action.router, prefix="/api/v1", tags=["actions"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/")
async def root():
    return {
        "service": "ML System", 
        "status": "running", 
        "agent_base_url": os.getenv("AGENT_BASE_URL", "http://agent:8000")
    }