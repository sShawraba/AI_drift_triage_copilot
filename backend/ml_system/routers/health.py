# backend/ml_system/routers/health.py
from fastapi import APIRouter
from backend.ml_system.services.model_service import model_service
from backend.ml_system.services.prediction_store import prediction_store

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check for docker-compose."""
    return {"status": "healthy", "service": "ml-system"}

@router.get("/ready")
async def readiness_check():
    """Readiness probe - checks if model is loaded."""
    try:
        # Check if model is loaded
        if model_service.model is None:
            return {"status": "not_ready", "reason": "model not loaded"}
        
        # Check if prediction store is accessible
        prediction_store.get_count()
        
        return {"status": "ready", "model_version": model_service.current_version}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}
    
@router.get("/status")
async def service_status():
    """Detailed status with model version."""
    from backend.ml_system.services.model_service import model_service
    
    return {
        "status": "healthy",
        "service": "ml-system",
        "model_version": model_service.get_current_version(),
        "threshold": model_service.get_threshold(),
        "predictions_stored": prediction_store.get_count()
    }

@router.get("/metrics")
async def system_metrics():
    """Metrics endpoint for dashboard - uses global drift_service from main."""
    # Import here to avoid circular import
    from backend.ml_system.main import drift_service
    
    # Get recent predictions
    recent_df = prediction_store.get_recent_as_dataframe(limit=200)
    total_predictions = prediction_store.get_count()
    recent_positive_rate = recent_df["prediction"].mean() if len(recent_df) > 0 else 0
    
    # Get drift status
    drift_status = None
    if drift_service and len(recent_df) >= 50:
        report = drift_service.compute_drift_report(recent_df)
        if report.get("enough_data"):
            drift_status = {
                "severity": drift_service.get_severity(
                    report["psi_numeric"],
                    report["chi2_categorical"],
                    report["output_drift"]
                ),
                "psi_numeric": round(report["psi_numeric"], 4),
                "chi2_categorical": round(report["chi2_categorical"], 4),
                "output_drift": round(report["output_drift"], 4),
                "last_severity": drift_service.get_last_severity(),
                "sample_size": report.get("sample_size", 0)
            }
    elif drift_service and len(recent_df) < 50:
        drift_status = {
            "severity": "insufficient_data",
            "message": f"Need 50 samples, have {len(recent_df)}",
            "sample_size": len(recent_df)
        }
    
    return {
        "model": {
            "name": "bank_marketing_model",
            "production_version": model_service.get_current_version(),
            "threshold": model_service.get_threshold()
        },
        "predictions": {
            "total_stored": total_predictions,
            "recent_positive_rate": round(recent_positive_rate, 4),
            "recent_window_size": len(recent_df)
        },
        "drift": drift_status,
        "service": {
            "status": "healthy",
            "poll_interval_seconds": 30
        }
    }