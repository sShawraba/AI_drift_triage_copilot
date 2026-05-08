# backend/ml_system/routers/action.py
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Optional
import threading
import subprocess
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from shared.schemas import ActionRequest, PlatformResponse
from backend.ml_system.services.model_service import model_service

router = APIRouter()

# Idempotency store (in‑memory – resets on restart)
_idempotency_store: Dict[str, Dict] = {}


def is_idempotent(key: str) -> bool:
    """Prevent duplicate actions from retried queue messages."""
    if key in _idempotency_store:
        elapsed = (datetime.utcnow() - _idempotency_store[key]["timestamp"]).total_seconds()
        if elapsed < 3600:
            return True
        else:
            del _idempotency_store[key]
    return False


def mark_processed(key: str, result: Dict) -> None:
    _idempotency_store[key] = {
        "timestamp": datetime.utcnow(),
        "result": result
    }


# ------------------------------------------------------------
# Retrain – background training
# ------------------------------------------------------------
def start_retrain():
    """Run the training script in the background using the correct module path."""
    def run():
        print("🔄 Starting retrain in background...")
        subprocess.run(["uv", "run", "python", "-m", "backend.ml_system.ml.train"], check=False)
        print("✅ Retrain complete")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# ------------------------------------------------------------
# Replay – evaluate current production model
# ------------------------------------------------------------
def run_replay() -> dict:
    """Replay the test set through the current model and return metrics."""
    try:
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    except FileNotFoundError:
        return {"error": "Test files not found"}

    y_probs = []
    for _, row in X_test.iterrows():
        prob, _ = model_service.predict(pd.DataFrame([row]))
        y_probs.append(prob)

    threshold = model_service.get_threshold()
    y_preds = [1 if p >= threshold else 0 for p in y_probs]

    auc = roc_auc_score(y_test, y_probs)
    f1 = f1_score(y_test, y_preds)

    return {
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "samples": len(y_test)
    }


# ------------------------------------------------------------
# Action endpoint
# ------------------------------------------------------------
@router.post("/action", response_model=PlatformResponse)
async def handle_action(request: ActionRequest):
    """
    Agent calls this after HIL approval.
    Handles: retrain, rollback, replay
    """
    from backend.ml_system.services.registry_service import RegistryService

    # 1. Idempotency
    if is_idempotent(request.idempotency_key):
        return PlatformResponse(
            success=True,
            message="Already processed (idempotent)",
            action_id=request.action_id
        )

    registry = RegistryService()

    # 2. Rollback
    if request.action_type == "rollback":
        if not registry.model_exists(request.model_version):
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_version} no longer exists"
            )

        result = registry.promote_to_production(
            model_version=request.model_version,
            investigation_id=request.investigation_id
        )
        model_service.force_reload()

        mark_processed(request.idempotency_key, result)

        return PlatformResponse(
            success=True,
            message=f"Rollback to version {request.model_version} completed",
            action_id=request.action_id
        )

    # 3. Retrain
    elif request.action_type == "retrain":
        mark_processed(request.idempotency_key, {})
        start_retrain()          # launch training in background
        return PlatformResponse(
            success=True,
            message="Retrain job started – new model will be registered shortly",
            action_id=request.action_id
        )

    # 4. Replay
    elif request.action_type == "replay":
        mark_processed(request.idempotency_key, {})
        metrics = run_replay()
        if "error" in metrics:
            message = f"Replay failed: {metrics['error']}"
        else:
            message = f"Replay complete – AUC: {metrics['auc']}, F1: {metrics['f1']}"
        return PlatformResponse(
            success=True,
            message=message,
            action_id=request.action_id
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action_type: {request.action_type}")