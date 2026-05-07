# backend/ml_system/routers/promote.py
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Optional

from shared.schemas import ActionRequest, PlatformResponse

router = APIRouter()

# Idempotency store
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
    
    # 2. Handle different action types
    if request.action_type == "rollback":
        # Rollback to previous model version
        if not registry.model_exists(request.model_version):
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_version} no longer exists"
            )
        
        result = registry.promote_to_production(
            model_version=request.model_version,
            investigation_id=request.investigation_id
        )
        
        mark_processed(request.idempotency_key, result)
        
        return PlatformResponse(
            success=True,
            message=f"Rollback to version {request.model_version} completed",
            action_id=request.action_id
        )
    
    elif request.action_type == "retrain":
        # Agent will trigger retrain via queue
        # Return accepted, actual retrain happens async
        mark_processed(request.idempotency_key, {})
        return PlatformResponse(
            success=True,
            message="Retrain job queued",
            action_id=request.action_id
        )
    
    elif request.action_type == "replay":
        # Replay test set through model
        mark_processed(request.idempotency_key, {})
        return PlatformResponse(
            success=True,
            message="Replay job queued",
            action_id=request.action_id
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action_type: {request.action_type}")