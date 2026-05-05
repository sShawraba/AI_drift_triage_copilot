"""Shared contract between model_service and agent_system. Version 1.0.0."""
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class DriftEvent(BaseModel):
    """Payload sent from Model Service to Agent when drift severity changes."""
    event_id: str
    timestamp: datetime
    model_version: str
    psi_numeric: float
    chi2_categorical: float
    output_drift: float
    severity: Literal["low", "medium", "high"]


class ActionRequest(BaseModel):
    """Agent -> Model Service: request to execute an approved action."""
    action_id: str
    investigation_id: str
    event_id: str
    action_type: Literal["retrain", "rollback", "replay"]
    model_version: str
    idempotency_key: str
    config: Optional[dict] = {}


class PlatformResponse(BaseModel):
    """Standard response from Model Service after an action request."""
    success: bool
    message: str
    action_id: Optional[str] = None