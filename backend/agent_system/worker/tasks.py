"""Worker tasks that call the model service action endpoint."""
import os
import httpx
from shared.schemas import ActionRequest, PlatformResponse

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model_service:8000")


def _call_model_service(
    action_type: str,
    investigation_id: str,
    model_version: str,
    idempotency_key: str,
    config: dict,
) -> PlatformResponse:
    """Send an ActionRequest to the model service and return the response."""
    event_id = config.get("event_id", "")
    req = ActionRequest(
        action_id=idempotency_key,
        investigation_id=investigation_id,
        event_id=event_id,
        action_type=action_type,
        model_version=model_version,
        idempotency_key=idempotency_key,
        config=config,
    )

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{MODEL_SERVICE_URL}/api/v1/action",
            json=req.model_dump(),
        )
        resp.raise_for_status()
        return PlatformResponse(**resp.json())


def retrain(investigation_id: str, model_version: str, **kwargs):
    print(f"⏳ Requesting retrain for model v{model_version} ...")
    result = _call_model_service(
        "retrain",
        investigation_id,
        model_version,
        kwargs["idempotency_key"],
        kwargs.get("config", {}),
    )
    if result.success:
        print(f"✅ Retrain accepted: {result.message}")
    else:
        raise Exception(f"Retrain rejected: {result.message}")


def rollback(investigation_id: str, model_version: str, **kwargs):
    print(f"⏳ Requesting rollback to v{model_version} ...")
    result = _call_model_service(
        "rollback",
        investigation_id,
        model_version,
        kwargs["idempotency_key"],
        kwargs.get("config", {}),
    )
    if result.success:
        print(f"✅ Rollback completed: {result.message}")
    else:
        raise Exception(f"Rollback failed: {result.message}")


def replay(investigation_id: str, model_version: str, **kwargs):
    print(f"⏳ Requesting replay for model v{model_version} ...")
    result = _call_model_service(
        "replay",
        investigation_id,
        model_version,
        kwargs["idempotency_key"],
        kwargs.get("config", {}),
    )
    if result.success:
        print(f"✅ Replay accepted: {result.message}")
    else:
        raise Exception(f"Replay rejected: {result.message}")


# Required by the worker to dispatch tasks
TASK_MAP = {
    "retrain": retrain,
    "rollback": rollback,
    "replay": replay,
}