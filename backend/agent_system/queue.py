"""
Redis producer for slow agent actions.
Pushes jobs into a Redis list. Each job has an idempotency key
to guarantee once‑only execution.
"""
import json
import hashlib
import os
import redis
from typing import Literal

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
_redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def enqueue_job(
    action_type: Literal["retrain", "rollback", "replay"],
    investigation_id: str,
    model_version: str,
    idempotency_key: str,
    config: dict | None = None,
) -> dict:
    """
    Push a job into the agent's work queue.
    Returns a dict with job details.
    """
    job_payload = {
        "action_type": action_type,
        "investigation_id": investigation_id,
        "model_version": model_version,
        "idempotency_key": idempotency_key,
        "config": config or {},
    }
    # Push to the right side of the list (FIFO)
    _redis_client.rpush("agent:job_queue", json.dumps(job_payload))
    return job_payload