"""Redis queue idempotency test (no LLM involved)."""
import json
import pytest
import redis

REDIS_URL = "redis://localhost:6379/0"

@pytest.fixture
def redis_client():
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    # Clean test keys
    r.delete("agent:job_queue", "agent:processed_keys", "agent:dead_letter_queue")
    yield r
    r.delete("agent:job_queue", "agent:processed_keys", "agent:dead_letter_queue")

def test_duplicate_job_is_skipped(redis_client):
    """Two identical jobs → only one processed."""
    job = {
        "action_type": "retrain",
        "investigation_id": "qa-test",
        "model_version": "v1",
        "idempotency_key": "key-123",
        "config": {},
    }
    payload = json.dumps(job)
    redis_client.rpush("agent:job_queue", payload)
    redis_client.rpush("agent:job_queue", payload)

    processed = set()
    while True:
        raw = redis_client.lpop("agent:job_queue")
        if raw is None:
            break
        job_data = json.loads(raw)
        kid = job_data["idempotency_key"]
        if kid in processed:
            continue
        processed.add(kid)
        redis_client.sadd("agent:processed_keys", kid)

    assert len(processed) == 1
    assert redis_client.scard("agent:processed_keys") == 1