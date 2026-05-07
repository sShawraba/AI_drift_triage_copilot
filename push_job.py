"""Push a test job into Redis with valid JSON."""
import redis
import json

r = redis.Redis.from_url("redis://localhost:6379/0")

job = {
    "action_type": "retrain",
    "investigation_id": "dlq-demo",
    "model_version": "v99.9",
    "idempotency_key": "dlq-final-test",
    "config": {},
}

r.rpush("agent:job_queue", json.dumps(job))
print("Job pushed successfully.")