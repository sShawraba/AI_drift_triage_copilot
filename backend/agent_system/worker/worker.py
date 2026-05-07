"""
Redis consumer that processes agent jobs with:
- idempotency keys (once‑only execution)
- exponential backoff (retry up to max_retries)
- dead‑letter queue for failed jobs
- safe JSON parsing (skips corrupt payloads)
"""
import json
import time
import os
import redis
from backend.agent_system.worker.tasks import TASK_MAP

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MAX_RETRIES = 3
BASE_BACKOFF = 2  # seconds


def main():
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    print("👷 Worker started – waiting for jobs...")
    while True:
        try:
            # Block until a job arrives (timeout=5 allows graceful shutdown)
            result = r.blpop("agent:job_queue", timeout=5)
            if result is None:
                continue
            _, raw_job = result

            # Parse JSON safely – skip malformed jobs
            try:
                job = json.loads(raw_job)
            except json.JSONDecodeError:
                print(f"🚫 Invalid JSON – discarding: {raw_job[:100]}...")
                continue

            job_id = job.get("idempotency_key")
            if not job_id:
                print("❌ Job missing idempotency key – discarding")
                continue

            # Idempotency check
            if r.sismember("agent:processed_keys", job_id):
                print(f"⏭️  Skipping duplicate job {job_id}")
                continue

            print(f"🔧 Processing job {job_id} ({job['action_type']})")

            # Attempt with retries
            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    task_fn = TASK_MAP[job["action_type"]]
                    task_fn(
                        investigation_id=job["investigation_id"],
                        model_version=job["model_version"],
                        **job.get("config", {}),
                    )
                    success = True
                    break
                except Exception as e:
                    wait = BASE_BACKOFF ** attempt
                    print(f"⚠️  Attempt {attempt} failed – {e}. Retrying in {wait}s...")
                    if attempt < MAX_RETRIES:
                        time.sleep(wait)

            if success:
                # Mark as processed
                r.sadd("agent:processed_keys", job_id)
                print(f"✅ Job {job_id} completed")
            else:
                # Move to dead‑letter queue after exhausting retries
                r.rpush("agent:dead_letter_queue", json.dumps(job))
                print(f"💀 Job {job_id} exhausted retries – moved to DLQ")

        except Exception as e:
            # Catch‑all to keep the worker alive
            print(f"❌ Unexpected worker error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()