# Runbook – Drift Triage Co‑Pilot (Agent Side)

## 1. Prerequisites
- Docker Desktop running
- `uv` installed (Python package manager)
- Groq API key (for LLM calls; mock is used in tests)

## 2. Clone & Environment
```bash
git clone <repo-url>
cd AI_drift_triage_copilot
cp .env.example .env
# Edit .env with your GROQ_API_KEY

Example
DATABASE_URL=postgresql://admin:admin@localhost:5432/mlops
REDIS_URL=redis://localhost:6379/0
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=llama3-70b-8192   # or your chosen Groq model

3. Start Infrastructure (Postgres + Redis)

docker run -d --name postgres -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=admin -e POSTGRES_DB=mlops -p 5432:5432 postgres
docker run -d --name redis -p 6379:6379 redis

4. Install Dependencies & Run DB Migrations
uv sync
uv run alembic -c backend/agent_system/state/alembic.ini upgrade head


5. Start the Agent & Worker

Open three terminals:

Terminal 1 – Agent Server

uv run uvicorn backend.agent_system.main:app --reload --port 8000


Terminal 2 – Worker

uv run python -m backend.agent_system.worker.worker

The worker will display: 👷 Worker started – waiting for jobs...


Terminal 3 – Used for sending requests (PowerShell / bash)


6. Full End‑to‑End Test (HIL + Queue)
6.1 Health Check

curl http://localhost:8000/health

Expected: {"status":"ok"}


6.2 Trigger a Drift Event (will pause for approval)


$body = @{
    event_id      = "evt-runbook-1"
    timestamp     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    model_version = "v2.5.0"
    psi_numeric   = 0.85
    chi2_categorical = 0.72
    output_drift  = 0.91
    severity      = "high"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://localhost:8000/webhook/drift-event -Method Post -Body $body -ContentType "application/json"
$response


The response should contain "status": "pending_approval" and a "proposed_action" (e.g., "retrain").

Save the investigation ID:
$invId = $response.investigation_id

6.3 List Pending Approvals
Invoke-RestMethod -Uri http://localhost:8000/webhook/pending

6.4 Approve the Action

Invoke-RestMethod -Uri "http://localhost:8000/webhook/investigations/$invId/approve" -Method Post -Body '{"approve":true}' -ContentType "application/json"

In the Worker terminal, you will see the job being processed:

🔧 Processing job abc123... (retrain)
⏳ Retraining model v2.5.0 ...
✅ Retrain complete for model v2.5.0
✅ Job abc123... completed

6.5 Verify Investigation Status

docker exec -it postgres psql -U admin -d mlops -c "SELECT id, status, proposed_action, final_decision FROM investigations WHERE id='$invId';"

Status should be COMPLETED and final_decision will mention the queue dispatch.

6.6 Check Queue Statistics

Invoke-RestMethod -Uri http://localhost:8000/webhook/queue/stats

Output includes queue_depth, dead_letter_depth, processed_count.


7. Testing the Dead‑Letter Queue (DLQ)
7.1 Simulate a Failed Job
Temporarily edit backend/agent_system/worker/tasks.py – change the retrain function to raise an exception:

def retrain(investigation_id: str, model_version: str, **kwargs):
    raise Exception("Simulated failure for DLQ test")

    7.2 Restart the Worker (Ctrl+C and run again)

    uv run python -m backend.agent_system.worker.worker

    7.3 Push a Failing Job (via Python for valid JSON)

    Run it: uv run python -m push_job


    7.4 Observe the Worker
It will retry three times, then move the job to the DLQ:
💀 Job dlq-test-key exhausted retries – moved to DLQ

7.5 Confirm DLQ Contents
docker exec -it redis redis-cli LRANGE agent:dead_letter_queue 0 -1

7.6 Revert the Task
Restore the original retrain function, restart the worker. The DLQ will still hold the old failed job for inspection.

8. Running Automated Tests (No API Key Required)

8.1 Record the Trajectory Fixture (one‑time)

uv run python scripts/record_trajectory.py

This creates tests/fixtures/trajectory_001.json. Commit this file.

8.2 Run All Tests Locally
uv run pytest tests/

Expected: two tests pass – trajectory regression test and queue idempotency test.

8.3 CI on GitHub
On every push to main, dev, or agent-work, a GitHub Actions workflow:

Spins up Postgres & Redis

Installs dependencies with uv

Runs uv run pytest tests/
A green checkmark means everything is healthy.

9. Common Debugging Commands
View all investigations: GET /webhook/investigations

View only pending: GET /webhook/pending

Inspect Redis job queue: docker exec -it redis redis-cli LRANGE agent:job_queue 0 -1

Inspect dead‑letter queue: docker exec -it redis redis-cli LRANGE agent:dead_letter_queue 0 -1

Clear all Redis state: docker exec -it redis redis-cli FLUSHALL

Check Postgres tables: docker exec -it postgres psql -U admin -d mlops -c "\dt"

View checkpoints: docker exec -it postgres psql -U admin -d mlops -c "SELECT thread_id, checkpoint_id FROM agent_checkpoints;"

View checkpoint writes: docker exec -it postgres psql -U admin -d mlops -c "SELECT * FROM checkpoint_writes LIMIT 5;"


10. Tear Down

# Stop containers
docker stop postgres redis
# Or remove them if you want to start fresh later
docker rm postgres redis




---

That `RUNBOOK.md` covers everything the agent does, including the complete HIL flow, queue/DLQ tests, automated testing, and debugging. You can place this file in the project root and commit it as part of Phase 8.