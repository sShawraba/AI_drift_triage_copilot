# Design Decisions

## 1. LangGraph Supervisor Topology (True Supervisor, Not a Chain)
**What**: Three sub‑agents (`triage`, `action`, `comms`) controlled by a `supervisor_router` using conditional edges.  
**Why**: The project demands a real supervisor – each node can decide the next step, and the graph can loop back (e.g., after human approval the action node re‑enters itself). A linear chain cannot pause for HIL and resume correctly.

## 2. Custom Postgres Checkpointer (not `langgraph_checkpoint`)
**What**: Self‑built `PostgresSaver` using `asyncpg` and `JsonPlusSerializer`, storing checkpoints in `agent_checkpoints` and writes in `checkpoint_writes`.  
**Why**: The official `langgraph_checkpoint` package was broken in our environment. Our implementation gives full control, adds stale‑approval‑guard queries, and guarantees persistence across container restarts.

## 3. Human‑in‑the‑Loop via `interrupt()` + Webhook API
**What**: The `action_node` calls `interrupt()` to pause the graph and returns `__interrupt__` to the webhook. The dashboard (or a simple API call) resumes via `Command(resume=...)`.  
**Why**: This keeps the agent logic inside the graph while allowing external approval. The investigation status is set to `PENDING_APPROVAL` and only becomes `COMPLETED` after human input.

## 4. Stale‑Approval Guard
**What**: Before executing an approved action, the agent checks if a newer investigation for the same model version exists (using the `investigations` table).  
**Why**: Prevents acting on outdated recommendations when a more recent drift event has already been handled.

## 5. Redis Job Queue with Idempotency and DLQ
**What**: Approved actions are dispatched to a Redis list (`agent:job_queue`). A separate worker process processes them with exponential backoff (max 3 retries) and moves permanently failed jobs to a Dead‑Letter Queue (`agent:dead_letter_queue`). Idempotency is guaranteed by a `SHA‑256` key per job, stored in a Redis set (`agent:processed_keys`).  
**Why**: Retraining/rollback/replay are slow operations; the agent must never block. Idempotency ensures duplicate approvals don’t trigger multiple trainings.

## 6. Prompt‑as‑Code
**What**: All LLM prompts live in `prompts/` as `.txt` files, loaded by `llm_client.load_prompt()`.  
**Why**: Prompts are part of the application logic and must be version‑controlled, not buried in code.

## 7. Mock LLM for CI/Testing
**What**: A deterministic mock (`mock_llm.py`) that returns fixed responses based on a hash of the prompt. `conftest.py` automatically swaps the real LLM with the mock in tests.  
**Why**: Trajectory snapshot tests run without an API key, ensuring graph logic is regression‑tested in CI.

## 8. Single Contract Layer (`shared/schemas.py`)
**What**: `DriftEvent`, `ActionRequest`, and `PlatformResponse` are defined once and shared between agent and model service.  
**Why**: Breaks the dependency – both sides can evolve independently as long as the contract is respected.