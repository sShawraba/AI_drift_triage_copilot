# Architecture – Drift Triage Co‑Pilot

## High‑level overview

The project is split into two independent services that communicate through a **shared contract**:

1. **Model Service** (Sara)  
   Trains ML models, serves predictions, computes drift, and exposes a promotion endpoint.

2. **Agent System** (Amer)  
   Consumes drift events, runs a LangGraph supervisor, pauses for human approval, and dispatches corrective actions via a Redis queue.

Both parts share a **Streamlit dashboard** that displays investigations, queue health, and the HIL inbox.

---

## Agent System – internal architecture


Webhook (FastAPI) → LangGraph Supervisor → Redis Queue
│ │
│ Postgres Checkpointer
│ │
Investigations Table Checkpoint Tables



### 1. FastAPI entrypoint (`main.py`)
- Loads environment variables and starts the server.
- On startup:
  - Creates database tables if missing.
  - Initialises the custom `PostgresSaver` and compiles the LangGraph graph with it.
- Exposes all agent endpoints via the webhook router.

### 2. Webhook router (`webhook.py`)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhook/drift-event` | Receive a `DriftEvent` from the model service. Invokes the graph. Returns `pending_approval` if the action requires human input, otherwise `completed`. |
| `GET` | `/webhook/pending` | List all investigations awaiting human approval (HIL inbox). |
| `POST` | `/webhook/investigations/{id}/approve` | Approve a pending action. Resumes the graph with `Command(resume={"approved":true})`. |
| `POST` | `/webhook/investigations/{id}/reject` | Reject a pending action. |
| `GET` | `/webhook/investigations` | Return all investigations, optionally filtered by `?status=...`. |
| `GET` | `/webhook/queue/stats` | Queue depth, DLQ depth, and processed‑key count. |
| `GET` | `/webhook/checkpoint/{id}` | Get the most recent LangGraph checkpoint for an investigation. |

### 3. LangGraph supervisor (`graph.py`)
- **State**: `AgentState` – a typed dict containing investigation ID, drift event, triage result, proposed action, final decision, routing info, and approval flag.
- **Nodes**:
  - `triage_node` – assesses severity and decides whether to escalate (`action`) or report (`comms`).
  - `action_node` – proposes a corrective action (retrain, rollback, replay). On first call it **interrupts** for human approval. After the human decision, it runs a **stale‑approval guard** and, if valid, enqueues the job via `queue.py`.
  - `comms_node` – generates a final summary.
- **Edges**:
  - After `triage`, conditional on `next_step` → `action` or `comms`.
  - After `action`, conditional on `next_step` → loops back to `action` (for approval resume) or `comms`.
  - `comms` → `END`.

### 4. Human‑in‑the‑loop (HIL) mechanism
- The `action_node` calls `interrupt()` from `langgraph.types`.
- The graph’s `ainvoke()` returns a state containing an `__interrupt__` marker.
- The webhook detects this marker, sets the investigation status to `PENDING_APPROVAL`, and stores the proposed action.
- When the dashboard (or API) calls `/approve` or `/reject`, the graph is resumed with `Command(resume={"approved": True/False})`. The decision is stored in `state["approval_decision"]`.
- The action node then re‑enters itself, processes the decision, runs the stale check, and either enqueues the job or marks the investigation as rejected.

### 5. Stale‑approval guard
Before executing an approved action, the agent queries the `investigations` table for a **newer** investigation with the same `model_version`. If one exists, the action is discarded with a `Stale approval` message. This prevents acting on outdated recommendations.

### 6. Redis job queue
- **Producer** (`queue.py`) – called by the action node when an action is approved and not stale. Pushes a JSON job into the Redis list `agent:job_queue`. Each job contains a unique `idempotency_key` (SHA‑256 of action type + investigation ID).
- **Worker** (`worker/worker.py`) – a standalone process that:
  - Blocks on `agent:job_queue` (FIFO).
  - Validates JSON and checks idempotency (`agent:processed_keys` set).
  - Executes the action via `tasks.py` with exponential backoff (max 3 retries).
  - On success, adds the key to the processed set.
  - On permanent failure, moves the job to the Dead‑Letter Queue (`agent:dead_letter_queue`).
- **Tasks** (`worker/tasks.py`) – implementations of `retrain`, `rollback`, `replay` (currently simulated; in production they would call the model service API).

### 7. Persistence layer
- **Database**: PostgreSQL, accessed via SQLAlchemy.
  - `investigations` table stores every investigation’s full lifecycle.
  - `agent_checkpoints` and `checkpoint_writes` store LangGraph state (using a custom async checkpointer).
- **ORM models** (`state/models.py`):
  - `Investigation` – id, status (open, pending_approval, approved, rejected, completed, failed), drift event JSON, triage/action/decision fields, model version, job ID.
- **Custom checkpointer** (`state/checkpoints.py`):
  - Implements `BaseCheckpointSaver` using `asyncpg` and `JsonPlusSerializer`.
  - Handles serialisation of special LangGraph objects (like `Interrupt`) by wrapping them in base64‑encoded JSON objects.
  - Supports `aget_tuple`, `aput`, `aput_writes`, `alist`, `adelete_thread`.

### 8. LLM integration
- **Provider**: Groq (configurable via env).
- **Prompt‑as‑code**: Prompts are stored as `.txt` files in `prompts/` and loaded by `llm_client.py`.
- **Mock fallback**: For testing, `conftest.py` replaces the LLM with a deterministic mock (`mock_llm.py`) – no API key required.

---

## Integration contract

Defined in `shared/schemas.py`:

- **DriftEvent** (Model Service → Agent)  
  `event_id`, `timestamp`, `model_version`, `psi_numeric`, `chi2_categorical`, `output_drift`, `severity`.

- **ActionRequest** (Agent → Model Service)  
  `action_id`, `investigation_id`, `event_id`, `action_type` (retrain|rollback|replay), `model_version`, `idempotency_key`, `config`.

- **PlatformResponse** (Model Service → Agent)  
  `success`, `message`, `action_id`.

---

## Cross‑cutting concerns

### Testing & CI
- **Snapshot trajectory test** (`tests/test_agent.py`) runs the full graph with the mock LLM and compares the state sequence against a committed fixture.
- **Queue idempotency test** (`tests/test_queue.py`) confirms that duplicate jobs are skipped.
- **CI** (`.github/workflows/ci.yml`) spins up Postgres + Redis on every push, runs all tests, and blocks merges on failure.

### Configurability
- All connection strings and API keys are read from `.env` (with sensible defaults).
- The LLM model and provider can be changed without touching code.

### Observability
- The dashboard polls `/webhook/investigations`, `/webhook/pending`, and `/webhook/queue/stats`.
- The worker prints rich logs for every job, including retries and DLQ transfers.