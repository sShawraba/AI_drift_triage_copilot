"""LangGraph supervisor with human‑in‑the‑loop (Groq LLM + model service linkage)."""
import json
import hashlib
from typing import TypedDict, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from backend.agent_system.llm_client import call_llm, load_prompt
from backend.agent_system.queue import enqueue_job

class AgentState(TypedDict):
    investigation_id: str
    drift_event: dict
    triage_result: str
    proposed_action: str
    final_decision: str
    next_step: str
    human_approval: bool
    approval_decision: Optional[bool]   # True = approved, False = rejected

# ── Nodes ────────────────────────────────────────────────────────
def triage_node(state: AgentState) -> AgentState:
    prompt = load_prompt(
        "triage",
        model_version=state["drift_event"].get("model_version", ""),
        psi_numeric=state["drift_event"].get("psi_numeric", ""),
        chi2_categorical=state["drift_event"].get("chi2_categorical", ""),
        output_drift=state["drift_event"].get("output_drift", ""),
        severity=state["drift_event"].get("severity", ""),
    )
    llm_out = call_llm(prompt)
    state["triage_result"] = llm_out.get("triage_result", "")
    state["next_step"] = llm_out.get("next_step", "comms")
    return state

def action_node(state: AgentState) -> AgentState:
    # ---- Resuming after a human decision ----
    if state.get("approval_decision") is not None:
        if not state["approval_decision"]:
            state["final_decision"] = "Human rejected the action"
            state["next_step"] = "comms"
            return state

        # Stale‑approval guard
        from backend.agent_system.state.session import SessionLocal
        from backend.agent_system.state.models import Investigation

        model_version = state["drift_event"]["model_version"]
        drift_ts = datetime.fromisoformat(state["drift_event"]["timestamp"])

        with SessionLocal() as session:
            newer = (
                session.query(Investigation)
                .filter(
                    Investigation.model_version == model_version,
                    Investigation.created_at > drift_ts,
                    Investigation.id != state["investigation_id"]
                )
                .first()
            )
        if newer:
            state["final_decision"] = (
                f"Stale approval – newer investigation {newer.id} exists"
            )
            state["next_step"] = "comms"
            return state

        # ---------- Approved & not stale → enqueue the job ----------
        action = state["proposed_action"]
        idem_key = hashlib.sha256(
            f"{action}-{state['investigation_id']}".encode()
        ).hexdigest()

        enqueue_job(
            action_type=action,
            investigation_id=state["investigation_id"],
            model_version=state["drift_event"]["model_version"],
            idempotency_key=idem_key,
            config={"event_id": state["drift_event"].get("event_id", "")},
        )
        state["final_decision"] = (
            f"Action {action} dispatched to queue "
            f"(idempotency_key={idem_key[:8]}...)"
        )
        state["next_step"] = "comms"
        return state

    # ---- First time in the action node: propose action & interrupt ----
    prompt = load_prompt(
        "action",
        triage_result=state["triage_result"],
        summary=json.dumps(state["drift_event"]),
    )
    llm_out = call_llm(prompt)
    state["proposed_action"] = llm_out.get("proposed_action", "")
    state["human_approval"] = llm_out.get("requires_approval", True)

    if state["human_approval"]:
        decision = interrupt({
            "proposed_action": state["proposed_action"],
            "message": (
                f"Proposed action: {state['proposed_action']}. "
                "Approve? (Send {'approved': True} or {'approved': False})"
            )
        })
        if isinstance(decision, dict):
            approved = decision.get("approved", False)
        else:
            approved = bool(decision)
        state["approval_decision"] = approved

        if not approved:
            state["final_decision"] = "Human rejected the action"
            state["next_step"] = "comms"
            return state

        # Route back to action after resume
        state["next_step"] = "action"
        return state

    # No approval required (LLM returned requires_approval=False)
    action = state["proposed_action"]
    idem_key = hashlib.sha256(
        f"{action}-{state['investigation_id']}".encode()
    ).hexdigest()
    enqueue_job(
        action_type=action,
        investigation_id=state["investigation_id"],
        model_version=state["drift_event"]["model_version"],
        idempotency_key=idem_key,
        config={"event_id": state["drift_event"].get("event_id", "")},
    )
    state["final_decision"] = (
        f"Action {action} dispatched to queue "
        f"(idempotency_key={idem_key[:8]}...)"
    )
    state["next_step"] = "comms"
    return state

def comms_node(state: AgentState) -> AgentState:
    prompt = load_prompt(
        "comms",
        proposed_action=state.get("proposed_action", ""),
        final_decision=state.get("final_decision", "Investigation completed"),
    )
    llm_out = call_llm(prompt)
    decision = llm_out.get("final_decision", "")
    if not decision:
        decision = state.get("final_decision", "Investigation completed")
    state["final_decision"] = decision
    return state

# ── Supervisor router ────────────────────────────────────────────
def supervisor_router(state: AgentState) -> str:
    if state.get("next_step") == "action":
        return "action"
    elif state.get("next_step") == "comms":
        return "comms"
    else:
        return END

# ── Graph builder ─────────────────────────────────────────────────
def create_state_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("action", action_node)
    graph.add_node("comms", comms_node)

    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage",
        supervisor_router,
        {"action": "action", "comms": "comms"},
    )
    graph.add_conditional_edges(
        "action",
        supervisor_router,
        {"action": "action", "comms": "comms"},
    )
    graph.add_edge("comms", END)
    return graph