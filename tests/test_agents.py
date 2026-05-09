"""Unit tests for the individual agent nodes (triage / action / comms).

These tests invoke each LangGraph node directly — no graph compilation, no
checkpointer, no Postgres, no Redis. The mock LLM autouse fixture in
conftest.py supplies deterministic LLM output; DB / queue side-effects are
stubbed via monkeypatch so the suite is hermetic.

Run with:  pytest tests/test_agents.py -v
"""
from unittest.mock import MagicMock

import pytest
from langgraph.graph import END

from backend.agent_system.graph import (
    AgentState,
    action_node,
    comms_node,
    supervisor_router,
    triage_node,
)


SAMPLE_DRIFT_EVENT = {
    "event_id": "evt-1",
    "timestamp": "2026-01-01T00:00:00",
    "model_version": "v1",
    "psi_numeric": 0.1,
    "chi2_categorical": 0.2,
    "output_drift": 0.3,
    "severity": "medium",
}


def _make_state(**overrides) -> AgentState:
    state: AgentState = {
        "investigation_id": "test-1",
        "drift_event": dict(SAMPLE_DRIFT_EVENT),
        "triage_result": "",
        "proposed_action": "",
        "final_decision": "",
        "next_step": "",
        "human_approval": False,
        "approval_decision": None,
    }
    state.update(overrides)
    return state


@pytest.fixture
def stub_session(monkeypatch):
    """Patch SessionLocal so the stale-approval guard returns no newer row."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.return_value.filter.return_value.first.return_value = None

    monkeypatch.setattr(
        "backend.agent_system.state.session.SessionLocal",
        lambda: session,
    )
    return session


@pytest.fixture
def stub_enqueue(monkeypatch):
    """Capture calls to enqueue_job instead of hitting Redis."""
    calls: list[dict] = []

    def fake(action_type, investigation_id, model_version,
             idempotency_key, config=None):
        payload = {
            "action_type": action_type,
            "investigation_id": investigation_id,
            "model_version": model_version,
            "idempotency_key": idempotency_key,
            "config": config or {},
        }
        calls.append(payload)
        return payload

    monkeypatch.setattr("backend.agent_system.graph.enqueue_job", fake)
    return calls


# ───────────────────────── triage node ──────────────────────────
class TestTriageNode:
    def test_produces_triage_result(self):
        state = triage_node(_make_state())
        assert state["triage_result"]
        assert "severity assessed as" in state["triage_result"]

    def test_routes_to_action(self):
        # The mock LLM always returns next_step="action" for triage prompts.
        state = triage_node(_make_state())
        assert state["next_step"] == "action"

    def test_does_not_mutate_drift_event(self):
        original = dict(SAMPLE_DRIFT_EVENT)
        state = triage_node(_make_state())
        assert state["drift_event"] == original


# ───────────────────────── action node ──────────────────────────
class TestActionNode:
    def test_rejection_short_circuits_to_comms(self, stub_enqueue):
        state = _make_state(
            triage_result="severity assessed as medium",
            proposed_action="retrain",
            approval_decision=False,
        )
        result = action_node(state)

        assert result["final_decision"] == "Human rejected the action"
        assert result["next_step"] == "comms"
        assert stub_enqueue == []  # no job dispatched

    def test_approval_enqueues_job(self, stub_session, stub_enqueue):
        state = _make_state(
            triage_result="severity assessed as medium",
            proposed_action="retrain",
            approval_decision=True,
        )
        result = action_node(state)

        assert len(stub_enqueue) == 1
        job = stub_enqueue[0]
        assert job["action_type"] == "retrain"
        assert job["investigation_id"] == "test-1"
        assert job["model_version"] == "v1"
        assert job["idempotency_key"]  # non-empty
        assert "Action retrain dispatched to queue" in result["final_decision"]
        assert result["next_step"] == "comms"

    def test_idempotency_key_is_stable(self, stub_session, stub_enqueue):
        """Same (action, investigation_id) must always hash to the same key."""
        for _ in range(2):
            action_node(_make_state(
                triage_result="x",
                proposed_action="rollback",
                approval_decision=True,
            ))
        assert stub_enqueue[0]["idempotency_key"] == stub_enqueue[1]["idempotency_key"]

    def test_stale_approval_blocks_dispatch(self, monkeypatch, stub_enqueue):
        newer = MagicMock()
        newer.id = "newer-investigation-id"

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter.return_value.first.return_value = newer
        monkeypatch.setattr(
            "backend.agent_system.state.session.SessionLocal",
            lambda: session,
        )

        state = _make_state(
            triage_result="severity assessed as medium",
            proposed_action="retrain",
            approval_decision=True,
        )
        result = action_node(state)

        assert "Stale approval" in result["final_decision"]
        assert "newer-investigation-id" in result["final_decision"]
        assert result["next_step"] == "comms"
        assert stub_enqueue == []

    def test_no_approval_required_dispatches_directly(
        self, monkeypatch, stub_enqueue
    ):
        """If the LLM returns requires_approval=False, no human gate."""
        monkeypatch.setattr(
            "backend.agent_system.graph.call_llm",
            lambda prompt: {
                "proposed_action": "replay",
                "requires_approval": False,
            },
        )

        state = _make_state(triage_result="severity assessed as low")
        result = action_node(state)

        assert len(stub_enqueue) == 1
        assert stub_enqueue[0]["action_type"] == "replay"
        assert "Action replay dispatched to queue" in result["final_decision"]
        assert result["next_step"] == "comms"


# ───────────────────────── comms node ───────────────────────────
class TestCommsNode:
    def test_uses_llm_decision_when_returned(self, monkeypatch):
        monkeypatch.setattr(
            "backend.agent_system.graph.call_llm",
            lambda prompt: {"final_decision": "Action retrain will be executed"},
        )
        state = _make_state(
            proposed_action="retrain",
            final_decision="(stale state value)",
        )
        result = comms_node(state)
        assert result["final_decision"] == "Action retrain will be executed"

    def test_falls_back_to_state_decision_when_llm_blank(self, monkeypatch):
        monkeypatch.setattr(
            "backend.agent_system.graph.call_llm",
            lambda prompt: {"final_decision": ""},
        )
        state = _make_state(final_decision="pre-existing decision")
        result = comms_node(state)
        assert result["final_decision"] == "pre-existing decision"

    def test_default_when_state_missing_decision(self, monkeypatch):
        """When the LLM blank and state has no `final_decision` key, default kicks in."""
        monkeypatch.setattr(
            "backend.agent_system.graph.call_llm",
            lambda prompt: {},
        )
        state = _make_state()
        del state["final_decision"]  # simulate a missing key, not just empty
        result = comms_node(state)
        assert result["final_decision"] == "Investigation completed"


# ─────────────────────── supervisor router ──────────────────────
class TestSupervisorRouter:
    def test_routes_to_action(self):
        assert supervisor_router(_make_state(next_step="action")) == "action"

    def test_routes_to_comms(self):
        assert supervisor_router(_make_state(next_step="comms")) == "comms"

    def test_routes_to_end_on_unknown(self):
        assert supervisor_router(_make_state(next_step="bogus")) == END
