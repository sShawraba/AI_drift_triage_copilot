"""Snapshot trajectory regression test (mock LLM, no API key)."""
import json
from pathlib import Path
import pytest
from langgraph.types import Command
from backend.agent_system.graph import create_state_graph, AgentState
from backend.agent_system.state.checkpoints import create_checkpointer, setup_checkpointer

FIXTURE = Path(__file__).parent / "fixtures" / "trajectory_001.json"

@pytest.mark.asyncio
async def test_full_trajectory():
    """Graph must produce the same state sequence as the recorded snapshot."""
    saver = create_checkpointer()
    await setup_checkpointer(saver)
    graph = create_state_graph().compile(checkpointer=saver)

    tid = "fix-001"
    initial = {
        "investigation_id": tid,
        "drift_event": {
            "event_id": "test-1",
            "timestamp": "2026-01-01T00:00:00",
            "model_version": "v1",
            "psi_numeric": 0.1,
            "chi2_categorical": 0.2,
            "output_drift": 0.3,
            "severity": "medium"
        },
        "triage_result": "",
        "proposed_action": "",
        "final_decision": "",
        "next_step": "",
        "human_approval": False,
        "approval_decision": None,
    }
    config = {"configurable": {"thread_id": tid}}

    # Run graph exactly as when snapshot was recorded
    step1 = await graph.ainvoke(initial, config)
    step2 = await graph.ainvoke(Command(resume={"approved": True}), config)

    # Keep only our state keys, ignore LangGraph internals
    keys = list(AgentState.__required_keys__)
    actual = [
        {k: v for k, v in step1.items() if k in keys},
        {k: v for k, v in step2.items() if k in keys},
    ]

    with open(FIXTURE, "r") as f:
        expected = json.load(f)

    assert actual == expected, "Trajectory diverged from snapshot"