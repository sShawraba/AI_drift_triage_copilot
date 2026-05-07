"""
Record a full trajectory using the mock LLM and save it as a
deterministic snapshot fixture for regression tests.

Run once:
  uv run python scripts/record_trajectory.py
"""
import asyncio
import json
from pathlib import Path
from langgraph.types import Command
from backend.agent_system.graph import create_state_graph, AgentState
from backend.agent_system.state.checkpoints import create_checkpointer, setup_checkpointer

# Force mock LLM (exactly what conftest does)
from backend.agent_system.utils.mock_llm import call_llm as mock_call
import backend.agent_system.graph
backend.agent_system.graph.call_llm = mock_call

async def main():
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

    step1 = await graph.ainvoke(initial, config)
    step2 = await graph.ainvoke(Command(resume={"approved": True}), config)

    keys = list(AgentState.__required_keys__)
    trajectory = [
        {k: v for k, v in step1.items() if k in keys},
        {k: v for k, v in step2.items() if k in keys},
    ]

    out_path = Path("tests/fixtures/trajectory_001.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trajectory, indent=2))
    print("Fixture written to", out_path)

asyncio.run(main())