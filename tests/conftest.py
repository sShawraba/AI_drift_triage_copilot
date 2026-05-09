# tests/conftest.py (full)
"""
Auto‑mock the LLM for all tests so they run without an API key.
Also creates the investigations table required by the stale‑check in action_node.
"""
import pytest
from backend.agent_system.utils.mock_llm import call_llm as mock_call
from backend.agent_system.state.models import Base
from backend.agent_system.state.session import engine


@pytest.fixture(autouse=True)
def mock_llm_for_tests(monkeypatch):
    """Replace graph.call_llm with the mock."""
    monkeypatch.setattr(
        "backend.agent_system.graph.call_llm",
        mock_call,
    )


@pytest.fixture(scope="session", autouse=True)
def create_tables():
    """Ensure the investigations table exists before any test runs.

    If the database isn't reachable (e.g. running tests on the host without
    docker compose up), skip silently — pure unit tests don't need it, and
    tests that *do* hit the DB will fail at their own call site.
    """
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        pass
    yield