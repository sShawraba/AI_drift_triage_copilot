"""
Auto‑mock the LLM for all tests so they run without an API key.
Also provides event loop for async tests.
"""
import pytest
from backend.agent_system.utils.mock_llm import call_llm as mock_call

@pytest.fixture(autouse=True)
def mock_llm_for_tests(monkeypatch):
    """Replace graph.call_llm with the mock."""
    monkeypatch.setattr(
        "backend.agent_system.graph.call_llm",
        mock_call,
    )