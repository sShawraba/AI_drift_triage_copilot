"""Mock LLM that returns deterministic responses based on prompt content.
Used for early development without API keys. Will be replaced by a real LLM client in Phase 5."""

import json
import hashlib
from typing import Dict, Any

def call_llm(prompt: str) -> Dict[str, Any]:
    """
    Simulates an LLM call by returning a fixed, deterministic JSON object
    derived from a hash of the prompt. This keeps the graph routing stable.
    """
    hash_val = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 10

    # ---------- Triage response ----------
    if "triage" in prompt.lower():
        severities = ["low", "medium", "high"]
        selected = severities[hash_val % 3]
        return {
            "triage_result": f"severity assessed as {selected}",
            "next_step": "action",              # <-- force routing to action node
            "explanation": f"Mock triage: drift detected at {selected} level."
        }

    # ---------- Action response ----------
    elif "action" in prompt.lower():
        actions = ["retrain", "rollback", "replay"]
        chosen = actions[hash_val % 3]
        return {
            "proposed_action": chosen,
            "reason": f"Mock action selection: {chosen} is appropriate.",
            "requires_approval": True,         # <-- always require approval
            "confidence": 0.8
        }

    # ---------- Comms response ----------
    elif "comms" in prompt.lower():
        return {
            "final_decision": "report",
            "summary": "The situation has been handled; no further action required."
        }

    # Fallback
    return {"message": "mock response"}