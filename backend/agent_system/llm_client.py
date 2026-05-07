"""Groq LLM client that loads prompts from files and calls the API.
Also provides a `call_llm` function compatible with the mock interface.
"""
import os
import json
import re
from pathlib import Path
from groq import Groq

# Prompt directory relative to this file
PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"

# Load prompt template from file
def load_prompt(name: str, **kwargs) -> str:
    """Read the prompt file and replace placeholders with kwargs."""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    template = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template

# Parse JSON from LLM response (may contain extra text)
def _extract_json(text: str) -> dict:
    # Try to find the first JSON object in the string
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {text[:200]}")

# Main call function – compatible replacement for mock_llm.call_llm
def call_llm(prompt: str) -> dict:
    """Send the prompt to Groq and return the parsed JSON response.
    This function can be easily mocked in tests.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=api_key)
    # Determine which model to use – can be overridden via env
    model = os.getenv("LLM_MODEL", "llama3-70b-8192")  # default, change if needed

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.2,
    )
    content = chat_completion.choices[0].message.content
    return _extract_json(content)