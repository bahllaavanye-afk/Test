"""Integration tests for the employee conversation runner."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT    = Path(__file__).resolve().parents[3]
RUNNER_PATH  = REPO_ROOT / ".github" / "scripts" / "employee_conversation_runner.py"
PROOF_PATH   = REPO_ROOT / ".github" / "state" / "collaboration_proof.md"
MEMORY_PATH  = REPO_ROOT / ".github" / "state" / "agent_memory.json"

_LLM_KEY_VARS = [
    "GROQ_API_KEY", "DEEPSEEK_API_KEY", "SAMBANOVA_API_KEY",
    "CEREBRAS_API_KEY", "HYPERBOLIC_API_KEY", "TOGETHER_API_KEY", "GEMINI_API_KEY",
]


def test_employee_runner_script_exists():
    assert RUNNER_PATH.exists(), f"Missing: {RUNNER_PATH}"


def test_employee_runner_syntax():
    source = RUNNER_PATH.read_text()
    try:
        compile(source, str(RUNNER_PATH), "exec")
    except SyntaxError as exc:
        pytest.fail(f"Syntax error in employee_conversation_runner.py: {exc}")


def test_collaboration_proof_schema():
    if not PROOF_PATH.exists():
        pytest.skip("collaboration_proof.md not yet generated")
    text = PROOF_PATH.read_text()
    assert "RESPONDED_COUNT" in text, "Missing RESPONDED_COUNT in proof file"
    # Date header should appear (YYYY-MM-DD format)
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", text), "Missing date header in proof file"


def test_agent_memory_conversations_schema():
    if not MEMORY_PATH.exists():
        pytest.skip("agent_memory.json not found")
    data = json.loads(MEMORY_PATH.read_text())
    convs = data.get("conversations", {})
    if not convs:
        pytest.skip("conversations dict is empty — no runs yet")
    for ts, entry in convs.items():
        assert "speaker" in entry, f"Entry {ts} missing 'speaker'"
        assert "message" in entry, f"Entry {ts} missing 'message'"
        assert "provider" in entry, f"Entry {ts} missing 'provider'"


def test_real_employee_conversations_in_ci():
    if not any(os.environ.get(k, "").strip() for k in _LLM_KEY_VARS):
        pytest.skip("No LLM API keys set — skipping live runner test")
    # Cap to a few employees so the live LLM run is bounded and reliable — the full
    # roster makes dozens of sequential calls and blows any fixed CI timeout. This
    # still exercises the real runner + real providers end-to-end.
    env = {**os.environ, "EMPLOYEE_RUNNER_LIMIT": "3"}
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )
    assert result.returncode == 0, (
        f"employee_conversation_runner.py exited {result.returncode}\n"
        f"stdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )
