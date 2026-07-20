"""Unit tests for .github workflow scripts — verify structure and safety guards."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

# Repo root: backend/tests/unit -> backend/tests -> backend -> Test
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


def _read_script(path: Path | None) -> str:
    """Safely read a script file.

    Returns an empty string if *path* is ``None`` or the file does not exist.
    This helper protects the test suite from ``None`` inputs and missing files,
    turning hard failures into graceful empty‑string returns.
    """
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def test_multi_agent_discussion_has_call_llm():
    """multi_agent_discussion.py must define a call_llm function."""
    script_path = SCRIPTS_DIR / "multi_agent_discussion.py"
    assert script_path.exists(), f"multi_agent_discussion.py not found at {script_path}"
    source = _read_script(script_path)
    assert "def call_llm" in source, (
        "multi_agent_discussion.py must define a 'call_llm' function"
    )


def test_multi_agent_has_all_providers():
    """multi_agent_discussion.py must reference all required LLM providers."""
    script_path = SCRIPTS_DIR / "multi_agent_discussion.py"
    assert script_path.exists(), f"multi_agent_discussion.py not found at {script_path}"
    source = _read_script(script_path).lower()
    required_providers: List[str] = [
        "groq",
        "deepseek",
        "sambanova",
        "cerebras",
        "hyperbolic",
        "together",
        "gemini",
    ]
    # Guard against off‑by‑one errors: ensure the list is not empty and has the expected length.
    expected_len = 7
    assert len(required_providers) == expected_len, (
        f"required_providers list should contain {expected_len} items, "
        f"found {len(required_providers)}"
    )
    missing = [p for p in required_providers if p not in source]
    assert not missing, (
        f"multi_agent_discussion.py is missing references to providers: {missing}"
    )


def test_continuous_improver_exists():
    """continuous_improver.py must exist under .github/scripts/."""
    script_path = SCRIPTS_DIR / "continuous_improver.py"
    assert script_path.exists(), f"continuous_improver.py not found at {script_path}"


def test_agent_health_monitor_exists():
    """agent_health_monitor.py must exist under .github/scripts/."""
    script_path = SCRIPTS_DIR / "agent_health_monitor.py"
    assert script_path.exists(), f"agent_health_monitor.py not found at {script_path}"


def test_no_paid_api_guard_missing():
    """Both multi_agent_discussion.py and continuous_improver.py must contain ALLOW_PAID_APIS guard."""
    scripts_to_check = [
        SCRIPTS_DIR / "multi_agent_discussion.py",
        SCRIPTS_DIR / "continuous_improver.py",
    ]
    guard = "ALLOW_PAID_APIS"
    for script_path in scripts_to_check:
        assert script_path.exists(), f"{script_path.name} not found at {script_path}"
        source = _read_script(script_path)
        assert guard in source, (
            f"{script_path.name} is missing the '{guard}' safety guard — "
            "this guard prevents accidental paid API usage"
        )


def test_read_script_with_none_input():
    """_read_script should gracefully handle a None input."""
    assert _read_script(None) == "", "Expected empty string when path is None"


def test_read_script_with_missing_file():
    """_read_script should return empty string for a non‑existent file."""
    missing_path = SCRIPTS_DIR / "non_existent_script.py"
    # Ensure the path truly does not exist for the test
    if missing_path.exists():
        missing_path.unlink()
    assert _read_script(missing_path) == "", "Expected empty string for missing file"