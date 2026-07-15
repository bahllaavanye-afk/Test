"""Unit tests for .github workflow scripts — verify structure and safety guards."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

# Repo root: backend/tests/unit -> backend/tests -> backend -> Test
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


def _resolve_script_path(script_name: str) -> Path:
    """Resolve the full path for a script safely.

    Handles None or empty inputs and ensures the base directories exist.
    """
    if not script_name:
        raise ValueError("script_name must be a non-empty string")
    if REPO_ROOT is None or not isinstance(REPO_ROOT, Path):
        raise RuntimeError("REPO_ROOT is not a valid Path")
    if SCRIPTS_DIR is None or not isinstance(SCRIPTS_DIR, Path):
        raise RuntimeError("SCRIPTS_DIR is not a valid Path")
    script_path = SCRIPTS_DIR / script_name
    return script_path


def _read_script_source(script_path: Path) -> str:
    """Read the script source safely, handling missing files or empty content."""
    if script_path is None or not isinstance(script_path, Path):
        raise ValueError("script_path must be a valid Path")
    if not script_path.exists():
        raise FileNotFoundError(f"{script_path.name} not found at {script_path}")
    source = script_path.read_text(encoding="utf-8")
    # Treat empty files as missing content
    if source == "":
        raise ValueError(f"{script_path.name} is empty")
    return source


def test_multi_agent_discussion_has_call_llm():
    """multi_agent_discussion.py must define a call_llm function."""
    script_path = _resolve_script_path("multi_agent_discussion.py")
    source = _read_script_source(script_path)
    assert "def call_llm" in source, (
        "multi_agent_discussion.py must define a 'call_llm' function"
    )


def test_multi_agent_has_all_providers():
    """multi_agent_discussion.py must reference all required LLM providers."""
    script_path = _resolve_script_path("multi_agent_discussion.py")
    source = _read_script_source(script_path).lower()
    required_providers: List[str] = [
        "groq",
        "deepseek",
        "sambanova",
        "cerebras",
        "hyperbolic",
        "together",
        "gemini",
    ]
    if not required_providers:
        raise AssertionError("required_providers list is empty")
    missing = [p for p in required_providers if p not in source]
    assert not missing, (
        f"multi_agent_discussion.py is missing references to providers: {missing}"
    )


def test_continuous_improver_exists():
    """continuous_improver.py must exist under .github/scripts/."""
    script_path = _resolve_script_path("continuous_improver.py")
    # Existence is verified by _read_script_source; any issue raises an exception.
    _ = _read_script_source(script_path)


def test_agent_health_monitor_exists():
    """agent_health_monitor.py must exist under .github/scripts/."""
    script_path = _resolve_script_path("agent_health_monitor.py")
    _ = _read_script_source(script_path)


def test_no_paid_api_guard_missing():
    """Both multi_agent_discussion.py and continuous_improver.py must contain ALLOW_PAID_APIS guard."""
    scripts_to_check = [
        _resolve_script_path("multi_agent_discussion.py"),
        _resolve_script_path("continuous_improver.py"),
    ]
    if not scripts_to_check:
        raise AssertionError("scripts_to_check list is empty")
    guard = "ALLOW_PAID_APIS"
    for script_path in scripts_to_check:
        source = _read_script_source(script_path)
        assert guard in source, (
            f"{script_path.name} is missing the '{guard}' safety guard — "
            "this guard prevents accidental paid API usage"
        )