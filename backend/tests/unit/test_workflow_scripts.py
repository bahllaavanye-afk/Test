"""Unit tests for .github workflow scripts — verify structure and safety guards."""
from __future__ import annotations

from pathlib import Path
import re

import pytest

# Repo root: backend/tests/unit -> backend/tests -> backend -> Test
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


def test_multi_agent_discussion_has_call_llm():
    """multi_agent_discussion.py must define a call_llm function."""
    script_path = SCRIPTS_DIR / "multi_agent_discussion.py"
    assert script_path.exists(), f"multi_agent_discussion.py not found at {script_path}"
    source = script_path.read_text(encoding="utf-8")
    assert "def call_llm" in source, (
        "multi_agent_discussion.py must define a 'call_llm' function"
    )


def test_multi_agent_has_all_providers():
    """multi_agent_discussion.py must reference all required LLM providers."""
    script_path = SCRIPTS_DIR / "multi_agent_discussion.py"
    assert script_path.exists(), f"multi_agent_discussion.py not found at {script_path}"
    source = script_path.read_text(encoding="utf-8").lower()
    required_providers = [
        "groq",
        "deepseek",
        "sambanova",
        "cerebras",
        "hyperbolic",
        "together",
        "gemini",
    ]
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
        source = script_path.read_text(encoding="utf-8")
        assert guard in source, (
            f"{script_path.name} is missing the '{guard}' safety guard — "
            "this guard prevents accidental paid API usage"
        )


def test_scripts_are_non_empty():
    """All workflow scripts should contain non‑trivial content."""
    script_paths = [
        SCRIPTS_DIR / "multi_agent_discussion.py",
        SCRIPTS_DIR / "continuous_improver.py",
        SCRIPTS_DIR / "agent_health_monitor.py",
    ]
    for script_path in script_paths:
        assert script_path.exists(), f"{script_path.name} not found at {script_path}"
        content = script_path.read_text(encoding="utf-8")
        # Strip whitespace to avoid counting empty lines
        stripped = content.strip()
        assert len(stripped) > 10, (
            f"{script_path.name} appears to be empty or trivial; "
            "ensure the script contains actual implementation."
        )


def test_multi_agent_discussion_has_main_guard():
    """multi_agent_discussion.py should have a typical __main__ guard."""
    script_path = SCRIPTS_DIR / "multi_agent_discussion.py"
    assert script_path.exists(), f"multi_agent_discussion.py not found at {script_path}"
    source = script_path.read_text(encoding="utf-8")
    pattern = r'if\s+__name__\s*==\s*["\']__main__["\']\s*:'
    assert re.search(pattern, source), (
        "multi_agent_discussion.py should contain a '__main__' guard to allow "
        "stand‑alone execution."
    )


def test_allow_paid_apis_guard_is_boolean():
    """ALLOW_PAID_APIS guard should be defined as a boolean literal."""
    scripts_to_check = [
        SCRIPTS_DIR / "multi_agent_discussion.py",
        SCRIPTS_DIR / "continuous_improver.py",
    ]
    for script_path in scripts_to_check:
        assert script_path.exists(), f"{script_path.name} not found at {script_path}"
        source = script_path.read_text(encoding="utf-8")
        # Look for a line like: ALLOW_PAID_APIS = True or False
        match = re.search(r'^\s*ALLOW_PAID_APIS\s*=\s*(True|False)\b', source, re.MULTILINE)
        assert match, (
            f"{script_path.name} must define ALLOW_PAID_APIS as a boolean literal "
            "(True/False) for explicit safety."
        )