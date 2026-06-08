"""Agent system integration tests — verify .github state files and scripts exist and are valid."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Repo root: backend/tests/integration -> backend/tests -> backend -> Test
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

STATE_DIR = REPO_ROOT / ".github" / "state"
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


def test_agent_memory_file_exists():
    """agent_memory.json must exist under .github/state/."""
    memory_path = STATE_DIR / "agent_memory.json"
    assert memory_path.exists(), f"agent_memory.json not found at {memory_path}"


def test_agent_memory_schema_valid():
    """agent_memory.json must contain the required top-level keys."""
    memory_path = STATE_DIR / "agent_memory.json"
    assert memory_path.exists(), f"agent_memory.json not found at {memory_path}"
    data = json.loads(memory_path.read_text())
    required_keys = {"version", "last_updated", "conversations", "platform_metrics"}
    missing = required_keys - set(data.keys())
    assert not missing, f"agent_memory.json is missing keys: {missing}"


def test_skill_library_exists():
    """skill_library.json must exist under .github/state/."""
    skill_path = STATE_DIR / "skill_library.json"
    assert skill_path.exists(), f"skill_library.json not found at {skill_path}"


def test_task_registry_exists():
    """task_registry.json must exist under .github/state/."""
    registry_path = STATE_DIR / "task_registry.json"
    assert registry_path.exists(), f"task_registry.json not found at {registry_path}"


def test_multi_agent_script_importable():
    """multi_agent_discussion.py must exist and be syntactically valid Python."""
    script_path = SCRIPTS_DIR / "multi_agent_discussion.py"
    assert script_path.exists(), f"multi_agent_discussion.py not found at {script_path}"
    source = script_path.read_text(encoding="utf-8")
    try:
        compile(source, str(script_path), "exec")
    except SyntaxError as exc:
        pytest.fail(f"multi_agent_discussion.py has a syntax error: {exc}")


def test_claude_conversations_script_importable():
    """claude_conversations.py must exist and be syntactically valid Python."""
    script_path = SCRIPTS_DIR / "claude_conversations.py"
    assert script_path.exists(), f"claude_conversations.py not found at {script_path}"
    source = script_path.read_text(encoding="utf-8")
    try:
        compile(source, str(script_path), "exec")
    except SyntaxError as exc:
        pytest.fail(f"claude_conversations.py has a syntax error: {exc}")


def test_agent_memory_conversations_structure():
    """If conversations key exists in agent_memory.json, it must be a dict (not None)."""
    memory_path = STATE_DIR / "agent_memory.json"
    assert memory_path.exists(), f"agent_memory.json not found at {memory_path}"
    data = json.loads(memory_path.read_text())
    if "conversations" in data:
        assert isinstance(data["conversations"], dict), (
            f"conversations must be a dict, got {type(data['conversations'])}"
        )


def test_platform_metrics_reasonable():
    """platform_metrics.strategies_live must be an int >= 0."""
    memory_path = STATE_DIR / "agent_memory.json"
    assert memory_path.exists(), f"agent_memory.json not found at {memory_path}"
    data = json.loads(memory_path.read_text())
    assert "platform_metrics" in data, "agent_memory.json missing 'platform_metrics'"
    metrics = data["platform_metrics"]
    assert "strategies_live" in metrics, "platform_metrics missing 'strategies_live'"
    strategies_live = metrics["strategies_live"]
    assert isinstance(strategies_live, int), (
        f"strategies_live must be an int, got {type(strategies_live)}"
    )
    assert strategies_live >= 0, (
        f"strategies_live={strategies_live} must be >= 0"
    )
