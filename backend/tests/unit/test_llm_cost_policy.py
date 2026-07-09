"""Enforce the LLM cost policy: Claude is a RARE backstop, not per-task.

Answers "is Claude used for each task?" — it must not be. These tests lock the
invariant so a future edit can't silently make every employee call the paid
tier and drain the prepaid balance. Static assertions over the shipped config
(no network, no import of heavy deps).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LLM_COMMON = ROOT / ".github" / "scripts" / "llm_common.py"
AGENT_TEAM = ROOT / ".github" / "scripts" / "slack_agent_team.py"


def _src(p: Path | None) -> str:
    """Read the source file safely.

    Returns an empty string if the path is None, does not exist, or cannot be
    read. This defensive behaviour allows the tests to fail with a clear
    assertion rather than raising an unexpected exception.
    """
    if p is None:
        return ""
    try:
        return p.read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def test_default_backstop_is_the_cheapest_claude_tier():
    """The default Claude backstop must be Haiku (cheap), never Opus/Sonnet."""
    src = _src(LLM_COMMON)
    assert src, f"Unable to read source from {LLM_COMMON!s}"
    m = re.search(r'_CLAUDE_BACKSTOP_MODEL\s*=\s*os\.environ\.get\(\s*"CLAUDE_BACKSTOP_MODEL"\s*,\s*"([^"]+)"', src)
    assert m, "default backstop model not found"
    default = m.group(1)
    assert "haiku" in default.lower(), f"default backstop must be Haiku-class, got {default!r}"
    assert "opus" not in default.lower(), "Opus must never be the default backstop"


def test_routing_tries_free_cascade_first():
    """Tier 1 of llm_routed must be the FREE cascade (paid never runs first)."""
    src = _src(LLM_COMMON)
    assert src, f"Unable to read source from {LLM_COMMON!s}"
    # The free parallel race is tier 1 and appears before any OpenRouter/Claude call.
    free_pos = src.find("_call_parallel_race")
    claude_pos = src.find("CLAUDE backstop")
    # Edge case: ensure both markers exist
    assert free_pos != -1 and claude_pos != -1, "Required markers not found in source"
    # Off‑by‑one guard: free must strictly precede Claude
    assert free_pos < claude_pos, "free cascade must be tried before the Claude backstop"


def test_claude_backstop_is_last_resort_only():
    """Claude tier must be gated to 'hard'/'auto' as a last resort, not default 'cheap'."""
    src = _src(LLM_COMMON)
    assert src, f"Unable to read source from {LLM_COMMON!s}"
    # The Claude tier guard must require tier hard/auto AND not-yet-resolved.
    assert re.search(r'if not result and \(tier == "hard" or tier == "auto"\)', src), \
        "Claude backstop must be gated on tier hard/auto AND prior failure"
    # "cheap" tier must be documented as free-only.
    assert re.search(r'"cheap".{0,80}free cascade only', src, re.DOTALL | re.IGNORECASE), \
        "cheap tier must be free-cascade-only"


def test_employees_are_zero_spend_by_default():
    """The hourly employee team must default to zero paid spend."""
    src = _src(AGENT_TEAM)
    assert src, f"Unable to read source from {AGENT_TEAM!s}"
    assert re.search(r'ALLOW_PAID_APIS[^\n]*=\s*False', src), \
        "employees must default ALLOW_PAID_APIS = False"