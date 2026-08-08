"""Enforce the LLM cost policy: Claude is a RARE backstop, not per-task.

Answers "is Claude used for each task?" — it must not be. These tests lock the
invariant so a future edit can't silently make every employee call the paid
tier and drain the prepaid balance. Static assertions over the shipped config
(no network, no import of heavy deps).
"""
from __future__ import annotations

import re
from pathlib import Path

# -------------------------------------------------------------------------
# Constants – extracted from magic strings / numbers used in tests
# -------------------------------------------------------------------------

# Regex to locate the default Claude backstop model definition
DEFAULT_BACKSTOP_REGEX = (
    r'_CLAUDE_BACKSTOP_MODEL\s*=\s*os\.environ\.get\(\s*"CLAUDE_BACKSTOP_MODEL"\s*,\s*"([^"]+)"'
)

# Expected substrings within the default backstop model name
DEFAULT_BACKSTOP_MODEL_SUBSTR = "haiku"
DISALLOWED_BACKSTOP_MODEL_SUBSTR = "opus"

# Identifiers used to verify ordering of cascade vs. Claude backstop
FREE_PARALLEL_RACE_STR = "_call_parallel_race"
CLAUDE_BACKSTOP_STR = "CLAUDE backstop"

# Regexes for Claude backstop gating logic and cheap tier documentation
CLAUDE_BACKSTOP_GUARD_REGEX = r'if not result and \(tier == "hard" or tier == "auto"\)'
CHEAP_TIER_DOC_REGEX = r'"cheap".{0,80}free cascade only'

# Regex to ensure employee team defaults to disallow paid APIs
ALLOW_PAID_APIS_REGEX = r'ALLOW_PAID_APIS[^\n]*=\s*False'

# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
LLM_COMMON = ROOT / ".github" / "scripts" / "llm_common.py"
AGENT_TEAM = ROOT / ".github" / "scripts" / "agent_team.py"


def _src(p: Path) -> str:
    return p.read_text()


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_default_backstop_is_the_cheapest_claude_tier():
    """The default Claude backstop must be Haiku (cheap), never Opus/Sonnet."""
    src = _src(LLM_COMMON)
    m = re.search(DEFAULT_BACKSTOP_REGEX, src)
    assert m, "default backstop model not found"
    default = m.group(1)
    assert (
        DEFAULT_BACKSTOP_MODEL_SUBSTR in default.lower()
    ), f"default backstop must be Haiku-class, got {default!r}"
    assert (
        DISALLOWED_BACKSTOP_MODEL_SUBSTR not in default.lower()
    ), "Opus must never be the default backstop"


def test_routing_tries_free_cascade_first():
    """Tier 1 of llm_routed must be the FREE cascade (paid never runs first)."""
    src = _src(LLM_COMMON)
    free_pos = src.find(FREE_PARALLEL_RACE_STR)
    claude_pos = src.find(CLAUDE_BACKSTOP_STR)
    assert free_pos != -1 and claude_pos != -1
    assert free_pos < claude_pos, "free cascade must be tried before the Claude backstop"


def test_claude_backstop_is_last_resort_only():
    """Claude tier must be gated to 'hard'/'auto' as a last resort, not default 'cheap'."""
    src = _src(LLM_COMMON)
    assert re.search(CLAUDE_BACKSTOP_GUARD_REGEX, src), (
        "Claude backstop must be gated on tier hard/auto AND prior failure"
    )
    assert re.search(CHEAP_TIER_DOC_REGEX, src, re.DOTALL | re.IGNORECASE), (
        "cheap tier must be free-cascade-only"
    )


def test_employees_are_zero_spend_by_default():
    """The hourly employee team must default to zero paid spend."""
    src = _src(AGENT_TEAM)
    assert re.search(ALLOW_PAID_APIS_REGEX, src), (
        "employees must default ALLOW_PAID_APIS = False"
    )


# -------------------------------------------------------------------------
# Additional edge‑case tests
# -------------------------------------------------------------------------

def test_default_backstop_model_name_format():
    """The captured backstop model name should consist only of allowed characters."""
    src = _src(LLM_COMMON)
    m = re.search(DEFAULT_BACKSTOP_REGEX, src)
    assert m, "default backstop model not found"
    model_name = m.group(1)
    # Allowed characters: lowercase letters, digits, hyphens, underscores
    assert re.fullmatch(r"[a-z0-9_-]+", model_name), (
        f"Backstop model name contains invalid characters: {model_name!r}"
    )


def test_allow_paid_apis_flag_occurs_once():
    """ALLOW_PAID_APIS should be set to False exactly once in agent_team.py."""
    src = _src(AGENT_TEAM)
    matches = list(re.finditer(ALLOW_PAID_APIS_REGEX, src))
    assert matches, "ALLOW_PAID_APIS flag not found"
    assert len(matches) == 1, f"ALLOW_PAID_APIS flag should appear once, found {len(matches)} times"


def test_claude_backstop_guard_is_precise():
    """The guard clause for Claude backstop must match the expected pattern exactly."""
    src = _src(LLM_COMMON)
    # Ensure there is a line that contains the guard regex without extra logical operators
    guard_lines = [ln for ln in src.splitlines() if re.search(CLAUDE_BACKSTOP_GUARD_REGEX, ln)]
    assert guard_lines, "Claude backstop guard clause not found"
    # Verify that the guard line does not contain unintended additional conditions
    for line in guard_lines:
        # After the guard condition, there should be a colon and the block start
        assert re.search(rf"{CLAUDE_BACKSTOP_GUARD_REGEX}\s*:", line), (
            f"Guard clause format unexpected in line: {line!r}"
        )