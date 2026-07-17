"""Enforce the LLM cost policy: Claude is a RARE backstop, not per-task.

Answers "is Claude used for each task?" — it must not be. These tests lock the
invariant so a future edit can't silently make every employee call the paid
tier and drain the prepaid balance. Static assertions over the shipped config
(no network, no import of heavy deps).
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, validator

ROOT = Path(__file__).resolve().parents[3]
LLM_COMMON = ROOT / ".github" / "scripts" / "llm_common.py"
AGENT_TEAM = ROOT / ".github" / "scripts" / "slack_agent_team.py"


def _src(p: Path) -> str:
    return p.read_text()


def test_default_backstop_is_the_cheapest_claude_tier():
    """The default Claude backstop must be Haiku (cheap), never Opus/Sonnet."""
    src = _src(LLM_COMMON)
    m = re.search(r'_CLAUDE_BACKSTOP_MODEL\s*=\s*os\.environ\.get\(\s*"CLAUDE_BACKSTOP_MODEL"\s*,\s*"([^"]+)"', src)
    assert m, "default backstop model not found"
    default = m.group(1)
    assert "haiku" in default.lower(), f"default backstop must be Haiku-class, got {default!r}"
    assert "opus" not in default.lower(), "Opus must never be the default backstop"


def test_routing_tries_free_cascade_first():
    """Tier 1 of llm_routed must be the FREE cascade (paid never runs first)."""
    src = _src(LLM_COMMON)
    # The free parallel race is tier 1 and appears before any OpenRouter/Claude call.
    free_pos = src.find("_call_parallel_race")
    claude_pos = src.find("CLAUDE backstop")
    assert free_pos != -1 and claude_pos != -1
    assert free_pos < claude_pos, "free cascade must be tried before the Claude backstop"


def test_claude_backstop_is_last_resort_only():
    """Claude tier must be gated to 'hard'/'auto' as a last resort, not default 'cheap'."""
    src = _src(LLM_COMMON)
    # The Claude tier guard must require tier hard/auto AND not-yet-resolved.
    assert re.search(r'if not result and \(tier == "hard" or tier == "auto"\)', src), \
        "Claude backstop must be gated on tier hard/auto AND prior failure"
    # "cheap" tier must be documented as free-only.
    assert re.search(r'"cheap".{0,80}free cascade only', src, re.DOTALL | re.IGNORECASE), \
        "cheap tier must be free-cascade-only"


def test_employees_are_zero_spend_by_default():
    """The hourly employee team must default to zero paid spend."""
    src = _src(AGENT_TEAM)
    assert re.search(r'ALLOW_PAID_APIS[^\n]*=\s*False', src), \
        "employees must default ALLOW_PAID_APIS = False"


class LLMBackstopConfig(BaseModel):
    """Configuration schema for the Claude backstop model used in routing.

    Attributes
    ----------
    model_name: str
        Name of the Claude backstop model. Must be a Haiku variant.
    tier: str
        Tier that enables the backstop. Accepted values are ``"hard"`` or ``"auto"``.
    allowed: bool
        Flag indicating whether the backstop is permitted.
    """
    model_name: str = Field(
        ...,
        description="Name of the Claude backstop model.",
        example="claude-3-haiku-20240307",
    )
    tier: str = Field(
        ...,
        description="Tier for the backstop usage.",
        example="hard",
    )
    allowed: bool = Field(
        ...,
        description="Whether the backstop is allowed.",
        example=True,
    )

    @validator("model_name")
    def ensure_haiku(cls, v: str) -> str:
        """Validate that the model name contains 'haiku' (case‑insensitive)."""
        if "haiku" not in v.lower():
            raise ValueError("Backstop model must be a Haiku variant")
        return v

    @validator("tier")
    def validate_tier(cls, v: str) -> str:
        """Validate that the tier is either 'hard' or 'auto'."""
        if v not in {"hard", "auto"}:
            raise ValueError('Tier must be "hard" or "auto"')
        return v