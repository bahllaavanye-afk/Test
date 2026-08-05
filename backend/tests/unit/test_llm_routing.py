"""Cost-tiered routing ladder (`llm_routed`): FREE → OpenRouter open-mid → Claude.

Escalation happens only on failure and `tier` caps how high it may climb, so Claude
stays a rare backstop. These are pure unit tests — every tier is monkeypatched, no
network and no real keys.
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_common as L  # noqa: E402

_KW = dict(use_cache=False, inject_company_context=False)


def test_cheap_tier_uses_free_only(monkeypatch):
    seen = {"or": False, "cl": False}
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: ("FREE", "groq"))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: seen.__setitem__("or", True) or "OPEN")
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: seen.__setitem__("cl", True) or "CLAUDE")
    out = L.llm_routed("hi", tier="cheap", **_KW)
    assert out == "FREE"
    assert seen == {"or": False, "cl": False}  # never escalated


def test_free_wins_even_when_higher_tiers_available(monkeypatch):
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: ("FREE", "gemini"))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: "OPEN")
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: "CLAUDE")
    assert L.llm_routed("hi", tier="hard", **_KW) == "FREE"


def test_escalates_to_openrouter_when_free_fails(monkeypatch):
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: "OPEN")
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: "CLAUDE")
    assert L.llm_routed("hi", tier="mid", **_KW) == "OPEN"


def test_hard_tier_falls_through_to_claude(monkeypatch):
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: "CLAUDE")
    assert L.llm_routed("hi", tier="hard", **_KW) == "CLAUDE"


def test_cheap_never_reaches_claude_even_if_free_fails(monkeypatch):
    seen = {"cl": False}
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: seen.__setitem__("cl", True) or "CLAUDE")
    out = L.llm_routed("hi", tier="cheap", **_KW)
    assert seen["cl"] is False
    assert out.startswith("[LLM unavailable")


def test_auto_tier_uses_claude_as_last_resort(monkeypatch):
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: "CLAUDE")
    assert L.llm_routed("hi", tier="auto", **_KW) == "CLAUDE"


# ── plain llm() must also escalate when the whole free cascade is down ─────────

def test_llm_escalates_to_openrouter_when_free_down(monkeypatch):
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: "OPEN")
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: "CLAUDE")
    out = L.llm("hi", use_cache=False, inject_company_context=False)
    assert out == "OPEN"


def test_llm_escalates_to_claude_when_free_and_open_down(monkeypatch):
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: "CLAUDE")
    out = L.llm("hi", use_cache=False, inject_company_context=False)
    assert out == "CLAUDE"


def test_llm_returns_sentinel_when_all_tiers_down(monkeypatch):
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: None)
    out = L.llm("hi", use_cache=False, inject_company_context=False)
    assert out.startswith("[LLM unavailable")


def test_env_keys_collects_numbered_variants_deduped(monkeypatch):
    for n in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "a")
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "a")   # dup → collapsed
    monkeypatch.setenv("OPENROUTER_API_KEY_3", "b")
    assert L._env_keys("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2",
                       "OPENROUTER_API_KEY_3") == ["a", "b"]
    # empty / "disabled" are skipped
    monkeypatch.setenv("X_K", "")
    monkeypatch.setenv("X_K2", "disabled")
    assert L._env_keys("X_K", "X_K2") == []


# ── Edge case tests ────────────────────────────────────────────────────────

def test_llm_routed_none_prompt_returns_sentinel(monkeypatch):
    """When the prompt is None, routing should gracefully return the sentinel."""
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: None)
    out = L.llm_routed(None, tier="auto", **_KW)
    assert out.startswith("[LLM unavailable")


def test_llm_routed_empty_string_prompt_uses_free_if_available(monkeypatch):
    """Empty string prompts should still allow the FREE tier to succeed."""
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: ("FREE", None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: "OPEN")
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: "CLAUDE")
    out = L.llm_routed("", tier="cheap", **_KW)
    assert out == "FREE"


def test_llm_routed_invalid_tier_falls_back_to_auto(monkeypatch):
    """An unrecognised tier should be treated like 'auto' and still reach Claude."""
    monkeypatch.setattr(L, "_call_parallel_race", lambda *a, **k: (None, None))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: "CLAUDE")
    out = L.llm_routed("hi", tier="unknown", **_KW)
    assert out == "CLAUDE"


def test_env_keys_with_none_and_missing_vars(monkeypatch):
    """_env_keys should ignore None arguments and skip missing environment variables."""
    monkeypatch.delenv("MISSING_KEY", raising=False)
    # Provide one valid key and one None; the None should be ignored.
    monkeypatch.setenv("VALID_KEY", "val")
    result = L._env_keys(None, "VALID_KEY", "MISSING_KEY")
    assert result == ["val"]


def test_env_keys_off_by_one_duplicate_handling(monkeypatch):
    """Ensure that duplicate values across sequentially numbered env vars are deduped."""
    for n in ("API_KEY", "API_KEY_2", "API_KEY_3"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("API_KEY", "x")
    monkeypatch.setenv("API_KEY_2", "x")   # duplicate
    monkeypatch.setenv("API_KEY_3", "y")
    assert L._env_keys("API_KEY", "API_KEY_2", "API_KEY_3") == ["x", "y"]



# End of test suite.