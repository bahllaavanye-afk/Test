"""Cost-tiered routing ladder (`llm_routed`): FREE → OpenRouter open-mid → Claude.

Escalation happens only on failure and `tier` caps how high it may climb, so Claude
stays a rare backstop. These are pure unit tests — every tier is monkeypatched, no
network and no real keys.
"""
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable, Any

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_common as L  # noqa: E402

# Default keyword arguments for llm calls used across tests
_KW: Dict[str, Any] = dict(use_cache=False, inject_company_context=False)


def _make_parallel_race(result: Tuple[Any, Any]) -> Callable:
    """Factory returning a lambda that mimics ``_call_parallel_race``."""
    return lambda *a, **k: result


def _make_returner(flag: str, store: Dict[str, bool] | None = None) -> Callable:
    """Factory returning a lambda that optionally records a call and returns ``flag``."""
    def _inner(*a, **k):
        if store is not None:
            store[flag.lower()] = True
        return flag.upper()
    return _inner


def test_cheap_tier_uses_free_only(monkeypatch):
    """Cheap tier should never invoke higher‑tier providers."""
    seen: Dict[str, bool] = {"or": False, "cl": False}
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race(("FREE", "groq")))
    monkeypatch.setattr(L, "_call_openrouter", _make_returner("OPEN", seen))
    monkeypatch.setattr(L, "_call_claude", _make_returner("CLAUDE", seen))
    out = L.llm_routed("hi", tier="cheap", **_KW)
    assert out == "FREE"
    assert seen == {"or": False, "cl": False}  # never escalated


def test_free_wins_even_when_higher_tiers_available(monkeypatch):
    """Free tier should win regardless of higher‑tier availability."""
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race(("FREE", "gemini")))
    monkeypatch.setattr(L, "_call_openrouter", _make_returner("OPEN"))
    monkeypatch.setattr(L, "_call_claude", _make_returner("CLAUDE"))
    assert L.llm_routed("hi", tier="hard", **_KW) == "FREE"


def test_escalates_to_openrouter_when_free_fails(monkeypatch):
    """Mid tier must fall back to OpenRouter if FREE fails."""
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race((None, None)))
    monkeypatch.setattr(L, "_call_openrouter", _make_returner("OPEN"))
    monkeypatch.setattr(L, "_call_claude", _make_returner("CLAUDE"))
    assert L.llm_routed("hi", tier="mid", **_KW) == "OPEN"


def test_hard_tier_falls_through_to_claude(monkeypatch):
    """Hard tier should reach Claude after FREE and OpenRouter both return None."""
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race((None, None)))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", _make_returner("CLAUDE"))
    assert L.llm_routed("hi", tier="hard", **_KW) == "CLAUDE"


def test_cheap_never_reaches_claude_even_if_free_fails(monkeypatch):
    """Cheap tier must not call Claude even when FREE fails."""
    seen: Dict[str, bool] = {"cl": False}
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race((None, None)))
    monkeypatch.setattr(L, "_call_claude", _make_returner("CLAUDE", seen))
    out = L.llm_routed("hi", tier="cheap", **_KW)
    assert seen["cl"] is False
    assert out.startswith("[LLM unavailable")


def test_auto_tier_uses_claude_as_last_resort(monkeypatch):
    """Auto tier should eventually reach Claude if all lower tiers are unavailable."""
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race((None, None)))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", _make_returner("CLAUDE"))
    assert L.llm_routed("hi", tier="auto", **_KW) == "CLAUDE"


# ── plain llm() must also escalate when the whole free cascade is down ─────────

def test_llm_escalates_to_openrouter_when_free_down(monkeypatch):
    """llm() should fall back to OpenRouter when FREE providers are down."""
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race((None, None)))
    monkeypatch.setattr(L, "_call_openrouter", _make_returner("OPEN"))
    monkeypatch.setattr(L, "_call_claude", _make_returner("CLAUDE"))
    out = L.llm("hi", use_cache=False, inject_company_context=False)
    assert out == "OPEN"


def test_llm_escalates_to_claude_when_free_and_open_down(monkeypatch):
    """llm() should fall back to Claude when both FREE and OpenRouter are down."""
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race((None, None)))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", _make_returner("CLAUDE"))
    out = L.llm("hi", use_cache=False, inject_company_context=False)
    assert out == "CLAUDE"


def test_llm_returns_sentinel_when_all_tiers_down(monkeypatch):
    """When every tier fails, llm() must return a sentinel error message."""
    monkeypatch.setattr(L, "_call_parallel_race", _make_parallel_race((None, None)))
    monkeypatch.setattr(L, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(L, "_call_claude", lambda *a, **k: None)
    out = L.llm("hi", use_cache=False, inject_company_context=False)
    assert out.startswith("[LLM unavailable")


def test_env_keys_collects_numbered_variants_deduped(monkeypatch):
    """_env_keys should dedupe numbered environment variables and ignore empty/disabled entries."""
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