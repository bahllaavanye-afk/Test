"""
Unit tests for the free LLM router's multi-key rotation and routing.

These lock in the fixes for the throughput-imbalance bugs:
  - numbered keys (GROQ_API_KEY_1/_2/...) must actually be discovered + used
  - rotation must spread calls across keys
  - call_routed's preferred provider names must exist in PROVIDERS (the old code
    referenced "groq_llama"/"together_llama" which silently fell through)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.tasks.free_llm_router as router
from app.tasks.free_llm_router import (
    PROVIDERS, _keys_for, _next_key, _record_usage, get_throughput_report,
    reset_usage, available_providers, available_keys,
)


@pytest.fixture(autouse=True)
def _clean_router(monkeypatch):
    # Clear all known provider env keys + numbered variants, and reset state.
    for p in PROVIDERS:
        for label in [p.env_key] + [f"{p.env_key}_{i}" for i in range(1, 13)]:
            monkeypatch.delenv(label, raising=False)
    router._RR_INDEX.clear()
    reset_usage()
    yield
    router._RR_INDEX.clear()
    reset_usage()


class TestKeyDiscovery:
    def test_base_key_discovered(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "base")
        assert _keys_for("GROQ_API_KEY") == [("GROQ_API_KEY", "base")]

    def test_numbered_keys_discovered(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY_1", "k1")
        monkeypatch.setenv("GROQ_API_KEY_2", "k2")
        monkeypatch.setenv("GROQ_API_KEY_3", "k3")
        keys = _keys_for("GROQ_API_KEY")
        assert [v for _, v in keys] == ["k1", "k2", "k3"]

    def test_base_plus_numbered(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "base")
        monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
        keys = _keys_for("GEMINI_API_KEY")
        assert ("GEMINI_API_KEY", "base") in keys
        assert ("GEMINI_API_KEY_2", "k2") in keys

    def test_disabled_and_empty_skipped(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "disabled")
        monkeypatch.setenv("GROQ_API_KEY_1", "")
        monkeypatch.setenv("GROQ_API_KEY_2", "good")
        assert _keys_for("GROQ_API_KEY") == [("GROQ_API_KEY_2", "good")]

    def test_duplicate_values_deduped(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "same")
        monkeypatch.setenv("GROQ_API_KEY_1", "same")
        assert _keys_for("GROQ_API_KEY") == [("GROQ_API_KEY", "same")]


class TestRotation:
    def test_round_robin_spreads(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY_1", "k1")
        monkeypatch.setenv("GROQ_API_KEY_2", "k2")
        monkeypatch.setenv("GROQ_API_KEY_3", "k3")
        picks = [_next_key("GROQ_API_KEY")[0] for _ in range(6)]
        # Cycles through all three, twice.
        assert picks == [
            "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3",
            "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3",
        ]

    def test_next_key_none_when_unset(self):
        assert _next_key("NONEXISTENT_KEY") is None


class TestUsageTracking:
    def test_record_and_report(self):
        _record_usage("GROQ_API_KEY_1", 100)
        _record_usage("GROQ_API_KEY_1", 50)
        _record_usage("GEMINI_API_KEY_2", 380)
        report = {r["key"]: r for r in get_throughput_report()}
        assert report["GROQ_API_KEY_1"] == {"key": "GROQ_API_KEY_1", "calls": 2, "tokens": 150}
        assert report["GEMINI_API_KEY_2"]["calls"] == 1


class TestAvailability:
    def test_available_providers_and_keys(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY_1", "k1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or")
        provs = available_providers()
        assert "groq" in provs
        assert "openrouter" in provs
        keys = available_keys()
        assert "GROQ_API_KEY_1" in keys
        assert "OPENROUTER_API_KEY" in keys


class TestRoutedNamesExist:
    def test_openrouter_provider_registered(self):
        assert any(p.name == "openrouter" for p in PROVIDERS)

    def test_call_routed_preferred_names_all_valid(self):
        """Regression: every name call_routed prefers must exist in PROVIDERS."""
        valid = {p.name for p in PROVIDERS}
        # Mirror the lists in call_routed.
        for names in (
            ["cerebras", "groq", "gemini"],
            ["gemini", "gemini_thinking", "together", "groq"],
            ["groq", "together", "cerebras", "openrouter", "gemini", "nvidia_nim"],
        ):
            for n in names:
                assert n in valid, f"call_routed references unknown provider '{n}'"
