"""Tests for the free-LLM gateway and autonomous employee reasoning.

These run with NO provider keys set, so they verify honest degradation:
the gateway returns None and callers never fabricate output.
"""
import pytest

from app.llm import gateway
from app.llm.employees import ROLE_PROMPTS, TASK_ROUTING, reason_about_task


def test_providers_configured_empty(monkeypatch):
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_1", "DEEPSEEK_API_KEY",
                "DEEPSEEK_API_KEY_1", "GEMINI_API_KEY", "GEMINI_API_KEY_1"):
        monkeypatch.delenv(var, raising=False)
    assert gateway.providers_configured() == []


def test_providers_configured_detects_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert "groq" in gateway.providers_configured()


def test_resolve_key_numbered_suffix(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_1", "numbered")
    assert gateway._resolve_key("GEMINI_API_KEY") == "numbered"


@pytest.mark.asyncio
async def test_complete_returns_none_without_keys(monkeypatch):
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_1", "DEEPSEEK_API_KEY",
                "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_KEY_3",
                "GEMINI_API_KEY", "GEMINI_API_KEY_1"):
        monkeypatch.delenv(var, raising=False)
    result = await gateway.complete([{"role": "user", "content": "hi"}])
    assert result is None


@pytest.mark.asyncio
async def test_reason_about_task_unavailable_without_llm(monkeypatch):
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_1", "DEEPSEEK_API_KEY",
                "DEEPSEEK_API_KEY_1", "GEMINI_API_KEY", "GEMINI_API_KEY_1"):
        monkeypatch.delenv(var, raising=False)
    out = await reason_about_task("risk_check", {"status": "ok", "regime": "bull"})
    assert out["llm"] == "unavailable"
    # Honest degradation: no fabricated analysis/recommendations.
    assert "analysis" not in out or not out.get("analysis")


def test_complete_sync_returns_none_without_keys(monkeypatch):
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_1", "DEEPSEEK_API_KEY",
                "DEEPSEEK_API_KEY_1", "GEMINI_API_KEY", "GEMINI_API_KEY_1"):
        monkeypatch.delenv(var, raising=False)
    assert gateway.complete_sync([{"role": "user", "content": "x"}]) is None


def test_every_task_route_has_a_role():
    for task_type, role in TASK_ROUTING.items():
        assert role in ROLE_PROMPTS, f"{task_type} routes to unknown role {role}"


@pytest.mark.asyncio
async def test_reason_parses_llm_json(monkeypatch):
    async def fake_complete(messages, max_tokens=400, agent=None):
        return '{"analysis": "Sharpe is weak.", "recommendations": ["disable laggards"]}'

    monkeypatch.setattr("app.llm.employees.complete", fake_complete)
    out = await reason_about_task("evaluate_strategies", {"poor_performers": []})
    assert out["agent"] == "strategy_agent"
    assert out["analysis"] == "Sharpe is weak."
    assert out["recommendations"] == ["disable laggards"]
