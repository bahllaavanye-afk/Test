"""
Regression test for the /qe risk Slack command.

CircuitBreaker.current_drawdown is a @property — a prior version called it as
current_drawdown(), which raises `TypeError: 'float' object is not callable`.
This locks in that _risk_blocks reads the property correctly and never crashes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class _FakeState:
    value = "NORMAL"


class _FakeBreaker:
    """Mimics CircuitBreaker: current_drawdown is a property, not a method."""
    state = _FakeState()

    @property
    def current_drawdown(self) -> float:
        return 0.0342


class _FakeRiskManager:
    global_breaker = _FakeBreaker()
    max_drawdown_pct = 0.10


@pytest.mark.asyncio
async def test_risk_blocks_reads_property(monkeypatch):
    import app.main as main_mod
    from app.tasks import slack_handler

    # Attach a fake risk manager to app.state.
    main_mod.app.state.risk_manager = _FakeRiskManager()

    blocks = await slack_handler._risk_blocks()
    text = blocks[0]["text"]["text"]
    # Property accessed (not called) → real percentage rendered, no TypeError.
    assert "3.4%" in text
    assert "NORMAL" in text
    assert "Unavailable" not in text


@pytest.mark.asyncio
async def test_risk_blocks_handles_missing_manager(monkeypatch):
    import app.main as main_mod
    from app.tasks import slack_handler

    main_mod.app.state.risk_manager = None
    blocks = await slack_handler._risk_blocks()
    text = blocks[0]["text"]["text"]
    assert "not yet attached" in text
