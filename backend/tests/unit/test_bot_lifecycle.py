"""Unit tests for the autonomous bot-lifecycle policy (pure, no DB).

The 'employee using Options Alpha' loop: disable proven losers, re-enable
recovered bots, grow the fleet from templates — with conservative evidence
thresholds so bots aren't churned on noise.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, validator

from app.bots.lifecycle import (
    DISABLE_WIN_RATE,
    MAX_CREATES_PER_RUN,
    MIN_TRADES_TO_JUDGE,
    MIN_TRADES_TO_PROMOTE,
    BotStats,
    decide_bot_actions,
)


def _s(name, enabled=True, archived=False, trades=0, pnl=0.0, wr=None):
    return BotStats(
        bot_id=name,
        name=name,
        is_enabled=enabled,
        is_archived=archived,
        trades=trades,
        total_pnl=pnl,
        win_rate=wr,
    )


class BotActionResult(BaseModel):
    """Schema representing the actions returned by ``decide_bot_actions``."""

    disable: List[BotStats] = Field(
        ...,
        description="List of bots that should be disabled.",
        example=[],
    )
    enable: List[BotStats] = Field(
        ...,
        description="List of bots that should be enabled.",
        example=[],
    )
    create: List[str] = Field(
        ...,
        description="Names of templates from which new bots should be created.",
        example=[],
    )

    @validator("disable", "enable", each_item=True)
    def check_bot_stats(cls, value):
        """Ensure that win_rate, if provided, is within a realistic range."""
        if value.win_rate is not None and not (0.0 <= value.win_rate <= 1.0):
            raise ValueError("win_rate must be between 0 and 1")
        return value

    @validator("create", each_item=True)
    def check_template_name(cls, value):
        """Template names must be non‑empty strings."""
        if not isinstance(value, str) or not value:
            raise ValueError("template name must be a non-empty string")
        return value


def test_losing_bot_with_evidence_is_disabled():
    a = decide_bot_actions([_s("loser", trades=MIN_TRADES_TO_JUDGE, pnl=-300.0, wr=0.30)], [])
    result = BotActionResult(**a)
    assert [b.name for b in result.disable] == ["loser"]


def test_losing_bot_without_evidence_is_left_alone():
    # 3 trades is noise, not evidence — never churn on it.
    a = decide_bot_actions([_s("young", trades=3, pnl=-500.0, wr=0.0)], [])
    result = BotActionResult(**a)
    assert result.disable == []


def test_losing_pnl_but_decent_win_rate_survives():
    # Positive expectancy profiles (few big wins) shouldn't die on win rate alone.
    a = decide_bot_actions(
        [_s("lumpy", trades=MIN_TRADES_TO_JUDGE, pnl=-50.0, wr=DISABLE_WIN_RATE + 0.05)], []
    )
    result = BotActionResult(**a)
    assert result.disable == []


def test_recovered_disabled_bot_is_promoted():
    a = decide_bot_actions(
        [_s("comeback", enabled=False, trades=MIN_TRADES_TO_PROMOTE, pnl=120.0, wr=0.6)], []
    )
    result = BotActionResult(**a)
    assert [b.name for b in result.enable] == ["comeback"]


def test_archived_bots_are_never_touched():
    a = decide_bot_actions(
        [_s("archived", enabled=False, archived=True, trades=20, pnl=999.0, wr=0.9)], []
    )
    result = BotActionResult(**a)
    assert result.enable == [] and result.disable == []


def test_template_creation_is_bounded():
    a = decide_bot_actions([], [f"tpl_{i}" for i in range(10)])
    result = BotActionResult(**a)
    assert len(result.create) == MAX_CREATES_PER_RUN


def test_profitable_enabled_bot_untouched():
    a = decide_bot_actions([_s("winner", trades=20, pnl=800.0, wr=0.7)], [])
    result = BotActionResult(**a)
    assert result.disable == [] and result.enable == []