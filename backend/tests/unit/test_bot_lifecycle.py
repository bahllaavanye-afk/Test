"""Unit tests for the autonomous bot-lifecycle policy (pure, no DB).

The 'employee using Options Alpha' loop: disable proven losers, re-enable
recovered bots, grow the fleet from templates — with conservative evidence
thresholds so bots aren't churned on noise.
"""
from __future__ import annotations

from app.bots.lifecycle import (
    DISABLE_WIN_RATE,
    MAX_CREATES_PER_RUN,
    MIN_TRADES_TO_JUDGE,
    MIN_TRADES_TO_PROMOTE,
    BotStats,
    decide_bot_actions,
)


def _s(name, enabled=True, archived=False, trades=0, pnl=0.0, wr=None):
    return BotStats(bot_id=name, name=name, is_enabled=enabled,
                    is_archived=archived, trades=trades, total_pnl=pnl, win_rate=wr)


def test_losing_bot_with_evidence_is_disabled():
    a = decide_bot_actions([_s("loser", trades=MIN_TRADES_TO_JUDGE, pnl=-300.0, wr=0.30)], [])
    assert [b.name for b in a["disable"]] == ["loser"]


def test_losing_bot_without_evidence_is_left_alone():
    # 3 trades is noise, not evidence — never churn on it.
    a = decide_bot_actions([_s("young", trades=3, pnl=-500.0, wr=0.0)], [])
    assert a["disable"] == []


def test_losing_pnl_but_decent_win_rate_survives():
    # Positive expectancy profiles (few big wins) shouldn't die on win rate alone.
    a = decide_bot_actions(
        [_s("lumpy", trades=MIN_TRADES_TO_JUDGE, pnl=-50.0, wr=DISABLE_WIN_RATE + 0.05)], []
    )
    assert a["disable"] == []


def test_recovered_disabled_bot_is_promoted():
    a = decide_bot_actions(
        [_s("comeback", enabled=False, trades=MIN_TRADES_TO_PROMOTE, pnl=120.0, wr=0.6)], []
    )
    assert [b.name for b in a["enable"]] == ["comeback"]


def test_archived_bots_are_never_touched():
    a = decide_bot_actions(
        [_s("archived", enabled=False, archived=True, trades=20, pnl=999.0, wr=0.9)], []
    )
    assert a["enable"] == [] and a["disable"] == []


def test_template_creation_is_bounded():
    a = decide_bot_actions([], [f"tpl_{i}" for i in range(10)])
    assert len(a["create"]) == MAX_CREATES_PER_RUN


def test_profitable_enabled_bot_untouched():
    a = decide_bot_actions([_s("winner", trades=20, pnl=800.0, wr=0.7)], [])
    assert a["disable"] == [] and a["enable"] == []
