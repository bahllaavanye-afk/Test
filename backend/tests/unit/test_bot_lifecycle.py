"""Unit tests for the autonomous bot‑lifecycle policy (pure, no DB).

The 'employee using Options Alpha' loop: disable proven losers, re‑enable
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

# Constants extracted from test cases
NOISE_TRADES = 3
PROFITABLE_TRADES = 20
PROFITABLE_PNL = 800.0
PROFITABLE_WR = 0.7
ARCHIVED_TRADES = 20
ARCHIVED_PNL = 999.0
ARCHIVED_WR = 0.9
TEMPLATE_COUNT = 10
TEMPLATE_PREFIX = "tpl_"

LOSER = "loser"
YOUNG = "young"
LUMPY = "lumpy"
COMEBACK = "comeback"
ARCHIVED = "archived"
WINNER = "winner"


def make_bot_stats(
    name: str,
    *,
    enabled: bool = True,
    archived: bool = False,
    trades: int = 0,
    pnl: float = 0.0,
    win_rate: float | None = None,
) -> BotStats:
    """Factory helper that builds a :class:`BotStats` instance for tests."""
    return BotStats(
        bot_id=name,
        name=name,
        is_enabled=enabled,
        is_archived=archived,
        trades=trades,
        total_pnl=pnl,
        win_rate=win_rate,
    )


def test_losing_bot_with_evidence_is_disabled() -> None:
    actions = decide_bot_actions(
        [make_bot_stats(LOSER, trades=MIN_TRADES_TO_JUDGE, pnl=-300.0, win_rate=0.30)],
        [],
    )
    assert [b.name for b in actions["disable"]] == [LOSER]


def test_losing_bot_without_evidence_is_left_alone() -> None:
    # NOISE_TRADES is noise, not evidence — never churn on it.
    actions = decide_bot_actions(
        [make_bot_stats(YOUNG, trades=NOISE_TRADES, pnl=-500.0, win_rate=0.0)],
        [],
    )
    assert actions["disable"] == []


def test_losing_pnl_but_decent_win_rate_survives() -> None:
    # Positive expectancy profiles (few big wins) shouldn't die on win rate alone.
    actions = decide_bot_actions(
        [
            make_bot_stats(
                LUMPY,
                trades=MIN_TRADES_TO_JUDGE,
                pnl=-50.0,
                win_rate=DISABLE_WIN_RATE + 0.05,
            )
        ],
        [],
    )
    assert actions["disable"] == []


def test_recovered_disabled_bot_is_promoted() -> None:
    actions = decide_bot_actions(
        [
            make_bot_stats(
                COMEBACK,
                enabled=False,
                trades=MIN_TRADES_TO_PROMOTE,
                pnl=120.0,
                win_rate=0.6,
            )
        ],
        [],
    )
    assert [b.name for b in actions["enable"]] == [COMEBACK]


def test_archived_bots_are_never_touched() -> None:
    actions = decide_bot_actions(
        [
            make_bot_stats(
                ARCHIVED,
                enabled=False,
                archived=True,
                trades=ARCHIVED_TRADES,
                pnl=ARCHIVED_PNL,
                win_rate=ARCHIVED_WR,
            )
        ],
        [],
    )
    assert actions["enable"] == [] and actions["disable"] == []


def test_template_creation_is_bounded() -> None:
    actions = decide_bot_actions(
        [],
        [f"{TEMPLATE_PREFIX}{i}" for i in range(TEMPLATE_COUNT)],
    )
    assert len(actions["create"]) == MAX_CREATES_PER_RUN


def test_profitable_enabled_bot_untouched() -> None:
    actions = decide_bot_actions(
        [
            make_bot_stats(
                WINNER,
                trades=PROFITABLE_TRADES,
                pnl=PROFITABLE_PNL,
                win_rate=PROFITABLE_WR,
            )
        ],
        [],
    )
    assert actions["disable"] == [] and actions["enable"] == []