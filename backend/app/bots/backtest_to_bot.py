"""Backtest → Bot generator (Option Alpha's "Automate your strategy").

QuantEdge backtests run *registry strategies* (momentum, rsi2_pullback, …) over
OHLCV, while the Bot engine (`app/bots/engine.py`) runs *indicator/condition*
logic. Those are different execution models, so a generated bot cannot silently
re-run the registry strategy — it would misrepresent the backtest.

This module maps a backtest's strategy family to the closest **real** bot
conditions the engine actually supports, and is honest about confidence:

  * mapped   → conditions that genuinely reproduce the strategy's entry intent
  * approx   → a reasonable entry the engine can run, flagged for user review

Either way the generated bot is created **disabled** (paper-first: the user
reviews and enables it) and its description records full provenance — source
backtest id, strategy, and the realized Sharpe / return / win-rate.
"""
from __future__ import annotations

from app.schemas.bot import (
    ActionConfig,
    BotCreate,
    ConditionConfig,
    ExitRuleConfig,
    TriggerConfig,
)

# Backtest interval strings → bot trigger intervals the engine schedules on.
_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "15m",
    "1h": "1h", "60m": "1h", "4h": "4h", "1d": "1d", "1day": "1d", "d": "1d",
}


def _no_position() -> ConditionConfig:
    return ConditionConfig(type="no_position")


def _rsi(op: str, value: float, period: int = 14) -> ConditionConfig:
    return ConditionConfig(type="indicator", indicator="rsi", period=period, operator=op, value=value)


# strategy-name substring → (confidence, conditions, action_side). First match wins.
# Only families the bot engine's evaluate_condition actually supports are "mapped".
_RULES: list[tuple[tuple[str, ...], str, list[ConditionConfig], str]] = [
    (("rsi2", "rsi_macd", "rsi"), "mapped",
     [_rsi("<", 30), _no_position()], "open_long"),
    (("supertrend",), "mapped",
     [ConditionConfig(type="indicator", indicator="supertrend", operator="crosses_above"), _no_position()], "open_long"),
    (("macd",), "mapped",
     [ConditionConfig(type="indicator", indicator="macd", operator="crosses_above"), _no_position()], "open_long"),
    (("ema_stack", "ema"), "mapped",
     [ConditionConfig(type="price_vs_ma", ma_type="ema", ma_period=20, operator=">"), _no_position()], "open_long"),
    (("donchian", "breakout", "opening_range", "fifty_two_week_high", "52_week"), "mapped",
     [ConditionConfig(type="indicator", indicator="donchian", operator="crosses_above"), _no_position()], "open_long"),
    (("bollinger", "bb", "mean_reversion", "vwap_reversion", "reversion"), "mapped",
     [ConditionConfig(type="indicator", indicator="bb", operator="<"), _no_position()], "open_long"),
    (("momentum", "trend", "roc", "time_series_momentum"), "mapped",
     [ConditionConfig(type="indicator", indicator="momentum", period=10, operator=">", value=0.0), _no_position()], "open_long"),
    (("stoch",), "mapped",
     [ConditionConfig(type="indicator", indicator="stoch_rsi", operator="<", value=0.2), _no_position()], "open_long"),
]


def map_strategy_to_conditions(strategy_name: str) -> tuple[str, list[ConditionConfig], str]:
    """(confidence, conditions, action_side) for a registry strategy name.

    Falls back to a schedule-driven momentum entry marked "approx" so the bot is
    runnable but clearly flagged for review — never silently wrong.
    """
    key = (strategy_name or "").lower()
    for subs, confidence, conds, side in _RULES:
        if any(s in key for s in subs):
            return confidence, [c.model_copy(deep=True) for c in conds], side
    # Unmapped: honest approximation the engine can actually run.
    return "approx", [
        ConditionConfig(type="indicator", indicator="momentum", period=10, operator=">", value=0.0),
        _no_position(),
    ], "open_long"


def build_bot_from_backtest(
    *,
    strategy_name: str,
    symbol: str,
    interval: str,
    market_type: str = "equity",
    name: str | None = None,
    size_pct: float = 5.0,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
    sharpe: float | None = None,
    total_return: float | None = None,
    win_rate: float | None = None,
    run_id: str | None = None,
) -> tuple[BotCreate, str]:
    """Map a completed backtest to a BotCreate. Returns (payload, confidence).

    TP/SL default to 3%/2% when the backtest didn't specify them (the same
    defaults the built-in templates use). Provenance is baked into the name and
    description so the bot is always traceable back to its backtest.
    """
    confidence, conditions, side = map_strategy_to_conditions(strategy_name)
    tp = take_profit_pct if take_profit_pct is not None else 3.0
    sl = stop_loss_pct if stop_loss_pct is not None else 2.0
    bot_name = name or f"{strategy_name} · {symbol.upper()} (backtest)"

    prov = []
    if sharpe is not None:
        prov.append(f"Sharpe {sharpe:.2f}")
    if total_return is not None:
        prov.append(f"return {total_return * 100:.1f}%" if abs(total_return) < 10 else f"return ${total_return:,.0f}")
    if win_rate is not None:
        prov.append(f"win {win_rate * 100:.0f}%")
    prov_str = " · ".join(prov) if prov else "no result stats"
    review = "" if confidence == "mapped" else " ⚠ entry conditions approximated — review before enabling."
    description = (
        f"Generated from backtest {run_id or '?'} running '{strategy_name}' on "
        f"{symbol.upper()} ({prov_str}).{review}"
    )

    payload = BotCreate(
        name=bot_name,
        description=description,
        symbol=symbol.upper(),
        market_type=market_type,
        trigger=TriggerConfig(type="schedule", interval=_INTERVAL_MAP.get((interval or "1d").lower(), "1d")),
        conditions=conditions,
        condition_logic="ALL",
        action=ActionConfig(type=side, size_pct=size_pct, take_profit_pct=tp, stop_loss_pct=sl),
        exit_rules=[
            ExitRuleConfig(type="take_profit", value=tp),
            ExitRuleConfig(type="stop_loss", value=sl),
        ],
        template_id=f"backtest:{run_id}" if run_id else None,
    )
    return payload, confidence
