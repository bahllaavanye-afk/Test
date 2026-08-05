"""Pydantic v2 schemas for the Bot builder."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Constants for default values
DEFAULT_TRIGGER_INTERVAL = "5m"  # 1m|5m|15m|1h|4h|1d
DEFAULT_TRIGGER_DIRECTION = "above"  # above|below
DEFAULT_TRIGGER_INDICATOR_OPERATOR = "<"  # <|>|crosses_above|crosses_below

DEFAULT_CONDITION_PERIOD = 14
DEFAULT_CONDITION_OPERATOR = "<"  # < | > | == | != | crosses_above | crosses_below
DEFAULT_CONDITION_MA_TYPE = "sma"  # "sma" or "ema"

DEFAULT_OPTION_DTE = 30  # days to expiration target
DEFAULT_OPTION_RATIO = 1  # contracts per 1x of the spread

DEFAULT_ACTION_SIZE_PCT = 5.0

DEFAULT_EXIT_RULE_PERIOD = 14
DEFAULT_EXIT_RULE_OPERATOR = ">"  # for indicator based exit rules

DEFAULT_BOT_MARKET_TYPE = "equity"
DEFAULT_BACKTEST_MARKET_TYPE = "equity"
DEFAULT_BACKTEST_SIZE_PCT = 5.0

DEFAULT_CONDITION_LOGIC = "ALL"


class TriggerConfig(BaseModel):
    type: Literal["schedule", "price_cross", "indicator"]
    interval: str = DEFAULT_TRIGGER_INTERVAL
    price_level: float | None = None   # price_cross
    direction: str = DEFAULT_TRIGGER_DIRECTION
    indicator: str | None = None       # rsi|macd|bb|sma|ema
    indicator_period: int = 14
    indicator_operator: str = DEFAULT_TRIGGER_INDICATOR_OPERATOR
    indicator_value: float | None = None


class ConditionConfig(BaseModel):
    type: Literal[
        "indicator",
        "price_vs_ma",
        "pnl",
        "time_window",
        "position_exists",
        "no_position",
        "ml_signal",
        "regime",
    ]
    indicator: str | None = None
    period: int = DEFAULT_CONDITION_PERIOD
    operator: str = DEFAULT_CONDITION_OPERATOR
    value: float | None = None
    ma_period: int | None = None
    start_time: str | None = None   # "09:30" ET
    end_time: str | None = None     # "16:00" ET
    pnl_pct: float | None = None
    # EMA cross custom periods
    fast_period: int | None = None
    slow_period: int | None = None
    # price_vs_ma MA type
    ma_type: str | None = DEFAULT_CONDITION_MA_TYPE
    # Stochastic periods
    k_period: int | None = None
    d_period: int | None = None
    # Supertrend multiplier
    multiplier: float | None = None
    # ml_signal: the trained model must predict `direction` with confidence ≥ min_confidence
    # (OA-style decision: "IF ML says up with ≥65% confidence THEN ..."). Fails safe:
    # no trained model / inference error → the condition is simply False.
    direction: str | None = None        # "up" | "down"
    min_confidence: float | None = None  # default 0.65 in the engine
    # regime: fire only in the listed market regimes (OA-style "trade only when
    # VIX/trend says so"). trend ∈ {bear,sideways,bull}; vol ∈ {calm,stressed}.
    regimes: list[str] | None = None    # e.g. ["bull", "sideways"] or ["stressed"]


class OptionLeg(BaseModel):
    """One leg of a multi-leg options order (spread, condor, straddle, ...)."""
    side: Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    delta: float | None = None        # target delta for strike selection (0-1)
    strike: float | None = None       # explicit strike (overrides delta if set)
    dte: int = DEFAULT_OPTION_DTE
    ratio: int = DEFAULT_OPTION_RATIO


class ActionConfig(BaseModel):
    type: Literal[
        "open_long",
        "open_short",
        "close_position",
        "send_alert",
        "reduce_position",
        "open_option_spread",
    ]
    size_pct: float = DEFAULT_ACTION_SIZE_PCT
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    alert_message: str | None = None
    reduce_by_pct: float | None = None
    legs: list[OptionLeg] | None = None   # required for open_option_spread
    # Option Alpha-style per-bot safeguards (checked before opening a position).
    # None = unlimited. Bot won't open once a limit is reached; existing
    # positions must close (or the day roll over) before it opens again.
    max_open_positions: int | None = None    # OA "Position limit" (max open at once)
    max_daily_positions: int | None = None    # OA "Daily positions limit"


class ExitRuleConfig(BaseModel):
    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
    value: float | None = None       # pct for TP/SL/trailing
    hours: int | None = None         # time_exit
    indicator: str | None = None
    period: int = DEFAULT_EXIT_RULE_PERIOD
    operator: str = DEFAULT_EXIT_RULE_OPERATOR
    indicator_value: float | None = None


class BotCreate(BaseModel):
    name: str
    description: str = ""
    symbol: str
    market_type: str = DEFAULT_BOT_MARKET_TYPE
    trigger: TriggerConfig
    conditions: list[ConditionConfig] = []
    condition_logic: str = DEFAULT_CONDITION_LOGIC
    action: ActionConfig
    exit_rules: list[ExitRuleConfig] = []
    template_id: str | None = None


class BotCreateFromBacktestBase(BaseModel):
    """Inputs for the OA-style 'Automate your strategy' (backtest → bot).

    Everything but overrides comes from the backtest run itself. `allocation` is
    accepted for parity with OA's Create Bot panel but is not persisted (the Bot
    model has no allocation column); position sizing is `size_pct` of the account.
    """
    name: str | None = None
    market_type: str = DEFAULT_BACKTEST_MARKET_TYPE
    allocation: float | None = None
    size_pct: float = DEFAULT_BACKTEST_SIZE_PCT
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None


class BotUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_enabled: bool | None = None
    conditions: list[ConditionConfig] | None = None
    condition_logic: str | None = None
    action: ActionConfig | None = None
    exit_rules: list[ExitRuleConfig] | None = None


class BotOut(BotCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str
    is_enabled: bool
    is_archived: bool = False
    archived_at: datetime | None = None
    run_count: int
    last_run_at: datetime | None
    last_signal: str | None
    last_result: dict | None
    created_at: datetime