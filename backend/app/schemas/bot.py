"""Pydantic v2 schemas for the Bot builder.

These models define the JSON structures used by the API to create, update, and
retrieve trading bots. They are deliberately lightweight – validation is handled
by Pydantic, while the business logic lives elsewhere in the service layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, List

from pydantic import BaseModel, ConfigDict


class TriggerConfig(BaseModel):
    """Configuration that determines *when* a bot should be evaluated.

    Attributes
    ----------
    type: Literal["schedule", "price_cross", "indicator"]
        The trigger mechanism.
    interval: str
        Frequency of evaluation (e.g., ``"5m"``). Supported values are
        ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``.
    price_level: float | None
        Target price for ``price_cross`` triggers.
    direction: str
        Whether the price must cross ``"above"`` or ``"below"`` the ``price_level``.
    indicator: str | None
        Name of the technical indicator used for ``indicator`` triggers.
    indicator_period: int
        Look‑back period for the indicator.
    indicator_operator: str
        Comparison operator for the indicator (e.g., ``"<"``, ``">"``,
        ``"crosses_above"``, ``"crosses_below"``).
    indicator_value: float | None
        Threshold value for the indicator comparison.
    """

    type: Literal["schedule", "price_cross", "indicator"]
    interval: str = "5m"               # 1m|5m|15m|1h|4h|1d
    price_level: float | None = None   # price_cross
    direction: str = "above"           # above|below
    indicator: str | None = None       # rsi|macd|bb|sma|ema
    indicator_period: int = 14
    indicator_operator: str = "<"      # <|>|crosses_above|crosses_below
    indicator_value: float | None = None


class ConditionConfig(BaseModel):
    """A single condition that must be satisfied for a bot to fire.

    The condition can be based on indicators, price relationships, P&L, time
    windows, position state, machine‑learning signals, or market regime.

    Attributes
    ----------
    type: Literal["indicator", "price_vs_ma", "pnl", "time_window",
                 "position_exists", "no_position", "ml_signal", "regime"]
        The type of condition.
    indicator: str | None
        Indicator name when ``type`` is ``"indicator"``.
    period: int
        Look‑back period for the indicator.
    operator: str
        Comparison operator (e.g., ``"<"``, ``">"``, ``"=="``, ``"!="``,
        ``"crosses_above"``, ``"crosses_below"``).
    value: float | None
        Threshold value for the comparison.
    ma_period: int | None
        Moving‑average period for ``price_vs_ma`` conditions.
    start_time: str | None
        Start of the allowed time window (e.g., ``"09:30"`` ET).
    end_time: str | None
        End of the allowed time window (e.g., ``"16:00"`` ET).
    pnl_pct: float | None
        P&L percentage threshold for ``pnl`` conditions.
    fast_period: int | None
        Fast EMA period for EMA cross conditions.
    slow_period: int | None
        Slow EMA period for EMA cross conditions.
    ma_type: str | None
        Type of moving average used in ``price_vs_ma`` (``"sma"`` or ``"ema"``).
    k_period: int | None
        %K period for stochastic oscillator conditions.
    d_period: int | None
        %D period for stochastic oscillator conditions.
    multiplier: float | None
        Multiplier for Supertrend conditions.
    direction: str | None
        Expected direction for ``ml_signal`` (``"up"`` or ``"down"``).
    min_confidence: float | None
        Minimum confidence required for an ``ml_signal`` condition.
    regimes: list[str] | None
        List of market regimes where the condition is valid (e.g., ``["bull",
        "sideways"]``).
    """

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
    period: int = 14
    operator: str = "<"   # < | > | == | != | crosses_above | crosses_below
    value: float | None = None
    ma_period: int | None = None
    start_time: str | None = None   # "09:30" ET
    end_time: str | None = None     # "16:00" ET
    pnl_pct: float | None = None
    fast_period: int | None = None
    slow_period: int | None = None
    ma_type: str | None = "sma"     # "sma" or "ema"
    k_period: int | None = None
    d_period: int | None = None
    multiplier: float | None = None
    direction: str | None = None        # "up" | "down"
    min_confidence: float | None = None  # default 0.65 in the engine
    regimes: List[str] | None = None    # e.g. ["bull", "sideways"] or ["stressed"]


class OptionLeg(BaseModel):
    """One leg of a multi‑leg options order (spread, condor, straddle, ...)."""

    side: Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    delta: float | None = None        # target delta for strike selection (0-1)
    strike: float | None = None       # explicit strike (overrides delta if set)
    dte: int = 30                     # days to expiration target
    ratio: int = 1                    # contracts per 1x of the spread


class ActionConfig(BaseModel):
    """Configuration describing the action a bot should take when triggered.

    Attributes
    ----------
    type: Literal[...]
        Action type (e.g., ``"open_long"``, ``"send_alert"``, etc.).
    size_pct: float
        Position size as a percentage of the account equity.
    stop_loss_pct: float | None
        Stop‑loss threshold expressed as a percentage.
    take_profit_pct: float | None
        Take‑profit threshold expressed as a percentage.
    trailing_stop_pct: float | None
        Trailing‑stop threshold expressed as a percentage.
    alert_message: str | None
        Message sent when ``type`` is ``"send_alert"``.
    reduce_by_pct: float | None
        Percentage by which to reduce an existing position.
    legs: list[OptionLeg] | None
        Required when ``type`` is ``"open_option_spread"``.
    max_open_positions: int | None
        Maximum number of concurrent open positions for this bot.
    max_daily_positions: int | None
        Maximum number of positions the bot may open in a single day.
    """

    type: Literal[
        "open_long",
        "open_short",
        "close_position",
        "send_alert",
        "reduce_position",
        "open_option_spread",
    ]
    size_pct: float = 5.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    alert_message: str | None = None
    reduce_by_pct: float | None = None
    legs: List[OptionLeg] | None = None   # required for open_option_spread
    max_open_positions: int | None = None    # OA "Position limit" (max open at once)
    max_daily_positions: int | None = None    # OA "Daily positions limit"


class ExitRuleConfig(BaseModel):
    """Rule governing when a position should be exited.

    Attributes
    ----------
    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
        Exit rule type.
    value: float | None
        Percentage for TP/SL/trailing‑stop rules.
    hours: int | None
        Number of hours after which a ``time_exit`` rule triggers.
    indicator: str | None
        Indicator name for indicator‑based exits.
    period: int
        Look‑back period for the indicator.
    operator: str
        Comparison operator for the indicator.
    indicator_value: float | None
        Threshold value for the indicator comparison.
    """

    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
    value: float | None = None       # pct for TP/SL/trailing
    hours: int | None = None         # time_exit
    indicator: str | None = None
    period: int = 14
    operator: str = ">"
    indicator_value: float | None = None


class BotCreate(BaseModel):
    """Schema for creating a new trading bot.

    Attributes
    ----------
    name: str
        Human‑readable bot name.
    description: str
        Optional free‑form description.
    symbol: str
        Ticker symbol the bot trades.
    market_type: str
        Market classification (e.g., ``"equity"``).
    trigger: TriggerConfig
        Trigger configuration.
    conditions: list[ConditionConfig]
        List of conditions that must be satisfied.
    condition_logic: str
        Logical combination of conditions (e.g., ``"ALL"`` or ``"ANY"``).
    action: ActionConfig
        Action to perform when conditions are met.
    exit_rules: list[ExitRuleConfig]
        Optional exit rules for the position.
    template_id: str | None
        Identifier of a bot template to copy settings from.
    """

    name: str
    description: str = ""
    symbol: str
    market_type: str = "equity"
    trigger: TriggerConfig
    conditions: List[ConditionConfig] = []
    condition_logic: str = "ALL"
    action: ActionConfig
    exit_rules: List[ExitRuleConfig] = []
    template_id: str | None = None


class BotCreateFromBacktestBase(BaseModel):
    """Inputs for the OA‑style 'Automate your strategy' (backtest → bot).

    All fields except overrides are derived from the backtest run. ``allocation`` is
    accepted for parity with OA's Create Bot panel but is not persisted; position
    sizing is controlled by ``size_pct``.
    """

    name: str | None = None
    market_type: str = "equity"
    allocation: float | None = None
    size_pct: float = 5.0
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None


class BotUpdate(BaseModel):
    """Schema for updating an existing bot's mutable fields."""

    name: str | None = None
    description: str | None = None
    is_enabled: bool | None = None
    conditions: List[ConditionConfig] | None = None
    condition_logic: str | None = None
    action: ActionConfig | None = None
    exit_rules: List[ExitRuleConfig] | None = None


class BotOut(BotCreate):
    """Representation of a bot returned from the API, extending ``BotCreate``."""

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