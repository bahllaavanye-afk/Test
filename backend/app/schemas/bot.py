"""Pydantic v2 schemas for the Bot builder."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, List, Optional

from pydantic import BaseModel, ConfigDict


class TriggerConfig(BaseModel):
    """Configuration that defines **when** a bot should be evaluated.

    Attributes
    ----------
    type: Literal["schedule", "price_cross", "indicator"]
        The trigger mechanism:
        * ``schedule`` – runs on a regular interval.
        * ``price_cross`` – fires when the price crosses a level.
        * ``indicator`` – fires based on an indicator condition.
    interval: str
        Cron‑style interval for ``schedule`` triggers (e.g. ``"5m"``).  Accepted
        values are ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``.
    price_level: Optional[float]
        Target price for ``price_cross`` triggers.
    direction: str
        Direction for ``price_cross`` – ``"above"`` or ``"below"``.
    indicator: Optional[str]
        Indicator name for ``indicator`` triggers (e.g. ``"rsi"``, ``"macd"``).
    indicator_period: int
        Look‑back period for the chosen indicator.
    indicator_operator: str
        Comparison operator used with the indicator (e.g. ``"<"``, ``">"``,
        ``"crosses_above"``, ``"crosses_below"``).
    indicator_value: Optional[float]
        Threshold value for the indicator comparison.
    """

    type: Literal["schedule", "price_cross", "indicator"]
    interval: str = "5m"               # 1m|5m|15m|1h|4h|1d
    price_level: Optional[float] = None   # price_cross
    direction: str = "above"           # above|below
    indicator: Optional[str] = None       # rsi|macd|bb|sma|ema
    indicator_period: int = 14
    indicator_operator: str = "<"      # <|>|crosses_above|crosses_below
    indicator_value: Optional[float] = None


class ConditionConfig(BaseModel):
    """Defines a **condition** that must be satisfied for a bot to act.

    The condition can be based on technical indicators, price relationships,
    portfolio P&L, time windows, position existence, machine‑learning signals,
    or market regime.

    Attributes
    ----------
    type: Literal[...]
        The condition category.
    indicator: Optional[str]
        Name of the indicator when ``type`` is ``"indicator"``.
    period: int
        Look‑back period for the indicator.
    operator: str
        Comparison operator (e.g. ``"<"``, ``">"``, ``"=="``, ``"!="``,
        ``"crosses_above"``, ``"crosses_below"``).
    value: Optional[float]
        Threshold value for the comparison.
    ma_period: Optional[int]
        Moving‑average period when ``type`` is ``"price_vs_ma"``.
    start_time / end_time: Optional[str]
        Time window in ``"HH:MM"`` (Eastern Time) for ``"time_window"``.
    pnl_pct: Optional[float]
        P&L percentage threshold for ``"pnl"`` conditions.
    fast_period / slow_period: Optional[int]
        Custom EMA cross periods.
    ma_type: Optional[str]
        Moving‑average type (``"sma"`` or ``"ema"``) for ``"price_vs_ma"``.
    k_period / d_period: Optional[int]
        Stochastic oscillator parameters.
    multiplier: Optional[float]
        Multiplier for the Supertrend indicator.
    direction: Optional[str]
        Expected direction for ``"ml_signal"`` – ``"up"`` or ``"down"``.
    min_confidence: Optional[float]
        Minimum confidence required for an ML signal (default ``0.65`` in the
        engine).
    regimes: Optional[list[str]]
        List of market regimes in which the condition is active (e.g.
        ``["bull", "sideways"]`` or ``["stressed"]``).
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
    indicator: Optional[str] = None
    period: int = 14
    operator: str = "<"   # < | > | == | != | crosses_above | crosses_below
    value: Optional[float] = None
    ma_period: Optional[int] = None
    start_time: Optional[str] = None   # "09:30" ET
    end_time: Optional[str] = None     # "16:00" ET
    pnl_pct: Optional[float] = None
    # EMA cross custom periods
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    # price_vs_ma MA type
    ma_type: Optional[str] = "sma"     # "sma" or "ema"
    # Stochastic periods
    k_period: Optional[int] = None
    d_period: Optional[int] = None
    # Supertrend multiplier
    multiplier: Optional[float] = None
    # ml_signal: the trained model must predict `direction` with confidence ≥ min_confidence
    # (OA-style decision: "IF ML says up with ≥65% confidence THEN ..."). Fails safe:
    # no trained model / inference error → the condition is simply False.
    direction: Optional[str] = None        # "up" | "down"
    min_confidence: Optional[float] = None  # default 0.65 in the engine
    # regime: fire only in the listed market regimes (OA-style "trade only when
    # VIX/trend says so"). trend ∈ {bear,sideways,bull}; vol ∈ {calm,stressed}.
    regimes: Optional[List[str]] = None    # e.g. ["bull", "sideways"] or ["stressed"]


class OptionLeg(BaseModel):
    """One leg of a multi‑leg options order (spread, condor, straddle, ...)."""

    side: Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    delta: Optional[float] = None        # target delta for strike selection (0-1)
    strike: Optional[float] = None       # explicit strike (overrides delta if set)
    dte: int = 30                     # days to expiration target
    ratio: int = 1                    # contracts per 1x of the spread


class ActionConfig(BaseModel):
    """Configuration that describes **what** action the bot should take when
    conditions are met.

    Attributes
    ----------
    type: Literal[...]
        The action to perform, such as opening/closing positions or sending an alert.
    size_pct: float
        Position size expressed as a percentage of the account equity.
    stop_loss_pct / take_profit_pct / trailing_stop_pct: Optional[float]
        Risk‑management parameters expressed as percentages.
    alert_message: Optional[str]
        Message to include when ``type`` is ``"send_alert"``.
    reduce_by_pct: Optional[float]
        Percentage by which to reduce a position when ``type`` is ``"reduce_position"``.
    legs: Optional[list[OptionLeg]]
        Required when ``type`` is ``"open_option_spread"``; defines the spread legs.
    max_open_positions / max_daily_positions: Optional[int]
        Upper limits used by the OA‑style safeguards.
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
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    alert_message: Optional[str] = None
    reduce_by_pct: Optional[float] = None
    legs: Optional[List[OptionLeg]] = None   # required for open_option_spread
    # Option Alpha-style per-bot safeguards (checked before opening a position).
    # None = unlimited. Bot won't open once a limit is reached; existing
    # positions must close (or the day roll over) before it opens again.
    max_open_positions: Optional[int] = None    # OA "Position limit" (max open at once)
    max_daily_positions: Optional[int] = None    # OA "Daily positions limit"


class ExitRuleConfig(BaseModel):
    """Specification of how a position should be exited.

    Attributes
    ----------
    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
        The exit strategy.
    value: Optional[float]
        Percentage value for TP/SL/trailing stop.
    hours: Optional[int]
        Number of hours after entry after which to exit (used for ``time_exit``).
    indicator: Optional[str]
        Indicator name when ``type`` is ``"indicator"``.
    period: int
        Look‑back period for the indicator.
    operator: str
        Comparison operator for the indicator condition.
    indicator_value: Optional[float]
        Threshold for the indicator.
    """

    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
    value: Optional[float] = None       # pct for TP/SL/trailing
    hours: Optional[int] = None         # time_exit
    indicator: Optional[str] = None
    period: int = 14
    operator: str = ">"
    indicator_value: Optional[float] = None


class BotCreate(BaseModel):
    """Schema used when creating a new bot via the API.

    Attributes
    ----------
    name: str
        Human‑readable name of the bot.
    description: str
        Optional longer description.
    symbol: str
        Trading symbol (e.g. ``"AAPL"``).
    market_type: str
        Market classification (default ``"equity"``).
    trigger: TriggerConfig
        When the bot should be evaluated.
    conditions: list[ConditionConfig]
        List of conditions that must be satisfied.
    condition_logic: str
        Logical combination of conditions (e.g. ``"ALL"`` or ``"ANY"``).
    action: ActionConfig
        Action to perform when conditions are met.
    exit_rules: list[ExitRuleConfig]
        Optional exit strategies.
    template_id: Optional[str]
        Identifier of a template bot to copy from.
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
    template_id: Optional[str] = None


class BotCreateFromBacktestBase(BaseModel):
    """Inputs for the OA‑style 'Automate your strategy' (backtest → bot).

    Everything but overrides comes from the backtest run itself. ``allocation`` is
    accepted for parity with OA's Create Bot panel but is not persisted (the Bot
    model has no allocation column); position sizing is ``size_pct`` of the
    account.
    """

    name: Optional[str] = None
    market_type: str = "equity"
    allocation: Optional[float] = None
    size_pct: float = 5.0
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None


class BotUpdate(BaseModel):
    """Schema for partial updates to an existing bot.

    All fields are optional; only provided values will be overwritten.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    is_enabled: Optional[bool] = None
    conditions: Optional[List[ConditionConfig]] = None
    condition_logic: Optional[str] = None
    action: Optional[ActionConfig] = None
    exit_rules: Optional[List[ExitRuleConfig]] = None


class BotOut(BotCreate):
    """Response model returned by the API when querying a bot.

    Extends :class:`BotCreate` with runtime‑generated fields.
    """

    model_config = ConfigDict(from_attributes=True)
    id: str
    is_enabled: bool
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    run_count: int
    last_run_at: Optional[datetime] = None
    last_signal: Optional[str] = None
    last_result: Optional[dict] = None
    created_at: datetime