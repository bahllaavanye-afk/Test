"""Pydantic v2 schemas for the Bot builder."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, List

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


_ALLOWED_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}
_ALLOWED_DIRECTIONS = {"above", "below"}
_ALLOWED_INDICATOR_OPERATORS = {"<", ">", "crosses_above", "crosses_below"}
_ALLOWED_CONDITION_OPERATORS = {
    "<",
    ">",
    "==",
    "!=",
    "crosses_above",
    "crosses_below",
}
_ALLOWED_MA_TYPES = {"sma", "ema"}
_ALLOWED_CONDITION_LOGIC = {"ALL", "ANY"}
_ALLOWED_EXIT_TYPES = {"take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"}
_ALLOWED_TRIGGER_TYPES = {"schedule", "price_cross", "indicator"}
_ALLOWED_ACTION_TYPES = {
    "open_long",
    "open_short",
    "close_position",
    "send_alert",
    "reduce_position",
    "open_option_spread",
}
_ALLOWED_OPTION_SIDES = {"buy", "sell"}
_ALLOWED_OPTION_TYPES = {"call", "put"}


class TriggerConfig(BaseModel):
    type: Literal["schedule", "price_cross", "indicator"]
    interval: str = "5m"               # 1m|5m|15m|1h|4h|1d
    price_level: float | None = None   # price_cross
    direction: str = "above"           # above|below
    indicator: str | None = None       # rsi|macd|bb|sma|ema
    indicator_period: int = 14
    indicator_operator: str = "<"      # <|>|crosses_above|crosses_below
    indicator_value: float | None = None

    @field_validator("interval")
    def validate_interval(cls, v: str) -> str:
        if v not in _ALLOWED_INTERVALS:
            raise ValueError(f"interval must be one of {_ALLOWED_INTERVALS}, got '{v}'")
        return v

    @field_validator("direction")
    def validate_direction(cls, v: str) -> str:
        if v not in _ALLOWED_DIRECTIONS:
            raise ValueError(f"direction must be one of {_ALLOWED_DIRECTIONS}, got '{v}'")
        return v

    @field_validator("indicator_operator")
    def validate_indicator_operator(cls, v: str) -> str:
        if v not in _ALLOWED_INDICATOR_OPERATORS:
            raise ValueError(
                f"indicator_operator must be one of {_ALLOWED_INDICATOR_OPERATORS}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_price_cross_requirements(cls, values):
        if values.type == "price_cross":
            if values.price_level is None:
                raise ValueError("price_level must be set for type 'price_cross'")
        return values


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
    period: int = 14
    operator: str = "<"   # < | > | == | != | crosses_above | crosses_below
    value: float | None = None
    ma_period: int | None = None
    start_time: str | None = None   # "09:30" ET
    end_time: str | None = None     # "16:00" ET
    pnl_pct: float | None = None
    # EMA cross custom periods
    fast_period: int | None = None
    slow_period: int | None = None
    # price_vs_ma MA type
    ma_type: str | None = "sma"     # "sma" or "ema"
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
    regimes: List[str] | None = None    # e.g. ["bull", "sideways"] or ["stressed"]

    @field_validator("operator")
    def validate_operator(cls, v: str) -> str:
        if v not in _ALLOWED_CONDITION_OPERATORS:
            raise ValueError(f"operator must be one of {_ALLOWED_CONDITION_OPERATORS}, got '{v}'")
        return v

    @field_validator("ma_type")
    def validate_ma_type(cls, v: str | None) -> str | None:
        if v is not None and v not in _ALLOWED_MA_TYPES:
            raise ValueError(f"ma_type must be one of {_ALLOWED_MA_TYPES}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_regime_fields(cls, values):
        if values.type == "regime":
            if not values.regimes:
                raise ValueError("regimes must be provided for condition type 'regime'")
        return values


class OptionLeg(BaseModel):
    """One leg of a multi-leg options order (spread, condor, straddle, ...)."""
    side: Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    delta: float | None = None        # target delta for strike selection (0-1)
    strike: float | None = None       # explicit strike (overrides delta if set)
    dte: int = 30                     # days to expiration target
    ratio: int = 1                    # contracts per 1x of the spread

    @field_validator("delta")
    def validate_delta(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 1):
            raise ValueError("delta must be between 0 and 1")
        return v

    @field_validator("strike")
    def validate_strike(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("strike must be positive")
        return v

    @field_validator("dte")
    def validate_dte(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("dte must be a positive integer")
        return v

    @field_validator("ratio")
    def validate_ratio(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ratio must be a positive integer")
        return v


class ActionConfig(BaseModel):
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
    # Option Alpha-style per-bot safeguards (checked before opening a position).
    # None = unlimited. Bot won't open once a limit is reached; existing
    # positions must close (or the day roll over) before it opens again.
    max_open_positions: int | None = None    # OA "Position limit" (max open at once)
    max_daily_positions: int | None = None    # OA "Daily positions limit"

    @field_validator("size_pct")
    def validate_size_pct(cls, v: float) -> float:
        if not (0 < v <= 100):
            raise ValueError("size_pct must be > 0 and <= 100")
        return v

    @field_validator("stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "reduce_by_pct")
    def validate_pct_fields(cls, v: float | None, info) -> float | None:
        if v is not None and not (0 < v <= 100):
            raise ValueError(f"{info.field_name} must be > 0 and <= 100")
        return v

    @field_validator("max_open_positions", "max_daily_positions")
    def validate_position_limits(cls, v: int | None, info) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @model_validator(mode="after")
    def validate_legs_requirement(cls, values):
        if values.type == "open_option_spread" and not values.legs:
            raise ValueError("legs must be provided for action type 'open_option_spread'")
        return values


class ExitRuleConfig(BaseModel):
    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
    value: float | None = None       # pct for TP/SL/trailing
    hours: int | None = None         # time_exit
    indicator: str | None = None
    period: int = 14
    operator: str = ">"
    indicator_value: float | None = None

    @field_validator("type")
    def validate_type(cls, v: str) -> str:
        if v not in _ALLOWED_EXIT_TYPES:
            raise ValueError(f"type must be one of {_ALLOWED_EXIT_TYPES}, got '{v}'")
        return v

    @field_validator("value")
    def validate_value(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("value must be positive")
        return v

    @field_validator("hours")
    def validate_hours(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("hours must be a positive integer")
        return v

    @field_validator("operator")
    def validate_operator(cls, v: str) -> str:
        if v not in {"<", ">", "==", "!=", "crosses_above", "crosses_below"}:
            raise ValueError(f"operator must be a valid comparison operator, got '{v}'")
        return v


class BotCreate(BaseModel):
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

    @field_validator("name", "symbol")
    def non_empty_strings(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v.strip()

    @field_validator("condition_logic")
    def validate_condition_logic(cls, v: str) -> str:
        if v not in _ALLOWED_CONDITION_LOGIC:
            raise ValueError(f"condition_logic must be one of {_ALLOWED_CONDITION_LOGIC}, got '{v}'")
        return v


class BotCreateFromBacktestBase(BaseModel):
    """Inputs for the OA-style 'Automate your strategy' (backtest → bot).

    Everything but overrides comes from the backtest run itself. `allocation` is
    accepted for parity with OA's Create Bot panel but is not persisted (the Bot
    model has no allocation column); position sizing is `size_pct` of the account.
    """
    name: str | None = None
    market_type: str = "equity"
    allocation: float | None = None
    size_pct: float = 5.0
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None

    @field_validator("size_pct")
    def validate_size_pct(cls, v: float) -> float:
        if not (0 < v <= 100):
            raise ValueError("size_pct must be > 0 and <= 100")
        return v

    @field_validator("take_profit_pct", "stop_loss_pct")
    def validate_pct_fields(cls, v: float | None, info) -> float | None:
        if v is not None and not (0 < v <= 100):
            raise ValueError(f"{info.field_name} must be > 0 and <= 100")
        return v


class BotUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_enabled: bool | None = None
    conditions: List[ConditionConfig] | None = None
    condition_logic: str | None = None
    action: ActionConfig | None = None
    exit_rules: List[ExitRuleConfig] | None = None

    @field_validator("name")
    def validate_name(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("name must be a non-empty string if provided")
        return v.strip() if v is not None else v

    @field_validator("condition_logic")
    def validate_condition_logic(cls, v: str | None) -> str | None:
        if v is not None and v not in _ALLOWED_CONDITION_LOGIC:
            raise ValueError(f"condition_logic must be one of {_ALLOWED_CONDITION_LOGIC}, got '{v}'")
        return v


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