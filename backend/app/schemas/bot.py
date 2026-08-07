"""Pydantic v2 schemas for the Bot builder."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, List

from pydantic import BaseModel, ConfigDict, field_validator, ValidationError


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
        allowed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if v not in allowed:
            raise ValueError(f"interval must be one of {sorted(allowed)}; got '{v}'")
        return v

    @field_validator("direction")
    def validate_direction(cls, v: str) -> str:
        if v not in {"above", "below"}:
            raise ValueError(f"direction must be 'above' or 'below'; got '{v}'")
        return v

    @field_validator("indicator_operator")
    def validate_indicator_operator(cls, v: str) -> str:
        allowed = {"<", ">", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"indicator_operator must be one of {sorted(allowed)}; got '{v}'")
        return v

    @field_validator("indicator_period")
    def validate_indicator_period(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("indicator_period must be a positive integer")
        return v


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

    @field_validator("period")
    def validate_period(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("period must be a positive integer")
        return v

    @field_validator("operator")
    def validate_operator(cls, v: str) -> str:
        allowed = {"<", ">", "==", "!=", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {sorted(allowed)}; got '{v}'")
        return v

    @field_validator("ma_type")
    def validate_ma_type(cls, v: str | None) -> str | None:
        if v is not None and v not in {"sma", "ema"}:
            raise ValueError("ma_type must be either 'sma' or 'ema'")
        return v

    @field_validator("direction")
    def validate_direction(cls, v: str | None) -> str | None:
        if v is not None and v not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        return v

    @field_validator("min_confidence")
    def validate_min_confidence(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")
        return v

    @field_validator("regimes")
    def validate_regimes(cls, v: List[str] | None) -> List[str] | None:
        if v is not None and not v:
            raise ValueError("regimes list cannot be empty")
        return v


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
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("delta must be between 0 and 1")
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
    max_daily_positions: int | None = None   # OA "Daily positions limit"

    @field_validator("size_pct")
    def validate_size_pct(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("size_pct must be a positive number")
        return v

    @field_validator("stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "reduce_by_pct")
    def validate_non_negative_pct(cls, v: float | None, info) -> float | None:
        name = info.field_name
        if v is not None and v < 0:
            raise ValueError(f"{name} cannot be negative")
        return v

    @field_validator("max_open_positions", "max_daily_positions")
    def validate_non_negative_int(cls, v: int | None, info) -> int | None:
        name = info.field_name
        if v is not None and v < 0:
            raise ValueError(f"{name} cannot be negative")
        return v

    @field_validator("legs")
    def validate_legs(cls, v: List[OptionLeg] | None, info) -> List[OptionLeg] | None:
        if info.data["type"] == "open_option_spread":
            if not v or len(v) == 0:
                raise ValueError("legs must be provided for open_option_spread actions")
        return v


class ExitRuleConfig(BaseModel):
    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
    value: float | None = None       # pct for TP/SL/trailing
    hours: int | None = None         # time_exit
    indicator: str | None = None
    period: int = 14
    operator: str = ">"
    indicator_value: float | None = None

    @field_validator("period")
    def validate_period(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("period must be a positive integer")
        return v

    @field_validator("operator")
    def validate_operator(cls, v: str) -> str:
        allowed = {"<", ">", "==", "!=", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {sorted(allowed)}; got '{v}'")
        return v

    @field_validator("value", "hours")
    def validate_type_specific(cls, v: float | int | None, info) -> float | int | None:
        rule_type = info.data["type"]
        field = info.field_name
        if rule_type in {"take_profit", "stop_loss", "trailing_stop"} and field == "value":
            if v is None or v <= 0:
                raise ValueError(f"value must be a positive number for {rule_type} rule")
        if rule_type == "time_exit" and field == "hours":
            if v is None or v <= 0:
                raise ValueError("hours must be a positive integer for time_exit rule")
        return v

    @field_validator("indicator")
    def validate_indicator(cls, v: str | None, info) -> str | None:
        if info.data["type"] == "indicator" and (v is None or not v):
            raise ValueError("indicator must be specified for indicator exit rule")
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

    @field_validator("condition_logic")
    def validate_condition_logic(cls, v: str) -> str:
        if v not in {"ALL", "ANY"}:
            raise ValueError("condition_logic must be either 'ALL' or 'ANY'")
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
        if v <= 0:
            raise ValueError("size_pct must be a positive number")
        return v

    @field_validator("allocation")
    def validate_allocation(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("allocation cannot be negative")
        return v

    @field_validator("take_profit_pct", "stop_loss_pct")
    def validate_pct(cls, v: float | None, info) -> float | None:
        name = info.field_name
        if v is not None and v < 0:
            raise ValueError(f"{name} cannot be negative")
        return v


class BotUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_enabled: bool | None = None
    conditions: List[ConditionConfig] | None = None
    condition_logic: str | None = None
    action: ActionConfig | None = None
    exit_rules: List[ExitRuleConfig] | None = None

    @field_validator("condition_logic")
    def validate_condition_logic(cls, v: str | None) -> str | None:
        if v is not None and v not in {"ALL", "ANY"}:
            raise ValueError("condition_logic must be either 'ALL' or 'ANY'")
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