"""Pydantic v2 schemas for the Bot builder."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, List

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class TriggerConfig(BaseModel):
    type: Literal["schedule", "price_cross", "indicator"]
    interval: str = "5m"               # 1m|5m|15m|1h|4h|1d
    price_level: float | None = None   # price_cross
    direction: str = "above"           # above|below
    indicator: str | None = None       # rsi|macd|bb|sma|ema
    indicator_period: int = 14
    indicator_operator: str = "<"      # <|>|crosses_above|crosses_below
    indicator_value: float | None = None

    @field_validator("indicator_period")
    @classmethod
    def validate_indicator_period(cls, v: int) -> int:
        if v < 1:
            raise ValueError("indicator_period must be >= 1")
        return v

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if v not in allowed:
            raise ValueError(f"interval must be one of {allowed}")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in {"above", "below"}:
            raise ValueError("direction must be 'above' or 'below'")
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

    @field_validator("period", "fast_period", "slow_period", "ma_period", "k_period", "d_period")
    @classmethod
    def non_negative_int(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("period values must be >= 1")
        return v

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        allowed = {"<", ">", "==", "!=", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {allowed}")
        return v

    @field_validator("ma_type")
    @classmethod
    def validate_ma_type(cls, v: str | None) -> str | None:
        if v is not None and v not in {"sma", "ema"}:
            raise ValueError("ma_type must be 'sma' or 'ema'")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str | None) -> str | None:
        if v is not None and v not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        return v

    @field_validator("regimes")
    @classmethod
    def validate_regimes(cls, v: List[str] | None) -> List[str] | None:
        if v is not None and not v:
            # Empty list is treated as None (no regime restriction)
            return None
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
    @classmethod
    def validate_delta(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 1):
            raise ValueError("delta must be between 0 and 1")
        return v

    @field_validator("dte")
    @classmethod
    def validate_dte(cls, v: int) -> int:
        if v < 0:
            raise ValueError("dte must be non‑negative")
        return v

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, v: int) -> int:
        if v < 1:
            raise ValueError("ratio must be >= 1")
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

    @field_validator("size_pct", "stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "reduce_by_pct")
    @classmethod
    def pct_between_0_and_100(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 100):
            raise ValueError("percentage fields must be between 0 and 100")
        return v

    @field_validator("max_open_positions", "max_daily_positions")
    @classmethod
    def non_negative_int(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("position limits must be non‑negative")
        return v

    @model_validator(mode="after")
    def validate_legs_for_option_spread(self) -> "ActionConfig":
        if self.type == "open_option_spread":
            if not self.legs:
                raise ValueError("legs must be provided for open_option_spread actions")
        return self


class ExitRuleConfig(BaseModel):
    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
    value: float | None = None       # pct for TP/SL/trailing
    hours: int | None = None         # time_exit
    indicator: str | None = None
    period: int = 14
    operator: str = ">"
    indicator_value: float | None = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("value must be non‑negative")
        return v

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("hours must be non‑negative")
        return v

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: int) -> int:
        if v < 1:
            raise ValueError("period must be >= 1")
        return v

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        allowed = {"<", ">", "==", "!=", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {allowed}")
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
    @classmethod
    def validate_condition_logic(cls, v: str) -> str:
        if v not in {"ALL", "ANY"}:
            raise ValueError("condition_logic must be either 'ALL' or 'ANY'")
        return v

    @field_validator("conditions", "exit_rules")
    @classmethod
    def default_empty_list(cls, v: List | None) -> List:
        return v or []


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

    @field_validator("size_pct", "take_profit_pct", "stop_loss_pct")
    @classmethod
    def pct_between_0_and_100(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 100):
            raise ValueError("percentage fields must be between 0 and 100")
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
    @classmethod
    def validate_condition_logic(cls, v: str | None) -> str | None:
        if v is not None and v not in {"ALL", "ANY"}:
            raise ValueError("condition_logic must be either 'ALL' or 'ANY'")
        return v

    @field_validator("conditions", "exit_rules")
    @classmethod
    def default_to_none_or_empty(cls, v: List | None) -> List | None:
        # Preserve None (meaning no change) but convert empty list to [] explicitly
        return v if v is None else v or []


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