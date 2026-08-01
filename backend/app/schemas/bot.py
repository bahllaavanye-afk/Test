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

    @field_validator("interval")
    def validate_interval(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if v not in allowed:
            raise ValueError(f"interval must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_type_dependent_fields(self) -> "TriggerConfig":
        if self.type == "price_cross":
            if self.price_level is None:
                raise ValueError("price_level must be set when type='price_cross'")
        elif self.type == "indicator":
            if self.indicator is None:
                raise ValueError("indicator must be set when type='indicator'")
            if self.indicator_value is None:
                raise ValueError("indicator_value must be set when type='indicator'")
        return self


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
        allowed = {
            "<",
            ">",
            "==",
            "!=",
            "crosses_above",
            "crosses_below",
        }
        if v not in allowed:
            raise ValueError(f"operator must be one of {allowed}")
        return v

    @field_validator("ma_type")
    def validate_ma_type(cls, v: str | None) -> str | None:
        if v is not None and v not in {"sma", "ema"}:
            raise ValueError("ma_type must be 'sma' or 'ema'")
        return v

    @model_validator(mode="after")
    def validate_type_dependent_fields(self) -> "ConditionConfig":
        t = self.type
        if t == "indicator":
            if self.indicator is None:
                raise ValueError("indicator must be set for type='indicator'")
            if self.value is None:
                raise ValueError("value must be set for type='indicator'")
        elif t == "price_vs_ma":
            if self.ma_period is None:
                raise ValueError("ma_period must be set for type='price_vs_ma'")
        elif t == "pnl":
            if self.pnl_pct is None:
                raise ValueError("pnl_pct must be set for type='pnl'")
        elif t == "time_window":
            if self.start_time is None or self.end_time is None:
                raise ValueError("start_time and end_time must be set for type='time_window'")
        elif t == "ml_signal":
            if self.direction is None or self.min_confidence is None:
                raise ValueError("direction and min_confidence required for type='ml_signal'")
        elif t == "regime":
            if not self.regimes:
                raise ValueError("regimes list must be provided for type='regime'")
        return self


class OptionLeg(BaseModel):
    """One leg of a multi-leg options order (spread, condor, straddle, ...)."""
    side: Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    delta: float | None = None        # target delta for strike selection (0-1)
    strike: float | None = None       # explicit strike (overrides delta if set)
    dte: int = 30                     # days to expiration target
    ratio: int = 1                    # contracts per 1x of the spread


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
            raise ValueError("size_pct must be >0 and <=100")
        return v

    @field_validator("stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "reduce_by_pct")
    def validate_pct_range(cls, v: float | None) -> float | None:
        if v is not None and not (0 < v <= 100):
            raise ValueError("percentage fields must be >0 and <=100")
        return v

    @model_validator(mode="after")
    def validate_option_spread(cls) -> "ActionConfig":
        if self.type == "open_option_spread":
            if not self.legs or len(self.legs) == 0:
                raise ValueError("legs must be provided for type='open_option_spread'")
        return self


class ExitRuleConfig(BaseModel):
    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"]
    value: float | None = None       # pct for TP/SL/trailing
    hours: int | None = None         # time_exit
    indicator: str | None = None
    period: int = 14
    operator: str = ">"
    indicator_value: float | None = None

    @field_validator("operator")
    def validate_operator(cls, v: str) -> str:
        allowed = {"<", ">", "==", "!=", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_type_dependent_fields(self) -> "ExitRuleConfig":
        t = self.type
        if t in {"take_profit", "stop_loss", "trailing_stop"}:
            if self.value is None:
                raise ValueError(f"value must be set for exit type '{t}'")
        elif t == "time_exit":
            if self.hours is None:
                raise ValueError("hours must be set for type='time_exit'")
        elif t == "indicator":
            if self.indicator is None or self.indicator_value is None:
                raise ValueError("indicator and indicator_value must be set for type='indicator'")
        return self


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
    def validate_logic(cls, v: str) -> str:
        if v not in {"ALL", "ANY"}:
            raise ValueError("condition_logic must be 'ALL' or 'ANY'")
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
            raise ValueError("size_pct must be >0 and <=100")
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
    def validate_logic(cls, v: str | None) -> str | None:
        if v is not None and v not in {"ALL", "ANY"}:
            raise ValueError("condition_logic must be 'ALL' or 'ANY'")
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