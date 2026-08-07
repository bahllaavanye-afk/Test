"""Pydantic v2 schemas for the Bot builder with enriched metadata and validation."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class TriggerConfig(BaseModel):
    """Configuration that determines *when* a bot evaluates its logic."""

    type: Literal["schedule", "price_cross", "indicator"] = Field(
        ...,
        description="Trigger mechanism type.",
        examples=["schedule"],
    )
    interval: str = Field(
        "5m",
        description="Time granularity for schedule‑type triggers.",
        examples=["1m", "5m", "15m", "1h", "4h", "1d"],
    )
    price_level: float | None = Field(
        None,
        description="Target price for `price_cross` triggers.",
        examples=[150.25],
    )
    direction: str = Field(
        "above",
        description="Direction of the price crossing.",
        examples=["above", "below"],
    )
    indicator: str | None = Field(
        None,
        description="Technical indicator name for `indicator` triggers.",
        examples=["rsi", "macd", "bb", "sma", "ema"],
    )
    indicator_period: int = Field(
        14,
        description="Look‑back period for the chosen indicator.",
        examples=[14],
    )
    indicator_operator: str = Field(
        "<",
        description="Comparison operator applied to the indicator value.",
        examples=["<", ">", "crosses_above", "crosses_below"],
    )
    indicator_value: float | None = Field(
        None,
        description="Static threshold for the indicator when using a comparison operator.",
        examples=[30.0],
    )

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if v not in allowed:
            raise ValueError(f"interval must be one of {sorted(allowed)}")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in {"above", "below"}:
            raise ValueError("direction must be 'above' or 'below'")
        return v

    @field_validator("indicator_operator")
    @classmethod
    def validate_indicator_operator(cls, v: str) -> str:
        allowed = {"<", ">", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"indicator_operator must be one of {sorted(allowed)}")
        return v


class ConditionConfig(BaseModel):
    """Logical condition evaluated after a trigger fires."""

    type: Literal[
        "indicator",
        "price_vs_ma",
        "pnl",
        "time_window",
        "position_exists",
        "no_position",
        "ml_signal",
        "regime",
    ] = Field(
        ...,
        description="Condition type.",
        examples=["indicator"],
    )
    indicator: str | None = Field(
        None,
        description="Name of the indicator when `type` is `indicator`.",
        examples=["rsi"],
    )
    period: int = Field(
        14,
        description="Look‑back period for the indicator.",
        examples=[14],
    )
    operator: str = Field(
        "<",
        description="Comparison operator for the condition.",
        examples=["<", ">", "==", "!=", "crosses_above", "crosses_below"],
    )
    value: float | None = Field(
        None,
        description="Static threshold for the condition.",
        examples=[30.0],
    )
    ma_period: int | None = Field(
        None,
        description="Moving‑average period for `price_vs_ma` conditions.",
        examples=[20],
    )
    start_time: str | None = Field(
        None,
        description='Window start time in "HH:MM" 24‑hour format (ET).',
        examples=["09:30"],
    )
    end_time: str | None = Field(
        None,
        description='Window end time in "HH:MM" 24‑hour format (ET).',
        examples=["16:00"],
    )
    pnl_pct: float | None = Field(
        None,
        description="Profit‑and‑loss percentage threshold.",
        examples=[5.0],
    )
    fast_period: int | None = Field(
        None,
        description="Fast EMA period for custom EMA cross conditions.",
        examples=[12],
    )
    slow_period: int | None = Field(
        None,
        description="Slow EMA period for custom EMA cross conditions.",
        examples=[26],
    )
    ma_type: str | None = Field(
        "sma",
        description="Type of moving average used in `price_vs_ma`.",
        examples=["sma", "ema"],
    )
    k_period: int | None = Field(
        None,
        description="Stochastic %K period.",
        examples=[14],
    )
    d_period: int | None = Field(
        None,
        description="Stochastic %D period.",
        examples=[3],
    )
    multiplier: float | None = Field(
        None,
        description="Multiplier for Supertrend indicator.",
        examples=[3.0],
    )
    direction: str | None = Field(
        None,
        description="Desired direction for ML predictions.",
        examples=["up", "down"],
    )
    min_confidence: float | None = Field(
        None,
        description="Minimum confidence required from the ML model.",
        examples=[0.65],
    )
    regimes: List[str] | None = Field(
        None,
        description="Allowed market regimes for the condition.",
        examples=[["bull", "sideways"]],
    )

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        allowed = {"<", ">", "==", "!=", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {sorted(allowed)}")
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

    @field_validator("min_confidence")
    @classmethod
    def validate_min_confidence(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")
        return v


class OptionLeg(BaseModel):
    """One leg of a multi‑leg options order (spread, condor, straddle, …)."""

    side: Literal["buy", "sell"] = Field(
        ...,
        description="Side of the leg.",
        examples=["buy"],
    )
    option_type: Literal["call", "put"] = Field(
        ...,
        description="Option type.",
        examples=["call"],
    )
    delta: float | None = Field(
        None,
        description="Target delta for strike selection (0‑1).",
        examples=[0.5],
    )
    strike: float | None = Field(
        None,
        description="Explicit strike price (overrides delta if set).",
        examples=[150.0],
    )
    dte: int = Field(
        30,
        description="Days to expiration target.",
        examples=[30],
    )
    ratio: int = Field(
        1,
        description="Number of contracts per unit of the spread.",
        examples=[1],
    )

    @field_validator("delta")
    @classmethod
    def validate_delta(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("delta must be between 0 and 1")
        return v

    @field_validator("strike")
    @classmethod
    def validate_strike(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("strike must be positive")
        return v

    @field_validator("dte")
    @classmethod
    def validate_dte(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("dte must be a positive integer")
        return v

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ratio must be a positive integer")
        return v


class ActionConfig(BaseModel):
    """Action to be taken when all trigger/condition logic evaluates to true."""

    type: Literal[
        "open_long",
        "open_short",
        "close_position",
        "send_alert",
        "reduce_position",
        "open_option_spread",
    ] = Field(
        ...,
        description="Action type.",
        examples=["open_long"],
    )
    size_pct: float = Field(
        5.0,
        description="Position size as a percentage of the account equity.",
        examples=[5.0],
    )
    stop_loss_pct: float | None = Field(
        None,
        description="Stop‑loss threshold as a percentage.",
        examples=[2.0],
    )
    take_profit_pct: float | None = Field(
        None,
        description="Take‑profit threshold as a percentage.",
        examples=[5.0],
    )
    trailing_stop_pct: float | None = Field(
        None,
        description="Trailing stop percentage.",
        examples=[1.5],
    )
    alert_message: str | None = Field(
        None,
        description="Message sent when `send_alert` is used.",
        examples=["Price crossed threshold!"],
    )
    reduce_by_pct: float | None = Field(
        None,
        description="Percentage by which to reduce an existing position.",
        examples=[20.0],
    )
    legs: List[OptionLeg] | None = Field(
        None,
        description="Option legs required for `open_option_spread` actions.",
        examples=[[{"side": "buy", "option_type": "call", "strike": 150.0}]],
    )
    max_open_positions: int | None = Field(
        None,
        description="Maximum number of concurrent open positions for this bot.",
        examples=[10],
    )
    max_daily_positions: int | None = Field(
        None,
        description="Maximum number of positions the bot may open in a single day.",
        examples=[20],
    )

    @field_validator("size_pct")
    @classmethod
    def validate_size_pct(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("size_pct must be positive")
        return v

    @field_validator("stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "reduce_by_pct")
    @classmethod
    def validate_percentages(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 100.0):
            raise ValueError("percentage fields must be between 0 and 100")
        return v

    @field_validator("max_open_positions", "max_daily_positions")
    @classmethod
    def validate_position_limits(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("position limits must be positive integers")
        return v

    @model_validator(mode="after")
    def validate_legs_for_option_spread(self) -> "ActionConfig":
        if self.type == "open_option_spread" and not self.legs:
            raise ValueError("legs must be provided for open_option_spread actions")
        return self


class ExitRuleConfig(BaseModel):
    """Rule that determines when a position should be exited."""

    type: Literal["take_profit", "stop_loss", "trailing_stop", "time_exit", "indicator"] = Field(
        ...,
        description="Exit rule type.",
        examples=["take_profit"],
    )
    value: float | None = Field(
        None,
        description="Numeric value associated with the rule (percentage for TP/SL/trailing).",
        examples=[5.0],
    )
    hours: int | None = Field(
        None,
        description="Number of hours after which a position exits (for `time_exit`).",
        examples=[24],
    )
    indicator: str | None = Field(
        None,
        description="Indicator name for indicator‑based exit rules.",
        examples=["rsi"],
    )
    period: int = Field(
        14,
        description="Look‑back period for the indicator.",
        examples=[14],
    )
    operator: str = Field(
        ">",
        description="Comparison operator for the indicator exit rule.",
        examples=[">", "<", "crosses_above", "crosses_below"],
    )
    indicator_value: float | None = Field(
        None,
        description="Static threshold for the indicator exit rule.",
        examples=[70.0],
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        # Basic validation is handled by Literal; additional cross‑field checks below.
        return v

    @model_validator(mode="after")
    def cross_field_validation(self) -> "ExitRuleConfig":
        if self.type in {"take_profit", "stop_loss", "trailing_stop"} and self.value is None:
            raise ValueError(f"value must be set for exit rule type {self.type}")
        if self.type == "time_exit" and self.hours is None:
            raise ValueError("hours must be set for time_exit exit rule")
        if self.type == "indicator":
            if not self.indicator:
                raise ValueError("indicator must be set for indicator exit rule")
            if self.indicator_value is None:
                raise ValueError("indicator_value must be set for indicator exit rule")
        return self

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        allowed = {"<", ">", "==", "!=", "crosses_above", "crosses_below"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {sorted(allowed)}")
        return v


class BotCreate(BaseModel):
    """Schema for creating a new bot."""

    name: str = Field(..., description="Human‑readable bot name.", examples=["Mean Reversion Bot"])
    description: str = Field("", description="Optional free‑form description.", examples=["Trades based on 20‑day SMA."])
    symbol: str = Field(..., description="Ticker symbol the bot trades.", examples=["AAPL"])
    market_type: str = Field("equity", description="Market classification (e.g., equity, futures).", examples=["equity"])
    trigger: TriggerConfig = Field(..., description="Trigger configuration.")
    conditions: List[ConditionConfig] = Field(
        default_factory=list,
        description="List of conditions that must be satisfied after a trigger fires.",
    )
    condition_logic: str = Field(
        "ALL",
        description="Logical combination of conditions: 'ALL' (AND) or 'ANY' (OR).",
        examples=["ALL"],
    )
    action: ActionConfig = Field(..., description="Action to execute when conditions are met.")
    exit_rules: List[ExitRuleConfig] = Field(
        default_factory=list,
        description="List of exit rules applied to the opened position.",
    )
    template_id: str | None = Field(
        None,
        description="Reference to a saved bot template.",
        examples=["template-1234"],
    )


class BotCreateFromBacktestBase(BaseModel):
    """Inputs for the OA‑style 'Automate your strategy' (backtest → bot)."""

    name: str | None = Field(
        None,
        description="Optional custom name for the generated bot.",
        examples=["Backtest Bot"],
    )
    market_type: str = Field("equity", description="Market type for the bot.", examples=["equity"])
    allocation: float | None = Field(
        None,
        description="Requested allocation (percentage of portfolio). Not persisted.",
        examples=[10.0],
    )
    size_pct: float = Field(5.0, description="Position size as percentage of account equity.", examples=[5.0])
    take_profit_pct: float | None = Field(
        None,
        description="Take‑profit threshold as a percentage.",
        examples=[5.0],
    )
    stop_loss_pct: float | None = Field(
        None,
        description="Stop‑loss threshold as a percentage.",
        examples=[2.0],
    )


class BotUpdate(BaseModel):
    """Schema for updating an existing bot."""

    name: str | None = Field(None, description="New name for the bot.", examples=["Updated Bot Name"])
    description: str | None = Field(None, description="Updated description.", examples=["New description."])
    is_enabled: bool | None = Field(None, description="Enable or disable the bot.")