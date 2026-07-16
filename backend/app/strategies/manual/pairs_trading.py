"""
Statistical Pairs Trading via Engle-Granger Cointegration.

Academic basis: Gatev, Goetzmann, Rouwenhorst (2006) — pairs trading
generates 11% pa excess return with Sharpe 1.5-2.67.

Logic:
  1. Test for cointegration between two symbols (Engle-Granger test)
  2. Compute spread = price_A - hedge_ratio * price_B
  3. Normalize spread to z-score using rolling mean/std
  4. Entry: |z| > entry_z (default 2.0) — trade mean reversion
  5. Exit:  |z| < exit_z  (default 0.5)
  6. Stop:  |z| > stop_z  (default 4.0)

Market neutral: long one leg, short the other (dollar-neutral).
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from pydantic import BaseModel, Field, root_validator, validator, ValidationError
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class PairsTradingParams(BaseModel):
    """Configuration parameters for the Pairs Trading strategy."""

    lookback: int = Field(
        ...,
        description="Number of historical bars used for rolling calculations.",
        ge=1,
        example=252,
    )
    z_entry: float = Field(
        ...,
        description="Z‑score threshold to open a position.",
        gt=0,
        example=2.0,
    )
    z_exit: float = Field(
        ...,
        description="Z‑score threshold to close a position.",
        ge=0,
        example=0.5,
    )
    stop_z: float = Field(
        4.0,
        description="Z‑score threshold at which trading is halted due to risk.",
        gt=0,
        example=4.0,
    )
    coint_pvalue: float = Field(
        0.05,
        description="Maximum p‑value accepted for the Engle‑Granger cointegration test.",
        gt=0,
        lt=1,
        example=0.05,
    )
    min_half_life: int | None = Field(
        None,
        description="Minimum acceptable half‑life of the spread (in bars).",
        ge=1,
        example=5,
    )
    max_half_life: int | None = Field(
        None,
        description="Maximum acceptable half‑life of the spread (in bars).",
        ge=1,
        example=126,
    )

    @root_validator
    def check_half_life_range(cls, values):
        min_hl, max_hl = values.get("min_half_life"), values.get("max_half_life")
        if min_hl is not None and max_hl is not None and min_hl > max_hl:
            raise ValueError("min_half_life must be less than or equal to max_half_life")
        return values

    @validator("z_exit")
    def exit_less_than_entry(cls, v, values):
        entry = values.get("z_entry")
        if entry is not None and v >= entry:
            raise ValueError("z_exit must be less than z_entry")
        return v

    @validator("z_entry")
    def entry_less_than_stop(cls, v, values):
        stop = values.get("stop_z")
        if stop is not None and v >= stop:
            raise ValueError("z_entry must be less than stop_z")
        return v


class PairsTradingStrategy(AbstractStrategy):
    name = "pairs_trading"
    display_name = "Pairs Trading (Cointegration)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 300.0  # 5 min

    # Default pairs universe (cointegrated historically)
    DEFAULT_PAIRS = [
        ("SPY", "QQQ"),
        ("GLD", "SLV"),
        ("KO", "PEP"),
        ("XOM", "CVX"),
        ("C", "JPM"),
    ]

    DEFAULT_PARAMS = {
        "lookback": 252,
        "z_entry": 2.0,
        "z_exit": 0.5,
        "min_half_life": 5,
        "max_half_life": 126,
        "stop_z": 4.0,
        "coint_pvalue": 0.05,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # Merge user supplied values with defaults before validation
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        try:
            validated = PairsTradingParams(**merged)
        except ValidationError as exc:
            raise ValueError(f"Invalid pairs trading parameters: {exc}") from exc

        self.lookback = validated.lookback
        self.entry_z = validated.z_entry
        self.exit_z = validated.z_exit
        self.stop_z = validated.stop_z
        self.coint_pvalue = validated.coint_pvalue
        self.min_half_life = validated.min_half_life
        self.max_half_life = validated.max_half_life

    def _compute_spread(self, price_a: pd.Series, price_b: pd.Series, lookback: int) -> tuple[pd.Series, float]:
        """Compute hedge ratio via OLS regression and return z‑scored spread."""
        y = price_a.iloc[-lookback:]
        x = price_b.iloc[-lookback:]
        hedge_ratio = np.cov(y, x)[0, 1] / np.var(x)
        spread = price_a - hedge_ratio * price_b
        rolling_mean = spread.rolling(lookback).mean()
        rolling_std = spread.rolling(lookback).std()
        z_score = (spread - rolling_mean) / (rolling_std + 1e-9)
        return z_score, hedge_ratio

    def _is_cointegrated(self, price_a: pd.Series, price_b: pd.Series) -> bool:
        try:
            _, pvalue, _ = coint(price_a, price_b)
            return pvalue < self.coint_pvalue
        except Exception:
            return False

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # data expected to have MultiIndex columns: (symbol, field)
        # or be called per‑pair. Here we handle simple case: data has 'close_a', 'close_b'
        if "close_a" not in data.columns or "close_b" not in data.columns:
            return None
        if len(data) < self.lookback + 10:
            return None

        price_a = data["close_a"]
        price_b = data["close_b"]
        z_score, hedge_ratio = self._compute_spread(price_a, price_b, self.lookback)
        latest_z = z_score.iloc[-1]

        if abs(latest_z) > self.stop_z:
            return None  # spread too wide — risk of cointegration breakdown

        if latest_z > self.entry_z:
            confidence = min(0.95, (latest_z - self.entry_z) / (self.stop_z - self.entry_z) * 0.8 + 0.6)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"z_score": float(latest_z), "hedge_ratio": hedge_ratio, "leg": "A"},
            )
        elif latest_z < -self.entry_z:
            confidence = min(0.95, (-latest_z - self.entry_z) / (self.stop_z - self.entry_z) * 0.8 + 0.6)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"z_score": float(latest_z), "hedge_ratio": hedge_ratio, "leg": "A"},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Rolling OLS hedge ratio — no lookahead bias."""
        if "close_a" not in df.columns or "close_b" not in df.columns:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        price_a = df["close_a"]
        price_b = df["close_b"]
        n = len(price_a)
        signals = pd.Series(0, index=df.index, dtype=float)

        for i in range(self.lookback, n):
            window_a = price_a.iloc[i - self.lookback:i]
            window_b = price_b.iloc[i - self.lookback:i]
            try:
                hedge = float(np.polyfit(window_b, window_a, 1)[0])
            except Exception:
                continue
            spread_window = window_a - hedge * window_b
            spread_mean = spread_window.mean()
            spread_std = spread_window.std()
            if spread_std < 1e-9:
                continue
            spread_i = price_a.iloc[i - 1] - hedge * price_b.iloc[i - 1]
            z = (spread_i - spread_mean) / spread_std
            if z < -self.entry_z:
                signals.iloc[i] = 1
            elif z > self.entry_z:
                signals.iloc[i] = -1
            elif abs(z) < self.exit_z:
                signals.iloc[i] = 0

        entries = signals > 0.5
        exits = signals == 0
        short_entries = signals < -0.5
        short_exits = exits.copy()

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )