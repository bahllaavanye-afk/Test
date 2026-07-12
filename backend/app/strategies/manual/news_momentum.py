"""
Post-Earnings Announcement Drift (PEAD) — News Momentum Strategy.

Academic basis: Ball & Brown (1968) — stocks with positive earnings surprises
continue to drift upward for 60 trading days after the announcement.

Signal: BUY when EPS surprise > 5% AND price gapped up > 2% on earnings day,
        within a 2‑day window of the announcement.

Sharpe target: 0.8–1.2
Risk bucket: directional (30% capital allocation)
"""
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class NewsMomentumParams(BaseModel):
    """Configuration parameters for the News Momentum strategy."""

    lookback_hours: int = Field(
        default=48,
        ge=1,
        description="Number of hours to look back for news sentiment.",
        example=48,
    )
    min_sentiment_score: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum sentiment score required to consider a news item relevant.",
        example=0.05,
    )
    position_size_pct: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Position size as a proportion of allocated capital (0‑1).",
        example=1.0,
    )
    min_earnings_surprise: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum earnings surprise (as a decimal) to trigger a signal.",
        example=0.05,
    )
    min_price_change: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum price gap on earnings day required for entry.",
        example=0.02,
    )
    drift_window_days: int = Field(
        default=2,
        ge=1,
        description="Maximum number of trading days after the earnings announcement to enter a position.",
        example=2,
    )

    @validator("min_sentiment_score", "position_size_pct", "min_earnings_surprise", "min_price_change")
    def _validate_percentage(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Value must be between 0 and 1 inclusive.")
        return v


class NewsMomentumStrategy(AbstractStrategy):
    name = "news_momentum"
    display_name = "Post-Earnings Drift (PEAD)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0  # Check hourly

    # Hard‑coded thresholds (fallback defaults)
    MIN_EARNINGS_SURPRISE = 0.05    # 5% EPS beat required
    MIN_PRICE_CHANGE = 0.02         # 2% price gap on earnings day
    DRIFT_WINDOW_DAYS = 2           # Enter within 2 trading days of announcement
    MAX_HOLDING_DAYS = 60           # Exit after 60 days (PEAD drift window)

    DEFAULT_PARAMS = {
        "lookback_hours": 48,
        "min_sentiment_score": 0.05,
        "position_size_pct": 1.0,
    }

    def __init__(self, params: dict | NewsMomentumParams | None = None):
        super().__init__(params)
        # Validate and normalise parameters via Pydantic
        if isinstance(params, NewsMomentumParams):
            cfg = params
        else:
            cfg = NewsMomentumParams(**{**self.DEFAULT_PARAMS, **(params or {})})

        self.lookback_hours = cfg.lookback_hours
        self.min_sentiment_score = cfg.min_sentiment_score
        self.position_size_pct = cfg.position_size_pct

        # Strategy‑specific thresholds – allow overrides via the incoming dict
        p = params or {}
        self.min_surprise = float(p.get("min_earnings_surprise", cfg.min_earnings_surprise))
        self.min_price_change = float(p.get("min_price_change", cfg.min_price_change))
        self.drift_window = int(p.get("drift_window_days", cfg.drift_window_days))

    def _compute_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a BUY signal when:
        1. earnings_surprise column present and last value > MIN_EARNINGS_SURPRISE
        2. Price change on the earnings bar > MIN_PRICE_CHANGE
        3. Signal is within DRIFT_WINDOW_DAYS of the earnings date

        Expected columns: ``close``, optional ``open``, ``earnings_surprise`` or ``eps_surprise_pct``,
        and optional ``earnings_date`` (datetime). The DataFrame index may be a datetime index.
        """
        if "close" not in data.columns or len(data) < 5:
            return None

        close = data["close"]
        latest_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close

        # Retrieve earnings surprise (support two possible column names)
        surprise_pct: float | None = None
        if "earnings_surprise" in data.columns:
            raw = data["earnings_surprise"].iloc[-1]
            if pd.notna(raw):
                surprise_pct = float(raw)
        elif "eps_surprise_pct" in data.columns:
            raw = data["eps_surprise_pct"].iloc[-1]
            if pd.notna(raw):
                surprise_pct = float(raw)

        if surprise_pct is None:
            return None

        if surprise_pct <= self.min_surprise:
            return None

        price_change = (latest_close - prev_close) / prev_close if prev_close > 0 else 0.0
        if price_change <= self.min_price_change:
            return None

        if "earnings_date" in data.columns:
            earnings_date = data["earnings_date"].iloc[-1]
            if pd.notna(earnings_date):
                today = data.index[-1] if hasattr(data.index, "__len__") else None
                if today is not None:
                    try:
                        days_since = (pd.Timestamp(today) - pd.Timestamp(earnings_date)).days
                        if days_since > self.drift_window:
                            return None
                    except Exception:
                        pass  # If date arithmetic fails, fall back to generating the signal

        confidence = min(
            0.90,
            0.50 + (surprise_pct / 0.10) * 0.20 + price_change * 2.0,
        )

        return Signal(
            symbol=symbol,
            side="buy",
            confidence=round(confidence, 4),
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "earnings_surprise_pct": round(surprise_pct * 100, 2),
                "price_change_pct": round(price_change * 100, 2),
                "drift_strategy": "PEAD",
                "max_holding_days": self.MAX_HOLDING_DAYS,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized back‑test signals for VectorBT.

        *Entry*: earnings_surprise > threshold **and** price_change > threshold (shifted 1 bar to avoid look‑ahead).  
        *Exit*: simple momentum fade – price_change < -1% (shifted 1 bar). No short entries are generated.
        """
        close = df["close"]
        price_change = close.pct_change()

        if "earnings_surprise" in df.columns:
            surprise = df["earnings_surprise"].fillna(0)
        elif "eps_surprise_pct" in df.columns:
            surprise = df["eps_surprise_pct"].fillna(0)
        else:
            false_series = pd.Series(False, index=df.index)
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        entries = (
            (surprise > self.min_surprise) & (price_change > self.min_price_change)
        ).shift(1).fillna(False)

        exits = (price_change < -0.01).shift(1).fillna(False)

        false_series = pd.Series(False, index=df.index)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=false_series,
            short_exits=false_series,
        )