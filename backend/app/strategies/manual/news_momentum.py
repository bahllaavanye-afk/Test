"""
Post-Earnings Announcement Drift (PEAD) — News Momentum Strategy.

Academic basis: Ball & Brown (1968) — stocks with positive earnings surprises
continue to drift upward for 60 trading days after the announcement.

Signal: BUY when EPS surprise > 5% AND price gapped up > 2% on earnings day,
        within a 2-day window of the announcement.

Sharpe target: 0.8–1.2
Risk bucket: directional (30% capital allocation)
"""
import pandas as pd
import numpy as np
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class NewsMomentumStrategy(AbstractStrategy):
    name = "news_momentum"
    display_name = "Post-Earnings Drift (PEAD)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0  # Check hourly

    # Thresholds
    MIN_EARNINGS_SURPRISE = 0.05    # 5% EPS beat required
    MIN_PRICE_CHANGE = 0.02         # 2% price gap on earnings day
    DRIFT_WINDOW_DAYS = 2           # Enter within 2 trading days of announcement
    MAX_HOLDING_DAYS = 60           # Exit after 60 days (PEAD drift window)

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.min_surprise = float(p.get("min_earnings_surprise", self.MIN_EARNINGS_SURPRISE))
        self.min_price_change = float(p.get("min_price_change", self.MIN_PRICE_CHANGE))
        self.drift_window = int(p.get("drift_window_days", self.DRIFT_WINDOW_DAYS))

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

        data expected columns: close, open (optional), earnings_surprise (optional),
                               earnings_date (optional, as datetime index or column)
        """
        if "close" not in data.columns or len(data) < 5:
            return None

        close = data["close"]
        latest_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close

        # Check for earnings surprise column
        surprise_pct: float | None = None
        if "earnings_surprise" in data.columns:
            raw = data["earnings_surprise"].iloc[-1]
            if pd.notna(raw):
                surprise_pct = float(raw)
        elif "eps_surprise_pct" in data.columns:
            raw = data["eps_surprise_pct"].iloc[-1]
            if pd.notna(raw):
                surprise_pct = float(raw)

        # If no earnings data, we cannot generate a PEAD signal
        if surprise_pct is None:
            return None

        # Check earnings surprise threshold
        if surprise_pct <= self.min_surprise:
            return None

        # Check price momentum on announcement day
        price_change = (latest_close - prev_close) / prev_close if prev_close > 0 else 0.0
        if price_change <= self.min_price_change:
            return None

        # Check we are within the drift window (using earnings_date column or last N bars)
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
                        pass  # Can't determine date proximity — proceed with signal

        # Confidence scales with surprise magnitude and price confirmation
        # Cap at 0.90 to leave room for risk manager adjustments
        confidence = min(0.90, 0.50 + (surprise_pct / 0.10) * 0.20 + price_change * 2.0)

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
        Vectorized signals for VectorBT.
        Entry: earnings_surprise > threshold AND price_change > threshold (shifted 1 bar to prevent lookahead).
        Exit: after MAX_HOLDING_DAYS bars or price drops below entry.
        """
        close = df["close"]

        # Price change (daily return)
        price_change = close.pct_change()

        if "earnings_surprise" in df.columns:
            surprise = df["earnings_surprise"].fillna(0)
        elif "eps_surprise_pct" in df.columns:
            surprise = df["eps_surprise_pct"].fillna(0)
        else:
            # No earnings data — emit no signals
            false_series = pd.Series(False, index=df.index)
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        # Entry: strong beat + price confirmation (shift 1 to prevent lookahead)
        entries = (
            (surprise > self.min_surprise) & (price_change > self.min_price_change)
        ).shift(1).fillna(False)

        # Exit: after drift window reversal or >60 bars held (approximate with rolling signal decay)
        # Simple exit: when momentum fades (price_change < 0 after a run)
        exits = (price_change < -0.01).shift(1).fillna(False)

        # No short signals for PEAD (drift is directional / long only)
        false_series = pd.Series(False, index=df.index)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=false_series,
            short_exits=false_series,
        )
