"""
Interest Rate Differential / EM Carry Trade Proxy.

FX carry trade implemented via ETFs:
  - Long high-yield EM ETFs (EMB = EM bonds, EEM = EM equities)
  - Short TLT (long-duration US Treasuries)

Entry condition: US 10Y yield > 4% AND rising (positive 20-day momentum).
This indicates:
  1. US rates are elevated → EM carry is attractive
  2. Rising rates → TLT is falling → short TLT also profits

Data: FRED 10Y yield via public API or fallback to TLT price proxy.
"""
import urllib.request

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

# FRED public API for US 10Y Treasury yield (no auth needed)
FRED_10Y_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"


def _fetch_10y_yield() -> float | None:
    """Fetch latest US 10Y yield from FRED (free, no API key)."""
    try:
        with urllib.request.urlopen(FRED_10Y_URL, timeout=8) as resp:
            lines = resp.read().decode().strip().split("\n")
        # Last valid row: DATE,VALUE
        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in (".", ""):
                return float(parts[1].strip())
    except Exception:
        pass
    return None


class InterestRateDifferentialStrategy(AbstractStrategy):
    name = "interest_rate_differential"
    display_name = "Interest Rate Differential EM Carry"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0  # daily

    YIELD_THRESHOLD = 4.0      # US 10Y > 4% → EM carry attractive
    YIELD_MOMENTUM_DAYS = 20   # rising if 20-day trend is positive
    EXIT_YIELD = 3.5           # unwind when yield falls below 3.5%

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.yield_threshold = p.get("yield_threshold", self.YIELD_THRESHOLD)
        self.yield_momentum_days = p.get("yield_momentum_days", self.YIELD_MOMENTUM_DAYS)
        self.exit_yield = p.get("exit_yield", self.EXIT_YIELD)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < self.yield_momentum_days + 5:
            return None

        # Try live FRED data
        live_yield = _fetch_10y_yield()

        if "us10y_yield" in data.columns:
            yield_series = data["us10y_yield"].dropna()
            current_yield = float(yield_series.iloc[-1])
            yield_ma = float(yield_series.rolling(self.yield_momentum_days).mean().iloc[-1])
        elif live_yield is not None:
            current_yield = live_yield
            # Use price inverse as yield proxy for trend
            close = data["close"]
            # TLT price inversely correlated with yields
            # Estimate: if TLT price falling → yields rising
            tlt_ret = close.pct_change().rolling(self.yield_momentum_days).mean().iloc[-1]
            yield_trend_rising = tlt_ret < 0  # falling TLT = rising yields
            yield_ma = current_yield - (0.1 if yield_trend_rising else -0.1)
        else:
            # Fallback: use inverse of price trend as yield proxy
            close = data["close"]
            pct = close.pct_change().rolling(self.yield_momentum_days).mean()
            current_yield = 4.0 - float(pct.iloc[-1]) * 100  # rough proxy
            yield_ma = current_yield - float(pct.iloc[-2]) * 10

        if np.isnan(current_yield):
            return None

        yield_rising = current_yield > yield_ma

        # MUTATION: add VIX volatility filter to avoid entering in high volatility regimes
        if "vix" in data.columns and float(data["vix"].iloc[-1]) > 20.0:
            return None

        if current_yield > self.yield_threshold and yield_rising:
            confidence = min(0.80, 0.60 + (current_yield - self.yield_threshold) * 0.1)
            return Signal(
                symbol=symbol,
                side="buy",     # long EM ETFs
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk=self.risk_bucket,
                entry_price=data["close"].iloc[-1],
                timestamp=data.index[-1]
            )

        # Exit condition
        if current_yield < self.exit_yield:
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=0.9,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk=self.risk_bucket,
                entry_price=data["close"].iloc[-1],
                timestamp=data.index[-1]
            )
        return None
