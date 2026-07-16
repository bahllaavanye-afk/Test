"""
Interest Rate Differential / EM Carry Trade Proxy.

FX carry trade implemented via ETFs:
  - Long high-yield EM ETFs (EMB = EM bonds, EEM = EM equities)
  - Short TLT (long-duration US Treasuries)

Entry condition: US 10Y yield > 4% AND rising (positive 20‑day momentum) with
additional confirmation from EM ETF price momentum and a low‑volatility environment.
Exit condition: Yield falls below 3.5% OR EM ETF price momentum turns negative.
Data: FRED 10Y yield via public API or fallback to TLT price proxy.
"""
import numpy as np
import pandas as pd
import urllib.request
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals

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
    YIELD_MOMENTUM_DAYS = 20   # rising if 20‑day trend is positive
    EXIT_YIELD = 3.5           # unwind when yield falls below 3.5%
    MIN_EM_MOMENTUM = 0.001    # minimum positive EM ETF 10‑day return
    MAX_VIX = 20.0             # optional volatility filter

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.yield_threshold = p.get("yield_threshold", self.YIELD_THRESHOLD)
        self.yield_momentum_days = p.get("yield_momentum_days", self.YIELD_MOMENTUM_DAYS)
        self.exit_yield = p.get("exit_yield", self.EXIT_YIELD)
        self.min_em_momentum = p.get("min_em_momentum", self.MIN_EM_MOMENTUM)
        self.max_vix = p.get("max_vix", self.MAX_VIX)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # Basic data validation
        required_cols = {"close"}
        if not required_cols.issubset(data.columns) or len(data) < self.yield_momentum_days + 5:
            return None

        # Fetch latest yield (live or from provided series)
        live_yield = _fetch_10y_yield()
        if "us10y_yield" in data.columns:
            yield_series = data["us10y_yield"].dropna()
            current_yield = float(yield_series.iloc[-1])
            yield_ma = float(yield_series.rolling(self.yield_momentum_days).mean().iloc[-1])
            # Momentum: difference between current and lagged value
            lagged_yield = float(yield_series.shift(self.yield_momentum_days).iloc[-1])
            yield_momentum = current_yield - lagged_yield
        elif live_yield is not None:
            current_yield = live_yield
            # Approximate trend using TLT price changes
            close = data["close"]
            tlt_ret = close.pct_change().rolling(self.yield_momentum_days).mean().iloc[-1]
            yield_trend_rising = tlt_ret < 0  # falling TLT = rising yields
            # Adjust moving average heuristically
            yield_ma = current_yield - (0.1 if yield_trend_rising else -0.1)
            yield_momentum = -0.05 if yield_trend_rising else 0.05
        else:
            # Fallback proxy based on price trend
            close = data["close"]
            pct = close.pct_change().rolling(self.yield_momentum_days).mean()
            current_yield = 4.0 - float(pct.iloc[-1]) * 100  # rough proxy
            yield_ma = current_yield - float(pct.iloc[-2]) * 10
            yield_momentum = current_yield - (4.0 - float(pct.iloc[-2]) * 100)

        if np.isnan(current_yield) or np.isnan(yield_ma):
            return None

        # Determine if yield is rising
        yield_rising = current_yield > yield_ma and yield_momentum > 0

        # EM ETF price momentum confirmation (10‑day)
        em_price = data["close"]
        em_momentum_10d = em_price.pct_change(10).iloc[-1]

        # Optional VIX filter
        vix_ok = True
        if "vix" in data.columns:
            latest_vix = float(data["vix"].iloc[-1])
            vix_ok = latest_vix <= self.max_vix

        # ENTRY LOGIC
        if (
            current_yield > self.yield_threshold
            and yield_rising
            and em_momentum_10d > self.min_em_momentum
            and vix_ok
        ):
            confidence = min(
                0.85,
                0.60
                + (current_yield - self.yield_threshold) * 0.08
                + (em_momentum_10d) * 0.2,
            )
            return Signal(
                symbol=symbol,
                side="buy",  # long EM ETFs, short TLT
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "us10y_yield": round(current_yield, 3),
                    "yield_trend": "rising",
                    "em_momentum_10d": round(em_momentum_10d, 5),
                    "vix": round(latest_vix, 2) if "vix" in data.columns else None,
                    "trade_type": "long_em_short_tlt",
                },
            )

        # EXIT LOGIC
        exit_signal = False
        if current_yield < self.exit_yield:
            exit_signal = True
        elif em_momentum_10d is not None and em_momentum_10d < 0:
            exit_signal = True

        if exit_signal:
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=0.80,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "us10y_yield": round(current_yield, 3),
                    "signal": "exit",
                    "em_momentum_10d": round(em_momentum_10d, 5) if em_momentum_10d is not None else None,
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        # Build yield series (real or proxy)
        if "us10y_yield" in df.columns:
            y = df["us10y_yield"]
        else:
            close = df["close"]
            pnorm = (close - close.rolling(252).min()) / (
                close.rolling(252).max() - close.rolling(252).min() + 1e-10
            )
            y = 5.0 - pnorm * 4.0  # map price to ~1‑5% yield range

        y_ma = y.rolling(self.yield_momentum_days).mean()
        y_shifted = y.shift(1)
        y_ma_shifted = y_ma.shift(1)

        # EM price momentum for backtest
        em_momentum_10d = df["close"].pct_change(10)

        # ENTRY: yield above threshold, rising, and EM momentum positive
        entries = (
            (y_shifted > self.yield_threshold)
            & (y_shifted > y_ma_shifted)
            & (em_momentum_10d > self.min_em_momentum)
        )

        # EXIT: yield below exit level OR EM momentum turns negative
        exits = (y_shifted < self.exit_yield) | (em_momentum_10d < 0)

        # SHORT side: opposite of long logic (carry when yields fall)
        short_entries = y_shifted < self.exit_yield
        short_exits = y_shifted > self.yield_threshold

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )