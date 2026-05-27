"""Options-specific strategy signal generators.

These are rule-based signal generators (no ML). They produce signals for
options strategies: covered call, cash-secured put, iron condor, and
directional long call.

Full execution logic (selecting specific contract, delta targeting, expiry
selection) will be layered on top in a future implementation phase. For now
these stubs define the signal conditions and backtest interface.
"""
from __future__ import annotations

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class CoveredCallStrategy(AbstractStrategy):
    """Sell an OTM call against an existing long equity position.

    Income generation strategy: targets ~2-5% monthly premium. Only
    generates a signal when:
      - We hold ≥100 shares of the underlying (checked via metadata parameter)
      - IV rank > 30 (elevated implied volatility means richer premium)
      - Stock is not in a strong uptrend (we don't want assignment risk)

    Execution hint: sell the nearest OTM call with ~30 delta, 21-45 DTE.
    """

    name = "covered_call"
    display_name = "Covered Call"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"          # premium collection = income / arb bucket
    tick_interval_seconds = 3600.0     # re-evaluate hourly
    confidence_threshold = 0.60

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # IV rank threshold above which we sell premium
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", 30.0)
        # Minimum shares to qualify for covered call
        self.min_shares: int = (params or {}).get("min_shares", 100)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Signal: IV rank > threshold and position ≥ 100 shares.

        The caller must inject 'iv_rank' and 'current_shares' into
        data.attrs or the last-row metadata before calling analyze().
        """
        if len(data) < 20:
            return None

        iv_rank: float = data.attrs.get("iv_rank", 0.0)
        current_shares: int = data.attrs.get("current_shares", 0)

        if current_shares < self.min_shares:
            return None
        if iv_rank < self.iv_rank_threshold:
            return None

        # Additional filter: avoid selling into strong uptrend (would cap gains)
        close = data["close"]
        sma20 = close.rolling(20).mean().iloc[-1]
        price = close.iloc[-1]
        if price > sma20 * 1.05:
            # Price running >5% above 20-day SMA — avoid capping upside
            return None

        confidence = min(0.85, 0.60 + (iv_rank - self.iv_rank_threshold) / 100)
        return Signal(
            symbol=symbol,
            side="sell",              # sell the call
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "covered_call",
                "iv_rank": round(iv_rank, 2),
                "current_shares": current_shares,
                "target_delta": 0.30,
                "target_dte_min": 21,
                "target_dte_max": 45,
                "hint": "Sell nearest OTM call at ~0.30 delta, 21-45 DTE",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Vectorised signal: enter when IV rank proxy > threshold.

        NOTE: Real IV rank requires historical IV data not in OHLCV.
        We approximate with realised volatility percentile as a proxy.
        Full implementation requires an options data feed.
        """
        close = df["close"]
        # Proxy for IV rank: 20-day realised vol as percentile of trailing 252 days
        rv20 = close.pct_change().rolling(20).std() * (252 ** 0.5)
        rv_rank = rv20.rolling(252).rank(pct=True) * 100

        # Signal: sell call when vol rank above threshold
        entries = (rv_rank.shift(1) > self.iv_rank_threshold).fillna(False)
        exits = (rv_rank.shift(1) < 20).fillna(False)  # buy back when vol collapses
        return BacktestSignals(entries=entries, exits=exits)


class CashSecuredPutStrategy(AbstractStrategy):
    """Sell an OTM put to enter a long position at a discount.

    Suited for sideways or mildly bullish markets. Collects premium while
    waiting to acquire shares at the chosen strike price.

    Signal conditions:
      - RSI(14) < 40: stock has pulled back, risk/reward is better
      - IV rank > 25: enough premium available to make the trade worthwhile

    Execution hint: sell ~30-delta put at 2-4 weeks expiry.
    """

    name = "cash_secured_put"
    display_name = "Cash Secured Put"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.60

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.rsi_threshold: float = (params or {}).get("rsi_threshold", 40.0)
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", 25.0)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Signal: RSI < threshold AND IV rank > threshold."""
        if len(data) < 30:
            return None

        iv_rank: float = data.attrs.get("iv_rank", 0.0)
        if iv_rank < self.iv_rank_threshold:
            return None

        close = data["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]

        if rsi_val >= self.rsi_threshold:
            return None

        confidence = min(0.80, 0.60 + (self.rsi_threshold - rsi_val) / 100)
        return Signal(
            symbol=symbol,
            side="sell",              # sell the put
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "cash_secured_put",
                "rsi": round(float(rsi_val), 2),
                "iv_rank": round(iv_rank, 2),
                "target_delta": -0.30,
                "target_dte_min": 14,
                "target_dte_max": 28,
                "hint": "Sell ~0.30-delta put, 2-4 weeks to expiry",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Approximate signal using RSI as proxy (IV rank not in OHLCV)."""
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        entries = (rsi.shift(1) < self.rsi_threshold).fillna(False)
        exits = (rsi.shift(1) > 55).fillna(False)
        return BacktestSignals(entries=entries, exits=exits)


class IronCondorStrategy(AbstractStrategy):
    """Sell OTM put spread + OTM call spread simultaneously.

    Profits when the underlying stays in a defined range until expiry.
    Best used on low-beta, range-bound stocks when IV is elevated.

    Signal conditions:
      - IV rank > 50 (need rich premium to justify width risk)
      - Low 20-day beta (sideways trending, not directional)
      - No strong trend: price within ±3% of 20-day SMA

    Execution hint: sell 16-delta short strikes, buy 5-delta long wings.
    Width of each spread typically $5–$10.
    """

    name = "iron_condor"
    display_name = "Iron Condor"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", 50.0)
        self.trend_pct_threshold: float = (params or {}).get("trend_pct_threshold", 0.03)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Signal: IV rank > 50 AND price within ±3% of SMA20 (range-bound)."""
        if len(data) < 25:
            return None

        iv_rank: float = data.attrs.get("iv_rank", 0.0)
        if iv_rank < self.iv_rank_threshold:
            return None

        close = data["close"]
        sma20 = close.rolling(20).mean().iloc[-1]
        price = close.iloc[-1]
        deviation = abs(price - sma20) / sma20
        if deviation > self.trend_pct_threshold:
            return None  # Stock is trending too strongly

        confidence = min(0.82, 0.65 + (iv_rank - self.iv_rank_threshold) / 200)
        return Signal(
            symbol=symbol,
            side="sell",              # net credit trade (sell the condor)
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "iron_condor",
                "iv_rank": round(iv_rank, 2),
                "sma20_deviation_pct": round(float(deviation) * 100, 2),
                "short_call_delta": 0.16,
                "short_put_delta": -0.16,
                "wing_delta": 0.05,
                "hint": "Sell 16-delta short strikes, buy 5-delta wings, ~30-45 DTE",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Approximate using realised vol rank + range-bound filter."""
        close = df["close"]
        rv20 = close.pct_change().rolling(20).std() * (252 ** 0.5)
        rv_rank = rv20.rolling(252).rank(pct=True) * 100
        sma20 = close.rolling(20).mean()
        deviation = (close - sma20).abs() / sma20

        high_iv = rv_rank.shift(1) > self.iv_rank_threshold
        range_bound = deviation.shift(1) < self.trend_pct_threshold
        entries = (high_iv & range_bound).fillna(False)
        exits = (deviation.shift(1) > self.trend_pct_threshold * 1.5).fillna(False)
        return BacktestSignals(entries=entries, exits=exits)


class LongCallMomentum(AbstractStrategy):
    """Buy in-the-money calls on breakout stocks for leveraged directional exposure.

    Instead of buying shares on a breakout, buys a deep ITM call (0.70 delta)
    to get similar delta exposure with defined risk (max loss = premium paid).

    Signal conditions: same as BreakoutStrategy (price > 52-week high with
    volume confirmation), but execution is via options.

    Execution hint: buy 0.70-delta call, nearest monthly expiry ≥ 30 DTE.
    """

    name = "long_call_momentum"
    display_name = "Long Call (Momentum)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0     # 15-minute bars
    confidence_threshold = 0.60

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.lookback: int = (params or {}).get("lookback", 52)
        self.vol_mult: float = (params or {}).get("vol_mult", 1.5)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Breakout signal → buy 0.70-delta call instead of shares."""
        if len(data) < self.lookback + 20:
            return None

        close = data["close"]
        high = data["high"]
        volume = data.get("volume", pd.Series(dtype=float))

        resistance = high.rolling(self.lookback).max().shift(1)
        vol_avg = volume.rolling(20).mean() if len(volume) > 0 else pd.Series(1, index=data.index)

        price = close.iloc[-1]
        res = resistance.iloc[-1]
        vol_curr = volume.iloc[-1] if len(volume) > 0 else 1
        vol_mean = vol_avg.iloc[-1] if len(volume) > 0 else 1

        if price > res and vol_curr > self.vol_mult * vol_mean:
            pct_break = (price - res) / res
            confidence = min(0.80, 0.55 + pct_break * 3)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": "long_call_momentum",
                    "resistance": round(float(res), 4),
                    "breakout_pct": round(float(pct_break) * 100, 2),
                    "target_delta": 0.70,
                    "target_dte_min": 30,
                    "target_dte_max": 60,
                    "hint": "Buy 0.70-delta call, 30-60 DTE. Max loss = premium paid.",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Identical entry conditions to BreakoutStrategy."""
        close = df["close"]
        high = df["high"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        resistance = high.rolling(self.lookback).max().shift(2)
        vol_avg = volume.rolling(20).mean()

        entries = (
            close.shift(1).gt(resistance) & volume.shift(1).gt(self.vol_mult * vol_avg.shift(1))
        ).fillna(False)
        exits = close.shift(1).lt(resistance).fillna(False)
        return BacktestSignals(entries=entries, exits=exits)
