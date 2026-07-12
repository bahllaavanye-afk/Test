"""Options-specific strategy signal generators.

These are rule-based signal generators (no ML). They produce signals for
options strategies: covered call, cash-secured put, iron condor, long call
momentum, earnings IV crush, and the wheel cycle.

Full execution logic (selecting specific contract, delta targeting, expiry
selection) will be layered on top in a future implementation phase. For now
these stubs define the signal conditions and backtest interface.

Research basis:
- Options Alpha backtests (180M+ strategies): Iron condors at 45 DTE,
  15-20 delta short strikes, 50% profit targets, 21 DTE exits → 78-82% win rate
- ORATS documented Sharpe 1.2-1.8 for managed iron condors
- Wheel income: 1-3% per cycle (monthly), ~12-24% annualized
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class CoveredCallParams(BaseModel):
    """Parameters for :class:`CoveredCallStrategy`."""

    iv_rank_threshold: float = Field(
        default=30,
        ge=0,
        le=100,
        description="Minimum IV rank required to trigger a signal.",
        example=35,
    )
    min_shares: int = Field(
        default=100,
        ge=1,
        description="Minimum number of shares held before the strategy can generate a signal.",
        example=200,
    )

    @validator("iv_rank_threshold")
    def iv_rank_must_be_reasonable(cls, v: float) -> float:
        if v > 100:
            raise ValueError("IV rank threshold cannot exceed 100")
        return v


class CashSecuredPutParams(BaseModel):
    """Parameters for :class:`CashSecuredPutStrategy`."""

    rsi_threshold: float = Field(
        default=40.0,
        ge=0,
        le=100,
        description="RSI upper bound; signal is generated when RSI falls below this value.",
        example=35.0,
    )
    iv_rank_threshold: float = Field(
        default=25,
        ge=0,
        le=100,
        description="Minimum IV rank required to trigger a signal.",
        example=30,
    )

    @validator("rsi_threshold")
    def rsi_must_be_reasonable(cls, v: float) -> float:
        if v > 100:
            raise ValueError("RSI threshold cannot exceed 100")
        return v

    @validator("iv_rank_threshold")
    def iv_rank_must_be_reasonable(cls, v: float) -> float:
        if v > 100:
            raise ValueError("IV rank threshold cannot exceed 100")
        return v


class CoveredCallStrategy(AbstractStrategy):
    """Sell an OTM call against an existing long equity position.

    Income generation strategy: targets ~2-5% monthly premium. Only
    generates a signal when:
      - We hold ≥100 shares of the underlying (checked via metadata parameter)
      - IV rank > 30 (elevated implied volatility means richer premium)
      - Stock is not in a strong uptrend (we don't want assignment risk)

    Execution hint: sell the nearest OTM call with ~30 delta, 21-45 DTE.
    Exit at 50% of max profit or roll at 21 DTE.
    """

    name = "covered_call"
    display_name = "Covered Call"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"          # premium collection = income / arb bucket
    tick_interval_seconds = 3600.0     # re-evaluate hourly
    confidence_threshold = 0.60

    # Parameters (Options Alpha research)
    TARGET_DTE_MIN = 21
    TARGET_DTE_MAX = 45
    TARGET_DELTA = 0.30
    MIN_IV_RANK = 30
    PROFIT_TARGET_PCT = 0.50   # exit at 50% of max premium received

    def __init__(self, params: dict | CoveredCallParams | None = None):
        super().__init__(params)
        if isinstance(params, CoveredCallParams):
            self.params_model = params
        elif isinstance(params, dict):
            self.params_model = CoveredCallParams(**params)
        else:
            self.params_model = CoveredCallParams()
        self.iv_rank_threshold: float = self.params_model.iv_rank_threshold
        self.min_shares: int = self.params_model.min_shares

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

        # Avoid selling into strong uptrend (would cap gains)
        close = data["close"]
        sma20 = close.rolling(20).mean().iloc[-1]
        price = close.iloc[-1]
        if price > sma20 * 1.05:
            # Price running >5% above 20-day SMA — avoid capping upside
            return None

        # RSI filter: don't sell calls if stock is oversold (could bounce hard)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / max(float(loss.iloc[-1]), 0.001)))
        if rsi < 35:
            return None  # oversold — stock may snap higher, don't cap upside

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
                "rsi": round(float(rsi), 1),
                "target_delta": self.TARGET_DELTA,
                "target_dte_min": self.TARGET_DTE_MIN,
                "target_dte_max": self.TARGET_DTE_MAX,
                "profit_target_pct": self.PROFIT_TARGET_PCT,
                "hint": f"Sell ~{self.TARGET_DELTA}-delta call, {self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. Exit at 50% profit.",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Vectorised signal: enter when IV rank proxy > threshold.

        NOTE: Real IV rank requires historical IV data not in OHLCV.
        We approximate with realised volatility percentile as a proxy.
        """
        close = df["close"]
        # HV20 percentile proxy for IV rank
        log_ret = np.log(close / close.shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv_min = hv20.rolling(252).min()
        hv_max = hv20.rolling(252).max()
        iv_rank = (hv20 - hv_min) / (hv_max - hv_min + 0.001) * 100

        # RSI filter
        d = close.diff()
        gain = d.clip(lower=0).rolling(14).mean()
        loss = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 0.001)))

        # Signal: sell call when vol rank above threshold and not oversold
        entries = ((iv_rank.shift(1) > self.iv_rank_threshold) & (rsi.shift(1) > 35)).fillna(False)
        exits = (iv_rank.shift(1) < 20).fillna(False)  # buy back when vol collapses
        return BacktestSignals(entries=entries, exits=exits)


class CashSecuredPutStrategy(AbstractStrategy):
    """Sell an OTM put to enter a long position at a discount.

    Suited for sideways or mildly bullish markets. Collects premium while
    waiting to acquire shares at the chosen strike price.

    Signal conditions:
      - RSI(14) < 40: stock has pulled back, risk/reward is better
      - IV rank > 25: enough premium available to make the trade worthwhile

    Execution hint: sell ~30-delta put at 2-4 weeks expiry.
    Exit at 50% of max premium or roll at 21 DTE.
    """

    name = "cash_secured_put"
    display_name = "Cash Secured Put"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.60

    TARGET_DTE_MIN = 14
    TARGET_DTE_MAX = 28
    TARGET_DELTA = -0.30
    MIN_IV_RANK = 25
    PROFIT_TARGET_PCT = 0.50

    def __init__(self, params: dict | CashSecuredPutParams | None = None):
        super().__init__(params)
        if isinstance(params, CashSecuredPutParams):
            self.params_model = params
        elif isinstance(params, dict):
            self.params_model = CashSecuredPutParams(**params)
        else:
            self.params_model = CashSecuredPutParams()
        self.rsi_threshold: float = self.params_model.rsi_threshold
        self.iv_rank_threshold: float = self.params_model.iv_rank_threshold

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
        rsi_val = float(rsi.iloc[-1])

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
                "rsi": round(rsi_val, 2),
                "iv_rank": round(iv_rank, 2),
                "target_delta": self.TARGET_DELTA,
                "target_dte_min": self.TARGET_DTE_MIN,
                "target_dte_max": self.TARGET_DTE_MAX,
                "profit_target_pct": self.PROFIT_TARGET_PCT,
                "hint": f"Sell ~{abs(self.TARGET_DELTA)}-delta put, {self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. Exit at 50% profit.",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Vectorised signal for cash secured put strategy."""
        close = df["close"]
        # Compute RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        # Approximate IV rank using realised volatility percentile
        log_ret = np.log(close / close.shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv_min = hv20.rolling(252).min()
        hv_max = hv20.rolling(252).max()
        iv_rank = (hv20 - hv_min) / (hv_max - hv_min + 0.001) * 100

        entries = ((rsi.shift(1) < self.rsi_threshold) & (iv_rank.shift(1) > self.iv_rank_threshold)).fillna(False)
        exits = (iv_rank.shift(1) < 20).fillna(False)  # exit when volatility collapses
        return BacktestSignals(entries=entries, exits=exits)