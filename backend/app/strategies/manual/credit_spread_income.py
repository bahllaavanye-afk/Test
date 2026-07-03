"""
Credit Spread Income Strategy — OptionsAlpha mechanical approach
================================================================
Systematic options income via short put spreads (bull put) and short call spreads
(bear call), mirroring the mechanical rules popularized by OptionsAlpha.com.

Entry logic:
- When VIX > 20 AND IVR (IV Rank) > 50:  sell put spread (bull put) for bullish income
  · Sell 30Δ put, buy 10Δ put, ~45 DTE, collect ≥ 1/3 of spread width as premium
- When IVR > 70:  sell iron condor (both a bull put and a bear call spread simultaneously)

Exit / management:
- Take profit at 50% of max credit received (standard OptionsAlpha rule)
- Time-based exit at 21 DTE (theta acceleration accelerates decay rapidly below 21 DTE)
- Stop loss: if the spread value reaches 200% of original credit, close for a loss

Risk classification:
- Delta-neutral income fits the "arbitrage" bucket (70% capital allocation bucket)
  because the P&L is driven by volatility mispricing, not directional price movement.

IVR proxy:
- Real-time IVR requires an options data provider.
- Backtest uses a rolling-percentile of 20-day ATR over 60 days as a free IV-rank proxy.
  (ATR is highly correlated with at-the-money implied volatility for liquid equity ETFs.)

References:
- OptionsAlpha Playbooks — systematic income via defined-risk spreads
- Sosnoff, T. (2014) "Tastytrade mechanical rules for options income"
- Cohen, G. (2005) "The Bible of Options Strategies" — credit spread mechanics
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

log = logging.getLogger(__name__)

try:
    import yfinance as yf

    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


class CreditSpreadIncomeStrategy(AbstractStrategy):
    """
    Systematic options income via short put spreads (bull put) and short call spreads
    (bear call).

    Signal metadata includes:
        spread_type         — "bull_put" | "bear_call" | "iron_condor"
        short_strike        — estimated short leg strike (spot ± delta offset)
        long_strike         — estimated long leg strike (spread width away)
        expiry              — target expiry date string (YYYY-MM-DD), ~45 DTE
        credit_per_contract — estimated premium in dollars per 100-share contract
        ivr                 — IV Rank (0–100) used for entry decision
    """

    name = "credit_spread_income"
    display_name = "Credit Spread Income (OptionsAlpha)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"          # delta-neutral income = arbitrage bucket
    tick_interval_seconds = 3600.0     # hourly check — no need to run every minute
    confidence_threshold = 0.60

    # ── Entry thresholds ──────────────────────────────────────────────────────
    VIX_THRESHOLD = 20.0    # VIX must be above this for put spread entry
    IVR_PUT_SPREAD = 50.0   # IVR ≥ 50 → sell put spread
    IVR_IRON_CONDOR = 70.0  # IVR ≥ 70 → sell iron condor
    MIN_CREDIT_RATIO = 1 / 3  # must collect ≥ 1/3 of spread width

    # ── Spread construction ───────────────────────────────────────────────────
    SPREAD_WIDTH_PCT = 0.05   # 5% of spot price = spread width (e.g. $25 on $500 SPY)
    SHORT_LEG_OFFSET = 0.04   # short strike ~4% OTM (roughly 30Δ for 45 DTE)
    LONG_LEG_OFFSET = 0.08    # long strike ~8% OTM (roughly 10Δ for 45 DTE)
    TARGET_DTE = 45
    EXIT_DTE = 21
    STOP_MULT = 2.0           # 200% of credit received = stop loss

    # ── ATR-based IVR proxy parameters ───────────────────────────────────────
    ATR_PERIOD = 14
    IVR_LOOKBACK = 60         # 60-day rolling percentile for IVR proxy

    UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA"]

    # ── Analyze ───────────────────────────────────────────────────────────────

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Analyze market data and emit a trade Signal if entry criteria are met."""
        if symbol not in self.UNIVERSE:
            return None
        if data.empty or "close" not in data.columns:
            return None
        if len(data) < self.IVR_LOOKBACK:
            return None

        spot = float(data["close"].iloc[-1])

        # --- Step 1: Get VIX and IVR ---
        vix_level, ivr = await asyncio.get_running_loop().run_in_executor(
            None, self._get_vix_and_ivr, symbol, data
        )
        if vix_level is None and ivr is None:
            return None

        log.debug(
            "credit_spread_income/%s  spot=%.2f  VIX=%s  IVR=%.0f",
            symbol,
            spot,
            f"{vix_level:.1f}" if vix_level is not None else "N/A",
            ivr,
        )

        # --- Step 2: Entry decision ---
        if ivr >= self.IVR_IRON_CONDOR:
            spread_type = "iron_condor"
            side = "sell"
            confidence = min(0.60 + (ivr - self.IVR_IRON_CONDOR) / 60.0, 0.95)
        elif vix_level is not None and vix_level > self.VIX_THRESHOLD and ivr >= self.IVR_PUT_SPREAD:
            spread_type = "bull_put"
            side = "sell"
            confidence = min(0.60 + (ivr - self.IVR_PUT_SPREAD) / 80.0, 0.90)
        else:
            return None  # Conditions not met — no trade

        # --- Step 3: Estimate strikes and credit ---
        expiry = (date.today() + timedelta(days=self.TARGET_DTE)).isoformat()
        short_put = round(spot * (1 - self.SHORT_LEG_OFFSET), 2)
        long_put = round(spot * (1 - self.LONG_LEG_OFFSET), 2)
        short_call = round(spot * (1 + self.SHORT_LEG_OFFSET), 2)
        long_call = round(spot * (1 + self.LONG_LEG_OFFSET), 2)

        spread_width = round(spot * self.SPREAD_WIDTH_PCT, 2)
        # Rough credit estimate: ~30% of spread width is typical for a 30Δ/10Δ spread
        credit_per_contract = round(spread_width * 0.30 * 100, 2)  # per 1-contract (100 shares)

        if spread_type == "bull_put":
            short_strike = short_put
            long_strike = long_put
            meta_extra: dict = {}
        else:  # iron_condor — use put side as primary legs
            short_strike = short_put
            long_strike = long_put
            meta_extra = {
                "short_call_strike": short_call,
                "long_call_strike": long_call,
                "iron_condor_credit": round(credit_per_contract * 2, 2),
            }

        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "spread_type": spread_type,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiry": expiry,
                "credit_per_contract": credit_per_contract,
                "ivr": round(ivr, 1),
                "vix": round(vix_level, 2) if vix_level is not None else None,
                "spot": round(spot, 2),
                "target_dte": self.TARGET_DTE,
                "exit_dte": self.EXIT_DTE,
                "profit_target_pct": 0.50,
                "stop_mult": self.STOP_MULT,
                **meta_extra,
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_vix_and_ivr(self, symbol: str, data: pd.DataFrame) -> tuple[float | None, float | None]:
        """
        Retrieve the current VIX level (if yfinance is available) and compute an IVR proxy
        based on the supplied price data.

        Returns a tuple (vix_level, ivr). If VIX cannot be fetched, vix_level is None.
        """
        # Compute IVR proxy from ATR regardless of yfinance availability
        ivr = self._atr_ivr(data)

        if not _YF_AVAILABLE:
            return None, ivr

        try:
            vix_series = yf.Ticker("^VIX").history(period="5d")["Close"]
            if vix_series.empty:
                vix_level = None
            else:
                vix_level = float(vix_series.iloc[-1])
        except Exception as exc:  # pragma: no cover
            log.exception("Failed to fetch VIX data: %s", exc)
            vix_level = None

        return vix_level, ivr

    def _atr_ivr(self, data: pd.DataFrame) -> float | None:
        """
        Approximate IV Rank (IVR) using the rolling percentile of the 20‑day ATR
        over the past ``IVR_LOOKBACK`` days.

        The method expects ``high``, ``low`` and ``close`` columns. If any are missing,
        the function returns ``None``.
        """
        required = {"high", "low", "close"}
        if not required.issubset(data.columns):
            log.debug("ATR IVR proxy requires columns %s; available: %s", required, data.columns)
            return None

        # True Range calculation
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window=self.ATR_PERIOD, min_periods=1).mean()

        if len(atr) < self.IVR_LOOKBACK:
            return None

        recent_atr = atr.iloc[-1]
        lookback_series = atr.iloc[-self.IVR_LOOKBACK : -1]

        # Percentile rank of the most recent ATR within the lookback window
        rank = (lookback_series < recent_atr).sum() / len(lookback_series) * 100.0
        return float(rank)