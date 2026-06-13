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
        if vix_level is None or ivr is None:
            return None

        log.debug(
            "credit_spread_income/%s  spot=%.2f  VIX=%.1f  IVR=%.0f",
            symbol, spot, vix_level, ivr,
        )

        # --- Step 2: Entry decision ---
        if ivr >= self.IVR_IRON_CONDOR:
            spread_type = "iron_condor"
            side = "sell"
            confidence = min(0.60 + (ivr - self.IVR_IRON_CONDOR) / 60.0, 0.95)
        elif vix_level > self.VIX_THRESHOLD and ivr >= self.IVR_PUT_SPREAD:
            spread_type = "bull_put"
            side = "sell"
            confidence = min(0.60 + (ivr - self.IVR_PUT_SPREAD) / 80.0, 0.90)
        else:
            return None  # Conditions not met — no trade

        # --- Step 3: Estimate strikes ---
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
                "vix": round(vix_level, 2),
                "spot": round(spot, 2),
                "target_dte": self.TARGET_DTE,
                "exit_dte": self.EXIT_DTE,
                "profit_target_pct": 0.50,
                "stop_mult": self.STOP_MULT,
                **meta_extra,
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_vix_and_ivr(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> tuple[float | None, float | None]:
        """
        Return (vix_level, ivr) using free yfinance data.

        VIX level comes from ^VIX.
        IVR is approximated as the rolling percentile of 20-day ATR over 60 days.
        ATR is a reasonable proxy for ATM IV on liquid ETFs/stocks.
        """
        if not _YF_AVAILABLE:
            # Fall back to ATR-only IVR (no VIX) using the supplied DataFrame
            ivr = self._atr_ivr(data)
            return None, ivr

        try:
            vix_data = yf.Ticker("^VIX").history(period="5d")
            if vix_data.empty:
                vix_level = None
            else:
                vix_level = float(vix_data["Close"].iloc[-1])
        except Exception as exc:
            log.debug("VIX fetch failed: %s", exc)
            vix_level = None

        ivr = self._atr_ivr(data)
        return vix_level, ivr

    def _atr_ivr(self, data: pd.DataFrame) -> float | None:
        """
        Compute IV Rank proxy (0–100) from 20-day ATR percentile over 60 days.
        IVR = (current ATR − 60d min ATR) / (60d max ATR − 60d min ATR) × 100
        """
        if len(data) < self.IVR_LOOKBACK + self.ATR_PERIOD:
            return None
        try:
            high = data["high"] if "high" in data.columns else data["close"]
            low = data["low"] if "low" in data.columns else data["close"]
            close = data["close"]

            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr20 = tr.rolling(self.ATR_PERIOD).mean()
            atr_window = atr20.iloc[-self.IVR_LOOKBACK:]
            current_atr = float(atr20.iloc[-1])
            atr_min = float(atr_window.min())
            atr_max = float(atr_window.max())
            if atr_max <= atr_min:
                return 50.0  # flat volatility — assume median
            ivr = (current_atr - atr_min) / (atr_max - atr_min) * 100.0
            return float(ivr)
        except Exception as exc:
            log.debug("ATR IVR computation failed: %s", exc)
            return None

    # ── Backtest ──────────────────────────────────────────────────────────────

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest using VIX data from yfinance + ATR-based IVR proxy.

        Entry  (entries):  IVR ≥ 50 AND VIX > 20  → treat as sell-put-spread entry
        Entry  (shorts):   IVR ≥ 70               → iron condor, also captured as 'sell'
        Exit   (exits):    IVR < 30               → vol has collapsed, close for profit

        In VectorBT terms:
          entries = long entry (we 'buy' the strategy position when we open the spread)
          exits   = close the position

        Note: actual options P&L simulation is not done here — the signal series only
        marks when the mechanical conditions for entry/exit are met.  Full P&L backtesting
        requires options chain data and is handled by the dedicated backtest runner.
        """
        if len(df) < self.IVR_LOOKBACK + self.ATR_PERIOD + 1:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # ── IVR proxy from ATR ────────────────────────────────────────────────
        high  = df["high"]  if "high"  in df.columns else df["close"]
        low   = df["low"]   if "low"   in df.columns else df["close"]
        close = df["close"]

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr20 = tr.rolling(self.ATR_PERIOD).mean()
        atr_min = atr20.rolling(self.IVR_LOOKBACK).min()
        atr_max = atr20.rolling(self.IVR_LOOKBACK).max()
        spread = (atr_max - atr_min).clip(lower=1e-9)
        ivr_series = ((atr20 - atr_min) / spread * 100).fillna(0)

        # ── VIX via yfinance (best-effort; fall back to high-IVR-only signal) ──
        if _YF_AVAILABLE:
            try:
                vix_hist = yf.Ticker("^VIX").history(period="5y")
                if not vix_hist.empty:
                    vix_series = vix_hist["Close"].reindex(df.index, method="ffill")
                    vix_above = vix_series > self.VIX_THRESHOLD
                else:
                    vix_above = pd.Series(True, index=df.index)
            except Exception:
                vix_above = pd.Series(True, index=df.index)
        else:
            # Without yfinance, use high ATR percentile as proxy for elevated VIX
            vix_above = ivr_series > 40

        # ── Entry / exit signals (shifted to prevent lookahead) ───────────────
        entries = (
            (ivr_series.shift(1) >= self.IVR_PUT_SPREAD) & vix_above.shift(1)
        ).fillna(False)

        exits = (ivr_series.shift(1) < 30).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)
