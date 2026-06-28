"""
Volatility Surface Arbitrage — VIX Term Structure Carry
=========================================================
Academic basis:
  - Mixon (2007) "What does implied volatility skew measure?" — VIX term structure
    predicts subsequent volatility ETF returns.
  - Simon & Campasano (2014) "The VIX Futures Basis: Evidence and Trading Strategies"
    Journal of Derivatives — VIX term structure carry earns ~15% p.a. risk-adjusted.
  - Whaley (2009), Eraker & Wu (2017): short-term VIX ETPs structurally lose money
    in contango via negative roll yield; strategies exploiting this are robust.

Mechanism:
  VIX futures typically trade in contango (long-dated > short-dated) because:
  1. Investors pay an insurance premium for distant protection.
  2. Short-term VIX spikes quickly revert.
  3. VIXY (1-2 month futures) continuously rolls into more expensive contracts.

  Roll yield = (near_price - far_price) / far_price per roll period.
  In contango this is negative for VIXY holders → systematic short opportunity.

VIX ETP proxies (Alpaca-tradeable):
  VIXY = ProShares VIX Short-Term Futures ETF (1-2 month)
  VIXM = ProShares VIX Mid-Term Futures ETF (4-7 month)

Term Structure Ratio = VIXY_close / VIXM_close:
  Ratio < 0.90  → steep contango → SHORT VIXY (collect roll yield)
  Ratio > 1.05  → backwardation → SHORT VIXY (spike reversion trade)
  0.90–1.05     → neutral zone → no position

Kelly-fraction confidence:
  In contango: confidence = (0.90 - ratio) / 0.90  (larger discount = larger bet)
  In backwardation: confidence = (ratio - 1.05) / 1.05  (larger spike = larger bet)
  Both capped at 0.90.

Documented Sharpe: 1.2-1.8 (Simon & Campasano 2014, various replication studies)
Risk: enormous tail risk during volatility spikes (VIX >50); circuit-breakers mandatory.
"""

from datetime import date, timedelta
from typing import Optional

import httpx
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"

VIXY = "VIXY"  # ProShares VIX Short-Term Futures ETF
VIXM = "VIXM"  # ProShares VIX Mid-Term Futures ETF
VIX_INDEX = "^VIX"  # CBOE Volatility Index ticker (Alpaca provides this symbol)


class VolTermStructureStrategy(AbstractStrategy):
    """
    VIX term structure carry — short VIXY in contango, manage tail risk in backwardation.

    Core insight: VIXY holders pay ~40-70% p.a. in roll costs during normal
    contango regimes. The strategy captures this roll yield by maintaining
    a short VIXY position, sized by the steepness of the term structure.
    """

    name = "vol_term_structure"
    display_name = "VIX Term Structure Carry"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0  # hourly — intraday regime changes matter

    # Term structure thresholds
    CONTANGO_THRESHOLD = 0.90   # VIXY/VIXM < 0.90 → steep contango → short VIXY
    BACKWARDATION_THRESHOLD = 1.05   # VIXY/VIXM > 1.05 → backwardation (spike) → short VIXY for reversion
    NEUTRAL_LOWER = 0.90
    NEUTRAL_UPPER = 1.05

    # Risk management
    MAX_CONFIDENCE = 0.90   # cap Kelly fraction
    STOP_RATIO = 1.20   # emergency stop: exit if ratio > 1.20 (VIX spike)
    LOOKBACK_DAYS = 30     # days of bars to fetch for current ratio

    # Rolling window for signal smoothing
    SIGNAL_SMOOTH_WINDOW = 5    # 5-bar (hour) smoothed ratio

    # Volatility filter
    VIX_TAIL_RISK_THRESHOLD = 40.0  # if VIX index > 40, suppress signals # MUTATION: added volatility filter to avoid tail risk

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)

    async def _fetch_bars(self, symbol: Optional[str], days: int = 30) -> pd.Series:
        """Fetch daily closing prices for a given symbol."""
        if not symbol:
            return pd.Series(dtype=float, name="unknown")

        # Guard against nonsensical day values
        days = max(1, days)

        start = (date.today() - timedelta(days=days + 10)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v1/bars/daily?symbols={symbol}&start={start}",
                    headers=alpaca_headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                df = pd.DataFrame(data.get(symbol, []))
                if df.empty:
                    return pd.Series(dtype=float, name=symbol)
                return df.set_index("t")["c"]
        except Exception as e:
            self.logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return pd.Series(dtype=float, name=symbol)

    async def _fetch_vix_index(self) -> float:
        """Fetch the latest VIX index level."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v1/bars/daily?symbols={VIX_INDEX}&limit=1",
                    headers=alpaca_headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                df = pd.DataFrame(data.get(VIX_INDEX, []))
                if df.empty:
                    return float("nan")
                return float(df.iloc[0]["c"])
        except Exception as e:
            self.logger.error(f"Failed to fetch VIX index: {e}")
            return float("nan")

    async def generate_signals(self, timestamp: pd.Timestamp) -> BacktestSignals:
        """Generate trading signals based on the VIX term structure."""
        # Fetch recent closes for VIXY and VIXM
        vixy_series = await self._fetch_bars(VIXY, self.LOOKBACK_DAYS)
        vixm_series = await self._fetch_bars(VIXM, self.LOOKBACK_DAYS)

        if vixy_series.empty or vixm_series.empty:
            self.logger.warning("Missing price data for VIXY or VIXM")
            return BacktestSignals.empty()

        # Align series by date index
        df = pd.concat([vixy_series, vixm_series], axis=1, join="inner")
        df.columns = ["VIXY", "VIXM"]
        df["ratio"] = df["VIXY"] / df["VIXM"]

        # Apply smoothing
        df["smooth_ratio"] = df["ratio"].rolling(self.SIGNAL_SMOOTH_WINDOW).mean()

        # Use the most recent smoothed ratio
        current_ratio = df["smooth_ratio"].iloc[-1]

        # Fetch current VIX index to apply tail risk filter
        vix_index = await self._fetch_vix_index()

        # If VIX index is above tail risk threshold, suppress any short signal
        if not pd.isna(vix_index) and vix_index > self.VIX_TAIL_RISK_THRESHOLD:
            self.logger.info(
                f"VIX index {vix_index:.2f} exceeds tail‑risk threshold "
                f"{self.VIX_TAIL_RISK_THRESHOLD}, suppressing signal."
            )
            return BacktestSignals.empty()

        # Emergency stop condition
        if current_ratio > self.STOP_RATIO:
            self.logger.info(
                f"Emergency stop triggered: ratio {current_ratio:.2f} > {self.STOP_RATIO}"
            )
            return BacktestSignals.empty()

        # Determine position and confidence
        if current_ratio < self.NEUTRAL_LOWER:
            # Contango – short VIXY
            confidence = (self.NEUTRAL_LOWER - current_ratio) / self.NEUTRAL_LOWER
            position = -1  # short
        elif current_ratio > self.NEUTRAL_UPPER:
            # Backwardation – short VIXY for reversion
            confidence = (current_ratio - self.NEUTRAL_UPPER) / self.NEUTRAL_UPPER
            position = -1  # short
        else:
            # Neutral zone – no position
            return BacktestSignals.empty()

        # Cap confidence
        confidence = min(confidence, self.MAX_CONFIDENCE)

        signal = Signal(
            asset=VIXY,
            direction=position,
            confidence=confidence,
            timestamp=timestamp,
            metadata={"ratio": current_ratio, "vix_index": vix_index},
        )
        return BacktestSignals([signal])
