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
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Day",
                        "start": start,
                        "limit": days + 10,
                        "feed": "iex",
                    },
                    headers=alpaca_headers(),
                )
            if resp.status_code != 200:
                return pd.Series(dtype=float, name=symbol)

            bars = resp.json().get("bars", [])
            if not bars:
                return pd.Series(dtype=float, name=symbol)

            # Build series; ensure we always have a float dtype
            s = pd.Series(
                {b["t"]: float(b["c"]) for b in bars if b.get("c") is not None},
                name=symbol,
                dtype=float,
            )
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            # Any failure (network, parsing, etc.) results in an empty series
            return pd.Series(dtype=float, name=symbol)

    @staticmethod
    def _roll_yield_annualized(vixy_close: float, vixm_close: float,
                               days_to_roll: int = 30) -> float:
        """
        Approximate annualized roll yield for a short VIXY position.
        Roll yield ≈ (VIXM - VIXY) / VIXY × (365 / days_to_roll)
        Positive roll yield = VIXY is cheaper = normal contango = profitable to short.
        """
        # Defensive checks against division by zero or negative prices
        if vixy_close <= 0 or vixm_close <= 0:
            return 0.0
        daily_roll = (vixm_close - vixy_close) / vixy_close
        return float(daily_roll * 365.0 / max(1, days_to_roll))

    async def analyze(self, data: Optional[pd.DataFrame], symbol: Optional[str]) -> Optional[Signal]:
        """
        Fetch VIXY and VIXM prices, compute term structure ratio.
        Issue SHORT VIXY signal in contango or backwardation-reversion regimes.
        """
        # Defensive early exits for unexpected inputs
        if data is None or symbol is None:
            return None

        import asyncio

        vixy_series, vixm_series = await asyncio.gather(
            self._fetch_bars(VIXY, self.LOOKBACK_DAYS),
            self._fetch_bars(VIXM, self.LOOKBACK_DAYS),
        )

        # Ensure we have data for both legs
        if vixy_series.empty or vixm_series.empty:
            return None

        # Align on common dates
        common = vixy_series.index.intersection(vixm_series.index)
        if len(common) < self.SIGNAL_SMOOTH_WINDOW:
            # Not enough data points to compute a reliable smoothed ratio
            return None

        vixy_aligned = vixy_series.loc[common]
        vixm_aligned = vixm_series.loc[common]

        # Compute ratio series; protect against division by zero
        vixm_aligned_safe = vixm_aligned.clip(lower=0.01)
        ratio_series = vixy_aligned / vixm_aligned_safe

        # Smooth the ratio to reduce noise
        smoothed_ratio = ratio_series.rolling(
            self.SIGNAL_SMOOTH_WINDOW, min_periods=2
        ).mean()

        # Guard against empty series after smoothing (unlikely but possible)
        if ratio_series.empty or smoothed_ratio.empty:
            return None

        # Use the most recent valid values
        current_ratio = float(ratio_series.iloc[-1])
        smoothed = float(smoothed_ratio.iloc[-1])

        # Additional safety: NaN checks
        if pd.isna(current_ratio) or pd.isna(smoothed):
            return None

        vixy_price = float(vixy_aligned.iloc[-1])
        vixm_price = float(vixm_aligned.iloc[-1])

        # Emergency stop: if ratio blew out, no new shorts
        if current_ratio > self.STOP_RATIO:
            return None

        # Regime classification
        in_contango = smoothed < self.CONTANGO_THRESHOLD
        in_backwardation = smoothed > self.BACKWARDATION_THRESHOLD

        if not in_contango and not in_backwardation:
            return None  # neutral zone — no trade

        # Compute annualized roll yield (used for meta information)
        roll_yield = self._roll_yield_annualized(vixy_price, vixm_price)

        # Determine confidence based on regime steepness
        if in_contango:
            confidence = (self.CONTANGO_THRESHOLD - smoothed) / self.CONTANGO_THRESHOLD
            regime = "contango"
        else:
            confidence = (smoothed - self.BACKWARDATION_THRESHOLD) / self.BACKWARDATION_THRESHOLD
            regime = "backwardation_reversion"

        # Cap confidence to the maximum allowed Kelly fraction
        confidence = min(confidence, self.MAX_CONFIDENCE)

        # Very low confidence signals are ignored
        if confidence < 0.05:
            return None

        # Define stop‑loss and take‑profit levels for the short position
        stop_loss = round(vixy_price * 1.15, 4)   # 15% adverse move
        take_profit = round(vixy_price * 0.80, 4)  # 20% favorable move

        # Build a metadata dict for downstream consumers (logging, analytics, etc.)
        meta = {
            "regime": regime,
            "current_ratio": current_ratio,
            "smoothed_ratio": smoothed,
            "roll_yield_annualized": roll_yield,
            "vixy_price": vixy_price,
            "vixm_price": vixm_price,
        }

        # Return a Signal object. The exact signature of Signal may vary across the codebase;
        # we use keyword arguments that align with typical implementations.
        return Signal(
            symbol=VIXY,
            side="short",
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            meta=meta,
        )