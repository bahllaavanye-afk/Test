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
  VIX futures typically trade in contango (long‑dated > short‑dated) because:
  1. Investors pay an insurance premium for distant protection.
  2. Short‑term VIX spikes quickly revert.
  3. VIXY (1‑2 month futures) continuously rolls into more expensive contracts.

  Roll yield = (near_price - far_price) / far_price per roll period.
  In contango this is negative for VIXY holders → systematic short opportunity.

VIX ETP proxies (Alpaca‑tradeable):
  VIXY = ProShares VIX Short‑Term Futures ETF (1‑2 month)
  VIXM = ProShares VIX Mid‑Term Futures ETF (4‑7 month)

Term Structure Ratio = VIXY_close / VIXM_close:
  Ratio < 0.90  → steep contango → SHORT VIXY (collect roll yield)
  Ratio > 1.05  → backwardation → SHORT VIXY (spike reversion trade)
  0.90–1.05     → neutral zone → no position

Kelly‑fraction confidence:
  In contango: confidence = (0.90 - ratio) / 0.90  (larger discount = larger bet)
  In backwardation: confidence = (ratio - 1.05) / 1.05  (larger spike = larger bet)
  Both capped at 0.90.

Documented Sharpe: 1.2‑1.8 (Simon & Campasano 2014, various replication studies)
Risk: enormous tail risk during volatility spikes (VIX >50); circuit‑breakers mandatory.
"""

from datetime import date, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"

VIXY = "VIXY"  # ProShares VIX Short‑Term Futures ETF
VIXM = "VIXM"  # ProShares VIX Mid‑Term Futures ETF
VIX = "^VIX"   # CBOE Volatility Index (used as a risk filter)


class VolTermStructureStrategy(AbstractStrategy):
    """
    VIX term structure carry — short VIXY in contango, manage tail risk in backwardation.

    Core insight: VIXY holders pay ~40‑70% p.a. in roll costs during normal
    contango regimes. The strategy captures this roll yield by maintaining
    a short VIXY position, sized by the steepness of the term structure.
    """

    name = "vol_term_structure"
    display_name = "VIX Term Structure Carry"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0  # hourly — intraday regime changes matter

    # --------------------------------------------------------------------- #
    # Configuration constants
    # --------------------------------------------------------------------- #
    CONTANGO_THRESHOLD = 0.90   # VIXY/VIXM < 0.90 → steep contango
    BACKWARDATION_THRESHOLD = 1.05   # VIXY/VIXM > 1.05 → backwardation
    NEUTRAL_LOWER = 0.90
    NEUTRAL_UPPER = 1.05

    MAX_CONFIDENCE = 0.90          # cap Kelly fraction
    STOP_RATIO = 1.20              # emergency stop: exit if ratio > 1.20
    LOOKBACK_DAYS = 30             # days of bars to fetch for ratio calculation
    SIGNAL_SMOOTH_WINDOW = 5       # smoothing window (bars)
    CONFIDENCE_FLOOR = 0.05        # minimum confidence to trigger a trade
    VIX_RISK_LIMIT = 50.0          # VIX level above which we stay flat
    TREND_WINDOW = 3               # periods to evaluate ratio trend
    TREND_THRESHOLD = 0.015        # minimum absolute change in smoothed ratio

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)

    # --------------------------------------------------------------------- #
    # Data helpers
    # --------------------------------------------------------------------- #
    async def _fetch_bars(self, symbol: str, days: int = LOOKBACK_DAYS) -> pd.Series:
        """Fetch *daily* closing prices for ``symbol``."""
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

            series = pd.Series(
                {b["t"]: float(b["c"]) for b in bars},
                name=symbol,
                dtype=float,
            )
            series.index = pd.to_datetime(series.index)
            return series.sort_index()
        except Exception:
            return pd.Series(dtype=float, name=symbol)

    # --------------------------------------------------------------------- #
    # Core calculations
    # --------------------------------------------------------------------- #
    @staticmethod
    def _roll_yield_annualized(vixy_close: float, vixm_close: float,
                               days_to_roll: int = 30) -> float:
        """
        Approximate annualized roll yield for a short VIXY position.

        Roll yield ≈ (VIXM - VIXY) / VIXY × (365 / days_to_roll).
        Positive roll yield = VIXY is cheaper = normal contango = profitable to short.
        """
        if vixm_close <= 0 or vixy_close <= 0:
            return 0.0
        daily_roll = (vixm_close - vixy_close) / vixy_close
        return float(daily_roll * 365.0 / days_to_roll)

    # --------------------------------------------------------------------- #
    # Signal generation
    # --------------------------------------------------------------------- #
    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Generate a short VIXY signal when the term structure indicates a
        profitable carry (contango) or a mean‑reversion opportunity
        (backwardation). The method applies several confirmation filters:

        1. **Liquidity filter** – require a minimum of 5 overlapping price points.
        2. **Risk filter** – stay flat when the VIX index exceeds ``VIX_RISK_LIMIT``.
        3. **Trend filter** – the smoothed ratio must be moving away from the
           neutral zone (decreasing in contango, increasing in backwardation)
           by at least ``TREND_THRESHOLD`` over ``TREND_WINDOW`` periods.
        4. **Confidence floor** – ignore weak signals (confidence < ``CONFIDENCE_FLOOR``).

        Exit logic (handled downstream) uses a fixed stop‑loss (15 % adverse move)
        and a take‑profit (20 % favourable move) together with the emergency
        ``STOP_RATIO`` guard.
        """
        import asyncio

        # ----------------------------------------------------------------- #
        # 1️⃣  Pull required market data in parallel
        # ----------------------------------------------------------------- #
        vixy_series, vixm_series, vix_series = await asyncio.gather(
            self._fetch_bars(VIXY, self.LOOKBACK_DAYS),
            self._fetch_bars(VIXM, self.LOOKBACK_DAYS),
            self._fetch_bars(VIX, self.LOOKBACK_DAYS),
        )

        # ----------------------------------------------------------------- #
        # 2️⃣  Basic sanity checks
        # ----------------------------------------------------------------- #
        if vixy_series.empty or vixm_series.empty or vix_series.empty:
            return None

        # Align all series on common dates
        common_idx = vixy_series.index.intersection(vixm_series.index).intersection(vix_series.index)
        if len(common_idx) < max(self.SIGNAL_SMOOTH_WINDOW, self.TREND_WINDOW) + 1:
            return None

        vixy = vixy_series[common_idx]
        vixm = vixm_series[common_idx]
        vix = vix_series[common_idx]

        # ----------------------------------------------------------------- #
        # 3️⃣  Risk filter – stay flat if VIX is too high
        # ----------------------------------------------------------------- #
        latest_vix = float(vix.iloc[-1])
        if latest_vix >= self.VIX_RISK_LIMIT:
            return None

        # ----------------------------------------------------------------- #
        # 4️⃣  Compute term‑structure ratio and smoothing
        # ----------------------------------------------------------------- #
        ratio_series = vixy / vixm.clip(lower=0.01)  # avoid division by zero
        smoothed_ratio = ratio_series.rolling(
            self.SIGNAL_SMOOTH_WINDOW, min_periods=2
        ).mean()

        # Current values (most recent bar)
        current_ratio = float(ratio_series.iloc[-1])
        smoothed = float(smoothed_ratio.iloc[-1])
        vixy_price = float(vixy.iloc[-1])
        vixm_price = float(vixm.iloc[-1])

        # ----------------------------------------------------------------- #
        # 5️⃣  Emergency stop – avoid entering when ratio is extreme
        # ----------------------------------------------------------------- #
        if current_ratio > self.STOP_RATIO:
            return None

        # ----------------------------------------------------------------- #
        # 6️⃣  Determine regime & apply trend confirmation
        # ----------------------------------------------------------------- #
        in_contango = smoothed < self.CONTANGO_THRESHOLD
        in_backwardation = smoothed > self.BACKWARDATION_THRESHOLD

        if not in_contango and not in_backwardation:
            return None  # Neutral zone – no trade

        # Trend filter: compare current smoothed value with its value ``TREND_WINDOW`` bars ago
        prior_smoothed = float(smoothed_ratio.iloc[-self.TREND_WINDOW])
        ratio_trend = smoothed - prior_smoothed

        if in_contango and ratio_trend > -self.TREND_THRESHOLD:
            # Ratio not falling enough → weak contango signal
            return None
        if in_backwardation and ratio_trend < self.TREND_THRESHOLD:
            # Ratio not rising enough → weak backwardation signal
            return None

        # ----------------------------------------------------------------- #
        # 7️⃣  Confidence calculation (Kelly‑fraction style)
        # ----------------------------------------------------------------- #
        if in_contango:
            raw_conf = (self.CONTANGO_THRESHOLD - smoothed) / self.CONTANGO_THRESHOLD
            regime_label = "contango"
        else:
            raw_conf = (smoothed - self.BACKWARDATION_THRESHOLD) / self.BACKWARDATION_THRESHOLD
            regime_label = "backwardation_reversion"

        confidence = min(max(raw_conf, 0.0), self.MAX_CONFIDENCE)

        if confidence < self.CONFIDENCE_FLOOR:
            return None

        # ----------------------------------------------------------------- #
        # 8️⃣  Position sizing – use confidence as a fraction of max capital.
        #     The actual sizing logic is delegated to the execution layer;
        #     we simply embed the confidence factor.
        # ----------------------------------------------------------------- #
        position_size = confidence  # interpreter will translate to appropriate qty

        # ----------------------------------------------------------------- #
        # 9️⃣  Exit parameters
        # ----------------------------------------------------------------- #
        stop_loss_price = round(vixy_price * 1.15, 4)   # 15 % adverse move
        take_profit_price = round(vixy_price * 0.80, 4)  # 20 % favourable move

        # ----------------------------------------------------------------- #
        # 10️⃣  Build and return the signal
        # ----------------------------------------------------------------- #
        signal = Signal(
            side="short",
            symbol=VIXY,
            price=vixy_price,
            quantity=position_size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            confidence=confidence,
            meta={
                "regime": regime_label,
                "ratio": current_ratio,
                "smoothed_ratio": smoothed,
                "vix_current": latest_vix,
                "roll_yield_annualized": self._roll_yield_annualized(vixy_price, vixm_price),
            },
        )
        return signal

    # --------------------------------------------------------------------- #
    # Optional: back‑testing helper (kept unchanged)
    # --------------------------------------------------------------------- #
    def backtest(self, data: pd.DataFrame) -> BacktestSignals:
        # Placeholder – the concrete back‑testing implementation lives in the
        # parent class.  Keeping the method here preserves the original API.
        return super().backtest(data)