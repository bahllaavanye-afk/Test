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

import httpx
import numpy as np
import pandas as pd

from app.config import settings
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
    CONTANGO_THRESHOLD      = 0.90   # VIXY/VIXM < 0.90 → steep contango → short VIXY
    BACKWARDATION_THRESHOLD = 1.05   # VIXY/VIXM > 1.05 → backwardation (spike) → short VIXY for reversion
    NEUTRAL_LOWER           = 0.90
    NEUTRAL_UPPER           = 1.05

    # Risk management
    MAX_CONFIDENCE     = 0.90   # cap Kelly fraction
    STOP_RATIO         = 1.20   # emergency stop: exit if ratio > 1.20 (VIX spike)
    LOOKBACK_DAYS      = 30     # days of bars to fetch for current ratio

    # Rolling window for signal smoothing
    SIGNAL_SMOOTH_WINDOW = 5    # 5-bar (hour) smoothed ratio

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

    async def _fetch_bars(self, symbol: str, days: int = 30) -> pd.Series:
        """Fetch daily closing prices."""
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
                    headers=self._headers(),
                )
            if resp.status_code != 200:
                return pd.Series(dtype=float, name=symbol)
            bars = resp.json().get("bars", [])
            if not bars:
                return pd.Series(dtype=float, name=symbol)
            s = pd.Series(
                {b["t"]: float(b["c"]) for b in bars},
                name=symbol,
            )
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            return pd.Series(dtype=float, name=symbol)

    @staticmethod
    def _roll_yield_annualized(vixy_close: float, vixm_close: float,
                                days_to_roll: int = 30) -> float:
        """
        Approximate annualized roll yield for a short VIXY position.
        Roll yield ≈ (VIXM - VIXY) / VIXY × (365 / days_to_roll)
        Positive roll yield = VIXY is cheaper = normal contango = profitable to short.
        """
        if vixm_close <= 0 or vixy_close <= 0:
            return 0.0
        daily_roll = (vixm_close - vixy_close) / vixy_close
        return float(daily_roll * 365.0 / days_to_roll)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Fetch VIXY and VIXM prices, compute term structure ratio.
        Issue SHORT VIXY signal in contango or backwardation-reversion regimes.
        """
        import asyncio

        vixy_series, vixm_series = await asyncio.gather(
            self._fetch_bars(VIXY, self.LOOKBACK_DAYS),
            self._fetch_bars(VIXM, self.LOOKBACK_DAYS),
        )

        if vixy_series.empty or vixm_series.empty:
            return None

        # Align on common dates
        common = vixy_series.index.intersection(vixm_series.index)
        if len(common) < 5:
            return None

        vixy_aligned = vixy_series[common]
        vixm_aligned = vixm_series[common]

        # Compute ratio series and smooth
        ratio_series = vixy_aligned / vixm_aligned.clip(lower=0.01)
        smoothed_ratio = ratio_series.rolling(
            self.SIGNAL_SMOOTH_WINDOW, min_periods=2
        ).mean()

        current_ratio  = float(ratio_series.iloc[-1])
        smoothed       = float(smoothed_ratio.iloc[-1])
        vixy_price     = float(vixy_aligned.iloc[-1])
        vixm_price     = float(vixm_aligned.iloc[-1])

        # Emergency stop: if ratio blew out, no new shorts
        if current_ratio > self.STOP_RATIO:
            return None

        # Regime classification
        in_contango      = smoothed < self.CONTANGO_THRESHOLD
        in_backwardation = smoothed > self.BACKWARDATION_THRESHOLD

        if not in_contango and not in_backwardation:
            return None  # neutral zone — no trade

        # Compute annualized roll yield
        roll_yield = self._roll_yield_annualized(vixy_price, vixm_price)

        if in_contango:
            # Classic carry trade: short VIXY to collect negative roll
            # Confidence: how steep is the contango?
            confidence = min(
                (self.CONTANGO_THRESHOLD - smoothed) / self.CONTANGO_THRESHOLD,
                self.MAX_CONFIDENCE,
            )
            regime = "contango"
        else:
            # Backwardation spike: short VIXY betting on mean reversion
            # Higher ratio = larger spike = stronger reversion expected
            confidence = min(
                (smoothed - self.BACKWARDATION_THRESHOLD) / self.BACKWARDATION_THRESHOLD,
                self.MAX_CONFIDENCE,
            )
            regime = "backwardation_reversion"

        if confidence < 0.05:
            return None

        # Stop-loss price: if VIXY rises 15% from here, cover short
        stop_loss   = round(vixy_price * 1.15, 4)
        take_profit = round(vixy_price * 0.80, 4)  # 20% gain on short

        return Signal(
            symbol=VIXY,
            side="sell",  # SHORT VIXY
            confidence=round(confidence, 4),
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=take_profit,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "vixy_price":        round(vixy_price,     4),
                "vixm_price":        round(vixm_price,     4),
                "current_ratio":     round(current_ratio,  4),
                "smoothed_ratio":    round(smoothed,       4),
                "regime":            regime,
                "roll_yield_ann":    round(roll_yield,     4),
                "contango_threshold":      self.CONTANGO_THRESHOLD,
                "backwardation_threshold": self.BACKWARDATION_THRESHOLD,
                "academic_ref": "Simon & Campasano (2014) VIX Futures Basis",
                "risk_note": "tail_risk_vix_spike_stop_at_ratio_1.20",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Single-ETF backtest on VIXY daily bars.
        Requires a DataFrame with both 'vixy' and 'vixm' columns OR a single
        'close' column (treated as VIXY) with 'vixm_close' if available.

        Short entries: ratio < CONTANGO_THRESHOLD or ratio > BACKWARDATION_THRESHOLD.
        Short exits: ratio enters neutral zone or hits stop.
        Apply shift(1) for lookahead prevention.
        """
        if "vixy" in df.columns and "vixm" in df.columns:
            vixy = df["vixy"].astype(float)
            vixm = df["vixm"].astype(float)
        elif "close" in df.columns and "vixm_close" in df.columns:
            vixy = df["close"].astype(float)
            vixm = df["vixm_close"].astype(float)
        elif "close" in df.columns:
            # Degenerate: approximate ratio using own rolling stats
            # (for API compatibility when only VIXY data available)
            close = df["close"].astype(float)
            rolling_high = close.rolling(20, min_periods=10).quantile(0.75)
            vixy = close
            vixm = rolling_high.fillna(close)
        else:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty,
                                   short_entries=empty, short_exits=empty)

        ratio = vixy / vixm.clip(lower=0.01)
        smoothed = ratio.rolling(self.SIGNAL_SMOOTH_WINDOW, min_periods=2).mean()

        in_contango      = smoothed < self.CONTANGO_THRESHOLD
        in_backwardation = (smoothed > self.BACKWARDATION_THRESHOLD) & (smoothed < self.STOP_RATIO)
        in_neutral       = ~in_contango & ~in_backwardation

        # Short entry: either contango or backwardation reversion signal
        short_entries = (in_contango | in_backwardation).shift(1).fillna(False)
        # Short exit: back to neutral or stop blown out
        short_exits   = (in_neutral | (smoothed >= self.STOP_RATIO)).shift(1).fillna(False)

        # No long leg (this is a volatility carry / short-vol strategy)
        empty = pd.Series(False, index=df.index)

        return BacktestSignals(
            entries=empty,
            exits=empty,
            short_entries=short_entries,
            short_exits=short_exits,
        )
