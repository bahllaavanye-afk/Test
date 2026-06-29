"""
Order Flow Imbalance (OFI)
============================
Academic basis:
  - Cont, Kukanov, Stoikov (2014) "The Price Impact of Order Book Events"
    Journal of Finance — order flow imbalance (OFI) is empirically the strongest
    short-term predictor of mid-price movements, outperforming all classical
    technical indicators at 1-5 minute horizons.
  - Kolm, Turiel, Westray (2021) "Deep Order Flow Imbalance" extended OFI to
    multi-level LOB using deep learning, confirming the baseline signal.

Mechanism:
  True OFI = Σ_t [ΔV_bid_t - ΔV_ask_t]  (changes at best bid and ask)
  Positive OFI → net buying pressure → price moves up in the next 1-5 min.

OHLCV Approximation (Alpaca 1-min bars, no full order book):
  For each 1-min bar:
    if close > open  → net buy flow  = +volume
    if close < open  → net sell flow = -volume
    if close == open → neutral       = 0
  OFI_t = sign(close_t - open_t) × volume_t
  OFI_ratio_N = Σ_{t-N}^{t} OFI_t / Σ_{t-N}^{t} volume_t  ∈ [-1, 1]

Entry: OFI_ratio > 0.60 AND 5-bar price momentum > 0
Exit:  OFI_ratio < 0.10 OR position hits +0.5% take-profit

Documented Sharpe: ~1.5-2.0 in academic studies (implementation varies by execution quality)
"""

from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.utils.logging import logger

_DATA_BASE = "https://data.alpaca.markets"

# Universe: high-volume equities with tight spreads (best OFI signal quality)
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "SPY", "QQQ", "AMD", "INTC", "JPM", "BAC", "GS",
]


class OrderFlowImbalanceStrategy(AbstractStrategy):
    """
    Intraday order flow imbalance using OHLCV-derived buying/selling pressure.

    Runs every minute. Fetches the last 20 1‑minute bars, computes a
    volume‑weighted signed flow ratio, and signals when imbalance is extreme
    and confirmed by additional filters (price momentum, volume spike,
    and short‑term trend).
    """

    name = "order_flow_imbalance"
    display_name = "Order Flow Imbalance"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 60.0  # runs every minute

    # Signal thresholds
    OFI_ENTRY_THRESHOLD = 0.60   # |OFI_ratio| > 0.60 to consider entry
    OFI_EXIT_THRESHOLD = 0.10    # |OFI_ratio| < 0.10 to exit
    MOMENTUM_BARS = 5            # bars for 5‑min momentum confirmation
    OFI_LOOKBACK = 10            # rolling window for OFI computation
    BARS_TO_FETCH = 20           # 1‑min bars to fetch from Alpaca

    # Additional filters
    VOLUME_SPIKE_MULTIPLIER = 1.20   # current volume must exceed avg by 20 %
    MA_PERIOD = 20                  # simple moving average period for trend filter

    # Risk parameters
    TAKE_PROFIT_PCT = 0.005       # 0.5 % take‑profit
    STOP_LOSS_PCT = 0.003         # 0.3 % stop‑loss

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    async def _fetch_minute_bars(self, symbol: str) -> pd.DataFrame:
        """Fetch the most recent 1‑minute bars for *symbol* from Alpaca."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Min",
                        "limit": self.BARS_TO_FETCH,
                        "feed": "iex",
                    },
                    headers=alpaca_headers(),
                )
            if resp.status_code != 200:
                logger.debug(
                    "order_flow_imbalance: failed to fetch bars",
                    symbol=symbol,
                    status_code=resp.status_code,
                )
                return pd.DataFrame()

            bars = resp.json().get("bars", [])
            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame(bars)
            df = df.rename(
                columns={
                    "t": "time",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
        except Exception as exc:  # pragma: no cover
            logger.error(
                "order_flow_imbalance: exception while fetching bars",
                symbol=symbol,
                error=str(exc),
            )
            return pd.DataFrame()

    @staticmethod
    def _compute_ofi(df: pd.DataFrame, lookback: int = 10) -> tuple[float, float]:
        """
        Compute OFI_ratio over the last ``lookback`` bars and the 5‑bar momentum.

        Returns
        -------
        ofi_ratio : float
            Signed flow ratio in the range [-1, 1].
        momentum : float
            Simple price return over ``MOMENTUM_BARS`` bars.
        """
        if len(df) < lookback + 5:
            return 0.0, 0.0

        window = df.iloc[-lookback:]
        signed_flow = np.sign(window["close"] - window["open"]) * window["volume"]
        total_volume = window["volume"].sum()
        ofi_ratio = float(signed_flow.sum() / total_volume) if total_volume > 0 else 0.0

        # 5‑bar price momentum (percentage change)
        momentum = float(df["close"].iloc[-1] / df["close"].iloc[-5] - 1.0)

        return ofi_ratio, momentum

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Produce a live signal for *symbol*.

        The method fetches the latest minute bars, computes the OFI ratio,
        validates additional filters (volume spike, short‑term trend) and,
        if all conditions are satisfied, returns a :class:`Signal` instance.
        """
        if symbol not in DEFAULT_SYMBOLS:
            logger.debug(
                "order_flow_imbalance: symbol outside default universe", symbol=symbol
            )

        df = await self._fetch_minute_bars(symbol)
        if df.empty or len(df) < self.OFI_LOOKBACK + self.MOMENTUM_BARS:
            return None

        ofi_ratio, momentum = self._compute_ofi(df, self.OFI_LOOKBACK)
        current_price = float(df["close"].iloc[-1])

        # ----- Additional confirmation filters -----
        # 1. Volume spike: current minute volume must exceed recent average.
        recent_avg_vol = df["volume"].iloc[-self.OFI_LOOKBACK :].mean()
        if recent_avg_vol == 0 or df["volume"].iloc[-1] < recent_avg_vol * self.VOLUME_SPIKE_MULTIPLIER:
            return None

        # 2. Short‑term trend filter using a simple moving average.
        if len(df) < self.MA_PERIOD:
            return None
        ma = df["close"].rolling(self.MA_PERIOD).mean().iloc[-1]

        # 3. Ensure OFI sign matches trend direction.
        if ofi_ratio > 0 and current_price < ma:
            return None
        if ofi_ratio < 0 and current_price > ma:
            return None

        # ----- Core entry logic -----
        abs_ofi = abs(ofi_ratio)
        if abs_ofi < self.OFI_ENTRY_THRESHOLD:
            return None

        # Momentum must be aligned with OFI direction.
        if ofi_ratio > 0 and momentum <= 0:
            return None
        if ofi_ratio < 0 and momentum >= 0:
            return None

        side = "buy" if ofi_ratio > 0 else "sell"
        confidence = min(abs_ofi, 1.0)

        # Price targets (rounded to 4 dp for consistency)
        if side == "buy":
            take_profit = round(current_price * (1 + self.TAKE_PROFIT_PCT), 4)
            stop_loss = round(current_price * (1 - self.STOP_LOSS_PCT), 4)
        else:
            take_profit = round(current_price * (1 - self.TAKE_PROFIT_PCT), 4)
            stop_loss = round(current_price * (1 + self.STOP_LOSS_PCT), 4)

        return Signal(
            symbol=symbol,
            side=side,
            confidence=round(confidence, 4),
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=take_profit,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "ofi_ratio": round(ofi_ratio, 4),
                "momentum_5bar": round(momentum, 6),
                "ofi_lookback_bars": self.OFI_LOOKBACK,
                "current_price": current_price,
                "average_volume": round(recent_avg_vol, 2),
                "moving_average": round(ma, 4),
                "bars_used": len(df),
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized back‑test implementation.

        The routine mirrors the live logic but works on a full OHLCV series.
        It returns four boolean Series:

        * ``entries``      – long entry signals
        * ``exits``        – long exit signals
        * ``short_entries``– short entry signals
        * ``short_exits``  – short exit signals
        """
        required = {"open", "close", "volume"}
        if not required.issubset(df.columns) or len(df) < self.OFI_LOOKBACK + self.MA_PERIOD:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(
                entries=empty, exits=empty, short_entries=empty, short_exits=empty
            )

        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        # Signed flow per bar
        signed_flow = np.sign(close - open_) * volume

        # OFI ratio (rolling)
        ofi_numer = signed_flow.rolling(self.OFI_LOOKBACK).sum()
        ofi_denom = volume.rolling(self.OFI_LOOKBACK).sum()
        ofi_ratio = ofi_numer / ofi_denom

        # 5‑bar momentum (percentage change)
        momentum = close.pct_change(periods=self.MOMENTUM_BARS)

        # Simple moving average for trend filter
        ma = close.rolling(self.MA_PERIOD).mean()

        # Volume spike filter: current bar volume > avg * multiplier
        avg_vol = volume.rolling(self.OFI_LOOKBACK).mean()
        volume_spike = volume > avg_vol * self.VOLUME_SPIKE_MULTIPLIER

        # ----- Entry conditions -----
        long_entry = (
            (ofi_ratio > self.OFI_ENTRY_THRESHOLD)
            & (momentum > 0)
            & (close > ma)
            & volume_spike
        )
        short_entry = (
            (ofi_ratio < -self.OFI_ENTRY_THRESHOLD)
            & (momentum < 0)
            & (close < ma)
            & volume_spike
        )

        # Shift to avoid look‑ahead bias
        long_entry = long_entry.shift(1).fillna(False)
        short_entry = short_entry.shift(1).fillna(False)

        # ----- Exit conditions -----
        long_exit = (
            (ofi_ratio.abs() < self.OFI_EXIT_THRESHOLD)
            | (momentum <= 0)
            | (close <= ma)  # trend reversal
        )
        short_exit = (
            (ofi_ratio.abs() < self.OFI_EXIT_THRESHOLD)
            | (momentum >= 0)
            | (close >= ma)
        )

        long_exit = long_exit.shift(1).fillna(False)
        short_exit = short_exit.shift(1).fillna(False)

        # Ensure boolean dtype
        long_entry = long_entry.astype(bool)
        short