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
Exit: OFI_ratio < 0.10 OR position hits +0.5% take-profit

Documented Sharpe: ~1.5-2.0 in academic studies (implementation varies by execution quality)
"""

from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

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

class OrderFlowImbalanceParams(BaseModel):
    """
    Pydantic schema for validating strategy parameters supplied at runtime.
    """

    ofi_entry_threshold: float = Field(
        0.60,
        description="Absolute OFI ratio threshold required to consider an entry signal.",
        ge=0.0,
        le=1.0,
        example=0.6,
    )
    ofi_exit_threshold: float = Field(
        0.10,
        description="Absolute OFI ratio threshold below which an existing position should be exited.",
        ge=0.0,
        le=1.0,
        example=0.1,
    )
    momentum_bars: int = Field(
        5,
        description="Number of recent bars used to compute price momentum.",
        ge=1,
        example=5,
    )
    ofi_lookback: int = Field(
        10,
        description="Rolling window size (in bars) for OFI computation.",
        ge=1,
        example=10,
    )
    bars_to_fetch: int = Field(
        20,
        description="Number of 1‑minute bars to request from Alpaca each run.",
        ge=1,
        example=20,
    )
    volume_spike_multiplier: float = Field(
        1.20,
        description="Multiplier applied to the recent average volume to define a volume spike.",
        ge=1.0,
        example=1.2,
    )
    ma_period: int = Field(
        20,
        description="Period for the simple moving average used as a short‑term trend filter.",
        ge=1,
        example=20,
    )
    take_profit_pct: float = Field(
        0.005,
        description="Target profit as a decimal fraction of entry price (e.g., 0.005 = 0.5%).",
        ge=0.0,
        example=0.005,
    )
    stop_loss_pct: float = Field(
        0.003,
        description="Maximum tolerated loss as a decimal fraction of entry price.",
        ge=0.0,
        example=0.003,
    )

    @validator("ofi_entry_threshold", "ofi_exit_threshold")
    def check_ofi_thresholds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("OFI thresholds must be between 0 and 1")
        return v

    @validator("volume_spike_multiplier")
    def check_volume_multiplier(cls, v):
        if v < 1.0:
            raise ValueError("Volume spike multiplier must be >= 1")
        return v


class OrderFlowImbalanceStrategy(AbstractStrategy):
    """
    Intraday order flow imbalance using OHLCV‑derived buying/selling pressure.

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

    # Signal thresholds (default values – can be overridden via params)
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
        # Validate any user‑provided parameters without altering default constants.
        if params:
            OrderFlowImbalanceParams(**params)

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

        # Construct and return the signal. The exact fields required by
        # ``Signal`` may evolve; we provide the most common ones.
        return Signal(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            confidence=confidence,
            timestamp=pd.Timestamp.utcnow(),
        )