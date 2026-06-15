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


import httpx
import numpy as np
import pandas as pd

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

    Runs every minute. Fetches the last 20 1-minute bars, computes a
    volume-weighted signed flow ratio, and signals when imbalance is extreme
    and confirmed by 5-bar price momentum.
    """

    name = "order_flow_imbalance"
    display_name = "Order Flow Imbalance"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 60.0  # runs every minute

    # Signal thresholds
    OFI_ENTRY_THRESHOLD  = 0.60   # |OFI_ratio| > 0.60 to enter
    OFI_EXIT_THRESHOLD   = 0.10   # |OFI_ratio| < 0.10 to exit
    MOMENTUM_BARS        = 5      # bars for 5-min momentum confirmation
    OFI_LOOKBACK         = 10     # rolling window for OFI computation
    BARS_TO_FETCH        = 20     # 1-min bars to fetch from Alpaca

    # Risk parameters
    TAKE_PROFIT_PCT = 0.005       # 0.5% take-profit
    STOP_LOSS_PCT   = 0.003       # 0.3% stop-loss

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    async def _fetch_minute_bars(self, symbol: str) -> pd.DataFrame:
        """Fetch last N 1-minute bars for symbol."""
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
                return pd.DataFrame()
            bars = resp.json().get("bars", [])
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            df = df.rename(columns={"t": "time", "o": "open", "h": "high",
                                    "l": "low",  "c": "close", "v": "volume"})
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _compute_ofi(df: pd.DataFrame, lookback: int = 10) -> tuple[float, float]:
        """
        Compute OFI_ratio over the last `lookback` bars.
        Returns (ofi_ratio, momentum_5bar).
        """
        if len(df) < lookback + 5:
            return 0.0, 0.0

        window = df.iloc[-lookback:]
        signed_flow = np.sign(window["close"] - window["open"]) * window["volume"]
        total_volume = window["volume"].sum()
        ofi_ratio = float(signed_flow.sum() / total_volume) if total_volume > 0 else 0.0

        # 5-bar price momentum
        if len(df) >= 5:
            momentum = float(df["close"].iloc[-1] / df["close"].iloc[-5] - 1.0)
        else:
            momentum = 0.0

        return ofi_ratio, momentum

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Fetch latest 1-minute bars, compute OFI ratio and momentum.
        Signal buy if OFI > 0.60 and momentum > 0.
        Signal sell if OFI < -0.60 and momentum < 0.
        """
        if symbol not in DEFAULT_SYMBOLS:
            logger.debug("order_flow_imbalance: symbol outside default universe", symbol=symbol)

        df = await self._fetch_minute_bars(symbol)
        if df.empty or len(df) < self.OFI_LOOKBACK + 5:
            return None

        ofi_ratio, momentum = self._compute_ofi(df, self.OFI_LOOKBACK)
        current_price = float(df["close"].iloc[-1])

        abs_ofi = abs(ofi_ratio)

        # No signal in neutral zone
        if abs_ofi < self.OFI_ENTRY_THRESHOLD:
            return None

        # Momentum confirmation required
        if ofi_ratio > 0 and momentum <= 0:
            return None
        if ofi_ratio < 0 and momentum >= 0:
            return None

        side = "buy" if ofi_ratio > 0 else "sell"
        confidence = min(abs_ofi, 1.0)

        # Price targets
        if side == "buy":
            take_profit = round(current_price * (1 + self.TAKE_PROFIT_PCT), 4)
            stop_loss   = round(current_price * (1 - self.STOP_LOSS_PCT),   4)
        else:
            take_profit = round(current_price * (1 - self.TAKE_PROFIT_PCT), 4)
            stop_loss   = round(current_price * (1 + self.STOP_LOSS_PCT),   4)

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
                "bars_used": len(df),
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest on 1-minute OHLCV data.

        For each bar compute OFI_ratio over last 10 bars and 5-bar momentum.
        Enter long when OFI_ratio > threshold and momentum > 0 (shift 1 to avoid lookahead).
        Enter short when OFI_ratio < -threshold and momentum < 0.
        Exit long when OFI_ratio drops below exit threshold.
        """
        required = {"open", "close", "volume"}
        if not required.issubset(df.columns) or len(df) < self.OFI_LOOKBACK + 10:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty,
                                   short_entries=empty, short_exits=empty)

        close  = df["close"].astype(float)
        open_  = df["open"].astype(float)
        volume = df["volume"].astype(float)

        # Signed flow per bar
        signed_flow = np.sign(close - open_) * volume

        # Rolling OFI ratio and total volume
        rolling_ofi   = signed_flow.rolling(self.OFI_LOOKBACK, min_periods=5).sum()
        rolling_vol   = volume.rolling(self.OFI_LOOKBACK, min_periods=5).sum()
        ofi_ratio     = (rolling_ofi / rolling_vol.clip(lower=1.0)).fillna(0.0)

        # 5-bar momentum
        momentum = (close / close.shift(self.MOMENTUM_BARS) - 1.0).fillna(0.0)

        # Signals (shift 1 for lookahead prevention)
        long_signal  = (ofi_ratio > self.OFI_ENTRY_THRESHOLD)  & (momentum > 0)
        short_signal = (ofi_ratio < -self.OFI_ENTRY_THRESHOLD) & (momentum < 0)
        long_exit    = (ofi_ratio < self.OFI_EXIT_THRESHOLD)
        short_exit   = (ofi_ratio > -self.OFI_EXIT_THRESHOLD)

        return BacktestSignals(
            entries      = long_signal.shift(1).fillna(False),
            exits        = long_exit.shift(1).fillna(False),
            short_entries = short_signal.shift(1).fillna(False),
            short_exits  = short_exit.shift(1).fillna(False),
        )
