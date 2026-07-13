"""
Earnings Accruals Factor (Sloan Anomaly)
==========================================
Academic basis:
  - Sloan (1996) "Do Stock Prices Fully Reflect Information in Accruals and Cash
    Flows about Future Earnings?" The Accounting Review — the definitive paper.
    Stocks with high accruals (earnings not backed by operating cash flow) earn
    significantly lower future returns. The effect is robust and persistent.
  - Hirshleifer, Hou, Teoh, Zhang (2004) extended the finding to the full cross-section.
  - Dechow, Ge, Schrand (2010) meta-review: accrual anomaly earns ~5% annualized.

Mechanism:
  Accruals = Net Income - Operating Cash Flow
  Accruals Ratio = Accruals / Average Total Assets

  High accruals → earnings are "managed" (not cash-backed) → mean-reverts.
  Short high-accrual stocks, long low-accrual.

OHLCV Proxy (no direct financial statement access via Alpaca):
  "Momentum-volume divergence" — the accrual signature in price data:
  1. Price appreciation without volume (> 15% 60‑day return, declining volume):
     indicates rally driven by sentiment / insider accumulation, not real demand.
  2. This mirrors accrual-driven earnings beats that get reversed later.

  Signal computation:
    price_mom_60  = close / close.shift(60) - 1
    volume_ratio  = rolling_mean_volume(20) / rolling_mean_volume(60)
    accrual_proxy = price_mom_60 × (1 - volume_ratio)  — high when price rose
                                                          but volume fell

  SHORT when: price_mom_60 > 0.15 AND volume_ratio < 0.85
  (large price gain with declining relative volume = likely accruals support)
  Confidence proportional to divergence magnitude.

Universe: high‑accrual candidate names (growth + special‑situation stocks
          historically prone to earnings management).

Documented Sharpe: ~0.7‑1.0 long‑short; ~0.4‑0.6 short‑only leg
"""

from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"


class EarningsAccrualsStrategy(AbstractStrategy):
    """
    Earnings accruals (Sloan) factor via price‑volume divergence proxy.

    Identifies stocks where price appreciation is NOT backed by volume
    (potential accruals signal), then shorts them expecting mean reversion.
    """

    name = "earnings_accruals"
    display_name = "Earnings Accruals Factor (Sloan)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0  # daily — fundamental factor, slow signal

    # Universe: growth / story stocks historically prone to accruals
    UNIVERSE = [
        "META", "SNAP", "UBER", "LYFT", "HOOD", "RIVN", "LCID", "SPCE",
        "NKLA", "BYND", "LAZR", "AEVA", "MSTR", "COIN", "SMCI", "NVAX",
        "TDOC", "ROKU", "ZM", "DKNG",
    ]

    # Signal parameters
    PRICE_MOM_WINDOW = 60      # 60‑day price momentum window
    VOLUME_SHORT_WINDOW = 20   # recent avg volume
    VOLUME_LONG_WINDOW = 60    # baseline avg volume
    PRICE_MOM_THRESHOLD = 0.15  # >15 % price rise triggers check
    VOLUME_RATIO_MAX = 0.85    # volume must have declined >15 %
    RSI_PERIOD = 14            # confirmation filter
    RSI_MIN = 65               # over‑bought threshold for short entry
    SMA_SHORT = 20
    SMA_LONG = 60
    ATR_PERIOD = 14            # exit‑logic volatility measure
    HISTORY_DAYS = 252         # bars to fetch

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    async def _fetch_daily_bars(self, symbol: str) -> pd.DataFrame:
        """Fetch daily OHLCV for signal computation."""
        start = (date.today() - timedelta(days=self.HISTORY_DAYS + 30)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Day",
                        "start": start,
                        "limit": self.HISTORY_DAYS + 30,
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
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _compute_accrual_signal(
        df: pd.DataFrame,
        price_window: int,
        vol_short: int,
        vol_long: int,
    ) -> tuple[float, float, float]:
        """
        Compute accrual proxy signal from the latest bar of df.
        Returns (price_momentum, volume_ratio, accrual_score).
        accrual_score > 0 indicates potential accruals (short candidate).
        """
        if len(df) < vol_long + 5:
            return 0.0, 1.0, 0.0

        close = df["close"]
        volume = df["volume"]

        # 60‑day price momentum
        price_mom = (
            float(close.iloc[-1] / close.iloc[-price_window] - 1.0)
            if len(close) >= price_window
            else 0.0
        )

        # Volume ratio: recent vs baseline
        avg_vol_short = float(volume.iloc[-vol_short:].mean())
        avg_vol_long = float(volume.iloc[-vol_long:].mean())
        volume_ratio = avg_vol_short / max(avg_vol_long, 1.0)

        # Accrual score – scaled to roughly [0, 1]
        if price_mom > 0 and volume_ratio < 1.0:
            accrual_score = price_mom * (1.0 - volume_ratio)
        else:
            accrual_score = 0.0

        return price_mom, volume_ratio, accrual_score

    @staticmethod
    def _sma(series: pd.Series, window: int) -> float:
        """Simple moving average of the last `window` points."""
        if len(series) < window:
            return np.nan
        return float(series.iloc[-window:].mean())

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        """Relative Strength Index (RSI) – returns the latest value."""
        if len(series) < period + 1:
            return np.nan
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(com=period - 1, adjust=False).mean()
        roll_down = down.ewm(com=period - 1, adjust=False).mean()
        rs = roll_up / roll_down.replace(to_replace=0, method="bfill")
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        """Average True Range (ATR) – returns the latest value."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        if len(df) < period + 1:
            return np.nan
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(com=period - 1, adjust=False).mean()
        return float(atr.iloc[-1])

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Compute the accrual proxy and apply tightened entry filters.
        Returns a Signal object when a short entry is justified.
        """
        if symbol not in self.UNIVERSE:
            return None

        df = await self._fetch_daily_bars(symbol)
        if df.empty or len(df) < self.VOLUME_LONG_WINDOW + 5:
            return None

        price_mom, volume_ratio, accrual_score = self._compute_accrual_signal(
            df,
            self.PRICE_MOM_WINDOW,
            self.VOLUME_SHORT_WINDOW,
            self.VOLUME_LONG_WINDOW,
        )

        # --- Primary entry filters -------------------------------------------------
        if price_mom < self.PRICE_MOM_THRESHOLD:
            return None  # insufficient price appreciation
        if volume_ratio >= self.VOLUME_RATIO_MAX:
            return None  # volume not sufficiently declining

        # --- Confirmation filters --------------------------------------------------
        rsi = self._rsi(df["close"], period=self.RSI_PERIOD)
        if np.isnan(rsi) or rsi < self.RSI_MIN:
            return None  # not over‑bought enough for a short

        sma_short = self._sma(df["close"], self.SMA_SHORT)
        sma_long = self._sma(df["close"], self.SMA_LONG)
        if np.isnan(sma_short) or np.isnan(sma_long) or sma_short <= sma_long:
            return None  # price not exhibiting a clear short‑term uptrend (needed for reversal)

        # --- Confidence computation ------------------------------------------------
        raw_confidence = min(accrual_score * 2.5, 0.95)
        if raw_confidence < 0.35:
            return None  # below confidence threshold

        # --- Exit logic (dynamic stop / target based on volatility) ----------------
        current_price = float(df["close"].iloc[-1])
        atr = self._atr(df, period=self.ATR_PERIOD)
        if np.isnan(atr) or atr == 0:
            # fallback to fixed percentages if ATR unavailable
            stop_loss = round(current_price * 1.07, 4)   # 7 % above entry
            take_profit = round(current_price * 0.88, 4)  # 12 % below entry
        else:
            stop_loss = round(current_price + 2 * atr, 4)   # 2 ×ATR above entry
            take_profit = round(current_price - 3 * atr, 4)  # 3 ×ATR below entry

        return Signal(
            symbol=symbol,
            side="sell",  # SHORT – accruals predict underperformance
            confidence=round(raw_confidence, 4),
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            entry_price=round(current_price, 4),
            stop_price=stop_loss,
            target_price=take_profit,
        )