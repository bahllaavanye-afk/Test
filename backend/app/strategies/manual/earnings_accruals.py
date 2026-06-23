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
  1. Price appreciation without volume (> 15% 60-day return, declining volume):
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

Universe: high-accrual candidate names (growth + special-situation stocks
          historically prone to earnings management).

Documented Sharpe: ~0.7-1.0 long-short; ~0.4-0.6 short-only leg
"""

from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, Signal

_DATA_BASE = "https://data.alpaca.markets"


class EarningsAccrualsStrategy(AbstractStrategy):
    """
    Earnings accruals (Sloan) factor via price-volume divergence proxy.

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
    PRICE_MOM_WINDOW = 60    # 60‑day price momentum window
    VOLUME_SHORT_WINDOW = 20  # recent avg volume
    VOLUME_LONG_WINDOW = 60   # baseline avg volume
    PRICE_MOM_THRESHOLD = 0.20  # tighter: >20% price rise
    VOLUME_RATIO_MAX = 0.80     # tighter: >20% volume decline
    MIN_CONFIDENCE = 0.35       # higher minimum confidence
    HISTORY_DAYS = 252         # bars to fetch

    # Additional confirmation parameters
    SHORT_TERM_MOM_WINDOW = 10
    SMA_SHORT = 20
    SMA_LONG = 60
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70

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

        # Accrual score: large positive when price rose sharply but volume dropped
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
        """Relative Strength Index (RSI) calculated on close series."""
        if len(series) < period + 1:
            return np.nan
        delta = series.diff().dropna()
        up = delta.clip(lower=0).rolling(window=period).mean()
        down = -delta.clip(upper=0).rolling(window=period).mean()
        rs = up / down.replace(to_replace=0, method="ffill")
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        """Average True Range (ATR) over the given period."""
        if len(df) < period + 1:
            return np.nan
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return float(atr.iloc[-1])

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Compute the earnings‑accruals short signal.

        Entry criteria (tightened):
        - 60‑day price momentum > PRICE_MOM_THRESHOLD (20%).
        - Volume ratio (20‑day / 60‑day) < VOLUME_RATIO_MAX (80%).
        - Short‑term momentum (10‑day) still positive (to avoid immediate reversals).
        - Price above its 20‑day SMA and 20‑day SMA above 60‑day SMA (overbought regime).
        - RSI > RSI_OVERBOUGHT (70) for additional overbought confirmation.

        Exit logic (dynamic):
        - Stop‑loss set to the larger of a fixed 5 % move or 1.5 × ATR.
        - Target‑price set to 2 × ATR below the current price (short profit zone).
        """
        if symbol not in self.UNIVERSE:
            return None

        df = await self._fetch_daily_bars(symbol)
        if df.empty or len(df) < self.VOLUME_LONG_WINDOW + 5:
            return None

        # Core accrual proxy
        price_mom, volume_ratio, accrual_score = self._compute_accrual_signal(
            df,
            self.PRICE_MOM_WINDOW,
            self.VOLUME_SHORT_WINDOW,
            self.VOLUME_LONG_WINDOW,
        )

        # --- Tightened entry filters ---
        if price_mom < self.PRICE_MOM_THRESHOLD:
            return None
        if volume_ratio >= self.VOLUME_RATIO_MAX:
            return None

        # Short‑term momentum confirmation
        if len(df["close"]) < self.SHORT_TERM_MOM_WINDOW:
            return None
        short_term_mom = float(df["close"].iloc[-1] / df["close"].iloc[-self.SHORT_TERM_MOM_WINDOW] - 1.0)
        if short_term_mom <= 0:
            return None

        # SMA overbought confirmation
        sma_short = self._sma(df["close"], self.SMA_SHORT)
        sma_long = self._sma(df["close"], self.SMA_LONG)
        if np.isnan(sma_short) or np.isnan(sma_long):
            return None
        if not (sma_short > sma_long and df["close"].iloc[-1] > sma_short):
            return None

        # RSI confirmation
        rsi = self._rsi(df["close"], self.RSI_PERIOD)
        if np.isnan(rsi) or rsi < self.RSI_OVERBOUGHT:
            return None

        # Confidence scaling (more aggressive now)
        raw_confidence = min(accrual_score * 4.0, 0.95)
        if raw_confidence < self.MIN_CONFIDENCE:
            return None

        # Dynamic exit parameters based on ATR
        atr = self._atr(df, period=14)
        if np.isnan(atr) or atr <= 0:
            # Fallback to fixed percentages if ATR unavailable
            current_price = float(df["close"].iloc[-1])
            stop_loss = round(current_price * 1.05, 4)   # 5 % stop‑loss
            target_price = round(current_price * 0.88, 4)  # 12 % target
        else:
            current_price = float(df["close"].iloc[-1])
            # For a short: stop is price up, target is price down
            stop_loss = round(current_price + max(0.05 * current_price, 1.5 * atr), 4)
            target_price = round(current_price - 2.0 * atr, 4)

        return Signal(
            symbol=symbol,
            side="sell",
            confidence=round(raw_confidence, 4),
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=target_price,
            stop_loss=stop_loss,
        )