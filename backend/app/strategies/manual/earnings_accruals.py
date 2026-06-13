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
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

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
    PRICE_MOM_WINDOW    = 60    # 60-day price momentum window
    VOLUME_SHORT_WINDOW = 20    # recent avg volume
    VOLUME_LONG_WINDOW  = 60    # baseline avg volume
    PRICE_MOM_THRESHOLD = 0.15  # > 15% price rise triggers check
    VOLUME_RATIO_MAX    = 0.85  # volume must have declined > 15%
    HISTORY_DAYS        = 252   # bars to fetch

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
    def _compute_accrual_signal(df: pd.DataFrame,
                                 price_window: int,
                                 vol_short: int,
                                 vol_long: int) -> tuple[float, float, float]:
        """
        Compute accrual proxy signal from the latest bar of df.
        Returns (price_momentum, volume_ratio, accrual_score).
        accrual_score > 0 indicates potential accruals (short candidate).
        """
        if len(df) < vol_long + 5:
            return 0.0, 1.0, 0.0

        close  = df["close"]
        volume = df["volume"]

        # 60-day price momentum
        if len(close) >= price_window:
            price_mom = float(close.iloc[-1] / close.iloc[-price_window] - 1.0)
        else:
            price_mom = 0.0

        # Volume ratio: recent vs baseline
        avg_vol_short = float(volume.iloc[-vol_short:].mean())
        avg_vol_long  = float(volume.iloc[-vol_long:].mean())
        volume_ratio  = avg_vol_short / max(avg_vol_long, 1.0)

        # Accrual score: large positive when price rose sharply but volume dropped
        # Scaled to [0, 1] approximately
        if price_mom > 0 and volume_ratio < 1.0:
            accrual_score = price_mom * (1.0 - volume_ratio)
        else:
            accrual_score = 0.0

        return price_mom, volume_ratio, accrual_score

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Fetch daily bars and compute price-volume divergence (accrual proxy).
        SHORT signal when price rose > 15% in 60 days but volume declined > 15%.
        """
        if symbol not in self.UNIVERSE:
            # Can be called with non-universe symbol; skip gracefully
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

        # Trigger conditions
        if price_mom < self.PRICE_MOM_THRESHOLD:
            return None  # not enough price appreciation to suspect accruals
        if volume_ratio >= self.VOLUME_RATIO_MAX:
            return None  # volume supported the rally — not an accrual signal

        # Confidence: proportional to divergence magnitude, capped at 0.90
        raw_confidence = min(accrual_score * 3.0, 0.90)
        if raw_confidence < 0.30:
            return None  # below minimum confidence threshold

        current_price = float(df["close"].iloc[-1])
        stop_loss    = round(current_price * 1.05, 4)  # 5% stop on short
        take_profit  = round(current_price * 0.88, 4)  # 12% target on short

        return Signal(
            symbol=symbol,
            side="sell",  # SHORT — accruals predict underperformance
            confidence=round(raw_confidence, 4),
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=take_profit,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "accrual_proxy": round(accrual_score, 4),
                "price_mom_60d": round(price_mom, 4),
                "volume_ratio_20_60": round(volume_ratio, 4),
                "interpretation": "high_accrual_short_candidate",
                "academic_ref": "Sloan (1996) Accruals Anomaly",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest on daily OHLCV.

        Short entry: price_mom_60 > 15% AND volume_ratio (20/60) < 0.85.
        Short exit:  accrual signal disappears (price_mom_60 drops below 5%
                     or volume_ratio recovers above 1.0).
        Apply shift(1) to prevent lookahead.
        """
        required = {"close", "volume"}
        if not required.issubset(df.columns) or len(df) < self.VOLUME_LONG_WINDOW + 10:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty,
                                   short_entries=empty, short_exits=empty)

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # 60-day price momentum
        price_mom = close / close.shift(self.PRICE_MOM_WINDOW) - 1.0

        # Rolling average volumes
        avg_vol_short = volume.rolling(self.VOLUME_SHORT_WINDOW, min_periods=10).mean()
        avg_vol_long  = volume.rolling(self.VOLUME_LONG_WINDOW,  min_periods=30).mean()
        volume_ratio  = avg_vol_short / avg_vol_long.clip(lower=1.0)

        # Accrual score
        accrual_score = price_mom.clip(lower=0) * (1.0 - volume_ratio).clip(lower=0)

        # Entry conditions
        high_accrual = (
            (price_mom > self.PRICE_MOM_THRESHOLD) &
            (volume_ratio < self.VOLUME_RATIO_MAX)
        )

        # Exit: momentum fades or volume recovers
        accrual_exit = (price_mom < 0.05) | (volume_ratio > 1.0)

        # This is a short strategy — entries are short entries
        short_entries = high_accrual.shift(1).fillna(False)
        short_exits   = accrual_exit.shift(1).fillna(False)

        # No long leg (pure short-side factor)
        empty = pd.Series(False, index=df.index)

        return BacktestSignals(
            entries=empty,
            exits=empty,
            short_entries=short_entries,
            short_exits=short_exits,
        )
