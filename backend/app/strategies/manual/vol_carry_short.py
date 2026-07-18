"""
Volatility Carry / Short VIX Term Structure Strategy.

When the VIX term structure is in contango (front month < back month)
AND VIX spot < 20, systematically:
  - Short VIX front-month (via VXX, UVXY, or short VIX futures)
  - Long VIX back-month (via VXZ or further dated)

The contango roll decay generates ~3-5% monthly carry in normal markets.
This strategy has a "short gamma" risk profile — avoid during volatility spikes.

Uses VIX front/back spread from CBOE (publicly available) or proxied
via VIX spot vs 3-month implied vol.
"""
import numpy as np
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class VolCarryShortStrategy(AbstractStrategy):
    name = "vol_carry_short"
    display_name = "Volatility Carry Short (VIX Contango)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0  # daily

    VIX_SPIKE_EXIT = 25.0    # exit when VIX > 25 (protect from vol spikes)
    VIX_ENTRY_MAX = 20.0     # only enter when VIX < 20
    CONTANGO_MIN = 0.05      # min contango (5% difference between back/front)
    BACKWARDATION_EXIT = 0.0 # exit if term structure flips to backwardation

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.vix_entry_max = p.get("vix_entry_max", self.VIX_ENTRY_MAX)
        self.vix_spike_exit = p.get("vix_spike_exit", self.VIX_SPIKE_EXIT)
        self.contango_min = p.get("contango_min", self.CONTANGO_MIN)

    async def analyze(self, data: pd.DataFrame | None, symbol: str) -> Signal | None:
        # Defensive checks for None or malformed DataFrame
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None
        if "close" not in data.columns or len(data) < 10:
            return None

        close = data["close"]
        # Ensure we have a valid last price
        if close.isna().all():
            return None
        current_price = float(close.iloc[-1])

        # VIX data check / proxy
        if "vix_close" in data.columns and not data["vix_close"].isna().all():
            vix_series = data["vix_close"]
            vix_current = float(vix_series.iloc[-1])
        else:
            # Proxy via 20‑day realized volatility
            ret = close.pct_change().dropna()
            if len(ret) < 20:
                return None
            vix_current = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)

        # Term‑structure calculation
        if "vix3m" in data.columns and "vix_close" in data.columns:
            front = data["vix_close"]
            back = data["vix3m"]
            # Guard against division by zero or NaN
            front_last = float(front.iloc[-1])
            back_last = float(back.iloc[-1])
            if front_last == 0 or np.isnan(front_last) or np.isnan(back_last):
                contango = 0.0
            else:
                contango = (back_last - front_last) / front_last
        else:
            # Proxy: ratio of 20‑day vs 5‑day realized vol
            ret = close.pct_change().dropna()
            if len(ret) < 20:
                return None
            rv5 = float(ret.rolling(5).std().iloc[-1] * np.sqrt(252) * 100)
            rv20 = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
            contango = (rv20 - rv5) / rv5 if rv5 > 0 else 0.0

        # Entry conditions: contango AND low VIX
        if contango > self.contango_min and vix_current < self.vix_entry_max:
            confidence = min(0.80, 0.60 + contango * 0.5)
            return Signal(
                symbol=symbol,
                side="sell",    # short volatility
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "vix_level": round(vix_current, 2),
                    "contango_pct": round(contango * 100, 2),
                    "trade_type": "short_front_long_back",
                },
            )
        elif vix_current > self.vix_spike_exit:
            # Exit signal: VIX spike, buy back to cover short
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=0.90,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"vix_level": round(vix_current, 2), "signal": "exit_vol_spike"},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame | None) -> BacktestSignals:
        # Defensive handling for None or empty inputs
        if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
            # Return empty signals to avoid breaking backtest pipelines
            empty = pd.Series([], dtype=bool)
            return BacktestSignals(
                entries=empty,
                exits=empty,
                short_entries=empty,
                short_exits=empty,
            )

        close = df["close"]
        ret = close.pct_change()

        # VIX series (real or proxy)
        if "vix_close" in df.columns and not df["vix_close"].isna().all():
            vix = df["vix_close"]
        else:
            vix = ret.rolling(20).std() * np.sqrt(252) * 100

        # Contango series
        if "vix3m" in df.columns and "vix_close" in df.columns:
            # Avoid division by zero
            front = df["vix_close"].replace(0, np.nan)
            contango = (df["vix3m"] - front) / front
        else:
            rv5 = ret.rolling(5).std() * np.sqrt(252) * 100
            rv20 = ret.rolling(20).std() * np.sqrt(252) * 100
            contango = (rv20 - rv5) / rv5.replace(0, np.nan)

        # Shift to avoid look‑ahead bias
        vix_s = vix.shift(1)
        contango_s = contango.shift(1)

        short_entries = (contango_s > self.contango_min) & (vix_s < self.vix_entry_max)
        short_exits = vix_s > self.vix_spike_exit

        # Inverse: buy when vol spikes to unwind
        entries = vix_s > self.vix_spike_exit
        exits = vix_s < self.vix_entry_max

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )