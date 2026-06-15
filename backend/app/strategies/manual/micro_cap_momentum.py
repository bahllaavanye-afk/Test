"""
Small-Cap Momentum Strategy.

Universe: IWM components (Russell 2000 small-cap index).
Signal: stocks with 3-month return in top decile AND volume > 2× 20-day average.

Combines price momentum with volume confirmation to identify
strong small-cap movers with conviction behind the move.
"""
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class MicroCapMomentumStrategy(AbstractStrategy):
    name = "micro_cap_momentum"
    display_name = "Small-Cap Momentum (IWM Universe)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0  # daily signal

    LOOKBACK_DAYS = 63     # ~3 months trading days
    VOL_WINDOW = 20        # 20-day volume average
    VOL_MULTIPLIER = 2.0   # volume must be >2× average
    TOP_DECILE = 0.90      # top 10% momentum rank
    EXIT_DECILE = 0.50     # exit when falls below median

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.lookback = p.get("lookback_days", self.LOOKBACK_DAYS)
        self.vol_window = p.get("vol_window", self.VOL_WINDOW)
        self.vol_multiplier = p.get("vol_multiplier", self.VOL_MULTIPLIER)
        self.top_decile = p.get("top_decile", self.TOP_DECILE)

    def _compute_momentum(self, close: pd.Series) -> float:
        """3-month return, skip last week (avoid reversal)."""
        if len(close) < self.lookback + 5:
            return 0.0
        past = close.iloc[-(self.lookback + 5)]
        now = close.iloc[-5]  # skip last 5 days
        return float((now - past) / past) if past > 0 else 0.0

    def _volume_spike(self, volume: pd.Series) -> bool:
        """True if latest volume > 2× 20-day average."""
        if len(volume) < self.vol_window + 1:
            return False
        avg_vol = float(volume.iloc[-self.vol_window - 1:-1].mean())
        latest_vol = float(volume.iloc[-1])
        return latest_vol > self.vol_multiplier * avg_vol if avg_vol > 0 else False

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < self.lookback + 10:
            return None

        close = data["close"]
        mom = self._compute_momentum(close)

        has_volume_spike = False
        if "volume" in data.columns:
            has_volume_spike = self._volume_spike(data["volume"])
        else:
            # Without volume data, use a looser criterion
            has_volume_spike = True

        # Require strong positive momentum AND volume confirmation
        if mom > 0.10 and has_volume_spike:   # top-decile proxy: >10% 3-month return
            confidence = min(0.85, 0.60 + mom * 0.5)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "momentum_3m": round(mom, 4),
                    "volume_spike": has_volume_spike,
                },
            )
        elif mom < -0.10 and has_volume_spike:
            confidence = min(0.80, 0.60 + abs(mom) * 0.4)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "momentum_3m": round(mom, 4),
                    "volume_spike": has_volume_spike,
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]

        # 3-month momentum (shifted 5 days + 1 for lookahead prevention)
        mom = (close.shift(5) / close.shift(self.lookback + 5) - 1).shift(1)

        # Volume condition
        if "volume" in df.columns:
            vol = df["volume"]
            vol_avg = vol.rolling(self.vol_window).mean().shift(1)
            vol_spike = vol.shift(1) > self.vol_multiplier * vol_avg
        else:
            vol_spike = pd.Series(True, index=close.index)

        entries = (mom > 0.10) & vol_spike
        exits = mom < 0.02
        short_entries = (mom < -0.10) & vol_spike
        short_exits = mom > -0.02

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
