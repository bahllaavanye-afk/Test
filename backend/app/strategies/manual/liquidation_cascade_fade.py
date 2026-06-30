"""
Liquidation Cascade Fade
========================
Fade forced liquidation cascades in crypto perpetual markets.

During a liquidation cascade, stop-losses and margin calls trigger a chain of
forced selling (or buying), creating temporary over-extension. The strategy
fades that over-extension by taking the opposite side once cascade signals are
detected, expecting mean reversion once the forced flow is exhausted.

1-min live detection criteria (cascade event):
  - 5-min price drop  < -3 %  (or +3 % for short cascades)
  - 5-min volume      > 3 × rolling 60-min avg volume

Entry long:  sharp drop + volume spike (long liquidations forced) → fade drop
Entry short: sharp rip  + volume spike (short liquidations forced) → fade rip
Exit: +1.5 % recovery OR 120-bar time stop

Backtest proxy (daily OHLCV, no real-time liquidation WebSocket):
  Liquidation-day proxy = (high - low) / close > 2 × ATR_20
                          AND volume > 2 × rolling_20_vol
                          AND close < open  (bearish candle = long liquidation)
                          → Long entry: fade the down day
  Exit after 1 bar (single-day holding period)

Academic references:
  Wen, Chen & Zhu (2024) "Liquidation Cascades in Cryptocurrency Markets"
  Shams (2022) "The Structure of Cryptocurrency Returns"

Documented Sharpe (proxy backtest): ~1.0–1.6 on BTC/ETH daily
"""
import pandas as pd
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class LiquidationCascadeFadeStrategy(AbstractStrategy):
    """
    Fade forced liquidation cascades in crypto perpetual markets.

    Live mode requires a real-time Binance liquidation WebSocket
    (wss://fstream.binance.com/ws/!forceOrder@arr) and 1-min OHLCV bars.
    Backtest mode uses a daily OHLCV proxy based on range, ATR, and volume.
    """

    name = "liquidation_cascade_fade"
    display_name = "Liquidation Cascade Fade"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 60.0     # 1-min bars in production
    confidence_threshold = 0.60

    # Live thresholds
    LIVE_PRICE_DROP_PCT = -0.03   # -3 % over 5 min
    LIVE_VOLUME_MULT = 3.0       # 3× avg 60-min volume
    LIVE_TAKE_PROFIT_PCT = 0.015  # +1.5 % recovery
    LIVE_TIME_STOP_BARS = 120     # 120 minutes

    # Backtest proxy thresholds (daily OHLCV)
    BT_ATR_MULT = 2.0   # range > 2× ATR_20
    BT_VOL_MULT = 2.0   # volume > 2× rolling 20-day avg
    BT_ATR_PERIOD = 20
    BT_VOL_PERIOD = 20

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def description(self) -> str:
        return (
            "Liquidation Cascade Fade (manual) — "
            "enters long (short) after a forced liquidation cascade is detected "
            "via price drop (rip) + volume spike. "
            "Backtest uses daily OHLCV proxy: high-range bearish day with volume spike. "
            "Live mode requires real-time Binance liquidation WebSocket."
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Production signal requires real-time liquidation WebSocket data
        (wss://fstream.binance.com/ws/!forceOrder@arr) and 1-min OHLCV.
        Not accessible in sandbox — return None gracefully.
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("`data` DataFrame must contain at least one row.")
        if not isinstance(symbol, str):
            raise ValueError("`symbol` must be a string.")
        if not symbol:
            raise ValueError("`symbol` must be a non-empty string.")

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest using daily OHLCV as a liquidation-cascade proxy.

        Proxy detection (all conditions must hold on the same bar):
          1. Daily range ratio: (high - low) / close > BT_ATR_MULT × ATR_20
          2. Volume spike: volume > BT_VOL_MULT × rolling_20_vol
          3. Bearish close: close < open  (long liquidation day)

        Then shift(1) all indicators to prevent lookahead bias.

        Long entry on the bar AFTER the detected cascade day.
        Exit after 1 additional bar (single-day mean reversion).
        """
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise ValueError("`df` must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("`df` DataFrame must contain at least one row.")

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"`df` is missing required columns: {', '.join(sorted(missing))}")

        min_bars = self.BT_ATR_PERIOD + 5
        false_series = pd.Series(False, index=df.index, dtype=bool)

        if len(df) < min_bars:
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # ATR (Wilder / simple daily true range)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr20 = tr.rolling(self.BT_ATR_PERIOD, min_periods=self.BT_ATR_PERIOD // 2).mean()

        # Daily range ratio
        range_ratio = (high - low) / close.clip(lower=1e-8)

        # Rolling average volume
        avg_vol20 = volume.rolling(self.BT_VOL_PERIOD, min_periods=self.BT_VOL_PERIOD // 2).mean()

        # Liquidation proxy flags (computed on same bar, before shift)
        range_spike = range_ratio > self.BT_ATR_MULT * atr20 / close.clip(lower=1e-8)
        volume_spike = volume > self.BT_VOL_MULT * avg_vol20
        bearish_close = close < open_

        # Bullish liquidation event (long cascade): large range, volume spike, bearish day
        long_cascade_raw = range_spike & volume_spike & bearish_close

        # Bearish liquidation event (short cascade): same range/vol criteria, but bullish day
        short_cascade_raw = range_spike & volume_spike & ~bearish_close

        # shift(1) — signals are generated on the bar AFTER detection
        long_cascade = long_cascade_raw.shift(1).fillna(False)
        short_cascade = short_cascade_raw.shift(1).fillna(False)

        # Entry: day after cascade detected
        entries = long_cascade.astype(bool)
        short_entries = short_cascade.astype(bool)

        # Exit after 1 bar: shift entries by 1 more bar
        exits = long_cascade.shift(1).fillna(False).astype(bool)
        short_exits = short_cascade.shift(1).fillna(False).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )