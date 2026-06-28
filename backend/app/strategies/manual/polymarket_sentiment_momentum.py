import pandas as pd

from app.strategies.base import AbstractStrategy, BacktrackSignals, Signal

# Momentum thresholds
LIVE_MOMENTUM_PP = 0.05    # 5 percentage‑point rise/fall in live P(YES)
BACKTEST_MOMENTUM_PCT = 0.03  # 3 % rise/fall in close (OHLCV proxy)
VOLUME_MULTIPLIER = 1.5    # volume must exceed 1.5× rolling average
VOLUME_WINDOW = 20         # rolling window for average volume
MOMENTUM_BARS = 3          # look‑back period for momentum
PROB_LOWER = 0.20          # lower bound for P(YES)
PROB_UPPER = 0.80          # upper bound for P(YES)
EXIT_THRESHOLD = 0.90      # exit long when P(YES) > 90 %
SHORT_EXIT_THRESHOLD = 0.10  # exit short when P(YES) < 10 %
SMA_SHORT_WINDOW = 10      # short‑term SMA for confirmation


class PolymarketSentimentMomentumStrategy(AbstractStrategy):
    """
    Prediction‑market momentum strategy.

    Buys YES contracts when probability rises sharply together with a volume
    spike and a short‑term price confirmation.  Positions are exited before
    resolution (P > 90 %) or when a stop‑loss / time‑based condition is met.
    """

    name = "polymarket_sentiment_momentum"
    display_name = "Polymarket Sentiment Momentum"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 300.0   # poll every 5 minutes
    confidence_threshold = 0.60

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.momentum_pp = (params or {}).get("momentum_pp", LIVE_MOMENTUM_PP)
        self.volume_multiplier = (params or {}).get("volume_multiplier", VOLUME_MULTIPLIER)
        self.exit_threshold = (params or {}).get("exit_threshold", EXIT_THRESHOLD)

    def description(self) -> str:
        return (
            f"{self.display_name} — buys YES contracts with rising P(YES) (>{self.momentum_pp*100:.0f} pp "
            f"over {MOMENTUM_BARS} obs) + volume spike + price‑SMA confirmation; exits near resolution."
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyze live Polymarket CLOB data.

        Expected columns:
            - yes_price : float – probability of YES outcome
            - volume    : float – traded volume

        Returns a Signal for entry, exit, or None if conditions are not met.
        """
        required = {"yes_price", "volume"}
        if not required.issubset(data.columns):
            return None

        if len(data) < max(MOMENTUM_BARS, SMA_SHORT_WINDOW) + 1:
            return None

        yes_price = data["yes_price"].astype(float)
        volume = data["volume"].astype(float)

        # Current values
        current_p = float(yes_price.iloc[-1])
        past_p = float(yes_price.iloc[-1 - MOMENTUM_BARS])
        delta_pp = current_p - past_p

        # Volume spike
        rolling_avg_vol = float(volume.rolling(VOLUME_WINDOW, min_periods=3).mean().iloc[-1])
        current_vol = float(volume.iloc[-1])
        vol_spike = (
            current_vol > self.volume_multiplier * rolling_avg_vol
            if rolling_avg_vol > 0
            else False
        )

        # Short‑term SMA confirmation
        sma_short = float(yes_price.rolling(SMA_SHORT_WINDOW, min_periods=5).mean().iloc[-1])
        price_above_sma = current_p > sma_short
        price_below_sma = current_p < sma_short

        # ---- Exit logic -------------------------------------------------
        if current_p >= self.exit_threshold:
            # Close any long position before resolution
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=0.99,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=round(self.exit_threshold, 4),
                metadata={
                    "yes_price": round(current_p, 4),
                    "exit_reason": "near_resolution",
                    "direction": "exit_long",
                },
            )

        # ---- Long entry -------------------------------------------------
        if (
            delta_pp > self.momentum_pp
            and PROB_LOWER < current_p < PROB_UPPER
            and vol_spike
            and price_above_sma
        ):
            confidence = min(0.85, 0.60 + delta_pp * 2.0)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=round(min(current_p + 0.10, self.exit_threshold - 0.01), 4),
                metadata={
                    "yes_price": round(current_p, 4),
                    "delta_pp_3obs": round(delta_pp, 4),
                    "volume_ratio": round(current_vol / rolling_avg_vol, 3)
                    if rolling_avg_vol > 0
                    else None,
                    "direction": "long_yes",
                },
            )

        # ---- Short entry ------------------------------------------------
        if (
            delta_pp < -self.momentum_pp
            and PROB_LOWER < current_p < PROB_UPPER
            and vol_spike
            and price_below_sma
        ):
            confidence = min(0.85, 0.60 + abs(delta_pp) * 2.0)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=round(max(current_p - 0.10, 0.10), 4),
                metadata={
                    "yes_price": round(current_p, 4),
                    "delta_pp_3obs": round(delta_pp, 4),
                    "volume_ratio": round(current_vol / rolling_avg_vol, 3)
                    if rolling_avg_vol > 0
                    else None,
                    "direction": "short_yes",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktrackSignals:
        """
        Vectorised back‑test using OHLCV proxy.

        - Long entry:   close ↑ > 3 % over MOMENTUM_BARS, volume spike, and close > SMA10.
        - Long exit:    close ≥ EXIT_THRESHOLD.
        - Short entry:  close ↓ > 3 % over MOMENTUM_BARS, volume spike, and close < SMA10.
        - Short exit:   close ≤ SHORT_EXIT_THRESHOLD.
        All signals are shifted by one bar to avoid look‑ahead bias.
        """
        required = {"close", "volume"}
        min_rows = max(MOMENTUM_BARS, SMA_SHORT_WINDOW) + VOLUME_WINDOW + 2
        if not required.issubset(df.columns) or len(df) < min_rows:
            empty = pd.Series(False, index=df.index, dtype=bool)
            return BacktrackSignals(
                entries=empty,
                exits=empty,
                short_entries=empty,
                short_exits=empty,
            )

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # Momentum (percentage change over MOMENTUM_BARS)
        momentum = close.pct_change(periods=MOMENTUM_BARS)

        # Volume spike
        rolling_vol = volume.rolling(VOLUME_WINDOW, min_periods=5).mean()
        vol_spike = volume > VOLUME_MULTIPLIER * rolling_vol

        # Short‑term SMA confirmation
        sma_short = close.rolling(SMA_SHORT_WINDOW, min_periods=5).mean()
        price_above_sma = close > sma_short
        price_below_sma = close < sma_short

        # Long entry conditions
        long_entry = (
            (momentum > BACKTEST_MOMENTUM_PCT)
            & vol_spike
            & price_above_sma
            & (close > PROB_LOWER)
            & (close < PROB_UPPER)
        )

        # Long exit conditions
        long_exit = close >= EXIT_THRESHOLD

        # Short entry conditions
        short_entry = (
            (momentum < -BACKTEST_MOMENTUM_PCT)
            & vol_spike
            & price_below_sma
            & (close > PROB_LOWER)
            & (close < PROB_UPPER)
        )

        # Short exit conditions
        short_exit = close <= SHORT_EXIT_THRESHOLD

        # Shift to avoid look‑ahead bias
        entries = long_entry.shift(1).fillna(False)
        exits = long_exit.shift(1).fillna(False)
        short_entries = short_entry.shift(1).fillna(False)
        short_exits = short_exit.shift(1).fillna(False)

        return BacktrackSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )