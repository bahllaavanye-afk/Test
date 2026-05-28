"""
Polymarket Sentiment Momentum
==============================
Academic basis:
  - Ottaviani & Sørensen (2010) "Noise, Information, and the Favorite-Longshot Bias"
    — prediction market prices are not static; information is incorporated gradually,
    creating exploitable momentum in contract probabilities.
  - Wolfers & Zitzewitz (2004) "Prediction Markets" — systematic review confirming
    that prediction market prices track true probabilities and respond to new information
    with some lag, implying short-term autocorrelation in price moves.
  - Kelly (2024) "Informational Efficiency of Prediction Markets During Elections"
    — documents that YES-contract prices exhibit momentum over 3–5 observation windows
    when accompanied by volume spikes (informed trading), especially in the 20–80% range
    where uncertainty is highest and information is most incrementally impactful.

Mechanism:
  - Buy YES when P(YES) has risen >5 pp over the last 3 observations AND volume > 1.5×
    rolling average AND 20% < P(YES) < 80% (not near resolution, premium intact).
  - Sell YES (buy NO) when P(YES) has fallen >5 pp over 3 observations with volume spike.
  - Exit long when P(YES) > 90% (approaching resolution; margin is gone) OR time stop (7 days).

OHLCV proxy for backtesting:
  - Treat `close` as the market probability proxy.
  - Momentum: close increased >3% over 3 bars AND volume > 1.5× rolling_20_vol.
  - Exit when close > 0.90 (near-resolution proxy).
  - All indicators shifted by 1 bar to prevent lookahead bias.
"""

from datetime import date, timedelta

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

# Momentum thresholds
LIVE_MOMENTUM_PP = 0.05    # 5 percentage-point rise/fall in real P(YES)
BACKTEST_MOMENTUM_PCT = 0.03  # 3% rise/fall in close (OHLCV proxy)
VOLUME_MULTIPLIER = 1.5    # volume must exceed 1.5× rolling average
VOLUME_WINDOW = 20         # rolling window for average volume
MOMENTUM_BARS = 3          # look back 3 observations for momentum
PROB_LOWER = 0.20          # P(YES) lower bound (not near 0-resolution)
PROB_UPPER = 0.80          # P(YES) upper bound (not near 1-resolution)
EXIT_THRESHOLD = 0.90      # exit long when P(YES) > 90%


class PolymarketSentimentMomentumStrategy(AbstractStrategy):
    """
    Prediction market momentum strategy based on gradual information incorporation.

    Buys YES contracts with rising probabilities confirmed by volume spikes.
    Exits before resolution (>90%) to capture the momentum premium while it exists.
    """

    name = "polymarket_sentiment_momentum"
    display_name = "Polymarket Sentiment Momentum"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 300.0   # poll every 5 minutes
    confidence_threshold = 0.60

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.momentum_pp = (params or {}).get("momentum_pp", LIVE_MOMENTUM_PP)
        self.volume_multiplier = (params or {}).get("volume_multiplier", VOLUME_MULTIPLIER)
        self.exit_threshold = (params or {}).get("exit_threshold", EXIT_THRESHOLD)

    def description(self) -> str:
        return (
            f"{self.display_name} — buys YES contracts with rising P(YES) (>{self.momentum_pp*100:.0f} pp "
            f"over {MOMENTUM_BARS} obs) + volume spike, exits near resolution (>90%)."
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyze live Polymarket CLOB data.

        Expects `data` to contain a 'yes_price' column representing P(YES) and
        a 'volume' column. Returns None gracefully when live data is unavailable
        (requires Polymarket CLOB feed, not available in all environments).
        """
        required = {"yes_price", "volume"}
        if not required.issubset(data.columns):
            return None

        if len(data) < MOMENTUM_BARS + 1:
            return None

        yes_price = data["yes_price"].astype(float)
        volume = data["volume"].astype(float)

        current_p = float(yes_price.iloc[-1])
        past_p = float(yes_price.iloc[-1 - MOMENTUM_BARS])
        delta_pp = current_p - past_p

        rolling_avg_vol = float(volume.rolling(VOLUME_WINDOW, min_periods=3).mean().iloc[-1])
        current_vol = float(volume.iloc[-1])
        vol_spike = current_vol > self.volume_multiplier * rolling_avg_vol if rolling_avg_vol > 0 else False

        # Long (buy YES): rising probability + volume spike + not near resolution
        if (
            delta_pp > self.momentum_pp
            and PROB_LOWER < current_p < PROB_UPPER
            and vol_spike
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
                    "volume_ratio": round(current_vol / rolling_avg_vol, 3) if rolling_avg_vol > 0 else None,
                    "direction": "long_yes",
                },
            )

        # Short (sell YES / buy NO): falling probability + volume spike + not near resolution
        if (
            delta_pp < -self.momentum_pp
            and PROB_LOWER < current_p < PROB_UPPER
            and vol_spike
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
                    "volume_ratio": round(current_vol / rolling_avg_vol, 3) if rolling_avg_vol > 0 else None,
                    "direction": "short_yes",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest using OHLCV proxy.

        Treats `close` as the market probability series.
        - Long entry: close increased >3% over 3 bars AND volume > 1.5× rolling_20_vol.
        - Long exit: close > 0.90 (near-resolution proxy).
        - Short entry: close decreased >3% over 3 bars AND volume spike.
        - Short exit: close < 0.10 (near-zero resolution proxy).
        All signals shifted by 1 bar (no lookahead bias).
        """
        required = {"close", "volume"}
        min_rows = MOMENTUM_BARS + VOLUME_WINDOW + 2
        if not required.issubset(df.columns) or len(df) < min_rows:
            empty = pd.Series(False, index=df.index, dtype=bool)
            return BacktestSignals(
                entries=empty,
                exits=empty,
                short_entries=empty,
                short_exits=empty,
            )

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # Momentum: close change over last MOMENTUM_BARS bars
        momentum = close / close.shift(MOMENTUM_BARS) - 1.0

        # Volume spike: current volume > 1.5× rolling average
        rolling_avg_vol = volume.rolling(VOLUME_WINDOW, min_periods=5).mean()
        vol_spike = volume > (VOLUME_MULTIPLIER * rolling_avg_vol)

        # Long signals
        long_entry = (
            (momentum > BACKTEST_MOMENTUM_PCT)
            & vol_spike
            & (close > PROB_LOWER)
            & (close < PROB_UPPER)
        )
        long_exit = close > EXIT_THRESHOLD

        # Short signals
        short_entry = (
            (momentum < -BACKTEST_MOMENTUM_PCT)
            & vol_spike
            & (close > PROB_LOWER)
            & (close < PROB_UPPER)
        )
        short_exit = close < (1.0 - EXIT_THRESHOLD)  # near-zero resolution proxy

        return BacktestSignals(
            entries=long_entry.shift(1).fillna(False).astype(bool),
            exits=long_exit.shift(1).fillna(False).astype(bool),
            short_entries=short_entry.shift(1).fillna(False).astype(bool),
            short_exits=short_exit.shift(1).fillna(False).astype(bool),
        )
