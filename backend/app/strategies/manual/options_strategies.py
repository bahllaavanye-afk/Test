"""Options-specific strategy signal generators.

These are rule-based signal generators (no ML). They produce signals for
options strategies: covered call, cash-secured put, iron condor, long call
momentum, earnings IV crush, and the wheel cycle.

Full execution logic (selecting specific contract, delta targeting, expiry
selection) will be layered on top in a future implementation phase. For now
these stubs define the signal conditions and backtest interface.

Research basis:
- Options Alpha backtests (180M+ strategies): Iron condors at 45 DTE,
  15-20 delta short strikes, 50% profit targets, 21 DTE exits → 78-82% win rate
- ORATS documented Sharpe 1.2-1.8 for managed iron condors
- Wheel income: 1-3% per cycle (monthly), ~12-24% annualized
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class CoveredCallStrategy(AbstractStrategy):
    """Sell an OTM call against an existing long equity position.

    Income generation strategy: targets ~2-5% monthly premium. Only
    generates a signal when:
      - We hold ≥100 shares of the underlying (checked via metadata parameter)
      - IV rank > 30 (elevated implied volatility means richer premium)
      - Stock is not in a strong uptrend (we don't want assignment risk)

    Execution hint: sell the nearest OTM call with ~30 delta, 21-45 DTE.
    Exit at 50% of max profit or roll at 21 DTE.
    """

    name = "covered_call"
    display_name = "Covered Call"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"          # premium collection = income / arb bucket
    tick_interval_seconds = 3600.0     # re-evaluate hourly
    confidence_threshold = 0.60

    # Parameters (Options Alpha research)
    TARGET_DTE_MIN = 21
    TARGET_DTE_MAX = 45
    TARGET_DELTA = 0.30
    MIN_IV_RANK = 30
    PROFIT_TARGET_PCT = 0.50   # exit at 50% of max premium received

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", self.MIN_IV_RANK)
        self.min_shares: int = (params or {}).get("min_shares", 100)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Signal: IV rank > threshold and position ≥ 100 shares.

        The caller must inject 'iv_rank' and 'current_shares' into
        data.attrs or the last-row metadata before calling analyze().
        """
        if len(data) < 20:
            return None

        iv_rank: float = data.attrs.get("iv_rank", 0.0)
        current_shares: int = data.attrs.get("current_shares", 0)

        if current_shares < self.min_shares:
            return None
        if iv_rank < self.iv_rank_threshold:
            return None

        # Avoid selling into strong uptrend (would cap gains)
        close = data["close"]
        sma20 = close.rolling(20).mean().iloc[-1]
        price = close.iloc[-1]
        if price > sma20 * 1.05:
            # Price running >5% above 20-day SMA — avoid capping upside
            return None

        # RSI filter: don't sell calls if stock is oversold (could bounce hard)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / max(float(loss.iloc[-1]), 0.001)))
        if rsi < 35:
            return None  # oversold — stock may snap higher, don't cap upside

        confidence = min(0.85, 0.60 + (iv_rank - self.iv_rank_threshold) / 100)
        return Signal(
            symbol=symbol,
            side="sell",              # sell the call
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "covered_call",
                "iv_rank": round(iv_rank, 2),
                "current_shares": current_shares,
                "rsi": round(float(rsi), 1),
                "target_delta": self.TARGET_DELTA,
                "target_dte_min": self.TARGET_DTE_MIN,
                "target_dte_max": self.TARGET_DTE_MAX,
                "profit_target_pct": self.PROFIT_TARGET_PCT,
                "hint": f"Sell ~{self.TARGET_DELTA}-delta call, {self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. Exit at 50% profit.",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Vectorised signal: enter when IV rank proxy > threshold.

        NOTE: Real IV rank requires historical IV data not in OHLCV.
        We approximate with realised volatility percentile as a proxy.
        """
        close = df["close"]
        # HV20 percentile proxy for IV rank
        log_ret = np.log(close / close.shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv_min = hv20.rolling(252).min()
        hv_max = hv20.rolling(252).max()
        iv_rank = (hv20 - hv_min) / (hv_max - hv_min + 0.001) * 100

        # RSI filter
        d = close.diff()
        gain = d.clip(lower=0).rolling(14).mean()
        loss = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 0.001)))

        # Signal: sell call when vol rank above threshold and not oversold
        entries = ((iv_rank.shift(1) > self.iv_rank_threshold) & (rsi.shift(1) > 35)).fillna(False)
        exits = (iv_rank.shift(1) < 20).fillna(False)  # buy back when vol collapses
        return BacktestSignals(entries=entries, exits=exits)


class CashSecuredPutStrategy(AbstractStrategy):
    """Sell an OTM put to enter a long position at a discount.

    Suited for sideways or mildly bullish markets. Collects premium while
    waiting to acquire shares at the chosen strike price.

    Signal conditions:
      - RSI(14) < 40: stock has pulled back, risk/reward is better
      - IV rank > 25: enough premium available to make the trade worthwhile

    Execution hint: sell ~30-delta put at 2-4 weeks expiry.
    Exit at 50% of max premium or roll at 21 DTE.
    """

    name = "cash_secured_put"
    display_name = "Cash Secured Put"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.60

    TARGET_DTE_MIN = 14
    TARGET_DTE_MAX = 28
    TARGET_DELTA = -0.30
    MIN_IV_RANK = 25
    PROFIT_TARGET_PCT = 0.50

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.rsi_threshold: float = (params or {}).get("rsi_threshold", 40.0)
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", self.MIN_IV_RANK)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Signal: RSI < threshold AND IV rank > threshold."""
        if len(data) < 30:
            return None

        iv_rank: float = data.attrs.get("iv_rank", 0.0)
        if iv_rank < self.iv_rank_threshold:
            return None

        close = data["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1])

        if rsi_val >= self.rsi_threshold:
            return None

        confidence = min(0.80, 0.60 + (self.rsi_threshold - rsi_val) / 100)
        return Signal(
            symbol=symbol,
            side="sell",              # sell the put
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "cash_secured_put",
                "rsi": round(rsi_val, 2),
                "iv_rank": round(iv_rank, 2),
                "target_delta": self.TARGET_DELTA,
                "target_dte_min": self.TARGET_DTE_MIN,
                "target_dte_max": self.TARGET_DTE_MAX,
                "profit_target_pct": self.PROFIT_TARGET_PCT,
                "hint": f"Sell ~{abs(self.TARGET_DELTA)}-delta put, {self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. Exit at 50% profit.",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Approximate signal using RSI as proxy (IV rank not in OHLCV)."""
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        entries = (rsi.shift(1) < self.rsi_threshold).fillna(False)
        exits = (rsi.shift(1) > 55).fillna(False)
        return BacktestSignals(entries=entries, exits=exits)


class IronCondorStrategy(AbstractStrategy):
    """Sell OTM put spread + OTM call spread simultaneously.

    Profits when the underlying stays in a defined range until expiry.
    Based on Options Alpha research: 78-82% win rate when managed at
    50% profit target and 21 DTE hard exit.

    Parameters (documented from Options Alpha research):
    - Entry: IV Rank > 50, 30-45 DTE
    - Short strikes: 15-20 delta (put spread below, call spread above)
    - Width: 5 strikes wide for liquid underlyings (SPY, QQQ, AAPL)
    - Profit target: 50% of max credit received
    - Stop loss: 200% of max credit (close when loss = 2x original credit)
    - Time exit: Close at 21 DTE regardless
    - Avoid: Do NOT sell within 5 days of earnings

    Win rate: 78-82% when managed (documented in ORATS backtest of 180M strategies)
    Sharpe target: 1.2-1.8
    """

    name = "iron_condor"
    display_name = "Iron Condor"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"          # non-directional, theta-positive
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    # Strategy parameters
    TARGET_DTE_MIN = 30
    TARGET_DTE_MAX = 45
    SHORT_DELTA_TARGET = 0.18          # 16-20 delta
    SPREAD_WIDTH = 5                   # strikes wide
    MIN_IV_RANK = 50                   # don't sell when IV cheap
    PROFIT_TARGET_PCT = 0.50           # exit at 50% of max profit
    STOP_LOSS_MULT = 2.0               # exit when loss = 2× credit received
    HARD_EXIT_DTE = 21                 # close all positions at 21 DTE

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", self.MIN_IV_RANK)
        self.trend_pct_threshold: float = (params or {}).get("trend_pct_threshold", 0.05)

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        """Generate iron condor entry signal.

        Signal fires when: IV rank > 50, RSI in 30-70 (range-bound), not near earnings.
        Uses HV20 percentile as IV rank proxy when no real options data is available.
        """
        if len(data) < 252:
            return None

        # Calculate 20-day HV as IV proxy
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        hv_20 = float(log_returns.tail(20).std() * np.sqrt(252))
        hv_series = log_returns.rolling(252).std() * np.sqrt(252)
        hv_min = float(hv_series.min())
        hv_max = float(hv_series.max())

        # IV Rank proxy using HV percentile
        iv_rank = (hv_20 - hv_min) / max(hv_max - hv_min, 0.001) * 100

        if iv_rank < self.MIN_IV_RANK:
            return None

        # RSI to check market is not trending strongly (iron condors need range-bound)
        close = data["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = float(100 - (100 / (1 + gain.iloc[-1] / max(float(loss.iloc[-1]), 0.001))))

        # Avoid entries when RSI is extreme (strongly trending market)
        if rsi > 70 or rsi < 30:
            return None

        # Also check that price is within range of SMA20 (not in parabolic move)
        sma20 = float(close.rolling(20).mean().iloc[-1])
        price = float(close.iloc[-1])
        deviation = abs(price - sma20) / sma20
        if deviation > self.trend_pct_threshold:
            return None

        # Scale confidence with IV rank (higher IV = better premium, more confidence)
        confidence = min(0.82, 0.65 + (iv_rank - self.MIN_IV_RANK) / 200)

        return Signal(
            symbol=symbol,
            side="sell",              # selling the condor (net credit)
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "iron_condor",
                "iv_rank": round(iv_rank, 1),
                "hv_20": round(hv_20, 4),
                "rsi": round(rsi, 1),
                "sma20_deviation_pct": round(deviation * 100, 2),
                "target_dte": f"{self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX}",
                "short_delta": self.SHORT_DELTA_TARGET,
                "spread_width": self.SPREAD_WIDTH,
                "profit_target_pct": self.PROFIT_TARGET_PCT,
                "stop_loss_mult": self.STOP_LOSS_MULT,
                "hard_exit_dte": self.HARD_EXIT_DTE,
                "hint": (
                    f"Sell {self.SHORT_DELTA_TARGET}-delta put spread and call spread, "
                    f"{self.SPREAD_WIDTH} strikes wide, "
                    f"{self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. "
                    f"Exit at 50% profit or {self.HARD_EXIT_DTE} DTE."
                ),
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Returns entries where IV rank > 50 and RSI is 35-65 (range-bound), else 0."""
        log_returns = np.log(df["close"] / df["close"].shift(1))
        hv_20 = log_returns.rolling(20).std() * np.sqrt(252)
        hv_min = hv_20.rolling(252).min()
        hv_max = hv_20.rolling(252).max()
        iv_rank = (hv_20 - hv_min) / (hv_max - hv_min + 0.001) * 100

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 0.001)))

        # Enter when IV elevated + market range-bound
        entries = (
            (iv_rank.shift(1) > self.MIN_IV_RANK)
            & (rsi.shift(1) > 35)
            & (rsi.shift(1) < 65)
        ).fillna(False)

        # Exit when vol collapses (IV rank < 30) or RSI goes extreme
        exits = (
            (iv_rank.shift(1) < 30)
            | (rsi.shift(1) > 72)
            | (rsi.shift(1) < 28)
        ).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)


class LongCallMomentum(AbstractStrategy):
    """Buy in-the-money calls on breakout stocks for leveraged directional exposure.

    Instead of buying shares on a breakout, buys a deep ITM call (0.70 delta)
    to get similar delta exposure with defined risk (max loss = premium paid).

    Best used when IV is LOW (cheap options) and momentum is strong. Avoid when
    IV rank > 50 (options expensive — theta drag will hurt).

    Signal conditions:
      - Price > 52-week high (breakout) with volume confirmation
      - IV rank < 40: options are cheap enough to buy
      - Volume > 1.5× 20-day average (institutional participation)

    Execution hint: buy 0.70-delta call, nearest monthly expiry ≥ 30 DTE.
    """

    name = "long_call_momentum"
    display_name = "Long Call (Momentum)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0     # 15-minute bars
    confidence_threshold = 0.60

    TARGET_DELTA = 0.70
    TARGET_DTE_MIN = 30
    TARGET_DTE_MAX = 60
    MAX_IV_RANK = 40    # only buy calls when IV is cheap

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.lookback: int = (params or {}).get("lookback", 52)
        self.vol_mult: float = (params or {}).get("vol_mult", 1.5)
        self.max_iv_rank: float = (params or {}).get("max_iv_rank", self.MAX_IV_RANK)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Breakout signal → buy 0.70-delta call instead of shares."""
        if len(data) < self.lookback + 20:
            return None

        close = data["close"]
        high = data["high"]
        volume = data.get("volume", pd.Series(dtype=float))

        # Check IV rank — don't buy options when they're expensive
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) >= 252:
            hv_20 = float(log_ret.tail(20).std() * np.sqrt(252))
            hv_series = log_ret.rolling(252).std() * np.sqrt(252)
            hv_min = float(hv_series.min())
            hv_max = float(hv_series.max())
            iv_rank = (hv_20 - hv_min) / max(hv_max - hv_min, 0.001) * 100
            if iv_rank > self.max_iv_rank:
                return None  # options too expensive; pass

        resistance = high.rolling(self.lookback).max().shift(1)
        vol_avg = volume.rolling(20).mean() if len(volume) > 0 else pd.Series(1, index=data.index)

        price = float(close.iloc[-1])
        res = float(resistance.iloc[-1])
        vol_curr = float(volume.iloc[-1]) if len(volume) > 0 else 1.0
        vol_mean = float(vol_avg.iloc[-1]) if len(volume) > 0 else 1.0

        if price > res and vol_curr > self.vol_mult * vol_mean:
            pct_break = (price - res) / res
            confidence = min(0.80, 0.55 + pct_break * 3)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": "long_call_momentum",
                    "resistance": round(res, 4),
                    "breakout_pct": round(pct_break * 100, 2),
                    "target_delta": self.TARGET_DELTA,
                    "target_dte_min": self.TARGET_DTE_MIN,
                    "target_dte_max": self.TARGET_DTE_MAX,
                    "hint": f"Buy {self.TARGET_DELTA}-delta call, {self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. Max loss = premium paid.",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Identical entry conditions to BreakoutStrategy."""
        close = df["close"]
        high = df["high"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        resistance = high.rolling(self.lookback).max().shift(2)
        vol_avg = volume.rolling(20).mean()

        entries = (
            close.shift(1).gt(resistance) & volume.shift(1).gt(self.vol_mult * vol_avg.shift(1))
        ).fillna(False)
        exits = close.shift(1).lt(resistance).fillna(False)
        return BacktestSignals(entries=entries, exits=exits)


class EarningsIVCrushStrategy(AbstractStrategy):
    """Sell ATM straddle (or strangle) 1-2 days BEFORE earnings, buy back after.

    IV typically drops 30-50% after earnings announcement regardless of direction.
    This strategy profits from the IV crush (volatility collapse) post-announcement.

    Entry: 1-2 days before earnings, IV Rank > 70
    Exit: Within 1 day after earnings announcement (IV crush is immediate)
    Max loss: 100% of premium (stock can move violently in wrong direction)
    Expected win rate: 70-75% (IV almost always drops after earnings)

    Risk management:
    - Never size > 2% of portfolio (tail risk from gap moves)
    - Use strangle (5-10% OTM) instead of straddle on volatile stocks
    - Avoid TSLA, MSTR (gap risk too high)
    - Best candidates: AAPL, MSFT, GOOGL (large-cap, liquid, contained moves)

    Execution hint: sell the nearest ATM straddle/strangle expiring 1-2 days
    after earnings (weekly options if available).
    """

    name = "earnings_iv_crush"
    display_name = "Earnings IV Crush"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    MIN_IV_RANK = 70      # only trade when IV very elevated (pre-earnings spike)
    TARGET_DTE = 2        # sell 1-2 days before earnings
    EXIT_DTE = 1          # buy back within 1 day after announcement
    MAX_POSITION_PCT = 0.02   # never more than 2% of portfolio

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", self.MIN_IV_RANK)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Signal: IV rank > 70 (pre-earnings spike), earnings within 1-2 days.

        The caller must inject 'iv_rank' and 'days_to_earnings' into
        data.attrs before calling analyze().
        days_to_earnings = 0 means today is earnings day (too late).
        days_to_earnings = 1-2 means ideal entry window.
        """
        if len(data) < 20:
            return None

        iv_rank: float = data.attrs.get("iv_rank", 0.0)
        days_to_earnings: int = data.attrs.get("days_to_earnings", 999)

        if iv_rank < self.iv_rank_threshold:
            return None

        # Only enter 1-2 days before earnings
        if days_to_earnings < 1 or days_to_earnings > 2:
            return None

        # IV crush candidate: high IV rank confirms pre-earnings premium spike
        # Scale confidence with proximity and IV elevation
        proximity_bonus = 0.05 if days_to_earnings == 1 else 0.0
        confidence = min(0.78, 0.65 + (iv_rank - self.MIN_IV_RANK) / 200 + proximity_bonus)

        return Signal(
            symbol=symbol,
            side="sell",              # sell the straddle/strangle (net credit)
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "earnings_iv_crush",
                "iv_rank": round(iv_rank, 1),
                "days_to_earnings": days_to_earnings,
                "max_position_pct": self.MAX_POSITION_PCT,
                "exit_dte": self.EXIT_DTE,
                "hint": (
                    "Sell ATM straddle or 5-10% OTM strangle expiring 1-2 days post-earnings. "
                    "Close within 1 day after announcement. Max 2% of portfolio."
                ),
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Approximate: fire when HV20-rank > 70 (earnings-like IV elevation).

        True earnings calendar data is needed for a real backtest. This proxy
        uses extreme IV spikes as a stand-in.
        """
        log_ret = np.log(df["close"] / df["close"].shift(1))
        hv_20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv_min = hv_20.rolling(252).min()
        hv_max = hv_20.rolling(252).max()
        iv_rank = (hv_20 - hv_min) / (hv_max - hv_min + 0.001) * 100

        # Enter on high-IV days (proxy for pre-earnings)
        entries = (iv_rank.shift(1) > self.MIN_IV_RANK).fillna(False)
        # Exit quickly — hold only 1-2 bars
        exits = iv_rank.shift(1).lt(iv_rank.shift(2)).fillna(False)  # exit when IV starts falling
        return BacktestSignals(entries=entries, exits=exits)


class WheelStrategy(AbstractStrategy):
    """The Wheel: Cash-Secured Put → Assignment → Covered Call → Called Away → Repeat.

    A systematic income strategy that cycles through two phases:
      Phase 1 (CSP): Sell 0.20-0.30 delta put at 30-45 DTE on bullish stocks
      Phase 2 (CC):  After assignment, sell 0.20-0.30 delta call at 30-45 DTE
      Phase 3:       If called away, restart with CSP; otherwise sell next CC

    Income: 1-3% per cycle (monthly), ~12-24% annualized
    Risk: Assignment risk (you own the stock at a below-market purchase cost basis)
    Best for: High-IV stocks you want to own (AAPL, MSFT, NVDA, etc.)

    Parameters:
    - CSP delta: 0.25 (25% probability of assignment)
    - CC delta:  0.25 (25% probability of being called away)
    - DTE: 30-45 days
    - Minimum IV Rank: 30 (need some premium)
    - Exit at 50% profit or roll at 21 DTE

    The strategy is bullish-neutral: it works best on stocks in a slow
    uptrend or sideways channel. Avoid during strong downtrends.
    """

    name = "wheel"
    display_name = "Wheel Strategy"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"        # takes directional exposure via assignment
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.60

    CSP_DELTA = 0.25
    CC_DELTA = 0.25
    TARGET_DTE_MIN = 30
    TARGET_DTE_MAX = 45
    MIN_IV_RANK = 30
    PROFIT_TARGET_PCT = 0.50     # exit at 50% of max premium
    ROLL_DTE = 21                # roll or exit at 21 DTE

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", self.MIN_IV_RANK)
        self.phase: str = (params or {}).get("phase", "csp")  # 'csp' | 'covered_call'

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate wheel entry signal based on current phase.

        Phase detection:
        - data.attrs['wheel_phase'] = 'csp' → we have cash, sell a put
        - data.attrs['wheel_phase'] = 'covered_call' → we own shares, sell a call

        IV and trend checks apply in both phases.
        """
        if len(data) < 252:
            # Fall back to shorter lookback if needed
            if len(data) < 30:
                return None

        iv_rank: float = data.attrs.get("iv_rank", 0.0)
        wheel_phase: str = data.attrs.get("wheel_phase", self.phase)
        current_shares: int = data.attrs.get("current_shares", 0)

        if iv_rank < self.iv_rank_threshold:
            return None

        # Calculate IV rank from HV if not provided externally
        if iv_rank == 0.0 and len(data) >= 252:
            log_ret = np.log(data["close"] / data["close"].shift(1)).dropna()
            hv_20 = float(log_ret.tail(20).std() * np.sqrt(252))
            hv_series = log_ret.rolling(252).std() * np.sqrt(252)
            hv_min = float(hv_series.min())
            hv_max = float(hv_series.max())
            iv_rank = (hv_20 - hv_min) / max(hv_max - hv_min, 0.001) * 100
            if iv_rank < self.iv_rank_threshold:
                return None

        close = data["close"]
        # Check trend direction — wheel works best in uptrend or sideways
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(data) >= 50 else float(close.mean())
        price = float(close.iloc[-1])

        # RSI for trend assessment
        d = close.diff()
        gain = d.clip(lower=0).rolling(14).mean()
        loss = (-d.clip(upper=0)).rolling(14).mean()
        rsi = float(100 - (100 / (1 + gain.iloc[-1] / max(float(loss.iloc[-1]), 0.001))))

        # Phase 1: CSP — need cash, stock should not be in strong downtrend
        if wheel_phase == "csp":
            if price < sma50 * 0.92:
                return None  # stock down >8% from SMA50 — avoid selling puts
            if rsi < 30:
                return None  # oversold momentum — could drop further

            confidence = min(0.78, 0.60 + (iv_rank - self.iv_rank_threshold) / 150)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": "wheel",
                    "phase": "cash_secured_put",
                    "iv_rank": round(iv_rank, 1),
                    "rsi": round(rsi, 1),
                    "csp_delta": self.CSP_DELTA,
                    "target_dte_min": self.TARGET_DTE_MIN,
                    "target_dte_max": self.TARGET_DTE_MAX,
                    "profit_target_pct": self.PROFIT_TARGET_PCT,
                    "roll_dte": self.ROLL_DTE,
                    "hint": (
                        f"Phase 1 — Sell {self.CSP_DELTA}-delta cash-secured put, "
                        f"{self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. "
                        f"Exit at 50% profit or roll at {self.ROLL_DTE} DTE."
                    ),
                },
            )

        # Phase 2: Covered Call — already own shares
        elif wheel_phase == "covered_call":
            if current_shares < 100:
                return None  # need at least 100 shares for 1 contract

            confidence = min(0.78, 0.60 + (iv_rank - self.iv_rank_threshold) / 150)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": "wheel",
                    "phase": "covered_call",
                    "iv_rank": round(iv_rank, 1),
                    "rsi": round(rsi, 1),
                    "current_shares": current_shares,
                    "cc_delta": self.CC_DELTA,
                    "target_dte_min": self.TARGET_DTE_MIN,
                    "target_dte_max": self.TARGET_DTE_MAX,
                    "profit_target_pct": self.PROFIT_TARGET_PCT,
                    "roll_dte": self.ROLL_DTE,
                    "hint": (
                        f"Phase 2 — Sell {self.CC_DELTA}-delta covered call against {current_shares} shares, "
                        f"{self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. "
                        f"Exit at 50% profit or roll at {self.ROLL_DTE} DTE."
                    ),
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Vectorized wheel signal: enter CSP phase when IV rank > 30 and uptrend."""
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        hv_20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv_min = hv_20.rolling(252).min()
        hv_max = hv_20.rolling(252).max()
        iv_rank = (hv_20 - hv_min) / (hv_max - hv_min + 0.001) * 100

        sma50 = close.rolling(50).mean()
        d = close.diff()
        gain = d.clip(lower=0).rolling(14).mean()
        loss = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 0.001)))

        # Enter wheel when: IV rank > threshold, not in strong downtrend, RSI > 35
        entries = (
            (iv_rank.shift(1) > self.MIN_IV_RANK)
            & (close.shift(1) > sma50.shift(1) * 0.92)
            & (rsi.shift(1) > 35)
        ).fillna(False)

        # Exit when IV collapses or stock breaks below SMA50 hard
        exits = (
            (iv_rank.shift(1) < 20)
            | (close.shift(1) < sma50.shift(1) * 0.90)
        ).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)


class PutProtectionOverlay(AbstractStrategy):
    """Portfolio hedge: buy OTM put on SPY/QQQ when VIX < 20 (cheap insurance).

    Acts as portfolio-level insurance, not an alpha source. Designed to run
    continuously alongside directional strategies to provide downside protection.

    Entry conditions:
      - VIX < 20: implied volatility is low, making puts cheap
      - Portfolio beta > 0.8 (inject via data.attrs['portfolio_beta'])
      - Buy 5% OTM put at 30-45 DTE on SPY or QQQ

    Risk note: buying puts means negative theta (time decay works against you).
    Keep allocation small (1-2% of portfolio for the hedge).
    """

    name = "put_protection_overlay"
    display_name = "Put Protection Overlay"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"           # hedging = risk reduction
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.60

    MAX_VIX_FOR_ENTRY = 20.0    # only buy when vol is cheap
    OTM_PCT = 0.05              # 5% out-of-the-money
    TARGET_DTE_MIN = 30
    TARGET_DTE_MAX = 45
    MIN_PORTFOLIO_BETA = 0.8

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.max_vix: float = (params or {}).get("max_vix", self.MAX_VIX_FOR_ENTRY)
        self.min_beta: float = (params or {}).get("min_portfolio_beta", self.MIN_PORTFOLIO_BETA)
        self.otm_pct: float = (params or {}).get("otm_pct", self.OTM_PCT)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Signal: VIX < threshold and portfolio is net long (beta > min_beta).

        Caller must inject 'vix' and 'portfolio_beta' into data.attrs.
        """
        if len(data) < 20:
            return None

        vix: float = data.attrs.get("vix", 25.0)
        portfolio_beta: float = data.attrs.get("portfolio_beta", 1.0)

        if vix >= self.max_vix:
            return None  # puts are too expensive right now

        if portfolio_beta < self.min_beta:
            return None  # portfolio already has low market exposure

        # Scale confidence inversely with VIX — lower VIX = cheaper puts = better deal
        confidence = min(0.80, 0.60 + (self.max_vix - vix) / 40.0)
        price = float(data["close"].iloc[-1])
        strike_hint = round(price * (1.0 - self.otm_pct), 2)

        return Signal(
            symbol=symbol,
            side="buy",               # buying the put (protection)
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "put_protection_overlay",
                "vix": round(vix, 2),
                "portfolio_beta": round(portfolio_beta, 3),
                "otm_pct": self.otm_pct,
                "strike_hint": strike_hint,
                "target_dte_min": self.TARGET_DTE_MIN,
                "target_dte_max": self.TARGET_DTE_MAX,
                "hint": (
                    f"Buy {self.otm_pct*100:.0f}% OTM put on {symbol} at ~{strike_hint}, "
                    f"{self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. "
                    f"Size: 1-2% of portfolio for insurance."
                ),
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Proxy: buy protection when realised vol is in low-percentile (cheap puts window)."""
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv_min = hv20.rolling(252).min()
        hv_max = hv20.rolling(252).max()
        # Low vol rank proxy = VIX low
        iv_rank = (hv20 - hv_min) / (hv_max - hv_min + 0.001) * 100

        # Enter when vol rank < 30 (cheap vol / low VIX environment)
        entries = (iv_rank.shift(1) < 30).fillna(False)
        # Exit when vol rank rises above 50 (put value has expanded, take profits or let run)
        exits = (iv_rank.shift(1) > 50).fillna(False)
        return BacktestSignals(entries=entries, exits=exits)


class DeltaNeutralStrangle(AbstractStrategy):
    """Sell OTM call + OTM put at equal delta, then delta-hedge with underlying.

    Collects theta decay while staying market-neutral by continuously
    rebalancing the net delta of the position.

    Entry conditions:
      - IV rank > 30 (adequate premium to collect)
      - Sell 10-15 delta call and 10-15 delta put simultaneously
    Delta hedging:
      - Rebalance net delta every day
      - Each rebalance: buy/sell underlying to bring net delta back to 0

    Risk: Gamma risk near expiry (position can lose if large move occurs).
    Exit: 50% profit target or 21 DTE, whichever first.
    """

    name = "delta_neutral_strangle"
    display_name = "Delta Neutral Strangle"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"           # net-neutral / theta-positive
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.62

    TARGET_DELTA = 0.12             # 10-15 delta per leg
    TARGET_DTE_MIN = 30
    TARGET_DTE_MAX = 45
    MIN_IV_RANK = 30
    PROFIT_TARGET_PCT = 0.50
    HARD_EXIT_DTE = 21
    REBALANCE_INTERVAL_DAYS = 1

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.iv_rank_threshold: float = (params or {}).get("iv_rank_threshold", self.MIN_IV_RANK)
        self.target_delta: float = (params or {}).get("target_delta", self.TARGET_DELTA)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Signal: IV rank > 30 and market is not in extreme trending mode.

        Caller may inject 'iv_rank' into data.attrs for real options data.
        Falls back to HV-percentile proxy when not provided.
        """
        if len(data) < 30:
            return None

        # Use injected IV rank or compute HV proxy
        iv_rank: float = data.attrs.get("iv_rank", 0.0)
        if iv_rank == 0.0 and len(data) >= 252:
            log_ret = np.log(data["close"] / data["close"].shift(1)).dropna()
            hv_20 = float(log_ret.tail(20).std() * np.sqrt(252))
            hv_series = log_ret.rolling(252).std() * np.sqrt(252)
            iv_rank = float(
                (hv_20 - float(hv_series.min()))
                / max(float(hv_series.max()) - float(hv_series.min()), 0.001)
                * 100
            )

        if iv_rank < self.iv_rank_threshold:
            return None

        # Avoid extremely trending markets (gamma risk too high)
        close = data["close"]
        d = close.diff()
        gain = d.clip(lower=0).rolling(14).mean()
        loss = (-d.clip(upper=0)).rolling(14).mean()
        rsi = float(100 - (100 / (1 + gain.iloc[-1] / max(float(loss.iloc[-1]), 0.001))))
        if rsi > 72 or rsi < 28:
            return None

        confidence = min(0.78, 0.62 + (iv_rank - self.iv_rank_threshold) / 150)

        return Signal(
            symbol=symbol,
            side="sell",              # selling both legs (net credit)
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "delta_neutral_strangle",
                "iv_rank": round(iv_rank, 1),
                "rsi": round(rsi, 1),
                "target_delta": self.target_delta,
                "target_dte_min": self.TARGET_DTE_MIN,
                "target_dte_max": self.TARGET_DTE_MAX,
                "profit_target_pct": self.PROFIT_TARGET_PCT,
                "hard_exit_dte": self.HARD_EXIT_DTE,
                "rebalance_interval_days": self.REBALANCE_INTERVAL_DAYS,
                "hint": (
                    f"Sell {self.target_delta}-delta OTM call and put on {symbol}, "
                    f"{self.TARGET_DTE_MIN}-{self.TARGET_DTE_MAX} DTE. "
                    f"Delta-hedge daily with underlying. "
                    f"Exit at 50% profit or {self.HARD_EXIT_DTE} DTE."
                ),
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Vectorized: enter when HV-rank > 30 and RSI is range-bound (35-65)."""
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        hv_20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv_min = hv_20.rolling(252).min()
        hv_max = hv_20.rolling(252).max()
        iv_rank = (hv_20 - hv_min) / (hv_max - hv_min + 0.001) * 100

        d = close.diff()
        gain = d.clip(lower=0).rolling(14).mean()
        loss = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 0.001)))

        entries = (
            (iv_rank.shift(1) > self.MIN_IV_RANK)
            & (rsi.shift(1) > 35)
            & (rsi.shift(1) < 65)
        ).fillna(False)

        exits = (
            (iv_rank.shift(1) < 20)
            | (rsi.shift(1) > 72)
            | (rsi.shift(1) < 28)
        ).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)


# ── Options Alpha Scoring Utility ─────────────────────────────────────────────

class OptionsAlphaScorer:
    """Score options opportunities by IV rank, expected move, and strategy fit.

    Based on Options Alpha research methodology (180M+ backtests).
    """

    @staticmethod
    def score_iv_rank(iv_rank: float) -> dict:
        """Return recommended strategy for current IV rank level."""
        if iv_rank > 70:
            return {
                "strategy": "iron_condor",
                "score": 0.9,
                "reason": "Very high IV — sell premium",
            }
        elif iv_rank > 50:
            return {
                "strategy": "covered_call",
                "score": 0.75,
                "reason": "High IV — income strategies",
            }
        elif iv_rank < 20:
            return {
                "strategy": "long_call_momentum",
                "score": 0.7,
                "reason": "Low IV — buy cheap options",
            }
        else:
            return {
                "strategy": "wheel",
                "score": 0.6,
                "reason": "Moderate IV — wheel income",
            }

    @staticmethod
    def expected_move(price: float, iv: float, dte: int) -> float:
        """Expected 1-sigma move: price * iv * sqrt(dte/365)"""
        import math
        return price * iv * math.sqrt(dte / 365)

    @staticmethod
    def iv_rank(current_iv: float, iv_52w_low: float, iv_52w_high: float) -> float:
        """IV Rank = (current - low) / (high - low) * 100"""
        if iv_52w_high == iv_52w_low:
            return 50.0
        return (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
