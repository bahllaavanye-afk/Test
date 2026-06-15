"""
Hidden Markov Model (HMM) Regime Detection + Adaptive Strategy Switching
=========================================================================
Most strategies only work in specific regimes. The HMM identifies:
  Regime 0: Bull/Low-Vol  (strong uptrend, low volatility) → run momentum
  Regime 1: Bear/High-Vol (downtrend, high volatility) → run VRP/mean reversion
  Regime 2: Crisis        (extreme volatility, panic) → cash or long vol

Academic: Ang & Timmermann (2012) "Regime Changes and Financial Markets"
          Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary"
Implementation: hmmlearn GaussianHMM with n_states=3
Features: [daily_return, rolling_vol_5d, rolling_vol_20d]
Documented: Regime-aware portfolios outperform buy-and-hold by 2-3% annually

This strategy:
1. Fits a 3-state Gaussian HMM on 2 years of daily returns + volatility
2. Identifies current regime
3. Returns the appropriate sub-strategy signal
"""
import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class HMMRegimeStrategy(AbstractStrategy):
    """
    Regime-aware strategy that switches between momentum (bull),
    mean-reversion (neutral), and defensive (bear/crisis) based on
    a fitted Hidden Markov Model.
    """
    name = "hmm_regime"
    display_name = "HMM Regime Detection"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    N_STATES = 3
    MIN_TRAIN_BARS = 252  # 1 year minimum

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        vol_5 = log_ret.rolling(5).std().fillna(log_ret.std())
        vol_20 = log_ret.rolling(20).std().fillna(log_ret.std())
        # Feature matrix: [return, short_vol, long_vol]
        X = np.column_stack([log_ret.values, vol_5.values, vol_20.values])
        return X

    def _fit_hmm(self, X: np.ndarray) -> tuple:
        """Fit HMM and return (model, current_state, regime_map)."""
        model = GaussianHMM(
            n_components=self.N_STATES,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)
        states = model.predict(X)
        current_state = int(states[-1])

        # Map states to regimes by volatility level
        # State with lowest mean vol → bull, highest → crisis
        state_vols = [X[states == s, 2].mean() if (states == s).any() else 0.0
                      for s in range(self.N_STATES)]
        sorted_states = np.argsort(state_vols)
        # sorted_states[0] = low vol (bull), [1] = medium (neutral), [2] = high (crisis)
        regime_map = {
            int(sorted_states[0]): "bull",
            int(sorted_states[1]): "neutral",
            int(sorted_states[2]): "crisis",
        }
        return model, current_state, regime_map

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        if not HMM_AVAILABLE:
            # Fallback: simple vol-based regime
            return await self._vol_regime_signal(data, symbol)

        if len(data) < self.MIN_TRAIN_BARS:
            return None

        X = self._extract_features(data)
        try:
            model, current_state, regime_map = self._fit_hmm(X)
        except Exception:
            return await self._vol_regime_signal(data, symbol)

        regime = regime_map.get(current_state, "neutral")

        # Generate signal based on regime
        recent_mom = (data["close"].iloc[-1] - data["close"].iloc[-5]) / data["close"].iloc[-5]

        if regime == "bull":
            side = "buy"
            confidence = 0.7
        elif regime == "neutral":
            # Mean reversion
            side = "sell" if recent_mom > 0.01 else "buy"
            confidence = 0.5
        else:  # crisis
            return None  # Stay in cash during crisis

        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "hmm_regime",
                "regime": regime,
                "hmm_state": current_state,
                "recent_momentum_5d": round(float(recent_mom), 4),
                "recommended_sub_strategy": {
                    "bull": "momentum",
                    "neutral": "vrp_systematic or mean_reversion",
                    "crisis": "stay_in_cash or long_vol",
                }.get(regime),
            },
        )

    async def _vol_regime_signal(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Fallback when hmmlearn not installed: vol-percentile regime."""
        if len(data) < 60:
            return None
        log_ret = np.log(data["close"] / data["close"].shift(1)).dropna()
        vol_5 = log_ret.tail(5).std() * np.sqrt(252)
        vol_252 = log_ret.std() * np.sqrt(252)
        vol_rank = float(vol_5 / max(vol_252, 0.001))

        if vol_rank > 1.5:
            return None  # Crisis: stay in cash
        elif vol_rank < 0.7:
            regime = "bull"
            mom = (data["close"].iloc[-1] - data["close"].iloc[-20]) / data["close"].iloc[-20]
            side = "buy" if mom > 0 else "sell"
            return Signal(
                symbol=symbol,
                side=side,
                confidence=0.65,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": "hmm_regime",
                    "regime": regime,
                    "vol_rank": round(vol_rank, 3),
                },
            )
        else:
            return None  # Neutral: no signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < self.MIN_TRAIN_BARS or not HMM_AVAILABLE:
            log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
            vol_5 = log_ret.rolling(5).std()
            vol_252 = log_ret.rolling(252).std()
            vol_rank = vol_5 / vol_252.clip(lower=0.0001)
            bull = vol_rank < 0.7
            crisis = vol_rank > 1.5
            mom = df["close"].pct_change(20)

            entries = (bull & (mom > 0)).shift(1).fillna(False)
            exits = (crisis | (bull & (mom < 0))).shift(1).fillna(False)
            short_entries = (bull & (mom < 0)).shift(1).fillna(False)
            short_exits = (crisis | (bull & (mom > 0))).shift(1).fillna(False)

            return BacktestSignals(
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
            )

        X = self._extract_features(df)
        try:
            model, _, regime_map = self._fit_hmm(X)
            states = model.predict(X)
            regimes = pd.Series(
                [regime_map.get(int(s), "neutral") for s in states],
                index=df.index,
            )
        except Exception:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )

        mom = df["close"].pct_change(20)
        is_bull = regimes == "bull"
        is_crisis = regimes == "crisis"

        entries = (is_bull & (mom > 0)).shift(1).fillna(False)
        exits = (is_crisis | (~is_bull)).shift(1).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)
