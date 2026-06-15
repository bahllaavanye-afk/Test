"""
Volatility Targeting — the single highest-impact technique for scaling Sharpe.

THEORY (why this is crucial):
  If you have N uncorrelated strategies each with Sharpe_i, the combined
  portfolio Sharpe ≈ sqrt(Σ Sharpe_i²).  For equal Sharpe=0.5 strategies:
    16 strategies → Sharpe 2.0
    36 strategies → Sharpe 3.0
    64 strategies → Sharpe 4.0

  This only holds if each strategy contributes EQUAL RISK. Without vol
  targeting, high-vol strategies dominate and the math breaks down.
  Vol targeting normalises each strategy to a common risk budget.

HOW IT WORKS:
  1. Measure each strategy's trailing 20-day annualised volatility.
  2. Compute scalar = target_vol / realised_vol.
  3. Multiply the base position by this scalar (capped at [0.1, 3.0]).
  4. At the portfolio level, additionally scale down if total vol > target.

REFERENCE:
  Hurst, Ooi, Pedersen (2012) — "A Century of Evidence on Trend-Following Returns"
  AQR Capital: "Managing the Risk of Trend-Following Strategies"
  Moreira & Muir (2017) — "Volatility-Managed Portfolios" (Journal of Finance)
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class VolatilityTargeter:
    """
    Per-strategy and portfolio-level volatility targeting.

    Usage (in strategy_runner.py):
        vol_targeter = VolatilityTargeter(target_vol=0.10)
        scalar = vol_targeter.get_scalar("momentum_SPY", latest_return)
        adjusted_qty = base_qty * scalar
    """

    TRADING_DAYS = 252
    MIN_SCALAR = 0.1    # never allocate less than 10% of base size
    MAX_SCALAR = 3.0    # never lever more than 3× even if vol is very low

    def __init__(
        self,
        target_vol: float = 0.10,      # 10% annualised portfolio vol
        lookback_days: int = 20,        # rolling window for realised vol estimation
        ewm_halflife: int = 10,         # EWM half-life for fast adaptation (days)
        use_ewm: bool = True,           # EWM (AQR-style) vs simple rolling
        portfolio_vol_cap: float = 0.15, # hard cap on estimated portfolio vol
    ):
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        self.ewm_halflife = ewm_halflife
        self.use_ewm = use_ewm
        self.portfolio_vol_cap = portfolio_vol_cap

        # Deques of daily returns per strategy key
        self._returns: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=lookback_days * 2)
        )
        # Cached scalars (recomputed when new returns arrive)
        self._scalars: dict[str, float] = {}

    def record_return(self, strategy_key: str, daily_return: float) -> None:
        """
        Call at end of each trading day with the strategy's realised P&L / equity.

        Args:
            strategy_key: e.g. "momentum_SPY" or "funding_rate_arb_BTCUSDT"
            daily_return: fraction (0.01 = +1%)
        """
        self._returns[strategy_key].append(daily_return)
        # Invalidate cached scalar so it is recomputed on next get_scalar call
        self._scalars.pop(strategy_key, None)

    def get_scalar(self, strategy_key: str, latest_return: float | None = None) -> float:
        """
        Return the volatility scalar for the given strategy.
        scalar > 1 means we're running below target vol → increase size.
        scalar < 1 means we're above target vol → reduce size.

        Args:
            strategy_key: same key used in record_return()
            latest_return: if provided, updates the history before computing
        """
        if latest_return is not None:
            self.record_return(strategy_key, latest_return)

        if strategy_key in self._scalars:
            return self._scalars[strategy_key]

        scalar = self._compute_scalar(strategy_key)
        self._scalars[strategy_key] = scalar
        return scalar

    def _compute_scalar(self, strategy_key: str) -> float:
        returns_deque = self._returns[strategy_key]
        if len(returns_deque) < 5:
            return 1.0  # not enough history — use base sizing

        returns = np.array(returns_deque)

        if self.use_ewm:
            # EWM variance (AQR-style): faster adaptation to volatility regimes
            alpha = 1.0 - 0.5 ** (1.0 / self.ewm_halflife)
            weights = np.array(
                [(1 - alpha) ** i for i in range(len(returns) - 1, -1, -1)]
            )
            weights /= weights.sum()
            mean = np.dot(weights, returns)
            variance = np.dot(weights, (returns - mean) ** 2)
        else:
            variance = np.var(returns, ddof=1)

        daily_vol = np.sqrt(max(variance, 1e-10))
        annualised_vol = daily_vol * np.sqrt(self.TRADING_DAYS)

        if annualised_vol < 1e-6:
            return self.MAX_SCALAR  # essentially no vol → full leverage

        scalar = self.target_vol / annualised_vol
        return float(np.clip(scalar, self.MIN_SCALAR, self.MAX_SCALAR))

    def get_portfolio_scalar(self, strategy_keys: list[str]) -> float:
        """
        Compute a portfolio-level scalar based on the cross-strategy
        average realised vol. Used to further scale down in high-vol regimes.

        Returns a scalar in [0.5, 1.0] — never levers the whole portfolio.
        """
        if not strategy_keys:
            return 1.0

        vols = []
        for key in strategy_keys:
            returns_deque = self._returns[key]
            if len(returns_deque) < 5:
                continue
            returns = np.array(returns_deque)
            daily_vol = np.std(returns, ddof=1)
            vols.append(daily_vol * np.sqrt(self.TRADING_DAYS))

        if not vols:
            return 1.0

        portfolio_vol = float(np.mean(vols))
        if portfolio_vol > self.portfolio_vol_cap:
            # Clamp total portfolio vol to cap
            return float(np.clip(self.portfolio_vol_cap / portfolio_vol, 0.5, 1.0))
        return 1.0

    def get_stats(self, strategy_key: str) -> dict:
        """Return diagnostics for the given strategy."""
        returns_deque = self._returns[strategy_key]
        if len(returns_deque) < 5:
            return {"strategy_key": strategy_key, "n_obs": len(returns_deque), "scalar": 1.0}
        returns = np.array(returns_deque)
        daily_vol = np.std(returns, ddof=1)
        ann_vol = daily_vol * np.sqrt(self.TRADING_DAYS)
        ann_return = np.mean(returns) * self.TRADING_DAYS
        sharpe = ann_return / ann_vol if ann_vol > 1e-6 else 0.0
        return {
            "strategy_key": strategy_key,
            "n_obs": len(returns_deque),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "ann_return_pct": round(ann_return * 100, 2),
            "sharpe": round(sharpe, 3),
            "scalar": round(self.get_scalar(strategy_key), 4),
            "target_vol_pct": round(self.target_vol * 100, 2),
        }

    def get_all_stats(self) -> list[dict]:
        return [self.get_stats(k) for k in sorted(self._returns.keys())]


# Module-level singleton — same lifecycle as the process
vol_targeter = VolatilityTargeter(target_vol=0.10, lookback_days=20, use_ewm=True)
