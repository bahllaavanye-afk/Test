"""
Cross-Sectional Residual Momentum
====================================
Source: Blitz, Huij, Martens (2011) "Residual Momentum", Journal of Empirical Finance.

Classical momentum (Jegadeesh & Titman 1993, 12-1 return ranking) earns positive
average returns but is heavily contaminated by static factor exposures: it tends
to be implicitly long high-beta, small-cap, growth (anti-value) names. Those
exposures are themselves time-varying risk premia, so a chunk of the "momentum
alpha" is just factor beta in disguise — and crashes whenever those factors
unwind together (e.g. Q1 2009, Feb 2021).

Residual momentum solves this by FIRST orthogonalising stock returns against a
small set of common factors, then ranking on the cumulative RESIDUAL return.

Our practical implementation uses tradable factor proxies in lieu of the
academic Fama-French series, because we ingest those from Alpaca anyway:
  - SPY  → market           (Mkt-Rf proxy)
  - IWM  → small caps        (SMB     proxy)
  - VTV  → large-cap value   (HML     proxy)

For each stock, rolling OLS:
    r_stock,t = a + b1 r_SPY,t + b2 r_IWM,t + b3 r_VTV,t + e_t

Signal = sum of e_t over t ∈ [-252, -21]  (skip-1 skip-12 window, same as
classical momentum).  Long the top-decile of cross-sectional residual scores.

Documented (Blitz et al. 2011, Tab. 4): residual momentum ~ 2× the Sharpe of
total-return momentum with a far smaller crash drawdown.
"""

import asyncio
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute OLS residuals given regressor matrix X (n x k, intercept included)."""
    # Numerically safe via lstsq
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


class ResidualMomentumStrategy(AbstractStrategy):
    name = "residual_momentum"
    display_name = "Residual Momentum (Blitz et al. 2011)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    FACTOR_PROXIES = ["SPY", "IWM", "VTV"]
    UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "JPM", "V", "MA", "UNH", "JNJ", "XOM", "HD", "LLY",
        "AVGO", "CRM", "ORCL", "AMD", "COST", "TMO", "AMAT", "MU",
    ]

    LOOKBACK_DAYS = 252
    SKIP_DAYS = 21
    TOP_N = 5
    RANK_THRESHOLD = 0.85

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

    async def _fetch_closes(self, symbol: str, days: int) -> pd.Series:
        start = (date.today() - timedelta(days=days + 30)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Day",
                        "start": start,
                        "limit": days + 30,
                        "feed": "iex",
                    },
                    headers=self._headers(),
                )
            if resp.status_code != 200:
                return pd.Series(dtype=float, name=symbol)
            bars = resp.json().get("bars", [])
            if not bars:
                return pd.Series(dtype=float, name=symbol)
            s = pd.Series(
                {b["t"]: float(b["c"]) for b in bars},
                name=symbol,
            )
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            return pd.Series(dtype=float, name=symbol)

    def _residual_score(self, stock_rets: pd.Series, factor_rets: pd.DataFrame) -> float | None:
        """
        Fit OLS over LOOKBACK_DAYS window, sum residuals over [-LOOKBACK:-SKIP].
        Returns None on insufficient data.
        """
        aligned = pd.concat([stock_rets, factor_rets], axis=1).dropna()
        if len(aligned) < self.LOOKBACK_DAYS - 30:
            return None
        window = aligned.tail(self.LOOKBACK_DAYS)
        y = window.iloc[:, 0].values.astype(float)
        F = window.iloc[:, 1:].values.astype(float)
        X = np.column_stack([np.ones(len(F)), F])
        resids = _ols_residuals(y, X)
        # sum residuals over the [-LOOKBACK : -SKIP] window
        usable = resids[: -self.SKIP_DAYS] if self.SKIP_DAYS > 0 else resids
        if len(usable) < 60:
            return None
        return float(np.sum(usable))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if symbol not in self.UNIVERSE:
            return None

        symbols_to_fetch = list(self.UNIVERSE) + list(self.FACTOR_PROXIES)
        series = await asyncio.gather(
            *(self._fetch_closes(s, self.LOOKBACK_DAYS) for s in symbols_to_fetch),
            return_exceptions=True,
        )
        closes: dict[str, pd.Series] = {}
        for s, r in zip(symbols_to_fetch, series):
            if isinstance(r, pd.Series) and not r.empty:
                closes[s] = r

        # Need all factor proxies + the asked-about symbol
        if any(f not in closes for f in self.FACTOR_PROXIES):
            return None
        if symbol not in closes:
            return None

        # Compute daily log-returns
        factor_rets = pd.DataFrame({f: np.log(closes[f]).diff() for f in self.FACTOR_PROXIES})

        scores: dict[str, float] = {}
        for s in self.UNIVERSE:
            ser = closes.get(s)
            if ser is None or len(ser) < self.LOOKBACK_DAYS - 30:
                continue
            stock_rets = np.log(ser).diff().rename(s)
            sc = self._residual_score(stock_rets, factor_rets)
            if sc is not None:
                scores[s] = sc

        if symbol not in scores or len(scores) < 10:
            return None

        # Cross-sectional rank of this symbol
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        rank_idx = next(i for i, (s, _) in enumerate(sorted_scores) if s == symbol)
        rank_pct = 1.0 - (rank_idx / max(len(scores) - 1, 1))

        if rank_idx >= self.TOP_N:
            return None
        if rank_pct < self.RANK_THRESHOLD:
            return None

        confidence = float(min(0.60 + 0.35 * rank_pct, 0.98))
        spot = float(closes[symbol].iloc[-1])

        return Signal(
            symbol=symbol,
            side="buy",
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=spot,
            metadata={
                "strategy": self.name,
                "rank": rank_idx + 1,
                "rank_pct": round(rank_pct, 3),
                "residual_score": round(scores[symbol], 5),
                "universe_size": len(scores),
                "factor_proxies": self.FACTOR_PROXIES,
                "lookback_days": self.LOOKBACK_DAYS,
                "skip_days": self.SKIP_DAYS,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Single-asset backtest fallback: residualise own returns against a rolling
        regression on the symbol's own past (effectively detrending), then rank
        rolling residual-momentum vs its own history. True cross-sectional
        residual momentum requires a panel and is computed live in analyze().
        """
        if "close" not in df.columns or len(df) < self.LOOKBACK_DAYS + 10:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )
        close = df["close"].astype(float)
        log_ret = np.log(close).diff()

        # Use rolling mean as a one-factor (market trend) proxy when no panel exists
        market_proxy = log_ret.rolling(63, min_periods=20).mean()
        # rolling beta of log_ret onto market_proxy via covariance/variance
        cov = log_ret.rolling(self.LOOKBACK_DAYS, min_periods=60).cov(market_proxy)
        var = market_proxy.rolling(self.LOOKBACK_DAYS, min_periods=60).var()
        beta = cov / var.replace(0.0, np.nan)
        residual = log_ret - beta * market_proxy

        # Residual momentum signal: sum residuals over (lookback - skip) past bars,
        # shifted by SKIP_DAYS to implement the skip-1 month convention.
        score = residual.rolling(self.LOOKBACK_DAYS - self.SKIP_DAYS, min_periods=120).sum()
        score = score.shift(self.SKIP_DAYS)

        # Self-history percentile rank of score (proxy for cross-sectional rank).
        rolling_rank = score.rolling(self.LOOKBACK_DAYS, min_periods=60).apply(
            lambda w: (w < w.iloc[-1]).mean(),
            raw=False,
        )

        entries = (rolling_rank.shift(1) > self.RANK_THRESHOLD).fillna(False)
        # Exit when rank falls out of the top tier
        exits = (rolling_rank.shift(1) < 0.50).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)
