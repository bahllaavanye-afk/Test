"""
PCA Statistical Arbitrage Strategy.

Academic basis:
  Avellaneda & Lee (2010) — Statistical Arbitrage in the US Equities Market.
  Generates market-neutral signals by trading residuals from PCA factor model.

Logic:
  1. Compute PCA on return matrix of N stocks (typically top 50 S&P 500 names)
  2. Use first K components (explaining >80% variance) as factor returns
  3. For each stock: residual = actual_return - beta @ factor_returns
  4. Compute cumulative residual (s-score): z-score of residual vs rolling mean/std
  5. Entry: |s-score| > 1.5 (mean reversion expected)
  6. Exit:  |s-score| < 0.5
  7. Stop:  |s-score| > 3.5 (regime break)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

# Default basket: top 20 liquid US equities (S&P 500 large-cap)
DEFAULT_BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "JNJ", "V", "PG",
    "UNH", "HD", "MA", "BAC", "XOM",
    "ABBV", "PFE", "BRK-B", "LLY", "AVGO",
]


class PCAStatArbStrategy(AbstractStrategy):
    """
    PCA-based Statistical Arbitrage.

    Decomposes a basket of stock returns into principal components, then trades
    each stock's unexplained residual as a mean-reverting spread (s-score).

    Risk bucket: arbitrage — market-neutral by construction.
    """

    name = "pca_stat_arb"
    display_name = "PCA Statistical Arbitrage (Avellaneda-Lee)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 86_400.0  # daily
    confidence_threshold = 0.60

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.basket: list[str] = p.get("basket", DEFAULT_BASKET)
        self.n_components: int = p.get("n_components", 5)
        self.lookback: int = p.get("lookback", 60)
        self.entry_z: float = p.get("entry_z", 1.5)
        self.exit_z: float = p.get("exit_z", 0.5)
        self.stop_z: float = p.get("stop_z", 3.5)
        self.min_var_explained: float = p.get("min_var_explained", 0.80)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_prices(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Return a DataFrame of close prices, one column per symbol.

        Supports two layouts:
          1. MultiIndex columns (symbol, field) — e.g. from a basket query
          2. Flat columns named  close_<SYMBOL>  — e.g. close_AAPL, close_MSFT
        """
        if isinstance(df.columns, pd.MultiIndex):
            try:
                prices = df.xs("close", axis=1, level=1)
                # Keep only basket symbols that are present
                cols = [c for c in self.basket if c in prices.columns]
                return prices[cols] if cols else None
            except KeyError:
                pass

        # Flat  close_<SYMBOL>  layout
        mapping = {}
        for sym in self.basket:
            col = f"close_{sym}"
            if col in df.columns:
                mapping[sym] = df[col]
        if not mapping:
            return None
        return pd.DataFrame(mapping, index=df.index)

    def _compute_s_scores(
        self, prices: pd.DataFrame, end_idx: int
    ) -> dict[str, float]:
        """
        Compute current s-scores for each symbol using a lookback window
        ending at *end_idx* (exclusive).

        Returns a dict  symbol → s_score  (NaN if not enough data).
        """
        window = prices.iloc[max(0, end_idx - self.lookback) : end_idx]
        if len(window) < max(self.n_components + 5, 20):
            return {}

        returns = window.pct_change().dropna()
        if returns.shape[0] < self.n_components + 2:
            return {}

        # Standardise returns for PCA
        ret_mean = returns.mean()
        ret_std = returns.std().replace(0, np.nan)
        returns_std = (returns - ret_mean) / ret_std
        returns_std = returns_std.fillna(0.0)

        # Fit PCA
        n_comp = min(self.n_components, returns_std.shape[1] - 1, returns_std.shape[0] - 1)
        if n_comp < 1:
            return {}
        pca = PCA(n_components=n_comp)
        try:
            factors = pca.fit_transform(returns_std)  # (T, K)
        except Exception:
            return {}

        # Beta of each stock on factors via OLS
        # X = factors (T, K),  y = returns_std (T, N)
        X = np.hstack([factors, np.ones((factors.shape[0], 1))])  # add intercept
        beta, _, _, _ = np.linalg.lstsq(X, returns_std.values, rcond=None)  # (K+1, N)

        # Residuals for the full window
        residuals = returns_std.values - X @ beta  # (T, N)

        # Cumulative residuals (approximate OU process)
        cum_resid = np.cumsum(residuals, axis=0)  # (T, N)

        # S-score = z-score of cumulative residual over the same window
        cr_mean = cum_resid.mean(axis=0)
        cr_std = cum_resid.std(axis=0)
        cr_std = np.where(cr_std < 1e-9, np.nan, cr_std)
        s_scores_arr = (cum_resid[-1] - cr_mean) / cr_std  # last bar

        return {
            sym: float(s_scores_arr[i])
            for i, sym in enumerate(returns_std.columns)
        }

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a signal for *symbol* using current PCA residuals.

        *data* must contain price columns for the full basket (not just the
        target symbol) so PCA can be computed on the cross-section.
        """
        prices = self._extract_prices(data)
        if prices is None or symbol not in prices.columns:
            return None
        if len(prices) < self.lookback + 10:
            return None

        # Exclude current bar from PCA estimation to avoid lookahead bias
        s_scores = self._compute_s_scores(prices, end_idx=len(prices) - 1)
        s = s_scores.get(symbol, float("nan"))
        if np.isnan(s):
            return None

        # Stop condition: regime break
        if abs(s) > self.stop_z:
            return None

        # Entry signals (mean reversion)
        confidence = min(
            0.95,
            (abs(s) - self.entry_z) / max(self.stop_z - self.entry_z, 1e-9) * 0.35 + 0.60,
        )

        if s > self.entry_z:
            # Stock overperformed factors → expect reversion → sell/short
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"s_score": s, "n_components": self.n_components},
            )
        if s < -self.entry_z:
            # Stock underperformed factors → expect reversion → buy/long
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"s_score": s, "n_components": self.n_components},
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorised s-score signals over the full price history.

        Applies shift(1) on every signal series to eliminate lookahead bias:
        the decision on bar *t* is made using only data up to and including
        bar *t-1*.
        """
        prices = self._extract_prices(df)
        empty = pd.Series(False, index=df.index, dtype=bool)

        if prices is None or prices.empty:
            return BacktestSignals(entries=empty, exits=empty)

        # Pick the first basket symbol present as the "primary" instrument.
        # For multi-instrument use, callers should loop over basket members.
        primary = None
        for sym in self.basket:
            if sym in prices.columns:
                primary = sym
                break
        if primary is None:
            return BacktestSignals(entries=empty, exits=empty)

        n = len(prices)
        raw_signal = pd.Series(0, index=prices.index, dtype=float)

        for i in range(self.lookback + 1, n):
            s_scores = self._compute_s_scores(prices, end_idx=i)
            s = s_scores.get(primary, float("nan"))
            if np.isnan(s):
                continue
            if abs(s) > self.stop_z:
                raw_signal.iloc[i] = 0  # stop — close position
            elif s < -self.entry_z:
                raw_signal.iloc[i] = 1  # long entry
            elif s > self.entry_z:
                raw_signal.iloc[i] = -1  # short entry
            elif abs(s) < self.exit_z:
                raw_signal.iloc[i] = 0  # exit

        # Shift(1) — decision at bar t uses only data known at bar t-1
        raw_signal = raw_signal.shift(1)

        entries = (raw_signal > 0.5).fillna(False)
        short_entries = (raw_signal < -0.5).fillna(False)
        exits = (raw_signal == 0).fillna(False)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=exits.copy(),
        )
