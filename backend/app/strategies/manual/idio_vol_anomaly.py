"""
Idiosyncratic Volatility Anomaly
====================================
Source: Ang, Hodrick, Xing, Zhang (2006) "The Cross-Section of Volatility and
Expected Returns", Journal of Finance.

Classical theory (CAPM, Merton's ICAPM) predicts that idiosyncratic risk is
diversifiable and should NOT be priced. Ang, Hodrick, Xing & Zhang (2006) found
exactly the opposite empirically: stocks with HIGH idiosyncratic volatility have
SIGNIFICANTLY LOWER subsequent returns than low-idio-vol peers. This is one of
the most replicated cross-sectional anomalies in finance.

Mechanism candidates (still debated):
  - Retail "lottery preference" for high-vol stocks bids them up beyond fair
    value, with subsequent under-performance.
  - Arbitrage-cost asymmetry: shorts are constrained, so overpricing persists.
  - High idio-vol correlates with poor earnings quality / opacity.

Strategy: cross-sectional dollar-neutral
  - For each stock, estimate idio-vol over rolling 21 trading days:
        r_stock,t = a + b r_SPY,t + e_t          → idio_vol = std(e_t) × √252
  - LONG  bottom decile of idio-vol (boring, low-vol names)
  - SHORT top decile     of idio-vol (lottery-like names)
  - Rebalance monthly; equal-weight within each leg.

We treat this asymmetrically: emit BUY signal for symbols in the bottom decile
and SELL signal for symbols in the top decile. The portfolio engine combines
these into a long/short book.
"""

import asyncio
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"


class IdiosyncraticVolAnomalyStrategy(AbstractStrategy):
    name = "idio_vol_anomaly"
    display_name = "Idiosyncratic Vol Anomaly (Ang et al. 2006)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"  # equity long/short
    tick_interval_seconds = 86400.0

    UNIVERSE = [
        # Large-cap quality
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "JPM", "V", "MA", "UNH", "JNJ", "XOM", "HD", "LLY",
        "AVGO", "CRM", "ORCL", "AMD", "COST", "TMO",
        "PG", "KO", "PEP", "WMT", "DIS", "BAC", "MRK", "CVX",
        "ABBV", "MCD", "ABT", "ACN", "ADBE", "T", "VZ", "NFLX",
        "NKE", "CSCO", "INTC", "AMGN", "GE", "HON", "UPS", "CAT",
        # Higher-vol / meme-prone
        "GME", "AMC", "PLTR", "RIVN", "LCID", "COIN", "MARA", "RIOT",
        "SOFI", "HOOD", "DKNG", "NIO",
    ]

    MARKET_PROXY = "SPY"
    IDIO_WINDOW = 21
    DECILE = 0.10  # top/bottom 10%

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    async def _fetch_closes(self, symbol: str, days: int = 40) -> pd.Series:
        start = (date.today() - timedelta(days=days + 10)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Day",
                        "start": start,
                        "limit": days + 10,
                        "feed": "iex",
                    },
                    headers=alpaca_headers(),
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

    @staticmethod
    def _idio_vol(stock_ret: np.ndarray, mkt_ret: np.ndarray) -> float:
        """OLS residual std (annualised) from regression r_stock = a + b*r_mkt + e."""
        if len(stock_ret) < 10:
            return float("nan")
        X = np.column_stack([np.ones(len(mkt_ret)), mkt_ret])
        coef, *_ = np.linalg.lstsq(X, stock_ret, rcond=None)
        resid = stock_ret - X @ coef
        return float(np.std(resid, ddof=1) * np.sqrt(252))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if symbol not in self.UNIVERSE:
            return None

        symbols = list(self.UNIVERSE) + [self.MARKET_PROXY]
        series = await asyncio.gather(
            *(self._fetch_closes(s, days=40) for s in symbols),
            return_exceptions=True,
        )
        closes: dict[str, pd.Series] = {
            s: r for s, r in zip(symbols, series)
            if isinstance(r, pd.Series) and not r.empty
        }
        if self.MARKET_PROXY not in closes or symbol not in closes:
            return None

        mkt_close = closes[self.MARKET_PROXY]
        mkt_ret_full = np.log(mkt_close).diff().dropna()

        idio_vols: dict[str, float] = {}
        for s, ser in closes.items():
            if s == self.MARKET_PROXY:
                continue
            r = np.log(ser).diff().dropna()
            aligned = pd.concat([r, mkt_ret_full], axis=1, join="inner").dropna().tail(self.IDIO_WINDOW)
            if len(aligned) < max(15, self.IDIO_WINDOW - 3):
                continue
            iv = self._idio_vol(
                aligned.iloc[:, 0].values.astype(float),
                aligned.iloc[:, 1].values.astype(float),
            )
            if np.isfinite(iv):
                idio_vols[s] = iv

        if symbol not in idio_vols or len(idio_vols) < 10:
            return None

        # Cross-sectional rank (0 = lowest idio-vol, 1 = highest)
        sorted_pairs = sorted(idio_vols.items(), key=lambda kv: kv[1])
        rank_idx = next(i for i, (s, _) in enumerate(sorted_pairs) if s == symbol)
        rank_pct = rank_idx / max(len(sorted_pairs) - 1, 1)

        n = len(sorted_pairs)
        bottom_cut = max(1, int(round(n * self.DECILE)))
        top_cut = n - bottom_cut

        spot = float(closes[symbol].iloc[-1])

        if rank_idx < bottom_cut:
            # Low idio-vol → LONG
            confidence = float(min(0.60 + 0.30 * (1.0 - rank_pct), 0.95))
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
                    "leg": "long_low_vol",
                    "idio_vol": round(idio_vols[symbol], 4),
                    "rank": rank_idx + 1,
                    "rank_pct": round(rank_pct, 3),
                    "universe_size": n,
                    "window_days": self.IDIO_WINDOW,
                },
            )

        if rank_idx >= top_cut:
            confidence = float(min(0.60 + 0.30 * rank_pct, 0.95))
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=spot,
                metadata={
                    "strategy": self.name,
                    "leg": "short_high_vol",
                    "idio_vol": round(idio_vols[symbol], 4),
                    "rank": rank_idx + 1,
                    "rank_pct": round(rank_pct, 3),
                    "universe_size": n,
                    "window_days": self.IDIO_WINDOW,
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Single-asset backtest fallback: use rolling 21-bar return std as idio-vol
        proxy (full residualisation against SPY requires a panel; computed live
        in analyze()). Self-history percentile decides the leg.
        """
        if "close" not in df.columns or len(df) < self.IDIO_WINDOW + 60:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )

        close = df["close"].astype(float)
        log_ret = np.log(close).diff()
        # Use a market detrend: subtract rolling mean (a 1-factor stand-in for SPY beta on a single series)
        market_proxy = log_ret.rolling(63, min_periods=20).mean()
        resid = log_ret - market_proxy
        idio_vol = resid.rolling(self.IDIO_WINDOW, min_periods=15).std() * np.sqrt(252)

        # Percentile rank of current idio_vol vs trailing 252 bars (proxy for cross-sectional rank)
        rank = idio_vol.rolling(252, min_periods=60).apply(
            lambda w: (w < w.iloc[-1]).mean(),
            raw=False,
        )

        long_entries = (rank.shift(1) < self.DECILE).fillna(False)
        long_exits = (rank.shift(1) > 0.50).fillna(False)
        short_entries = (rank.shift(1) > (1.0 - self.DECILE)).fillna(False)
        short_exits = (rank.shift(1) < 0.50).fillna(False)

        return BacktestSignals(
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
