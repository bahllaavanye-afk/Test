"""
Multi-Factor Cross-Sectional Equity Strategy (Citadel GQS model)
=================================================================
Scores equities on 5 Fama-French + momentum + quality factors:
  1. Value:    earnings yield (E/P) from yfinance info
  2. Momentum: 12-1 month return (skip last month — Jegadeesh & Titman 1993)
  3. Quality:  ROE from yfinance fundamentals
  4. Low-Vol:  1-year realized volatility (lower = better, Frazzini & Pedersen 2014)
  5. Size:     log(market_cap) — small-cap premium (Fama & French 1993)

Combines into composite z-score. Universe: 13 liquid sector ETFs.
Long top 3 by composite score, short bottom 3. Rebalance monthly.

Academic:
  - Fama & French (1993, 2015) 3-factor and 5-factor models
  - Jegadeesh & Titman (1993) momentum
  - Frazzini & Pedersen (2014) Betting Against Beta (low-vol premium)
  - Novy-Marx (2013) quality / profitability factor
  - Documented Sharpe: 0.9-1.6 for multi-factor combination (Israel et al. 2017)
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

try:
    import yfinance as yf  # free data, no key needed
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


class MultiFactorEquity(AbstractStrategy):
    """
    Multi-factor cross-sectional equity strategy (Fama-French + Momentum + Quality).

    Scores each equity on 5 factors and combines into composite z-score.
    Universe: ["SPY","QQQ","IWM","XLE","XLF","XLK","XLV","XLI","XLP","XLRE","XLU","XLB","XLC"]
    Long top 3 by score, short bottom 3. Rebalance monthly.

    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    """

    name = "multi_factor_equity"
    display_name = "Multi-Factor Equity (GQS)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0  # daily — rebalance monthly

    UNIVERSE = [
        "SPY", "QQQ", "IWM",
        "XLE", "XLF", "XLK", "XLV", "XLI",
        "XLP", "XLRE", "XLU", "XLB", "XLC",
    ]

    # Factor weights (sum to 1.0)
    FACTOR_WEIGHTS = {
        "value":    0.20,
        "momentum": 0.30,
        "quality":  0.20,
        "low_vol":  0.15,
        "size":     0.15,
    }

    TOP_N = 3    # long top N
    BOT_N = 3    # short bottom N

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _get_factor_data(self, ticker: str) -> dict:
        """
        Fetch factor data from yfinance (free).
        Returns dict with raw factor values; missing values set to NaN.
        Raises RuntimeError if yfinance is not installed.
        """
        if not _YF_AVAILABLE:
            raise RuntimeError(
                "yfinance is required for MultiFactorEquity. Install with: pip install yfinance"
            )

        t = yf.Ticker(ticker)
        info = t.info or {}

        # ── Factor 1: Value (earnings yield = E/P) ──────────────────────────
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        ep_yield = 1.0 / pe_ratio if pe_ratio and pe_ratio > 0 else float("nan")

        # ── Factor 3: Quality (ROE) ──────────────────────────────────────────
        roe = info.get("returnOnEquity")
        if roe is None:
            roe = float("nan")
        else:
            roe = float(roe)

        # ── Factor 5: Size (log market cap) ─────────────────────────────────
        mkt_cap = info.get("marketCap")
        log_size = math.log(mkt_cap) if mkt_cap and mkt_cap > 0 else float("nan")

        return {
            "value": ep_yield,
            "quality": roe,
            "size": log_size,
        }

    def _get_price_factors(self, ticker: str, lookback_days: int = 252) -> dict:
        """
        Compute price-based factors from yfinance historical prices.
        Returns momentum (12-1 month) and low_vol (realized vol).
        """
        if not _YF_AVAILABLE:
            raise RuntimeError("yfinance is required for MultiFactorEquity")

        t = yf.Ticker(ticker)
        hist = t.history(period="14mo")  # ~14 months of daily data

        if hist.empty or len(hist) < 200:
            return {"momentum": float("nan"), "low_vol": float("nan")}

        closes = hist["Close"]

        # ── Factor 2: Momentum (12-1 month, skip last month) ─────────────────
        # 12-month return excluding most recent month
        # approx: return from 252 bars ago to 21 bars ago
        if len(closes) >= 252:
            mom = float(closes.iloc[-21] / closes.iloc[-252] - 1.0)
        else:
            mom = float("nan")

        # ── Factor 4: Low-Vol (1-year realized vol — lower is better) ────────
        log_rets = np.log(closes / closes.shift(1)).dropna()
        if len(log_rets) >= 20:
            realized_vol = float(log_rets.iloc[-252:].std() * np.sqrt(252))
        else:
            realized_vol = float("nan")

        return {"momentum": mom, "low_vol": realized_vol}

    @staticmethod
    def _zscore_series(values: dict[str, float]) -> dict[str, float]:
        """Cross-sectional z-score across all tickers in the universe."""
        vals = np.array(list(values.values()), dtype=float)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std < 1e-10:
            return {k: 0.0 for k in values}
        return {k: float((v - mean) / std) if not math.isnan(v) else 0.0
                for k, v in values.items()}

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        """
        Compute multi-factor scores across the universe.
        Returns buy signal for top-3, sell for bottom-3.
        Signals are issued for the passed-in symbol if it is in the top/bottom buckets.
        """
        if not _YF_AVAILABLE:
            raise RuntimeError("yfinance is required for MultiFactorEquity")

        # Gather factor data concurrently in a thread pool
        loop = asyncio.get_running_loop()

        def _gather_all():
            results = {}
            for tkr in self.UNIVERSE:
                try:
                    fd = self._get_factor_data(tkr)
                    pf = self._get_price_factors(tkr)
                    results[tkr] = {**fd, **pf}
                except Exception:
                    results[tkr] = {
                        "value": float("nan"),
                        "momentum": float("nan"),
                        "quality": float("nan"),
                        "low_vol": float("nan"),
                        "size": float("nan"),
                    }
            return results

        factor_data = await loop.run_in_executor(None, _gather_all)

        # ── Cross-sectional z-scores ────────────────────────────────────────
        # For each factor build {ticker: raw_value} then z-score
        factor_zscores: dict[str, dict[str, float]] = {}
        for factor in ["value", "momentum", "quality", "low_vol", "size"]:
            raw = {tkr: factor_data[tkr].get(factor, float("nan")) for tkr in self.UNIVERSE}
            zs = self._zscore_series(raw)
            # Low-vol: invert so lower vol = higher score
            if factor == "low_vol":
                zs = {k: -v for k, v in zs.items()}
            # Size: invert so smaller cap = higher score
            if factor == "size":
                zs = {k: -v for k, v in zs.items()}
            factor_zscores[factor] = zs

        # ── Composite score ────────────────────────────────────────────────
        composite: dict[str, float] = {}
        for tkr in self.UNIVERSE:
            score = sum(
                self.FACTOR_WEIGHTS[f] * factor_zscores[f].get(tkr, 0.0)
                for f in self.FACTOR_WEIGHTS
            )
            composite[tkr] = score

        sorted_tickers = sorted(composite, key=composite.__getitem__, reverse=True)
        top_tickers = sorted_tickers[:self.TOP_N]
        bot_tickers = sorted_tickers[-self.BOT_N:]

        if symbol not in self.UNIVERSE:
            return None

        if symbol in top_tickers:
            rank_score = composite[symbol]
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=min(abs(rank_score) / 2.0, 1.0),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "composite_score": round(rank_score, 4),
                    "rank": sorted_tickers.index(symbol) + 1,
                    "factor_scores": {
                        f: round(factor_zscores[f].get(symbol, 0.0), 4)
                        for f in self.FACTOR_WEIGHTS
                    },
                    "long_basket": top_tickers,
                    "short_basket": bot_tickers,
                    "academic_ref": "Fama-French 5-factor + Momentum",
                },
            )
        elif symbol in bot_tickers:
            rank_score = composite[symbol]
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=min(abs(rank_score) / 2.0, 1.0),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "composite_score": round(rank_score, 4),
                    "rank": sorted_tickers.index(symbol) + 1,
                    "factor_scores": {
                        f: round(factor_zscores[f].get(symbol, 0.0), 4)
                        for f in self.FACTOR_WEIGHTS
                    },
                    "long_basket": top_tickers,
                    "short_basket": bot_tickers,
                    "academic_ref": "Fama-French 5-factor + Momentum",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Single-ETF backtest using monthly factor z-scores from price data only.

        Uses rolling 252-day window to compute:
          - momentum z-score (12-1 month return)
          - low-vol z-score (realized vol, inverted)

        Composite signal > 0 = long, < 0 = short.
        Applies .shift(1).fillna(0) to prevent lookahead bias.
        """
        if "close" not in df.columns or len(df) < 252:
            empty = pd.Series(0, index=df.index)
            return BacktestSignals(
                entries=empty.astype(bool),
                exits=empty.astype(bool),
            )

        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))

        # Momentum: 12-1 month (skip last 21 days)
        mom_12 = close.shift(21) / close.shift(252) - 1.0
        mom_z = (mom_12 - mom_12.rolling(252).mean()) / (mom_12.rolling(252).std() + 1e-10)

        # Low-vol: 1-year realized vol (inverted: lower vol = better)
        rv = log_ret.rolling(252).std() * np.sqrt(252)
        rv_z = -((rv - rv.rolling(252).mean()) / (rv.rolling(252).std() + 1e-10))

        composite = (
            self.FACTOR_WEIGHTS["momentum"] * mom_z
            + self.FACTOR_WEIGHTS["low_vol"] * rv_z
        )

        # Top quintile = long, bottom quintile = short
        q_hi = composite.rolling(252).quantile(0.80)
        q_lo = composite.rolling(252).quantile(0.20)

        entries = (composite > q_hi).shift(1).fillna(False)
        exits = (composite < 0).shift(1).fillna(False)
        short_entries = (composite < q_lo).shift(1).fillna(False)
        short_exits = (composite > 0).shift(1).fillna(False)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
