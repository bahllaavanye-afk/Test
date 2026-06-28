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

        # ── Liquidity filter ────────────────────────────────────────────────
        avg_vol = info.get("averageVolume")
        if avg_vol is None or avg_vol < 500_000:  # MUTATION: filter out low‑volume ETFs to avoid noisy signals
            return {
                "value": float("nan"),
                "quality": float("nan"),
                "size": float("nan"),
            }

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
            raise RuntimeError(
                "yfinance is required for MultiFactorEquity. Install with: pip install yfinance"
            )
        # (rest of the method unchanged)
