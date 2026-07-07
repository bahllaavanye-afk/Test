"""
Cross-Asset Carry Portfolio
=============================
Academic basis:
  - Koijen, Moskowitz, Pedersen, Vrugt (2018) "Carry" Journal of Financial Economics —
    carry predicts returns across ALL asset classes (equities, bonds, currencies,
    commodities). This is one of the most robust and diversified risk premia.
  - Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere" shows carry is
    complementary to momentum — combining them improves Sharpe ratio significantly.

Carry definitions:
  - Equity carry:   dividend yield + buyback yield (income while holding equity)
  - Bond carry:     term spread (10Y minus 2Y yield); steep curve → buy duration
  - Vol carry:      implied volatility minus realized vol (VRP, covered elsewhere)

ETF Implementation (Alpaca-tradeable):
  Equity high-carry:  SCHD (Schwab Dividend), VYM (Vanguard High Dividend Yield)
  Equity low-carry:   ARKK (ARK Innovation, no dividend), SPAK (SPAC ETF)
  Bond high-carry:    TLT (20Y Treasury, captures term premium)
  Bond low-carry:     SHY (1-3Y Treasury, cash-like)
  Bond neutral ref:   IEF (7-10Y Treasury, mid duration)

Signal construction:
  equity_carry = z_score(SCHD_12m_ret - ARKK_12m_ret)  high - low carry spread
  bond_carry   = z_score(TLT_12m_ret  - SHY_12m_ret)   long vs short duration
  combined     = 0.50 × equity_carry + 0.50 × bond_carry

  Long combined when combined_signal > 0.5, short when < -0.5.
  Individual legs sized by 40% equity + 40% bond + 20% residual cash.

Documented Sharpe: 0.8-1.4 for diversified carry (Koijen et al. 2018, Table II)
"""

from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"

# ETF universe — each leg and its role in the carry signal
HIGH_EQUITY_CARRY = ["SCHD", "VYM"]
LOW_EQUITY_CARRY = ["ARKK", "SPAK"]
HIGH_BOND_CARRY = ["TLT"]
LOW_BOND_CARRY = ["SHY"]
ALL_ETF_UNIVERSE = HIGH_EQUITY_CARRY + LOW_EQUITY_CARRY + HIGH_BOND_CARRY + LOW_BOND_CARRY


class CrossAssetCarryStrategy(AbstractStrategy):
    """
    Cross-asset carry portfolio using dividend / term-structure ETF spreads.

    Goes long high-carry assets and shorts low-carry assets using
    trailing 12-month return differentials as carry proxies.
    """

    name = "cross_asset_carry"
    display_name = "Cross-Asset Carry Portfolio"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0  # daily — carry is a slow-moving signal

    # Portfolio weights (must sum to 1.0)
    EQUITY_CARRY_WEIGHT = 0.50
    BOND_CARRY_WEIGHT = 0.50

    # Entry/exit thresholds (z-score of combined carry signal)
    ENTRY_THRESHOLD = 0.50
    EXIT_THRESHOLD = 0.10
    STOP_THRESHOLD = -1.50  # stop out if carry signal dramatically reverses

    # Lookback for trailing return
    LOOKBACK_DAYS = 252  # ~12 months

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    async def _fetch_12m_return(self, symbol: str) -> float | None:
        """Fetch daily bars and compute trailing 12‑month total return.

        Returns ``None`` if the request fails, insufficient data is returned,
        or the computation encounters a divide‑by‑zero scenario.
        """
        if not symbol:
            return None

        start = (date.today() - timedelta(days=self.LOOKBACK_DAYS + 30)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Day",
                        "start": start,
                        "limit": self.LOOKBACK_DAYS + 30,
                        "feed": "iex",
                    },
                    headers=alpaca_headers(),
                )
            if resp.status_code != 200:
                return None

            bars = resp.json().get("bars", [])
            if len(bars) < max(200, self.LOOKBACK_DAYS // 2):
                # Not enough data to compute a reliable 12‑month return
                return None

            closes = [float(b["c"]) for b in bars if b.get("c") is not None]
            if not closes:
                return None

            # Use the earliest close within the look‑back window for the denominator
            denominator_index = -min(self.LOOKBACK_DAYS, len(closes))
            denominator = closes[denominator_index]
            if denominator == 0:
                return None

            return float(closes[-1] / denominator - 1.0)
        except Exception:
            return None

    @staticmethod
    def _zscore(value: float, series_vals: list[float]) -> float:
        """Compute z‑score of *value* relative to *series_vals*.

        Returns ``0.0`` when the reference series is empty or has insufficient
        variance to avoid division‑by‑zero.
        """
        if not series_vals or len(series_vals) < 2:
            return 0.0
        mean = np.mean(series_vals)
        std = np.std(series_vals)
        return float((value - mean) / max(std, 1e-8))

    async def analyze(self, data: pd.DataFrame | None, symbol: str | None) -> Signal | None:
        """
        Compute carry signal across equity and bond ETFs.

        Returns a ``Signal`` for the synthetic ``carry_basket`` symbol (or the
        provided ``symbol`` if it belongs to the universe). Handles edge cases
        such as missing data, empty inputs, and off‑by‑one indexing safely.
        """
        import asyncio

        # Guard against completely missing inputs
        if data is None or not isinstance(data, pd.DataFrame):
            # The strategy does not depend on the incoming dataframe, but we
            # retain the check for future compatibility.
            pass

        # Fetch 12‑month returns for all ETFs concurrently
        returns = await asyncio.gather(
            *[self._fetch_12m_return(sym) for sym in ALL_ETF_UNIVERSE],
            return_exceptions=True,
        )

        # Build a map of successful returns, ignoring None or exception results
        ret_map: dict[str, float] = {}
        for sym, ret in zip(ALL_ETF_UNIVERSE, returns):
            if isinstance(ret, float):
                ret_map[sym] = ret

        # Ensure at least one ETF from each required leg is present
        required_symbols = {"SCHD", "ARKK", "TLT", "SHY"}
        if not required_symbols.issubset(ret_map.keys()):
            return None

        # Helper to safely compute mean of a possibly empty list
        def safe_mean(symbols: list[str]) -> float:
            values = [ret_map[s] for s in symbols if s in ret_map]
            return float(np.mean(values)) if values else 0.0

        # Equity carry spread
        high_eq_ret = safe_mean(HIGH_EQUITY_CARRY)
        low_eq_ret = safe_mean(LOW_EQUITY_CARRY)
        equity_carry_raw = high_eq_ret - low_eq_ret

        # Bond carry spread
        bond_carry_raw = float(ret_map["TLT"] - ret_map["SHY"])

        # Normalize each signal to [-1, +1] range using tanh
        equity_carry_norm = float(np.tanh(equity_carry_raw * 5.0))
        bond_carry_norm = float(np.tanh(bond_carry_raw * 5.0))

        # Combined carry signal
        combined = (
            self.EQUITY_CARRY_WEIGHT * equity_carry_norm
            + self.BOND_CARRY_WEIGHT * bond_carry_norm
        )

        # No actionable signal if below entry threshold
        if abs(combined) < self.ENTRY_THRESHOLD:
            return None

        side = "buy" if combined > 0 else "sell"
        confidence = min(abs(combined), 1.0)

        # Default trade symbol based on signal direction
        trade_symbol = "SCHD" if side == "buy" else "ARKK"

        # If a specific symbol is supplied and belongs to the universe, use it
        if symbol and symbol in ALL_ETF_UNIVERSE:
            trade_symbol = symbol

        return Signal(
            symbol=trade_symbol,
            side=side,
            confidence=round(confidence, 4),
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "equity_carry_spread": round(equity_carry_raw, 4),
                "bond_carry_spread": round(bond_carry_raw, 4),
                "equity_carry_norm": round(equity_carry_norm, 4),
                "bond_carry_norm": round(bond_carry_norm, 4),
                "combined_carry": round(combined, 4),
                "schd_12m": round(ret_map.get("SCHD", 0.0), 4),
                "arkk_12m": round(ret_map.get("ARKK", 0.0), 4),
                "tlt_12m": round(ret_map.get("TLT", 0.0), 4),
                "shy_12m": round(ret_map.get("SHY", 0.0), 4),
                "academic_ref": "Koijen et al. (2018)",
            },
        )