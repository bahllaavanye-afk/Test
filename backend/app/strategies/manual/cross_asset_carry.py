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
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"

# Item 10: FRED free API endpoint for 10Y-2Y yield curve spread
# No API key required — the CSV endpoint is completely public
_FRED_T10Y2Y = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"

# yfinance tickers for commodity front/second contract spread proxies
_COMMODITY_TICKERS = {
    "oil": ("CL=F", "CLH26.NYM"),   # WTI front vs deferred (proxy)
    "gold": ("GC=F", "GCM26.CMX"),  # Gold front vs deferred
}


# ETF universe — each leg and its role in the carry signal
HIGH_EQUITY_CARRY = ["SCHD", "VYM"]
LOW_EQUITY_CARRY  = ["ARKK", "SPAK"]
HIGH_BOND_CARRY   = ["TLT"]
LOW_BOND_CARRY    = ["SHY"]
ALL_ETF_UNIVERSE  = HIGH_EQUITY_CARRY + LOW_EQUITY_CARRY + HIGH_BOND_CARRY + LOW_BOND_CARRY


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
    BOND_CARRY_WEIGHT   = 0.50

    # Entry/exit thresholds (z-score of combined carry signal)
    ENTRY_THRESHOLD =  0.50
    EXIT_THRESHOLD  =  0.10
    STOP_THRESHOLD  = -1.50  # stop out if carry signal dramatically reverses

    # Lookback for trailing return
    LOOKBACK_DAYS = 252   # ~12 months

    # Item 10: weights for the extended composite signal
    RATES_CARRY_WEIGHT     = 0.20
    COMMODITY_CARRY_WEIGHT = 0.10
    EQUITY_CARRY_WEIGHT    = 0.40
    BOND_CARRY_WEIGHT      = 0.30

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

    async def _fetch_12m_return(self, symbol: str) -> float | None:
        """Fetch daily bars and compute trailing 12-month total return."""
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
                    headers=self._headers(),
                )
            if resp.status_code != 200:
                return None
            bars = resp.json().get("bars", [])
            if len(bars) < 200:
                return None
            closes = [float(b["c"]) for b in bars]
            # 12-month return: most recent close vs close ~252 bars ago
            return float(closes[-1] / closes[-min(self.LOOKBACK_DAYS, len(closes))] - 1.0)
        except Exception:
            return None

    @staticmethod
    def _zscore(value: float, series_vals: list[float]) -> float:
        """Compute z-score of value relative to a reference distribution."""
        if not series_vals or len(series_vals) < 2:
            return 0.0
        mean = np.mean(series_vals)
        std  = np.std(series_vals)
        return float((value - mean) / max(std, 1e-8))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Compute carry signal across equity and bond ETFs.
        Signal is issued for the 'carry_basket' (treated as a synthetic symbol).
        The actual executor will decompose into individual ETF legs.
        """
        import asyncio

        # Fetch 12-month returns for all ETFs concurrently
        returns = await asyncio.gather(
            *[self._fetch_12m_return(sym) for sym in ALL_ETF_UNIVERSE],
            return_exceptions=True,
        )
        ret_map: dict[str, float] = {}
        for sym, ret in zip(ALL_ETF_UNIVERSE, returns):
            if isinstance(ret, float) and ret is not None:
                ret_map[sym] = ret

        # Need at least one from each leg
        if not all(s in ret_map for s in ["SCHD", "ARKK", "TLT", "SHY"]):
            return None

        # Equity carry spread: avg(high-carry ETFs) - avg(low-carry ETFs)
        high_eq_ret = np.mean([ret_map[s] for s in HIGH_EQUITY_CARRY if s in ret_map])
        low_eq_ret  = np.mean([ret_map[s] for s in LOW_EQUITY_CARRY  if s in ret_map])
        equity_carry_raw = float(high_eq_ret - low_eq_ret)

        # Bond carry spread: TLT - SHY (duration premium)
        bond_carry_raw = float(ret_map["TLT"] - ret_map["SHY"])

        # Normalize each signal to [-1, +1] range
        # We use tanh normalization: tanh maps any real to (-1, 1) smoothly
        equity_carry_norm = float(np.tanh(equity_carry_raw * 5.0))
        bond_carry_norm   = float(np.tanh(bond_carry_raw   * 5.0))

        # Combined carry signal
        combined = (
            self.EQUITY_CARRY_WEIGHT * equity_carry_norm +
            self.BOND_CARRY_WEIGHT   * bond_carry_norm
        )

        if abs(combined) < self.ENTRY_THRESHOLD:
            return None  # no actionable carry signal

        side       = "buy" if combined > 0 else "sell"
        confidence = min(abs(combined), 1.0)

        # Determine which ETF to trade (signal issued for the triggering symbol)
        # Prefer to trade the most liquid ETF in the appropriate leg
        if side == "buy":
            trade_symbol = "SCHD"   # long high-carry equity
        else:
            trade_symbol = "ARKK"   # short low-carry equity (or TLT short if bond-driven)

        # Override with provided symbol if it's in the universe
        if symbol in ALL_ETF_UNIVERSE:
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
                "bond_carry_spread":   round(bond_carry_raw,   4),
                "equity_carry_norm":   round(equity_carry_norm, 4),
                "bond_carry_norm":     round(bond_carry_norm,   4),
                "combined_carry":      round(combined, 4),
                "schd_12m":  round(ret_map.get("SCHD", 0), 4),
                "arkk_12m":  round(ret_map.get("ARKK", 0), 4),
                "tlt_12m":   round(ret_map.get("TLT",  0), 4),
                "shy_12m":   round(ret_map.get("SHY",  0), 4),
                "academic_ref": "Koijen et al. (2018) JFE Carry",
                "portfolio_weights": {
                    "equity_carry": self.EQUITY_CARRY_WEIGHT,
                    "bond_carry":   self.BOND_CARRY_WEIGHT,
                },
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Single-ETF backtest: use trailing 12-month return momentum as carry proxy.
        Long when trailing return is positive (ETF in carry regime), else flat.

        For multi-asset backtesting the full cross-sectional version is needed;
        this implementation provides the single-symbol backbone.
        """
        if "close" not in df.columns or len(df) < self.LOOKBACK_DAYS:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"].astype(float)

        # Rolling 12-month return as carry signal
        carry_signal = close / close.shift(self.LOOKBACK_DAYS) - 1.0

        # Entry: positive carry (long when carry > 0)
        # Apply shift(1) to prevent lookahead
        entries = (carry_signal > 0.0).shift(1).fillna(False)
        exits   = (carry_signal < 0.0).shift(1).fillna(False)

        # Short leg: negative carry
        short_entries = (carry_signal < -0.05).shift(1).fillna(False)
        short_exits   = (carry_signal > -0.01).shift(1).fillna(False)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
