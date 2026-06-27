"""
Volatility Risk Premium (VRP) Systematic Harvesting
=====================================================
The volatility risk premium is the persistent difference between
implied volatility (IV) and subsequent realized volatility (RV).
On average, IV > RV by 3-5 volatility points — options are systematically overpriced.

Strategy: Sell 1-month ATM straddles on SPY/QQQ when IV/RV ratio > 1.15.
Buy back at 50% profit or at expiry. Roll monthly.

Theory: Variance risk premium exists because options buyers pay for insurance.
Market makers and sophisticated sellers collect this premium systematically.

Key metric: VRP = IV² - E[RV²] (in variance terms)
  When VRP > 0: sell options (implied > realized → earn the premium)
  When VRP < 0: avoid selling (options are cheap, realized vol may spike)

Parameters (Carr & Wu 2009, Bollerslev et al. 2009):
- Entry: IV_30d / RV_20d > 1.15 (options pricing in 15%+ more vol than realized)
- Exit: 50% of max profit, OR 21 DTE
- Stop: 2× credit received
- Universe: SPY, QQQ, IWM (liquid, tight spreads)
- Expected Sharpe: 1.5-2.0 (documented in academic literature)
- Win rate: ~72% of months profitable

Academic:
- Carr & Wu (2009) "Variance Risk Premia"
- Bollerslev, Tauchen, Zhou (2009) "Expected Stock Returns and Variance Risk Premia"
- Ilmanen (2011) "Expected Returns" Chapter on volatility risk premium
"""

from datetime import date, timedelta
from typing import List, Optional

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class VRPSystematicParams(BaseModel):
    """Configuration parameters for the VRP systematic strategy."""

    iv_rv_threshold: float = Field(
        default=1.15,
        gt=0,
        description="IV/RV ratio threshold that triggers a sell entry.",
        example=1.15,
    )
    rv_lookback: int = Field(
        default=20,
        gt=0,
        description="Number of days to look back when computing realized volatility.",
        example=20,
    )
    profit_target: float = Field(
        default=0.50,
        gt=0,
        le=1,
        description="Target profit as a fraction of the initial credit received.",
        example=0.5,
    )
    stop_mult: float = Field(
        default=2.0,
        gt=0,
        description="Multiplier of the initial credit at which a stop loss is triggered.",
        example=2.0,
    )
    min_dte: int = Field(
        default=21,
        gt=0,
        description="Minimum days to expiry required for a new entry.",
        example=21,
    )
    target_dte: int = Field(
        default=30,
        gt=0,
        description="Target days to expiry when entering a position.",
        example=30,
    )
    universe: List[str] = Field(
        default_factory=lambda: ["SPY", "QQQ", "IWM"],
        description="List of tradable symbols that the strategy may operate on.",
        example=["SPY", "QQQ", "IWM"],
    )

    @validator("universe")
    def non_empty_universe(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("universe must contain at least one symbol")
        return v


class VRPSystematicStrategy(AbstractStrategy):
    name = "vrp_systematic"
    display_name = "VRP Systematic Harvesting"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0

    _DATA_BASE = "https://data.alpaca.markets"
    _ALPACA_BASE = "https://paper-api.alpaca.markets"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        # Resolve configuration using the Pydantic schema; fallback to defaults.
        self.cfg = VRPSystematicParams(**(params or {}))

    async def _get_realized_vol(self, symbol: str) -> Optional[float]:
        """Compute realized volatility over the configured lookback period."""
        start = (date.today() - timedelta(days=40)).isoformat()
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{self._DATA_BASE}/v2/stocks/{symbol}/bars",
                params={"timeframe": "1Day", "start": start, "limit": 30},
                headers=alpaca_headers(),
            )
        if resp.status_code != 200:
            return None
        bars = resp.json().get("bars", [])
        if len(bars) < self.cfg.rv_lookback:
            return None
        closes = [float(b["c"]) for b in bars]
        log_rets = np.diff(np.log(closes))
        rv = float(np.std(log_rets[-self.cfg.rv_lookback :]) * np.sqrt(252))
        return rv

    async def _get_implied_vol(self, symbol: str, spot: float) -> Optional[float]:
        """Retrieve the ATM implied volatility for the given symbol."""
        today = date.today().isoformat()
        async with httpx.AsyncClient(timeout=10.0) as client:
            contracts_resp = await client.get(
                f"{self._ALPACA_BASE}/v2/options/contracts",
                params={
                    "underlying_symbols": symbol,
                    "expiration_date_gte": today,
                    "expiration_date_lte": (date.today() + timedelta(days=45)).isoformat(),
                    "limit": 100,
                },
                headers=alpaca_headers(),
            )
            if contracts_resp.status_code != 200:
                return None
            contracts = contracts_resp.json().get("option_contracts", [])
            calls = [c for c in contracts if c.get("type") == "call"]
            if not calls:
                return None
            atm = min(calls, key=lambda c: abs(float(c.get("strike_price", 0)) - spot))
            atm_sym = atm.get("symbol")
            if not atm_sym:
                return None
            snap_resp = await client.get(
                f"{self._ALPACA_BASE}/v2/options/snapshots",
                params={"symbols": atm_sym, "feed": "indicative"},
                headers=alpaca_headers(),
            )
        if snap_resp.status_code != 200:
            return None
        snapshots = snap_resp.json().get("snapshots", {})
        snap = snapshots.get(atm_sym, {})
        iv = snap.get("impliedVolatility")
        return float(iv) if iv is not None else None

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Optional[Signal]:
        if symbol not in self.cfg.universe:
            return None
        if data.empty or "close" not in data.columns:
            return None
        spot = float(data["close"].iloc[-1])

        rv = await self._get_realized_vol(symbol)
        iv = await self._get_implied_vol(symbol, spot)

        if rv is None or iv is None or rv < 0.001:
            return None

        iv_rv_ratio = iv / rv
        vrp = iv - rv  # volatility risk premium in annualized vol points

        if iv_rv_ratio < self.cfg.iv_rv_threshold:
            return None  # Options not rich enough

        confidence = min((iv_rv_ratio - self.cfg.iv_rv_threshold) / 0.3, 1.0)

        return Signal(
            symbol=symbol,
            side="sell",  # Sell the straddle
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "vrp_systematic",
                "implied_vol": round(iv, 4),
                "realized_vol": round(rv, 4),
                "iv_rv_ratio": round(iv_rv_ratio, 3),
                "vrp": round(vrp, 4),
                "order_type": "straddle",
                "target_dte": self.cfg.target_dte,
                "profit_target_pct": self.cfg.profit_target,
                "stop_mult": self.cfg.stop_mult,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Proxy: IV/RV ratio using HV20 vs HV60 (HV60 as IV proxy in absence of option data)."""
        log_ret = np.log(df["close"] / df["close"].shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv60 = log_ret.rolling(60).std() * np.sqrt(252)
        ratio = hv60 / hv20.clip(lower=0.01)

        entries = (ratio.shift(1) > self.cfg.iv_rv_threshold).fillna(False)
        exits = (ratio.shift(1) < 1.0).fillna(False)  # buy back when premium normalizes

        return BacktestSignals(entries=entries, exits=exits)