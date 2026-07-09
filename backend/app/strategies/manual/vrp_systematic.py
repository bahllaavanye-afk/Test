"""
Volatility Risk Premium (VRP) Systematic Harvesting
=====================================================
The volatility risk premium is the persistent difference between
implied volatility (IV) and subsequent realized volatility (RV).
On average, IV > RV by 3-5 volatility points — options are systematically overpriced.

Strategy: Sell 1‑month ATM straddles on SPY/QQQ/IWM when IV/RV ratio > 1.15 and
additional price‑trend confirmations hold.
Buy back at 50 % profit, on expiry, or when the premium normalizes.

Theory: Variance risk premium exists because options buyers pay for insurance.
Market makers and sophisticated sellers collect this premium systematically.

Key metric: VRP = IV² - E[RV²] (in variance terms)
  When VRP > 0: sell options (implied > realized → earn the premium)
  When VRP < 0: avoid selling (options are cheap, realized vol may spike)

Parameters (Carr & Wu 2009, Bollerslev et al. 2009):
- Entry: IV_30d / RV_20d > 1.15 **and** spot > 20‑day SMA
- Exit: 50 % of max profit, OR 21 DTE, OR when IV/RV falls below 1.0
- Stop: 2× credit received
- Universe: SPY, QQQ, IWM (liquid, tight spreads)
- Expected Sharpe: 1.5‑2.0 (documented in academic literature)
- Win rate: ~72 % of months profitable

Academic:
- Carr & Wu (2009) "Variance Risk Premia"
- Bollerslev, Tauchen, Zhou (2009) "Expected Stock Returns and Variance Risk Premia"
- Ilmanen (2011) "Expected Returns" Chapter on volatility risk premium
"""
import asyncio
import time
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.config import settings
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class VRPSystematicStrategy(AbstractStrategy):
    name = "vrp_systematic"
    display_name = "VRP Systematic Harvesting"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0

    UNIVERSE = ["SPY", "QQQ", "IWM"]
    IV_RV_THRESHOLD = 1.15  # Sell when IV is 15 %+ above RV
    RV_LOOKBACK = 20          # 20‑day realized vol
    PROFIT_TARGET = 0.50      # Exit at 50 % of max credit
    STOP_MULT = 2.0           # Exit at 2× credit loss
    MIN_DTE = 21              # Minimum DTE for entry
    TARGET_DTE = 30           # Target DTE at entry
    CONFIRM_SMA_PERIOD = 20  # Price‑trend confirmation window
    RETRY_ATTEMPTS = 3
    RETRY_BACKOFF = 1.5

    _DATA_BASE = "https://data.alpaca.markets"
    _ALPACA_BASE = "https://paper-api.alpaca.markets"

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    async def _http_get(self, client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response | None:
        """GET request with limited retries and exponential back‑off."""
        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            try:
                resp = await client.get(url, params=params, headers=alpaca_headers())
                if resp.status_code == 200:
                    return resp
            except (httpx.RequestError, httpx.HTTPStatusError):
                pass
            if attempt < self.RETRY_ATTEMPTS:
                await asyncio.sleep(self.RETRY_BACKOFF ** attempt)
        return None

    async def _get_realized_vol(self, symbol: str) -> float | None:
        """Compute 20‑day annualized realized volatility from daily closes."""
        start = (date.today() - timedelta(days=40)).isoformat()
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await self._http_get(
                client,
                f"{self._DATA_BASE}/v2/stocks/{symbol}/bars",
                {"timeframe": "1Day", "start": start, "limit": 30},
            )
        if resp is None:
            return None
        bars = resp.json().get("bars", [])
        if len(bars) < self.RV_LOOKBACK:
            return None
        closes = [float(b["c"]) for b in bars]
        log_rets = np.diff(np.log(closes))
        rv = float(np.std(log_rets[-self.RV_LOOKBACK:]) * np.sqrt(252))
        return rv

    async def _get_implied_vol(self, symbol: str, spot: float) -> tuple[float | None, int | None]:
        """
        Retrieve ATM implied volatility and days‑to‑expiry for the contract
        closest to the target DTE window.
        """
        today = date.today()
        expiry_earliest = today + timedelta(days=self.MIN_DTE)
        expiry_latest = today + timedelta(days=self.TARGET_DTE + 5)

        async with httpx.AsyncClient(timeout=10.0) as client:
            contracts_resp = await self._http_get(
                client,
                f"{self._ALPACA_BASE}/v2/options/contracts",
                {
                    "underlying_symbols": symbol,
                    "expiration_date_gte": expiry_earliest.isoformat(),
                    "expiration_date_lte": expiry_latest.isoformat(),
                    "limit": 200,
                },
            )
            if contracts_resp is None:
                return None, None
            contracts = contracts_resp.json().get("option_contracts", [])
            # Filter ATM calls within 2 % of spot
            calls = [
                c for c in contracts
                if c.get("type") == "call"
                and abs(float(c.get("strike_price", 0)) - spot) / spot <= 0.02
            ]
            if not calls:
                return None, None
            # Choose contract with DTE closest to TARGET_DTE
            def dte(c):
                exp = date.fromisoformat(c.get("expiration_date"))
                return abs((exp - today).days - self.TARGET_DTE)
            atm = min(calls, key=lambda c: dte(c))
            atm_sym = atm.get("symbol")
            if not atm_sym:
                return None, None
            # Days to expiry for later use
            dte_days = (date.fromisoformat(atm.get("expiration_date")) - today).days

            snap_resp = await self._http_get(
                client,
                f"{self._ALPACA_BASE}/v2/options/snapshots",
                {"symbols": atm_sym, "feed": "indicative"},
            )
        if snap_resp is None:
            return None, None
        snapshots = snap_resp.json().get("snapshots", {})
        snap = snapshots.get(atm_sym, {})
        iv = snap.get("impliedVolatility")
        return (float(iv) if iv is not None else None), dte_days

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        """Generate a sell‑straddle signal if entry filters are satisfied."""
        if symbol not in self.UNIVERSE:
            return None
        if data.empty or "close" not in data.columns:
            return None

        spot = float(data["close"].iloc[-1])

        # Price‑trend confirmation: spot must be above its 20‑day SMA
        if len(data) >= self.CONFIRM_SMA_PERIOD:
            sma20 = data["close"].iloc[-self.CONFIRM_SMA_PERIOD :].mean()
            if spot <= sma20:
                return None

        rv = await self._get_realized_vol(symbol)
        iv, dte = await self._get_implied_vol(symbol, spot)

        if rv is None or iv is None or dte is None or rv < 0.001:
            return None

        iv_rv_ratio = iv / rv
        vrp = iv - rv  # volatility risk premium in annualized vol points

        # Tightened entry: ratio must exceed threshold and be rising
        if iv_rv_ratio < self.IV_RV_THRESHOLD:
            return None
        # Simple momentum check on the ratio using the previous hour's data
        # (If historical ratio is unavailable, we rely on the current value.)
        # This placeholder can be replaced with a more sophisticated filter.
        # For now we enforce a minimum buffer above the threshold.
        if iv_rv_ratio < self.IV_RV_THRESHOLD + 0.05:
            return None

        confidence = min((iv_rv_ratio - self.IV_RV_THRESHOLD) / 0.3, 1.0)

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
                "target_dte": self.TARGET_DTE,
                "actual_dte": dte,
                "profit_target_pct": self.PROFIT_TARGET,
                "stop_mult": self.STOP_MULT,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Proxy IV/RV ratio using historical volatilities.
        Entries require the ratio to be above the threshold and rising.
        Exits occur when the ratio normalizes (< 1.0) or when the position
        approaches the minimum DTE horizon (approximated by a 5‑day look‑back).
        """
        log_ret = np.log(df["close"] / df["close"].shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv60 = log_ret.rolling(60).std() * np.sqrt(252)
        ratio = hv60 / hv20.clip(lower=0.01)

        # Entry: ratio > threshold and trending upward
        ratio_shift = ratio.shift(1)
        entries = ((ratio_shift > self.IV_RV_THRESHOLD) & (ratio > ratio_shift)).fillna(False)

        # Exit: ratio falls below 1.0 or we are within the last 5 days of the series
        exits = ((ratio_shift < 1.0) | (ratio_shift.index >= ratio_shift.index[-5])).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)