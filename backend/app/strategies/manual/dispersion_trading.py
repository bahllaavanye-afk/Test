"""
Dispersion Trading
==================
Trade the spread between index implied volatility and component stock volatility.

Core insight: Index IV is usually HIGHER than the weighted average of component IVs
because investors buy index options for portfolio protection. This "correlation premium"
can be harvested by:
  → Short index variance (sell SPY/QQQ straddles)
  → Long single-stock variance (buy straddles on top components)

When to enter: implied correlation > realized correlation (correlation premium is rich)
When to exit: spread reverts to historical mean

Implied correlation = (σ²_index - Σ w²ᵢσ²ᵢ) / (2 Σᵢ<ⱼ wᵢwⱼσᵢσⱼ)
where σ_index = SPY IV, σᵢ = component IVs, wᵢ = weights

Simplified 5-stock implementation:
  Index = QQQ (tech-heavy)
  Components: AAPL (13%), MSFT (12%), NVDA (8%), AMZN (7%), META (5%)
  Short: QQQ straddle
  Long: AAPL + MSFT straddles (proportional to weights)

Documented:
- Credit Suisse Dispersion Trading (2006) — average Sharpe 1.4
- O'Brien & Srivastava (2013) — 2-3% monthly premium
- Best entered after earnings season (correlation typically drops)

Entry trigger: implied_corr / realized_corr > 1.20 (correlation 20% rich)
"""
import asyncio
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class DispersionTradingStrategy(AbstractStrategy):
    name = "dispersion_trading"
    display_name = "Dispersion Trading"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"

    # QQQ component weights (approximate)
    INDEX = "QQQ"
    COMPONENTS = {
        "AAPL": 0.13, "MSFT": 0.12, "NVDA": 0.08,
        "AMZN": 0.07, "META": 0.05,
    }
    LOOKBACK = 30  # days for realized correlation
    MIN_CORR_PREMIUM = 0.20  # Enter when implied corr 20%+ above realized

    _DATA_BASE = "https://data.alpaca.markets"
    _ALPACA_BASE = "https://paper-api.alpaca.markets"

    async def _fetch_hv(self, symbol: str, days: int = 30) -> float | None:
        start = (date.today() - timedelta(days=days + 10)).isoformat()
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{self._DATA_BASE}/v2/stocks/{symbol}/bars",
                params={"timeframe": "1Day", "start": start, "limit": days + 5},
                headers=alpaca_headers(),
            )
        if resp.status_code != 200:
            return None
        bars = resp.json().get("bars", [])
        if len(bars) < days:
            return None
        closes = [float(b["c"]) for b in bars[-days:]]
        log_rets = np.diff(np.log(closes))
        return float(np.std(log_rets) * np.sqrt(252))

    async def _get_atm_iv(self, symbol: str) -> float | None:
        """Get ATM implied vol for ~30 DTE options."""
        today = date.today()
        exp_min = (today + timedelta(days=21)).isoformat()
        exp_max = (today + timedelta(days=45)).isoformat()
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get current price
            quote_resp = await client.get(
                f"{self._DATA_BASE}/v2/stocks/{symbol}/bars",
                params={"timeframe": "1Day", "limit": 1},
                headers=alpaca_headers(),
            )
            if quote_resp.status_code != 200:
                return None
            bars = quote_resp.json().get("bars", [])
            if not bars:
                return None
            spot = float(bars[-1]["c"])

            # Get ATM options
            contracts_resp = await client.get(
                f"{self._ALPACA_BASE}/v2/options/contracts",
                params={
                    "underlying_symbols": symbol,
                    "expiration_date_gte": exp_min,
                    "expiration_date_lte": exp_max,
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
        snap = snap_resp.json().get("snapshots", {}).get(atm_sym, {})
        iv = snap.get("impliedVolatility")
        return float(iv) if iv is not None else None

    async def _compute_realized_correlation(self) -> float | None:
        """Compute average pairwise realized correlation of top 5 components."""
        syms = list(self.COMPONENTS.keys())
        tasks = [self._fetch_daily_returns(s) for s in syms]
        all_returns = await asyncio.gather(*tasks, return_exceptions=True)

        valid = [(s, r) for s, r in zip(syms, all_returns)
                 if not isinstance(r, Exception) and r is not None and len(r) > 10]
        if len(valid) < 2:
            return None

        # Build return matrix
        min_len = min(len(r) for _, r in valid)
        matrix = np.column_stack([r[-min_len:] for _, r in valid])
        corr_matrix = np.corrcoef(matrix.T)
        n = corr_matrix.shape[0]
        # Average pairwise correlation (off-diagonal elements)
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        return float(corr_matrix[mask].mean())

    async def _fetch_daily_returns(self, symbol: str) -> np.ndarray | None:
        start = (date.today() - timedelta(days=45)).isoformat()
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{self._DATA_BASE}/v2/stocks/{symbol}/bars",
                params={"timeframe": "1Day", "start": start, "limit": 40},
                headers=alpaca_headers(),
            )
        if resp.status_code != 200:
            return None
        bars = resp.json().get("bars", [])
        if len(bars) < 10:
            return None
        closes = np.array([float(b["c"]) for b in bars])
        return np.diff(np.log(closes))

    async def analyze(self, data: pd.DataFrame, symbol: str = "QQQ") -> Signal | None:
        # Fetch IVs concurrently
        syms = [self.INDEX] + list(self.COMPONENTS.keys())[:2]
        iv_tasks = [self._get_atm_iv(s) for s in syms]
        hv_tasks = [self._fetch_hv(s) for s in syms]

        ivs_raw, hvs_raw = await asyncio.gather(
            asyncio.gather(*iv_tasks, return_exceptions=True),
            asyncio.gather(*hv_tasks, return_exceptions=True),
        )

        index_iv = ivs_raw[0] if not isinstance(ivs_raw[0], Exception) and ivs_raw[0] else None
        if index_iv is None:
            return None

        comp_ivs = [iv for iv in ivs_raw[1:] if not isinstance(iv, Exception) and iv]
        if not comp_ivs:
            return None

        # Implied correlation approximation:
        # implied_corr ≈ (index_iv² - weighted_avg_comp_iv²) / cross_terms
        weights = list(self.COMPONENTS.values())[:len(comp_ivs)]
        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]

        weighted_var_sum = sum(w * iv**2 for w, iv in zip(norm_w, comp_ivs))
        if index_iv**2 <= weighted_var_sum:
            implied_corr = 0.0
        else:
            cross_term_approx = 2 * sum(norm_w) ** 2 * np.mean(comp_ivs) ** 2
            implied_corr = min((index_iv**2 - weighted_var_sum) / max(cross_term_approx, 0.001), 1.0)

        realized_corr = await self._compute_realized_correlation()
        if realized_corr is None or realized_corr <= 0:
            return None

        corr_ratio = implied_corr / max(realized_corr, 0.01)

        if corr_ratio < (1 + self.MIN_CORR_PREMIUM):
            return None  # Correlation premium not rich enough

        confidence = min((corr_ratio - 1) / 0.5, 1.0)
        return Signal(
            symbol=self.INDEX,
            side="sell",  # Sell index straddle, buy component straddles
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "dispersion_trading",
                "index_iv": round(index_iv, 4),
                "avg_component_iv": round(float(np.mean(comp_ivs)), 4),
                "implied_correlation": round(implied_corr, 4),
                "realized_correlation": round(realized_corr, 4),
                "correlation_premium_pct": round((corr_ratio - 1) * 100, 1),
                "trade": f"Short {self.INDEX} straddle + Long AAPL/MSFT straddles",
                "components_to_long": list(self.COMPONENTS.keys())[:2],
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        # Proxy: high short-term vs long-term vol ratio suggests dispersion opportunity
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
        vol_60 = log_ret.rolling(60).std() * np.sqrt(252)
        ratio = vol_20 / vol_60.clip(lower=0.01)
        # High ratio → dispersion opportunity → short index (sell signal)
        short_entry = (ratio.shift(1) > 1.2).fillna(False)
        short_exit = (ratio.shift(1) <= 1.0).fillna(False)
        return BacktestSignals(
            entries=pd.Series(False, index=df.index),
            exits=pd.Series(False, index=df.index),
            short_entries=short_entry,
            short_exits=short_exit,
        )
