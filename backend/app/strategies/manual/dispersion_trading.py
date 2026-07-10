import logging
import asyncio
from datetime import date, timedelta
from time import perf_counter
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.config import settings
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

logger = logging.getLogger(__name__)


class DispersionTradingStrategy(AbstractStrategy):
    name = "dispersion_trading"
    display_name = "Dispersion Trading"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"

    # QQQ component weights (approximate)
    INDEX = "QQQ"
    COMPONENTS = {
        "AAPL": 0.13,
        "MSFT": 0.12,
        "NVDA": 0.08,
        "AMZN": 0.07,
        "META": 0.05,
    }
    LOOKBACK = 30  # days for realized correlation
    MIN_CORR_PREMIUM = 0.20  # Enter when implied corr 20%+ above realized

    _DATA_BASE = "https://data.alpaca.markets"
    _ALPACA_BASE = "https://paper-api.alpaca.markets"

    async def _fetch_hv(self, symbol: str, days: int = 30) -> Optional[float]:
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

    async def _get_atm_iv(self, symbol: str) -> Optional[float]:
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

    async def _fetch_daily_returns(self, symbol: str) -> Optional[np.ndarray]:
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

    async def _compute_realized_correlation(self) -> Optional[float]:
        """Compute average pairwise realized correlation of top 5 components."""
        syms = list(self.COMPONENTS.keys())
        tasks = [self._fetch_daily_returns(s) for s in syms]
        all_returns = await asyncio.gather(*tasks, return_exceptions=True)

        valid = [
            r
            for r in all_returns
            if not isinstance(r, Exception) and r is not None and len(r) > 10
        ]
        if len(valid) < 2:
            return None

        min_len = min(len(r) for r in valid)
        matrix = np.column_stack([r[-min_len:] for r in valid])
        corr_matrix = np.corrcoef(matrix.T)

        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        return float(corr_matrix[mask].mean())

    async def analyze(self, data: pd.DataFrame, symbol: str = "QQQ") -> Optional[Signal]:
        """Generate a trading signal based on implied vs realized correlation."""
        start_time = perf_counter()

        # Fetch IVs concurrently (index + first two components)
        syms = [self.INDEX] + list(self.COMPONENTS.keys())[:2]
        iv_tasks = [self._get_atm_iv(s) for s in syms]
        hv_tasks = [self._fetch_hv(s) for s in syms]

        ivs_raw, hvs_raw = await asyncio.gather(
            asyncio.gather(*iv_tasks, return_exceptions=True),
            asyncio.gather(*hv_tasks, return_exceptions=True),
        )

        # Validate index IV
        index_iv = ivs_raw[0] if not isinstance(ivs_raw[0], Exception) and ivs_raw[0] else None
        if index_iv is None:
            logger.info("dispersion_trading analyze aborted: missing index IV")
            return None

        # Component IVs
        comp_ivs = [
            iv for iv in ivs_raw[1:] if not isinstance(iv, Exception) and iv
        ]
        if not comp_ivs:
            logger.info("dispersion_trading analyze aborted: missing component IVs")
            return None

        # Approximate implied correlation
        weights = list(self.COMPONENTS.values())[: len(comp_ivs)]
        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]

        weighted_var_sum = sum(w * iv ** 2 for w, iv in zip(norm_w, comp_ivs))
        if index_iv ** 2 <= weighted_var_sum:
            implied_corr = 0.0
        else:
            cross_term_approx = 2 * (sum(norm_w) ** 2) * np.mean(comp_ivs) ** 2
            implied_corr = min(
                (index_iv ** 2 - weighted_var_sum) / max(cross_term_approx, 0.001), 1.0
            )

        # Realized correlation
        realized_corr = await self._compute_realized_correlation()
        if realized_corr is None or realized_corr <= 0:
            logger.info(
                "dispersion_trading analyze aborted: unable to compute realized correlation"
            )
            return None

        # Decision rule
        corr_ratio = implied_corr / realized_corr
        signal: Optional[Signal] = None
        if corr_ratio > 1 + self.MIN_CORR_PREMIUM:
            # Generate a short signal on the index (sell variance) and long on components
            signal = Signal(
                name=self.name,
                direction="short",
                metadata={
                    "implied_corr": implied_corr,
                    "realized_corr": realized_corr,
                    "corr_ratio": corr_ratio,
                },
            )
            logger.info(
                "dispersion_trading signal generated",
                signal_name=signal.name,
                direction=signal.direction,
                implied_corr=implied_corr,
                realized_corr=realized_corr,
                corr_ratio=corr_ratio,
            )
        else:
            logger.info(
                "dispersion_trading no signal",
                implied_corr=implied_corr,
                realized_corr=realized_corr,
                corr_ratio=corr_ratio,
            )

        exec_time = perf_counter() - start_time
        logger.info(
            "dispersion_trading analyze completed",
            signal_count=1 if signal else 0,
            execution_time=exec_time,
            pnl=getattr(signal, "pnl", None),
        )
        return signal