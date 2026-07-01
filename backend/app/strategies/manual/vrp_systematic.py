import logging
import time
import numpy as np
import pandas as pd
import httpx
from datetime import date, timedelta
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.config import settings
from app.brokers.alpaca_headers import alpaca_headers

logger = logging.getLogger(__name__)


class VRPSystematicStrategy(AbstractStrategy):
    name = "vrp_systematic"
    display_name = "VRP Systematic Harvesting"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0

    UNIVERSE = ["SPY", "QQQ", "IWM"]
    IV_RV_THRESHOLD = 1.15   # Sell when IV is 15%+ above RV
    RV_LOOKBACK = 20          # 20-day realized vol
    PROFIT_TARGET = 0.50      # Exit at 50% of max credit
    STOP_MULT = 2.0           # Exit at 2× credit loss
    MIN_DTE = 21              # Minimum DTE for entry
    TARGET_DTE = 30           # Target DTE at entry

    _DATA_BASE = "https://data.alpaca.markets"
    _ALPACA_BASE = "https://paper-api.alpaca.markets"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._signal_counter = 0

    async def _get_realized_vol(self, symbol: str) -> float | None:
        """Compute 20-day annualized realized volatility from daily closes."""
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
        if len(bars) < self.RV_LOOKBACK:
            return None
        closes = [float(b["c"]) for b in bars]
        log_rets = np.diff(np.log(closes))
        rv = float(np.std(log_rets[-self.RV_LOOKBACK:]) * np.sqrt(252))
        return rv

    async def _get_implied_vol(self, symbol: str, spot: float) -> float | None:
        """Get ATM implied vol from Alpaca options snapshots."""
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
            # Find ATM call (strike closest to spot)
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

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        start_time = time.time()
        if symbol not in self.UNIVERSE:
            logger.info(
                "VRP analyze skipped",
                extra={"symbol": symbol, "reason": "outside_universe", "execution_time": time.time() - start_time},
            )
            return None
        if data.empty or "close" not in data.columns:
            logger.info(
                "VRP analyze skipped",
                extra={"symbol": symbol, "reason": "invalid_data", "execution_time": time.time() - start_time},
            )
            return None
        spot = float(data["close"].iloc[-1])

        rv = await self._get_realized_vol(symbol)
        iv = await self._get_implied_vol(symbol, spot)

        if rv is None or iv is None or rv < 0.001:
            logger.info(
                "VRP analyze skipped",
                extra={"symbol": symbol, "reason": "missing_vol_data", "execution_time": time.time() - start_time},
            )
            return None

        iv_rv_ratio = iv / rv
        vrp = iv - rv  # volatility risk premium in annualized vol points

        if iv_rv_ratio < self.IV_RV_THRESHOLD:
            logger.info(
                "VRP analyze skipped",
                extra={"symbol": symbol, "reason": "ratio_below_threshold", "iv_rv_ratio": iv_rv_ratio, "execution_time": time.time() - start_time},
            )
            return None  # Options not rich enough

        confidence = min((iv_rv_ratio - self.IV_RV_THRESHOLD) / 0.3, 1.0)

        signal = Signal(
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
                "profit_target_pct": self.PROFIT_TARGET,
                "stop_mult": self.STOP_MULT,
            },
        )

        self._signal_counter += 1
        elapsed = time.time() - start_time
        logger.info(
            "VRP signal generated",
            extra={
                "symbol": symbol,
                "signal_count": self._signal_counter,
                "execution_time": elapsed,
                "iv_rv_ratio": iv_rv_ratio,
                "confidence": confidence,
                "pnl": None,  # P&L to be filled post‑execution
            },
        )
        return signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Proxy: IV/RV ratio using HV20 vs HV60 (HV60 as IV proxy in absence of option data)."""
        log_ret = np.log(df["close"] / df["close"].shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv60 = log_ret.rolling(60).std() * np.sqrt(252)
        ratio = hv60 / hv20.clip(lower=0.01)

        # Sell straddle when HV60 (IV proxy) >> HV20 (recent realized vol)
        entries = (ratio.shift(1) > self.IV_RV_THRESHOLD).fillna(False)
        exits = (ratio.shift(1) < 1.0).fillna(False)  # buy back when premium normalizes

        entry_count = int(entries.sum())
        exit_count = int(exits.sum())
        logger.info(
            "VRP backtest signal generation",
            extra={
                "entry_signals": entry_count,
                "exit_signals": exit_count,
            },
        )
        return BacktestSignals(entries=entries, exits=exits)