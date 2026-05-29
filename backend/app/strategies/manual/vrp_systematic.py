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
import numpy as np
import pandas as pd
import httpx
from datetime import date, timedelta
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.config import settings

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


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

    def _headers(self):
        return {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

    async def _get_realized_vol(self, symbol: str) -> float | None:
        """Compute 20-day annualized realized volatility from daily closes."""
        start = (date.today() - timedelta(days=40)).isoformat()
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{self._DATA_BASE}/v2/stocks/{symbol}/bars",
                params={"timeframe": "1Day", "start": start, "limit": 30},
                headers=self._headers(),
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
                headers=self._headers(),
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
                headers=self._headers(),
            )
        if snap_resp.status_code != 200:
            return None
        snapshots = snap_resp.json().get("snapshots", {})
        snap = snapshots.get(atm_sym, {})
        iv = snap.get("impliedVolatility")
        return float(iv) if iv is not None else None

    def _get_vix_vrp(self) -> dict | None:
        """
        Item 9: Compute VRP using free yfinance data.
        VRP = 30-day implied vol (VIX/√12 as 30-day proxy) minus
              30-day realized vol of SPY.

        Signal:
          VRP > 4 vol points → SELL (short variance, collect premium)
          VRP < -2 vol points → BUY (long variance, buy protection)
        """
        if not _YF_AVAILABLE:
            return None
        try:
            # VIX is the 30-calendar-day IV for SPX (approximately = SPY IV)
            vix_data = yf.Ticker("^VIX").history(period="5d")
            if vix_data.empty:
                return None
            vix_level = float(vix_data["Close"].iloc[-1])
            iv_30d = vix_level / 100.0  # convert percentage to decimal

            # SPY 30-day realized vol
            spy_data = yf.Ticker("SPY").history(period="60d")
            if len(spy_data) < 22:
                return None
            spy_log_rets = np.log(spy_data["Close"] / spy_data["Close"].shift(1)).dropna()
            rv_30d = float(spy_log_rets.iloc[-22:].std() * np.sqrt(252))

            vrp_vol_pts = (iv_30d - rv_30d) * 100  # in vol-point units (e.g. 4 = 4 pp)
            return {
                "iv_30d": iv_30d,
                "rv_30d": rv_30d,
                "vrp_vol_pts": vrp_vol_pts,
                "vix_level": vix_level,
            }
        except Exception:
            return None

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        if symbol not in self.UNIVERSE:
            return None
        if data.empty or "close" not in data.columns:
            return None
        spot = float(data["close"].iloc[-1])

        # Item 9: Try free VIX-based VRP first (no API key needed)
        import asyncio
        loop = asyncio.get_event_loop()
        vix_vrp = await loop.run_in_executor(None, self._get_vix_vrp)

        if vix_vrp is not None:
            vrp_pts = vix_vrp["vrp_vol_pts"]
            iv_30d = vix_vrp["iv_30d"]
            rv_30d = vix_vrp["rv_30d"]

            if vrp_pts > 4.0:
                # Short variance: implied > realized by >4 vol points
                confidence = min((vrp_pts - 4.0) / 6.0, 1.0)
                return Signal(
                    symbol=symbol,
                    side="sell",
                    confidence=confidence,
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    metadata={
                        "strategy": "vrp_systematic",
                        "vrp_method": "vix_proxy",
                        "implied_vol_30d": round(iv_30d, 4),
                        "realized_vol_30d": round(rv_30d, 4),
                        "vrp_vol_pts": round(vrp_pts, 2),
                        "vix_level": round(vix_vrp["vix_level"], 2),
                        "signal": "short_variance",
                        "order_type": "straddle",
                        "target_dte": self.TARGET_DTE,
                        "profit_target_pct": self.PROFIT_TARGET,
                        "stop_mult": self.STOP_MULT,
                    },
                )
            elif vrp_pts < -2.0:
                # Long variance: realized > implied → buy protection
                confidence = min((abs(vrp_pts) - 2.0) / 4.0, 1.0)
                return Signal(
                    symbol=symbol,
                    side="buy",
                    confidence=confidence,
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    metadata={
                        "strategy": "vrp_systematic",
                        "vrp_method": "vix_proxy",
                        "implied_vol_30d": round(iv_30d, 4),
                        "realized_vol_30d": round(rv_30d, 4),
                        "vrp_vol_pts": round(vrp_pts, 2),
                        "vix_level": round(vix_vrp["vix_level"], 2),
                        "signal": "long_variance_protection",
                        "order_type": "straddle",
                    },
                )
            return None  # VRP in neutral zone, no trade

        # Fallback: use Alpaca options data (original implementation)
        rv = await self._get_realized_vol(symbol)
        iv = await self._get_implied_vol(symbol, spot)

        if rv is None or iv is None or rv < 0.001:
            return None

        iv_rv_ratio = iv / rv
        vrp = iv - rv  # volatility risk premium in annualized vol points

        if iv_rv_ratio < self.IV_RV_THRESHOLD:
            return None  # Options not rich enough

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
                "vrp_method": "alpaca_options",
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

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Proxy: IV/RV ratio using HV20 vs HV60 (HV60 as IV proxy in absence of option data)."""
        log_ret = np.log(df["close"] / df["close"].shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv60 = log_ret.rolling(60).std() * np.sqrt(252)
        ratio = hv60 / hv20.clip(lower=0.01)

        # Sell straddle when HV60 (IV proxy) >> HV20 (recent realized vol)
        entries = (ratio.shift(1) > self.IV_RV_THRESHOLD).fillna(False)
        exits = (ratio.shift(1) < 1.0).fillna(False)  # buy back when premium normalizes

        return BacktestSignals(entries=entries, exits=exits)
