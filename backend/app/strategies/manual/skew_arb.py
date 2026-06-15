"""
Options Skew Arbitrage
=======================
The volatility skew is the difference in implied volatility between
OTM puts and OTM calls. Normally, OTM puts trade at higher IV than calls
(investors pay a premium for downside protection → "fear" asymmetry).

When skew is EXTREME (puts much more expensive than calls relative to history):
→ Puts are rich: sell them and buy calls (delta-neutral)
→ When skew normalizes, profit from IV compression of puts + drift of calls

When skew is LOW (unusual):
→ Unusual → reversion expected
→ Buy puts, sell calls (risk reversal)

Key metrics:
- 25-delta put IV vs 25-delta call IV (risk reversal)
- Skew = IV_put_25Δ - IV_call_25Δ (normal = 3-5% for SPY)
- CBOE SKEW Index > 140 = extreme fear → sell puts

Entry: Skew > historical 80th percentile → sell puts relative to calls
Exit: Skew reverts to 50th percentile

Academic: Bates (2003), Foresi & Wu (2005), Santa-Clara & Yan (2010)
Documented: Risk reversal strategies earn ~0.5% monthly (6% annually)
Implementation: Requires options with clear 25-delta strikes
"""
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class SkewArbitrageStrategy(AbstractStrategy):
    name = "skew_arb"
    display_name = "Skew Arbitrage"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"

    UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    TARGET_DELTA = 0.25   # Use 25-delta options
    TARGET_DTE_MIN = 21
    TARGET_DTE_MAX = 45
    HIGH_SKEW_PCTILE = 80  # Enter when skew > 80th percentile
    LOW_SKEW_PCTILE = 20   # Enter (reverse) when skew < 20th percentile

    _ALPACA_BASE = "https://paper-api.alpaca.markets"
    _DATA_BASE = "https://data.alpaca.markets"

    async def _get_skew(self, symbol: str) -> dict | None:
        """Get current skew: IV difference between 25Δ put and 25Δ call."""
        today = date.today()
        exp_min = (today + timedelta(days=self.TARGET_DTE_MIN)).isoformat()
        exp_max = (today + timedelta(days=self.TARGET_DTE_MAX)).isoformat()

        async with httpx.AsyncClient(timeout=15.0) as client:
            contracts_resp = await client.get(
                f"{self._ALPACA_BASE}/v2/options/contracts",
                params={
                    "underlying_symbols": symbol,
                    "expiration_date_gte": exp_min,
                    "expiration_date_lte": exp_max,
                    "limit": 200,
                },
                headers=alpaca_headers(),
            )
            if contracts_resp.status_code != 200:
                return None
            contracts = contracts_resp.json().get("option_contracts", [])
            syms = [c["symbol"] for c in contracts if c.get("symbol")]
            if not syms:
                return None

            snap_resp = await client.get(
                f"{self._ALPACA_BASE}/v2/options/snapshots",
                params={"symbols": ",".join(syms[:50]), "feed": "indicative"},
                headers=alpaca_headers(),
            )
        if snap_resp.status_code != 200:
            return None
        snapshots = snap_resp.json().get("snapshots", {})

        # Find ~25-delta call and ~25-delta put
        put_25 = None
        call_25 = None
        put_25_iv = None
        call_25_iv = None
        put_25_best_diff = float("inf")
        call_25_best_diff = float("inf")

        for contract in contracts:
            sym = contract.get("symbol", "")
            snap = snapshots.get(sym, {})
            greeks = snap.get("greeks", {})
            delta = greeks.get("delta")
            iv = snap.get("impliedVolatility")
            option_type = contract.get("type")

            if delta is None or iv is None:
                continue

            delta_abs = abs(float(delta))
            iv_val = float(iv)
            diff = abs(delta_abs - self.TARGET_DELTA)

            if option_type == "put" and diff < 0.05 and diff < put_25_best_diff:
                put_25 = sym
                put_25_iv = iv_val
                put_25_best_diff = diff

            if option_type == "call" and diff < 0.05 and diff < call_25_best_diff:
                call_25 = sym
                call_25_iv = iv_val
                call_25_best_diff = diff

        if put_25_iv is None or call_25_iv is None:
            return None

        skew = put_25_iv - call_25_iv
        return {
            "skew": skew,
            "put_25d_iv": put_25_iv,
            "call_25d_iv": call_25_iv,
            "put_symbol": put_25,
            "call_symbol": call_25,
        }

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        if symbol not in self.UNIVERSE:
            return None

        skew_data = await self._get_skew(symbol)
        if not skew_data:
            return None

        skew = skew_data["skew"]

        # Historical skew estimate: normal SPY skew ≈ 0.04-0.07 (4-7% IV difference)
        # High skew (> 0.10) = puts expensive relative to calls → sell puts, buy calls
        # Low skew (< 0.02) = unusual, buy puts relative to calls
        HIGH_SKEW_THRESHOLD = 0.09
        LOW_SKEW_THRESHOLD = 0.02

        if skew > HIGH_SKEW_THRESHOLD:
            # Puts expensive: sell puts (delta-neutral with calls)
            confidence = min((skew - HIGH_SKEW_THRESHOLD) / 0.04, 1.0)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": "skew_arb",
                    "skew": round(skew, 4),
                    "regime": "high_skew_sell_puts",
                    "put_25d_iv": round(skew_data["put_25d_iv"], 4),
                    "call_25d_iv": round(skew_data["call_25d_iv"], 4),
                    "trade": "Sell 25-delta puts, buy 25-delta calls (delta neutral)",
                    "put_symbol": skew_data["put_symbol"],
                    "call_symbol": skew_data["call_symbol"],
                },
            )
        elif skew < LOW_SKEW_THRESHOLD:
            # Puts cheap: buy puts (relative to calls)
            confidence = min((LOW_SKEW_THRESHOLD - skew) / 0.03, 1.0)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": "skew_arb",
                    "skew": round(skew, 4),
                    "regime": "low_skew_buy_puts",
                    "trade": "Buy 25-delta puts, sell 25-delta calls",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
        # High vol → high skew historically → sell put-heavy structures (short)
        # Low vol → low skew → buy puts (long)
        vol_pctile = vol_20.rolling(252).rank(pct=True)
        high_skew = (vol_pctile.shift(1) > 0.8).fillna(False)
        low_skew = (vol_pctile.shift(1) < 0.2).fillna(False)
        neutral = (~high_skew & ~low_skew)

        return BacktestSignals(
            entries=low_skew,          # buy when skew unusually low
            exits=(high_skew | neutral).fillna(False),
            short_entries=high_skew,   # sell/short when skew unusually high
            short_exits=(low_skew | neutral).fillna(False),
        )
