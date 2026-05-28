"""
Post-Earnings Announcement Drift (PEAD) with SUE Factor
=========================================================
One of the most robust anomalies in finance: after large positive earnings surprises,
stocks continue to drift upward for 30-90 days. After large negative surprises, drift down.

SUE (Standardized Unexpected Earnings) = (EPS_actual - EPS_estimate) / std(surprise_history)
  SUE > 2.0: Strong positive drift (BUY, hold 60 days)
  SUE < -2.0: Strong negative drift (SELL/SHORT, hold 60 days)

Why it persists: Investors under-react to earnings news. Analysts are slow to revise.
Institutional investors gradually accumulate after the announcement.

Academic: Ball & Brown (1968), Bernard & Thomas (1989), Chan et al. (1996)
Documented:
  - Quartile 4 (highest SUE) outperforms Quartile 1 by 8-10% annually
  - Effect is stronger for smaller stocks, less analyst coverage
  - Persists after 60 days but fades by 90 days → optimal hold = 45-60 days

Data: Alpaca earnings/corporate actions API (or proxy from price gap on earnings day)

Gap proxy: If no earnings API data available, use the earnings-day price gap as SUE proxy:
  Gap > 5%: Strong positive surprise → BUY
  Gap < -5%: Strong negative surprise → SELL
"""
import numpy as np
import pandas as pd
import httpx
from datetime import date, timedelta
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.config import settings


class PEADStrategy(AbstractStrategy):
    name = "pead_sue"
    display_name = "PEAD / SUE Factor"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"

    MIN_GAP_PCT = 0.04   # 4% gap minimum (proxy for significant surprise)
    HOLD_DAYS = 45       # Optimal hold period (Bernard & Thomas 1989)
    MAX_HOLD_DAYS = 60   # Exit by 60 days regardless

    _DATA_BASE = "https://data.alpaca.markets"
    _ALPACA_BASE = "https://paper-api.alpaca.markets"

    def _headers(self):
        return {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

    async def _get_earnings_gap(self, symbol: str, days_lookback: int = 5) -> dict | None:
        """
        Detect recent earnings by looking for unusual overnight gaps.
        A gap > MIN_GAP_PCT on high volume = likely earnings announcement.
        """
        start = (date.today() - timedelta(days=days_lookback + 5)).isoformat()
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{self._DATA_BASE}/v2/stocks/{symbol}/bars",
                params={"timeframe": "1Day", "start": start, "limit": days_lookback + 3},
                headers=self._headers(),
            )
        if resp.status_code != 200:
            return None
        bars = resp.json().get("bars", [])
        if len(bars) < 3:
            return None

        # Look for large overnight gaps in recent bars
        for i in range(1, len(bars)):
            prev_close = float(bars[i-1]["c"])
            curr_open = float(bars[i]["o"])
            curr_vol = float(bars[i].get("v", 0))
            avg_vol = np.mean([float(b.get("v", 0)) for b in bars[:i]])

            gap_pct = (curr_open - prev_close) / prev_close
            volume_spike = curr_vol / max(avg_vol, 1)

            # Earnings = large gap + volume spike
            if abs(gap_pct) >= self.MIN_GAP_PCT and volume_spike > 1.5:
                return {
                    "gap_pct": gap_pct,
                    "volume_spike": volume_spike,
                    "date": bars[i]["t"],
                    "days_ago": len(bars) - i - 1,
                    "current_price": float(bars[-1]["c"]),
                }
        return None

    async def _try_alpaca_earnings(self, symbol: str) -> dict | None:
        """Try to get actual earnings data from Alpaca corporate actions."""
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"https://data.alpaca.markets/v1beta1/corporate-actions",
                params={"types": "earnings", "symbols": symbol, "limit": 5},
                headers=self._headers(),
            )
        if resp.status_code != 200:
            return None
        data = resp.json()
        items = data if isinstance(data, list) else data.get("earnings", [])
        if not items:
            return None
        latest = items[0]
        eps_est = latest.get("estimate_eps") or latest.get("eps_estimate")
        eps_act = latest.get("reported_eps") or latest.get("eps_actual")
        if eps_est and eps_act and float(eps_est) != 0:
            sue = (float(eps_act) - float(eps_est)) / abs(float(eps_est))
            return {"sue": sue, "source": "alpaca_earnings"}
        return None

    async def analyze(self, data: pd.DataFrame, symbol: str = "AAPL") -> Signal | None:
        # Try actual earnings data first
        earnings_data = await self._try_alpaca_earnings(symbol)

        if earnings_data and abs(earnings_data.get("sue", 0)) >= 0.05:
            sue = earnings_data["sue"]
            if abs(sue) < 0.05:
                return None
            side = "buy" if sue > 0 else "sell"
            confidence = min(abs(sue) / 0.20, 1.0)
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": "pead_sue",
                    "sue": round(sue, 4),
                    "source": "actual_earnings",
                    "hold_days": self.HOLD_DAYS,
                    "max_hold_days": self.MAX_HOLD_DAYS,
                    "academic_basis": "Bernard & Thomas (1989)",
                },
            )

        # Fallback: detect earnings via price gap
        gap_data = await self._get_earnings_gap(symbol)
        if gap_data is None:
            return None

        gap_pct = gap_data["gap_pct"]
        days_since = gap_data["days_ago"]

        if days_since > self.HOLD_DAYS:
            return None  # Drift has already played out

        side = "buy" if gap_pct > 0 else "sell"
        confidence = min(abs(gap_pct) / 0.10, 1.0)

        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "pead_sue",
                "earnings_gap_pct": round(gap_pct * 100, 2),
                "volume_spike": round(gap_data["volume_spike"], 1),
                "days_since_earnings": days_since,
                "days_remaining_in_hold": self.HOLD_DAYS - days_since,
                "source": "gap_proxy",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "open" not in df.columns:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )
        # Detect large overnight gaps
        gap = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        vol_spike = df["volume"] / df["volume"].rolling(20).mean().clip(lower=1)
        earnings_day = (gap.abs() >= self.MIN_GAP_PCT) & (vol_spike > 1.5)

        # Forward-fill signal for HOLD_DAYS after each earnings detection
        gap_sign = np.sign(gap)
        long_signal = pd.Series(False, index=df.index)
        short_signal = pd.Series(False, index=df.index)

        for i in range(len(df)):
            if earnings_day.iloc[i]:
                end = min(i + self.HOLD_DAYS, len(df))
                if gap_sign.iloc[i] > 0:
                    long_signal.iloc[i:end] = True
                else:
                    short_signal.iloc[i:end] = True

        entries = long_signal.shift(1).fillna(False)
        exits = (~long_signal).shift(1).fillna(False)
        short_entries = short_signal.shift(1).fillna(False)
        short_exits = (~short_signal).shift(1).fillna(False)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
