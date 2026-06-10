"""
Macro Risk Barometer Strategy
===============================
Academic basis:
  - Multi-factor risk-on/risk-off regime detection.
  - VIX as fear gauge: Whaley (2000) "The Investor Fear Gauge", JPortfolio Management.
  - Credit spreads as leading indicators: Gilchrist & Zakrajsek (2012), AER.
  - Yield curve as recession predictor: Harvey (1989).

Three independent risk indicators, each scores +1 (risk-on) or 0 (risk-off):
  1. VIX < 20 → risk-on (+1)
  2. TLT/SHY ratio > 252-day median → curve not inverted (+1)
  3. HYG/LQD ratio rising (20-day momentum > 0) → credit spreads tightening (+1)

  Score 3: full risk-on  → long SPY with high confidence.
  Score 2: moderate risk-on → long SPY with medium confidence.
  Score 1: cautious      → flat.
  Score 0: full risk-off → long TLT / defensive.

ETFs used (all via yfinance):
  ^VIX  — CBOE Volatility Index
  TLT   — 20Y Treasury (long duration)
  SHY   — 1-3Y Treasury (short duration)
  HYG   — iShares HY Corporate Bond ETF
  LQD   — iShares IG Corporate Bond ETF
  SPY   — S&P 500 ETF
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_VIX_THRESHOLD  = 20.0
_CREDIT_WINDOW  = 20
_CURVE_WINDOW   = 252


def _fetch_yf(symbol: str, period: str = "3y") -> pd.Series | None:
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes
    except Exception:
        return None


class MacroRiskBarometerStrategy(AbstractStrategy):
    """
    Composite risk-on/risk-off barometer using VIX, yield curve, and HYG credit spread.
    Scores: VIX < 20 (+1), yield curve > 0 (+1), HYG/LQD ratio rising (+1).
    Score 3: full risk-on (long SPY). Score 0: full risk-off (long TLT).
    Risk bucket: directional, market_type: equity
    """

    name = "macro_risk_barometer"
    display_name = "Macro Risk Barometer (VIX+Curve+Credit)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.vix_threshold   = float(p.get("vix_threshold",  _VIX_THRESHOLD))
        self.credit_window   = int(p.get("credit_window",    _CREDIT_WINDOW))
        self.curve_window    = int(p.get("curve_window",     _CURVE_WINDOW))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        vix = _fetch_yf("^VIX")
        tlt = _fetch_yf("TLT")
        shy = _fetch_yf("SHY")
        hyg = _fetch_yf("HYG")
        lqd = _fetch_yf("LQD")

        if any(s is None for s in (vix, tlt, shy, hyg, lqd)):
            return None

        score = 0
        details: dict = {}

        # 1. VIX signal
        vix_now = float(vix.iloc[-1])
        details["vix"] = round(vix_now, 2)
        if vix_now < self.vix_threshold:
            score += 1
            details["vix_signal"] = "risk_on"
        else:
            details["vix_signal"] = "risk_off"

        # 2. Yield curve signal (TLT/SHY ratio vs rolling median)
        df_curve = pd.DataFrame({"tlt": tlt, "shy": shy}).dropna()
        if len(df_curve) >= self.curve_window:
            ratio_curve = df_curve["tlt"] / df_curve["shy"].clip(lower=1e-8)
            curve_median = float(ratio_curve.tail(self.curve_window).median())
            curve_now = float(ratio_curve.iloc[-1])
            details["curve_ratio"] = round(curve_now, 4)
            details["curve_median"] = round(curve_median, 4)
            if curve_now > curve_median:
                score += 1
                details["curve_signal"] = "risk_on"
            else:
                details["curve_signal"] = "risk_off"

        # 3. Credit spread signal (HYG/LQD ratio 20-day momentum)
        df_credit = pd.DataFrame({"hyg": hyg, "lqd": lqd}).dropna()
        if len(df_credit) > self.credit_window:
            ratio_credit = df_credit["hyg"] / df_credit["lqd"].clip(lower=1e-8)
            mom_credit = float(ratio_credit.pct_change(self.credit_window).iloc[-1])
            details["hyg_lqd_momentum"] = round(mom_credit, 4)
            if not np.isnan(mom_credit) and mom_credit > 0:
                score += 1
                details["credit_signal"] = "risk_on"
            else:
                details["credit_signal"] = "risk_off"

        details["composite_score"] = score

        if score >= 3:
            trade_sym = "SPY"
            side = "buy"
            confidence = 0.90
        elif score == 2:
            trade_sym = "SPY"
            side = "buy"
            confidence = 0.72
        elif score == 0:
            trade_sym = "TLT"
            side = "buy"
            confidence = 0.80
        else:
            return None  # score == 1: ambiguous

        if symbol in ("SPY", "TLT", "HYG"):
            trade_sym = symbol

        details["academic_ref"] = "Whaley (2000) Fear Gauge + Harvey (1989) + Gilchrist & Zakrajsek (2012)"
        return Signal(
            symbol=trade_sym,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata=details,
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Backtest proxy: use df['close'] as SPY, with VIX proxy derived from rolling
        realized vol. Curve and credit signals approximated from price data.
        """
        if "close" not in df.columns or len(df) < max(self.curve_window, self.credit_window) + 5:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"].astype(float)
        returns = close.pct_change()

        # VIX proxy: 20-day realized vol annualized
        rvol = returns.rolling(20).std() * np.sqrt(252) * 100
        vix_signal = rvol < self.vix_threshold

        # Curve proxy: price trend (rising = steepening proxy)
        sma_short = close.rolling(self.credit_window).mean()
        sma_long  = close.rolling(self.curve_window).mean()
        curve_signal = sma_short > sma_long

        # Credit proxy: momentum > 0 (risk-on when price momentum positive)
        mom = close.pct_change(self.credit_window)
        credit_signal = mom > 0

        score = vix_signal.astype(int) + curve_signal.astype(int) + credit_signal.astype(int)

        raw_entries = (score >= 2)
        raw_exits   = (score <= 1)

        entries = raw_entries.shift(1).fillna(False).astype(bool)
        exits   = raw_exits.shift(1).fillna(False).astype(bool)

        short_entries = (score == 0).shift(1).fillna(False).astype(bool)
        short_exits   = (score >= 2).shift(1).fillna(False).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
