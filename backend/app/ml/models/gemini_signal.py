"""
Gemini Signal Engine — uses Google AI Studio's Gemini API to analyze
market data and generate directional probability signals.

Free tier: 1500 req/day (Gemini 2.0 Flash).
No GPU, no model files needed — inference runs in Google's cloud.

The signal is computed by:
1. Summarizing recent OHLCV + technical indicators as a structured prompt
2. Asking Gemini to reason about market regime and direction probability
3. Returning a calibrated probability (0.0–1.0) for upward price movement
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()

_SYSTEM_PROMPT = """You are a quantitative trading analyst. Given OHLCV data and
technical indicators, output ONLY a JSON object with your directional forecast.
No explanation, no markdown — just the JSON."""

_ANALYSIS_TEMPLATE = """Analyze the following market data and forecast the next-bar direction.

Symbol: {symbol}
Interval: {interval}
Last {n} bars summary:
- Current price: {price:.4f}
- 5-bar return: {ret5:+.2%}
- 20-bar return: {ret20:+.2%}
- RSI(14): {rsi:.1f}
- Price vs 20-SMA: {vs_sma:+.2%}
- ATR(14) / price: {atr_pct:.3f} (volatility)
- Volume ratio vs 20-bar avg: {vol_ratio:.2f}x
- Recent high/low range: {range_pct:.2%}
- Trend: {trend}

Respond with ONLY this JSON (no other text):
{{"direction_prob_up": <float 0.0-1.0>, "confidence": <"low"|"medium"|"high">, "regime": <"trending"|"ranging"|"volatile">}}"""


def _compute_summary(df: pd.DataFrame, symbol: str, interval: str) -> str:
    """Compute a compact technical summary for Gemini."""
    close = df["close"].astype(float)
    n = len(close)
    price = float(close.iloc[-1])

    ret5 = float(close.pct_change(5).iloc[-1]) if n >= 5 else 0.0
    ret20 = float(close.pct_change(20).iloc[-1]) if n >= 20 else 0.0

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    rsi = float(100 - 100 / (1 + rs.iloc[-1]))

    sma20 = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else price
    vs_sma = (price - sma20) / (sma20 + 1e-9)

    # ATR
    if "high" in df.columns and "low" in df.columns:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
    else:
        atr = float(close.rolling(14).std().iloc[-1]) if n >= 14 else 0.0
    atr_pct = atr / (price + 1e-9)

    vol_ratio = 1.0
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        avg_vol = float(vol.rolling(20).mean().iloc[-1]) if n >= 20 else float(vol.mean())
        vol_ratio = float(vol.iloc[-1]) / (avg_vol + 1e-9)

    high_20 = float(df["high"].astype(float).rolling(20).max().iloc[-1]) if "high" in df.columns and n >= 20 else price
    low_20 = float(df["low"].astype(float).rolling(20).min().iloc[-1]) if "low" in df.columns and n >= 20 else price
    range_pct = (high_20 - low_20) / (low_20 + 1e-9)

    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if n >= 50 else price
    trend = "uptrend" if price > ema50 else "downtrend" if price < ema50 * 0.98 else "neutral"

    return _ANALYSIS_TEMPLATE.format(
        symbol=symbol, interval=interval, n=n, price=price,
        ret5=ret5, ret20=ret20, rsi=rsi, vs_sma=vs_sma,
        atr_pct=atr_pct, vol_ratio=vol_ratio, range_pct=range_pct, trend=trend,
    )


def _call_gemini_json(prompt: str, api_key: str) -> dict[str, Any]:
    """Synchronous Gemini call returning parsed JSON dict."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={"temperature": 0.1, "max_output_tokens": 128},
            system_instruction=_SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        text = response.text.strip() if response.text else ""

        # Extract JSON
        m = re.search(r'\{.*?"direction_prob_up".*?\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Gemini signal call failed", error=str(e))
    return {}


class GeminiSignalEngine:
    """
    Wraps Gemini API as an ML-style signal generator.
    Implements the AbstractModel interface (predict method returns probability).
    """
    model_type = "gemini_signal"

    def __init__(self):
        self._key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY_1")
            or ""
        )
        self._available = bool(self._key)
        if not self._available:
            logger.debug("GeminiSignalEngine: no API key — signals disabled")

    @property
    def is_available(self) -> bool:
        return self._available

    async def predict_proba(self, df: pd.DataFrame, symbol: str, interval: str = "1d") -> float | None:
        """
        Returns probability of upward price movement (0.0–1.0), or None if unavailable.
        """
        if not self._available or df is None or len(df) < 20:
            return None

        prompt = _compute_summary(df, symbol, interval)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _call_gemini_json, prompt, self._key)

        prob = result.get("direction_prob_up")
        if prob is not None:
            return float(np.clip(prob, 0.0, 1.0))
        return None

    def predict_proba_sync(self, df: pd.DataFrame, symbol: str, interval: str = "1d") -> float | None:
        """Synchronous version for use outside async context."""
        if not self._available or df is None or len(df) < 20:
            return None
        prompt = _compute_summary(df, symbol, interval)
        result = _call_gemini_json(prompt, self._key)
        prob = result.get("direction_prob_up")
        return float(np.clip(prob, 0.0, 1.0)) if prob is not None else None


# Module-level singleton
_gemini_engine: GeminiSignalEngine | None = None


def get_gemini_engine() -> GeminiSignalEngine:
    global _gemini_engine
    if _gemini_engine is None:
        _gemini_engine = GeminiSignalEngine()
    return _gemini_engine
