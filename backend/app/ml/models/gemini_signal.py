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

import os
import re
import json
import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

# Gemini model configuration
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.1
GEMINI_MAX_OUTPUT_TOKENS = 128

# Prompt handling
JSON_EXTRACT_REGEX = r'\{.*?"direction_prob_up".*?\}'
DIRECTION_PROB_KEY = "direction_prob_up"

# Technical indicator windows
RSI_PERIOD = 14
ATR_PERIOD = 14
RETURN_5_PERIOD = 5
RETURN_20_PERIOD = 20
SMA_PERIOD = 20
VOL_AVG_PERIOD = 20
EMA_PERIOD = 50

# Miscellaneous
EPS = 1e-9
MIN_DF_LENGTH = 20
DEFAULT_INTERVAL = "1d"
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEY_1")

# System and analysis prompts
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

    ret5 = float(close.pct_change(RETURN_5_PERIOD).iloc[-1]) if n >= RETURN_5_PERIOD else 0.0
    ret20 = float(close.pct_change(RETURN_20_PERIOD).iloc[-1]) if n >= RETURN_20_PERIOD else 0.0

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = gain / (loss + EPS)
    rsi = float(100 - 100 / (1 + rs.iloc[-1]))

    sma20 = float(close.rolling(SMA_PERIOD).mean().iloc[-1]) if n >= SMA_PERIOD else price
    vs_sma = (price - sma20) / (sma20 + EPS)

    # ATR
    if "high" in df.columns and "low" in df.columns:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(tr.ewm(span=ATR_PERIOD, adjust=False).mean().iloc[-1])
    else:
        atr = float(close.rolling(ATR_PERIOD).std().iloc[-1]) if n >= ATR_PERIOD else 0.0
    atr_pct = atr / (price + EPS)

    vol_ratio = 1.0
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        avg_vol = (
            float(vol.rolling(VOL_AVG_PERIOD).mean().iloc[-1])
            if n >= VOL_AVG_PERIOD
            else float(vol.mean())
        )
        vol_ratio = float(vol.iloc[-1]) / (avg_vol + EPS)

    high_20 = (
        float(df["high"].astype(float).rolling(SMA_PERIOD).max().iloc[-1])
        if "high" in df.columns and n >= SMA_PERIOD
        else price
    )
    low_20 = (
        float(df["low"].astype(float).rolling(SMA_PERIOD).min().iloc[-1])
        if "low" in df.columns and n >= SMA_PERIOD
        else price
    )
    range_pct = (high_20 - low_20) / (low_20 + EPS)

    ema50 = float(close.ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1]) if n >= EMA_PERIOD else price
    trend = "uptrend" if price > ema50 else "downtrend" if price < ema50 * 0.98 else "neutral"

    return _ANALYSIS_TEMPLATE.format(
        symbol=symbol,
        interval=interval,
        n=n,
        price=price,
        ret5=ret5,
        ret20=ret20,
        rsi=rsi,
        vs_sma=vs_sma,
        atr_pct=atr_pct,
        vol_ratio=vol_ratio,
        range_pct=range_pct,
        trend=trend,
    )


def _call_gemini_json(prompt: str, api_key: str) -> dict[str, Any]:
    """Synchronous Gemini call returning parsed JSON dict."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
                "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            },
            system_instruction=_SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        text = response.text.strip() if response.text else ""

        # Extract JSON
        m = re.search(JSON_EXTRACT_REGEX, text, re.DOTALL)
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
        self._key = next((os.environ.get(var) for var in API_KEY_ENV_VARS if os.environ.get(var)), "")
        self._available = bool(self._key)
        if not self._available:
            logger.debug("GeminiSignalEngine: no API key — signals disabled")

    @property
    def is_available(self) -> bool:
        return self._available

    async def predict_proba(self, df: pd.DataFrame, symbol: str, interval: str = DEFAULT_INTERVAL) -> float | None:
        """
        Returns probability of upward price movement (0.0–1.0), or None if unavailable.
        """
        if not self._available or df is None or len(df) < MIN_DF_LENGTH:
            return None

        prompt = _compute_summary(df, symbol, interval)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _call_gemini_json, prompt, self._key)

        prob = result.get(DIRECTION_PROB_KEY)
        if prob is not None:
            return float(np.clip(prob, 0.0, 1.0))
        return None

    def predict_proba_sync(self, df: pd.DataFrame, symbol: str, interval: str = DEFAULT_INTERVAL) -> float | None:
        """Synchronous version for use outside async context."""
        if not self._available or df is None or len(df) < MIN_DF_LENGTH:
            return None
        prompt = _compute_summary(df, symbol, interval)
        result = _call_gemini_json(prompt, self._key)
        prob = result.get(DIRECTION_PROB_KEY)
        return float(np.clip(prob, 0.0, 1.0)) if prob is not None else None


# Module-level singleton
_gemini_engine: GeminiSignalEngine | None = None


def get_gemini_engine() -> GeminiSignalEngine:
    global _gemini_engine
    if _gemini_engine is None:
        _gemini_engine = GeminiSignalEngine()
    return _gemini_engine