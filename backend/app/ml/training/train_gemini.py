"""
Gemini Free Cloud Training — uses Google AI Studio's code execution tool
to train XGBoost/sklearn classifiers inside Gemini's Python sandbox.

Free tier: 1500 requests/day (Gemini 2.0 Flash).
No local GPU needed — all compute happens in Google's cloud.

Usage:
    python -m app.ml.training.train_gemini --symbol BTC-USD --interval 1d
    python -m app.ml.training.train_gemini --config experiments/configs/lstm_btc_1h.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger()

ARTIFACTS_DIR = Path(__file__).parents[4] / "models_artifacts"
RESULTS_DIR = Path(__file__).parents[4] / "experiments" / "results"

# Gemini limits: truncate CSV to this many rows to stay within token budget
MAX_ROWS_FOR_GEMINI = 1000


def _sanitize_data(df: pd.DataFrame, max_rows: int = MAX_ROWS_FOR_GEMINI) -> str:
    """Convert OHLCV DataFrame to a compact CSV string for Gemini."""
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    subset = df[cols].tail(max_rows).copy()
    subset.index = subset.index.astype(str)
    return subset.to_csv()


def _build_training_prompt(csv_data: str, symbol: str, interval: str) -> str:
    return f"""You are a quantitative ML researcher. Train a binary direction classifier
on the following OHLCV data for {symbol} ({interval} bars).

Data (CSV, most recent {MAX_ROWS_FOR_GEMINI} bars):
```
{csv_data}
```

Use Python with pandas, numpy, and scikit-learn (XGBoost if available, else
GradientBoostingClassifier). Follow this exact pipeline:

1. Parse the CSV (index is date).
2. Engineer features:
   - rsi_14: RSI with 14-period window using EWM
   - macd: 12-period EMA minus 26-period EMA
   - bb_width: (upper - lower) / middle, Bollinger Bands 20-period
   - atr_14: Average True Range 14 periods
   - vol_ratio: volume / volume.rolling(20).mean()
   - price_momentum_5: close.pct_change(5)
   - price_momentum_20: close.pct_change(20)
3. Label: 1 if close shifts -1 > 0 (next bar up), else 0.
4. Drop NaN rows. Use 70% train / 15% val / 15% test split (chronological).
5. Train classifier. For XGBoost use: n_estimators=200, max_depth=4,
   learning_rate=0.05, subsample=0.8, eval_metric='auc', early_stopping_rounds=20.
6. Evaluate on test set. Compute:
   - test_accuracy: float
   - test_auc: float
   - test_sharpe: Sharpe ratio of strategy returns (signal * next_return, annualized)
   - feature_importance: dict of feature_name → importance score (top 7)
7. Output ONLY a JSON object on the last line (no trailing text):
{{"status": "success", "symbol": "{symbol}", "interval": "{interval}",
  "n_train": int, "n_test": int,
  "test_accuracy": float, "test_auc": float, "test_sharpe": float,
  "feature_importance": {{}},
  "model_type": "xgboost_or_gbc"
}}

Ensure the JSON is valid and on its own line at the very end.
"""


def _parse_result_from_response(text: str) -> dict[str, Any]:
    """Extract the JSON result block from Gemini's response."""
    # Look for last JSON object in the output
    matches = re.findall(r'\{[^{}]*"status"[^{}]*\}', text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object with status key
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "status" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {"status": "error", "reason": "could not parse JSON from Gemini response"}


def _call_gemini_with_code_execution(prompt: str, api_key: str) -> dict[str, Any]:
    """
    Call Gemini with code execution enabled.
    Returns parsed result dict from executed code output.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            tools="code_execution",
        )
        response = model.generate_content(prompt)

        # Collect all text parts (including code execution output)
        full_text = ""
        for part in response.parts:
            if hasattr(part, "text") and part.text:
                full_text += part.text + "\n"
            if hasattr(part, "executable_code") and part.executable_code:
                full_text += f"[CODE]\n{part.executable_code.code}\n[/CODE]\n"
            if hasattr(part, "code_execution_result") and part.code_execution_result:
                full_text += f"[OUTPUT]\n{part.code_execution_result.output}\n[/OUTPUT]\n"

        logger.info("Gemini code execution complete", response_length=len(full_text))
        return _parse_result_from_response(full_text)

    except ImportError:
        logger.error("google-generativeai not installed — run: pip install google-generativeai")
        return {"status": "error", "reason": "google-generativeai not installed"}
    except Exception as e:
        logger.error("Gemini call failed", error=str(e))
        return {"status": "error", "reason": str(e)}


async def train_with_gemini(
    ohlcv_df: pd.DataFrame,
    symbol: str = "BTC-USD",
    interval: str = "1d",
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """
    Train a model using Gemini's free code execution cloud.
    Downloads data, sends to Gemini, returns result with metrics.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY_1") or ""
    if not api_key:
        return {"status": "error", "reason": "GEMINI_API_KEY not set"}

    if ohlcv_df is None or len(ohlcv_df) < 100:
        return {"status": "error", "reason": "insufficient data (need ≥100 bars)"}

    csv_data = _sanitize_data(ohlcv_df)
    prompt = _build_training_prompt(csv_data, symbol, interval)

    logger.info("Starting Gemini cloud training", symbol=symbol, interval=interval,
                rows=min(len(ohlcv_df), MAX_ROWS_FOR_GEMINI))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _call_gemini_with_code_execution, prompt, api_key
    )

    if result.get("status") == "success":
        exp_name = experiment_name or f"gemini_{symbol.lower().replace('-','_')}_{interval}"
        result["experiment_name"] = exp_name
        result["trained_at"] = datetime.now(timezone.utc).isoformat()
        result["model"] = "gemini_code_execution"
        result["symbol"] = symbol
        result["interval"] = interval

        # Save result JSON
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{exp_name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info("Gemini training result saved", path=str(out_path),
                    sharpe=result.get("test_sharpe"), accuracy=result.get("test_accuracy"))

    return result


async def train_all_symbols() -> list[dict]:
    """Train models for all standard symbols using Gemini."""
    import yfinance as yf

    targets = [
        ("BTC-USD", "1d"),
        ("ETH-USD", "1d"),
        ("SPY", "1d"),
        ("QQQ", "1d"),
        ("SOL-USD", "1d"),
    ]
    results = []
    for symbol, interval in targets:
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=730)
            df = yf.download(symbol, start=str(start.date()), end=str(end.date()),
                             interval=interval, auto_adjust=True, progress=False)
            if df is None or len(df) < 100:
                logger.warning("Insufficient data", symbol=symbol)
                continue
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            result = await train_with_gemini(df, symbol=symbol, interval=interval)
            results.append(result)
            # Rate limit: 1 request per 2 seconds to stay within free tier
            await asyncio.sleep(2)
        except Exception as e:
            logger.error("Training failed", symbol=symbol, error=str(e))
            results.append({"status": "error", "symbol": symbol, "reason": str(e)})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Free Cloud ML Training")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading symbol")
    parser.add_argument("--interval", default="1d", help="Bar interval")
    parser.add_argument("--all", action="store_true", help="Train all standard symbols")
    parser.add_argument("--config", help="Path to experiment YAML config")
    args = parser.parse_args()

    async def main():
        if args.all:
            results = await train_all_symbols()
            for r in results:
                print(json.dumps(r, indent=2))
        elif args.config:
            import yaml
            cfg_path = Path(args.config)
            if not cfg_path.is_absolute():
                cfg_path = Path(__file__).parents[4] / "experiments" / "configs" / args.config
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            exp = cfg.get("experiment", {})
            symbol = exp.get("symbol", args.symbol)
            interval = exp.get("interval", args.interval)
            import yfinance as yf
            data_cfg = cfg.get("data", {})
            df = yf.download(symbol, start=data_cfg.get("train_start", "2021-01-01"),
                             end=data_cfg.get("test_end", "2024-12-31"),
                             interval=interval, auto_adjust=True, progress=False)
            if df is not None and len(df) >= 100:
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                result = await train_with_gemini(df, symbol=symbol, interval=interval,
                                                 experiment_name=exp.get("name"))
                print(json.dumps(result, indent=2))
        else:
            import yfinance as yf
            df = yf.download(args.symbol, period="2y", interval=args.interval,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) >= 100:
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                result = await train_with_gemini(df, symbol=args.symbol, interval=args.interval)
                print(json.dumps(result, indent=2))
            else:
                print(json.dumps({"status": "error", "reason": "no data downloaded"}))

    asyncio.run(main())
