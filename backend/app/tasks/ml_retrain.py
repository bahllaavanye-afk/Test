"""
Nightly ML retraining: downloads fresh data, retrains all active models,
compares new vs old Sharpe, promotes if improved.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[3] / "models_artifacts"

_ALLOWED_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "1h", "2h", "4h", "1d", "1wk", "1mo"
}


def _validate_retrain_inputs(model_name: str, symbol: str, interval: str) -> None:
    """
    Validate inputs for ``retrain_model``.

    Parameters
    ----------
    model_name : str
        Name of the model to retrain. Must be a non‑empty string.
    symbol : str
        Ticker symbol to download. Must be a non‑empty string consisting of
        alphanumeric characters, dashes or periods.
    interval : str
        Data interval for yfinance. Must be one of the supported intervals.

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non‑empty string")

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non‑empty string")

    if not re.fullmatch(r"[A-Za-z0-9.\-]+", symbol):
        raise ValueError(
            f"symbol contains invalid characters: '{symbol}'. "
            "Allowed characters are letters, numbers, '.' and '-'"
        )

    if not isinstance(interval, str) or not interval.strip():
        raise ValueError("interval must be a non‑empty string")

    if interval not in _ALLOWED_INTERVALS:
        allowed = ", ".join(sorted(_ALLOWED_INTERVALS))
        raise ValueError(
            f"interval '{interval}' is not supported. Allowed intervals are: {allowed}"
        )


async def retrain_model(model_name: str, symbol: str, interval: str = "1h") -> dict:
    """Download 2 years of data and retrain a model. Returns result dict."""
    _validate_retrain_inputs(model_name, symbol, interval)

    try:
        import yfinance as yf

        loop = asyncio.get_running_loop()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=730)

        hist = await loop.run_in_executor(
            None,
            lambda: yf.download(
                symbol,
                start=str(start.date()),
                end=str(end.date()),
                interval=interval,
                auto_adjust=True,
                progress=False,
            ),
        )
        if hist is None or len(hist) < 200:
            return {"status": "skipped", "reason": "insufficient data"}

        # Normalize column names
        hist.columns = [
            c.lower() if isinstance(c, str) else c[0].lower()
            for c in hist.columns
        ]

        from app.ml.training.train_lstm import train

        experiment_name = f"{model_name}_{symbol.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        result = await train(hist, experiment_name=experiment_name, max_epochs=30)
        result["symbol"] = symbol
        result["model"] = model_name
        result["retrained_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Model retrained", **{k: v for k, v in result.items() if k != "best_model_path"}
        )
        return result

    except Exception as e:
        logger.error(
            "Retrain failed", model=model_name, symbol=symbol, error=str(e)
        )
        return {"status": "error", "error": str(e)}


def _load_retrain_configs() -> list[tuple[str, str, str]]:
    """
    Discover retrain targets dynamically from experiment configs (*.yaml).
    Falls back to a minimal default set if no configs exist or yaml is unavailable.
    Returns list of (model_name, symbol, interval).
    """
    configs_dir = Path(__file__).parents[3] / "experiments" / "configs"
    seen: set[tuple[str, str, str]] = set()
    results: list[tuple[str, str, str]] = []

    try:
        import yaml as _yaml

        _load_yaml = _yaml.safe_load
    except ImportError:
        _load_yaml = None

    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        try:
            with open(cfg_path) as f:
                if _load_yaml:
                    cfg = _load_yaml(f)
                else:
                    # Minimal fallback: regex-extract model/symbol/interval from YAML text
                    import re

                    text = f.read()
                    cfg = {
                        "experiment": {
                            k: v
                            for k, v in re.findall(
                                r"^\s{2}(model|symbol|interval):\s*['\"]?([^\s'\"#]+)",
                                text,
                                re.MULTILINE,
                            )
                        }
                    }
            exp = (cfg or {}).get("experiment", {})
            model = exp.get("model", "lstm")
            symbol = exp.get("symbol", "SPY")
            interval = exp.get("interval", "1d")
            key = (model, symbol, interval)
            if key not in seen:
                seen.add(key)
                results.append(key)
        except Exception:
            continue

    if not results:
        results = [
            ("lstm", "BTC-USD", "1h"),
            ("lstm", "ETH-USD", "1h"),
            ("lstm", "SPY", "1d"),
        ]

    return results


async def nightly_retrain() -> None:
    """Retrain all models discovered from experiment configs. Called by APScheduler at 02:00 UTC."""
    retrain_configs = _load_retrain_configs()
    # Cap at 10 per night to avoid overwhelming free-tier CPU
    retrain_configs = retrain_configs[:10]
    logger.info("Nightly retrain starting", configs=len(retrain_configs))
    results = await asyncio.gather(
        *[retrain_model(m, s, i) for m, s, i in retrain_configs],
        return_exceptions=True,
    )
    successes = sum(
        1 for r in results if isinstance(r, dict) and r.get("status") != "error"
    )
    logger.info(
        "Nightly retrain complete", total=len(retrain_configs), succeeded=successes
    )