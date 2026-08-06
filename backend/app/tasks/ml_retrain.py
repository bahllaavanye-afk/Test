"""
Nightly ML retraining: downloads fresh data, retrains all active models,
compares new vs old Sharpe, promotes if improved.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

from app.utils.logging import logger

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_INTERVAL: str = "1h"
MIN_HIST_LENGTH: int = 200
MAX_EPOCHS: int = 30
DEFAULT_TRAIN_DAYS: int = 730
CONFIGS_DIR: Path = Path(__file__).parents[3] / "experiments" / "configs"
DEFAULT_RETRAIN_CONFIGS: List[Tuple[str, str, str]] = [
    ("lstm", "BTC-USD", "1h"),
    ("lstm", "ETH-USD", "1h"),
    ("lstm", "SPY", "1d"),
]
MAX_RETRAIN_PER_NIGHT: int = 10
DATE_FORMAT: str = "%Y%m%d"
REGEX_EXTRACT_PATTERN: str = r"^\s{2}(model|symbol|interval):\s*['\"]?([^\s'\"#]+)"

# --------------------------------------------------------------------------- #
# Global in‑process cache for downloaded price data.
# Key: (symbol, interval) -> (timestamp, DataFrame)
# The cache lives only for the duration of the nightly job, avoiding repeated
# network calls when multiple models share the same symbol/interval.
# --------------------------------------------------------------------------- #
_DATA_CACHE: Dict[Tuple[str, str], Tuple[datetime, pd.DataFrame]] = {}

# --------------------------------------------------------------------------- #
# Import heavy or optional dependencies once at module load time.
# --------------------------------------------------------------------------- #
try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    # Optional dependency: absent in CI and slim deploys. A hard raise here
    # crashes the scheduler at import — degrade instead: retrain simply skips
    # data downloads (matches how torch-backed models degrade elsewhere).
    yf = None
    logger.error(
        "yfinance unavailable — nightly retrain will skip downloads",
        error=str(exc),
        exc_info=exc,
    )

try:
    import yaml as _yaml
    _load_yaml = _yaml.safe_load
except Exception as exc:  # pragma: no cover
    _load_yaml = None
    logger.error(
        "yaml library unavailable — falling back to regex parsing",
        error=str(exc),
        exc_info=exc,
    )


async def _download_hist(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame | None:
    """
    Retrieve historical price data, using an in‑process cache to avoid duplicate
    downloads within the same nightly run.

    Returns a pandas DataFrame or ``None`` on failure.
    """
    cache_key = (symbol, interval)
    cached = _DATA_CACHE.get(cache_key)
    if cached:
        cache_ts, df = cached
        # Cached data is considered fresh if it covers the requested date range.
        if cache_ts >= end and not df.empty:
            logger.debug("Using cached data for %s %s", symbol, interval)
            return df.copy()

    if yf is None:
        logger.warning(
            "yfinance not available; cannot download data",
            symbol=symbol,
            interval=interval,
        )
        return None  # yfinance not installed (CI / slim deploy) — skip download

    loop = asyncio.get_running_loop()
    try:
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
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to download data",
            symbol=symbol,
            interval=interval,
            error=str(exc),
            exc_info=exc,
        )
        return None

    if hist is None or len(hist) < MIN_HIST_LENGTH:
        logger.warning(
            "Insufficient historical data returned",
            symbol=symbol,
            interval=interval,
            rows=len(hist) if hist is not None else 0,
        )
        return None

    # Normalize column names once.
    hist.columns = [
        c.lower() if isinstance(c, str) else c[0].lower() for c in hist.columns
    ]

    # Store in cache for potential reuse.
    _DATA_CACHE[cache_key] = (datetime.now(timezone.utc), hist.copy())
    return hist


async def retrain_model(
    model_name: str,
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
) -> dict:
    """Download 2 years of data and retrain a model. Returns result dict."""
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=DEFAULT_TRAIN_DAYS)

        hist = await _download_hist(symbol, interval, start, end)
        if hist is None:
            return {"status": "skipped", "reason": "insufficient data"}

        try:
            from app.ml.training.train_lstm import train
        except ImportError as exc:
            logger.error(
                "Training module missing",
                model=model_name,
                symbol=symbol,
                error=str(exc),
                exc_info=exc,
            )
            return {"status": "error", "error": "training module unavailable"}

        experiment_name = f"{model_name}_{symbol.lower()}_{datetime.now(timezone.utc).strftime(DATE_FORMAT)}"
        result = await train(hist, experiment_name=experiment_name, max_epochs=MAX_EPOCHS)

        result["symbol"] = symbol
        result["model"] = model_name
        result["retrained_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Model retrained",
            **{k: v for k, v in result.items() if k != "best_model_path"},
        )
        return result

    except Exception as e:  # pragma: no cover
        logger.error(
            "Retrain failed",
            model=model_name,
            symbol=symbol,
            error=str(e),
            exc_info=e,
        )
        return {"status": "error", "error": str(e)}


def _load_retrain_configs() -> List[Tuple[str, str, str]]:
    """
    Discover retrain targets dynamically from experiment configs (*.yaml).
    Falls back to a minimal default set if no configs exist or yaml is unavailable.
    Returns list of (model_name, symbol, interval).
    """
    configs_dir = CONFIGS_DIR
    seen: set[Tuple[str, str, str]] = set()
    results: List[Tuple[str, str, str]] = []

    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                if _load_yaml:
                    try:
                        cfg = _load_yaml(f)
                    except _yaml.YAMLError as exc:
                        logger.error(
                            "YAML parsing error",
                            path=str(cfg_path),
                            error=str(exc),
                            exc_info=exc,
                        )
                        continue
                else:
                    # Minimal fallback: regex‑extract model/symbol/interval from YAML text
                    text = f.read()
                    cfg = {
                        "experiment": {
                            k: v
                            for k, v in re.findall(
                                REGEX_EXTRACT_PATTERN,
                                text,
                                re.MULTILINE,
                            )
                        }
                    }
            exp = cfg.get("experiment", {})
            model = exp.get("model", "lstm")
            symbol = exp.get("symbol", "SPY")
            interval = exp.get("interval", "1d")
            key = (model, symbol, interval)
            if key not in seen:
                seen.add(key)
                results.append(key)
        except FileNotFoundError as exc:
            logger.error(
                "Config file not found",
                path=str(cfg_path),
                error=str(exc),
                exc_info=exc,
            )
            continue
        except PermissionError as exc:
            logger.error(
                "Permission denied when reading config",
                path=str(cfg_path),
                error=str(exc),
                exc_info=exc,
            )
            continue
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Unexpected error loading config",
                path=str(cfg_path),
                error=str(exc),
                exc_info=exc,
            )
            continue

    if not results:
        logger.info("No valid config files found; using default retrain set")
        results = list(DEFAULT_RETRAIN_CONFIGS)

    return results


async def nightly_retrain() -> None:
    """Retrain all models discovered from experiment configs. Called by APScheduler at 02:00 UTC."""
    retrain_configs = _load_retrain_configs()
    # Cap at 10 per night to avoid overwhelming free‑tier CPU
    retrain_configs = retrain_configs[:MAX_RETRAIN_PER_NIGHT]

    if not retrain_configs:
        logger.info("No retrain configurations found")
        return

    logger.info("Nightly retrain starting", configs=len(retrain_configs))
    results = await asyncio.gather(
        *(retrain_model(m, s, i) for m, s, i in retrain_configs),
        return_exceptions=True,
    )
    successes = 0
    errors = 0
    for r in results:
        if isinstance(r, dict):
            if r.get("status") != "error":
                successes += 1
            else:
                errors += 1
        else:
            # An exception object was returned because of return_exceptions=True
            errors += 1
            logger.error(
                "Retrain task raised exception",
                exception=str(r),
                exc_info=r if isinstance(r, BaseException) else None,
            )
    logger.info(
        "Nightly retrain complete",
        total=len(retrain_configs),
        succeeded=successes,
        errors=errors,
    )