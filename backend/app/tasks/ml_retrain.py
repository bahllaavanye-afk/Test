"""
Nightly ML retraining: downloads fresh data, retrains all active models,
compares new vs old Sharpe, promotes if improved.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Tuple, List, Optional

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
    yf = None
    logger.error(
        "yfinance unavailable — nightly retrain will skip downloads",
        error=str(exc),
    )

try:
    import yaml as _yaml
    _load_yaml = _yaml.safe_load
except Exception:  # pragma: no cover
    _load_yaml = None


def _validate_hist(df: pd.DataFrame) -> bool:
    """
    Confirm that the historical DataFrame contains the required columns,
    has no NaNs in price fields, and meets the minimum length requirement.

    Returns True if the data passes all checks.
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols.difference(set(df.columns.str.lower()))
    if missing:
        logger.warning("Missing required columns in downloaded data", missing=missing)
        return False

    if len(df) < MIN_HIST_LENGTH:
        logger.warning(
            "Insufficient history length",
            length=len(df),
            required=MIN_HIST_LENGTH,
        )
        return False

    if df[["open", "high", "low", "close"]].isnull().any().any():
        logger.warning("NaN values detected in price columns")
        return False

    return True


async def _download_hist(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    retries: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Retrieve historical price data, using an in‑process cache to avoid duplicate
    downloads within the same nightly run. Retries on transient failures.

    Returns a pandas DataFrame or ``None`` on failure.
    """
    cache_key = (symbol, interval)
    cached = _DATA_CACHE.get(cache_key)
    if cached:
        cache_ts, df = cached
        if cache_ts >= end and not df.empty:
            logger.debug("Using cached data for %s %s", symbol, interval)
            return df.copy()

    if yf is None:
        return None  # yfinance not installed — skip download

    loop = asyncio.get_running_loop()
    attempt = 0
    while attempt <= retries:
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
            if hist is None or len(hist) < MIN_HIST_LENGTH:
                logger.debug(
                    "Download returned insufficient data",
                    attempt=attempt,
                    symbol=symbol,
                    interval=interval,
                )
                return None
            # Normalise column names
            hist.columns = [
                c.lower() if isinstance(c, str) else c[0].lower()
                for c in hist.columns
            ]
            if not _validate_hist(hist):
                return None
            _DATA_CACHE[cache_key] = (datetime.now(timezone.utc), hist.copy())
            return hist
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to download data",
                symbol=symbol,
                interval=interval,
                attempt=attempt,
                error=str(exc),
            )
            attempt += 1
            await asyncio.sleep(2 ** attempt)  # exponential back‑off
    return None


async def retrain_model(model_name: str, symbol: str, interval: str = DEFAULT_INTERVAL) -> dict:
    """Download 2 years of data and retrain a model. Returns result dict."""
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=DEFAULT_TRAIN_DAYS)

        hist = await _download_hist(symbol, interval, start, end)
        if hist is None:
            return {"status": "skipped", "reason": "insufficient or invalid data"}

        from app.ml.training.train_lstm import train

        experiment_name = f"{model_name}_{symbol.lower()}_{datetime.now(timezone.utc).strftime(DATE_FORMAT)}"
        result = await train(hist, experiment_name=experiment_name, max_epochs=MAX_EPOCHS)

        # Confirmation filter: require a minimum Sharpe improvement before promotion
        new_sharpe = result.get("sharpe")
        old_sharpe = result.get("previous_sharpe")
        if new_sharpe is not None and old_sharpe is not None:
            if new_sharpe <= old_sharpe * 1.01:  # at least 1% improvement
                result["status"] = "skipped"
                result["reason"] = "sharpe_not_improved"
                logger.info(
                    "Model not promoted – Sharpe improvement insufficient",
                    symbol=symbol,
                    model=model_name,
                    old_sharpe=old_sharpe,
                    new_sharpe=new_sharpe,
                )
                return result

        result["symbol"] = symbol
        result["model"] = model_name
        result["retrained_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Model retrained",
            **{k: v for k, v in result.items() if k != "best_model_path"},
        )
        return result

    except Exception as e:
        logger.error(
            "Retrain failed",
            model=model_name,
            symbol=symbol,
            error=str(e),
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
            with open(cfg_path) as f:
                if _load_yaml:
                    cfg = _load_yaml(f)
                else:
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
        except Exception:
            continue

    if not results:
        results = list(DEFAULT_RETRAIN_CONFIGS)

    return results


async def nightly_retrain() -> None:
    """Retrain all models discovered from experiment configs. Called by APScheduler at 02:00 UTC."""
    retrain_configs = _load_retrain_configs()
    retrain_configs = retrain_configs[:MAX_RETRAIN_PER_NIGHT]

    if not retrain_configs:
        logger.info("No retrain configurations found")
        return

    logger.info("Nightly retrain starting", configs=len(retrain_configs))
    results = await asyncio.gather(
        *(retrain_model(m, s, i) for m, s, i in retrain_configs),
        return_exceptions=True,
    )
    successes = sum(
        1
        for r in results
        if isinstance(r, dict) and r.get("status") not in {"error", "skipped"}
    )
    logger.info(
        "Nightly retrain complete",
        total=len(retrain_configs),
        succeeded=successes,
    )