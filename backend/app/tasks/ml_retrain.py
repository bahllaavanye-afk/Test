"""
Nightly ML retraining: downloads fresh data, retrains all active models,
compares new vs old Sharpe, promotes if improved.
"""
from __future__ import annotations

import asyncio
import json
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
# Confirmation thresholds
MIN_SHARPE: float = 0.5
MAX_DRAWDOWN: float = 0.20

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


def _is_data_quality_ok(df: pd.DataFrame) -> bool:
    """
    Basic quality checks for downloaded price data.
    - Must contain at least MIN_HIST_LENGTH rows after dropping NaNs.
    - No more than 5 % missing values in any column.
    """
    if df.empty:
        return False
    df_clean = df.dropna()
    if len(df_clean) < MIN_HIST_LENGTH:
        return False
    missing_ratio = df.isna().mean().max()
    return missing_ratio <= 0.05


async def _download_hist(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> Optional[pd.DataFrame]:
    """
    Retrieve historical price data, using an in‑process cache to avoid duplicate
    downloads within the same nightly run.

    Returns a pandas DataFrame or ``None`` on failure or insufficient quality.
    """
    cache_key = (symbol, interval)
    cached = _DATA_CACHE.get(cache_key)
    if cached:
        cache_ts, df = cached
        if cache_ts >= end and not df.empty:
            logger.debug("Using cached data for %s %s", symbol, interval)
            return df.copy()

    if yf is None:
        return None

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
        )
        return None

    if hist is None or len(hist) < MIN_HIST_LENGTH:
        return None

    # Normalize column names.
    hist.columns = [
        c.lower() if isinstance(c, str) else c[0].lower()
        for c in hist.columns
    ]

    if not _is_data_quality_ok(hist):
        logger.warning("Data quality check failed for %s %s", symbol, interval)
        return None

    _DATA_CACHE[cache_key] = (datetime.now(timezone.utc), hist.copy())
    return hist


def _existing_sharpe_path(model_name: str, symbol: str) -> Path:
    """
    Returns the path where the historical performance JSON is stored for a
    given model/symbol pair.
    """
    base_dir = Path(__file__).parents[3] / "model_performance"
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_{symbol.lower()}.json"
    return base_dir / filename


def _load_existing_sharpe(model_name: str, symbol: str) -> Optional[float]:
    """
    Load the previously recorded Sharpe ratio for a model/symbol pair.
    Returns ``None`` if not available.
    """
    path = _existing_sharpe_path(model_name, symbol)
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return float(data.get("sharpe", 0))
    except Exception:  # pragma: no cover
        return None


def _save_sharpe(model_name: str, symbol: str, sharpe: float) -> None:
    """
    Persist the Sharpe ratio for a model/symbol pair.
    """
    path = _existing_sharpe_path(model_name, symbol)
    with open(path, "w") as f:
        json.dump({"sharpe": sharpe, "saved_at": datetime.now(timezone.utc).isoformat()}, f)


async def retrain_model(
    model_name: str,
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
) -> dict:
    """Download data, retrain a model, and apply confirmation filters."""
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=DEFAULT_TRAIN_DAYS)

        hist = await _download_hist(symbol, interval, start, end)
        if hist is None:
            return {
                "status": "skipped",
                "reason": "insufficient or low‑quality data",
                "model": model_name,
                "symbol": symbol,
            }

        from app.ml.training.train_lstm import train

        experiment_name = f"{model_name}_{symbol.lower()}_{datetime.now(timezone.utc).strftime(DATE_FORMAT)}"
        result = await train(hist, experiment_name=experiment_name, max_epochs=MAX_EPOCHS)

        # Expected result structure: {"metrics": {"sharpe": ..., "max_drawdown": ...}, ...}
        metrics = result.get("metrics", {})
        sharpe = float(metrics.get("sharpe", 0))
        max_dd = float(metrics.get("max_drawdown", 1))

        # Confirmation filters
        if sharpe < MIN_SHARPE or max_dd > MAX_DRAWDOWN:
            logger.info(
                "Model failed confirmation filters",
                model=model_name,
                symbol=symbol,
                sharpe=sharpe,
                max_drawdown=max_dd,
            )
            return {
                "status": "skipped",
                "reason": "confirmation filters not met",
                "model": model_name,
                "symbol": symbol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
            }

        # Compare with existing Sharpe ratio
        prev_sharpe = _load_existing_sharpe(model_name, symbol)
        if prev_sharpe is not None and sharpe <= prev_sharpe:
            logger.info(
                "New model did not improve Sharpe",
                model=model_name,
                symbol=symbol,
                new_sharpe=sharpe,
                previous_sharpe=prev_sharpe,
            )
            status = "not_promoted"
        else:
            _save_sharpe(model_name, symbol, sharpe)
            status = "promoted"

        result.update(
            {
                "symbol": symbol,
                "model": model_name,
                "retrained_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
            }
        )
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
        return {"status": "error", "error": str(e), "model": model_name, "symbol": symbol}


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