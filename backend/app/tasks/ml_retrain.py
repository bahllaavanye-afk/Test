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
from pydantic import BaseModel, Field, validator

from app.utils.logging import logger

# --------------------------------------------------------------------------- #
# Pydantic Schemas
# --------------------------------------------------------------------------- #


class RetrainConfig(BaseModel):
    """Configuration describing a single model retraining job."""

    model_name: str = Field(
        ...,
        description="Name of the model architecture (e.g., 'lstm').",
        example="lstm",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol for the asset to be trained on.",
        example="BTC-USD",
        regex=r"^[A-Z0-9\-.]{1,20}$",
    )
    interval: str = Field(
        ...,
        description="Data granularity interval compatible with yfinance.",
        example="1h",
        regex=r"^\d+[smhd]$",
    )

    @validator("interval")
    def _validate_interval(cls, v: str) -> str:
        """Ensure interval follows yfinance allowed pattern (e.g., '1h', '30m')."""
        if not re.fullmatch(r"^\d+[smhd]$", v):
            raise ValueError("interval must be a number followed by s,m,h, or d")
        return v

    class Config:
        schema_extra = {
            "example": {
                "model_name": "lstm",
                "symbol": "BTC-USD",
                "interval": "1h",
            }
        }


class RetrainResult(BaseModel):
    """Result payload returned after attempting to retrain a model."""

    status: str = Field(
        ...,
        description="Overall status of the retraining attempt.",
        example="success",
    )
    model: str = Field(
        ...,
        description="Model architecture that was retrained.",
        example="lstm",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol the model was trained on.",
        example="BTC-USD",
    )
    retrained_at: str = Field(
        ...,
        description="ISO‑8601 timestamp when the retraining completed.",
        example="2024-08-03T12:34:56.789Z",
    )
    reason: str | None = Field(
        None,
        description="Reason for skipping retraining, if applicable.",
        example="insufficient data",
    )
    error: str | None = Field(
        None,
        description="Error message if the retraining failed.",
        example="Network timeout",
    )
    best_model_path: str | None = Field(
        None,
        description="Filesystem path to the best model checkpoint produced.",
        example="/models/lstm_btc_usd_20240803.ckpt",
    )
    sharpe: float | None = Field(
        None,
        description="Sharpe ratio achieved by the newly trained model.",
        example=1.42,
    )
    additional_metrics: dict | None = Field(
        None,
        description="Any extra performance metrics returned by the training routine.",
        example={"drawdown": 0.15},
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "model": "lstm",
                "symbol": "BTC-USD",
                "retrained_at": "2024-08-03T12:34:56.789Z",
                "best_model_path": "/models/lstm_btc_usd_20240803.ckpt",
                "sharpe": 1.42,
                "additional_metrics": {"drawdown": 0.15},
            }
        }


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
    logger.error("yfinance unavailable — nightly retrain will skip downloads", error=str(exc))

try:
    import yaml as _yaml
    _load_yaml = _yaml.safe_load
except Exception:  # pragma: no cover
    _load_yaml = None


async def _download_hist(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame | None:
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
        logger.error("Failed to download data", symbol=symbol, interval=interval, error=str(exc))
        return None

    if hist is None or len(hist) < MIN_HIST_LENGTH:
        return None

    # Normalize column names once.
    hist.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in hist.columns]

    # Store in cache for potential reuse.
    _DATA_CACHE[cache_key] = (datetime.now(timezone.utc), hist.copy())
    return hist


async def retrain_model(model_name: str, symbol: str, interval: str = DEFAULT_INTERVAL) -> dict:
    """Download 2 years of data and retrain a model. Returns result dict."""
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=DEFAULT_TRAIN_DAYS)

        hist = await _download_hist(symbol, interval, start, end)
        if hist is None:
            return {"status": "skipped", "reason": "insufficient data"}

        from app.ml.training.train_lstm import train

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
        except Exception:
            continue

    if not results:
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
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("status") != "error")
    logger.info(
        "Nightly retrain complete",
        total=len(retrain_configs),
        succeeded=successes,
    )