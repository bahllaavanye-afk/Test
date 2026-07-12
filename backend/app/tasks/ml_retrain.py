"""
Nightly ML retraining: downloads fresh data, retrains all active models,
compares new vs old Sharpe, promotes if improved.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, validator

from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[3] / "models_artifacts"


class RetrainResult(BaseModel):
    """Schema for the result of a model retraining run."""

    status: str = Field(
        ...,
        description="Result status of the retraining operation.",
        examples=["success", "skipped", "error"],
    )
    reason: str | None = Field(
        None,
        description="Reason for skipping the retraining (used when status is 'skipped').",
        examples=["insufficient data"],
    )
    error: str | None = Field(
        None,
        description="Error message if the retraining failed (used when status is 'error').",
        examples=["Connection timeout"],
    )
    symbol: str | None = Field(
        None,
        description="Ticker symbol that was retrained.",
        examples=["BTC-USD"],
    )
    model: str | None = Field(
        None,
        description="Name of the model that was retrained.",
        examples=["lstm"],
    )
    retrained_at: datetime | None = Field(
        None,
        description="UTC timestamp when the retraining completed.",
        examples=["2023-10-12T08:15:30Z"],
    )

    @validator("status")
    def _validate_status(cls, v: str) -> str:
        allowed = {"success", "skipped", "error"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v

    class Config:
        extra = "allow"


class RetrainConfig(BaseModel):
    """Schema describing a single retraining target."""

    model_name: str = Field(
        ...,
        description="Identifier of the model to be retrained.",
        examples=["lstm"],
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol for which data will be fetched.",
        examples=["BTC-USD"],
    )
    interval: str = Field(
        ...,
        description="Data granularity interval accepted by yfinance.",
        examples=["1h"],
    )


async def retrain_model(model_name: str, symbol: str, interval: str = "1h") -> dict:
    """Download 2 years of data and retrain a model. Returns result dict."""
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
            result = {"status": "skipped", "reason": "insufficient data"}
            # Validate schema before returning
            validated = RetrainResult(**result)
            return validated.dict()

        # Normalize column names
        hist.columns = [
            c.lower() if isinstance(c, str) else c[0].lower() for c in hist.columns
        ]

        from app.ml.training.train_lstm import train

        experiment_name = f"{model_name}_{symbol.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        result = await train(hist, experiment_name=experiment_name, max_epochs=30)
        result["symbol"] = symbol
        result["model"] = model_name
        result["retrained_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Model retrained", **{k: v for k, v in result.items() if k != "best_model_path"})

        # Validate the result against the schema; any validation errors are logged and turned into an error dict
        try:
            validated = RetrainResult(**result)
            return validated.dict()
        except ValidationError as ve:
            logger.error(
                "Retrain result validation failed",
                model=model_name,
                symbol=symbol,
                error=str(ve),
            )
            return {"status": "error", "error": str(ve)}

    except Exception as e:
        logger.error("Retrain failed", model=model_name, symbol=symbol, error=str(e))
        return {"status": "error", "error": str(e)}


def _load_retrain_configs() -> List[RetrainConfig]:
    """
    Discover retrain targets dynamically from experiment configs (*.yaml).
    Falls back to a minimal default set if no configs exist or yaml is unavailable.
    Returns list of RetrainConfig objects.
    """
    configs_dir = Path(__file__).parents[3] / "experiments" / "configs"
    seen: set[tuple[str, str, str]] = set()
    results: List[RetrainConfig] = []

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
                    # Minimal fallback: regex‑extract model/symbol/interval from YAML text
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
                results.append(RetrainConfig(model_name=model, symbol=symbol, interval=interval))
        except Exception:
            continue

    if not results:
        results = [
            RetrainConfig(model_name="lstm", symbol="BTC-USD", interval="1h"),
            RetrainConfig(model_name="lstm", symbol="ETH-USD", interval="1h"),
            RetrainConfig(model_name="lstm", symbol="SPY", interval="1d"),
        ]

    return results


async def nightly_retrain() -> None:
    """Retrain all models discovered from experiment configs. Called by APScheduler at 02:00 UTC."""
    retrain_configs = _load_retrain_configs()
    # Cap at 10 per night to avoid overwhelming free-tier CPU
    retrain_configs = retrain_configs[:10]
    logger.info("Nightly retrain starting", configs=len(retrain_configs))
    results = await asyncio.gather(
        *[
            retrain_model(cfg.model_name, cfg.symbol, cfg.interval)
            for cfg in retrain_configs
        ],
        return_exceptions=True,
    )
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("status") != "error")
    logger.info(
        "Nightly retrain complete",
        total=len(retrain_configs),
        succeeded=successes,
    )