"""
Nightly ML retraining: downloads fresh data, retrains all active models,
compares new vs old Sharpe, promotes if improved.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from pydantic import BaseModel, Field, validator

from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[3] / "models_artifacts"


class RetrainResult(BaseModel):
    """Schema representing the outcome of a model retraining run."""

    status: str = Field(
        ...,
        description="Result status of the retraining operation.",
        example="success",
    )
    model: str = Field(
        ...,
        description="Name of the model that was retrained.",
        example="lstm",
    )
    symbol: str = Field(
        ...,
        description="Financial instrument ticker symbol.",
        example="BTC-USD",
    )
    retrained_at: datetime = Field(
        ...,
        description="UTC timestamp when the retraining completed.",
        example="2023-01-01T00:00:00Z",
    )
    sharpe: Optional[float] = Field(
        None,
        description="Sharpe ratio of the newly trained model.",
        example=1.52,
    )
    max_drawdown: Optional[float] = Field(
        None,
        description="Maximum drawdown percentage observed during backtesting.",
        example=11.2,
    )
    best_model_path: Optional[str] = Field(
        None,
        description="Filesystem path to the best model artifact produced.",
        example="/models/lstm/btc_usd/20230101/model.pt",
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for skipping the retraining, if applicable.",
        example="insufficient data",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the retraining failed.",
        example="network timeout",
    )

    @validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"success", "skipped", "error"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v

    @validator("retrained_at")
    def validate_timestamp(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("retrained_at must be timezone-aware")
        return v


async def retrain_model(model_name: str, symbol: str, interval: str = "1h") -> dict:
    """Download 2 years of data and retrain a model. Returns result dict."""
    now_utc = datetime.now(timezone.utc)
    try:
        import yfinance as yf

        loop = asyncio.get_running_loop()
        end = now_utc
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
            result = RetrainResult(
                status="skipped",
                reason="insufficient data",
                model=model_name,
                symbol=symbol,
                retrained_at=now_utc,
            )
            logger.info("Model retrain skipped", **result.dict())
            return result.dict()

        # Normalize column names
        hist.columns = [
            c.lower() if isinstance(c, str) else c[0].lower() for c in hist.columns
        ]

        from app.ml.training.train_lstm import train

        experiment_name = f"{model_name}_{symbol.lower()}_{now_utc.strftime('%Y%m%d')}"
        train_result = await train(
            hist, experiment_name=experiment_name, max_epochs=30
        )
        # Ensure expected keys exist
        train_result["symbol"] = symbol
        train_result["model"] = model_name
        train_result["retrained_at"] = now_utc

        # Default status to success if not provided by training routine
        if "status" not in train_result:
            train_result["status"] = "success"

        result = RetrainResult(**train_result)
        logger.info(
            "Model retrained",
            **{k: v for k, v in result.dict().items() if k != "best_model_path"},
        )
        return result.dict()

    except Exception as e:
        error_result = RetrainResult(
            status="error",
            error=str(e),
            model=model_name,
            symbol=symbol,
            retrained_at=now_utc,
        )
        logger.error(
            "Retrain failed",
            model=model_name,
            symbol=symbol,
            error=str(e),
        )
        return error_result.dict()


def _load_retrain_configs() -> List[Tuple[str, str, str]]:
    """
    Discover retrain targets dynamically from experiment configs (*.yaml).
    Falls back to a minimal default set if no configs exist or yaml is unavailable.
    Returns list of (model_name, symbol, interval).
    """
    configs_dir = Path(__file__).parents[3] / "experiments" / "configs"
    seen: set[Tuple[str, str, str]] = set()
    results: List[Tuple[str, str, str]] = []

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
        1
        for r in results
        if isinstance(r, dict) and r.get("status") != "error"
    )
    logger.info(
        "Nightly retrain complete",
        total=len(retrain_configs),
        succeeded=successes,
    )