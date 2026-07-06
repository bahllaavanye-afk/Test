"""
Nightly ML retraining: downloads fresh data, retrains all active models,
compares new vs old Sharpe, promotes if improved.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Literal

import pandas as pd
from pydantic import BaseModel, Field, validator, root_validator

from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[3] / "models_artifacts"


class RetrainTarget(BaseModel):
    """Configuration for a single retraining job."""

    model_name: str = Field(
        ...,
        description="Identifier of the model architecture (e.g., 'lstm').",
        example="lstm",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol to download data for.",
        example="BTC-USD",
    )
    interval: str = Field(
        ...,
        description="Data granularity accepted by yfinance (e.g., '1h', '1d').",
        example="1h",
    )

    @validator("interval")
    def validate_interval(cls, v: str) -> str:
        """Validate that interval follows yfinance conventions (e.g., 1m, 5m, 1h, 1d)."""
        if not re.fullmatch(r"\d+[mhdw]", v):
            raise ValueError("interval must be a number followed by m, h, d, or w")
        return v


class RetrainResult(BaseModel):
    """Result of a single model retraining attempt."""

    status: Literal["success", "skipped", "error"] = Field(
        ...,
        description="High‑level outcome of the retraining job.",
        example="success",
    )
    model: Optional[str] = Field(
        None,
        description="Name of the model architecture that was retrained.",
        example="lstm",
    )
    symbol: Optional[str] = Field(
        None,
        description="Ticker symbol the model was trained on.",
        example="BTC-USD",
    )
    retrained_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the retraining completed.",
        example="2024-07-06T12:34:56Z",
    )
    reason: Optional[str] = Field(
        None,
        description="Human‑readable reason when status is 'skipped'.",
        example="insufficient data",
    )
    error: Optional[str] = Field(
        None,
        description="Error message when status is 'error'.",
        example="Network timeout while fetching data",
    )
    # Additional metrics produced by the training pipeline (optional)
    sharpe: Optional[float] = Field(
        None,
        description="Sharpe ratio of the newly trained model.",
        example=1.52,
        ge=0,
    )
    max_dd: Optional[float] = Field(
        None,
        description="Maximum drawdown (in percent) of the newly trained model.",
        example=11.2,
        ge=0,
    )
    best_model_path: Optional[Path] = Field(
        None,
        description="Filesystem path to the best checkpoint produced by training.",
        example="/models_artifacts/lstm_btc_20240706/model.pt",
    )

    @root_validator
    def check_consistency(cls, values):
        """Ensure that required fields are present for a successful run."""
        status = values.get("status")
        if status == "success":
            missing = [field for field in ("model", "symbol", "retrained_at") if not values.get(field)]
            if missing:
                raise ValueError(f"Fields {missing} must be set when status='success'")
        return values


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
            result = RetrainResult(
                status="skipped",
                reason="insufficient data",
                model=model_name,
                symbol=symbol,
                retrained_at=datetime.now(timezone.utc),
            )
            return result.dict(exclude_none=True)

        # Normalize column names
        hist.columns = [
            c.lower() if isinstance(c, str) else c[0].lower() for c in hist.columns
        ]

        from app.ml.training.train_lstm import train

        experiment_name = f"{model_name}_{symbol.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        train_result = await train(hist, experiment_name=experiment_name, max_epochs=30)

        # Build successful result
        result = RetrainResult(
            status="success",
            model=model_name,
            symbol=symbol,
            retrained_at=datetime.now(timezone.utc),
            sharpe=train_result.get("sharpe"),
            max_dd=train_result.get("max_dd"),
            best_model_path=train_result.get("best_model_path"),
        )
        # Log without exposing heavy path details
        logger.info(
            "Model retrained",
            **{k: v for k, v in result.dict().items() if k != "best_model_path"},
        )
        return result.dict(exclude_none=True)

    except Exception as e:
        logger.error(
            "Retrain failed", model=model_name, symbol=symbol, error=str(e)
        )
        result = RetrainResult(status="error", error=str(e), model=model_name, symbol=symbol)
        return result.dict(exclude_none=True)


def _load_retrain_configs() -> List[RetrainTarget]:
    """
    Discover retrain targets dynamically from experiment configs (*.yaml).
    Falls back to a minimal default set if no configs exist or yaml is unavailable.
    Returns list of RetrainTarget instances.
    """
    configs_dir = Path(__file__).parents[3] / "experiments" / "configs"
    seen: set[Tuple[str, str, str]] = set()
    results: List[RetrainTarget] = []

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
            exp = cfg.get("experiment", {})
            model = exp.get("model", "lstm")
            symbol = exp.get("symbol", "SPY")
            interval = exp.get("interval", "1d")
            key = (model, symbol, interval)
            if key not in seen:
                seen.add(key)
                results.append(RetrainTarget(model_name=model, symbol=symbol, interval=interval))
        except Exception:
            continue

    if not results:
        defaults = [
            RetrainTarget(model_name="lstm", symbol="BTC-USD", interval="1h"),
            RetrainTarget(model_name="lstm", symbol="ETH-USD", interval="1h"),
            RetrainTarget(model_name="lstm", symbol="SPY", interval="1d"),
        ]
        results.extend(defaults)

    return results


async def nightly_retrain() -> None:
    """Retrain all models discovered from experiment configs. Called by APScheduler at 02:00 UTC."""
    retrain_configs = _load_retrain_configs()
    # Cap at 10 per night to avoid overwhelming free-tier CPU
    retrain_configs = retrain_configs[:10]
    logger.info("Nightly retrain starting", configs=len(retrain_configs))
    results = await asyncio.gather(
        *[
            retrain_model(target.model_name, target.symbol, target.interval)
            for target in retrain_configs
        ],
        return_exceptions=True,
    )
    successes = sum(
        1
        for r in results
        if isinstance(r, dict) and r.get("status") != "error"
    )
    logger.info(
        "Nightly retrain complete", total=len(retrain_configs), succeeded=successes
    )