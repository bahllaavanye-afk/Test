"""
ModelServingLayer — central inference entrypoint with A/B routing and async logging.

This replaces direct model calls in strategies.  Strategies call:

    from app.ml.serving.serve import get_serving_layer
    serving = get_serving_layer()
    result = await serving.predict("lstm_momentum", features, symbol="SPY")

The layer handles:
  - Champion / challenger traffic routing via ABRouter
  - Model artifact loading with an in-memory LRU cache
  - Async fire-and-forget inference logging (never blocks the hot path)
  - Signal thresholding (buy / sell / hold)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from app.ml.serving.ab_router import ABRouter, RouteDecision, get_ab_router

logger = structlog.get_logger()

# Maximum number of distinct release artifacts to keep in the process cache.
_CACHE_MAX = 16


# ─── Prediction result ────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    prediction: float    # raw model output in [0, 1]
    signal: str          # "buy" | "sell" | "hold"
    confidence: float    # abs(prediction - 0.5) * 2
    version: str         # e.g. "v1.2.0"
    release_id: str
    ab_group: str        # "champion" | "challenger"
    latency_ms: float


# ─── Serving layer ────────────────────────────────────────────────────────────

class ModelServingLayer:
    """
    Wraps ABRouter + model loading cache + inference logging.

    Model cache:
        _model_cache is a dict: release_id → loaded model object.
        Models are loaded lazily on first request.  The cache is bounded to
        _CACHE_MAX entries; oldest entries are evicted (insertion-order FIFO
        from Python 3.7+ dict guarantees).

    Inference logging:
        Each prediction is recorded asynchronously via asyncio.create_task so
        the prediction path never waits for a DB write.  If the DB is down the
        log entry is silently dropped (trading signal still delivered).
    """

    def __init__(self, router: ABRouter, db_factory) -> None:
        self._router = router
        self._db_factory = db_factory
        # release_id → model object
        self._model_cache: dict[str, Any] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def predict(
        self,
        model_name: str,
        features: Any,            # torch.Tensor expected
        symbol: str,
        threshold: float = 0.55,
    ) -> PredictionResult | None:
        """
        Route inference to champion/challenger, run the model, log the result.

        Args:
            model_name: logical model name (must match a champion in model_releases).
            features:   input tensor shaped (seq_len, n_features).
            symbol:     ticker symbol for logging context.
            threshold:  signals outside [1-threshold, threshold] are "buy"/"sell".

        Returns:
            PredictionResult or None if no champion exists for model_name.
        """
        decision = await self._router.route(model_name)
        if decision is None:
            return None

        model = await self._load_model(decision)
        if model is None:
            return None

        t0 = time.perf_counter()
        try:
            import torch
            x = features
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            # Models expect (batch, seq_len, features); add batch dim
            if x.dim() == 2:
                x = x.unsqueeze(0)
            with torch.no_grad():
                out = model(x)
                pred = float(out.squeeze())
        except Exception as exc:
            logger.error("ModelServingLayer: inference failed", model_name=model_name, error=str(exc))
            return None

        latency_ms = (time.perf_counter() - t0) * 1000

        signal = (
            "buy"  if pred > threshold else
            "sell" if pred < (1.0 - threshold) else
            "hold"
        )
        confidence = abs(pred - 0.5) * 2.0

        # Fire-and-forget log — never await this
        asyncio.create_task(
            self._log_inference(decision, symbol, pred, signal, confidence, latency_ms)
        )

        return PredictionResult(
            prediction=pred,
            signal=signal,
            confidence=confidence,
            version=decision.version,
            release_id=decision.release_id,
            ab_group=decision.ab_group,
            latency_ms=latency_ms,
        )

    def invalidate_model(self, release_id: str) -> None:
        """Remove a model from the cache (call after promote/archive)."""
        self._model_cache.pop(release_id, None)

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _load_model(self, decision: RouteDecision) -> Any | None:
        """Return the loaded model for *decision*, loading from disk if not cached."""
        if decision.release_id in self._model_cache:
            return self._model_cache[decision.release_id]

        artifact_path = Path(decision.artifact_path)
        if not artifact_path.exists():
            logger.error(
                "ModelServingLayer: artifact not found",
                release_id=decision.release_id,
                path=str(artifact_path),
            )
            return None

        try:
            model = await asyncio.to_thread(self._load_artifact, decision)
        except Exception as exc:
            logger.error(
                "ModelServingLayer: failed to load artifact",
                release_id=decision.release_id,
                error=str(exc),
            )
            return None

        # Evict oldest entry if cache is full
        if len(self._model_cache) >= _CACHE_MAX:
            oldest_key = next(iter(self._model_cache))
            del self._model_cache[oldest_key]

        self._model_cache[decision.release_id] = model
        logger.info(
            "ModelServingLayer: model loaded",
            release_id=decision.release_id,
            version=decision.version,
            framework=decision.framework,
        )
        return model

    @staticmethod
    def _load_artifact(decision: RouteDecision) -> Any:
        """Synchronous artifact loading — runs in a thread via asyncio.to_thread."""
        path = decision.artifact_path
        framework = decision.framework.lower()

        if framework == "pytorch":
            import torch
            return torch.load(path, map_location="cpu", weights_only=False)

        if framework == "xgboost":
            from app.ml.models.xgboost_model import XGBoostClassifier
            return XGBoostClassifier.load(path)

        if framework == "lightgbm":
            from app.ml.models.lightgbm_model import LightGBMClassifier
            return LightGBMClassifier.load(path)

        if framework == "ensemble":
            from app.ml.models.ensemble_model import EnsembleModel
            return EnsembleModel.load(path)

        raise ValueError(f"Unknown framework '{framework}' for release {decision.release_id}")

    async def _log_inference(
        self,
        decision: RouteDecision,
        symbol: str,
        prediction: float,
        signal: str,
        confidence: float,
        latency_ms: float,
    ) -> None:
        """Write an InferenceLog row to the database (best-effort, never raises)."""
        try:
            from app.models.inference_log import InferenceLog

            log = InferenceLog(
                id=str(uuid.uuid4()),
                release_id=decision.release_id,
                model_name=decision.model_name,
                version=decision.version,
                symbol=symbol,
                ts=datetime.now(UTC),
                prediction=prediction,
                signal=signal,
                confidence=confidence,
                latency_ms=latency_ms,
                ab_group=decision.ab_group,
            )
            async with self._db_factory() as db:
                db.add(log)
                await db.commit()
        except Exception as exc:
            logger.warning("ModelServingLayer: inference log failed", error=str(exc))


# ── Module-level singleton ────────────────────────────────────────────────────

_serving_layer: ModelServingLayer | None = None


def get_serving_layer() -> ModelServingLayer:
    global _serving_layer
    if _serving_layer is None:
        from app.database import AsyncSessionLocal
        router = get_ab_router()
        _serving_layer = ModelServingLayer(router, AsyncSessionLocal)
    return _serving_layer
