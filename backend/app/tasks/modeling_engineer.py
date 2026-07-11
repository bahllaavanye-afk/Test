"""
Principal Modeling Engineer — autonomous agent that runs ML experiments,
monitors model performance, triggers retraining when performance degrades,
and promotes best models to production.

Loop (every 1800s):
  1. Check active model performance (accuracy, prediction quality)
  2. Detect model drift (if accuracy < threshold for N consecutive checks)
  3. Trigger retraining for drifted models
  4. Run grid search on best-performing configs
  5. Promote champion models that beat incumbents
  6. Log all decisions to modeling_log.jsonl
"""
from __future__ import annotations

import asyncio
import json
import random
import itertools
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.utils.logging import logger

MODELING_LOG = Path(__file__).parents[3] / "experiments" / "modeling_log.jsonl"
MODELING_LOG.parent.mkdir(parents=True, exist_ok=True)

MODEL_TYPES = ["lstm", "xgboost", "lorentzian", "ensemble"]

# Baseline Sharpe thresholds per model type (based on historical paper-trading)
INCUMBENT_SHARPE: dict[str, float] = {
    "lstm": 1.2,
    "xgboost": 0.9,
    "lorentzian": 0.8,
    "ensemble": 1.5,
}

# Hyperparameter search spaces per model type
HYPERPARAM_SPACES: dict[str, dict] = {
    "lstm": {
        "hidden_size": [64, 128, 256],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "sequence_length": [20, 30, 60],
    },
    "xgboost": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "lorentzian": {
        "neighbors_count": [8, 16, 32],
        "feature_count": [4, 5, 6],
        "max_bars_back": [2000, 2500, 3000],
    },
    "ensemble": {
        "lstm_weight": [0.3, 0.4, 0.5],
        "xgboost_weight": [0.3, 0.4, 0.5],
        "lorentzian_weight": [0.1, 0.2, 0.3],
    },
}


@dataclass
class ModelPerformanceRecord:
    model_id: str
    accuracy: float          # fraction of correct directional predictions
    sharpe: float            # rolling Sharpe of model-guided returns
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    n_predictions: int = 0
    drift_detected: bool = False


@dataclass
class ModelingDecision:
    decision_type: Literal["retrain", "promote", "demote", "monitor"]
    model_id: str
    reason: str
    confidence: float        # 0-1
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ModelingEngineer:
    """
    Principal Modeling Engineer: monitors ML models, triggers retraining,
    promotes champion configs, and runs hyperparameter sweeps.
    Runs as a background asyncio task every 30 minutes.
    """

    def __init__(
        self,
        interval_seconds: int = 1800,
        drift_threshold: float = 0.52,
        retrain_after_n_drift: int = 3,
    ):
        self.interval_seconds = interval_seconds
        self.drift_threshold = drift_threshold
        self.retrain_after_n_drift = retrain_after_n_drift

        self._cycle = 0
        self._decisions: list[ModelingDecision] = []

        # Rolling window of performance records per model_id
        self._perf_cache: dict[str, deque[ModelPerformanceRecord]] = defaultdict(
            lambda: deque(maxlen=10)
        )

        # Consecutive drift count per model_id
        self._drift_counts: dict[str, int] = defaultdict(int)

        # Best known Sharpe per model (starts from incumbents)
        self._best_sharpe: dict[str, float] = dict(INCUMBENT_SHARPE)

        # Best known hyperparams per model
        self._best_params: dict[str, dict] = {}

        # Cache of evaluated hyperparameter combinations per model to avoid recomputation
        self._evaluated_combos: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run forever."""
        logger.info("ModelingEngineer started", interval=self.interval_seconds)
        while True:
            try:
                await self._engineering_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:  # pragma: no cover
                logger.error(f"ModelingEngineer cycle failed: {e}")
            await asyncio.sleep(self.interval_seconds)

    async def _engineering_cycle(self) -> None:
        """One full cycle: check all models, detect drift, retrain if needed."""
        self._cycle += 1
        logger.info("ModelingEngineer: starting cycle", cycle=self._cycle)

        for model_type in MODEL_TYPES:
            model_id = model_type  # simple 1:1 mapping for now

            # 1. Check current performance
            record = await self.check_model_performance(model_id)
            self._perf_cache[model_id].append(record)

            # 2. Detect drift
            drifted = await self.detect_drift(model_id)
            if drifted:
                self._drift_counts[model_id] += 1
                logger.warning(
                    "ModelingEngineer: drift detected",
                    model=model_id,
                    consecutive=self._drift_counts[model_id],
                    accuracy=round(record.accuracy, 3),
                )
            else:
                self._drift_counts[model_id] = 0  # reset on good performance

            # 3. Trigger retraining if drift persists
            if self._drift_counts[model_id] >= self.retrain_after_n_drift:
                await self.trigger_retraining(model_id)
                self._drift_counts[model_id] = 0  # reset after scheduling retrain

            # 4. Evaluate champion
            decision = await self.evaluate_champion(model_id, record)
            self._decisions.append(decision)
            self._log_decision(decision)

        # 5. Run one hyperparameter sweep per cycle (rotate through model types)
        sweep_model = MODEL_TYPES[(self._cycle - 1) % len(MODEL_TYPES)]
        await self.run_hyperparameter_sweep(sweep_model)

        logger.info(
            "ModelingEngineer: cycle complete",
            cycle=self._cycle,
            models_checked=len(MODEL_TYPES),
            retrain_pending=[m for m, c in self._drift_counts.items() if c > 0],
        )

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    async def check_model_performance(
        self, model_id: str | None, n_recent: int = 100
    ) -> ModelPerformanceRecord:
        """
        Pull recent predictions from DB and compute accuracy + Sharpe.
        In production this queries the predictions table.
        Currently uses heuristic simulation based on known model quality.

        Handles None or empty model_id by returning a neutral record.
        """
        if not model_id:
            logger.warning("check_model_performance called with empty model_id")
            return ModelPerformanceRecord(
                model_id="unknown",
                accuracy=0.0,
                sharpe=0.0,
                n_predictions=n_recent,
                drift_detected=True,
            )

        # Simulate realistic accuracy distribution per model type
        base_accuracy = {
            "lstm": 0.56,
            "xgboost": 0.54,
            "lorentzian": 0.53,
            "ensemble": 0.58,
        }.get(model_id, 0.52)

        # Add realistic noise
        noise = random.gauss(0, 0.025)
        accuracy = max(0.40, min(0.75, base_accuracy + noise))

        # Sharpe roughly correlated with accuracy edge
        accuracy_edge = accuracy - 0.50
        sharpe = max(-0.5, accuracy_edge * 20 + random.gauss(0, 0.15))

        drift_detected = accuracy < self.drift_threshold

        return ModelPerformanceRecord(
            model_id=model_id,
            accuracy=round(accuracy, 4),
            sharpe=round(sharpe, 4),
            n_predictions=n_recent,
            drift_detected=drift_detected,
        )

    async def detect_drift(self, model_id: str | None) -> bool:
        """
        Determine if recent performance indicates drift.
        Returns False for None/unknown model_id or empty performance history.
        """
        if not model_id:
            logger.warning("detect_drift called with empty model_id")
            return False

        recent_records = self._perf_cache.get(model_id)
        if not recent_records:
            logger.debug("No performance records for model_id=%s", model_id)
            return False

        # Use the latest record's drift flag
        latest = recent_records[-1]
        return latest.drift_detected

    async def trigger_retraining(self, model_id: str | None) -> None:
        """
        Initiate retraining pipeline for the given model.
        Safely no‑ops if model_id is None or unknown.
        """
        if not model_id:
            logger.error("trigger_retraining called with empty model_id")
            return

        logger.info("Triggering retraining", model=model_id)
        # Placeholder for actual retraining logic; in production this would enqueue a job.
        await asyncio.sleep(0)  # simulate async context switch

    async def evaluate_champion(
        self,
        model_id: str | None,
        record: ModelPerformanceRecord | None,
    ) -> ModelingDecision:
        """
        Compare current performance against best known Sharpe.
        Handles missing inputs gracefully and defaults to monitoring.
        """
        if not model_id or not record:
            logger.warning("evaluate_champion received None inputs")
            return ModelingDecision(
                decision_type="monitor",
                model_id=model_id or "unknown",
                reason="Missing data",
                confidence=0.0,
            )

        current_sharpe = record.sharpe
        best_sharpe = self._best_sharpe.get(model_id, float("-inf"))

        if current_sharpe > best_sharpe:
            self._best_sharpe[model_id] = current_sharpe
            decision_type = "promote"
            reason = f"Improved Sharpe {best_sharpe:.2f}->{current_sharpe:.2f}"
            confidence = min(1.0, (current_sharpe - best_sharpe) / max(0.1, best_sharpe))
        else:
            decision_type = "monitor"
            reason = f"No improvement (best {best_sharpe:.2f}, current {current_sharpe:.2f})"
            confidence = max(0.0, 1 - (best_sharpe - current_sharpe) / max(0.1, best_sharpe))

        return ModelingDecision(
            decision_type=decision_type,
            model_id=model_id,
            reason=reason,
            confidence=round(confidence, 3),
        )

    async def run_hyperparameter_sweep(self, model_type: str | None) -> None:
        """
        Iterate over the hyperparameter grid for a given model type.
        Safely handles unknown or empty model_type and empty search spaces.
        """
        if not model_type:
            logger.error("run_hyperparameter_sweep called with empty model_type")
            return

        space = HYPERPARAM_SPACES.get(model_type)
        if not space:
            logger.warning("No hyperparameter space defined for model_type=%s", model_type)
            return

        # Generate all possible combos; guard against empty dimensions
        keys, values = zip(*[(k, v) for k, v in space.items() if v])
        if not keys:
            logger.warning("Hyperparameter space for %s contains no values", model_type)
            return

        for combo in itertools.product(*values):
            combo_dict = dict(zip(keys, combo))
            combo_id = json.dumps(combo_dict, sort_keys=True)

            if combo_id in self._evaluated_combos[model_type]:
                continue  # skip already evaluated

            # Simulate evaluation; in production this would train/evaluate the model.
            logger.debug("Evaluating combo", model=model_type, combo=combo_dict)
            await asyncio.sleep(0)  # async placeholder

            self._evaluated_combos[model_type].add(combo_id)

        # Optionally store the best combo (placeholder logic)
        if self._evaluated_combos[model_type]:
            self._best_params[model_type] = json.loads(
                min(self._evaluated_combos[model_type])
            )

    def _log_decision(self, decision: ModelingDecision | None) -> None:
        """Append a single decision to the persistent JSON lines log."""
        if not decision:
            logger.error("Attempted to log a None decision")
            return

        try:
            with MODELING_LOG.open("a", encoding="utf-8") as f:
                json.dump(asdict(decision), f)
                f.write("\n")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to write modeling decision: {e}")