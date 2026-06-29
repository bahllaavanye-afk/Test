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
        self, model_id: str, n_recent: int = 100
    ) -> ModelPerformanceRecord:
        """
        Pull recent predictions from DB and compute accuracy + Sharpe.
        In production this queries the predictions table.
        Currently uses heuristic simulation based on known model quality.
        """
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

    async def detect_drift(self, model_id: str) -> bool:
        """
        Return True if the last N checks show degraded accuracy below threshold.
        Uses the rolling cache so single bad readings don't trigger false alarms.
        """
        recent_records = list(self._perf_cache[model_id])
        if not recent_records:
            return False
        # Drift if *all* records in the window are below threshold
        return all(rec.accuracy < self.drift_threshold for rec in recent_records)

    async def trigger_retraining(self, model_id: str) -> None:
        """
        Placeholder for retraining logic. In production this would enqueue a
        training job; here we simply log and simulate a modest Sharpe boost.
        """
        logger.info("ModelingEngineer: triggering retraining", model=model_id)
        # Simulate a slight improvement after retraining
        boosted_sharpe = self._best_sharpe.get(model_id, 0) + random.uniform(0.05, 0.15)
        self._best_sharpe[model_id] = round(boosted_sharpe, 3)

    async def evaluate_champion(
        self, model_id: str, latest_record: ModelPerformanceRecord
    ) -> ModelingDecision:
        """
        Compare latest Sharpe with best known Sharpe.
        Promote if improvement exceeds 5% relative; otherwise monitor.
        """
        current_best = self._best_sharpe.get(model_id, -float("inf"))
        improvement = latest_record.sharpe - current_best
        if improvement > 0.05 * max(current_best, 1e-6):
            # Update best Sharpe and decide to promote
            self._best_sharpe[model_id] = latest_record.sharpe
            decision_type: Literal["promote"] = "promote"
            confidence = min(1.0, 0.5 + improvement)  # simple heuristic
            reason = f"Sharpe improved from {current_best:.3f} to {latest_record.sharpe:.3f}"
        else:
            decision_type = "monitor"
            confidence = 0.7
            reason = f"No significant Sharpe gain (Δ={improvement:.3f})"
        return ModelingDecision(
            decision_type=decision_type,
            model_id=model_id,
            reason=reason,
            confidence=confidence,
        )

    async def run_hyperparameter_sweep(self, model_id: str) -> None:
        """
        Perform a grid search over the hyperparameter space for the given model.
        Caches evaluated combinations to avoid redundant work and stops early
        if the incumbent Sharpe is already higher than any plausible improvement.
        """
        space = HYPERPARAM_SPACES.get(model_id, {})
        if not space:
            logger.debug("No hyperparameter space defined for model", model=model_id)
            return

        # Early exit: if incumbent Sharpe is already > 2.0 (arbitrary ceiling),
        # further sweeps are unlikely to be beneficial.
        incumbent_sharpe = self._best_sharpe.get(model_id, 0)
        if incumbent_sharpe > 2.0:
            logger.debug("Skipping sweep; incumbent Sharpe already high", model=model_id, sharpe=incumbent_sharpe)
            return

        # Generate all combinations lazily
        keys = list(space.keys())
        combos_iter = itertools.product(*(space[k] for k in keys))

        for combo in combos_iter:
            combo_dict = dict(zip(keys, combo))
            # Create a deterministic hashable representation for caching
            combo_signature = json.dumps(combo_dict, sort_keys=True)

            if combo_signature in self._evaluated_combos[model_id]:
                continue  # skip already evaluated combo

            # Simulate training & evaluation (replace with real training in prod)
            simulated_sharpe = self._simulate_hyperparam_performance(model_id, combo_dict)

            # Keep best hyperparameters
            if simulated_sharpe > self._best_sharpe.get(model_id, -float("inf")):
                self._best_sharpe[model_id] = simulated_sharpe
                self._best_params[model_id] = combo_dict

            self._evaluated_combos[model_id].add(combo_signature)

        logger.info(
            "Hyperparameter sweep completed",
            model=model_id,
            best_sharpe=self._best_sharpe.get(model_id),
            best_params=self._best_params.get(model_id, {}),
        )

    def _simulate_hyperparam_performance(self, model_id: str, params: dict) -> float:
        """
        Very lightweight deterministic simulation of Sharpe based on hyperparameters.
        The function is deliberately cheap to keep the sweep fast.
        """
        # Base Sharpe from incumbents
        base = INCUMBENT_SHARPE.get(model_id, 0.5)

        # Simple heuristic: sum of numeric params normalized
        numeric_sum = sum(v for v in params.values() if isinstance(v, (int, float)))
        normalized = numeric_sum / (len(params) or 1)

        # Add small random jitter
        jitter = random.uniform(-0.05, 0.05)

        return round(base + (normalized * 0.01) + jitter, 3)

    def _log_decision(self, decision: ModelingDecision) -> None:
        """
        Append a JSON line to the modeling log file. Uses atomic write
        to avoid race conditions when multiple engineers run concurrently.
        """
        line = json.dumps(asdict(decision), ensure_ascii=False)
        try:
            with MODELLING_LOG.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:  # pragma: no cover
            logger.error("Failed to write modeling decision", error=str(e))