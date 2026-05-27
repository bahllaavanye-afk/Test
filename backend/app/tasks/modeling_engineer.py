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
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
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
            except Exception as e:
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
            decision = await self.evaluate_champion(model_id)
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

        record = ModelPerformanceRecord(
            model_id=model_id,
            accuracy=round(accuracy, 4),
            sharpe=round(sharpe, 4),
            n_predictions=n_recent,
            drift_detected=drift_detected,
        )
        return record

    async def detect_drift(self, model_id: str) -> bool:
        """
        Return True if the last N checks show degraded accuracy below threshold.
        Uses the rolling cache so single bad readings don't trigger false alarms.
        """
        history = list(self._perf_cache[model_id])
        if not history:
            return False

        # Check most recent record first (fast path)
        latest = history[-1]
        if latest.accuracy >= self.drift_threshold:
            return False

        # Confirm with sliding window: if majority are below threshold → drift
        recent = history[-min(3, len(history)):]
        below = sum(1 for r in recent if r.accuracy < self.drift_threshold)
        return below >= max(1, len(recent) // 2 + 1)

    async def trigger_retraining(self, model_id: str) -> None:
        """
        Log the retrain decision and create an experiment config entry.
        In production this enqueues a Celery/Redis job or calls the ML training API.
        """
        decision = ModelingDecision(
            decision_type="retrain",
            model_id=model_id,
            reason=(
                f"Accuracy below {self.drift_threshold} for "
                f"{self.retrain_after_n_drift} consecutive checks"
            ),
            confidence=0.9,
        )
        self._decisions.append(decision)
        self._log_decision(decision)

        # Write an experiment config to experiments/
        experiment_path = (
            Path(__file__).parents[3]
            / "experiments"
            / "configs"
            / f"retrain_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
        )
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            experiment_path.write_text(
                json.dumps(
                    {
                        "model_id": model_id,
                        "trigger": "drift_detected",
                        "params": self._best_params.get(model_id, {}),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
        except Exception as e:
            logger.warning(f"ModelingEngineer: could not write experiment config: {e}")

        logger.info(
            "ModelingEngineer: retraining triggered",
            model=model_id,
            experiment=str(experiment_path.name),
        )

    async def evaluate_champion(self, model_id: str) -> ModelingDecision:
        """
        Compare latest Sharpe vs incumbent. Promote if significantly better,
        demote if significantly worse, otherwise monitor.
        """
        history = list(self._perf_cache[model_id])
        if not history:
            return ModelingDecision(
                decision_type="monitor",
                model_id=model_id,
                reason="No performance data yet",
                confidence=0.5,
            )

        latest = history[-1]
        incumbent = self._best_sharpe.get(model_id, INCUMBENT_SHARPE.get(model_id, 0.8))
        delta = latest.sharpe - incumbent

        if delta > 0.15:
            # New champion — promote
            self._best_sharpe[model_id] = latest.sharpe
            decision = ModelingDecision(
                decision_type="promote",
                model_id=model_id,
                reason=f"Sharpe improved by {delta:.3f} vs incumbent ({incumbent:.3f}→{latest.sharpe:.3f})",
                confidence=min(0.95, 0.7 + delta),
            )
            logger.info(
                "ModelingEngineer: model promoted",
                model=model_id,
                new_sharpe=round(latest.sharpe, 3),
                improvement=round(delta, 3),
            )
        elif delta < -0.30:
            # Significantly underperforming — demote
            decision = ModelingDecision(
                decision_type="demote",
                model_id=model_id,
                reason=f"Sharpe degraded by {abs(delta):.3f} vs incumbent",
                confidence=0.8,
            )
            logger.warning(
                "ModelingEngineer: model demoted",
                model=model_id,
                current_sharpe=round(latest.sharpe, 3),
                incumbent_sharpe=round(incumbent, 3),
            )
        else:
            decision = ModelingDecision(
                decision_type="monitor",
                model_id=model_id,
                reason=f"Performance within acceptable range (delta={delta:.3f})",
                confidence=0.75,
            )

        return decision

    async def run_hyperparameter_sweep(self, model_type: str) -> None:
        """
        Sample N random configs from the search space and log them as experiments.
        In production this would launch actual training jobs.
        """
        space = HYPERPARAM_SPACES.get(model_type, {})
        if not space:
            return

        configs = []
        for trial in range(3):
            params = {k: random.choice(v) for k, v in space.items()}
            # Simulate trial outcome
            trial_sharpe = self._best_sharpe.get(model_type, 0.8) + random.gauss(0, 0.2)
            configs.append(
                {
                    "trial": trial + 1,
                    "model_type": model_type,
                    "params": params,
                    "estimated_sharpe": round(trial_sharpe, 4),
                }
            )

        # Pick best trial config
        best = max(configs, key=lambda c: c["estimated_sharpe"])
        current_best = self._best_sharpe.get(model_type, 0.8)

        if best["estimated_sharpe"] > current_best * 1.05:
            self._best_params[model_type] = best["params"]
            logger.info(
                "ModelingEngineer: sweep found better config",
                model=model_type,
                sharpe=best["estimated_sharpe"],
                params=best["params"],
            )

        # Log sweep results
        sweep_entry = {
            "type": "hyperparameter_sweep",
            "model_type": model_type,
            "cycle": self._cycle,
            "configs_tried": len(configs),
            "best_sharpe": best["estimated_sharpe"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._log_raw(sweep_entry)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _log_decision(self, decision: ModelingDecision) -> None:
        """Append decision to modeling log."""
        entry = asdict(decision)
        entry["log_type"] = "decision"
        self._log_raw(entry)

    def _log_raw(self, entry: dict) -> None:
        """Append arbitrary dict to modeling_log.jsonl."""
        try:
            with open(MODELING_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"ModelingEngineer: failed to write log entry: {e}")

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def get_engineering_summary(self) -> dict:
        """Summary for API endpoint."""
        recent_decisions = self._decisions[-20:] if self._decisions else []

        # Latest performance per model
        latest_perf = {}
        for model_id, history in self._perf_cache.items():
            if history:
                rec = history[-1]
                latest_perf[model_id] = {
                    "accuracy": rec.accuracy,
                    "sharpe": rec.sharpe,
                    "drift_detected": rec.drift_detected,
                    "checked_at": rec.checked_at,
                }

        return {
            "cycles_completed": self._cycle,
            "models_monitored": MODEL_TYPES,
            "drift_threshold": self.drift_threshold,
            "latest_performance": latest_perf,
            "consecutive_drift_counts": dict(self._drift_counts),
            "best_sharpe": self._best_sharpe,
            "best_params": self._best_params,
            "recent_decisions": [
                {
                    "type": d.decision_type,
                    "model": d.model_id,
                    "reason": d.reason,
                    "confidence": d.confidence,
                    "at": d.decided_at,
                }
                for d in recent_decisions
            ],
            "promote_count": sum(
                1 for d in self._decisions if d.decision_type == "promote"
            ),
            "retrain_count": sum(
                1 for d in self._decisions if d.decision_type == "retrain"
            ),
        }
