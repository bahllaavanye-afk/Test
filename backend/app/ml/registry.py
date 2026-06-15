"""
ML Model Registry — tracks trained model artifacts, metadata, and performance.

Each registered model is recorded in a JSON index file with:
  model_type, created_at, val_sharpe, val_accuracy, artifact_path, tags

Usage:
    registry = ModelRegistry()
    registry.register("my_tft", model, metrics, artifact_path="models/tft_v1.pt")
    registry.get_best("tft", metric="val_sharpe")
    registry.compare_models(["tft_v1", "lgbm_v2"])
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.utils.logging import logger

_DEFAULT_INDEX = Path("experiments/model_registry.json")


class ModelRegistry:
    """
    Persistent registry for trained ML models.

    The registry stores a JSON index at `index_path` mapping model names to
    their metadata records.  Artifacts (weight files, etc.) are stored at
    paths provided by the caller.
    """

    def __init__(self, index_path: str | Path = _DEFAULT_INDEX):
        self.index_path = Path(index_path)
        self._records: dict[str, dict[str, Any]] = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Load the JSON index from disk (creates an empty one if missing)."""
        if self.index_path.exists():
            try:
                self._records = json.loads(self.index_path.read_text())
            except json.JSONDecodeError as exc:
                logger.warning(f"ModelRegistry: corrupt index at {self.index_path}: {exc}. Starting fresh.")
                self._records = {}
        else:
            self._records = {}

    def _save_index(self) -> None:
        """Persist the in-memory index to disk atomically."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._records, indent=2, default=str))
        tmp.replace(self.index_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model_type: str,
        artifact_path: str,
        val_sharpe: float = 0.0,
        val_accuracy: float = 0.0,
        val_auc: float = 0.5,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Register a trained model.

        Args:
            name:          Unique name for this model version (e.g. "tft_spy_v3").
            model_type:    Class identifier (e.g. "tft", "lightgbm", "lstm").
            artifact_path: Path where the model weights / files are saved.
            val_sharpe:    Out-of-sample Sharpe ratio on validation set.
            val_accuracy:  Direction accuracy on validation set (0-1).
            val_auc:       ROC-AUC on validation set.
            tags:          Free-form labels, e.g. ["production", "SPY", "1h"].
            extra:         Any additional metadata to store.

        Returns:
            The metadata record dict that was persisted.
        """
        record: dict[str, Any] = {
            "name": name,
            "model_type": model_type,
            "artifact_path": str(artifact_path),
            "created_at": datetime.now(UTC).isoformat(),
            "val_sharpe": round(float(val_sharpe), 4),
            "val_accuracy": round(float(val_accuracy), 4),
            "val_auc": round(float(val_auc), 4),
            "tags": tags or [],
            **(extra or {}),
        }
        self._records[name] = record
        self._save_index()
        logger.info(f"ModelRegistry: registered '{name}' ({model_type}) sharpe={val_sharpe:.3f}")
        return record

    def load(self, name: str) -> dict[str, Any]:
        """
        Return the metadata record for *name*.

        The caller is responsible for actually loading the weights from
        ``record["artifact_path"]`` using the appropriate model class.

        Raises:
            KeyError: if *name* is not found in the registry.
        """
        if name not in self._records:
            raise KeyError(f"ModelRegistry: model '{name}' not found. Available: {list(self._records)}")
        return self._records[name]

    def list_models(
        self,
        model_type: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List registered models, optionally filtered by type or tag.

        Returns records sorted by ``val_sharpe`` descending.
        """
        records = list(self._records.values())
        if model_type:
            records = [r for r in records if r.get("model_type") == model_type]
        if tag:
            records = [r for r in records if tag in r.get("tags", [])]
        return sorted(records, key=lambda r: r.get("val_sharpe", 0.0), reverse=True)

    def get_best(
        self,
        model_type: str | None = None,
        metric: str = "val_sharpe",
        tag: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Return the best model record according to *metric*.

        Args:
            model_type: Restrict to a specific model type (optional).
            metric:     Key to rank by — must exist in records.
                        Typical values: "val_sharpe", "val_accuracy", "val_auc".
            tag:        Restrict to models with this tag (optional).

        Returns:
            Metadata record dict, or None if the registry is empty / no match.
        """
        candidates = self.list_models(model_type=model_type, tag=tag)
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.get(metric, float("-inf")))

    def compare_models(
        self,
        names: list[str],
        sort_by: str = "val_sharpe",
    ) -> list[dict[str, Any]]:
        """
        Return a leaderboard for the specified model names.

        Args:
            names:   List of registered model names to compare.
            sort_by: Metric to rank by (default ``val_sharpe``).

        Returns:
            List of records sorted by *sort_by* descending.  Models not found
            in the registry are silently skipped with a warning.
        """
        records = []
        for name in names:
            try:
                records.append(self.load(name))
            except KeyError:
                logger.warning(f"ModelRegistry.compare_models: '{name}' not found, skipping.")
        return sorted(records, key=lambda r: r.get(sort_by, float("-inf")), reverse=True)

    def delete(self, name: str) -> bool:
        """
        Remove a model from the registry index (does NOT delete artifact files).

        Returns True if removed, False if not found.
        """
        if name in self._records:
            del self._records[name]
            self._save_index()
            logger.info(f"ModelRegistry: deleted '{name}'")
            return True
        return False

    def update(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """
        Update fields on an existing record.

        Useful for adding post-hoc metrics (e.g. live Sharpe after paper trading).

        Raises:
            KeyError: if *name* is not found.
        """
        record = self.load(name)
        record.update(kwargs)
        self._records[name] = record
        self._save_index()
        return record

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"ModelRegistry(index={self.index_path}, n_models={len(self)})"
