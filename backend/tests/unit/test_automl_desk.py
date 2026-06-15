"""
Unit tests for the AutoML desk orchestration.

Hermetic: no network, no live inference singleton, no real artifacts dir. The
torch-backed cold-start path runs against a synthetic random-walk frame and a
patched, empty InferenceService + temp artifacts directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tasks.automl_desk import AutoMLDesk, SymbolResult, CycleReport, get_automl_desk

torch = pytest.importorskip("torch")
import pandas as pd  # noqa: E402


def _synthetic_ohlcv(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.012, size=n)
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000, 100_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class _FakeInference:
    def __init__(self):
        self.models = {}
        self.scalers = {}


class TestDataclasses:
    def test_symbol_result_defaults(self):
        r = SymbolResult(symbol="SPY", action="skipped")
        assert r.reason == ""
        assert r.n_val == 0

    def test_cycle_report_defaults(self):
        rep = CycleReport(timestamp="2026-06-13T00:00:00+00:00")
        assert rep.promotions == 0
        assert rep.results == []


class TestDeskBasics:
    def test_singleton(self):
        assert get_automl_desk() is get_automl_desk()

    def test_custom_symbols(self):
        desk = AutoMLDesk(symbols=["AAA", "BBB"], interval_seconds=10)
        assert desk.symbols == ["AAA", "BBB"]
        assert desk._running is False


class TestUpdateSymbolSync:
    def test_short_data_skipped_not_errored(self, monkeypatch):
        import app.ml.inference as inf_mod
        monkeypatch.setattr(inf_mod, "get_inference_service", lambda: _FakeInference())
        desk = AutoMLDesk()
        result = desk._update_symbol_sync("SPY", _synthetic_ohlcv(n=40))
        assert result.action in ("skipped", "error")
        assert result.action == "skipped"  # ValueError → skipped, never error

    def test_cold_start_path_runs(self, monkeypatch, tmp_path):
        import app.ml.inference as inf_mod
        import app.tasks.automl_desk as desk_mod
        fake = _FakeInference()
        monkeypatch.setattr(inf_mod, "get_inference_service", lambda: fake)
        monkeypatch.setattr(desk_mod, "ARTIFACTS_DIR", tmp_path)
        desk = AutoMLDesk(fine_tune_epochs=1)
        result = desk._update_symbol_sync("SPY", _synthetic_ohlcv(n=400))
        # Either it cold-started a champion (and hot-swapped it into fake), or the
        # fresh model didn't clear the quality bar — both are valid, neither errors.
        assert result.action in ("cold_start", "skipped")
        assert result.action != "error"
        if result.action == "cold_start":
            assert "lstm" in fake.models
            assert (tmp_path / "lstm_latest.pt").exists()

    def test_error_isolated_per_symbol(self, monkeypatch):
        import app.ml.inference as inf_mod

        def _boom():
            raise RuntimeError("inference exploded")

        monkeypatch.setattr(inf_mod, "get_inference_service", _boom)
        desk = AutoMLDesk()
        result = desk._update_symbol_sync("SPY", _synthetic_ohlcv(n=400))
        assert result.action == "error"
        assert "exploded" in result.reason


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_run_cycle_skips_when_no_data(self, monkeypatch):
        desk = AutoMLDesk(symbols=["SPY", "QQQ"])

        async def _no_data(_symbol):
            return None

        monkeypatch.setattr(desk, "_fetch_recent", _no_data)
        # Don't touch Redis / disk during the cycle.
        async def _noop(_report):
            return None
        monkeypatch.setattr(desk, "_persist_and_broadcast", _noop)

        report = await desk.run_cycle()
        assert report.symbols_processed == 0
        assert report.promotions == 0
        assert len(report.results) == 2
        assert all(r.action == "skipped" for r in report.results)
