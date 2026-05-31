"""
Integration tests: background task imports and initialization.
Verifies all 11 employee tasks can be imported without crashing.
A broken import silently kills tasks that run 24/7 in production.
"""
from __future__ import annotations

import importlib
import pytest


# (module_path, class_or_fn_name, has_class)
TASK_MODULES = [
    ("app.tasks.scheduler",          "start_scheduler",         False),
    ("app.tasks.strategy_runner",    "ContinuousStrategyRunner", False),
    ("app.tasks.price_feed",         "run_price_feed",          False),
    ("app.tasks.algo_agent",         "AlgoAgent",               True),
    ("app.tasks.self_improver",      "SelfImprover",            True),
    ("app.tasks.code_quality_loop",  "CodeQualityLoop",         True),
    ("app.tasks.qa_monitor",         "QAMonitor",               True),
    ("app.tasks.research_scientist", "ResearchScientist",       True),
    ("app.tasks.modeling_engineer",  "ModelingEngineer",        True),
    ("app.tasks.regime_monitor",     "RegimeMonitor",           True),
    ("app.tasks.ml_retrain",         "nightly_retrain",         False),
]


@pytest.mark.parametrize("module_path,symbol,is_class", TASK_MODULES)
def test_task_module_importable(module_path, symbol, is_class):
    """Every background task module must import without errors."""
    mod = importlib.import_module(module_path)
    available = [x for x in dir(mod) if not x.startswith("_")]
    assert hasattr(mod, symbol), (
        f"{module_path} is missing '{symbol}'. Available: {available}"
    )


@pytest.mark.parametrize("module_path,symbol,is_class", [
    t for t in TASK_MODULES if t[2]  # only test classes, not functions
])
def test_task_class_instantiable(module_path, symbol, is_class):
    """Every background task class must instantiate with no required args."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, symbol)
    instance = cls()
    assert instance is not None


@pytest.mark.parametrize("module_path,symbol,is_class", [
    t for t in TASK_MODULES if t[2]
])
def test_task_class_has_run_or_start(module_path, symbol, is_class):
    """Every task class must expose run() or start() for the supervisor."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, symbol)
    instance = cls()
    assert hasattr(instance, "run") or hasattr(instance, "start"), (
        f"{symbol} has neither 'run' nor 'start' — main.py cannot supervise it"
    )
