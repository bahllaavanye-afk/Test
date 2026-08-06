"""Every ML model must satisfy the AbstractModel contract — checked WITHOUT torch.

CI runs `pytest tests/` with:

    --ignore=tests/unit/test_ml_models.py
    --ignore=tests/unit/test_a3c_lstm.py

because torch is not in the CI dependency list. The result is that ~15 model
implementations have had **no CI coverage at all**, and it showed:

  * `lorentzian_knn.backtest_signals()` built `BacktestSignals(positions=…)` —
    four wrong keywords and both required fields missing. The strategy could
    never be backtested.
  * `itransformer.evaluate()` omitted the required `EvalMetrics.sharpe`, so it
    raised TypeError on every call. The model could never be evaluated.

Neither was caught by a test. Both were found by static analysis, because
static analysis is the only thing that *can* run here.

So this checks the contract the same way: parse each model module and verify
the abstract methods are declared. It needs no torch, no GPU, and no fixtures,
which is precisely why it runs on every PR.

Scope note: this proves a method EXISTS with the right name. It cannot prove
the body is correct — `test_dataclass_kwargs.py` covers the return-value
construction that broke in both cases above. Together they cover the failure
mode; alone, neither does.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

# File system
MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "app" / "ml" / "models"

# Declared @abstractmethod on AbstractModel (app/ml/models/base_model.py).
REQUIRED_METHODS: set[str] = {"forward", "train_epoch", "evaluate"}

# Modules that are not model implementations.
SKIP_MODULES: set[str] = {"__init__", "base_model"}

# Validation thresholds
MIN_EXPECTED_MODELS: int = 10

# Error messages
ERROR_MODEL_DISCOVERY: str = "expected the model package, found {} modules"
ERROR_MISSING_METHODS: str = (
    "AbstractModel subclass does not implement the interface. torch is not "
    "in the CI dep list, so nothing else exercises these files:\n  "
)
ERROR_SYNTAX: str = "model module(s) with syntax errors:\n  "
ERROR_DICT_RETURN: str = "evaluate() returns a dict"
ERROR_UNPARSEABLE: str = "{}: unparseable — {}"
ERROR_MISSING_FORMAT: str = "{}::{} missing {}"
ERROR_REQUIRED_NOT_IN_BASE: str = (
    "REQUIRED_METHODS lists {!r}, but AbstractModel does not declare "
    "it as @abstractmethod. Declared: {}"
)

# AST attribute names
ATTR_ABSTRACTMETHOD: str = "abstractmethod"
ATTR_ABSTRACTMODEL: str = "AbstractModel"


def _model_modules() -> list[pathlib.Path]:
    return sorted(
        p for p in MODELS_DIR.glob("*.py") if p.stem not in SKIP_MODULES
    )


def _classes_with_methods(tree: ast.Module) -> dict[str, set[str]]:
    """{class name: {method names}} for every class in the module."""
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            out[node.name] = {
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return out


def _subclasses_abstract_model(tree: ast.Module) -> list[str]:
    """Class names that inherit AbstractModel (directly, by name)."""
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = getattr(base, "id", None) or getattr(base, "attr", None)
            if base_name == ATTR_ABSTRACTMODEL:
                names.append(node.name)
    return names


def test_model_modules_are_discoverable():
    mods = _model_modules()
    assert len(mods) >= MIN_EXPECTED_MODELS, ERROR_MODEL_DISCOVERY.format(len(mods))


def test_every_abstractmodel_subclass_declares_the_required_methods():
    """A model missing `evaluate` fails only when something calls it."""
    violations: list[str] = []

    for path in _model_modules():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            violations.append(ERROR_UNPARSEABLE.format(path.name, exc))
            continue

        classes = _classes_with_methods(tree)
        for cls in _subclasses_abstract_model(tree):
            missing = REQUIRED_METHODS - classes.get(cls, set())
            if missing:
                violations.append(
                    ERROR_MISSING_FORMAT.format(
                        path.name, cls, ", ".join(sorted(missing))
                    )
                )

    assert not violations, ERROR_MISSING_METHODS + "\n  ".join(violations)


def test_every_model_module_parses():
    """A syntax error here is invisible to CI — the tests are --ignore'd."""
    broken = []
    for path in _model_modules():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            broken.append(f"{path.name}:{exc.lineno} {exc.msg}")
    assert not broken, ERROR_SYNTAX + "\n  ".join(broken)


def test_no_model_calls_evaluate_returning_a_bare_dict():
    """evaluate() must return EvalMetrics, not an ad-hoc dict.

    A dict silently loses the schema — callers reading `.sharpe` get
    AttributeError at runtime instead of a clear construction error.
    """
    offenders = []
    for path in _model_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "evaluate":
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    offenders.append(f"{path.name}:{sub.lineno} {ERROR_DICT_RETURN}")
    assert not offenders, "\n  ".join(offenders)


@pytest.mark.parametrize("required", sorted(REQUIRED_METHODS))
def test_required_method_list_matches_the_abstract_base(required):
    """If base_model gains an @abstractmethod, this list must follow it."""
    base = MODELS_DIR / "base_model.py"
    tree = ast.parse(base.read_text(encoding="utf-8"))

    declared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == ATTR_ABSTRACTMODEL:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in item.decorator_list:
                        name = getattr(dec, "id", None) or getattr(dec, "attr", None)
                        if name == ATTR_ABSTRACTMETHOD:
                            declared.add(item.name)

    assert required in declared, ERROR_REQUIRED_NOT_IN_BASE.format(
        required, sorted(declared)
    )