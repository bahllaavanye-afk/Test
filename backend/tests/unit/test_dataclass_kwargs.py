"""Every dataclass construction must use keywords that are actually fields.

This defect class has now shipped three times, and each instance sat in code
that looked completely ordinary:

  * `OrderResult(reason="price_tolerance_exceeded")` — the bracket price guard.
    The TypeError was caught by the surrounding `except Exception`, logged as a
    quote-fetch failure, and the entry submitted anyway. The guard never once
    rejected an order.
  * `OrderResult(order_id=..., symbol=...)` — the RL branch of SmartOrderRouter.
    Raised on every *successful* execution, after the fills had happened.
  * `BacktestSignals(positions=, returns=, probabilities=, execution_time_ms=)`
    — Lorentzian KNN. All four wrong and both required fields missing, so the
    strategy could never be backtested at all.

Python only raises at call time, so a construction on a rare error path — which
is exactly where these live — stays invisible until production hits it. This
check is static: it parses every module and compares each dataclass call's
keywords against the fields that dataclass actually declares.

Deliberately conservative. It skips anything it cannot resolve with certainty
(`**kwargs` unpacking, classes with a hand-written `__init__`, unresolvable
bases), so a pass is not proof of correctness — but a failure is always real.
"""
from __future__ import annotations

import ast
import pathlib

APP_ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"


def _is_dataclass(node: ast.ClassDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (getattr(target, "attr", None) or getattr(target, "id", None)) == "dataclass":
            return True
    return False


def _declared_fields(node: ast.ClassDef) -> tuple[set[str], list[str]]:
    """(all fields, required fields) from the annotated class-level assignments.

    Required means annotated with no default — `x: int` rather than `x: int = 0`
    or `x: list = field(default_factory=list)`. ClassVar/InitVar are not
    constructor fields at all.
    """
    allf: set[str] = set()
    required: list[str] = []
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        annotation = ast.unparse(stmt.annotation)
        if "ClassVar" in annotation or "InitVar" in annotation:
            continue
        allf.add(stmt.target.id)
        if stmt.value is None:
            required.append(stmt.target.id)
    return allf, required


def _define(node: ast.ClassDef) -> dict:
    allf, required = _declared_fields(node)
    return {
        "fields": allf,
        "required": required,
        "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
        "custom_init": any(
            isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == "__init__"
            for s in node.body
        ),
    }


def _collect() -> tuple[dict[str, list[dict]], dict[pathlib.Path, ast.Module]]:
    definitions: dict[str, list[dict]] = {}
    trees: dict[pathlib.Path, ast.Module] = {}

    for path in sorted(APP_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        trees[path] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_dataclass(node):
                definitions.setdefault(node.name, []).append(_define(node))
    return definitions, trees


def _resolve(name: str, definitions: dict, seen: tuple = ()) -> set[str] | None:
    """Union of fields across every definition of `name`.

    Returns None when the answer cannot be established — a hand-written
    __init__, an unresolvable base class, or a cyclic reference. Callers must
    treat None as "skip", never as "no fields".
    """
    entries = definitions.get(name)
    if not entries or name in seen:
        return None

    total: set[str] = set()
    for entry in entries:
        if entry["custom_init"]:
            return None
        total |= entry["fields"]
        for base in entry["bases"]:
            inherited = _resolve(base, definitions, seen + (name,))
            if inherited is None:
                return None
            total |= inherited
    return total


def _resolve_required(name: str, definitions: dict, seen: tuple = ()) -> list[str] | None:
    """Fields with no default, base classes first (dataclass __init__ order).

    Stricter than _resolve: a name defined more than once is ambiguous here,
    because unioning "required" across two different classes would invent
    requirements neither of them has.
    """
    entries = definitions.get(name)
    if not entries or name in seen or len(entries) > 1:
        return None

    entry = entries[0]
    if entry["custom_init"]:
        return None

    inherited: list[str] = []
    for base in entry["bases"]:
        got = _resolve_required(base, definitions, seen + (name,))
        if got is None:
            return None
        inherited += got
    return inherited + entry["required"]


def test_no_dataclass_is_constructed_with_a_nonexistent_field():
    definitions, trees = _collect()
    assert definitions, "found no dataclasses — the scanner is looking in the wrong place"

    resolved_cache: dict[str, set[str] | None] = {}
    violations: list[str] = []

    for path, tree in trees.items():
        rel = path.relative_to(APP_ROOT.parent)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            cls = node.func.id
            if cls not in definitions:
                continue
            if cls not in resolved_cache:
                resolved_cache[cls] = _resolve(cls, definitions)
            fields = resolved_cache[cls]
            if fields is None:
                continue
            for kw in node.keywords:
                if kw.arg is None:      # **unpacking — not statically checkable
                    continue
                if kw.arg not in fields:
                    violations.append(
                        f"{rel}:{node.lineno} — {cls}({kw.arg}=...) is not a field. "
                        f"Valid: {', '.join(sorted(fields))}"
                    )

    assert not violations, (
        "dataclass constructed with keywords that are not fields — this raises "
        "TypeError at call time, and these calls tend to live on error paths "
        "where nothing notices:\n  " + "\n  ".join(violations)
    )


def test_no_dataclass_construction_omits_a_required_field():
    """The mirror image: every field with no default must be supplied.

    Found `EvalMetrics(loss=, accuracy=, auc=)` in ml/models/itransformer.py —
    `sharpe` has no default, so `iTransformerPredictor.evaluate()` raised
    TypeError on every call and that model could never be evaluated. Every
    other model in the package passes `sharpe=0.0`.
    """
    definitions, trees = _collect()
    cache: dict[str, list[str] | None] = {}
    violations: list[str] = []

    for path, tree in trees.items():
        rel = path.relative_to(APP_ROOT.parent)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            cls = node.func.id
            if cls not in definitions:
                continue
            # Positional args fill fields in declaration order. Resolving that
            # correctly means honouring kw_only, inherited ordering and
            # field(kw_only=...), so skip those calls rather than guess.
            if node.args or any(k.arg is None for k in node.keywords):
                continue
            if cls not in cache:
                cache[cls] = _resolve_required(cls, definitions)
            required = cache[cls]
            if required is None:
                continue
            provided = {k.arg for k in node.keywords}
            missing = [f for f in required if f not in provided]
            if missing:
                violations.append(
                    f"{rel}:{node.lineno} — {cls}(...) omits required "
                    f"{', '.join(missing)}. Passed: {', '.join(sorted(provided)) or '(nothing)'}"
                )

    assert not violations, (
        "dataclass constructed without a field that has no default — this "
        "raises TypeError at call time:\n  " + "\n  ".join(violations)
    )


def test_scanner_detects_a_planted_violation():
    """The checks above only mean something if they can actually fail."""
    tree = ast.parse('''
from dataclasses import dataclass

@dataclass
class Thing:
    good: int
    needed: float
    other: str = ""

t = Thing(good=1, bogus=2)
''')
    definitions = {
        node.name: [_define(node)]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _is_dataclass(node)
    }

    assert _resolve("Thing", definitions) == {"good", "needed", "other"}
    assert _resolve_required("Thing", definitions) == ["good", "needed"], (
        "a field with a default must not be reported as required"
    )

    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Thing"
    )
    provided = {k.arg for k in call.keywords}
    assert [k for k in provided if k not in _resolve("Thing", definitions)] == ["bogus"]
    assert [f for f in _resolve_required("Thing", definitions) if f not in provided] == ["needed"]


def test_classes_with_a_custom_init_are_skipped():
    """A hand-written __init__ can accept anything — never flag those."""
    tree = ast.parse('''
from dataclasses import dataclass

@dataclass
class Flexible:
    known: int
    def __init__(self, **kw):
        pass
''')
    definitions = {
        node.name: [_define(node)]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _is_dataclass(node)
    }
    assert _resolve("Flexible", definitions) is None
    assert _resolve_required("Flexible", definitions) is None


def test_inherited_fields_are_counted():
    tree = ast.parse('''
from dataclasses import dataclass

@dataclass
class Base:
    a: int

@dataclass
class Child(Base):
    b: int
''')
    definitions = {
        node.name: [_define(node)]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _is_dataclass(node)
    }

    assert _resolve("Child", definitions) == {"a", "b"}, (
        "a field inherited from a dataclass base must not be reported as unknown"
    )
    assert _resolve_required("Child", definitions) == ["a", "b"], (
        "base-class fields come first in the generated __init__ signature"
    )


def test_classvar_is_not_a_constructor_field():
    """ClassVar is class-level state, not a dataclass field — never required."""
    tree = ast.parse('''
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class WithClassVar:
    registry: ClassVar[dict] = {}
    real: int
''')
    definitions = {
        node.name: [_define(node)]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _is_dataclass(node)
    }
    assert _resolve("WithClassVar", definitions) == {"real"}
    assert _resolve_required("WithClassVar", definitions) == ["real"]
