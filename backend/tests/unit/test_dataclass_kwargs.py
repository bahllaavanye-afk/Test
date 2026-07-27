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


def _declared_fields(node: ast.ClassDef) -> set[str]:
    """Annotated class-level assignments — how a dataclass declares its fields."""
    return {
        stmt.target.id
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


def _collect() -> tuple[dict[str, list[tuple]], dict[pathlib.Path, ast.Module]]:
    definitions: dict[str, list[tuple[set[str], list[str], bool]]] = {}
    trees: dict[pathlib.Path, ast.Module] = {}

    for path in sorted(APP_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        trees[path] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_dataclass(node):
                custom_init = any(
                    isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and s.name == "__init__"
                    for s in node.body
                )
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                definitions.setdefault(node.name, []).append(
                    (_declared_fields(node), bases, custom_init)
                )
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
    for fields, bases, custom_init in entries:
        if custom_init:
            return None
        total |= fields
        for base in bases:
            inherited = _resolve(base, definitions, seen + (name,))
            if inherited is None:
                return None
            total |= inherited
    return total


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


def test_scanner_detects_a_planted_violation(tmp_path):
    """The check above only means something if it can actually fail."""
    source = '''
from dataclasses import dataclass

@dataclass
class Thing:
    good: int
    other: str = ""

t = Thing(good=1, bogus=2)
'''
    tree = ast.parse(source)
    definitions: dict[str, list[tuple]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_dataclass(node):
            definitions[node.name] = [(_declared_fields(node), [], False)]

    fields = _resolve("Thing", definitions)
    assert fields == {"good", "other"}

    bad = [
        kw.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "Thing"
        for kw in node.keywords
        if kw.arg is not None and kw.arg not in fields
    ]
    assert bad == ["bogus"]


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
        node.name: [(_declared_fields(node), [], True)]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _is_dataclass(node)
    }
    assert _resolve("Flexible", definitions) is None


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
    definitions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_dataclass(node):
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            definitions[node.name] = [(_declared_fields(node), bases, False)]

    assert _resolve("Child", definitions) == {"a", "b"}, (
        "a field inherited from a dataclass base must not be reported as unknown"
    )
