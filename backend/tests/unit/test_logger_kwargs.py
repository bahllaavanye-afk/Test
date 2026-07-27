"""A stdlib logger called with structlog-style fields raises TypeError.

This codebase runs two logging conventions side by side:

    from app.utils.logging import logger      # structlog — logger.info("m", k=v) is CORRECT
    logger = logging.getLogger(__name__)      # stdlib   — logger.info("m", k=v) RAISES

Currently 32 modules bind a stdlib logger and 79 bind the structlog one. The
call sites look identical, so the only thing distinguishing correct from
fatal is a binding several hundred lines away.

There are ZERO violations today — this check is preventive. It exists because
the autonomous improver edits these files unattended, which is the same reason
`TestSlackStaysRemoved` exists. I introduced exactly this bug myself while
fixing `lorentzian_knn.py` (a stdlib-logger module) and caught it by eye
rather than by any test.

Conservative by construction: only modules where the binding is unambiguous
are checked at all, and only keywords the stdlib logging methods genuinely
reject are flagged.
"""
from __future__ import annotations

import ast
import pathlib

APP_ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "critical", "exception", "log"}

# Keywords the stdlib logging methods actually accept. Anything else is a
# structlog-style field and will raise.
STDLIB_KWARGS = {"exc_info", "stack_info", "stacklevel", "extra"}


def _logger_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """(names bound to a stdlib logger, names bound to a structlog logger)."""
    stdlib: set[str] = set()
    structlog_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            attr = getattr(fn, "attr", None)
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if attr == "getLogger":
                stdlib.update(targets)
            elif attr == "get_logger":       # structlog.get_logger()
                structlog_names.update(targets)

        # `from app.utils.logging import logger` — the project's structlog instance
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("utils.logging") or node.module.endswith(".logging"):
                for alias in node.names:
                    if alias.name in ("logger", "log"):
                        structlog_names.add(alias.asname or alias.name)

    return stdlib, structlog_names


def _scan(tree: ast.Module) -> list[tuple[int, str, str, list[str]]]:
    stdlib, structlog_names = _logger_bindings(tree)
    # A name bound both ways in one module is ambiguous — say nothing.
    stdlib -= structlog_names
    if not stdlib:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr not in LOG_METHODS:
            continue
        if not isinstance(fn.value, ast.Name) or fn.value.id not in stdlib:
            continue
        bad = [k.arg for k in node.keywords if k.arg is not None and k.arg not in STDLIB_KWARGS]
        if bad:
            found.append((node.lineno, fn.value.id, fn.attr, bad))
    return found


def test_no_stdlib_logger_is_called_with_structlog_fields():
    violations: list[str] = []
    stdlib_modules = 0

    for path in sorted(APP_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if _logger_bindings(tree)[0]:
            stdlib_modules += 1
        rel = path.relative_to(APP_ROOT.parent)
        for lineno, name, method, bad in _scan(tree):
            violations.append(
                f"{rel}:{lineno} — {name}.{method}(..., {'=..., '.join(bad)}=...) "
                f"on a stdlib logger. Use %-style formatting, or the structlog "
                f"logger from app.utils.logging."
            )

    assert stdlib_modules, "found no stdlib loggers — the scanner is looking in the wrong place"
    assert not violations, (
        "stdlib logger called with structlog-style keyword fields — this raises "
        "TypeError when the line executes, and log lines on error paths are "
        "exactly where that goes unnoticed:\n  " + "\n  ".join(violations)
    )


def test_scanner_flags_a_stdlib_logger_with_fields():
    """The check above only means something if it can actually fail."""
    tree = ast.parse('''
import logging
logger = logging.getLogger(__name__)

def f(symbol):
    logger.info("something happened", symbol=symbol)
''')
    assert _scan(tree) == [(6, "logger", "info", ["symbol"])]


def test_structlog_logger_with_fields_is_fine():
    """The same call is correct against the project's structlog logger."""
    tree = ast.parse('''
from app.utils.logging import logger

def f(symbol):
    logger.info("something happened", symbol=symbol)
''')
    assert _scan(tree) == []


def test_stdlib_kwargs_are_not_flagged():
    """exc_info/stack_info/stacklevel/extra are real stdlib parameters."""
    tree = ast.parse('''
import logging
logger = logging.getLogger(__name__)

def f(exc):
    logger.error("boom", exc_info=exc, stack_info=True, stacklevel=2, extra={"a": 1})
''')
    assert _scan(tree) == []


def test_percent_style_positional_args_are_not_flagged():
    """The correct stdlib form — positional args, not keywords."""
    tree = ast.parse('''
import logging
logger = logging.getLogger(__name__)

def f(symbol, n):
    logger.warning("dropped %s after %d tries", symbol, n)
''')
    assert _scan(tree) == []


def test_a_module_binding_both_conventions_is_skipped():
    """Ambiguous binding — the check must stay quiet rather than guess."""
    tree = ast.parse('''
import logging
from app.utils.logging import logger
logger = logging.getLogger(__name__)

def f(symbol):
    logger.info("m", symbol=symbol)
''')
    assert _scan(tree) == []
