"""Logging must actually be configured, not left on library defaults.

Found by the same unreferenced-function sweep as the risk gate and the exit
path. `configure_logging()` had exactly ONE textual occurrence in the package —
its own `def`. Nothing called it, so structlog ran on its library defaults and
this app's configuration was decorative:

  * renderer was `ConsoleRenderer`, not `JSONRenderer` — Render received
    unstructured text, so nothing downstream could parse a log line
  * wrapper_class was `BoundLoggerFilteringAtNotset`, which filters NOTHING,
    so all 105 `logger.debug()` call sites in `app/` emitted on every
    production run

Both were verified by importing structlog and reading `get_config()` before
the fix, not inferred from reading the source.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

import structlog

APP = pathlib.Path(__file__).resolve().parents[2] / "app"


def test_configure_logging_is_actually_called():
    """A config function nothing calls is not configuration.

    AST-based, not `str.count`: an explanatory comment naming the function
    must not count as a call site. That mistake was made once already, in the
    reachability guard in test_risk_gate_wiring.
    """
    referenced = False
    # Guard against APP being None or not existing
    if not APP or not APP.exists():
        pytest.fail(f"Application directory not found at {APP}")

    for path in APP.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            # Skip files that cannot be parsed; they do not affect the call check
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "configure_logging"
            ):
                referenced = True
                break  # No need to continue scanning once found
        if referenced:
            break
    assert referenced, (
        "configure_logging() is never called, so structlog runs on library "
        "defaults: console output instead of JSON, and no level filtering"
    )


def test_importing_the_app_installs_the_configuration():
    """The call must be reached by merely importing the app.

    static_server.py imports app.main, so this covers every entrypoint. If the
    call moved inside lifespan, module-level logging before startup would still
    be unconfigured.
    """
    import app.main  # noqa: F401 — import is the thing under test

    cfg = structlog.get_config()
    # Defensive check for None or missing keys
    processors = cfg.get("processors") if isinstance(cfg, dict) else None
    assert processors, "structlog has no processors — never configured"


def test_production_logs_are_machine_readable():
    """Console text cannot be parsed by a log aggregator."""
    import app.main  # noqa: F401

    from app.config import settings

    cfg = structlog.get_config()
    processors = cfg.get("processors") if isinstance(cfg, dict) else None
    # Guard against empty processors list
    assert processors, "structlog processors list is missing or empty"
    renderer = type(processors[-1]).__name__
    expected = "ConsoleRenderer" if settings.debug else "JSONRenderer"
    assert renderer == expected, (
        f"with debug={settings.debug} the renderer must be {expected}, got "
        f"{renderer} — the unconfigured default is ConsoleRenderer in BOTH modes"
    )


def test_debug_logs_do_not_escape_into_production():
    """105 debug call sites in app/ were emitting on every run."""
    import app.main  # noqa: F401

    from app.config import settings

    cfg = structlog.get_config()
    wrapper_class = cfg.get("wrapper_class") if isinstance(cfg, dict) else None
    # Ensure wrapper_class is present
    assert wrapper_class, "structlog wrapper_class is missing"
    wrapper = wrapper_class.__name__
    assert "Notset" not in wrapper, (
        "BoundLoggerFilteringAtNotset filters nothing — every logger.debug() "
        f"in app/ reaches production output (wrapper_class={wrapper})"
    )
    if not settings.debug:
        assert any(level in wrapper for level in ("Info", "Warning", "Error")), (
            f"expected an INFO-or-higher filter outside debug mode, got {wrapper}"
        )


def test_a_debug_line_is_actually_suppressed(capsys):
    """The config is only real if it changes what reaches stdout.

    structlog does not propagate to stdlib logging, so this asserts on captured
    output rather than caplog.
    """
    import app.main  # noqa: F401

    from app.config import settings

    if settings.debug:
        pytest.skip("debug mode intentionally emits debug lines")

    from app.utils.logging import logger

    # Ensure any prior output is cleared
    capsys.readouterr()
    logger.debug("SENTINEL_DEBUG_MUST_NOT_APPEAR")
    logger.info("SENTINEL_INFO_MUST_APPEAR")
    out = capsys.readouterr()
    combined = out.out + out.err

    assert "SENTINEL_DEBUG_MUST_NOT_APPEAR" not in combined
    assert "SENTINEL_INFO_MUST_APPEAR" in combined