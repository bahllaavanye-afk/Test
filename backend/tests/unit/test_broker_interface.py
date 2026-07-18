"""Broker interface guard: every AbstractBroker subclass must be concrete.

PR #420 (autonomous improver) truncated brokers/alpaca.py, deleting 6 of the
7 interface methods. The file still compiled, so nothing failed in CI — but
AlpacaBroker became un-instantiable (abstract TypeError), the exception was
swallowed at construction sites, and the backend silently fell back to
yfinance quotes with every Alpaca order path dead. This test makes that
failure mode loud: a broker class missing ANY interface method fails here
with the exact method names.
"""
from __future__ import annotations

import importlib
import os
from typing import Iterable, Mapping

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-32-bytes-hex-xxxxxx")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_broker_iface.db")

from app.brokers.base import AbstractBroker  # noqa: E402

_BROKER_MODULES = [
    "app.brokers.alpaca",
    "app.brokers.binance",
    "app.brokers.polymarket",
    "app.brokers.tradestation",
]


def _all_subclasses(cls: type) -> set[type]:
    """Recursively collect all subclasses of ``cls``."""
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def _import_modules(modules: Iterable[str]) -> Mapping[str, str]:
    """Import each module in ``modules`` and return a mapping of failures.

    The mapping keys are module names and values are formatted exception strings.
    """
    failures: dict[str, str] = {}
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            failures[mod] = f"{type(exc).__name__}: {exc}"
    return failures


def _collect_abstract_subclasses(base_cls: type) -> Mapping[str, list[str]]:
    """Return a mapping of subclass name to its sorted abstract methods."""
    abstract: dict[str, list[str]] = {}
    for cls in _all_subclasses(base_cls):
        abstract_methods = getattr(cls, "__abstractmethods__", None)
        if abstract_methods:
            abstract[cls.__name__] = sorted(abstract_methods)
    return abstract


def test_all_broker_modules_import():
    failures = _import_modules(_BROKER_MODULES)
    assert not failures, f"broker modules failed to import: {failures}"


def test_no_broker_subclass_is_abstract():
    # Ensure modules are importable; failures are reported by the import test above.
    _import_modules(_BROKER_MODULES)

    still_abstract = _collect_abstract_subclasses(AbstractBroker)
    assert not still_abstract, (
        f"broker classes missing interface methods (un-instantiable, silently "
        f"disables the broker): {still_abstract}"
    )


def test_alpaca_broker_declares_every_interface_method_directly():
    """Belt and braces: the primary broker must implement the full interface in
    its own body (not inherit stubs) — truncation deletes method bodies."""
    pytest.importorskip("alpaca")
    from app.brokers.alpaca import AlpacaBroker

    required = {
        "place_order",
        "cancel_order",
        "get_order",
        "get_positions",
        "get_account",
        "get_quote",
        "get_historical",
    }
    own = set(vars(AlpacaBroker))
    missing = required - own
    assert not missing, f"AlpacaBroker missing methods in class body: {sorted(missing)}"