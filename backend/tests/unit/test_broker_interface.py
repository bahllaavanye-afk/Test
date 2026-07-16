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
from collections.abc import Iterable
from typing import Any

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


def _all_subclasses(cls: type | None) -> set[type]:
    """Recursively collect all subclasses of *cls*.

    Handles ``None`` gracefully by returning an empty set.
    """
    if cls is None:
        return set()
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def _validate_module_iterable(mods: Any) -> Iterable[str]:
    """Validate that *mods* is an iterable of strings.

    Returns an empty iterable for ``None`` or non‑iterable inputs to avoid
    TypeError during iteration.
    """
    if mods is None:
        return ()
    if isinstance(mods, str):
        # A single string should be treated as a collection with one element.
        return (mods,)
    if isinstance(mods, Iterable):
        return mods
    return ()


def test_all_broker_modules_import():
    """Ensure each broker module can be imported.

    Handles the case where the module list is ``None`` or empty.
    """
    modules = _validate_module_iterable(_BROKER_MODULES)
    if not modules:
        pytest.skip("No broker modules defined to import.")
    failures: dict[str, str] = {}
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            failures[mod] = f"{type(exc).__name__}: {exc}"
    assert not failures, f"broker modules failed to import: {failures}"


def test_no_broker_subclass_is_abstract():
    """Verify that no concrete subclass of ``AbstractBroker`` remains abstract.

    Safeguards against ``None`` or empty module collections.
    """
    modules = _validate_module_iterable(_BROKER_MODULES)
    if not modules:
        pytest.skip("No broker modules defined for abstract‑method check.")
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            # Import failures are reported by the import test above.
            pass
    still_abstract = {
        cls.__name__: sorted(cls.__abstractmethods__)  # type: ignore[attr-defined]
        for cls in _all_subclasses(AbstractBroker)
        if getattr(cls, "__abstractmethods__", None)
    }
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