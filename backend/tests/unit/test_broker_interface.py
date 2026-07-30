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
from typing import Dict, Set

import pytest
from pydantic import BaseSettings, Field, SecretStr, validator

# -------------------------------------------------------------------------
# Pydantic schema for test environment configuration
# -------------------------------------------------------------------------

class TestEnvSettings(BaseSettings):
    """Environment configuration used by broker interface tests.

    Attributes
    ----------
    secret_key: SecretStr
        32‑byte (64‑hex‑character) secret key required by the application.
        Example: ``"a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"``.
    database_url: str
        SQLAlchemy compatible database URL for the test SQLite database.
        Example: ``"sqlite+aiosqlite:///./test_broker_iface.db"``.
    """

    secret_key: SecretStr = Field(
        ...,
        description="32‑byte (64‑hex‑character) secret key for cryptographic operations.",
        example="a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
    )
    database_url: str = Field(
        ...,
        description="SQLAlchemy database URL pointing to the test SQLite instance.",
        example="sqlite+aiosqlite:///./test_broker_iface.db",
    )

    @validator("secret_key")
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        if len(raw) != 64 or any(c not in "0123456789abcdefABCDEF" for c in raw):
            raise ValueError("SECRET_KEY must be exactly 64 hexadecimal characters")
        return v

    @validator("database_url")
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("sqlite"):
            raise ValueError("Only SQLite URLs are allowed for test environments")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


# Ensure required environment variables have sensible defaults for test runs
os.environ.setdefault("SECRET_KEY", "test-secret-key-32-bytes-hex-xxxxxx")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_broker_iface.db")

# Load and validate the settings; will raise if defaults are invalid
settings = TestEnvSettings()


from app.brokers.base import AbstractBroker  # noqa: E402

_BROKER_MODULES = [
    "app.brokers.alpaca",
    "app.brokers.binance",
    "app.brokers.polymarket",
    "app.brokers.tradestation",
]


def _all_subclasses(cls: type) -> Set[type]:
    """Recursively collect all subclasses of ``cls``."""
    out: Set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def test_all_broker_modules_import() -> None:
    """All broker modules must import without raising exceptions."""
    failures: Dict[str, str] = {}
    for mod in _BROKER_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            failures[mod] = f"{type(exc).__name__}: {exc}"
    assert not failures, f"broker modules failed to import: {failures}"


def test_no_broker_subclass_is_abstract() -> None:
    """Every concrete broker must implement all abstract methods."""
    for mod in _BROKER_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 — reported by the import test above
            pass
    still_abstract = {
        cls.__name__: sorted(cls.__abstractmethods__)  # type: ignore[attr-defined]
        for cls in _all_subclasses(AbstractBroker)
        if getattr(cls, "__abstractmethods__", None)
    }
    assert not still_abstract, (
        "broker classes missing interface methods (un‑instantiable, silently "
        f"disables the broker): {still_abstract}"
    )


def test_alpaca_broker_declares_every_interface_method_directly() -> None:
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