"""Strategy contract: every registry strategy is safe to run on a desk.

Contract, enforced per strategy over the whole STRATEGY_REGISTRY:
  1. analyze(df, symbol) returns a Signal (valid side + confidence) or None
  2. it finishes fast — no hangs (a strategy once froze a desk run for 2
     minutes doing network I/O in analyze())
  3. it NEVER crashes the caller when the network is unavailable — sockets
     are blocked below, so any strategy that fetches must catch and return
     None. Strategies currently known to raise instead are quarantined in
     KNOWN_RAISERS; that list may only shrink.

This is the test that catches all 90+ strategies at once when someone adds a
module-level fetch, an unguarded HTTP call, or a hanging loop.
"""
from __future__ import annotations

import asyncio
import os
import socket

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-32-bytes-hex-xxxxxx")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_contract.db")

from app.strategies import STRATEGY_REGISTRY  # noqa: E402

# CONTRACT DEBT (2026-07-11 audit: 97/115 clean): these strategies either do
# BLOCKING network I/O inside analyze() (hang the desk loop when the source is
# slow/unreachable — the observed 2-minute desk freezes) or raise instead
# of returning None. Quarantined = skipped here AND flagged in IMPROVEMENTS.md.
# Fix = fail-soft fetch with a hard timeout, then REMOVE from this list.
# Adding a name in a PR is a red flag; this list may only shrink.
QUARANTINED: set[str] = set()
# 2026-07-15: EMPTY. The last 6 (yfinance retry-sleeps offline) now go through
# app/strategies/_failsoft.apply_hard_budget — analyze runs in a worker thread
# and returns None past STRATEGY_ANALYZE_BUDGET_S (3.5s default), so the 5s
# contract budget holds even when the blocking fetch can't be cancelled.

PER_STRATEGY_TIMEOUT_S = 5.0


def _bars(n: int = 300) -> pd.DataFrame:
    rs = np.random.RandomState(7)
    close = 100 * np.exp(np.cumsum(rs.normal(0.0003, 0.015, n)))
    df = pd.DataFrame(
        {
            "open": close * (1 + rs.normal(0, 0.002, n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rs.uniform(1e6, 5e6, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    df.attrs["iv_rank"] = 60.0  # options-income strategies gate on this
    return df


@pytest.fixture()
def no_network(monkeypatch):
    """Kill all outbound network instantly — DNS and connects fail — while
    leaving socketpair() intact (asyncio's event loop needs it internally)."""
    def _blocked(*a, **k):
        raise OSError("network disabled by strategy contract test")
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=True)


_LOADED = sorted(n for n, c in STRATEGY_REGISTRY.items() if c is not None)


def test_registry_is_populated():
    assert len(_LOADED) >= 80, f"registry unexpectedly small: {len(_LOADED)}"


@pytest.mark.parametrize("name", _LOADED)
def test_strategy_honors_contract(name: str, no_network):
    if name in QUARANTINED:
        pytest.skip("quarantined: blocking I/O or raise in analyze() — see QUARANTINED")
    cls = STRATEGY_REGISTRY[name]
    try:
        strat = cls()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{name}: constructor raised {type(exc).__name__}: {exc}")

    df = _bars()

    async def run():
        return await asyncio.wait_for(strat.analyze(df, "SPY"), PER_STRATEGY_TIMEOUT_S)

    try:
        sig = asyncio.run(run())
    except asyncio.TimeoutError:
        pytest.fail(f"{name}: analyze() exceeded {PER_STRATEGY_TIMEOUT_S}s (hang)")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"{name}: analyze() raised {type(exc).__name__}: {str(exc)[:120]} "
            f"— strategies must catch and return None"
        )
        return

    if sig is None:
        return  # no setup — a valid, honest answer
    side = str(getattr(sig, "side", "")).lower()
    assert side in ("buy", "sell"), f"{name}: bad side {side!r}"
    conf = getattr(sig, "confidence", None)
    assert conf is not None and 0.0 <= float(conf) <= 1.0, f"{name}: bad confidence {conf!r}"


@pytest.mark.parametrize("name", _LOADED)
def test_strategy_analyze_none_inputs(name: str, no_network):
    """Ensure strategies gracefully handle None inputs without raising."""
    if name in QUARANTINED:
        pytest.skip("quarantined")
    cls = STRATEGY_REGISTRY[name]
    strat = cls()
    async def run():
        return await asyncio.wait_for(strat.analyze(None, None), PER_STRATEGY_TIMEOUT_S)
    try:
        result = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{name}: analyze(None, None) raised {type(exc).__name__}: {exc}")
    else:
        assert result is None, f"{name}: expected None for None inputs, got {result}"


@pytest.mark.parametrize("name", _LOADED)
def test_strategy_analyze_empty_dataframe(name: str, no_network):
    """Verify that an empty DataFrame does not cause crashes or unexpected signals."""
    if name in QUARANTINED:
        pytest.skip("quarantined")
    cls = STRATEGY_REGISTRY[name]
    strat = cls()
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    async def run():
        return await asyncio.wait_for(strat.analyze(empty_df, "SPY"), PER_STRATEGY_TIMEOUT_S)
    try:
        result = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{name}: analyze(empty_df) raised {type(exc).__name__}: {exc}")
    else:
        # An empty input should result in None, not a malformed signal
        assert result is None, f"{name}: expected None for empty DataFrame, got {result}"


@pytest.mark.parametrize("name", _LOADED)
def test_strategy_analyze_one_row_dataframe(name: str, no_network):
    """Check off‑by‑one handling: a single‑row DataFrame should be processed without error."""
    if name in QUARANTINED:
        pytest.skip("quarantined")
    cls = STRATEGY_REGISTRY[name]
    strat = cls()
    one_row_df = _bars(1)
    async def run():
        return await asyncio.wait_for(strat.analyze(one_row_df, "SPY"), PER_STRATEGY_TIMEOUT_S)
    try:
        result = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{name}: analyze(one_row_df) raised {type(exc).__name__}: {exc}")
    else:
        # Result may be None or a valid signal; if a signal, validate its fields
        if result is not None:
            side = str(getattr(result, "side", "")).lower()
            assert side in ("buy", "sell"), f"{name}: bad side {side!r}"
            conf = getattr(result, "confidence", None)
            assert conf is not None and 0.0 <= float(conf) <= 1.0, f"{name}: bad confidence {conf!r}"


def test_quarantine_names_are_real():
    """Every quarantined name must exist in the registry (stale entries out)."""
    unknown = QUARANTINED - set(STRATEGY_REGISTRY)
    assert not unknown, f"QUARANTINED has unknown strategies: {sorted(unknown)}"