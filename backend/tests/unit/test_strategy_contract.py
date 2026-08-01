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
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field, ValidationError, validator

os.environ.setdefault("SECRET_KEY", "test-secret-key-32-bytes-hex-xxxxxx")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_contract.db")

from app.strategies import STRATEGY_REGISTRY  # noqa: E402

# CONTRACT DEBT (2026-07-11 audit: 97/115 clean): these strategies either do
# BLOCKING network I/O inside analyze() (hang the desk loop when the source is
# slow/unreachable — the observed 2-minute desk freezes) or raise instead of
# returning None. Quarantined = skipped here AND flagged in IMPROVEMENTS.md.
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
    """Kill all outbound network INSTANTLY — including yfinance's.

    Patching `socket` alone did not work, and the cost was severe. yfinance
    fetches through **curl_cffi**, which talks to libcurl directly and never
    touches Python's socket module — a fact `app/strategies/_failsoft.py`
    already documents ("via curl_cffi, which bypasses Python's socket module,
    so socket-level network kills don't even reach it"). So every fetching
    strategy in this file was doing REAL network I/O and real retry-backoff,
    despite the fixture's name.

    Measured before this fix: **7m15s wall, 5.75s CPU** for this file alone —
    i.e. ~99% of it was waiting on Yahoo, and in CI (`--dist loadfile`) all 115
    parametrised cases land on ONE worker, serialised, while three sit idle.
    It also meant a Yahoo outage could turn any PR red for reasons unrelated to
    the diff.

    Blocking libcurl too makes the test hermetic AND fast, and it does not
    weaken the contract — the contract is "fail soft when the data source is
    unavailable", which is precisely the condition being simulated. It is now
    actually simulated instead of merely intended.
    """
    def _blocked(*a, **k):
        raise OSError("network disabled by strategy contract test")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=True)

    # The seam that actually matters for yfinance.
    try:
        import curl_cffi.requests as _curl_requests
    except Exception:            # pragma: no cover — dep absent, socket patch stands
        return
    monkeypatch.setattr(_curl_requests.Session, "request", _blocked, raising=False)
    for _fn in ("get", "post", "request"):
        monkeypatch.setattr(_curl_requests, _fn, _blocked, raising=False)


# ---------------------------------------------------------------------------
# Pydantic schema for the signal objects returned by strategies.
# ---------------------------------------------------------------------------
class SignalSchema(BaseModel):
    """Validated representation of a strategy signal.

    Attributes
    ----------
    side : Literal['buy', 'sell']
        The intended trade direction. Must be lower‑case.
    confidence : float
        Confidence score in the range [0.0, 1.0] indicating the strength of the
        signal.
    """

    side: Literal["buy", "sell"] = Field(
        ...,
        description="Trade direction: either 'buy' or 'sell'.",
        examples=["buy"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 (no confidence) and 1 (full confidence).",
        examples=[0.85],
    )

    @validator("side", pre=True)
    def normalize_side(cls, v: str) -> str:
        """Ensure side is a lower‑case string."""
        if not isinstance(v, str):
            raise ValueError("side must be a string")
        v_norm = v.strip().lower()
        if v_norm not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        return v_norm

    class Config:
        schema_extra = {
            "example": {"side": "buy", "confidence": 0.92},
        }


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
        pytest.fail(f"{name}: analyze() raised {type(exc).__name__}: {str(exc)[:120]} "
                    f"— strategies must catch and return None")
        return

    if sig is None:
        return  # no setup — a valid, honest answer

    # Validate the returned signal against the schema.
    try:
        SignalSchema.parse_obj(sig)
    except ValidationError as ve:
        pytest.fail(f"{name}: signal validation failed: {ve}")

    # Existing assertions remain for clarity.
    side = str(getattr(sig, "side", "")).lower()
    assert side in ("buy", "sell"), f"{name}: bad side {side!r}"
    conf = getattr(sig, "confidence", None)
    assert conf is not None and 0.0 <= float(conf) <= 1.0, f"{name}: bad confidence {conf!r}"


def test_quarantine_names_are_real():
    """Every quarantined name must exist in the registry (stale entries out)."""
    unknown = QUARANTINED - set(STRATEGY_REGISTRY)
    assert not unknown, f"QUARANTINED has unknown strategies: {sorted(unknown)}"


def test_the_network_kill_actually_reaches_yfinance(no_network):
    """The socket patch alone did NOT stop yfinance, and nothing said so.

    yfinance fetches through curl_cffi → libcurl, never touching Python's
    socket module. So this file ran 115 strategies against the REAL Yahoo API
    while its fixture was named `no_network`: 7m15s wall for 5.75s of CPU, all
    of it retry-backoff, and any Yahoo outage could redden an unrelated PR.

    A silent regression here is invisible — the suite would still pass, just
    slowly and non-deterministically — so it is asserted directly.
    """
    curl_requests = pytest.importorskip("curl_cffi.requests")

    with pytest.raises(OSError):
        curl_requests.Session().request("GET", "https://query1.finance.yahoo.com/")

    with pytest.raises(OSError):
        curl_requests.get("https://query1.finance.yahoo.com/")

    # And the plain-socket path is still blocked for non-curl fetchers.
    with pytest.raises(OSError):
        socket.getaddrinfo("query1.finance.yahoo.com", 443)