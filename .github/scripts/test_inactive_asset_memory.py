"""`asset MKR/USD is not active` came back, and the code could not say why.

Measured live in desk run 30394875861 (2026-07-28), FIVE times in one process:

    ⚠ alpaca POST /v2/orders → 422: {"code":40010001,
        "message":"asset MKR/USD is not active"}

This class was supposedly closed by `_filter_tradable_crypto`, which drops
delisted pairs before a signal is ever computed. But the run printed no
`ⓘ skipping N non-tradable` line, so `dropped` was empty — and there was NO
way to tell which of two very different things had happened:

  A. the /v2/assets lookup failed, `_tradable_crypto_symbols()` returned None,
     and the filter fail-softly kept the whole universe (correct behaviour,
     invisible); or
  B. Alpaca's asset metadata says MKR/USD is active while its own order engine
     refuses it — in which case no amount of pre-filtering will ever help.

A fail-soft path that returns None silently is indistinguishable from a path
that had nothing to do. Same silent-miss family as the `prices:{symbol}`
topic-vs-key bug in app/tasks/CLAUDE.md. So `_tradable_crypto_symbols()` now
narrates its own failure, and the next live run will name the branch.

Independently of WHICH branch it is: the first 422 is a definitive answer, and
re-submitting the same asset for four more desks in the same process is a
guaranteed-wasted round trip plus four duplicate error lines that make the log
look like four separate problems. The rejection is now remembered for the rest
of the run.

Deliberately NOT persisted across runs: a delisting can be reversed, and a
process-lifetime memory self-heals on the next scheduled run without needing
anyone to clear state.
"""
from __future__ import annotations

import asyncio
import importlib.util
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"
_spec = importlib.util.spec_from_file_location("dop_inactive_test", _MOD)
dop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dop)  # type: ignore[union-attr]


@pytest.fixture(autouse=True)
def _clean_run_state():
    """Each test starts with a fresh process-lifetime memory."""
    getattr(dop, "_inactive_assets", set()).clear()
    yield
    getattr(dop, "_inactive_assets", set()).clear()


class _Rejection(urllib.error.HTTPError):
    def __init__(self, code: int, body: str):
        super().__init__("https://paper-api.alpaca.markets", code, "err", {}, None)
        self._body = body.encode()

    def read(self, *a, **kw):  # noqa: D102
        return self._body


def _alpaca_rejects(monkeypatch, detail: str, code: int = 422):
    def fake_urlopen(req, timeout=None):
        raise _Rejection(code, detail)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _capture_posts(monkeypatch) -> list:
    posted: list = []

    async def fake_post(path, body):
        posted.append((path, body))
        return {"id": "order-1", "status": "new"}

    monkeypatch.setattr(dop, "_alpaca_post", fake_post)
    return posted


_MKR_422 = '{"code":40010001,"message":"asset MKR/USD is not active"}'


# ── the regression: the same reject, five times in one run ───────────────────

def test_a_rejected_asset_is_not_resubmitted_by_the_next_desk(monkeypatch):
    """The whole point. Desk 1 gets the 422; desks 2..5 must not repeat it.

    Drives the REAL entry points only — no reference to the new memory — so
    on the pre-fix tree this fails on the assertion, not on an AttributeError.
    """
    _alpaca_rejects(monkeypatch, _MKR_422)
    with pytest.raises(urllib.error.HTTPError):
        dop._alpaca_post_sync("/v2/orders", {"symbol": "MKR/USD", "side": "buy"})

    posted = _capture_posts(monkeypatch)
    for _ in range(4):                       # the four later desks
        out = asyncio.run(dop._place_order("MKR/USD", "buy", 100.0, limit_price=1000.0))
        # Stronger than the previous `is None`: a known-inactive asset is a
        # deliberate skip, not a placement failure, and the two must stay
        # distinguishable so the run log does not report a decision as an error.
        assert dop._was_skipped(out), f"expected a deliberate skip, got {out!r}"
    assert posted == [], f"re-submitted a known-inactive asset: {posted}"


def test_the_skip_is_announced_not_silent(monkeypatch, capsys):
    """A silent skip would look identical to a desk that had no signal."""
    _alpaca_rejects(monkeypatch, _MKR_422)
    with pytest.raises(urllib.error.HTTPError):
        dop._alpaca_post_sync("/v2/orders", {"symbol": "MKR/USD", "side": "buy"})
    _capture_posts(monkeypatch)
    capsys.readouterr()
    asyncio.run(dop._place_order("MKR/USD", "buy", 100.0, limit_price=1000.0))
    out = capsys.readouterr().out
    assert "MKR/USD" in out and "not active" in out


def test_the_first_rejection_names_the_asset_it_is_blacklisting(monkeypatch, capsys):
    _alpaca_rejects(monkeypatch, _MKR_422)
    with pytest.raises(urllib.error.HTTPError):
        dop._alpaca_post_sync("/v2/orders", {"symbol": "MKR/USD", "side": "buy"})
    out = capsys.readouterr().out
    assert "MKR/USD marked INACTIVE" in out


# ── the blacklist must stay NARROW ───────────────────────────────────────────
# A too-eager memory is worse than none: it would strand a perfectly good
# symbol for the rest of the run on a transient or unrelated refusal.

@pytest.mark.parametrize("detail", [
    '{"code":42210000,"message":"fractional orders cannot be sold short"}',
    '{"code":42210000,"message":"asset \\"EIDO\\" cannot be sold short"}',
    '{"code":40310000,"message":"insufficient buying power"}',
    '{"code":42910000,"message":"too many requests"}',
    "",
])
def test_an_unrelated_refusal_does_not_blacklist_the_symbol(monkeypatch, detail):
    """These are retryable or side-specific — the asset itself is fine."""
    _alpaca_rejects(monkeypatch, detail)
    with pytest.raises(urllib.error.HTTPError):
        dop._alpaca_post_sync("/v2/orders", {"symbol": "UNG", "side": "sell"})

    posted = _capture_posts(monkeypatch)
    asyncio.run(dop._place_order("UNG", "buy", 100.0, limit_price=50.0))
    assert posted, "a healthy symbol was blacklisted by an unrelated 422"


def test_only_the_orders_endpoint_can_blacklist(monkeypatch):
    """A 422 from some other POST says nothing about tradability."""
    _alpaca_rejects(monkeypatch, _MKR_422)
    with pytest.raises(urllib.error.HTTPError):
        dop._alpaca_post_sync("/v2/positions/close", {"symbol": "MKR/USD"})

    posted = _capture_posts(monkeypatch)
    asyncio.run(dop._place_order("MKR/USD", "buy", 100.0, limit_price=1000.0))
    assert posted, "a non-order endpoint blacklisted the asset"


def test_other_symbols_are_unaffected(monkeypatch):
    _alpaca_rejects(monkeypatch, _MKR_422)
    with pytest.raises(urllib.error.HTTPError):
        dop._alpaca_post_sync("/v2/orders", {"symbol": "MKR/USD", "side": "buy"})

    posted = _capture_posts(monkeypatch)
    asyncio.run(dop._place_order("BTC/USD", "buy", 100.0, limit_price=60000.0))
    assert len(posted) == 1 and posted[0][1]["symbol"] == "BTC/USD"


def test_a_clean_run_blacklists_nothing(monkeypatch):
    posted = _capture_posts(monkeypatch)
    asyncio.run(dop._place_order("MKR/USD", "buy", 100.0, limit_price=1000.0))
    assert len(posted) == 1


# ── unit-level: the recorder ─────────────────────────────────────────────────

@pytest.mark.parametrize("message", [
    "asset MKR/USD is not active",
    "asset is not active",
    "ASSET MKR/USD IS NOT ACTIVE",
    "asset MKR/USD is not tradable",
])
def test_the_inactive_phrasings_are_recognised(message):
    dop._note_inactive_asset({"symbol": "MKR/USD"}, message)
    assert "MKR/USD" in dop._inactive_assets


def test_a_missing_symbol_is_not_recorded_as_empty():
    """An empty entry would blacklist nothing — but must not blacklist ''."""
    dop._note_inactive_asset({}, "asset is not active")
    assert dop._inactive_assets == set()


def test_recording_is_idempotent_and_announced_once(capsys):
    dop._note_inactive_asset({"symbol": "MKR/USD"}, "asset MKR/USD is not active")
    capsys.readouterr()
    dop._note_inactive_asset({"symbol": "MKR/USD"}, "asset MKR/USD is not active")
    assert capsys.readouterr().out == ""


def test_the_symbol_is_normalised(monkeypatch):
    dop._note_inactive_asset({"symbol": " mkr/usd "}, "asset is not active")
    assert "MKR/USD" in dop._inactive_assets
    posted = _capture_posts(monkeypatch)
    asyncio.run(dop._place_order("MKR/USD", "buy", 100.0, limit_price=1000.0))
    assert posted == []


def test_a_none_body_does_not_raise():
    dop._note_inactive_asset(None, "asset is not active")  # type: ignore[arg-type]
    assert dop._inactive_assets == set()


# ── the lookup now narrates its own failure ──────────────────────────────────
# Branch A vs branch B (see module docstring) were indistinguishable in the log.

def test_a_failed_tradable_lookup_says_so(monkeypatch, capsys):
    dop._tradable_crypto_cache = None

    async def boom(path, params=None, data_api=False):
        raise RuntimeError("alpaca 503 service unavailable")

    monkeypatch.setattr(dop, "_alpaca_get", boom)
    assert asyncio.run(dop._tradable_crypto_symbols()) is None
    out = capsys.readouterr().out
    assert "tradable-crypto lookup FAILED" in out
    assert "alpaca 503" in out, "the reason must be in the line, not just the fact"
    assert "not filtering" in out


def test_a_bad_response_shape_says_so(monkeypatch, capsys):
    """Alpaca returning an error envelope instead of a list is branch A too."""
    dop._tradable_crypto_cache = None

    async def wrong(path, params=None, data_api=False):
        return {"message": "forbidden"}

    monkeypatch.setattr(dop, "_alpaca_get", wrong)
    assert asyncio.run(dop._tradable_crypto_symbols()) is None
    out = capsys.readouterr().out
    assert "returned dict" in out and "not a list" in out


def test_a_healthy_lookup_stays_quiet(monkeypatch, capsys):
    """No news is good news — this line must not appear every run."""
    dop._tradable_crypto_cache = None

    async def ok(path, params=None, data_api=False):
        return [{"symbol": "BTC/USD", "tradable": True},
                {"symbol": "ETH/USD", "tradable": True}]

    monkeypatch.setattr(dop, "_alpaca_get", ok)
    asyncio.run(dop._tradable_crypto_symbols())
    dop._tradable_crypto_cache = None
    assert "tradable-crypto lookup" not in capsys.readouterr().out


def test_the_failure_reason_is_truncated_not_dumped(monkeypatch, capsys):
    """A stack-trace-sized exception must not flood the desk log."""
    dop._tradable_crypto_cache = None

    async def boom(path, params=None, data_api=False):
        raise RuntimeError("x" * 5000)

    monkeypatch.setattr(dop, "_alpaca_get", boom)
    asyncio.run(dop._tradable_crypto_symbols())
    assert len(capsys.readouterr().out) < 300


def test_the_fail_soft_contract_is_unchanged(monkeypatch):
    """Narrating the failure must not change what the filter DOES."""
    dop._tradable_crypto_cache = None

    async def boom(path, params=None, data_api=False):
        raise RuntimeError("down")

    monkeypatch.setattr(dop, "_alpaca_get", boom)
    universe = ["BTC/USD", "ETH/USD", "MKR/USD"]
    kept, dropped = dop._filter_tradable_crypto(
        universe, asyncio.run(dop._tradable_crypto_symbols())
    )
    assert kept == universe and dropped == []
