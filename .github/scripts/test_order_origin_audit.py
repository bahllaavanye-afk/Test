"""A second backend has been trading this Alpaca account, and no run said so.

`quantedge-api-agb8.onrender.com` — re-verified live 2026-08-05 19:00:

    background_tasks: {"running": 11, "total": 11}
    alpaca:           {"ok": true, "note": "connected"}
    strategies:       {"count": 113}
    database:         {"ok": false, "error": "Name or service not known"}

It cannot record what it does, but it can still place orders — and it moves the
equity, buying power and positions that Kelly sizing, the daily loss cap and
`is_risk_reducing` all read. Suspending it is an operator action and has been
item #1 on that list for days.

What code can do is stop it being rediscovered by hand every session. The only
origin report that existed (`_report_recent_closes`) ran solely when the daily
loss cap tripped *on a flat book* — a corner almost never reached — so on every
ordinary run a second writer was invisible.

`client_order_id` is the discriminator, and it is the right one because it lives
at the broker: it survives the backend DB being on its ephemeral sqlite
fallback, which is where it has been all week.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SECRET_KEY", "x" * 64)
os.environ.setdefault("DATABASE_URL", "")
import desk_order_placer as dop  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent


def _o(coid, symbol="SPY"):
    return {"client_order_id": coid, "symbol": symbol, "side": "buy",
            "qty": "1", "status": "filled", "submitted_at": "2026-08-05T18:00:00Z"}


# ── Classification ───────────────────────────────────────────────────────────

def test_our_own_orders_are_recognised():
    ours, foreign, sample = dop.summarise_origins([
        _o("qe-momentum-SPY-1754400000"), _o("qe-low_volatility-BAC-1754400001")])
    assert (ours, foreign, sample) == (2, 0, [])


def test_a_foreign_order_is_counted_and_sampled():
    ours, foreign, sample = dop.summarise_origins([
        _o("qe-momentum-SPY-1"), _o("bot-run-9f3a2", "NVDA")])
    assert (ours, foreign) == (1, 1)
    assert sample[0]["symbol"] == "NVDA"


def test_an_order_with_no_client_id_is_foreign_not_ours():
    """Alpaca auto-liquidation and hand-placed orders carry no coid. Defaulting
    those to 'ours' would hide exactly the writer this exists to find."""
    ours, foreign, _ = dop.summarise_origins([_o(None), _o("")])
    assert (ours, foreign) == (0, 2)


def test_a_lookalike_prefix_is_not_ours():
    """`qe` without the hyphen is a different tag. The check is
    `startswith("qe-")` and it must stay that literal."""
    ours, foreign, _ = dop.summarise_origins([_o("qeX-momentum-SPY-1")])
    assert (ours, foreign) == (0, 1)


def test_empty_and_none_inputs_do_not_raise():
    """A broker returning nothing is not evidence of a clean account, and this
    must not crash the desk on the way to saying so."""
    assert dop.summarise_origins([]) == (0, 0, [])
    assert dop.summarise_origins(None) == (0, 0, [])


def test_the_sample_is_bounded():
    """50 foreign orders must not print 50 lines into every desk log."""
    ours, foreign, sample = dop.summarise_origins([_o(f"other-{i}") for i in range(40)])
    assert foreign == 40
    assert len(sample) == 5


# ── The audit ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_audit_speaks_on_the_CLEAN_path_too(monkeypatch, capsys):
    """A guard that only prints when it fires is indistinguishable from a guard
    that stopped running. This codebase has shipped that mistake repeatedly."""
    async def fake_get(path, params=None):
        return [_o("qe-momentum-SPY-1")]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    ours, foreign, _ = await dop.audit_order_origins()
    assert (ours, foreign) == (1, 0)
    assert "order-origin audit" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_second_writer_is_named_loudly(monkeypatch, capsys):
    async def fake_get(path, params=None):
        return [_o("qe-momentum-SPY-1"), _o("agb8-runner-7", "TSLA")]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    _, foreign, _ = await dop.audit_order_origins()
    out = capsys.readouterr().out
    assert foreign == 1
    assert "second writer" in out
    assert "TSLA" in out and "agb8-runner-7" in out


@pytest.mark.asyncio
async def test_a_broker_failure_cannot_take_the_desk_down(monkeypatch, capsys):
    """Diagnostic only. It must never be the reason a trading run dies."""
    async def boom(path, params=None):
        raise RuntimeError("429 rate limited")
    monkeypatch.setattr(dop, "_alpaca_get", boom)
    assert await dop.audit_order_origins() == (0, 0, [])
    assert "unavailable" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_it_asks_for_all_orders_not_just_closed_ones(monkeypatch):
    """`_report_recent_closes` uses status=closed, which is right for its
    question and wrong for this one: a second writer's *open* orders are the
    ones about to move the book."""
    seen = {}

    async def fake_get(path, params=None):
        seen.update(params or {})
        return []
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    await dop.audit_order_origins()
    assert seen.get("status") == "all"


# ── The call site ────────────────────────────────────────────────────────────

def test_the_desk_actually_runs_the_audit_every_run():
    """Not inside the loss-cap branch, which is where the only previous origin
    report lived and why a second writer went unreported for days."""
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    stage = src.split('"Fetch account and market bars"', 1)[1][:3000]
    assert "audit_order_origins()" in stage, (
        "the account stage never calls audit_order_origins()")
    # It must precede the sizing inputs it exists to warn about.
    assert stage.index("audit_order_origins()") < stage.index("recover_negative_cash")
