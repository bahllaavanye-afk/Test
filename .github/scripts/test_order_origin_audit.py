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

import asyncio
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

def test_the_audit_speaks_on_the_CLEAN_path_too(monkeypatch, capsys):
    """A guard that only prints when it fires is indistinguishable from a guard
    that stopped running. This codebase has shipped that mistake repeatedly."""
    async def fake_get(path, params=None):
        return [_o("qe-momentum-SPY-1")]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    ours, foreign, _ = asyncio.run(dop.audit_order_origins())
    assert (ours, foreign) == (1, 0)
    assert "order-origin audit" in capsys.readouterr().out


def test_a_second_writer_is_named_loudly(monkeypatch, capsys):
    """One untagged order, not part of any bulk close-all — the case the audit
    exists for. It must still be loud, and it must still name the order."""
    async def fake_get(path, params=None):
        return [_o("qe-momentum-SPY-1"), _o("agb8-runner-7", "TSLA")]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    _, foreign, _ = asyncio.run(dop.audit_order_origins())
    out = capsys.readouterr().out
    assert foreign == 1
    assert "ORDER-ORIGIN AUDIT" in out and "one at a time" in out
    assert "TSLA" in out and "agb8-runner-7" in out


def test_a_broker_failure_cannot_take_the_desk_down(monkeypatch, capsys):
    """Diagnostic only. It must never be the reason a trading run dies."""
    async def boom(path, params=None):
        raise RuntimeError("429 rate limited")
    monkeypatch.setattr(dop, "_alpaca_get", boom)
    assert asyncio.run(dop.audit_order_origins()) == (0, 0, [])
    assert "unavailable" in capsys.readouterr().out


def test_it_asks_for_all_orders_not_just_closed_ones(monkeypatch):
    """`_report_recent_closes` uses status=closed, which is right for its
    question and wrong for this one: a second writer's *open* orders are the
    ones about to move the book."""
    seen = {}

    async def fake_get(path, params=None):
        seen.update(params or {})
        return []
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    asyncio.run(dop.audit_order_origins())
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


# ── The false alarm ──────────────────────────────────────────────────────────
#
# On 2026-08-06 this audit printed "50 of 50 recent orders were NOT placed by
# this desk — a second writer is on this account" and it was escalated as an
# intruder. It was this desk's own recovery flatten: `recover_negative_cash`
# fired one run earlier and closed 17 positions via `DELETE /v2/positions`,
# which Alpaca fulfils by creating the closing orders itself — untagged.
#
# The five sampled orders, verbatim from run 31072118909:
#
#   2026-08-06T04:07:09.55037457Z USO  sell qty=24.019098096 coid='72f6155b-…'
#   2026-08-06T04:07:09.54981564Z URA  sell qty=91.937715946 coid='692927e8-…'
#   2026-08-06T04:07:09.54924723Z TSLA buy  qty=3            coid='2f8a25a2-…'
#   2026-08-06T04:07:09.54871309Z THD  sell qty=113.66       coid='63495dbe-…'
#   2026-08-06T04:07:09.54811399Z QQQ  sell qty=3.93         coid='3d611e35-…'
#
# Five orders inside 2.1 milliseconds. No strategy loop decides that way.

REAL_FLATTEN = [
    {"client_order_id": "72f6155b-5610-4fa7-857c-d6e5d56bd22b", "symbol": "USO",
     "side": "sell", "qty": "24.019098096", "status": "accepted",
     "submitted_at": "2026-08-06T04:07:09.55037457Z"},
    {"client_order_id": "692927e8-4969-45b8-b65b-74a562893146", "symbol": "URA",
     "side": "sell", "qty": "91.937715946", "status": "accepted",
     "submitted_at": "2026-08-06T04:07:09.54981564Z"},
    {"client_order_id": "2f8a25a2-ac91-4918-bd80-f43370caaa91", "symbol": "TSLA",
     "side": "buy", "qty": "3", "status": "accepted",
     "submitted_at": "2026-08-06T04:07:09.54924723Z"},
    {"client_order_id": "63495dbe-204c-441f-90b4-1f2e06a21208", "symbol": "THD",
     "side": "sell", "qty": "113.66", "status": "accepted",
     "submitted_at": "2026-08-06T04:07:09.54871309Z"},
    {"client_order_id": "3d611e35-18b9-464a-96a1-e09cdcfcbdb1", "symbol": "QQQ",
     "side": "sell", "qty": "3.93", "status": "accepted",
     "submitted_at": "2026-08-06T04:07:09.54811399Z"},
]


def test_alpaca_nanosecond_timestamps_parse():
    """Alpaca stamps nanoseconds. Unparsed, every order looks isolated and the
    false alarm comes straight back. Python 3.11 truncates these natively, so
    the shim in `_parse_ts` is insurance for older interpreters rather than
    something this suite can kill a mutation on — hence the equivalence check
    below, which holds on every version."""
    assert dop._parse_ts("2026-08-06T04:07:09.55037457Z") is not None
    assert (dop._parse_ts("2026-08-06T04:07:09.55037457Z")
            == dop._parse_ts("2026-08-06T04:07:09.550374Z"))


def test_the_real_flatten_is_recognised_as_one_bulk_action():
    assert dop.bulk_burst_count(REAL_FLATTEN) == 5


def test_the_real_flatten_does_not_report_a_second_writer(monkeypatch, capsys):
    async def fake_get(path, params=None):
        return REAL_FLATTEN
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    ours, foreign, _ = asyncio.run(dop.audit_order_origins())
    out = capsys.readouterr().out
    assert (ours, foreign) == (0, 5)          # still counted as untagged …
    assert "second writer" not in out         # … but no longer accused
    assert "close-all" in out


def test_orders_placed_one_at_a_time_still_raise_the_alarm():
    """The fix must not silence the case the audit exists for. Same five
    symbols, decided minutes apart instead of milliseconds."""
    spread = [dict(o, submitted_at=f"2026-08-06T0{4 + i}:07:09Z")
              for i, o in enumerate(REAL_FLATTEN)]
    assert dop.bulk_burst_count(spread) == 0


def test_a_real_writer_alongside_a_flatten_is_still_reported(monkeypatch, capsys):
    intruder = dict(REAL_FLATTEN[0], symbol="AAPL", side="buy",
                    client_order_id="c0ffee00-0000-0000-0000-000000000001",
                    submitted_at="2026-08-06T09:31:00Z")
    async def fake_get(path, params=None):
        return REAL_FLATTEN + [intruder]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    asyncio.run(dop.audit_order_origins())
    out = capsys.readouterr().out
    assert "ORDER-ORIGIN AUDIT" in out
    assert "one at a time" in out


def test_two_orders_are_not_a_burst():
    """A pair landing together is a coincidence, not a close-all. The floor
    keeps the exemption from swallowing ordinary paired activity."""
    pair = REAL_FLATTEN[:2]
    assert dop.bulk_burst_count(pair) == 0


def test_a_malformed_timestamp_does_not_crash_a_diagnostic():
    assert dop._parse_ts(None) is None
    assert dop._parse_ts("not-a-date") is None
    assert dop.bulk_burst_count([{"submitted_at": "nonsense"}]) == 0


# ── The recovery's promise ───────────────────────────────────────────────────

def test_recovery_does_not_promise_recovery_into_a_closed_market(monkeypatch, capsys):
    """Measured 2026-08-06: the 04:06 and 04:45 runs flattened the same 17
    positions and reported cash -$48,471.29 both times, to the cent. Closes
    submitted into a shut market cannot fill, so cash never frees."""
    monkeypatch.setattr(dop, "_alpaca_delete_sync",
                        lambda path: {"body": [{"symbol": "SPY"}] * 17})
    monkeypatch.setattr(dop, "ALPACA_PAPER_BASE", "https://paper-api.alpaca.markets")
    monkeypatch.setattr(dop, "AUTO_FLATTEN_ON_NEGATIVE_CASH", True)
    account = {"cash": -48471.29, "non_marginable_buying_power": 0.0, "buying_power": 0.0}

    asyncio.run(dop.recover_negative_cash(account, market_is_open=False))
    closed_out = capsys.readouterr().out
    assert "next run trades normally" not in closed_out
    assert "market is CLOSED" in closed_out

    asyncio.run(dop.recover_negative_cash(account, market_is_open=True))
    assert "next run trades normally" in capsys.readouterr().out


def test_a_bug_in_the_burst_analysis_cannot_take_the_desk_down(monkeypatch, capsys):
    """The fetch was already fail-soft; the burst logic added after it was not,
    and a missing `import re` in it would have crashed a live trading run."""
    async def fake_get(path, params=None):
        return REAL_FLATTEN

    def boom(*a, **k):
        raise RuntimeError("synthetic burst failure")
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    monkeypatch.setattr(dop, "bulk_burst_count", boom)
    ours, foreign, _ = asyncio.run(dop.audit_order_origins())
    assert (ours, foreign) == (0, 5)
    assert "burst analysis unavailable" in capsys.readouterr().out


def test_the_alarm_names_its_candidates_instead_of_crying_intruder(monkeypatch, capsys):
    """The old headline said only "a second writer is on this account" and was
    escalated as an intruder. The candidates must be named and the limit stated.

    This previously asserted the message said orders "CANNOT be told apart".
    That blanket claim is now itself too strong and was removed deliberately:
    an equity market+GTC order DOES carry the exit loop's signature, so exits
    and openers CAN be separated. What genuinely cannot be separated is WHICH
    backend — `9jz0` and `agb8` run identical code. The assertion below pins
    that narrower, true limit instead of the old over-claim."""
    async def fake_get(path, params=None):
        return [_o("qe-momentum-SPY-1"), _o("agb8-runner-7", "TSLA")]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    asyncio.run(dop.audit_order_origins())
    out = capsys.readouterr().out
    assert "PositionMonitor" in out
    assert "agb8" in out
    assert "not WHICH backend" in out


# ── Narrowing "these cannot be told apart" ───────────────────────────────────
#
# Live 2026-08-07 00:29, run 31134848144 — the alarm path firing for real:
#
#   ⚠️ ORDER-ORIGIN AUDIT: 10 of 50 recent orders are untagged AND were not part
#   of a bulk close-all ... These CANNOT be told apart until backend orders carry
#   a client_order_id
#
# True, but weaker than the evidence allows. `PositionMonitor._close_position`
# (position_monitor.py:340) submits order_type="market", time_in_force="GTC".
# The desk submits equities as `"gtc" if is_crypto else "day"` — so on an EQUITY,
# market+GTC is the exit loop's shape and day is not. That separates "a position
# was closed" from "something opened one", which is the part worth acting on.
#
# It does NOT say which backend: 9jz0 and agb8 run the same code. Only a
# client_order_id answers that, and that file is Do-Not-Modify. [P0] stands.

def _eq(tif="day", typ="limit", symbol="SPY"):
    return {"client_order_id": "x", "symbol": symbol, "side": "sell", "qty": "1",
            "status": "filled", "submitted_at": "2026-08-07T00:00:00Z",
            "time_in_force": tif, "type": typ}


def test_the_exit_loop_signature_is_recognised():
    assert dop.looks_like_a_position_monitor_exit(_eq(tif="gtc", typ="market")) is True


def test_a_desk_style_equity_order_is_not_an_exit():
    """The desk uses day orders on equities, limit-first."""
    assert dop.looks_like_a_position_monitor_exit(_eq(tif="day", typ="limit")) is False
    assert dop.looks_like_a_position_monitor_exit(_eq(tif="day", typ="market")) is False


def test_a_gtc_limit_is_not_an_exit():
    """PositionMonitor submits MARKET. GTC alone is not the signature."""
    assert dop.looks_like_a_position_monitor_exit(_eq(tif="gtc", typ="limit")) is False


def test_crypto_is_excluded_because_the_desk_also_uses_gtc():
    """On crypto the desk itself submits GTC, so the signature cannot
    distinguish — claiming it could would be a false attribution."""
    assert dop.looks_like_a_position_monitor_exit(
        _eq(tif="gtc", typ="market", symbol="BTC/USD")) is False


def test_a_lone_order_is_not_in_a_burst():
    assert dop._in_a_burst(REAL_FLATTEN[0], [REAL_FLATTEN[0]]) is False


def test_burst_members_are_identified_individually():
    """`bulk_burst_count` gives a total; attributing the remainder needs to know
    which specific orders stood alone."""
    for o in REAL_FLATTEN:
        assert dop._in_a_burst(o, REAL_FLATTEN) is True


def test_the_audit_separates_exits_from_openers(monkeypatch, capsys):
    opener = _eq(tif="day", typ="limit", symbol="AAPL")
    opener["client_order_id"] = "untagged-1"
    opener["submitted_at"] = "2026-08-07T01:00:00Z"
    exit_order = _eq(tif="gtc", typ="market", symbol="MSFT")
    exit_order["client_order_id"] = "untagged-2"
    exit_order["submitted_at"] = "2026-08-07T02:00:00Z"

    async def fake_get(path, params=None):
        return [opener, exit_order]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    asyncio.run(dop.audit_order_origins())
    out = capsys.readouterr().out
    assert "1 carry the backend exit loop's signature" in out
    assert "1 do not" in out
    assert "worth explaining" in out


def test_all_exits_reads_as_our_own_backend(monkeypatch, capsys):
    a = _eq(tif="gtc", typ="market", symbol="MSFT"); a["client_order_id"] = "u1"
    a["submitted_at"] = "2026-08-07T02:00:00Z"
    async def fake_get(path, params=None):
        return [a]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    asyncio.run(dop.audit_order_origins())
    out = capsys.readouterr().out
    assert "all of them look like position exits" in out
    assert "worth explaining" not in out


def test_the_which_backend_limit_is_always_stated():
    """The narrowing must not be mistaken for identification. 9jz0 and agb8
    produce identical signatures."""
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    assert "narrows WHAT happened, not WHICH backend" in src
