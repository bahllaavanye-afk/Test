"""Truncating the strategy name in `client_order_id` merged P&L across strategies.

Every desk order is tagged `qe-{strategy}-{sym}-{ts}` so fills can be
attributed back. The strategy token used to be truncated to 10 characters,
which broke three separate things:

**1. Attribution was AMBIGUOUS.** Two prefix groups collide in the registry:

    commodity_  ->  commodity_momentum, commodity_reversion, commodity_trend
    supertrend  ->  supertrend, supertrend_rsi_tv

So three distinct commodity strategies shared one P&L bucket, and every
`supertrend_rsi_tv` fill was booked against plain `supertrend`. That bad
attribution then feeds the trimmer and the auto-tuner — a strategy could be
retired for another's losses.

**2. The trimmer could never fire.** `strategy_trims.json` is keyed by whatever
`strategy_performance.json` contains — the TRUNCATED token — while the desk
checks `if sname in _trimmed` using the FULL registry name. `vol_of_vol` never
equals `vol_of_vol_timing`, so the retirement path was inert even once its
input file was finally being committed (#1191). This was the fourth broken link
in that chain, after: file never committed, trims never written, desk never
read them.

**3. `fill_tracker._parse_strategy` was fragile.** It took `split("-")[1]`,
which only works while the token is truncated and hyphen-free. It now splits
from the right like `desk_trade_sync.parse_strategy_from_coid` already did.

The fix is to stop truncating. The longest desk strategy name is 29 chars,
which yields a 48-char coid — EXACTLY Alpaca's cap, with zero margin. That is
safe today and fragile tomorrow, so the boundary is a hard test rather than a
comment: add a 30-character strategy name and CI fails here instead of the
desk silently emitting a corrupted id (`[:48]` would clip the timestamp, not
the name, since the name comes first).

Note on the transition: `strategy_performance.json` has never existed in the
repo, so there is no legacy data to migrate. Orders placed in the 7-day lookback
before this change still carry truncated ids and will produce a few short-lived
truncated keys; they receive no new fills and age out of the window.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_HERE = Path(__file__).parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_coid_test", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


dop = _load("desk_order_placer")
tracker = _load("fill_tracker")

_ALPACA_COID_CAP = dop.ALPACA_COID_CAP
_DESK_NAMES = sorted({n for d in dop.DESKS for n in d.strategy_names})


def _coid(strategy: str, symbol: str = "ABCD", ts: int | None = None) -> str:
    """The PRODUCTION constructor — never a local re-implementation.

    The first version of this file built the id itself and so passed happily
    against the truncating code it existed to catch. A guard that does not call
    the real thing is not a guard.
    """
    return dop.make_coid(strategy, symbol, ts if ts is not None else int(time.time()))


def test_there_are_desk_strategies_to_check():
    assert _DESK_NAMES, "no desk strategy names found — the guard would be vacuous"


# ── the boundary that must not be crossed silently ───────────────────────────

@pytest.mark.parametrize("name", _DESK_NAMES, ids=lambda n: n)
def test_every_strategy_name_fits_the_broker_cap(name):
    """`[:48]` clips the TIMESTAMP, not the name — a silent id corruption."""
    coid = _coid(name)
    assert len(coid) <= _ALPACA_COID_CAP, (
        f"{name} ({len(name)} chars) produces a {len(coid)}-char client_order_id. "
        f"Shorten the name, or shorten the timestamp encoding — do not let it clip."
    )


@pytest.mark.parametrize("name", _DESK_NAMES, ids=lambda n: n)
def test_every_strategy_name_round_trips(name):
    """Emit -> parse must return the name EXACTLY. This is the whole contract."""
    assert tracker._parse_strategy(_coid(name)) == name


def test_the_cap_has_no_hidden_slack():
    """Documents the real margin so a future edit knows what it is spending."""
    longest = max(_DESK_NAMES, key=len)
    assert len(_coid(longest)) == _ALPACA_COID_CAP, (
        f"margin changed: longest name {longest!r} now yields "
        f"{len(_coid(longest))} chars (cap {_ALPACA_COID_CAP})"
    )


# ── the collisions that made attribution wrong ───────────────────────────────

def test_no_two_desk_strategies_share_a_10_char_prefix_after_the_fix():
    """Not a constraint on names — proof the OLD scheme was lossy.

    These are the exact groups whose P&L was being merged.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for n in _DESK_NAMES:
        groups[n[:10]].append(n)
    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    assert collisions, (
        "expected the historical collisions to still exist in the registry — "
        "if this fails the test has lost its subject, not the bug its danger"
    )
    # and every one of them is now distinguishable
    for names in collisions.values():
        assert len({tracker._parse_strategy(_coid(n)) for n in names}) == len(names)


@pytest.mark.parametrize("a,b", [
    ("commodity_momentum", "commodity_reversion"),
    ("commodity_momentum", "commodity_trend"),
    ("supertrend", "supertrend_rsi_tv"),
])
def test_previously_colliding_pairs_are_now_distinct(a, b):
    assert tracker._parse_strategy(_coid(a)) != tracker._parse_strategy(_coid(b))


# ── parser robustness ────────────────────────────────────────────────────────

def test_a_crypto_symbol_slash_is_stripped_and_does_not_confuse_the_parser():
    assert tracker._parse_strategy(_coid("vol_of_vol_timing", "ETH/USD")) == "vol_of_vol_timing"


def test_untagged_orders_are_ignored():
    for junk in [None, "", "broker-generated-123", "manual", "qe"]:
        assert tracker._parse_strategy(junk) is None


def test_a_malformed_tag_is_ignored_not_guessed():
    assert tracker._parse_strategy("qe-onlystrategy") is None
    assert tracker._parse_strategy("qe--ABCD-123") is None


def test_the_backend_parser_agrees():
    """desk_trade_sync must attribute identically or the two ledgers diverge."""
    import sys
    sys.path.insert(0, str(_HERE.parents[2] / "backend"))
    try:
        from app.tasks.desk_trade_sync import parse_strategy_from_coid
    except Exception:  # noqa: BLE001
        pytest.skip("backend deps unavailable in this environment")
    for name in _DESK_NAMES[:25]:
        coid = _coid(name)
        assert parse_strategy_from_coid(coid, _DESK_NAMES) == tracker._parse_strategy(coid) == name


def test_historical_truncated_ids_still_parse():
    """The 7-day lookback still contains pre-change orders — must not crash."""
    assert tracker._parse_strategy("qe-vol_of_vol-ETHU-1785289292") == "vol_of_vol"
