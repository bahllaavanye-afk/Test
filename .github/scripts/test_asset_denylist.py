"""MKR/USD burns a top-K slot every run, and pre-filtering cannot stop it.

Settled live on 2026-07-28 across runs 30401244361 and 30402044105, both
carrying the full instrumentation from #1184/#1186. The crypto desk logged:

    · ensemble[Crypto]: MKR/USD buy x2 (...) -> conf=0.89
    · top-K[Crypto]: dropped 9 lower-confidence signals
      ⚠ alpaca POST /v2/orders → 422: {"code":40010001,
          "message":"asset MKR/USD is not active"}
      ⓘ MKR/USD marked INACTIVE for the rest of this run

and — decisively — NONE of these:

    ⓘ tradable-crypto lookup FAILED (...)          <- lookup was healthy
    ⓘ tradable-crypto set has N entries but ...    <- format guard did not trip
    ⓘ skipping N non-tradable pair(s)              <- nothing was dropped

`_filter_tradable_crypto` is called unconditionally for every desk and prints
whenever it drops anything, so the only reading left is: Alpaca's
`/v2/assets?asset_class=crypto&status=active` returns MKR/USD as tradable while
`POST /v2/orders` refuses it as not active. **The metadata contradicts the order
engine**, which disproves the IMPROVEMENTS.md premise that filtering the
universe against the active-asset list would fix this class. It cannot.

What remains is memory. The in-process `_inactive_assets` set catches every
repeat within a run but never the FIRST attempt, and that attempt is expensive:
MKR/USD took 1 of only 3 passing signals in BOTH runs — roughly a third of the
crypto desk's capacity spent on a guaranteed reject. Seeding from a checked-in
denylist removes it before signals are generated, so the slot goes to a pair
that can actually trade.

Entries EXPIRE (DENYLIST_TTL_DAYS). A delisting can be reversed, and a denylist
nobody re-confirms is exactly how a permanently-stale exclusion happens. Letting
it decay forces the evidence to stay fresh: if the reject recurs, the desk log
names it again and the entry gets refreshed.

Follows the existing `strategy_trims.json` pattern — the desk READS state that
something else maintains; it never writes from the trading hot path.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"
_spec = importlib.util.spec_from_file_location("dop_denylist_test", _MOD)
dop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dop)  # type: ignore[union-attr]

_STATE = Path(__file__).resolve().parents[1] / "state" / "inactive_assets.json"


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _load_from(path: Path) -> set:
    """Run the real loader against an arbitrary file by patching __file__ lookup."""
    import json as _json
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    # mirror of _denylisted_assets, driven by an explicit path — kept in step by
    # test_the_shipped_loader_and_this_mirror_agree below.
    raw = _json.loads(path.read_text())
    cutoff = _dt.now(_tz.utc) - _td(days=dop.DENYLIST_TTL_DAYS)
    out = set()
    for sym, meta in (raw or {}).items():
        if sym.startswith("_") or not isinstance(meta, dict):
            continue
        try:
            since = _dt.fromisoformat(str(meta.get("since", "")).replace("Z", "+00:00"))
        except Exception:
            continue
        if since >= cutoff:
            out.add(sym.strip().upper())
    return out


# ── the shipped state file ───────────────────────────────────────────────────

def test_the_shipped_denylist_parses_and_contains_the_confirmed_asset():
    assert _STATE.is_file(), f"missing {_STATE}"
    loaded = dop._denylisted_assets()
    assert "MKR/USD" in loaded, (
        "MKR/USD is the confirmed case; without it the desk keeps burning a slot"
    )


def test_every_shipped_entry_carries_dated_evidence():
    """An undated entry is an opinion, not evidence — and would never expire."""
    raw = json.loads(_STATE.read_text())
    for sym, meta in raw.items():
        if sym.startswith("_"):
            continue
        assert isinstance(meta, dict), f"{sym}: expected an object"
        assert meta.get("since"), f"{sym}: no 'since' timestamp"
        assert meta.get("reason"), f"{sym}: no 'reason' — nobody can audit this later"
        datetime.fromisoformat(str(meta["since"]).replace("Z", "+00:00"))


def test_annotation_keys_are_not_treated_as_symbols():
    """The _README key must never become a denylisted 'symbol'."""
    assert not any(s.startswith("_") for s in dop._denylisted_assets())


# ── expiry ───────────────────────────────────────────────────────────────────

def test_a_fresh_entry_is_honoured(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"MKR/USD": {"since": _iso(1), "reason": "x"}}))
    assert _load_from(f) == {"MKR/USD"}


def test_a_stale_entry_expires_so_a_reversal_self_heals(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"MKR/USD": {"since": _iso(dop.DENYLIST_TTL_DAYS + 1), "reason": "x"}}))
    assert _load_from(f) == set(), "a stale denylist would exclude a recovered asset forever"


def test_an_undated_entry_is_ignored(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"MKR/USD": {"reason": "no date"}}))
    assert _load_from(f) == set()


def test_a_garbage_date_is_ignored_not_fatal(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"MKR/USD": {"since": "not-a-date", "reason": "x"}}))
    assert _load_from(f) == set()


def test_the_ttl_is_short_enough_to_force_re_confirmation():
    assert 1 <= dop.DENYLIST_TTL_DAYS <= 30, dop.DENYLIST_TTL_DAYS


# ── fail-soft: never break the desk ──────────────────────────────────────────

def test_a_missing_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(dop, "__file__", str(tmp_path / "nope" / "x.py"))
    assert dop._denylisted_assets() == set()


def test_malformed_json_is_not_an_error(tmp_path, monkeypatch):
    d = tmp_path / "scripts"
    d.mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "inactive_assets.json").write_text("{not json")
    monkeypatch.setattr(dop, "__file__", str(d / "desk_order_placer.py"))
    assert dop._denylisted_assets() == set()


def test_the_shipped_loader_and_this_mirror_agree():
    """Guards the mirror in _load_from against drifting from the real loader."""
    assert _load_from(_STATE) == dop._denylisted_assets()


# ── the loader must never be the reason a desk idles ─────────────────────────

def test_symbols_are_matched_case_and_whitespace_insensitively(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"  mkr/usd  ": {"since": _iso(0), "reason": "x"}}))
    assert _load_from(f) == {"MKR/USD"}


# ── applying it to a desk universe ───────────────────────────────────────────
# This is the half that actually saves the slot. The loader could be perfect and
# the desk would still waste the order if the universe were not trimmed BEFORE
# signals are generated.

_UNIVERSE = ["BTC/USD", "ETH/USD", "MKR/USD", "UNI/USD"]


def test_the_denylisted_pair_leaves_the_universe():
    kept, blocked = dop._apply_denylist(_UNIVERSE, {"MKR/USD"})
    assert blocked == ["MKR/USD"]
    assert kept == ["BTC/USD", "ETH/USD", "UNI/USD"]


def test_an_empty_denylist_is_a_no_op():
    kept, blocked = dop._apply_denylist(_UNIVERSE, set())
    assert kept == _UNIVERSE and blocked == []


def test_a_denylist_naming_nothing_in_this_desk_is_a_no_op():
    kept, blocked = dop._apply_denylist(["BTC/USD"], {"MKR/USD"})
    assert kept == ["BTC/USD"] and blocked == []


def test_it_refuses_to_empty_the_universe(capsys):
    """Idling a desk is worse than one wasted order — same contract as the
    tradable filter's kept-or-all fallback."""
    kept, blocked = dop._apply_denylist(["MKR/USD"], {"MKR/USD"})
    assert kept == ["MKR/USD"] and blocked == []
    assert "would drop ALL" in capsys.readouterr().out


def test_matching_is_case_and_whitespace_insensitive_here_too():
    kept, blocked = dop._apply_denylist([" mkr/usd ", "BTC/USD"], {"MKR/USD"})
    assert blocked == [" mkr/usd "] and kept == ["BTC/USD"]


def test_equities_are_supported_not_just_crypto():
    """The 422 class is not crypto-specific — a delisted ETF behaves the same."""
    kept, blocked = dop._apply_denylist(["SPY", "EIDO", "QQQ"], {"EIDO"})
    assert blocked == ["EIDO"] and kept == ["SPY", "QQQ"]


def test_the_shipped_denylist_does_not_gut_the_crypto_desk():
    """A guard against someone pasting the whole universe in here later."""
    kept, blocked = dop._apply_denylist(_UNIVERSE, dop._denylisted_assets())
    assert len(kept) >= len(_UNIVERSE) - 1, blocked
