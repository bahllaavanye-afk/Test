"""The Indian session is only worth reading if the desk actually reads it.

Three times this week a feature was defined, tested, merged, and called by
nothing — `rank_within_categories` was the last one, and it shipped green.
`test_the_desk_actually_applies_the_tilt` is the guard that makes this one
different, and it is the most important test in the file: everything else here
is arithmetic that would still be correct in a module nobody imports.

The second theme is that an absent read must never look like a flat one. A
symbol with no usable data, a session too old to be "overnight", and a session
that genuinely moved 0.02% are three different facts, and all three would
serialise to `tilt: 0.0` if the code were careless. Each is pinned separately.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SECRET_KEY", "x" * 64)
os.environ.setdefault("DATABASE_URL", "")

import india_nse_signal as ins  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
NOW = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)   # a US session, NSE closed


def _closes(*pairs) -> list[tuple[date, float]]:
    return [(date.fromisoformat(d), c) for d, c in pairs]


def _up_2pct(day: str = "2026-08-05") -> list[tuple[date, float]]:
    prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    return _closes((prev, 100.0), (day, 102.0))


# ── The move itself ──────────────────────────────────────────────────────────

def test_move_is_the_last_two_closes():
    assert ins.session_move_pct(_up_2pct())[1] == pytest.approx(2.0)


def test_one_close_is_not_a_move():
    """None, not 0.0. A single bar means "cannot say", and a 0.0 here would be
    published as a flat Indian session — a claim the data does not support."""
    assert ins.session_move_pct(_closes(("2026-08-05", 100.0))) is None
    assert ins.session_move_pct([]) is None


def test_unsorted_input_still_uses_the_two_most_recent():
    out_of_order = _closes(("2026-08-05", 102.0), ("2026-08-03", 90.0),
                           ("2026-08-04", 100.0))
    day, move = ins.session_move_pct(out_of_order)
    assert day == date(2026, 8, 5)
    assert move == pytest.approx(2.0)


def test_zero_and_negative_closes_are_discarded():
    """A 0.0 close is a yfinance artefact, and dividing by it raises."""
    assert ins.session_move_pct(_closes(("2026-08-04", 0.0),
                                        ("2026-08-05", 102.0))) is None


# ── The tilt ─────────────────────────────────────────────────────────────────

def test_tilt_tracks_direction():
    assert ins.tilt_from_move(2.0) > 0
    assert ins.tilt_from_move(-2.0) < 0


def test_tilt_is_capped_in_both_directions():
    """A 20% day is a crash or a limit-up, and neither should hand a signal an
    arbitrarily large confidence boost."""
    assert ins.tilt_from_move(20.0) == ins.TILT_MAX
    assert ins.tilt_from_move(-20.0) == -ins.TILT_MAX


def test_noise_floor_produces_no_tilt():
    assert ins.tilt_from_move(0.05) == 0.0
    assert ins.tilt_from_move(-0.05) == 0.0


def test_weight_scales_the_tilt_down():
    """SMIN is small-cap and the Nifty 50 is not, so the same session must move
    it less than it moves INDA. If the weights stopped being applied this is
    the test that notices."""
    full = ins.tilt_from_move(1.0, weight=1.0)
    small = ins.tilt_from_move(1.0, weight=0.6)
    assert 0 < small < full


# ── The payload ──────────────────────────────────────────────────────────────

def test_adr_move_tilts_its_own_us_listing():
    payload = ins.build_payload({"INFY.NS": _up_2pct()}, now=NOW)
    assert payload["tilts"]["INFY"]["tilt"] > 0
    assert payload["tilts"]["INFY"]["source"] == "INFY.NS"
    assert payload["tilts"]["INFY"]["kind"] == "adr"


def test_index_move_tilts_every_broad_etf():
    payload = ins.build_payload({"^NSEI": _up_2pct()}, now=NOW)
    for etf in ("INDA", "INDY", "EPI", "SMIN"):
        assert etf in payload["tilts"], f"{etf} missing"
    assert payload["tilts"]["SMIN"]["tilt"] < payload["tilts"]["INDA"]["tilt"]


def test_a_stale_session_is_skipped_with_a_reason_not_tilted():
    """The failure this guards is a workflow that stops running. The file keeps
    sitting in the repo looking valid, and the desk keeps tilting on a session
    from last month."""
    old = _up_2pct("2026-07-01")
    payload = ins.build_payload({"INFY.NS": old}, now=NOW)
    assert "INFY" not in payload["tilts"]
    assert "INFY.NS" in payload["skipped"]
    assert "old" in payload["skipped"]["INFY.NS"]


def test_a_session_dated_in_the_future_is_skipped():
    payload = ins.build_payload({"INFY.NS": _up_2pct("2026-09-01")}, now=NOW)
    assert "INFY" not in payload["tilts"]
    assert "future" in payload["skipped"]["INFY.NS"]


def test_a_genuinely_flat_session_is_recorded_as_such():
    """Distinct from stale and from missing: the data arrived, it was fresh,
    and India did nothing. That belongs in `skipped` with the move quoted, so
    the reader can tell it apart from a broken fetch."""
    flat = _closes(("2026-08-04", 100.0), ("2026-08-05", 100.02))
    payload = ins.build_payload({"INFY.NS": flat}, now=NOW)
    assert "INFY" not in payload["tilts"]
    assert re.search(r"noise floor", payload["skipped"]["INFY.NS->INFY"])


def test_every_source_symbol_is_accounted_for():
    """Nothing may vanish. A symbol in neither bucket is indistinguishable
    from one that was never requested."""
    payload = ins.build_payload({}, now=NOW)
    mentioned = set(payload["skipped"]) | {
        v["source"] for v in payload["tilts"].values()}
    for src in list(ins.ADR_MAP) + list(ins.INDEX_MAP):
        assert any(m == src or m.startswith(f"{src}->") for m in mentioned), src


def test_an_empty_fetch_still_produces_a_complete_payload():
    """The file is rewritten every run. If a total fetch failure produced no
    payload, the previous day's file would stay in place and keep tilting."""
    payload = ins.build_payload({}, now=NOW)
    assert payload["tilts"] == {}
    assert payload["skipped"], "a total failure must say so, not be silent"
    assert payload["generated_at"].endswith("Z")


def test_the_stored_timestamp_is_the_format_the_desk_parses():
    """These two live in different files; a format drift would be caught only
    at runtime, by the consumer's except-clause, which swallows it."""
    stamp = ins.build_payload({}, now=NOW)["generated_at"]
    assert datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


# ── The map ──────────────────────────────────────────────────────────────────

def test_no_indian_ticker_is_ever_a_tilt_target():
    """Alpaca cannot route NSE. The whole design is that the *source* is Indian
    and the *target* is US-listed; a `.NS` on the target side would produce
    orders that 404."""
    targets = {t for t, _ in ins.ADR_MAP.values()}
    targets |= {t for m in ins.INDEX_MAP.values() for t in m}
    bad = [t for t in targets if t.endswith((".NS", ".BO")) or t.startswith("^")]
    assert bad == [], f"untradeable tilt targets: {bad}"


def test_every_target_is_a_symbol_some_desk_actually_trades():
    """A tilt for a symbol no desk holds is a file nobody reads."""
    import desk_order_placer as dop
    traded = {s for d in dop.DESKS for s in d.symbols}
    targets = {t for t, _ in ins.ADR_MAP.values()}
    targets |= {t for m in ins.INDEX_MAP.values() for t in m}
    assert targets <= traded, f"tilt targets no desk trades: {sorted(targets - traded)}"


def test_mmyt_is_not_mapped():
    """MakeMyTrip is US-listed with no NSE line. Mapping it to the index would
    invent an overnight read that does not exist."""
    all_targets = {t for t, _ in ins.ADR_MAP.values()}
    all_targets |= {t for m in ins.INDEX_MAP.values() for t in m}
    assert "MMYT" not in all_targets


# ── The Discord post ─────────────────────────────────────────────────────────

def test_a_silent_day_explains_itself():
    body = ins.discord_body(ins.build_payload({}, now=NOW))
    assert "No tilt" in body
    assert "·" in body, "the reasons must be listed, not just the headline"


def test_a_tilted_day_names_the_symbols():
    body = ins.discord_body(ins.build_payload({"INFY.NS": _up_2pct()}, now=NOW))
    assert "INFY" in body and "INFY.NS" in body


# ── The consumer — the part that makes any of this matter ────────────────────

def test_the_desk_actually_applies_the_tilt():
    """THE call-site guard. `india_tilt` existing and being correct is worth
    nothing if the confidence gate never calls it. Three features this week
    passed every unit test while being dead code."""
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    gate = src.split("Apply confidence threshold + top-K filter", 1)
    assert len(gate) == 2, "the threshold stage was renamed — re-point this test"
    stage = gate[1][:4000]
    assert "india_tilt(" in stage, "the confidence gate never calls india_tilt()"
    assert 'item["confidence"] = conf' in stage, (
        "the tilt was computed but not written back — top-K would rank on the "
        "pre-tilt confidence")


def test_the_desk_reads_the_file_this_script_writes():
    """Two hardcoded paths, one file. A rename on either side is silent."""
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    assert ins.STATE_FILE.name in src


def test_tilt_sign_follows_the_side():
    import desk_order_placer as dop
    dop._INDIA_TILTS = {"INDA": {"tilt": 0.04, "source": "^NSEI", "move_pct": 2.0}}
    try:
        assert dop.india_tilt("INDA", "buy") == pytest.approx(0.04)
        assert dop.india_tilt("INDA", "sell") == pytest.approx(-0.04)
    finally:
        dop._INDIA_TILTS = {}


def test_untilted_symbols_are_untouched():
    import desk_order_placer as dop
    dop._INDIA_TILTS = {"INDA": {"tilt": 0.04}}
    try:
        assert dop.india_tilt("SPY", "buy") == 0.0
    finally:
        dop._INDIA_TILTS = {}


def test_the_consumer_caps_whatever_the_file_claims():
    """Defence in depth. The producer caps at ±0.06; if a hand-edited or
    corrupted file claimed 0.9, an uncapped consumer would let it force any
    signal past any threshold."""
    import desk_order_placer as dop
    dop._INDIA_TILTS = {"INDA": {"tilt": 0.9}}
    try:
        assert dop.india_tilt("INDA", "buy") == pytest.approx(dop.INDIA_TILT_HARD_CAP)
        assert dop.india_tilt("INDA", "sell") == pytest.approx(-dop.INDIA_TILT_HARD_CAP)
    finally:
        dop._INDIA_TILTS = {}


def test_the_consumer_rechecks_age_rather_than_trusting_the_file():
    """The producer's freshness check runs when the file is written. The file
    then lives in the repo indefinitely. Only a check at read time can notice
    that the producing workflow stopped."""
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    block = src.split("_INDIA_TILT_FILE", 1)[1][:2000]
    assert "max_age_hours" in block
    assert "stale" in block
