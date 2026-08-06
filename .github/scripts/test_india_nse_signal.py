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
    """A month-ahead date and a market that is merely still trading are the same
    defect at different scales, so they share one guard and one message."""
    payload = ins.build_payload({"INFY.NS": _up_2pct("2026-09-01")}, now=NOW)
    assert "INFY" not in payload["tilts"]
    assert "not closed yet" in payload["skipped"]["INFY.NS"]


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


# ── The fallback that made live verification possible ────────────────────────

def test_chart_api_parses_a_real_shaped_payload(monkeypatch):
    """Bars are stamped at the session OPEN (03:45 UTC for NSE), so the date
    must come from the timestamp and the close is reconstructed later. A `null`
    close is a real thing Yahoo returns for a halted session and must be
    dropped, not coerced to 0.0."""
    payload = {"chart": {"result": [{
        "timestamp": [1785900300, 1785986700, 1786073100],  # 03:45 UTC each
        "indicators": {"quote": [{"close": [24774.3, None, 24624.65]}]},
    }]}}

    class _Resp:
        status_code = 200
        def json(self): return payload

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    rows = ins._via_chart_api("^NSEI")
    assert [c for _, c in rows] == [24774.3, 24624.65]
    assert all(isinstance(d, date) for d, _ in rows)


def test_chart_api_raises_on_a_non_200(monkeypatch):
    """The crypto feed returned HTTP 451 and the caller treated a bare None as
    'no data', dropping every symbol with no log line at all. A non-200 here
    must be loud."""
    class _Resp:
        status_code = 429
        def json(self): return {}

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="429"):
        ins._via_chart_api("^NSEI")


def test_a_yfinance_transport_failure_falls_through_rather_than_reading_flat(monkeypatch):
    """yfinance ships its own HTTP stack (curl_cffi) which fails behind an
    egress proxy while plain requests to the same URL succeeds. Without the
    fallback that transport failure is indistinguishable from a flat session."""
    monkeypatch.setattr(ins, "_via_chart_api",
                        lambda sym: _closes(("2026-08-04", 100.0), ("2026-08-05", 102.0)))

    class _Broken:
        def Ticker(self, sym): raise OSError("Connection reset by peer")

    monkeypatch.setitem(sys.modules, "yfinance", _Broken())
    out = ins.fetch_closes(["INFY.NS"])
    assert out["INFY.NS"], "the fallback was never consulted"
    assert ins.session_move_pct(out["INFY.NS"])[1] == pytest.approx(2.0)


def test_both_sources_failing_yields_no_entry_at_all(monkeypatch):
    """Not an empty list, which `build_payload` would report as 'fewer than 2
    closes' — the same message a genuinely thin ticker gets. Absent means the
    fetch is what broke, and the log line above says so."""
    def _boom(sym): raise RuntimeError("chart API down")
    monkeypatch.setattr(ins, "_via_chart_api", _boom)

    class _Broken:
        def Ticker(self, sym): raise OSError("reset")

    monkeypatch.setitem(sys.modules, "yfinance", _Broken())
    assert ins.fetch_closes(["INFY.NS"]) == {}


def test_the_tilt_log_line_states_the_side():
    """The first live application printed

        INFY conf 0.73 → 0.72 (-0.011 from INFY.NS +0.56%)

    a down-tilt from an up session, which reads as a sign error until you learn
    it was a SELL. The sign is *derived* from the side, so a line that omits the
    side cannot be checked by the person reading it — and these lines exist to
    be checked."""
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    stage = src.split("Apply confidence threshold + top-K filter", 1)[1][:4000]
    tilt_log = [ln for ln in stage.splitlines() if "🇮🇳" in ln and "print(" in ln]
    assert tilt_log, "the tilt application no longer logs anything"
    block = stage.split("🇮🇳", 1)[1][:400]
    assert "_sd" in block or "side" in block, "the tilt log line omits the signal side"
    assert "agrees" in block, "the log does not say whether India agreed or disagreed"


# ── Asia-Pacific extension (2026-08-06) ─────────────────────────────────────

def test_every_source_market_closes_before_the_us_open():
    """THE property that makes an overnight read useful. A market closing after
    13:30 UTC cannot inform an order at the open, and adding one would look
    like more coverage while delivering a stale number."""
    US_OPEN_UTC = 13.5
    late = {src: h for src, h in ins.SESSION_CLOSE_UTC.items() if h >= US_OPEN_UTC}
    assert not late, f"these close at or after the US open: {late}"


def test_europe_is_not_in_the_map():
    """Explicit, because the desks trade EWG/EWQ/EWU so the temptation is real.
    DAX/CAC close 15:30 UTC and the FTSE 16:30 — two to three hours late."""
    euro = {"^GDAXI", "^FCHI", "^FTSE", "^STOXX50E", "^IBEX", "^AEX"}
    assert not (euro & set(ins.INDEX_MAP)), "a European index cannot inform the US open"
    assert not (euro & set(ins.SESSION_CLOSE_UTC))


def test_each_market_uses_its_own_close_hour_not_a_shared_constant():
    """A single constant (10, NSE's) overstated a 05:30 close's age by 4.5h.
    Harmless inside a 30h window, wrong on its own terms, and a trap if the
    window is ever tightened."""
    from datetime import date as _date
    tw = ins._session_close_utc(_date(2026, 8, 5), "^TWII")
    ns = ins._session_close_utc(_date(2026, 8, 5), "^NSEI")
    assert tw.hour == 5 and tw.minute == 30, tw
    assert ns.hour == 10, ns
    assert tw < ns


def test_a_half_hour_close_is_not_truncated_to_the_hour():
    """Korea closes 06:30 and Taiwan 05:30. Integer hours would lose both."""
    from datetime import date as _date
    assert ins._session_close_utc(_date(2026, 8, 5), "^KS11").minute == 30


def test_an_adr_inherits_its_home_market_close():
    """`INFY.NS` is not a key in SESSION_CLOSE_UTC; its suffix is. Without the
    suffix lookup every single-name silently falls back to the 10:00 default —
    right for India by luck, wrong everywhere the map grows to."""
    assert ins._market_close_for("INFY.NS") == ins.SESSION_CLOSE_UTC["^NSEI"]
    assert ins._market_close_for("7203.T") == ins.SESSION_CLOSE_UTC["^N225"]
    assert ins._market_close_for("NOTAMARKET") == ins.DEFAULT_CLOSE_UTC_HOUR


def test_every_new_target_is_a_symbol_some_desk_trades():
    """Same rule as the India targets: a tilt for a symbol no desk holds is a
    file nobody reads."""
    import desk_order_placer as dop
    traded = {s for d in dop.DESKS for s in d.symbols}
    targets = {t for m in ins.INDEX_MAP.values() for t in m}
    assert targets <= traded, f"tilt targets no desk trades: {sorted(targets - traded)}"


def test_the_hong_kong_link_is_weighted_below_the_direct_ones():
    """FXI holds China H-shares; the Hang Seng is a Hong Kong index. Related,
    not the same market — the weight has to say so, exactly as SMIN's does."""
    assert ins.INDEX_MAP["^HSI"]["FXI"] < ins.INDEX_MAP["^N225"]["EWJ"]


def test_asia_sources_produce_tilts_end_to_end():
    payload = ins.build_payload({"^N225": _up_2pct(), "^HSI": _up_2pct()}, now=NOW)
    assert payload["tilts"]["EWJ"]["tilt"] > payload["tilts"]["FXI"]["tilt"] > 0
    assert payload["tilts"]["EWJ"]["kind"] == "index"


def test_a_session_that_has_not_closed_yet_is_refused():
    """Caught live 2026-08-06 05:15: a run produced `EWT -0.0107 <- ^TWII
    -0.60%` from a Taiwan session with 15 minutes left to trade. The old ±1h
    tolerance was harmless when NSE (10:00) was the only source and the workflow
    ran at 10:20; with markets closing 05:30-10:00 it admitted partial sessions.

    An intraday snapshot reported as a close is worse than no read, because it
    is indistinguishable from a real one."""
    from datetime import datetime as _dt
    # 05:15 UTC — Taiwan (05:30) is still trading.
    mid_session = _dt(2026, 8, 6, 5, 15, tzinfo=timezone.utc)
    payload = ins.build_payload(
        {"^TWII": _up_2pct("2026-08-06")}, now=mid_session)
    assert "EWT" not in payload["tilts"], "tilted on an unclosed session"
    assert "not closed yet" in payload["skipped"]["^TWII"]


def test_a_session_that_just_closed_is_accepted():
    """The tolerance must not swing the other way and reject fresh closes."""
    from datetime import datetime as _dt
    just_closed = _dt(2026, 8, 6, 5, 35, tzinfo=timezone.utc)   # TW closed 05:30
    payload = ins.build_payload({"^TWII": _up_2pct("2026-08-06")}, now=just_closed)
    assert "EWT" in payload["tilts"], payload["skipped"]
