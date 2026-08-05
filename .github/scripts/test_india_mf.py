"""Indian mutual funds: AMFI parsing, universe filtering, and the ranking trap.

AMFI publishes every scheme's NAV daily, free and unauthenticated. Verified
2026-08-05: 14,237 schemes across 52 AMCs, of which **3,156** are investable
(Direct + Growth).

Two things these tests pin that cost real debugging:

**1. The host redirects.** `www.amfiindia.com/spages/NAVAll.txt` 302s to
`portal.amfiindia.com/...`. Without following it the payload is a 169-byte HTML
stub that parses to zero schemes — an empty universe that looks like "no funds
today" rather than a broken fetch.

**2. Ranking the raw history returns the same fund four times.** Direct/Regular
x Growth/IDCW all track one portfolio and post near-identical returns. Measured
on Axis (mf=53), the unfiltered top 5 was: Axis IT ETF, then Nifty IT Index Fund
as Direct-Growth, Direct-IDCW, Regular-Growth and Regular-IDCW — four slots for
one fund, two of them the strictly-worse Regular plan. Passing the investable
set gives five distinct funds. `test_ranking_without_the_filter_duplicates_a_fund`
pins the failure mode itself, so nobody "simplifies" the argument away.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import india_mf as M  # noqa: E402

# A faithful slice of NAVAll.txt: AMC header line, category header, and rows for
# the four plan/option variants of one fund plus an unpriced scheme.
SAMPLE = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Sectoral/Thematic)
Axis Mutual Fund
100001;INF0001;INF0002;Axis Nifty IT Index Fund - Direct Plan - Growth;25.5000;04-Aug-2026
100002;INF0003;-;Axis Nifty IT Index Fund - Direct Plan - IDCW;25.4000;04-Aug-2026
100003;INF0004;-;Axis Nifty IT Index Fund - Regular Plan - Growth;24.8000;04-Aug-2026
100004;INF0005;-;Axis Nifty IT Index Fund - Regular Plan - IDCW;24.7000;04-Aug-2026
100005;INF0006;-;Axis Unpriced Fund - Direct Plan - Growth;N.A.;04-Aug-2026

Open Ended Schemes(Equity Scheme - Large Cap)
SBI Mutual Fund
200001;INF0007;-;SBI Bluechip Fund - Direct Plan - Growth;95.1000;04-Aug-2026
"""

HIST = """Scheme Code;Scheme Name;ISIN Div Payout/ISIN Growth;ISIN Div Reinvestment;Net Asset Value;Repurchase Price;Sale Price;Date
100001;Axis Nifty IT Index Fund - Direct Plan - Growth;INF0001;;20.00;;;01-Jul-2026
100001;Axis Nifty IT Index Fund - Direct Plan - Growth;INF0001;;25.50;;;04-Aug-2026
100003;Axis Nifty IT Index Fund - Regular Plan - Growth;INF0004;;20.00;;;01-Jul-2026
100003;Axis Nifty IT Index Fund - Regular Plan - Growth;INF0004;;24.80;;;04-Aug-2026
200001;SBI Bluechip Fund - Direct Plan - Growth;INF0007;;90.00;;;01-Jul-2026
200001;SBI Bluechip Fund - Direct Plan - Growth;INF0007;;95.10;;;04-Aug-2026
"""


@pytest.fixture(scope="module")
def schemes():
    return M.parse_navall(SAMPLE)


def test_it_parses_schemes_and_attributes_the_amc(schemes):
    """The AMC is section state, not a column on the row."""
    assert len(schemes) == 5, f"expected 5 priced schemes, got {len(schemes)}"
    by_code = {s.code: s for s in schemes}
    assert by_code["100001"].amc == "Axis Mutual Fund"
    assert by_code["200001"].amc == "SBI Mutual Fund", (
        "the AMC did not switch at the second section header — every scheme "
        "would be attributed to the first fund house in the file."
    )


def test_category_headers_are_not_mistaken_for_an_amc(schemes):
    """'Open Ended Schemes(...)' has no semicolon and would otherwise stick."""
    assert not any("Open Ended" in s.amc for s in schemes)


def test_unpriced_schemes_are_dropped(schemes):
    """AMFI writes 'N.A.' for schemes that did not price that day."""
    assert "100005" not in {s.code for s in schemes}, (
        "an N.A. NAV was parsed; it would become 0.0 and rank as a total loss"
    )


def test_the_universe_is_direct_and_growth_only(schemes):
    uni = M.investable_universe(schemes)
    names = [s.name for s in uni]
    assert len(uni) == 2, f"expected Direct+Growth only, got {names}"
    assert all("Direct" in n for n in names), "a Regular plan survived the filter"
    assert all("IDCW" not in n for n in names), "an IDCW variant survived"


def test_growth_matches_on_absence_not_the_word_growth():
    """Many schemes name the growth option implicitly; requiring the word drops them."""
    s = M.Scheme(code="1", name="Some Fund - Direct Plan", isin="X",
                 nav=10.0, date="04-Aug-2026", amc="A")
    assert s.is_growth, (
        "a Direct plan with no explicit option word was excluded — matching on "
        "the presence of 'growth' silently shrinks the universe."
    )
    idcw = M.Scheme(code="2", name="Some Fund - Direct Plan - IDCW", isin="X",
                    nav=10.0, date="04-Aug-2026", amc="A")
    assert not idcw.is_growth


def test_history_is_sorted_oldest_first():
    """Return maths reads [0] and [-1]; reversed order flips the sign."""
    h = M.parse_history(HIST)
    dates = [d for d, _ in h["100001"]]
    assert dates == ["01-Jul-2026", "04-Aug-2026"], dates


def test_total_return_is_computed_from_the_endpoints():
    h = M.parse_history(HIST)
    assert M.total_return_pct(h["100001"]) == pytest.approx(27.5)


def test_total_return_refuses_an_unusable_series():
    assert M.total_return_pct([]) is None
    assert M.total_return_pct([("01-Jul-2026", 10.0)]) is None
    assert M.total_return_pct([("01-Jul-2026", 0.0), ("04-Aug-2026", 5.0)]) is None


def test_ranking_with_the_investable_filter_returns_distinct_funds():
    h = M.parse_history(HIST)
    sch = M.parse_navall(SAMPLE)
    names = {s.code: s.name for s in sch}
    codes = {s.code for s in M.investable_universe(sch)}
    top = M.rank_by_momentum(h, names, min_points=2, top_n=5, investable=codes)
    assert [r["code"] for r in top] == ["100001", "200001"], (
        f"expected the two Direct-Growth funds, got {[r['name'] for r in top]}"
    )


def test_ranking_without_the_filter_duplicates_a_fund():
    """Pins the trap itself, so the argument is not 'simplified' away later."""
    h = M.parse_history(HIST)
    names = {s.code: s.name for s in M.parse_navall(SAMPLE)}
    top = M.rank_by_momentum(h, names, min_points=2, top_n=5)
    got = [r["name"] for r in top]
    assert any("Regular" in n for n in got), (
        "the unfiltered ranking no longer includes the Regular plan — if that "
        "is now filtered elsewhere this test is obsolete, but silently ranking "
        "both plans of one fund is the bug it documents."
    )


def test_min_points_rejects_a_thin_series():
    """Two stale prints must not rank as a top performer."""
    h = M.parse_history(HIST)
    names = {s.code: s.name for s in M.parse_navall(SAMPLE)}
    assert M.rank_by_momentum(h, names, min_points=15, top_n=5) == []


def test_the_url_points_at_the_host_that_does_not_redirect():
    """www.amfiindia.com 302s to portal.; the stub parses to zero schemes."""
    assert "portal.amfiindia.com" in M.NAV_ALL_URL, (
        "NAV_ALL_URL points at the redirecting host. Without following the 302 "
        "the payload is a 169-byte HTML stub and the universe silently empties."
    )


def test_the_persisted_history_is_bounded():
    assert 0 < M.HISTORY_KEEP <= 500


def test_save_state_applies_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "STATE_FILE", tmp_path / "india_mf.json")
    M.save_state({"runs": [{"n": i} for i in range(M.HISTORY_KEEP + 40)]})
    import json
    kept = json.loads((tmp_path / "india_mf.json").read_text())["runs"]
    assert len(kept) == M.HISTORY_KEEP
    assert kept[-1]["n"] == M.HISTORY_KEEP + 39, "the cap kept the oldest runs"


# ── the expanded-IDCW gap, and the daily workflow ─────────────────────────────

@pytest.mark.parametrize("name,growth", [
    ("X Fund - Direct Plan - Growth", True),
    ("X Fund - Direct Plan - Growth Option", True),
    ("X Fund - Direct Plan", True),
    ("X Fund - Direct Plan - IDCW", False),
    ("X Fund - Direct Plan - Monthly Dividend", False),
    # AMFI spells IDCW out in full, and that defeated the abbreviation match.
    # Live run 2026-08-05 ranked "SBI Innovative Opportunities Fund - Direct
    # Plan - Growth" and "... - Income Distribution cum Capital Withdrawal" as
    # two separate 13.58% entries, plus the same for SBI Automotive: one
    # portfolio, two slots, and the payout variant is the wrong one to hold.
    ("SBI Innovative Opportunities Fund - Direct Plan - Income Distribution cum "
     "Capital Withdrawal", False),
    ("X Fund - Direct - INCOME DISTRIBUTION CUM CAPITAL WITHDRAWAL", False),
])
def test_growth_detection_covers_both_spellings(name, growth):
    s = M.Scheme(code="1", name=name, isin="", nav=10.0, date="", amc="")
    assert s.is_growth is growth, f"is_growth={s.is_growth} for {name!r}"


def test_the_amc_codes_are_present_and_plausible():
    """Resolved by probe — they are not published machine-readably."""
    assert len(M.AMC_CODES) >= 5, "the ranking universe shrank to a handful of AMCs"
    assert all(isinstance(k, int) and v for k, v in M.AMC_CODES.items())
    assert "HDFC" in M.AMC_CODES.values() and "SBI" in M.AMC_CODES.values(), (
        "the two largest Indian AMCs are missing from the ranking universe"
    )


def test_the_workflow_can_commit_what_it_writes():
    """quick-backtest had no permissions block and persisted nothing for months."""
    yaml = pytest.importorskip("yaml")
    wf = Path(__file__).resolve().parents[1] / "workflows" / "india-mf.yml"
    assert wf.exists(), "the daily India MF workflow is gone"
    doc = yaml.safe_load(wf.read_text())
    assert (doc.get("permissions") or {}).get("contents") == "write", (
        "india-mf.yml cannot write, so its NAV state can never be committed"
    )
    (job,) = doc["jobs"].values()
    assert job.get("timeout-minutes"), "no timeout — a hung fetch burns the runner"


def test_the_workflow_runs_daily_not_on_the_desk_beat():
    """MFs price once a day; a 50-minute cadence would be 29 identical runs."""
    yaml = pytest.importorskip("yaml")
    wf = Path(__file__).resolve().parents[1] / "workflows" / "india-mf.yml"
    doc = yaml.safe_load(wf.read_text())
    sched = (doc.get("on") or doc.get(True)).get("schedule")
    assert sched, "no schedule — the module would only run by hand"
    cron = sched[0]["cron"].split()
    assert cron[1] != "*", f"cron {sched[0]['cron']!r} fires hourly or faster"


def test_the_discord_post_says_something_when_there_is_no_ranking():
    """A silent channel and a broken fetch must not look identical."""
    wf = (Path(__file__).resolve().parents[1] / "workflows" / "india-mf.yml").read_text()
    assert "no ranking produced this run" in wf, (
        "the Discord step posts nothing when the ranking is empty — the failure "
        "this repo keeps rediscovering, in a brand-new channel."
    )


# ── category-relative ranking, and AMFI's truncated IDCW ──────────────────────

def test_the_category_is_captured(schemes):
    """AMFI nests category -> AMC -> rows, NOT the other way round.

    An earlier version reset the category on each AMC line, which left 2,998 of
    3,010 schemes uncategorised in production while the unit test passed — the
    fixture had the two lines reversed. The fixture was the thing that was wrong.
    """
    by_code = {s.code: s for s in schemes}
    assert "Sectoral" in by_code["100001"].category
    assert "Large Cap" in by_code["200001"].category, (
        "the category did not advance at the second section — or an AMC line is "
        "resetting it, which silently uncategorises almost everything."
    )


@pytest.mark.parametrize("name,growth", [
    # AMFI truncates 'IDCW' on some Direct rows. Observed 2026-08-05:
    # 'Regular Plan - Half Yearly IDCW' was caught, 'Direct Plan - Half Yearly'
    # was not — same payout option, one spelling. A payout FREQUENCY at the end
    # is the tell: a growth option does not distribute, so it has no frequency.
    ("Axis Conservative Hybrid Fund - Direct Plan - Half Yearly", False),
    ("Axis Conservative Hybrid Fund - Direct Plan - Quarterly", False),
    ("X Fund - Direct Plan - Annual", False),
    ("X Fund - Direct Plan - Monthly Option", False),
    ("Axis Conservative Hybrid Fund - Direct Plan - Growth Option", True),
    # Anchored to the end, so a frequency word mid-name is not swept up.
    ("Some Quarterly Interval Fund - Direct Plan - Growth", True),
])
def test_truncated_payout_frequencies_are_excluded(name, growth):
    s = M.Scheme(code="1", name=name, isin="", nav=10.0, date="", amc="")
    assert s.is_growth is growth, f"is_growth={s.is_growth} for {name!r}"


def test_category_ranking_compares_like_with_like():
    """A thematic fund beating a large-cap fund is not evidence about either."""
    sch = M.parse_navall(SAMPLE)
    hist = M.parse_history(HIST)
    out = M.rank_within_categories(hist, sch, min_points=2, per_category=3, min_peers=1)
    assert out, "no category produced a ranking"
    for cat, rows in out.items():
        assert all(r["category"] == cat for r in rows), "a fund leaked across categories"
        assert rows[0]["rank"] == 1 and rows[0]["peers"] >= 1


def test_a_category_with_too_few_peers_is_dropped():
    """Coming first out of two is not a percentile."""
    sch = M.parse_navall(SAMPLE)
    hist = M.parse_history(HIST)
    out = M.rank_within_categories(hist, sch, min_points=2, per_category=3, min_peers=99)
    assert out == {}, "a category ranked despite having fewer peers than min_peers"


def test_ranking_is_ordered_best_first():
    sch = M.parse_navall(SAMPLE)
    hist = M.parse_history(HIST)
    out = M.rank_within_categories(hist, sch, min_points=2, per_category=5, min_peers=1)
    for rows in out.values():
        rets = [r["return_pct"] for r in rows]
        assert rets == sorted(rets, reverse=True), f"not descending: {rets}"
