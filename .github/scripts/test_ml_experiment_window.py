"""The ML experiment was not reproducible, and persistence is what revealed it.

Four runs on 2026-08-05, identical params, `oos_days` alternating 688 / 1147:

    09:28  SPY alpaca   940 rows   buyhold sharpe 1.482
    09:35  SPY yfinance 1399 rows  buyhold sharpe 0.789   ← 7 minutes later
    14:44  SPY alpaca   941 rows   buyhold sharpe 1.490
    16:54  SPY yfinance 1399 rows  buyhold sharpe 0.791

Alpaca's free IEX feed carries ~940 usable rows; yfinance carries 1399. The
old code took Alpaca and fell back to yfinance only when Alpaca returned
nothing, so a transient failure silently changed the evaluation period — and
did it per symbol, so 09:35 ran SPY on 1399 rows and QQQ on 940 and reported
them side by side as one cross-sectional result.

**The benchmark moves with the window**, because the longer series includes
the 2022 bear market. That is what made both "beats buy-and-hold" results
(NVDA 09:35, QQQ 16:54) meaningless: the strategy Sharpe barely moved between
runs, the buy-and-hold Sharpe halved. The model did not improve; the yardstick
changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_experiment as mlx  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent


def _frame(start: str, periods: int, end: str | None = None) -> pd.DataFrame:
    """`periods` business days from `start`, or a window ending at `end` — real
    feeds all end at the last close, so most fixtures pin the end."""
    if end is not None:
        idx = pd.date_range(start, end, freq="B")
    else:
        idx = pd.date_range(start, periods=periods, freq="B")
    periods = len(idx)
    return pd.DataFrame(
        {"close": range(1, periods + 1), "high": range(1, periods + 1),
         "low": range(1, periods + 1), "volume": [1] * periods},
        index=idx, dtype=float,
    )


# ── The source pick ──────────────────────────────────────────────────────────

def test_the_longer_history_wins_regardless_of_which_source_it_is(monkeypatch):
    """The real numbers: alpaca 940, yfinance 1399. Whichever source happens to
    be longer must win, so the window stops depending on which fetch succeeded
    first."""
    monkeypatch.setattr(mlx, "fetch_alpaca", lambda s: _frame("2022-01-03", 940))
    monkeypatch.setattr(mlx, "fetch_yfinance", lambda s: _frame("2020-01-02", 1399))
    df, source = mlx.fetch_bars("SPY")
    assert source == "yfinance"
    assert len(df) == 1399

    # …and the other way round, so this is not just "yfinance is preferred".
    monkeypatch.setattr(mlx, "fetch_alpaca", lambda s: _frame("2020-01-02", 1399))
    monkeypatch.setattr(mlx, "fetch_yfinance", lambda s: _frame("2022-01-03", 940))
    df, source = mlx.fetch_bars("SPY")
    assert source == "alpaca"
    assert len(df) == 1399


def test_a_dead_source_does_not_change_the_window(monkeypatch):
    """The old bug in one line: Alpaca fails, yfinance answers, and the
    evaluation period silently grows by 49%."""
    monkeypatch.setattr(mlx, "fetch_alpaca", lambda s: None)
    monkeypatch.setattr(mlx, "fetch_yfinance", lambda s: _frame("2020-01-02", 1399))
    df, source = mlx.fetch_bars("SPY")
    assert source == "yfinance" and len(df) == 1399


def test_both_sources_dead_is_reported_not_guessed(monkeypatch):
    monkeypatch.setattr(mlx, "fetch_alpaca", lambda s: None)
    monkeypatch.setattr(mlx, "fetch_yfinance", lambda s: None)
    assert mlx.fetch_bars("SPY") == (None, "none")


def test_an_empty_frame_is_not_a_source(monkeypatch):
    """`len(df) == 0` is not None, and `max()` on row count would happily pick
    it over nothing — leaving a symbol with an empty window and status ok."""
    monkeypatch.setattr(mlx, "fetch_alpaca", lambda s: _frame("2022-01-03", 0))
    monkeypatch.setattr(mlx, "fetch_yfinance", lambda s: _frame("2020-01-02", 500))
    df, source = mlx.fetch_bars("SPY")
    assert source == "yfinance" and len(df) == 500


# ── The common window ────────────────────────────────────────────────────────

def test_symbols_are_trimmed_to_a_shared_start():
    """Run 09:35 reported SPY on 1399 rows next to QQQ on 940 as one
    cross-sectional result. They were different periods. Both feeds end at the
    last close, which is why only the start differs in the real failure."""
    out = mlx.common_window({
        "SPY": _frame("2020-01-02", 0, end="2026-08-05"),
        "QQQ": _frame("2022-01-03", 0, end="2026-08-05"),
    })
    assert out["SPY"].index.min() == out["QQQ"].index.min()
    assert len(out["SPY"]) == len(out["QQQ"])


def test_a_lagging_feed_cannot_leave_one_symbol_a_day_short():
    """The same defect at the other end. A feed that has not published today's
    bar would otherwise give one symbol a shorter window, and the row counts
    stay close enough that nobody would notice."""
    out = mlx.common_window({
        "SPY": _frame("2022-01-03", 0, end="2026-08-05"),
        "QQQ": _frame("2022-01-03", 0, end="2026-08-04"),   # a day behind
    })
    assert out["SPY"].index.max() == out["QQQ"].index.max() == pd.Timestamp("2026-08-04")
    assert len(out["SPY"]) == len(out["QQQ"])


def test_the_shared_start_is_the_latest_one_not_the_earliest():
    """Taking the earliest would keep a symbol's missing history as NaN or
    silently shorten only some symbols — the bug, restated."""
    out = mlx.common_window({
        "A": _frame("2020-01-02", 1399),
        "B": _frame("2022-01-03", 940),
        "C": _frame("2021-01-04", 1100),
    })
    starts = {s: d.index.min() for s, d in out.items()}
    assert len(set(starts.values())) == 1
    assert starts["A"] == pd.Timestamp("2022-01-03")


def test_a_single_symbol_run_is_left_alone():
    """Nothing to align against; trimming would be a no-op that could only
    lose data."""
    out = mlx.common_window({"SPY": _frame("2020-01-02", 1399)})
    assert len(out["SPY"]) == 1399


def test_an_empty_frame_does_not_poison_the_shared_window():
    """`len(df) == 0` is not None, and an empty DatetimeIndex has `min() ==
    NaT`. Let one through and `max(...)` over the starts becomes NaT, every
    `>=` comparison is False, and *every* symbol comes back with zero rows —
    the run reports "only 0 usable rows" for symbols whose data was fine."""
    out = mlx.common_window({
        "SPY": _frame("2022-01-03", 0, end="2026-08-05"),
        "QQQ": _frame("2022-01-03", 0, end="2026-08-05"),
        "DEAD": _frame("2022-01-03", 0),          # zero rows
    })
    assert "DEAD" not in out
    assert len(out["SPY"]) > 0 and len(out["QQQ"]) > 0


def test_symbols_that_failed_to_fetch_do_not_truncate_the_rest():
    """A None entry must not participate in the max(), or one dead symbol
    would drag every other symbol's window with it."""
    out = mlx.common_window({"SPY": _frame("2020-01-02", 1399), "QQQ": None})
    assert "QQQ" not in out
    assert len(out["SPY"]) == 1399


# ── The call sites ───────────────────────────────────────────────────────────

def test_main_actually_uses_both_fixes():
    """Neither helper is worth anything sitting unused next to the old
    alpaca-then-fallback code. Three features shipped dead this week."""
    src = (SCRIPTS / "ml_experiment.py").read_text()
    body = src.split("def main(", 1)[1]
    assert "fetch_bars(" in body, "main() still picks its own source"
    assert "common_window(" in body, "main() never aligns the symbols"
    assert "fetch_alpaca(sym)" not in body, (
        "main() still calls fetch_alpaca directly — the old alternating path")


def test_the_window_is_recorded_in_the_payload():
    """`rows` alone made this take four runs and a diff to spot. The dates make
    a changed window visible in one line of the state file."""
    src = (SCRIPTS / "ml_experiment.py").read_text()
    body = src.split("def main(", 1)[1]
    assert "first_date" in body and "last_date" in body


# ── Sub-windows: is the edge consistent, or one lucky stretch? ───────────────

def _series(n, val):
    import numpy as np
    return np.full(n, val, dtype=float)


def test_sub_windows_split_the_oos_series_evenly():
    import numpy as np
    dates = pd.date_range("2021-01-04", periods=900, freq="B")
    out = mlx.sub_window_stats(_series(900, 0.001), _series(900, 0.001), dates)
    assert len(out) == 3
    assert sum(w["days"] for w in out) == 900
    assert out[0]["from"] == "2021-01-04"
    assert out[-1]["to"] == str(dates[-1].date())


def test_a_short_series_reports_nothing_rather_than_noise():
    """Under 30 OOS days a slice's Sharpe is noise dressed as a number, and a
    reader would weigh it the same as a real one."""
    import numpy as np
    dates = pd.date_range("2026-01-01", periods=60, freq="B")
    assert mlx.sub_window_stats(_series(60, 0.001), _series(60, 0.001), dates) == []


def test_an_edge_concentrated_in_one_period_is_visible():
    """THE point of this. A model that beats the benchmark only in the middle
    third must not read the same as one that beats it throughout — those imply
    opposite decisions about wiring ML into live orders."""
    import numpy as np
    n = 900
    bench = _series(n, 0.0005)
    strat = _series(n, 0.0001)
    rng = np.random.default_rng(0)
    bench = bench + rng.normal(0, 0.001, n)
    strat = strat + rng.normal(0, 0.001, n)
    strat[300:600] += 0.004                      # edge ONLY in the middle third
    out = mlx.sub_window_stats(strat, bench, pd.date_range("2021-01-04", periods=n, freq="B"))
    assert [w["beats"] for w in out] == [False, True, False], [w["beats"] for w in out]


def test_a_consistent_edge_shows_in_every_window():
    import numpy as np
    n = 900
    rng = np.random.default_rng(1)
    bench = rng.normal(0.0002, 0.001, n)
    strat = bench + 0.003                        # uniformly better
    out = mlx.sub_window_stats(strat, bench, pd.date_range("2021-01-04", periods=n, freq="B"))
    assert all(w["beats"] for w in out)


def test_walk_forward_actually_reports_sub_windows():
    """Call-site guard. The helper is worthless if the payload never carries it
    — the exact mistake made four times this week."""
    src = (SCRIPTS / "ml_experiment.py").read_text()
    body = src.split("def walk_forward(", 1)[1]
    assert '"sub_windows": sub_window_stats(' in body, (
        "walk_forward does not put sub_windows in its result")


# ── The integration the string-match guard cannot cover ──────────────────────

def test_walk_forward_runs_end_to_end_and_emits_sub_windows(monkeypatch):
    """`walk_forward` had never been EXECUTED by any test — only string-matched.

    `test_walk_forward_actually_reports_sub_windows` greps the source, so if
    `feat.index[mask]` raised at runtime every test here would still pass and
    the weekly ML run would die. sklearn is not installed in the agent-test
    environment (CI installs pandas/numpy/pyyaml/requests only), so the
    classifier is stubbed: the point is the plumbing — masking, indexing,
    the sub-window call — not the model.
    """
    import sys as _sys
    import types
    import numpy as np

    class _StubClf:
        def __init__(self, **kw): pass
        def fit(self, X, y): return self
        def predict_proba(self, X):
            return np.tile([[0.4, 0.6]], (len(X), 1))

    fake = types.ModuleType("sklearn.ensemble")
    fake.GradientBoostingClassifier = _StubClf
    pkg = types.ModuleType("sklearn")
    pkg.ensemble = fake
    monkeypatch.setitem(_sys.modules, "sklearn", pkg)
    monkeypatch.setitem(_sys.modules, "sklearn.ensemble", fake)

    rng = np.random.default_rng(7)
    n = 700
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)
    raw = pd.DataFrame({"close": close, "high": close * 1.01,
                        "low": close * 0.99, "volume": rng.integers(1e6, 2e6, n)},
                       index=idx).astype(float)

    feat = mlx.build_features(raw)
    assert len(feat) > mlx.MIN_TRAIN + 120, f"fixture too short: {len(feat)} rows"

    out = mlx.walk_forward(feat)

    for key in ("oos_days", "hit_rate", "strategy_sharpe", "buyhold_sharpe", "sub_windows"):
        assert key in out, f"walk_forward dropped {key}"
    assert out["oos_days"] == len(feat) - mlx.MIN_TRAIN

    wins = out["sub_windows"]
    assert len(wins) == 3, wins
    assert sum(w["days"] for w in wins) == out["oos_days"]
    for w in wins:
        # Dates must be real and ordered — this is what `feat.index[mask]`
        # silently getting the wrong type would break.
        assert len(w["from"]) == 10 and len(w["to"]) == 10, w
        assert w["from"] <= w["to"], w
    assert wins[0]["to"] <= wins[1]["from"] <= wins[2]["from"]

    # The dates must be the OUT-OF-SAMPLE ones. Passing `feat.index` instead of
    # `feat.index[mask]` keeps every slice index in range against the longer
    # index, so the windows still look well-formed while every date is shifted
    # by MIN_TRAIN, roughly a year. That mutation survived the first pass;
    # these two assertions are what kill it.
    assert wins[0]["from"] == str(feat.index[mlx.MIN_TRAIN].date())
    assert wins[-1]["to"] == str(feat.index[-1].date())


def test_sub_window_stats_rejects_a_date_index_that_does_not_match():
    """The mismatch is silent by construction: a longer index keeps every slice
    in range, so it has to be rejected explicitly rather than caught later."""
    import numpy as np
    dates = pd.date_range("2021-01-04", periods=500, freq="B")
    with pytest.raises(ValueError, match="masked"):
        mlx.sub_window_stats(np.zeros(300), np.zeros(300), dates)


# ── The noise floor: stop near-zero comparisons reading as evidence ──────────

def test_the_floor_tightens_as_the_window_grows():
    """More samples, smaller detectable difference. If this ever inverted, long
    windows would demand a bigger edge than short ones."""
    assert mlx.sharpe_noise_floor(100) > mlx.sharpe_noise_floor(380) > mlx.sharpe_noise_floor(1147)


def test_a_single_sample_distinguishes_nothing():
    """`inf`, not a small number. One observation supports no verdict at all,
    and returning e.g. 0.0 would make every comparison decisive."""
    assert mlx.sharpe_noise_floor(1) == float("inf")
    assert mlx.sharpe_noise_floor(0) == float("inf")


def test_the_real_spy_window_that_motivated_this_is_now_inconclusive():
    """Run 2026-08-06T00:13 reported SPY 0.102 vs 0.087 over 382 days as
    `beats`. Arithmetically true, indistinguishable from zero, and it counted
    equally with QQQ's 2.057 vs 0.176 in the same table."""
    import numpy as np
    n = 382 * 3
    rng = np.random.default_rng(3)
    bench = rng.normal(0.00002, 0.01, n)
    strat = bench.copy()                       # margin ~0 by construction
    dates = pd.date_range("2022-01-06", periods=n, freq="B")
    for w in mlx.sub_window_stats(strat, bench, dates):
        assert w["verdict"] == "inconclusive", w
        assert w["beats"] is False, "a zero margin must not read as a win"


def test_a_decisive_edge_still_reads_as_a_win():
    """The floor must not swallow QQQ's 2.057-vs-0.176 case."""
    import numpy as np
    n = 382 * 3
    rng = np.random.default_rng(4)
    bench = rng.normal(0.0, 0.01, n)
    strat = bench + 0.004                       # large, sustained
    dates = pd.date_range("2022-01-06", periods=n, freq="B")
    for w in mlx.sub_window_stats(strat, bench, dates):
        assert w["verdict"] == "beats" and w["beats"] is True, w
        assert w["margin"] > w["noise_floor"]


def test_a_decisive_loss_is_named_as_one_not_merely_not_a_win():
    """Three states, not two. 'loses' and 'inconclusive' imply different
    decisions, and collapsing them would hide a real underperformance."""
    import numpy as np
    n = 382 * 3
    rng = np.random.default_rng(5)
    bench = rng.normal(0.0, 0.01, n)
    strat = bench - 0.004
    dates = pd.date_range("2022-01-06", periods=n, freq="B")
    for w in mlx.sub_window_stats(strat, bench, dates):
        assert w["verdict"] == "loses" and w["beats"] is False, w


def test_margin_and_floor_are_both_reported():
    """A verdict the reader cannot check is just an assertion. Both inputs to
    it are in the payload."""
    import numpy as np
    n = 382 * 3
    dates = pd.date_range("2022-01-06", periods=n, freq="B")
    w = mlx.sub_window_stats(np.zeros(n), np.zeros(n), dates)[0]
    assert "margin" in w and "noise_floor" in w
    assert w["noise_floor"] == round(mlx.sharpe_noise_floor(w["days"]), 3)


def test_a_SMALL_POSITIVE_margin_is_inconclusive_not_a_win():
    """The case the floor exists for, and the one an identical-copy fixture
    cannot reach. SPY's real window was +0.015 against a 0.145 floor: a genuine
    positive margin that is still noise. `margin > 0` passes an exact-zero
    fixture, so without this the old bug survives mutation testing."""
    import numpy as np
    n = 382 * 3
    rng = np.random.default_rng(11)
    bench = rng.normal(0.0, 0.01, n)
    strat = bench + 3e-5                      # margin ~+0.05, floor ~0.145
    dates = pd.date_range("2022-01-06", periods=n, freq="B")
    for w in mlx.sub_window_stats(strat, bench, dates):
        assert 0 < w["margin"] < w["noise_floor"], w
        assert w["verdict"] == "inconclusive", w
        assert w["beats"] is False, "a positive margin inside the floor is not a win"


def test_a_SMALL_NEGATIVE_margin_is_inconclusive_not_a_loss():
    """Symmetric, and it matters for the recommendation: SPY's most recent
    window was -0.100 against a 0.145 floor. Reporting that as `loses` is how
    'behind on 2 of 3' got overstated."""
    import numpy as np
    n = 382 * 3
    rng = np.random.default_rng(12)
    bench = rng.normal(0.0, 0.01, n)
    strat = bench - 3e-5
    dates = pd.date_range("2022-01-06", periods=n, freq="B")
    for w in mlx.sub_window_stats(strat, bench, dates):
        assert -w["noise_floor"] < w["margin"] < 0, w
        assert w["verdict"] == "inconclusive", w


# ── The summary a human actually reads ──────────────────────────────────────

def test_the_step_summary_surfaces_the_sub_windows_not_just_the_aggregate():
    """Computed, persisted, and invisible is the failure mode this codebase
    keeps hitting. The overall Sharpe is the statistic that misled twice in one
    evening; a summary showing only it invites the same read."""
    src = (SCRIPTS / "ml_experiment.py").read_text()
    block = src.split("GITHUB_STEP_SUMMARY", 1)[1]
    assert "sub_windows" in block, "the step summary never reads sub_windows"
    for field in ("margin", "noise_floor", "verdict"):
        assert field in block, f"the summary omits {field}"


def test_the_promotion_bar_is_not_stated_on_the_bare_aggregate():
    """It previously read 'a strategy only earns promotion when its OOS Sharpe
    beats buy-and-hold' — the exact criterion the sub-windows disproved, stated
    as policy in the artifact a human reads."""
    src = (SCRIPTS / "ml_experiment.py").read_text()
    block = src.split("GITHUB_STEP_SUMMARY", 1)[1]
    assert "only earns promotion when its OOS Sharpe beats buy-and-hold" not in block
    assert "noise floor" in block and "necessary and not" in block


def test_a_run_too_short_to_split_says_so_rather_than_printing_nothing():
    """An empty section is indistinguishable from a broken one."""
    src = (SCRIPTS / "ml_experiment.py").read_text()
    block = src.split("GITHUB_STEP_SUMMARY", 1)[1]
    assert "any_windows" in block
    assert "too few out-of-sample days" in block
