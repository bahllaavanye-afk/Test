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
