"""28% of the shared brain's `peer_learnings` was the LLM restating its prompt.

Measured 2026-08-05 against the live `agent_memory.json`: **56 of 200 entries**
were instruction echoes, not learnings —

    [investor_pipeline @ ...] The user asks: "Give a one-sentence status update:
                              what are you actively working on RIGHT NOW..."
    [self_improver @ ...]     We need to respond as self_improver agent,
                              autonomous, 2 sentences max, first person...

`agent_status_checker.py` and `multi_agent_discussion.py` append the raw LLM
reply with no quality check, so a *failed* generation is stored as a learning.
That was merely wasteful until the retrieval fix landed the same night; now
these entries are retrieved into other agents' prompts, so the noise compounds.

The other half is IMPROVEMENTS #814. `backend/performance_log/
strategy_performance.json` is written by `fill_tracker.py` from real filled
orders and IS committed — 22 strategies, 247 tracked order ids. **Nothing
consumed it into agent context.** So the daily discussion ran entirely on
self-reported status while the actual results sat in a file:

    avellaneda_stoikov_mm:  10 trades,  90% win,  +53.48%
    stat_arb_etf:           30 trades,  93% win,  +31.89%
    stat_arb_e:             21 trades,   0% win,  -37.17%   <- never discussed

The worst performer is always included for exactly that reason: reporting only
winners is how status theater starts, and a 0%-win strategy is the more useful
thing for the agents to be arguing about.

The echo patterns are deliberately narrow. A false positive silently discards a
real learning — worse than keeping one echo — so
`test_ordinary_learnings_are_never_dropped` pins realistic entries that must
survive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from shared_context import (  # noqa: E402
    clean_learnings,
    is_low_quality_learning,
    outcome_learnings,
)

ECHOES = [
    '[investor_pipeline @ 2026-08-02T02:44] The user asks: "Give a one-sentence status update: what are you working on"',
    "[self_improver @ 2026-08-02T02:44] We need to respond as self_improver agent, autonomous, 2 sentences max, first person",
    '[peer_reviewer @ 2026-08-02T06:32] We need to respond as peer_reviewer agent. The instruction: "Give a one-sentence status"',
    "[algo_agent @ 2026-08-05T02:28] Give a one-sentence status update: what are you actively working on right now",
]

REAL = [
    "[desk_trader @ 2026-08-05T01:10] Crypto desk placed 0 orders for the 4th run — non_marginable_buying_power is $0.00",
    "[continuous_improver @ 2026-08-05T02:00] error_handling on backend/app/api/v1/trades.py — added specific exception types",
    "[research_scientist @ 2026-08-05T02:28] Walk-forward on the momentum ensemble shows Sharpe 0.4 out of sample vs 1.9 in",
    "[attribution @ 2026-08-05T03:40] stat_arb_e: 21 trades, 0% win rate, -37.17% total return, -1.77% avg (period 30d)",
]


@pytest.mark.parametrize("text", ECHOES)
def test_instruction_echoes_are_rejected(text):
    assert is_low_quality_learning(text), (
        f"an instruction echo was accepted as a learning: {text[:70]!r}. It "
        "will be retrieved into other agents' prompts."
    )


@pytest.mark.parametrize("text", REAL)
def test_ordinary_learnings_are_never_dropped(text):
    """False positives are the expensive failure — they lose real signal."""
    assert not is_low_quality_learning(text), (
        f"a real learning was filtered out: {text[:70]!r}. The echo patterns "
        "must stay narrow; silently discarding findings is worse than keeping "
        "an echo."
    )


def test_too_short_is_rejected_after_stripping_the_prefix():
    """`[agent @ ts] ` is ~30 chars — a naive length check passes on emptiness."""
    assert is_low_quality_learning("[continuous_improver @ 2026-08-05T02:00] ok")
    assert is_low_quality_learning("[continuous_improver @ 2026-08-05T02:00] ")


def test_non_strings_are_rejected_not_crashed():
    for bad in (None, 42, {"a": 1}, []):
        assert is_low_quality_learning(bad)


def test_clean_learnings_preserves_order_of_survivors():
    batch = [REAL[0], ECHOES[0], REAL[1], ECHOES[1], REAL[2]]
    assert clean_learnings(batch) == [REAL[0], REAL[1], REAL[2]]


def test_clean_learnings_handles_empty_and_none():
    assert clean_learnings([]) == []
    assert clean_learnings(None) == []


# ── outcome linkage (IMPROVEMENTS #814) ───────────────────────────────────────

def _perf(tmp_path: Path, strategies: dict, period: int = 30) -> Path:
    p = tmp_path / "strategy_performance.json"
    p.write_text(json.dumps({
        "generated_at": "2026-08-05T00:55:23+00:00",
        "period_days": period,
        "strategies": strategies,
        "tracked_order_ids": [],
    }))
    return p


def _s(trades, win, total, avg):
    return {"trades": trades, "wins": int(trades * win), "win_rate": win,
            "avg_return_pct": avg, "total_return_pct": total}


def test_it_reports_real_numbers_from_the_artifact(tmp_path):
    p = _perf(tmp_path, {"options_pc": _s(14, 1.0, 24.0265, 1.7162)})
    lines = outcome_learnings(top_n=5, path=p)
    assert len(lines) == 1
    line = lines[0]
    for token in ("options_pc", "14 trades", "100% win rate", "+24.03%", "+1.72%", "30d"):
        assert token in line, f"{token!r} missing from {line!r}"


def test_the_worst_performer_is_always_included(tmp_path):
    """The load-bearing test: a losing strategy must not be croppable.

    Ranked by total return descending, so `stat_arb_e` at -37% falls outside any
    top_n. Surfacing only winners is the status theater #814 exists to end.
    """
    strategies = {f"win{i}": _s(10, 0.9, 50 - i, 5.0) for i in range(6)}
    strategies["stat_arb_e"] = _s(21, 0.0, -37.17, -1.77)
    lines = outcome_learnings(top_n=3, path=_perf(tmp_path, strategies))
    assert any("stat_arb_e" in l for l in lines), (
        "the worst performer was cropped by top_n. A 0%-win strategy is the "
        "most useful thing in this list."
    )
    assert len(lines) == 4, "expected top_n winners plus the single worst"

    # Composition, not just presence. Reversing the sort keeps stat_arb_e in the
    # list (it lands in the leading slice instead) while silently turning
    # "top 3 winners + the worst" into "worst 3 + the best" — a different
    # briefing entirely. Pin both ends.
    assert "win0" in lines[0], (
        "the best performer (+50%) is not first — the ranking is not descending "
        "by total return."
    )
    assert "stat_arb_e" in lines[-1], "the worst performer must be the trailing entry"
    assert not any("win5" in l for l in lines[:3]), (
        "a weaker winner displaced a stronger one in the top slice"
    )


def test_the_worst_is_not_duplicated_when_it_fits(tmp_path):
    strategies = {"a": _s(10, 0.9, 50, 5.0), "b": _s(10, 0.5, -3, -0.3)}
    lines = outcome_learnings(top_n=5, path=_perf(tmp_path, strategies))
    assert len(lines) == 2, f"expected 2 lines, got {len(lines)}: {lines}"


def test_zero_trade_strategies_are_excluded(tmp_path):
    """A strategy with no fills has no outcome to learn from."""
    strategies = {"live": _s(4, 1.0, 8.0, 2.0), "never_ran": _s(0, 0.0, 0.0, 0.0)}
    lines = outcome_learnings(top_n=5, path=_perf(tmp_path, strategies))
    assert len(lines) == 1 and "never_ran" not in lines[0]


@pytest.mark.parametrize("payload", ['{"strategies": {}}', '{}', 'not json', ''])
def test_a_missing_or_empty_artifact_yields_nothing_not_an_error(tmp_path, payload):
    """This runs inside the discussion's save path — it must never raise."""
    p = tmp_path / "strategy_performance.json"
    p.write_text(payload)
    assert outcome_learnings(path=p) == []


def test_a_nonexistent_path_is_silent(tmp_path):
    assert outcome_learnings(path=tmp_path / "nope.json") == []


def test_outcome_lines_survive_the_quality_filter(tmp_path):
    """Both halves ship together — the filter must not eat the outcomes.

    They pass through `clean_learnings` in multi_agent_discussion, so an echo
    pattern that happened to match this format would delete the very thing #814
    adds.
    """
    p = _perf(tmp_path, {"stat_arb_e": _s(21, 0.0, -37.17, -1.77)})
    lines = outcome_learnings(path=p)
    assert clean_learnings(lines) == lines, (
        "attribution lines are being rejected by the echo filter."
    )


def test_the_discussion_wires_in_both(tmp_path):
    """A helper nothing calls is the same absence in a new shape."""
    src = (_DIR / "multi_agent_discussion.py").read_text()
    extend = src.index('mem["peer_learnings"].extend(new_learnings)')
    before = src[:extend]
    assert "clean_learnings(new_learnings)" in before, (
        "the discussion no longer filters echoes before extending peer_learnings"
    )
    assert "outcome_learnings()" in before, (
        "the discussion no longer injects P&L attribution — #814 is unwired and "
        "the agents are back to discussing self-reported status."
    )


def test_the_status_checker_filters_too():
    """It is the higher-volume producer of raw replies of the two."""
    src = (_DIR / "agent_status_checker.py").read_text()
    assert "is_low_quality_learning" in src, (
        "agent_status_checker appends raw LLM replies unfiltered again — it is "
        "where the measured 28% came from."
    )
