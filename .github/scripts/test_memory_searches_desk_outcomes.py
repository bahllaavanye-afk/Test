"""Upgrading to semantic retrieval silently dropped what the desks actually did.

`SemanticRetriever.search` had a hardcoded default category list:

    ["episodic", "skills", "chat_insights",
     "github_insights", "trade_outcomes", "experiment_results"]

Three of those six — `skills`, `chat_insights`, `trade_outcomes` — have never
existed in `company_brain.json`. And `desk_outcomes`, which DOES exist and held
**100 of the brain's 403 entries**, was not in the list. Each of those entries is
one desk run:

    {"channel": "desk-commodities", "source": "desk_run_summary",
     "summary": "*Commodities Desk* — 3 order(s) placed\\n🟢 `time_series_momentum/SLV`
                 BUY $200 conf=100%  id=`66755b3f…` …"}

Side, notional, confidence, per order. For a trading firm this is the most
decision-relevant memory in the brain, and it reached no agent prompt.

**It is a regression, not an oversight.** The recency path this replaced —
`llm_common.get_company_context`, line ~951 — did include it:
`desk_outcomes = brain.get("desk_outcomes", [])[-3:]`. So the "retrieval upgrade"
made agent context strictly worse for trading outcomes while looking like an
improvement.

Nothing caught it because the failure mode is a **narrower** result set, not an
empty one. `search("crypto desk orders placed")` still returned four plausible
`episodic` hits. A silent narrowing is invisible from the outside — which is why
`test_a_brain_of_only_desk_outcomes_returns_hits` builds a brain where any
non-empty result must have come from the category under test.

The runtime drift warning is deliberately NOT a CI assertion: the brain is
written by background bots, and a test asserting on live mutable state can turn
the agent suite red at 3am and block every PR under `pytest -x` (recorded
2026-08-04, the denylist TTL).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_manager as mm  # noqa: E402


@pytest.fixture
def retriever(monkeypatch):
    """A retriever whose brain is whatever the test hands it."""
    def _make(brain: dict) -> mm.SemanticRetriever:
        r = mm.SemanticRetriever()
        r._brain_cache = brain
        return r
    return _make


# IDF here is `log(N / (1 + df))`, so on a tiny or homogeneous corpus every term
# scores <= 0 and `score > 0` filters out everything. That is a property of the
# existing scoring, not of the category fix — the live brain has ~400 documents,
# where it is positive for any realistic df. Fixtures therefore need enough
# VARIED documents for the discriminating term to be rare. Do not "fix" this by
# changing the IDF formula: that re-ranks the entire brain and needs its own
# justification.
_FILLER = [
    "FX desk quiet zero signals above threshold zero orders",
    "Rates desk no qualifying curve trades this session",
    "International desk ADR basket unchanged no rebalance",
    "Options desk spread width outside tolerance skipped",
    "Macro desk regime unchanged bull calm no action",
    "Prediction desk no order path venue unavailable",
    "Equity desk funnel narrowed by confidence gate",
    "Crypto desk insufficient available cash under minimum",
]


def _desk_brain(*distinctive: str) -> dict:
    """A desk_outcomes-only brain: any hit MUST come from that category."""
    entries = [{"channel": "desk-x", "source": "desk_run_summary", "summary": t}
               for t in _FILLER]
    for t in distinctive:
        entries.append({"channel": "desk-commodities", "source": "desk_run_summary",
                        "summary": t})
    return {"desk_outcomes": entries}


def test_desk_outcomes_is_in_the_default_categories():
    assert "desk_outcomes" in mm.DEFAULT_SEARCH_CATEGORIES, (
        "desk_outcomes dropped out of the search set again. That is 100 of the "
        "brain's 403 entries — every order the desks placed, with side, notional "
        "and confidence — invisible to every agent prompt."
    )


def test_a_brain_of_only_desk_outcomes_returns_hits(retriever):
    """Listing the category is not the same as searching it.

    The brain here contains nothing else, so a non-empty result can only have
    come from desk_outcomes. A version that lists the name but never reads it
    returns [].
    """
    r = retriever(_desk_brain(
        "*Commodities Desk* — 3 order(s) placed time_series_momentum/SLV BUY $200 conf=100%"))
    hits = r.search("SLV time_series_momentum placed", n=5)
    assert hits, (
        "desk_outcomes is named in the default categories but produced no hits "
        "from a brain containing only desk_outcomes — it is listed, not searched."
    )
    assert all(h["_category"] == "desk_outcomes" for h in hits)


def test_the_summary_field_is_what_gets_indexed(retriever):
    """desk_outcomes entries carry `summary`, not `lesson`.

    If the text extractor stops falling through to `summary`, every entry
    degrades to `str(entry)` — still indexable, but the dict punctuation and key
    names dilute the term statistics and the ranking quietly rots.
    """
    r = retriever(_desk_brain("kangaroo divergence in AUDUSD carry"))
    hits = r.search("kangaroo", n=2)
    assert hits and "kangaroo" in hits[0]["summary"], (
        "a term present only in `summary` did not rank first, so summary is no "
        "longer the indexed field for desk_outcomes."
    )


def test_the_old_recency_path_still_includes_desk_outcomes():
    """The fallback must not regress the same way the retriever did."""
    src = (Path(__file__).resolve().parent / "llm_common.py").read_text()
    assert 'brain.get("desk_outcomes", [])' in src, (
        "get_company_context no longer reads desk_outcomes. That path is the "
        "fallback when memory_manager fails to import — losing it there too "
        "would remove desk outcomes from agent context entirely."
    )


def test_unsearched_categories_flags_a_populated_orphan():
    brain = {"episodic": [{"lesson": "x"}], "brand_new_feed": [{"summary": "y"}]}
    assert mm._unsearched_categories(brain) == ["brand_new_feed"], (
        "a populated brain category outside the search set was not reported. "
        "This is exactly how desk_outcomes went missing for months."
    )


def test_unsearched_categories_ignores_empty_and_non_lists():
    """Only categories that actually hold entries are worth reporting."""
    # `future_feed` is the load-bearing case: NOT in the search set, so only the
    # emptiness check can exclude it. Using an already-searched name here (as an
    # earlier draft did) makes the test pass whether or not that check exists.
    brain = {
        "future_feed": [],                  # unsearched AND empty — not a drift
        "learnings": [],                    # declared, empty
        "platform": {"name": "QuantEdge"},  # dict, not a memory list
        "version": "3",                     # scalar
        "episodic": [{"lesson": "x"}],      # searched
    }
    assert mm._unsearched_categories(brain) == [], (
        "empty lists, dicts or scalars were reported as unsearched categories — "
        "the warning would fire on every run and stop being read."
    )


def test_unsearched_categories_survives_a_missing_brain():
    assert mm._unsearched_categories({}) == []
    assert mm._unsearched_categories(None) == []


def test_drift_is_warned_at_runtime_not_asserted_in_ci(retriever, capsys):
    r = retriever({"desk_outcomes": [{"summary": "a"}], "mystery_feed": [{"summary": "b"}]})
    r.search("a", n=1)
    err = capsys.readouterr().err
    assert "mystery_feed" in err, (
        "a new populated category produced no runtime warning, so the next "
        "drift is as invisible as this one was."
    )


def test_the_warning_fires_once_per_process(retriever, capsys):
    """A per-call warning on every LLM prompt would be pure log noise."""
    r = retriever({"episodic": [{"lesson": "a"}], "mystery_feed": [{"summary": "b"}]})
    for _ in range(3):
        r.search("a", n=1)
    assert capsys.readouterr().err.count("mystery_feed") == 1, (
        "the drift warning repeats per search call."
    )


def test_an_explicit_category_list_does_not_warn(retriever, capsys):
    """Callers narrowing the search on purpose are not drifting."""
    r = retriever({"episodic": [{"lesson": "a"}], "mystery_feed": [{"summary": "b"}]})
    r.search("a", n=1, categories=["episodic"])
    assert "mystery_feed" not in capsys.readouterr().err, (
        "a deliberate `categories=[...]` call was reported as drift."
    )


def test_the_shipped_brain_has_no_unsearched_categories_right_now():
    """Informational, and deliberately tolerant of the file being absent.

    NOT an assertion on live mutable state — see the module docstring. It runs
    against the file only when present and only checks the categories THIS
    commit knows about, so a bot adding a new feed cannot fail CI; the runtime
    warning covers that case instead.
    """
    f = Path(__file__).resolve().parents[1] / "state" / "company_brain.json"
    if not f.exists():
        pytest.skip("no shipped brain in this checkout")
    brain = json.loads(f.read_text())
    known_at_commit = {"episodic", "desk_outcomes", "github_insights",
                       "experiment_results", "learnings", "experiments"}
    still_missing = [c for c in known_at_commit
                     if brain.get(c) and c not in mm.DEFAULT_SEARCH_CATEGORIES]
    assert not still_missing, (
        f"categories known to exist at this commit are not searched: {still_missing}"
    )
