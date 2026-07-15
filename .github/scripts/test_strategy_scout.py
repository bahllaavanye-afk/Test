"""Strategy Scout guards: desk suggestion + digest are pure and stable."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from strategy_scout import RESEARCH_BACKLOG, build_digest, suggest_desk


def test_suggest_desk_routes_by_family():
    assert suggest_desk("poly_orderbook_arb") == "Polymarket"
    assert suggest_desk("crypto_basis_roll") == "Crypto"
    assert suggest_desk("options_pcr_reversal") == "Options"
    assert suggest_desk("yield_curve_momentum") == "Macro/FX"
    assert suggest_desk("commodity_trend") == "Commodities"
    assert suggest_desk("kalman_pairs") == "StatArb"
    assert suggest_desk("hull_suite_tv") == "TV Indicators"
    assert suggest_desk("fifty_two_week_high") == "Equities"   # default


def test_digest_lists_unwired_and_coverage():
    d = build_digest({"foo_tv": "TV Indicators", "poly_x": "Polymarket"},
                     [("Equities", 26, 30), ("Crypto", 14, 20)],
                     [("meta_labeling_gate", "Equities", "P(win) gate")])
    assert "2 registry strategies trade on NO desk" in d
    assert "`foo_tv` → suggest **TV Indicators**" in d
    assert "Equities: 26 × 30" in d
    assert "meta_labeling_gate" in d


def test_digest_all_wired_is_positive():
    d = build_digest({}, [("Equities", 26, 30)], [])
    assert "Every wirable registry strategy is on a desk" in d


def test_digest_marks_known_exclusions_as_blocked():
    d = build_digest({"covered_call": "Options"}, [("Options", 17, 8)], [])
    assert "excluded pending a data source" in d
    assert "share inventory" in d
    assert "wire these" not in d          # exclusions are not free wins


def test_research_backlog_is_well_formed():
    assert len(RESEARCH_BACKLOG) >= 10
    for key, desk, desc in RESEARCH_BACKLOG:
        assert key and desk and len(desc) > 10
