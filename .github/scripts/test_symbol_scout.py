"""Symbol Scout guards: validation/proposal/digest are pure and honest."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from symbol_scout import (
    EQUITY_WATCHLIST,
    build_digest,
    propose_crypto,
    propose_equities,
    validate_universe,
)


def test_validate_flags_only_unlisted():
    dead = validate_universe({"SPY", "FAKE1", "GLD"}, {"SPY", "GLD", "QQQ"})
    assert dead == ["FAKE1"]


def test_propose_crypto_only_usd_pairs_not_wired():
    out = propose_crypto({"PEPE/USD", "BTC/USD", "ETH/BTC"}, {"BTC/USD"})
    assert out == ["PEPE/USD"]                     # ETH/BTC (non-USD) excluded


def test_propose_equities_requires_tradable_and_unwired():
    out = propose_equities(["XLV", "SMH", "GLD"], {"XLV", "GLD"}, {"GLD"})
    assert out == ["XLV"]                          # SMH untradable, GLD wired


def test_digest_dead_symbols_are_loud():
    d = build_digest(["FAKE1"], [], [])
    assert "NOT tradable" in d and "`FAKE1`" in d


def test_digest_clean_run_is_positive():
    d = build_digest([], [], [])
    assert "Every desk symbol is an active, tradable Alpaca asset" in d
    assert "No new symbol proposals" in d


def test_watchlist_is_sane():
    assert len(EQUITY_WATCHLIST) >= 15
    assert len(set(EQUITY_WATCHLIST)) == len(EQUITY_WATCHLIST)
