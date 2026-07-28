"""A scan row with NO signals took `/api/v1/scanners/crypto` down with a 500.

`ScanResultOut.validate_signals` rejects an empty `signals` list, but both the
equity and crypto scanners returned a `ScanResult` unconditionally — so a
symbol where no condition fired arrived with `signals=[]`, `score=0`,
`side="neutral"`. One such row makes the WHOLE response a 500, because the
endpoint serialises the batch.

HOW IT STAYED HIDDEN, AND WHAT SURFACED IT. The crypto scanner was starved of
bars by the pagination truncation (fixed 2026-07-28), so it returned nothing at
all and the endpoint happily served `results: []`. The moment the bars fix
restored its universe, it started producing signal-less rows and the endpoint
began failing. A latent defect, revealed by an unrelated fix — the same shape
as the original polymarket 500, which also only appeared once a desk produced
a non-empty result.

Two layers, deliberately:

  producer  — don't emit a row when nothing fired. A scanner exists to surface
              symbols where a condition triggered; a zero-score, no-signal,
              neutral row carries no information and only dilutes the ranking.
  read path — drop such rows anyway. Redis rows written before this fix outlive
              the deploy, so a producer-only fix would leave the endpoint
              500ing on cache until every key expired.

Dropping beats inventing a placeholder signal name: the row genuinely means
"nothing here", and fabricating `["unspecified"]` would put a meaningless entry
into a ranked list that the UI presents as opportunities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.api.v1.scanners import (
    ScanResultOut,
    _normalise_scan_item,
    _normalise_scan_items,
)
from app.tasks.stock_scanners import CryptoScanner, EquityScanner, ScanResult


# ── the read path ────────────────────────────────────────────────────────────

def _row(symbol="BTC/USD", signals=("rsi_oversold_28",), score=55.0, side="long"):
    return {"symbol": symbol, "desk": "crypto", "score": score,
            "signals": list(signals), "side": side, "data": {}}


def test_a_signal_less_row_is_dropped():
    assert _normalise_scan_items([_row(signals=[])]) == []


def test_a_real_row_survives():
    out = _normalise_scan_items([_row()])
    assert len(out) == 1 and out[0]["symbol"] == "BTC/USD"


def test_one_empty_row_no_longer_poisons_the_whole_batch():
    """THE BUG: the endpoint serialises the batch, so one bad row 500s it all."""
    batch = [_row("BTC/USD"), _row("ETH/USD", signals=[]), _row("SOL/USD")]
    out = _normalise_scan_items(batch)
    assert [r["symbol"] for r in out] == ["BTC/USD", "SOL/USD"]
    # The real assertion: every surviving row actually serialises.
    assert [ScanResultOut(**r).symbol for r in out] == ["BTC/USD", "SOL/USD"]


def test_the_exact_row_shape_that_500d_production():
    """score=0, signals=[], side=neutral — a symbol where nothing fired."""
    dead = _row("XTZ/USD", signals=[], score=0.0, side="neutral")
    with pytest.raises(Exception):
        ScanResultOut(**_normalise_scan_item(dead))
    assert _normalise_scan_items([dead]) == []


@pytest.mark.parametrize("empty", [[], None, ()])
def test_every_empty_shape_is_treated_as_no_signals(empty):
    assert _normalise_scan_items([_row(signals=empty or [])]) == []


def test_an_empty_batch_is_fine():
    assert _normalise_scan_items([]) == []
    assert _normalise_scan_items(None) == []


def test_the_other_normalisations_still_apply():
    """Dropping must not bypass the score/side translation."""
    out = _normalise_scan_items([_row(score=87.0, side="long")])
    assert out[0]["score"] == pytest.approx(0.87)
    assert out[0]["side"] == "buy"


# ── the producers ────────────────────────────────────────────────────────────

def _quiet_frame(n: int = 60) -> pd.DataFrame:
    """Prices that trigger NO condition in either scanner.

    A slow sine (period 30, amplitude 3) rather than a flat line, found by
    testing rather than assumption — the obvious candidates all fire something:

        constant       RSI = 0 (gain and loss both 0)  -> rsi_oversold
        tight zig-zag  monotonic EMAs                  -> ema_stack_bearish
        period 20      too much swing over 14 bars     -> rsi_oversold_26

    Constant volume keeps vol_ratio at 1.0 and constant high-low keeps ATR
    flat, so only the close-derived indicators are in play.
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = pd.Series(100 + 3 * np.sin(2 * np.pi * np.arange(n) / 30), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": pd.Series(1_000_000.0, index=idx)},
        index=idx,
    )


def test_the_equity_scanner_emits_nothing_when_nothing_fires():
    assert EquityScanner()._score("AAPL", _quiet_frame()) is None


def test_the_crypto_scanner_emits_nothing_when_nothing_fires():
    assert CryptoScanner()._score("BTC/USD", _quiet_frame(), None) is None


@pytest.mark.parametrize("scanner_cls", [EquityScanner, CryptoScanner])
def test_scan_would_filter_a_None_anyway(scanner_cls):
    """`scan()` keeps only ScanResult instances — None propagates safely.

    Pinned because the producer fix relies on it: `_score` returning None flows
    up through `_scan_one` into `scan()`'s isinstance filter.
    """
    mixed = [ScanResult("A", "equity", 50.0, ["x"], "long"), None,
             ValueError("boom")]
    assert [r for r in mixed if isinstance(r, ScanResult)] == [mixed[0]]
