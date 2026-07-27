"""Implemented features must actually reach the model's input matrix.

Found by the same unreferenced-function sweep that caught the risk gate and the
exit path. Three feature builders were fully implemented and unit-tested and
called from nowhere, so every model trained without them:

    add_microstructure_features   pure OHLCV arithmetic — now wired
    add_sentiment_features        needs a caller-supplied history — not wired
    add_alternative_features      calls the Binance API — deliberately NOT wired

Only the first is wired. The other two are network- or caller-dependent, and
feature engineering runs inside backtests and training, which must stay
deterministic and work offline. `test_feature_engineering_makes_no_network_calls`
pins that.

Safe to widen FEATURE_COLS right now specifically because there are zero
trained models: no `.pt`/`.pth`/`.pkl` artifacts anywhere in the repo and the
live `/health/detailed` reports `ml_models: {count: 0}`. Changing the input
width later, after a model exists, would break inference silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.features.engineer import FEATURE_COLS, engineer_features
from app.ml.features.microstructure import MICROSTRUCTURE_FEATURE_COLS


def _ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.uniform(1e5, 1e6, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


def test_microstructure_columns_are_in_the_canonical_feature_list():
    """FEATURE_COLS is what create_sequences slices — absent here means unused."""
    for col in MICROSTRUCTURE_FEATURE_COLS:
        assert col in FEATURE_COLS, (
            f"{col} is computed but never selected, so no model ever sees it"
        )


def test_engineer_features_actually_emits_them():
    """Listing a column is not the same as producing it."""
    out = engineer_features(_ohlcv())
    for col in MICROSTRUCTURE_FEATURE_COLS:
        assert col in out.columns, f"{col} listed in FEATURE_COLS but never computed"
        assert out[col].notna().any(), f"{col} present but entirely NaN"


def test_microstructure_features_are_finite():
    """`(close-open)/(high-low)` divides by zero on a flat bar."""
    df = _ohlcv()
    # Force a flat bar — high == low — which is the divide-by-zero case.
    df.iloc[50, df.columns.get_loc("high")] = df.iloc[50]["close"]
    df.iloc[50, df.columns.get_loc("low")] = df.iloc[50]["close"]
    df.iloc[50, df.columns.get_loc("open")] = df.iloc[50]["close"]

    out = engineer_features(df)
    for col in MICROSTRUCTURE_FEATURE_COLS:
        vals = out[col].to_numpy(dtype=float)
        assert np.isfinite(vals).all(), f"{col} produced inf/NaN on a flat bar"


def test_microstructure_features_do_not_look_ahead():
    """A feature at bar t must not change when FUTURE bars change.

    The label is `close.pct_change(h).shift(-h)` — bar t predicts t+h — so bar
    t's own OHLC is legitimately known. What must never happen is bar t's
    feature depending on t+1 onward.
    """
    df = _ohlcv()
    base = engineer_features(df)

    tampered = df.copy()
    cut = len(tampered) - 20
    tampered.iloc[cut:, tampered.columns.get_loc("close")] *= 3.0
    tampered.iloc[cut:, tampered.columns.get_loc("high")] *= 3.0
    tampered.iloc[cut:, tampered.columns.get_loc("low")] *= 3.0
    after = engineer_features(tampered)

    common = base.index.intersection(after.index)
    common = common[common < df.index[cut - 5]]
    assert len(common) > 50, "need enough untouched history to compare"

    for col in MICROSTRUCTURE_FEATURE_COLS:
        pd.testing.assert_series_equal(
            base.loc[common, col], after.loc[common, col],
            check_names=False,
            obj=f"{col} changed on past bars when future bars were altered",
        )


def test_feature_engineering_makes_no_network_calls():
    """Backtests and training must be deterministic and work offline.

    This is why add_alternative_features (Binance API) is NOT wired into
    engineer_features despite being implemented — pinning the reason so a
    future change has to argue with a test rather than a comment.
    """
    import httpx

    calls = []

    class _Boom(httpx.Client):
        def request(self, *a, **kw):  # pragma: no cover - must not run
            calls.append(a)
            raise AssertionError("engineer_features made a network call")

    original_client, original_async = httpx.Client, httpx.AsyncClient
    httpx.Client = _Boom
    httpx.AsyncClient = _Boom
    try:
        engineer_features(_ohlcv())
    finally:
        httpx.Client = original_client
        httpx.AsyncClient = original_async

    assert not calls


def test_feature_columns_have_no_duplicates():
    """A duplicated column silently double-weights a feature in the matrix."""
    dupes = {c for c in FEATURE_COLS if FEATURE_COLS.count(c) > 1}
    assert not dupes, f"duplicate feature columns: {sorted(dupes)}"


@pytest.mark.parametrize("market_type", ["equity", "crypto"])
def test_both_market_types_still_engineer_cleanly(market_type):
    out = engineer_features(_ohlcv(), market_type=market_type, symbol="BTC")
    assert len(out) > 0
    for col in MICROSTRUCTURE_FEATURE_COLS:
        assert col in out.columns
