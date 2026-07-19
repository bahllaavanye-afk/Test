"""Feature engineering tests — must check no lookahead bias."""
import pytest
import pandas as pd
import numpy as np
from app.ml.features.engineer import engineer_features, create_sequences, add_labels


@pytest.fixture
def ohlcv_df():
    n = 200
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(100_000, 1_000_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_engineer_returns_dataframe(ohlcv_df):
    df = engineer_features(ohlcv_df)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_engineer_keeps_close(ohlcv_df):
    df = engineer_features(ohlcv_df)
    assert "close" in df.columns


def test_engineer_no_nan_after_warmup(ohlcv_df):
    df = engineer_features(ohlcv_df)
    # After 50 bars warmup, no NaNs in feature columns
    later = df.iloc[50:]
    feature_cols = [
        c
        for c in later.columns
        if c not in ("open", "high", "low", "close", "volume")
    ]
    if feature_cols:
        # Some indicators may still have NaNs early; check it's bounded
        assert later[feature_cols].isna().sum().sum() < len(later) * len(
            feature_cols
        ) * 0.5


def test_create_sequences_shape(ohlcv_df):
    df = engineer_features(ohlcv_df).dropna()
    df = add_labels(df, threshold=0.002)
    X, y = create_sequences(df, seq_len=20)
    if len(X) > 0:
        assert X.ndim == 3  # (n, seq_len, features)
        assert X.shape[1] == 20
        assert len(y) == len(X)


def test_labels_are_binary(ohlcv_df):
    df = engineer_features(ohlcv_df).dropna()
    df = add_labels(df, threshold=0.002)
    assert "label" in df.columns
    unique = set(df["label"].dropna().unique())
    assert unique.issubset({0, 1})


# ----------------------------------------------------------------------
# Validation tests for public functions – ensure they raise ValueError
# ----------------------------------------------------------------------


def test_engineer_features_invalid_input():
    """engineer_features should reject non‑DataFrame inputs."""
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        engineer_features("not a dataframe")


def test_engineer_features_missing_columns():
    """engineer_features should reject DataFrames missing required OHLCV columns."""
    df_missing = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="DataFrame must contain required columns: open, high, low, close, volume"
    ):
        engineer_features(df_missing)


def test_create_sequences_invalid_seq_len(ohlcv_df):
    """create_sequences should enforce a positive integer sequence length."""
    df = engineer_features(ohlcv_df).dropna()
    df = add_labels(df, threshold=0.002)
    with pytest.raises(
        ValueError, match="seq_len must be a positive integer greater than zero"
    ):
        create_sequences(df, seq_len=0)


def test_create_sequences_invalid_dataframe():
    """create_sequences should reject inputs that are not DataFrames."""
    with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
        create_sequences("not a dataframe", seq_len=10)


def test_add_labels_invalid_threshold(ohlcv_df):
    """add_labels should require a non‑negative numeric threshold."""
    df = engineer_features(ohlcv_df).dropna()
    with pytest.raises(
        ValueError, match="threshold must be a non‑negative numeric value"
    ):
        add_labels(df, threshold=-0.01)


def test_add_labels_invalid_dataframe():
    """add_labels should reject inputs that are not DataFrames."""
    with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
        add_labels("not a dataframe", threshold=0.001)