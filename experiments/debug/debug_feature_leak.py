"""
Detect lookahead bias in feature engineering.
Checks whether any feature has a negative lag (uses future data).
"""
from __future__ import annotations
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def check_lookahead(df_features: pd.DataFrame, df_prices: pd.DataFrame) -> dict:
    """
    For each feature, check if it's correlated with FUTURE returns.
    A perfectly predictive feature with 0-lag is suspicious — may be leaked.
    """
    results = {}
    future_return = df_prices["close"].pct_change().shift(-1).rename("future_return")

    for col in df_features.columns:
        if df_features[col].dtype not in [np.float32, np.float64, float, int]:
            continue
        try:
            series = df_features[col].dropna()
            aligned = pd.concat([series, future_return], axis=1).dropna()
            corr_lag0 = aligned.corr().iloc[0, 1]

            # Check lagged correlation (lag 1 = proper usage)
            lag1_series = series.shift(1).dropna()
            aligned_lag1 = pd.concat([lag1_series, future_return], axis=1).dropna()
            corr_lag1 = aligned_lag1.corr().iloc[0, 1]

            leak_suspected = abs(corr_lag0) > abs(corr_lag1) + 0.10

            results[col] = {
                "corr_lag0": round(float(corr_lag0), 4),
                "corr_lag1": round(float(corr_lag1), 4),
                "leak_suspected": leak_suspected,
            }
        except Exception:
            pass

    leaky = [k for k, v in results.items() if v["leak_suspected"]]
    return {"features": results, "leaky_features": leaky, "clean": len(leaky) == 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    args = parser.parse_args()

    import yfinance as yf
    from app.ml.features.engineer import engineer_features

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True) if Path(args.csv).exists() \
        else yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False)
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    features = engineer_features(df)

    result = check_lookahead(features, df)
    if result["clean"]:
        print("✓ No lookahead bias detected")
    else:
        print(f"⚠ Potential lookahead in: {result['leaky_features']}")
    for col, info in result["features"].items():
        if info.get("leak_suspected"):
            print(f"  {col}: lag0={info['corr_lag0']}, lag1={info['corr_lag1']}")
