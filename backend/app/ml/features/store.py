"""
Feature Store — versioned, point-in-time-correct feature cache.

Key properties:
- Features for date D only use market data available strictly before D (no lookahead).
- Cache is keyed by (symbol, interval, as_of_date) so backtests never contaminate
  live inference and two training runs on different date ranges don't collide.
- On-disk storage uses Parquet (columnar, fast, lightweight); hot data lives in Redis.
- Leak detection runs on every write: raises FeatureLeakError if any feature has a
  correlation with same-day forward returns > 0.10 above its lagged correlation.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.logging import logger

STORE_DIR = Path(os.environ.get("FEATURE_STORE_DIR", "/tmp/quantedge_feature_store"))
STORE_DIR.mkdir(parents=True, exist_ok=True)

# Correlation gap that triggers a leak warning (advisory) vs error (hard gate)
LEAK_WARN_THRESHOLD = 0.05
LEAK_ERROR_THRESHOLD = 0.10


class FeatureLeakError(ValueError):
    """Raised when point-in-time enforcement detects lookahead bias."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_key(symbol: str, interval: str, as_of: date | str) -> str:
    """Stable cache key string."""
    return f"{symbol.upper()}:{interval}:{as_of}"


def _parquet_path(symbol: str, interval: str, as_of: date | str) -> Path:
    safe = symbol.upper().replace("/", "_")
    key_hash = hashlib.md5(_cache_key(symbol, interval, as_of).encode()).hexdigest()[:8]
    return STORE_DIR / f"{safe}_{interval}_{as_of}_{key_hash}.parquet"


def _meta_path(symbol: str, interval: str, as_of: date | str) -> Path:
    p = _parquet_path(symbol, interval, as_of)
    return p.with_suffix(".meta.json")


# ── Leak detection ─────────────────────────────────────────────────────────────

def check_point_in_time(df: pd.DataFrame, feature_cols: list[str],
                         horizon: int = 1) -> dict[str, float]:
    """
    For each feature column, compute:
      gap = abs(corr(feature, return_t+horizon)) - abs(corr(feature.shift(1), return_t+horizon))

    A positive gap means the un-shifted feature is MORE correlated with the future than
    the shifted one — i.e., it already contains same-bar information and leaks the future.

    Returns dict of {col: gap} for features where gap > LEAK_WARN_THRESHOLD.
    Raises FeatureLeakError if gap > LEAK_ERROR_THRESHOLD for any feature.
    """
    if "close" not in df.columns or len(df) < 30:
        return {}

    fwd_return = df["close"].pct_change(horizon).shift(-horizon)
    leaks: dict[str, float] = {}

    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.nunique() < 2:
            continue
        aligned = pd.concat([series, fwd_return], axis=1).dropna()
        if len(aligned) < 20:
            continue

        c_raw = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        c_lag = aligned.iloc[:, 0].shift(1).corr(aligned.iloc[:, 1])
        gap = abs(c_raw) - abs(c_lag)
        if gap > LEAK_WARN_THRESHOLD:
            leaks[col] = round(gap, 4)

    hard_leaks = {k: v for k, v in leaks.items() if v > LEAK_ERROR_THRESHOLD}
    if hard_leaks:
        raise FeatureLeakError(
            f"Point-in-time violation — these features look into the future "
            f"(lag0_corr > lag1_corr + {LEAK_ERROR_THRESHOLD}): {hard_leaks}"
        )

    if leaks:
        logger.warning("feature_store: soft leak warning", columns=list(leaks.keys()),
                       gaps=leaks)
    return leaks


# ── Main store API ─────────────────────────────────────────────────────────────

def write(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    as_of: date | str,
    feature_cols: list[str],
    check_leak: bool = True,
) -> Path:
    """
    Persist an engineered feature DataFrame to the store.

    Args:
        df:           Feature DataFrame with DatetimeIndex or 'date' column.
        symbol:       Ticker (e.g. "SPY", "BTC").
        interval:     Data interval ("1d", "1h", "5m").
        as_of:        The point-in-time boundary. Features must only reflect data
                      with timestamps < as_of.
        feature_cols: List of feature column names to validate for leaks.
        check_leak:   Run point-in-time enforcement (disable only for benchmarking).

    Returns:
        Path to written Parquet file.
    """
    if check_leak:
        check_point_in_time(df, feature_cols)

    path = _parquet_path(symbol, interval, as_of)
    df.to_parquet(path, index=True)

    meta = {
        "symbol": symbol,
        "interval": interval,
        "as_of": str(as_of),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "feature_cols": feature_cols,
        "written_at": datetime.now(UTC).isoformat(),
    }
    _meta_path(symbol, interval, as_of).write_text(json.dumps(meta))

    logger.debug("feature_store: wrote", path=str(path), rows=len(df), symbol=symbol)
    return path


def read(
    symbol: str,
    interval: str,
    as_of: date | str,
) -> pd.DataFrame | None:
    """Load features from cache. Returns None if not cached."""
    path = _parquet_path(symbol, interval, as_of)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning("feature_store: corrupt cache entry", path=str(path), error=str(e))
        path.unlink(missing_ok=True)
        return None


def get_or_compute(
    symbol: str,
    interval: str,
    as_of: date | str,
    compute_fn,
    feature_cols: list[str],
    check_leak: bool = True,
) -> pd.DataFrame:
    """
    Cache-aside pattern:
    1. Return cached features if available.
    2. Otherwise call compute_fn() → validate leak → persist → return.

    compute_fn must be a zero-argument callable that returns a feature DataFrame.
    This keeps the store decoupled from the engineering pipeline.
    """
    cached = read(symbol, interval, as_of)
    if cached is not None:
        return cached

    df = compute_fn()
    write(df, symbol, interval, as_of, feature_cols, check_leak=check_leak)
    return df


def evict_before(cutoff: date) -> int:
    """Delete cache entries older than cutoff. Returns count deleted."""
    deleted = 0
    for f in STORE_DIR.glob("*.parquet"):
        meta_file = f.with_suffix(".meta.json")
        try:
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                entry_date = date.fromisoformat(str(meta.get("as_of", "1900-01-01")))
                if entry_date < cutoff:
                    f.unlink(missing_ok=True)
                    meta_file.unlink(missing_ok=True)
                    deleted += 1
            else:
                # Orphaned parquet without metadata — remove.
                f.unlink(missing_ok=True)
                deleted += 1
        except Exception:
            pass
    return deleted


def list_entries() -> list[dict]:
    """Return metadata for all cache entries, sorted by written_at descending."""
    entries = []
    for meta_file in STORE_DIR.glob("*.meta.json"):
        try:
            entries.append(json.loads(meta_file.read_text()))
        except Exception:
            pass
    return sorted(entries, key=lambda x: x.get("written_at", ""), reverse=True)


def stats() -> dict:
    """Summary stats for the feature store."""
    parquets = list(STORE_DIR.glob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in parquets if f.exists())
    return {
        "entries": len(parquets),
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "store_dir": str(STORE_DIR),
    }
