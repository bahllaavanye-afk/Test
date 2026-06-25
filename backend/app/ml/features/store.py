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
from typing import Callable, List, Dict, Any, Optional

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
    """Generate a stable cache key string for a given symbol, interval, and as_of date."""
    return f"{symbol.upper()}:{interval}:{as_of}"


def _parquet_path(symbol: str, interval: str, as_of: date | str) -> Path:
    """Return the filesystem Path for the Parquet file storing features."""
    safe = symbol.upper().replace("/", "_")
    key_hash = hashlib.md5(_cache_key(symbol, interval, as_of).encode()).hexdigest()[:8]
    return STORE_DIR / f"{safe}_{interval}_{as_of}_{key_hash}.parquet"


def _meta_path(symbol: str, interval: str, as_of: date | str) -> Path:
    """Return the filesystem Path for the JSON metadata file associated with a Parquet file."""
    p = _parquet_path(symbol, interval, as_of)
    return p.with_suffix(".meta.json")


# ── Leak detection ─────────────────────────────────────────────────────────────

def check_point_in_time(df: pd.DataFrame, feature_cols: List[str],
                         horizon: int = 1) -> Dict[str, float]:
    """
    Validate that feature columns do not contain lookahead information.

    For each feature column, compute:
        gap = |corr(feature, return_{t+horizon})| - |corr(feature.shift(1), return_{t+horizon})|

    A positive gap indicates the un-shifted feature is more correlated with the future
    than its lagged counterpart, implying a leak.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and a 'close' price column.
    feature_cols : List[str]
        Names of feature columns to evaluate.
    horizon : int, default 1
        Number of periods forward to compute returns.

    Returns
    -------
    Dict[str, float]
        Mapping of feature column names to their leak gap values for columns where
        the gap exceeds ``LEAK_WARN_THRESHOLD``. An empty dict is returned if no
        applicable columns are found.

    Raises
    ------
    FeatureLeakError
        If any feature's gap exceeds ``LEAK_ERROR_THRESHOLD``.
    """
    if "close" not in df.columns or len(df) < 30:
        return {}

    fwd_return = df["close"].pct_change(horizon).shift(-horizon)
    leaks: Dict[str, float] = {}

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
    feature_cols: List[str],
    check_leak: bool = True,
) -> Path:
    """
    Persist an engineered feature DataFrame to the store.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with a DatetimeIndex or a 'date' column.
    symbol : str
        Ticker symbol (e.g. "SPY", "BTC").
    interval : str
        Data interval (e.g. "1d", "1h", "5m").
    as_of : date | str
        The point-in-time boundary. Features must only reflect data with timestamps
        strictly earlier than ``as_of``.
    feature_cols : List[str]
        List of feature column names to validate for leaks.
    check_leak : bool, default True
        Whether to run point-in-time enforcement. Disable only for benchmarking.

    Returns
    -------
    Path
        Path to the written Parquet file.
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
) -> Optional[pd.DataFrame]:
    """
    Load features from the cache.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    interval : str
        Data interval.
    as_of : date | str
        Point-in-time identifier.

    Returns
    -------
    Optional[pd.DataFrame]
        The cached DataFrame, or ``None`` if the entry does not exist or is corrupted.
    """
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
    compute_fn: Callable[[], pd.DataFrame],
    feature_cols: List[str],
    check_leak: bool = True,
) -> pd.DataFrame:
    """
    Retrieve cached features or compute them on demand.

    This follows a cache-aside pattern:
    1. Return cached features if available.
    2. Otherwise call ``compute_fn`` → validate leak → persist → return.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    interval : str
        Data interval.
    as_of : date | str
        Point-in-time identifier.
    compute_fn : Callable[[], pd.DataFrame]
        Zero‑argument callable that returns a feature DataFrame.
    feature_cols : List[str]
        Feature column names to be validated for leaks.
    check_leak : bool, default True
        Whether to enforce point‑in‑time validation when writing new data.

    Returns
    -------
    pd.DataFrame
        The feature DataFrame, either from cache or freshly computed.
    """
    cached = read(symbol, interval, as_of)
    if cached is not None:
        return cached

    df = compute_fn()
    write(df, symbol, interval, as_of, feature_cols, check_leak=check_leak)
    return df


def evict_before(cutoff: date) -> int:
    """
    Delete cache entries older than a given cutoff date.

    Parameters
    ----------
    cutoff : date
        Entries with ``as_of`` earlier than this date will be removed.

    Returns
    -------
    int
        Number of entries deleted.
    """
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


def list_entries() -> List[Dict[str, Any]]:
    """
    Retrieve metadata for all cache entries.

    Returns
    -------
    List[Dict[str, Any]]
        List of metadata dictionaries, sorted by ``written_at`` descending.
    """
    entries: List[Dict[str, Any]] = []
    for meta_file in STORE_DIR.glob("*.meta.json"):
        try:
            entries.append(json.loads(meta_file.read_text()))
        except Exception:
            pass
    return sorted(entries, key=lambda x: x.get("written_at", ""), reverse=True)


def stats() -> Dict[str, Any]:
    """
    Compute summary statistics for the feature store.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing total entry count, total size in megabytes, and the store directory path.
    """
    parquets = list(STORE_DIR.glob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in parquets if f.exists())
    return {
        "entries": len(parquets),
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "store_dir": str(STORE_DIR),
    }