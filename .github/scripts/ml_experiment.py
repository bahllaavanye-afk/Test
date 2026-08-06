"""Weekly CPU ML experiment: walk-forward direction model, honestly scored.

torch cannot run on the 512MB Render dyno, so the deep models degrade there.
This script is the durable alternative: a scikit-learn gradient-boosting
classifier trained walk-forward (expanding window, monthly refits) on daily
bars, scored ONLY out-of-sample. No look-ahead: features at t predict the
direction of the t+1 close, and every prediction comes from a model fitted
strictly on data before its refit date.

Data: Alpaca daily bars when keys are present, else yfinance. If neither
yields data the symbol is reported as skipped — never fabricated.

Output: JSON to stdout + markdown to $GITHUB_STEP_SUMMARY when set.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
# Rolling history of walk-forward runs. Enough to see a trend across weeks
# without the file becoming another unbounded state blob (see the 2026-08-05
# agent_memory.json entry — 47% of the git repo).
_HISTORY_KEEP = 60
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SYMBOLS = [s.strip() for s in os.environ.get("ML_SYMBOLS", "SPY,QQQ,NVDA").split(",") if s.strip()]
LOOKBACK_DAYS = int(os.environ.get("ML_LOOKBACK_DAYS", "1460"))  # ~4y of dailies
MIN_TRAIN = 252          # first fit needs a year
REFIT_EVERY = 21         # monthly refits
LONG_THRESHOLD = 0.55    # go long only when P(up) clears this
ANNUALIZE = np.sqrt(252)


def fetch_alpaca(symbol: str) -> pd.DataFrame | None:
    key, sec = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", "")
    if not (key and sec):
        return None
    import httpx

    start = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    try:
        r = httpx.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            params={"symbols": symbol, "timeframe": "1Day", "start": start,
                    "limit": 10000, "adjustment": "split", "feed": "iex"},
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"  [{symbol}] alpaca HTTP {r.status_code}", file=sys.stderr)
            return None
        bars = (r.json().get("bars") or {}).get(symbol) or []
        if len(bars) < MIN_TRAIN + 100:
            return None
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["t"])
        df = df.rename(columns={"c": "close", "h": "high", "l": "low", "v": "volume"})
        return df.set_index("date")[["close", "high", "low", "volume"]].astype(float)
    except Exception as exc:
        print(f"  [{symbol}] alpaca error: {exc}", file=sys.stderr)
        return None


def fetch_yfinance(symbol: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        hist = yf.download(symbol, period=f"{LOOKBACK_DAYS}d", interval="1d",
                           auto_adjust=True, progress=False)
        if hist is None or len(hist) < MIN_TRAIN + 100:
            return None
        hist.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in hist.columns]
        return hist[["close", "high", "low", "volume"]].astype(float)
    except Exception as exc:
        print(f"  [{symbol}] yfinance error: {exc}", file=sys.stderr)
        return None


def fetch_bars(symbol: str) -> "tuple[pd.DataFrame | None, str]":
    """Bars for `symbol` from whichever source has the LONGER history.

    Not a preference — a correctness fix. The two sources disagree by ~49%:
    Alpaca's free IEX feed yields 940 usable rows, yfinance 1399, and the old
    code took Alpaca and silently fell back only when it returned nothing. So
    the evaluation window flipped between runs, and even between symbols
    inside one run (2026-08-05 09:35: SPY yfinance/1399, QQQ alpaca/940).

    That is not a cosmetic difference. The *benchmark* moves with the window —
    SPY buy-and-hold Sharpe was 1.482 on 940 rows and 0.789 on 1399, because
    the longer series includes the 2022 bear market. Two runs 7 minutes apart
    disagreed about the benchmark by 2x, so no two runs were comparable and
    "beats buy-and-hold" meant nothing without knowing which window it used.

    Taking the longest available makes the choice deterministic for a given
    symbol and date instead of dependent on which fetch happened to succeed.
    """
    candidates: list[tuple[pd.DataFrame, str]] = []
    alpaca = fetch_alpaca(symbol)
    if alpaca is not None and len(alpaca):
        candidates.append((alpaca, "alpaca"))
    yfin = fetch_yfinance(symbol)
    if yfin is not None and len(yfin):
        candidates.append((yfin, "yfinance"))
    if not candidates:
        return None, "none"
    # Ties go to the first source listed, so the pick stays stable.
    df, source = max(candidates, key=lambda c: len(c[0]))
    if len(candidates) == 2:
        other = [c for c in candidates if c[1] != source][0]
        print(f"  [{symbol}] {source} {len(df)} rows > {other[1]} {len(other[0])} rows",
              file=sys.stderr)
    return df, source


def common_window(frames: "dict[str, pd.DataFrame]") -> "dict[str, pd.DataFrame]":
    """Truncate every symbol to the window all of them cover.

    Comparing SPY on 1399 rows against QQQ on 940 in the same run is comparing
    two different periods and calling it a cross-sectional result. Trimming to
    the shared window costs the extra history of the longest symbol and buys
    the ability to read the run as one experiment.

    Both ends are clamped, not just the start. In practice every source ends at
    the last close, but a lagging feed that has not published today's bar would
    otherwise leave one symbol a day short — the same defect at the other end,
    and much harder to notice because the row counts stay close.
    """
    usable = {s: d for s, d in frames.items() if d is not None and len(d)}
    if len(usable) < 2:
        return usable
    start = max(d.index.min() for d in usable.values())
    end = min(d.index.max() for d in usable.values())
    return {s: d[(d.index >= start) & (d.index <= end)] for s, d in usable.items()}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ret = df["close"].pct_change()
    for lag in (1, 2, 3, 5, 10):
        out[f"ret_{lag}d"] = df["close"].pct_change(lag)
    out["vol_20d"] = ret.rolling(20).std()
    out["mom_60d"] = df["close"].pct_change(60)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    out["dist_sma50"] = df["close"] / df["close"].rolling(50).mean() - 1
    out["target"] = (ret.shift(-1) > 0).astype(int)  # direction of NEXT day
    out["next_ret"] = ret.shift(-1)
    return out.dropna()


def sharpe_noise_floor(n_days: int) -> float:
    """Smallest Sharpe difference worth calling a win over `n_days` samples.

    A Sharpe estimated from n daily returns carries a standard error of roughly
    `sqrt(1/n)` annualised out; the difference of two such estimates is about
    `sqrt(2/n)`. Two of those is the usual bar for "not noise", so the floor is
    `2 * sqrt(2 / n)` — about **0.145 over a 380-day sub-window**, tightening as
    the window grows.

    This is an approximation, deliberately. It ignores the correlation between
    the strategy and its benchmark (which makes the true SE smaller, so the
    floor is conservative) and the fat tails of daily returns (which makes it
    larger). It exists to stop near-zero comparisons being reported as
    evidence, not to be a significance test.
    """
    import math

    if n_days <= 1:
        return float("inf")     # nothing is distinguishable from one sample
    return 2.0 * math.sqrt(2.0 / n_days)


def sub_window_stats(strat_ret, bench_ret, dates, n_windows: int = 3) -> list[dict]:
    """Per-sub-period Sharpe for an already-computed out-of-sample series.

    The 2026-08-05 19:22 run beat buy-and-hold on QQQ 1.201 vs 0.734 and NVDA
    1.184 vs 1.130 — on **one** window. A single number cannot distinguish "the
    model has an edge" from "the model was long through one good stretch and
    flat through a crash", and those imply opposite decisions about wiring it
    into orders.

    This costs nothing: the walk-forward already produced both return series,
    so slicing them adds no model fits. Equal-length slices by row count, which
    for daily bars is equal trading time.
    """
    import numpy as np

    out: list[dict] = []
    total = len(strat_ret)
    # A length mismatch is a programming error, and a silent one: the slice
    # indices stay in range against a LONGER index, so every reported date is
    # simply wrong — shifted by MIN_TRAIN, about a year. Passing `feat.index`
    # instead of `feat.index[mask]` does exactly that and changes no other
    # output, which is why it survived the first mutation pass.
    if len(dates) != total:
        raise ValueError(
            f"sub_window_stats: {total} returns but {len(dates)} dates — "
            f"the dates must be masked to the out-of-sample rows")
    if total < n_windows * 30:      # <30 OOS days a slice says nothing
        return out
    edges = [round(i * total / n_windows) for i in range(n_windows + 1)]

    def _sharpe(x) -> float:
        return float(ANNUALIZE * x.mean() / x.std()) if len(x) and x.std() > 0 else 0.0

    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        s, b = strat_ret[lo:hi], bench_ret[lo:hi]
        ss, bs = _sharpe(s), _sharpe(b)
        margin = ss - bs
        floor = sharpe_noise_floor(hi - lo)
        out.append({
            "from": str(dates[lo])[:10],
            "to": str(dates[hi - 1])[:10],
            "days": hi - lo,
            "strategy_sharpe": round(ss, 3),
            "buyhold_sharpe": round(bs, 3),
            "margin": round(margin, 3),
            "noise_floor": round(floor, 3),
            # `beats` now REQUIRES the margin to clear the floor, so a caller
            # tallying it is not counting noise. See the 2026-08-06 01:20 entry:
            # SPY reported 0.102 vs 0.087 as `beats`, which is arithmetically
            # true, indistinguishable from zero, and counted equally with QQQ's
            # 2.057 vs 0.176 in the same table.
            "beats": bool(margin > floor),
            "verdict": ("beats" if margin > floor
                        else "loses" if margin < -floor
                        else "inconclusive"),
        })
    return out


def walk_forward(feat: pd.DataFrame) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier

    cols = [c for c in feat.columns if c not in ("target", "next_ret")]
    X, y = feat[cols].values, feat["target"].values
    next_ret = feat["next_ret"].values

    proba = np.full(len(feat), np.nan)
    model = None
    for i in range(MIN_TRAIN, len(feat)):
        if model is None or (i - MIN_TRAIN) % REFIT_EVERY == 0:
            model = GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ).fit(X[:i], y[:i])
        proba[i] = model.predict_proba(X[i : i + 1])[0, 1]

    mask = ~np.isnan(proba)
    p, r, actual = proba[mask], next_ret[mask], y[mask]
    long_sig = p > LONG_THRESHOLD
    strat_ret = np.where(long_sig, r, 0.0)

    def sharpe(x: np.ndarray) -> float:
        return float(ANNUALIZE * x.mean() / x.std()) if x.std() > 0 else 0.0

    hit = float(((p > 0.5).astype(int) == actual).mean())
    curve = np.cumprod(1 + strat_ret)
    dd = float((1 - curve / np.maximum.accumulate(curve)).max())
    return {
        # Same arrays, sliced — no extra fits. Answers whether the headline
        # edge is consistent or one lucky stretch.
        "sub_windows": sub_window_stats(strat_ret, r, feat.index[mask]),
        "oos_days": int(mask.sum()),
        "hit_rate": round(hit, 4),
        "time_in_market": round(float(long_sig.mean()), 4),
        "strategy_sharpe": round(sharpe(strat_ret), 3),
        "buyhold_sharpe": round(sharpe(r), 3),
        "strategy_total_return": round(float(curve[-1] - 1), 4),
        "buyhold_total_return": round(float(np.prod(1 + r) - 1), 4),
        "max_drawdown": round(dd, 4),
    }


def main() -> int:
    results = {}
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for sym in SYMBOLS:
        print(f"[{sym}] fetching…", file=sys.stderr)
        df, source = fetch_bars(sym)
        if df is None:
            results[sym] = {"status": "skipped", "reason": "no data from alpaca or yfinance"}
            continue
        frames[sym] = df
        sources[sym] = source

    # One window for the whole run, so the symbols can be read against each
    # other and against previous runs.
    frames = common_window(frames)

    for sym, df in frames.items():
        feat = build_features(df)
        if len(feat) < MIN_TRAIN + 60:
            results[sym] = {"status": "skipped", "reason": f"only {len(feat)} usable rows"}
            continue
        print(f"[{sym}] walk-forward on {len(feat)} rows…", file=sys.stderr)
        metrics = walk_forward(feat)
        metrics.update({
            "status": "ok",
            "source": sources[sym],
            "rows": len(feat),
            # The window is the thing that made every earlier comparison
            # invalid, so it is recorded explicitly rather than inferred from
            # the row count.
            "first_date": str(feat.index.min().date()),
            "last_date": str(feat.index.max().date()),
        })
        results[sym] = metrics

    payload = {
        "experiment": "gbc_walkforward_daily",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "params": {"min_train": MIN_TRAIN, "refit_every": REFIT_EVERY,
                   "long_threshold": LONG_THRESHOLD},
        "results": results,
    }
    print(json.dumps(payload, indent=2))

    # Persist. Until 2026-08-05 this function's entire output was a print: the
    # walk-forward ran on real bars every time, produced real out-of-sample
    # numbers, and threw them away. There was no history, so nothing could
    # answer "is the ML edge improving or decaying?" — the one question a
    # weekly experiment exists to answer.
    #
    # Same failure the attribution file had (computed in an ephemeral runner,
    # never committed) and the same fix: write it where a consumer can read it.
    # Rolling window, newest last, so the file stays bounded.
    try:
        hist_path = _REPO / ".github" / "state" / "ml_experiments.json"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            history = json.loads(hist_path.read_text())
            if not isinstance(history, list):
                history = []
        except Exception:  # noqa: BLE001 — a corrupt file must not lose this run
            history = []
        history.append(payload)
        hist_path.write_text(json.dumps(history[-_HISTORY_KEEP:], indent=2))
        print(f"[ml_experiment] appended to {hist_path} ({len(history[-_HISTORY_KEEP:])} runs kept)",
              flush=True)
    except Exception as exc:  # noqa: BLE001 — never fail the experiment over bookkeeping
        print(f"[ml_experiment] could not persist results: {exc}", flush=True)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write("## 🧪 Weekly ML experiment — walk-forward GBC (out-of-sample only)\n\n")
            fh.write("| Symbol | OOS days | Hit rate | Strat Sharpe | B&H Sharpe | Strat ret | B&H ret | Max DD | In market |\n")
            fh.write("|---|---|---|---|---|---|---|---|---|\n")
            for sym, m in results.items():
                if m.get("status") != "ok":
                    fh.write(f"| {sym} | — | — | — | — | — | — | — | {m.get('reason')} |\n")
                    continue
                fh.write(
                    f"| {sym} | {m['oos_days']} | {m['hit_rate']:.1%} | {m['strategy_sharpe']} "
                    f"| {m['buyhold_sharpe']} | {m['strategy_total_return']:.1%} "
                    f"| {m['buyhold_total_return']:.1%} | {m['max_drawdown']:.1%} "
                    f"| {m['time_in_market']:.1%} |\n"
                )
            fh.write("\nA strategy only earns promotion when its OOS Sharpe beats buy-and-hold — "
                     "walk-forward numbers above are the honest bar, not in-sample fits.\n")

    ok = [m for m in results.values() if m.get("status") == "ok"]
    return 0 if ok else 1  # no symbol trained at all → fail loudly


if __name__ == "__main__":
    sys.exit(main())
