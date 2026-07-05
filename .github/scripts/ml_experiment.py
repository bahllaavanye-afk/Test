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

    hit = float((p > 0.5).astype(int).mean() * 0 + ((p > 0.5).astype(int) == actual).mean())
    curve = np.cumprod(1 + strat_ret)
    dd = float((1 - curve / np.maximum.accumulate(curve)).max())
    return {
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
    for sym in SYMBOLS:
        print(f"[{sym}] fetching…", file=sys.stderr)
        df = fetch_alpaca(sym)
        source = "alpaca"
        if df is None:
            df, source = fetch_yfinance(sym), "yfinance"
        if df is None:
            results[sym] = {"status": "skipped", "reason": "no data from alpaca or yfinance"}
            continue
        feat = build_features(df)
        if len(feat) < MIN_TRAIN + 60:
            results[sym] = {"status": "skipped", "reason": f"only {len(feat)} usable rows"}
            continue
        print(f"[{sym}] walk-forward on {len(feat)} rows…", file=sys.stderr)
        metrics = walk_forward(feat)
        metrics.update({"status": "ok", "source": source, "rows": len(feat)})
        results[sym] = metrics

    payload = {
        "experiment": "gbc_walkforward_daily",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "params": {"min_train": MIN_TRAIN, "refit_every": REFIT_EVERY,
                   "long_threshold": LONG_THRESHOLD},
        "results": results,
    }
    print(json.dumps(payload, indent=2))

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
