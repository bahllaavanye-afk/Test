#!/usr/bin/env python3
"""FX Desk — OANDA practice account, 7 majors, 24/5 paper orders.

Closes the IMPROVEMENTS P0 "OANDA FX desk": the FX strategies (fx_trend,
fx_reversion, interest_rate_differential, dollar_carry) were written and
registry-tested but had no venue. This desk mirrors desk_order_placer's
shape — fetch bars, run strategies (hard 10s timeout each), confidence-gate
at the 0.60 break-even, vol-target the size, place real practice orders —
against OANDA's v20 REST API (keys already relayed to secrets).

Honest guards: no keys -> clean exit 0 with a message (never fake success);
FX weekend (Fri 21:00 - Sun 21:05 UTC) -> skip; every OANDA call retries 429
and surfaces the response body on failure (the lesson of the Alpaca 403).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

OANDA_API_KEY = os.environ.get("OANDA_API_KEY", "").strip()
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "").strip()
OANDA_BASE = os.environ.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"]
STRATEGIES = ["fx_trend", "fx_reversion", "interest_rate_differential", "dollar_carry"]
CONFIDENCE_MIN = 0.60          # break-even in the Kelly model (see desk_order_placer)
BASE_UNITS = 1_000             # micro-lot-ish; conf + vol scaling applies
MAX_ORDERS_PER_RUN = 3
ANALYZE_TIMEOUT_S = 10.0       # same guard as the main desks
_UA = "Mozilla/5.0 (X11; Linux x86_64) QuantEdge-FXDesk/1.0"


def fx_market_open(now: datetime | None = None) -> bool:
    """FX trades 24/5: closed from Fri 21:00 UTC to Sun 21:05 UTC."""
    now = now or datetime.now(timezone.utc)
    wd, t = now.weekday(), now.hour * 60 + now.minute
    if wd == 4 and t >= 21 * 60:      # Friday night
        return False
    if wd == 5:                        # Saturday
        return False
    if wd == 6 and t < 21 * 60 + 5:    # Sunday until reopen
        return False
    return True


def _oanda(method: str, path: str, body: dict | None = None) -> dict:
    """OANDA v20 call with 429 retry; surfaces the response body on HTTP errors."""
    url = OANDA_BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": _UA,
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode(errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if exc.code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"  ⚠ OANDA {method} {path} → {exc.code}: {detail}", flush=True)
            raise
    raise RuntimeError("unreachable")


def fetch_candles(pair: str, count: int = 300):
    """Daily mid candles → OHLCV DataFrame (None on failure — desk skips pair)."""
    import pandas as pd
    try:
        data = _oanda("GET", f"/v3/instruments/{pair}/candles?granularity=D&price=M&count={count}")
        rows = [
            {"time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
             "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": float(c["volume"])}
            for c in data.get("candles", []) if c.get("complete")
        ]
        if len(rows) < 50:
            print(f"  ⚠ {pair}: only {len(rows)} candles", flush=True)
            return None
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time").sort_index()
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ {pair}: candles failed: {str(exc)[:100]}", flush=True)
        return None


def vol_scalar(df) -> float:
    """Same Moreira-Muir scaling as the main desks (target 20%, clamp [0.5,2])."""
    try:
        import numpy as np
        rets = np.log(df["close"] / df["close"].shift(1)).dropna().tail(20)
        realized = float(rets.std() * (252 ** 0.5))
        if not realized > 0:
            return 1.0
        return float(min(max(0.20 / realized, 0.5), 2.0))
    except Exception:  # noqa: BLE001
        return 1.0


def place_order(pair: str, side: str, units: int) -> dict | None:
    """MARKET order on the practice account. Negative units = short (OANDA v20)."""
    signed = units if side == "buy" else -units
    body = {"order": {"type": "MARKET", "instrument": pair, "units": str(signed),
                      "timeInForce": "FOK", "positionFill": "DEFAULT",
                      "clientExtensions": {"tag": "qe-fx", "id": f"qe-fx-{pair}-{int(time.time())}"}}}
    try:
        out = _oanda("POST", f"/v3/accounts/{OANDA_ACCOUNT_ID}/orders", body)
        tx = out.get("orderFillTransaction") or out.get("orderCreateTransaction") or {}
        return out if tx else None
    except Exception:  # noqa: BLE001 — _oanda already printed the reason
        return None


async def run() -> int:
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        print("FX desk: OANDA keys absent — skipping honestly (set OANDA_API_KEY/OANDA_ACCOUNT_ID).")
        return 0
    if not fx_market_open():
        print("FX desk: weekend — market closed, 0 orders.")
        return 0

    os.environ.setdefault("SECRET_KEY", "d" * 64)
    os.environ.setdefault("DATABASE_URL", "")
    from app.strategies import STRATEGY_REGISTRY

    strategies = [(n, STRATEGY_REGISTRY[n]()) for n in STRATEGIES if STRATEGY_REGISTRY.get(n)]
    print(f"FX desk: {len(PAIRS)} pairs × {len(strategies)} strategies")

    signals: list[dict] = []
    for pair in PAIRS:
        df = fetch_candles(pair)
        if df is None:
            continue
        for name, strat in strategies:
            try:
                sig = await asyncio.wait_for(strat.analyze(df, pair.replace("_", "")), ANALYZE_TIMEOUT_S)
            except asyncio.TimeoutError:
                print(f"  ⚠ {name}/{pair} timed out (>{ANALYZE_TIMEOUT_S}s) — skipped", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ {name}/{pair} error: {str(exc)[:80]}", flush=True)
                continue
            if sig is None:
                continue
            conf = float(getattr(sig, "confidence", 0) or 0)
            side = str(getattr(sig, "side", "")).lower()
            if side not in ("buy", "sell"):
                continue
            if conf < CONFIDENCE_MIN:
                print(f"  · {name}/{pair} conf={conf:.2f} < {CONFIDENCE_MIN} — skipped", flush=True)
                continue
            signals.append({"pair": pair, "side": side, "conf": conf, "strategy": name,
                            "units": max(int(BASE_UNITS * conf * vol_scalar(df)), 100)})

    signals.sort(key=lambda s: -s["conf"])
    placed = 0
    for s in signals[:MAX_ORDERS_PER_RUN]:
        print(f"  ► {s['strategy']}/{s['pair']} {s['side'].upper()} conf={s['conf']:.2f} "
              f"units={s['units']}", flush=True)
        if place_order(s["pair"], s["side"], s["units"]):
            placed += 1
            print("    ✓ order placed", flush=True)
        else:
            print("    ✗ order failed (reason above)", flush=True)

    print(f"FX desk done: {len(signals)} signal(s) ≥ gate, {placed} order(s) placed.")
    try:
        import notify
        notify.post("#desk-fx-rates",
                    f"FX desk: {len(signals)} signals ≥ {CONFIDENCE_MIN}, {placed} orders "
                    f"({', '.join(s['pair'] for s in signals[:MAX_ORDERS_PER_RUN]) or 'none'})",
                    username="QuantEdge FX Desk")
    except Exception as exc:  # noqa: BLE001
        print(f"  (notify skipped: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
