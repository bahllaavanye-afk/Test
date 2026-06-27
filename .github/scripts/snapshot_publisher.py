#!/usr/bin/env python3
"""Static snapshot publisher — the no-backend alternative.

The customer-facing site is dark whenever the Render backend is down (free
build-minute quota). But the data the site needs already exists: Alpaca paper
P&L/positions, the strategy library, the desk roster, brain health. This writes
all of it to ``frontend/public/data/snapshot.json``, which Vercel serves
statically at ``/data/snapshot.json`` — so the frontend can show REAL numbers
with no backend at all.

Every section is best-effort: missing keys or network just omit that section,
the rest still publishes. Designed to run on a schedule and commit the JSON.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "frontend" / "public" / "data" / "snapshot.json"
ALPACA_BASE = "https://paper-api.alpaca.markets"


def _alpaca_get(path: str, params: dict | None = None):
    key = os.environ.get("ALPACA_API_KEY", "") or os.environ.get("ALPACA_API_KEY_1", "")
    sec = os.environ.get("ALPACA_SECRET_KEY", "") or os.environ.get("ALPACA_SECRET_KEY_1", "")
    if not (key and sec):
        return None
    url = ALPACA_BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        print(f"alpaca {path} failed: {e}", file=sys.stderr)
        return None


def repo_facts() -> dict:
    manual = glob.glob(str(REPO / "backend/app/strategies/manual/*.py"))
    ml = glob.glob(str(REPO / "backend/app/strategies/ml_enhanced/*.py"))
    strategies = sorted(
        Path(f).stem for f in manual + ml if "__init__" not in f
    )
    models = [Path(f).stem for f in glob.glob(str(REPO / "backend/app/ml/models/*.py"))
              if "__init__" not in f and "base" not in f]
    return {
        "strategy_count": len(strategies),
        "strategies": strategies,
        "model_count": len(models),
        "models": models,
    }


def desk_roster() -> list[dict]:
    """Parse desk names + symbols from desk_order_placer without importing it."""
    desks = []
    try:
        src = (REPO / ".github/scripts/desk_order_placer.py").read_text()
        import re
        for m in re.finditer(r'name="([^"]+)".*?symbols=\[([^\]]*)\]', src, re.S):
            name = m.group(1)
            syms = re.findall(r'"([^"]+)"', m.group(2))
            desks.append({"name": name, "symbols": syms, "count": len(syms)})
    except Exception as e:  # noqa: BLE001
        print(f"desk parse failed: {e}", file=sys.stderr)
    return desks


def brain_health() -> dict:
    try:
        sys.path.insert(0, str(REPO / ".github/scripts"))
        import llm_common  # type: ignore
        st = llm_common.cascade_status(probe=True)
        return {"healthy": st.get("healthy"), "working": st.get("working", [])}
    except Exception as e:  # noqa: BLE001
        return {"healthy": None, "working": [], "note": str(e)[:80]}


def trading_snapshot() -> dict:
    acct = _alpaca_get("/v2/account")
    if not acct:
        return {"available": False, "reason": "no Alpaca keys or unreachable"}
    positions = _alpaca_get("/v2/positions") or []
    orders = _alpaca_get("/v2/orders", {"status": "all", "limit": 20}) or []
    clock = _alpaca_get("/v2/clock") or {}
    equity = float(acct.get("equity", 0) or 0)
    last_equity = float(acct.get("last_equity", equity) or equity)
    return {
        "available": True,
        "equity": round(equity, 2),
        "day_pnl": round(equity - last_equity, 2),
        "day_pnl_pct": round((equity - last_equity) / last_equity * 100, 3) if last_equity else 0,
        "cash": round(float(acct.get("cash", 0) or 0), 2),
        "market_open": bool(clock.get("is_open", False)),
        "open_positions": [
            {"symbol": p.get("symbol"), "qty": p.get("qty"),
             "market_value": round(float(p.get("market_value", 0) or 0), 2),
             "unrealized_pl": round(float(p.get("unrealized_pl", 0) or 0), 2)}
            for p in positions
        ],
        "recent_orders": [
            {"symbol": o.get("symbol"), "side": o.get("side"), "qty": o.get("qty"),
             "status": o.get("status"), "submitted_at": o.get("submitted_at")}
            for o in orders[:10]
        ],
    }


def main() -> int:
    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_ts": int(time.time()),
        "platform": "QuantEdge",
        "trading_mode": os.environ.get("TRADING_MODE", "paper"),
        "repo": repo_facts(),
        "desks": desk_roster(),
        "brain": brain_health(),
        "trading": trading_snapshot(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snap, indent=2))
    t = snap["trading"]
    print(f"✓ wrote {OUT.relative_to(REPO)} | strategies={snap['repo']['strategy_count']} "
          f"desks={len(snap['desks'])} brain={snap['brain'].get('working')} "
          f"trading={'$'+str(t.get('equity')) if t.get('available') else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
