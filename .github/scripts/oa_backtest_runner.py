"""Backtest every OA clone + factory variant on real underlying history.

Pulls 2y of daily closes from Alpaca for each template's symbol, scores every
options template with the synthetic-BS backtester, posts a ranked table to
Discord #alpha-research, and writes backend/performance_log/oa_backtests.json
(committed by the workflow) so the lifecycle manager and humans share one
ranking. New clones (from OA Scout issues or screenshots) are picked up
automatically — anything oa_*/gen_* in BOT_TEMPLATES gets scored.

Caveat rides every output: synthetic-BS ranks templates; it does not promise
returns (no skew, no bid/ask).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

KEY, SECRET = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", "")
OUT = REPO / "backend" / "performance_log" / "oa_backtests.json"


def daily_closes(symbol: str, days: int = 730) -> list[float]:
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = urllib.parse.urlencode({"timeframe": "1Day", "limit": 1000,
                                     "adjustment": "split", "start": start})
    req = urllib.request.Request(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?{params}",
        headers={"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET,
                 "User-Agent": "QuantEdge-OA-Backtests/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        bars = json.loads(r.read()).get("bars") or []
    return [float(b["c"]) for b in bars]


def main() -> int:
    if not (KEY and SECRET):
        print("ALPACA keys absent — cannot backtest (honest skip)")
        return 0
    from app.backtest.options_synth import backtest_template
    from app.bots.templates import BOT_TEMPLATES

    targets = {tid: t for tid, t in BOT_TEMPLATES.items()
               if tid.startswith(("oa_", "gen_")) and t.get("market_type") == "options"}
    closes_cache: dict[str, list[float]] = {}
    results = []
    for tid, t in sorted(targets.items()):
        sym = t.get("symbol", "SPY")
        if sym not in closes_cache:
            try:
                closes_cache[sym] = daily_closes(sym)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ bars for {sym} failed: {exc}")
                closes_cache[sym] = []
        closes = closes_cache[sym]
        if len(closes) < 60:
            print(f"  ⚠ {tid}: insufficient history — skipped")
            continue
        r = backtest_template(t, closes)
        r.update({"template": tid, "name": t["name"], "symbol": sym})
        results.append(r)
        print(f"  {tid:42s} trades={r['trades']:>3} win={r['win_rate']} pnl=${r['total_pnl']:>10}")

    results.sort(key=lambda r: r["total_pnl"], reverse=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "as_of": datetime.now(timezone.utc).isoformat(),
        "method": "synthetic-BS (HV20×1.1) over 2y daily bars — ranking only",
        "results": results,
    }, indent=1))

    if results:
        top = results[:5]
        bottom = results[-3:]
        lines = ["📐 **OA clone backtests** (synthetic-BS, 2y — ranking, not a promise)",
                 "**Top:**"]
        lines += [f"• {r['name'][:40]} — ${r['total_pnl']:,.0f}, win {r['win_rate']:.0%}, {r['trades']} trades"
                  for r in top if r["win_rate"] is not None]
        lines.append("**Bottom:**")
        lines += [f"• {r['name'][:40]} — ${r['total_pnl']:,.0f}" for r in bottom]
        try:
            from notify import discord_post

            discord_post("alpha-research", "\n".join(lines), username="QuantEdge Backtest Desk")
        except Exception as exc:  # noqa: BLE001
            print(f"(Discord skipped: {exc})")
    print(f"\nScored {len(results)} templates → {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
