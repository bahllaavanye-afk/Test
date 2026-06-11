"""
Fill Tracker — checks Alpaca paper account for filled orders and computes PnL.

Strategy names are encoded in client_order_id (format: qe-{strategy[:10]}-{sym[:4]}-{ts})
by desk_order_placer.py so fills can be attributed back to strategies.

Flow:
  1. Fetch all filled orders from last 7 days via Alpaca REST
  2. Parse strategy name from client_order_id
  3. For fills >= 24h old: fetch next-day price, compute win/loss + return
  4. Write cumulative stats to backend/performance_log/strategy_performance.json
  5. Post report to Slack #pnl-daily

Run daily at 22:00 UTC via fill-tracking.yml workflow.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
SLACK_BOT_TOKEN   = os.environ.get("SLACK_BOT_TOKEN", "")
TRADING_MODE      = os.environ.get("TRADING_MODE", "paper")
ALLOW_PAID        = os.environ.get("ALLOW_PAID_APIS", "False")

REPO_ROOT    = Path(__file__).parent.parent.parent
OUTPUT_FILE  = REPO_ROOT / "backend" / "performance_log" / "strategy_performance.json"

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE  = "https://data.alpaca.markets"

# Hard safety blocks
if ALLOW_PAID.lower() == "true":
    print("ALLOW_PAID_APIS=True is blocked in fill_tracker.py")
    sys.exit(1)
if TRADING_MODE == "live":
    print("TRADING_MODE=live is blocked in fill_tracker.py")
    sys.exit(1)


# ── Alpaca helpers ─────────────────────────────────────────────────────────────

def _alpaca_get(path: str, params: dict | None = None, data_api: bool = False) -> dict | list:
    base = ALPACA_DATA_BASE if data_api else ALPACA_PAPER_BASE
    url  = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    })
    with urllib.request.urlopen(req, timeout=12) as resp:
        return json.loads(resp.read())


def _post_slack(channel: str, text: str) -> None:
    if not SLACK_BOT_TOKEN:
        return
    try:
        payload = json.dumps({"channel": channel, "text": text}).encode()
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload,
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        print(f"  ⚠ Slack post failed: {e}", flush=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_strategy(client_order_id: str | None) -> str | None:
    """Extract strategy name from qe-{strategy[:10]}-{sym[:4]}-{ts} format."""
    if not client_order_id or not client_order_id.startswith("qe-"):
        return None
    parts = client_order_id.split("-")
    # parts: ["qe", strategy_name, symbol, timestamp]
    return parts[1] if len(parts) >= 3 else None


def _next_day_price(symbol: str, fill_dt: datetime) -> float | None:
    """Return the close price of `symbol` on the day after fill_dt."""
    next_dt = fill_dt + timedelta(days=1)
    start   = (next_dt - timedelta(days=3)).strftime("%Y-%m-%d")
    end_    = (next_dt + timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        if "/" in symbol:
            data = _alpaca_get(
                "/v1beta3/crypto/us/bars",
                {"symbols": symbol, "timeframe": "1Day", "start": start, "end": end_},
                data_api=True,
            )
            bars = data.get("bars", {}).get(symbol, [])
        else:
            data = _alpaca_get(
                f"/v2/stocks/{symbol}/bars",
                {"timeframe": "1Day", "start": start, "end": end_, "adjustment": "split"},
                data_api=True,
            )
            bars = data.get("bars", [])
        if not bars:
            return None
        # Find the bar closest to (but not before) fill_dt + 1 day
        target_date = next_dt.strftime("%Y-%m-%d")
        for bar in sorted(bars, key=lambda b: b["t"]):
            if bar["t"][:10] >= target_date:
                return float(bar["c"])
        return float(bars[-1]["c"])
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"QuantEdge Fill Tracker — {datetime.now(timezone.utc).isoformat()}", flush=True)

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("⚠ ALPACA credentials not set — skipping (no real orders to track)", flush=True)
        return

    # Fetch closed orders from last 7 days
    since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        orders = _alpaca_get("/v2/orders", {
            "status": "closed", "after": since,
            "limit": "500", "direction": "desc",
        })
    except Exception as e:
        print(f"✗ failed to fetch orders: {e}", flush=True)
        return

    if not isinstance(orders, list):
        print(f"✗ unexpected orders response type: {type(orders)}", flush=True)
        return

    # Only process filled orders >= 24h old (need next-day price)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    filled_orders = [
        o for o in orders
        if o.get("status") == "filled"
        and o.get("filled_at")
        and o.get("client_order_id", "").startswith("qe-")
        and datetime.fromisoformat(o["filled_at"].replace("Z", "+00:00")) < cutoff
    ]
    print(f"✓ {len(filled_orders)} QuantEdge-tagged filled orders >= 24h old", flush=True)

    # Load existing cumulative performance
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing_perf: dict = {}
    already_tracked: set = set()
    if OUTPUT_FILE.exists():
        try:
            saved = json.loads(OUTPUT_FILE.read_text())
            existing_perf = saved.get("strategies", {})
            already_tracked = set(saved.get("tracked_order_ids", []))
        except Exception:
            pass

    new_wins:    dict[str, list[bool]]  = {}
    new_returns: dict[str, list[float]] = {}
    new_tracked: list[str]              = []

    for order in filled_orders:
        oid  = order.get("id", "")
        coid = order.get("client_order_id", "")
        if oid in already_tracked:
            continue

        strategy_name = _parse_strategy(coid)
        if not strategy_name:
            continue

        symbol = order.get("symbol", "")
        side   = order.get("side", "")
        fill_price_str = order.get("filled_avg_price")
        filled_at_str  = order.get("filled_at", "")

        if not fill_price_str or not filled_at_str:
            continue

        fill_price = float(fill_price_str)
        fill_dt    = datetime.fromisoformat(filled_at_str.replace("Z", "+00:00"))
        next_price = _next_day_price(symbol, fill_dt)

        if next_price is None or fill_price <= 0:
            print(f"  ⚠ {strategy_name}/{symbol}: no next-day price available", flush=True)
            continue

        # Return from fill price to next-day close
        ret_pct = (next_price - fill_price) / fill_price * 100.0
        if side == "sell":
            ret_pct = -ret_pct  # short positions profit when price falls

        is_win = ret_pct > 0
        new_wins.setdefault(strategy_name, []).append(is_win)
        new_returns.setdefault(strategy_name, []).append(ret_pct)
        new_tracked.append(oid)
        print(
            f"  · {strategy_name}/{symbol} {side} fill={fill_price:.4f} → next={next_price:.4f} "
            f"ret={ret_pct:+.2f}% {'✓' if is_win else '✗'}",
            flush=True,
        )

    # Merge into cumulative performance
    now_str = datetime.now(timezone.utc).isoformat()
    all_strategies = set(list(new_wins.keys()) + list(existing_perf.keys()))
    updated_perf: dict = {}
    for sname in all_strategies:
        prev = existing_perf.get(sname, {"trades": 0, "wins": 0, "total_return_pct": 0.0})
        added_wins    = new_wins.get(sname, [])
        added_returns = new_returns.get(sname, [])

        total_trades = prev.get("trades", 0) + len(added_wins)
        total_wins   = prev.get("wins", 0)   + sum(added_wins)
        total_ret    = prev.get("total_return_pct", 0.0) + sum(added_returns)

        win_rate = total_wins / total_trades if total_trades > 0 else 0.0
        avg_ret  = total_ret  / total_trades if total_trades > 0 else 0.0
        updated_perf[sname] = {
            "trades":           total_trades,
            "wins":             total_wins,
            "win_rate":         round(win_rate, 4),
            "avg_return_pct":   round(avg_ret, 4),
            "total_return_pct": round(total_ret, 4),
            "last_updated":     now_str,
        }

    all_tracked = sorted(already_tracked | set(new_tracked))

    output = {
        "generated_at":      now_str,
        "period_days":       30,
        "strategies":        updated_perf,
        "tracked_order_ids": all_tracked[-2000:],  # cap at 2000 to prevent unbounded growth
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Saved performance data: {len(updated_perf)} strategies, {len(new_tracked)} new fills", flush=True)

    # Slack summary
    if SLACK_BOT_TOKEN and (new_wins or updated_perf):
        qualified = {k: v for k, v in updated_perf.items() if v["trades"] >= 3}
        if qualified:
            by_wr = sorted(qualified.items(), key=lambda x: x[1]["win_rate"], reverse=True)
            lines = [f"*📊 Daily Fill Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*"]
            lines.append(f"New fills processed: *{len(new_tracked)}*  |  Strategies tracked: *{len(qualified)}*\n")

            top = [x for x in by_wr[:5] if x[1]["trades"] >= 3]
            bot = [x for x in by_wr[-5:] if x[1]["trades"] >= 3]
            if top:
                lines.append("*Top performers:*")
                for n, d in top:
                    lines.append(f"  ✅ `{n}`: {d['win_rate']:.0%} win ({d['trades']} trades, avg {d['avg_return_pct']:+.2f}%)")
            if bot and bot != top:
                lines.append("*Underperformers:*")
                for n, d in bot:
                    lines.append(f"  ⚠️ `{n}`: {d['win_rate']:.0%} win ({d['trades']} trades, avg {d['avg_return_pct']:+.2f}%)")

            _post_slack("#pnl-daily", "\n".join(lines))

    print("Fill tracker complete.", flush=True)


if __name__ == "__main__":
    main()
