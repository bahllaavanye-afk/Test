"""
Daily P&L Attribution Report
Runs after market close and posts a strategy-level Slack summary to #pnl-daily.

Data source: Alpaca paper account closed orders for today.
"""
from __future__ import annotations
import os
import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

try:
    import requests
except ImportError:
    sys.exit("Missing: pip install requests")

# ── Config ─────────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
SLACK_BOT_TOKEN   = os.environ.get("SLACK_BOT_TOKEN", "")
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DAYS              = int(os.environ.get("DAYS", "1"))

SLACK_CHANNEL = "#pnl-daily"


def _alpaca(path: str, params: dict | None = None) -> dict | list:
    resp = requests.get(
        f"{ALPACA_BASE_URL}{path}",
        headers={
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        },
        params=params or {},
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"Alpaca {path} → {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return {}
    return resp.json()


def _post_slack(text: str, blocks: list | None = None) -> None:
    if not SLACK_BOT_TOKEN:
        print(f"[SLACK (dry-run)]\n{text}")
        return
    payload: dict = {"channel": SLACK_CHANNEL, "text": text}
    if blocks:
        payload["blocks"] = blocks
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )
    data = resp.json()
    if not data.get("ok"):
        print(f"Slack error: {data.get('error')}", file=sys.stderr)


def main() -> None:
    if not ALPACA_API_KEY:
        print("ALPACA_API_KEY not set — skipping report", file=sys.stderr)
        return

    # ── Fetch account ──────────────────────────────────────────────────────────
    acct = _alpaca("/v2/account")
    equity       = float(acct.get("equity", 0))
    last_equity  = float(acct.get("last_equity", 0))
    day_pnl      = equity - last_equity
    day_pnl_pct  = day_pnl / last_equity * 100 if last_equity > 0 else 0.0
    cash         = float(acct.get("cash", 0))
    buying_power = float(acct.get("buying_power", 0))

    # ── Fetch today's closed orders ────────────────────────────────────────────
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat()
    orders_raw = _alpaca("/v2/orders", {
        "status": "closed", "after": since, "limit": 500, "direction": "desc",
    })
    orders: list[dict] = orders_raw if isinstance(orders_raw, list) else []

    # Group by symbol → compute realized P&L (approximate: qty × (fill − avg_cost))
    # Since Alpaca doesn't return per-order P&L directly, we sum fill values
    symbol_stats: dict[str, dict] = defaultdict(lambda: {"orders": 0, "side_vol": 0.0})
    filled = [o for o in orders if o.get("status") == "filled"]

    for o in filled:
        sym = o.get("symbol", "?")
        side = o.get("side", "")
        qty = float(o.get("filled_qty") or 0)
        fill = float(o.get("filled_avg_price") or 0)
        dollar_vol = qty * fill
        symbol_stats[sym]["orders"] += 1
        symbol_stats[sym]["side_vol"] += dollar_vol if side == "buy" else -dollar_vol

    # Sort by absolute notional
    ranked = sorted(symbol_stats.items(), key=lambda kv: abs(kv[1]["side_vol"]), reverse=True)

    # ── Build Slack message ────────────────────────────────────────────────────
    now_et = datetime.now(timezone.utc) - timedelta(hours=4)  # approx ET
    date_str = now_et.strftime("%a %b %-d %Y")
    pnl_emoji = "🟢" if day_pnl >= 0 else "🔴"
    sign = "+" if day_pnl >= 0 else ""

    header = (
        f"{pnl_emoji} *Daily P&L Report — {date_str}*\n"
        f"Account equity: *${equity:,.0f}* | "
        f"Day P&L: *{sign}${day_pnl:,.2f} ({sign}{day_pnl_pct:.2f}%)* | "
        f"Cash: ${cash:,.0f}"
    )

    lines = [f"*Filled orders today ({len(filled)}):*"]
    for sym, stats in ranked[:15]:
        vol = stats["side_vol"]
        side_label = "NET LONG" if vol > 0 else "NET SHORT"
        lines.append(f"  • `{sym}` — {stats['orders']} orders, notional ${abs(vol):,.0f} ({side_label})")
    if not ranked:
        lines.append("  _No filled orders today_")

    body = "\n".join(lines)

    footer = (
        f"_Paper trading | Buying power: ${buying_power:,.0f} | "
        f"Lookback: {DAYS}d | QuantEdge_"
    )

    full_text = f"{header}\n{body}\n{footer}"
    _post_slack(full_text)

    # ── Save JSON artifact ─────────────────────────────────────────────────────
    report = {
        "date": date_str,
        "equity": equity,
        "day_pnl": round(day_pnl, 2),
        "day_pnl_pct": round(day_pnl_pct, 4),
        "filled_orders": len(filled),
        "top_symbols": [{"symbol": k, "net_notional": round(v["side_vol"], 2)} for k, v in ranked[:15]],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path("experiments/results") / f"pnl_report_{now_et.strftime('%Y%m%d')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out_path}")
    print(f"Day P&L: {sign}${day_pnl:,.2f} ({sign}{day_pnl_pct:.2f}%)")


if __name__ == "__main__":
    main()
