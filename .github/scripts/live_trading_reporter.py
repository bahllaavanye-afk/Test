"""
QuantEdge Live Trading Reporter
=================================
Reads the REAL Alpaca paper account (positions, orders, P&L, clock)
and posts live status to Slack channels so every employee sees real numbers.

Runs every 30 minutes via live-trading-reporter.yml.
Posts to:
  #pnl-daily       — account P&L, positions
  #desk-equities   — equity positions
  #desk-crypto     — crypto positions
  #engineering     — system health (order fill rate, latency)
  #risk-alerts     — any breach of drawdown/concentration limits
  #allquantedge    — concise exec summary

SECURITY: TRADING_MODE must be 'paper'. Live trading is blocked.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing: pip install requests")

# ── Constants ──────────────────────────────────────────────────────────────────
ALPACA_KEY     = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET  = os.environ.get("ALPACA_SECRET_KEY", "")
SLACK_TOKEN    = os.environ.get("SLACK_BOT_TOKEN", "")
TRADING_MODE   = os.environ.get("TRADING_MODE", "paper").lower()
ALPACA_BASE    = "https://paper-api.alpaca.markets"
DATA_BASE      = "https://data.alpaca.markets"

# Hard block: never run against live account from CI
if TRADING_MODE == "live":
    print("BLOCKED: TRADING_MODE=live is not allowed in CI. Set TRADING_MODE=paper.", file=sys.stderr)
    sys.exit(1)

# Drawdown limit: if equity drops more than this % from high-water mark, post risk alert
MAX_DRAWDOWN_PCT = 5.0
# Position concentration limit: no single position > this % of portfolio
MAX_POSITION_PCT = 20.0


# ── Alpaca helpers ─────────────────────────────────────────────────────────────
def _hdr() -> dict:
    return {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }


def get(path: str, params: dict | None = None, base: str = ALPACA_BASE) -> dict | list:
    if not ALPACA_KEY:
        return {}
    try:
        r = requests.get(f"{base}{path}", headers=_hdr(), params=params or {}, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"  [alpaca] {path} → HTTP {r.status_code}: {r.text[:120]}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"  [alpaca] {path} error: {e}", file=sys.stderr)
        return {}


# ── Slack helpers ──────────────────────────────────────────────────────────────
def post_slack(channel: str, text: str, username: str = "Live Trading Bot",
               emoji: str = ":chart_with_upwards_trend:") -> bool:
    if not SLACK_TOKEN:
        print(f"[SLACK:{channel}]\n{text}\n")
        return True
    r = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
        json={"channel": channel, "text": text, "username": username, "icon_emoji": emoji},
        timeout=10,
    )
    data = r.json()
    if not data.get("ok"):
        print(f"  [slack] #{channel}: {data.get('error')}", file=sys.stderr)
        return False
    return True


def _pct_color(pct: float) -> str:
    """Return emoji based on P&L percentage."""
    if pct > 1.0:  return ":green_heart:"
    if pct > 0:    return ":white_check_mark:"
    if pct > -1.0: return ":yellow_heart:"
    return ":red_circle:"


def _side_arrow(side: str) -> str:
    return ":arrow_up:" if side.lower() == "long" else ":arrow_down:"


# ── Fetch real account data ────────────────────────────────────────────────────
def fetch_all() -> dict:
    account   = get("/v2/account")
    positions = get("/v2/positions") or []
    clock     = get("/v2/clock")
    # Last 50 filled orders from today
    since = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    orders = get("/v2/orders", {"status": "all", "after": since, "limit": "50"}) or []
    # Portfolio history (1 day, 1-min bars for equity curve)
    port_hist = get("/v2/account/portfolio/history",
                    {"period": "1D", "timeframe": "5Min", "extended_hours": "true"}) or {}

    return {
        "account":   account,
        "positions": positions if isinstance(positions, list) else [],
        "clock":     clock,
        "orders":    orders if isinstance(orders, list) else [],
        "port_hist": port_hist,
    }


# ── Build reports ──────────────────────────────────────────────────────────────
def report_pnl(data: dict) -> str:
    a = data["account"]
    if not a:
        return ":warning: Could not fetch Alpaca account — check API keys"

    equity      = float(a.get("equity", 0))
    last_eq     = float(a.get("last_equity", equity))
    day_pnl     = equity - last_eq
    day_pnl_pct = day_pnl / last_eq * 100 if last_eq else 0.0
    cash        = float(a.get("cash", 0))
    bp          = float(a.get("buying_power", 0))
    positions   = data["positions"]
    orders      = data["orders"]
    clock       = data["clock"]

    filled   = [o for o in orders if o.get("status") == "filled"]
    canceled = [o for o in orders if o.get("status") == "canceled"]
    pending  = [o for o in orders if o.get("status") in ("new", "partially_filled")]

    is_open   = clock.get("is_open", False) if isinstance(clock, dict) else False
    mkt_time  = clock.get("timestamp", "")[:16] if isinstance(clock, dict) else ""
    mkt_badge = ":green_circle: OPEN" if is_open else ":red_circle: CLOSED"

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    color   = _pct_color(day_pnl_pct)
    sign    = "+" if day_pnl >= 0 else ""

    lines = [
        f"*:money_with_wings: Live P&L Report* | {now_str}",
        f"Market: {mkt_badge} | {mkt_time}",
        "",
        f"{color} *Day P&L:* {sign}${day_pnl:,.2f} ({sign}{day_pnl_pct:.2f}%)",
        f"  Portfolio equity: *${equity:,.2f}*",
        f"  Cash: ${cash:,.2f}   Buying power: ${bp:,.2f}",
        "",
        f"*Orders today:* {len(filled)} filled · {len(pending)} pending · {len(canceled)} canceled",
        f"*Open positions:* {len(positions)}",
    ]

    if positions:
        lines.append("")
        lines.append("*Open positions:*")
        for pos in sorted(positions, key=lambda p: abs(float(p.get("unrealized_pl", 0))), reverse=True)[:8]:
            sym   = pos.get("symbol", "?")
            side  = pos.get("side", "long")
            qty   = float(pos.get("qty", 0))
            mktv  = float(pos.get("market_value", 0))
            upl   = float(pos.get("unrealized_pl", 0))
            uplpct= float(pos.get("unrealized_plpc", 0)) * 100
            arr   = _side_arrow(side)
            ucolor= ":green_heart:" if upl >= 0 else ":red_circle:"
            lines.append(
                f"  {arr} *{sym}* {qty:g} @ ${mktv:,.0f} | "
                f"UPL: {ucolor} {'+' if upl>=0 else ''}{upl:.2f} ({'+' if uplpct>=0 else ''}{uplpct:.1f}%)"
            )

    if filled:
        lines.append("")
        lines.append(f"*Recent fills ({min(len(filled),5)}/{len(filled)}):*")
        for o in filled[-5:]:
            sym  = o.get("symbol", "?")
            side = o.get("side", "?")
            qty  = o.get("filled_qty", "?")
            price= o.get("filled_avg_price", "?")
            arr  = ":arrow_up:" if side == "buy" else ":arrow_down:"
            lines.append(f"  {arr} {sym} {side} {qty} @ ${price}")

    return "\n".join(lines)


def report_risk(data: dict) -> str | None:
    """Returns a risk alert string if any limit is breached, else None."""
    a         = data["account"]
    positions = data["positions"]
    if not a or not positions:
        return None

    equity    = float(a.get("equity", 1))
    last_eq   = float(a.get("last_equity", equity))
    day_pnl_pct = (equity - last_eq) / last_eq * 100 if last_eq else 0.0

    alerts = []

    # Drawdown check
    if day_pnl_pct < -MAX_DRAWDOWN_PCT:
        alerts.append(
            f":rotating_light: *DRAWDOWN ALERT* — Day P&L {day_pnl_pct:.1f}% "
            f"(limit: -{MAX_DRAWDOWN_PCT}%). Consider reducing exposure."
        )

    # Concentration check
    for pos in positions:
        mktv = abs(float(pos.get("market_value", 0)))
        conc = mktv / equity * 100 if equity else 0
        if conc > MAX_POSITION_PCT:
            sym = pos.get("symbol", "?")
            alerts.append(
                f":warning: *CONCENTRATION* — {sym} is {conc:.1f}% of portfolio "
                f"(limit: {MAX_POSITION_PCT}%). Reduce or hedge."
            )

    if not alerts:
        return None

    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    return f":shield: *Risk Alert — {now_str}*\n" + "\n".join(alerts)


def report_equity_desk(data: dict) -> str:
    """Equity-specific positions for #desk-equities."""
    positions = [p for p in data["positions"] if "/" not in p.get("symbol", "")]
    if not positions:
        return ""
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [f"*:chart_with_upwards_trend: Equities Desk — {now_str}* | {len(positions)} positions"]
    for pos in sorted(positions, key=lambda p: abs(float(p.get("market_value", 0))), reverse=True)[:10]:
        sym    = pos.get("symbol", "?")
        side   = pos.get("side", "long")
        qty    = float(pos.get("qty", 0))
        entry  = float(pos.get("avg_entry_price", 0))
        curr   = float(pos.get("current_price", entry))
        upl    = float(pos.get("unrealized_pl", 0))
        uplpct = float(pos.get("unrealized_plpc", 0)) * 100
        arrow  = ":arrow_up:" if side == "long" else ":arrow_down:"
        ucolor = ":green_heart:" if upl >= 0 else ":red_circle:"
        lines.append(
            f"  {arrow} *{sym}* ×{qty:g} | entry ${entry:.2f} → ${curr:.2f} | "
            f"UPL {ucolor} {'+' if upl>=0 else ''}{upl:.2f} ({'+' if uplpct>=0 else ''}{uplpct:.1f}%)"
        )
    return "\n".join(lines)


def report_crypto_desk(data: dict) -> str:
    """Crypto positions for #desk-crypto."""
    positions = [p for p in data["positions"] if "/" in p.get("symbol", "")]
    if not positions:
        return ""
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [f"*:coin: Crypto Desk — {now_str}* | {len(positions)} positions"]
    for pos in sorted(positions, key=lambda p: abs(float(p.get("market_value", 0))), reverse=True)[:8]:
        sym    = pos.get("symbol", "?")
        qty    = float(pos.get("qty", 0))
        entry  = float(pos.get("avg_entry_price", 0))
        curr   = float(pos.get("current_price", entry))
        upl    = float(pos.get("unrealized_pl", 0))
        uplpct = float(pos.get("unrealized_plpc", 0)) * 100
        ucolor = ":green_heart:" if upl >= 0 else ":red_circle:"
        lines.append(
            f"  :coin: *{sym}* ×{qty:.4f} | ${entry:.2f} → ${curr:.2f} | "
            f"UPL {ucolor} {'+' if upl>=0 else ''}{upl:.2f} ({'+' if uplpct>=0 else ''}{uplpct:.1f}%)"
        )
    return "\n".join(lines)


def report_exec_summary(data: dict) -> str:
    """Concise executive summary for #allquantedge."""
    a       = data["account"]
    orders  = data["orders"]
    clock   = data["clock"]
    if not a:
        return ""

    equity     = float(a.get("equity", 0))
    last_eq    = float(a.get("last_equity", equity))
    day_pnl    = equity - last_eq
    day_pnl_pct= day_pnl / last_eq * 100 if last_eq else 0.0
    n_pos      = len(data["positions"])
    filled     = len([o for o in orders if o.get("status") == "filled"])
    is_open    = clock.get("is_open", False) if isinstance(clock, dict) else False
    mkt_badge  = ":green_circle:" if is_open else ":red_circle:"

    color  = _pct_color(day_pnl_pct)
    sign   = "+" if day_pnl >= 0 else ""
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    mode   = "PAPER"

    return (
        f"*QuantEdge {mode} — {now_str}* {mkt_badge}\n"
        f"{color} Day P&L: {sign}${day_pnl:,.2f} ({sign}{day_pnl_pct:.2f}%)\n"
        f"Equity: *${equity:,.2f}* | Positions: {n_pos} | Fills today: {filled}"
    )


def report_system_health(data: dict) -> str:
    """Order fill-rate and API health for #engineering."""
    orders  = data["orders"]
    account = data["account"]
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")

    total    = len(orders)
    filled   = len([o for o in orders if o.get("status") == "filled"])
    rejected = len([o for o in orders if o.get("status") in ("rejected", "expired")])
    fill_rate = filled / total * 100 if total > 0 else 0.0
    account_ok = bool(account and account.get("status") == "ACTIVE")

    health_emoji = ":white_check_mark:" if account_ok and fill_rate > 80 else ":warning:"

    return (
        f"{health_emoji} *System Health — {now_str}*\n"
        f"  Alpaca account: {'✅ ACTIVE (paper)' if account_ok else '⚠️ NOT ACTIVE'}\n"
        f"  Orders today: {total} total | {filled} filled | {rejected} rejected\n"
        f"  Fill rate: {fill_rate:.0f}%\n"
        f"  Mode: *PAPER TRADING* (live blocked in CI)"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    if not ALPACA_KEY:
        print("ALPACA_API_KEY not set — skipping live reporter", file=sys.stderr)
        return 0

    print(f"📊 Live Trading Reporter | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Mode: {TRADING_MODE.upper()}")

    data = fetch_all()
    account = data.get("account", {})
    if not account:
        print("⚠️ Could not reach Alpaca API — check secrets")
        if SLACK_TOKEN:
            post_slack("engineering",
                       ":warning: *Live Reporter*: Could not reach Alpaca paper API — check ALPACA_API_KEY / ALPACA_SECRET_KEY secrets",
                       username="System Monitor", emoji=":warning:")
        return 1

    equity = float(account.get("equity", 0))
    print(f"   Account equity: ${equity:,.2f}")
    print(f"   Positions: {len(data['positions'])}")
    print(f"   Orders today: {len(data['orders'])}")

    posts = []

    # 1. P&L report → #pnl-daily
    pnl_msg = report_pnl(data)
    if pnl_msg:
        posts.append(("pnl-daily", pnl_msg, "P&L Bot", ":money_with_wings:"))

    # 2. Equity desk → #desk-equities
    eq_msg = report_equity_desk(data)
    if eq_msg:
        posts.append(("desk-equities", eq_msg, "Equities Desk Bot", ":chart_with_upwards_trend:"))

    # 3. Crypto desk → #desk-crypto
    cr_msg = report_crypto_desk(data)
    if cr_msg:
        posts.append(("desk-crypto", cr_msg, "Crypto Desk Bot", ":coin:"))

    # 4. Risk alerts → #risk-alerts (only if breach)
    risk_msg = report_risk(data)
    if risk_msg:
        posts.append(("risk-alerts", risk_msg, "Risk Monitor", ":shield:"))
        print("  ⚠️  Risk alert detected")

    # 5. System health → #engineering
    health_msg = report_system_health(data)
    if health_msg:
        posts.append(("engineering", health_msg, "System Monitor", ":gear:"))

    # 6. Exec summary → #allquantedge
    exec_msg = report_exec_summary(data)
    if exec_msg:
        posts.append(("allquantedge", exec_msg, "Trading Desk", ":bar_chart:"))

    # Post all
    posted = 0
    for channel, text, username, emoji in posts:
        ok = post_slack(channel, text, username=username, emoji=emoji)
        if ok:
            posted += 1
            print(f"  ✓ #{channel}")
        else:
            print(f"  ✗ #{channel}")
        time.sleep(0.5)

    print(f"\n✅ {posted}/{len(posts)} reports posted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
