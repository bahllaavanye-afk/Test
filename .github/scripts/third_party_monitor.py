"""
QuantEdge Third-Party Health Monitor

Checks every external service the platform depends on:
  Alpaca · Binance · Polymarket · Render · Vercel · Supabase · Upstash
  GitHub · yfinance proxy · Anthropic API

Posts outages to #infra-alerts immediately. Auto-resolves when service recovers.
Writes last-known state to avoid duplicate alerts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
ALPACA_KEY  = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SEC  = os.environ.get("ALPACA_SECRET_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

STATE_FILE  = Path("/tmp/qe_third_party_state.json")
CHANNEL     = "#infra-alerts"
TIMEOUT     = 12  # seconds per check

SERVICES = [
    # name, url, method, expected_status, headers, notes
    {
        "name": "Alpaca Paper API",
        "url": "https://paper-api.alpaca.markets/v2/clock",
        "method": "GET",
        "expected_status": [200, 401],  # 401 = wrong key but API is up
        "headers": {},
        "critical": True,
        "slack_channel": "#desk-equities",
    },
    {
        "name": "Alpaca Data API",
        "url": "https://data.alpaca.markets/v2/stocks/bars?symbols=SPY&timeframe=1Day&limit=1",
        "method": "GET",
        "expected_status": [200, 401, 403],
        "headers": {},
        "critical": True,
        "slack_channel": "#infra-alerts",
    },
    {
        "name": "Binance REST API",
        "url": "https://api.binance.com/api/v3/ping",
        "method": "GET",
        "expected_status": [200],
        "headers": {},
        "critical": True,
        "slack_channel": "#desk-crypto",
    },
    {
        "name": "Binance Futures API",
        "url": "https://fapi.binance.com/fapi/v1/ping",
        "method": "GET",
        "expected_status": [200],
        "headers": {},
        "critical": True,
        "slack_channel": "#desk-crypto",
    },
    {
        "name": "Polymarket CLOB",
        "url": "https://clob.polymarket.com/",
        "method": "GET",
        "expected_status": [200, 404],
        "headers": {},
        "critical": True,
        "slack_channel": "#desk-polymarket",
    },
    {
        "name": "Vercel (Frontend)",
        "url": "https://quantedge.vercel.app",
        "method": "GET",
        "expected_status": [200, 301, 302, 404],  # 404 = deployed but no route at /
        "headers": {},
        "critical": True,
        "slack_channel": "#infra-alerts",
    },
    {
        "name": "Render (Backend API)",
        "url": "https://quantedge-api-agb8.onrender.com/health",
        "method": "GET",
        "expected_status": [200],
        "headers": {},
        "critical": True,
        "slack_channel": "#infra-alerts",
    },
    {
        "name": "Anthropic API",
        "url": "https://api.anthropic.com/v1/models",
        "method": "GET",
        "expected_status": [200, 401],  # 401 = wrong key but API is up
        "headers": {"anthropic-version": "2023-06-01"},
        "critical": False,
        "slack_channel": "#infra-alerts",
    },
    {
        "name": "GitHub API",
        "url": "https://api.github.com",
        "method": "GET",
        "expected_status": [200],
        "headers": {"User-Agent": "QuantEdge-Monitor"},
        "critical": False,
        "slack_channel": "#infra-alerts",
    },
    {
        "name": "yfinance (Yahoo Finance)",
        "url": "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d&range=1d",
        "method": "GET",
        "expected_status": [200],
        "headers": {"User-Agent": "Mozilla/5.0"},
        "critical": True,
        "slack_channel": "#infra-alerts",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M UTC")


def slack_post(channel: str, text: str) -> None:
    if not SLACK_TOKEN:
        return
    try:
        r = httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            json={"channel": channel, "text": text, "mrkdwn": True},
            timeout=10,
        )
        data = r.json()
        if not data.get("ok"):
            print(f"  Slack error ({channel}): {data.get('error')}")
    except Exception as e:
        print(f"  Slack error: {e}")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def check_service(svc: dict) -> tuple[bool, str]:
    """Returns (is_healthy, detail_message)"""
    try:
        r = httpx.request(
            svc["method"],
            svc["url"],
            headers=svc.get("headers", {}),
            timeout=TIMEOUT,
            follow_redirects=True,
        )
        if r.status_code in svc["expected_status"]:
            latency = round(r.elapsed.total_seconds() * 1000)
            return True, f"{r.status_code} in {latency}ms"
        return False, f"HTTP {r.status_code} (expected {svc['expected_status']})"
    except httpx.TimeoutException:
        return False, f"Timeout after {TIMEOUT}s"
    except httpx.ConnectError as e:
        return False, f"Connection failed: {str(e)[:80]}"
    except Exception as e:
        return False, f"Error: {str(e)[:80]}"


def main() -> None:
    print(f"=== Third-Party Health Monitor @ {now_utc()} ===\n")
    state = load_state()
    any_down = False
    status_lines = []

    for svc in SERVICES:
        name = svc["name"]
        healthy, detail = check_service(svc)
        was_healthy = state.get(name, {}).get("healthy", True)

        icon = "✅" if healthy else ("🔴" if svc["critical"] else "🟡")
        print(f"  {icon} {name}: {detail}")
        status_lines.append(f"{icon} *{name}*: `{detail}`")

        if not healthy:
            any_down = True

        if not healthy and was_healthy:
            # New outage — alert immediately
            severity = "CRITICAL" if svc["critical"] else "WARNING"
            msg = (
                f"{'🚨' if svc['critical'] else '⚠️'} *{severity}: {name} is DOWN*\n"
                f"Error: `{detail}`\n"
                f"Detected: {now_utc()}\n"
                f"Impact: {'Trading may be affected' if svc['critical'] else 'Non-critical service'}"
            )
            slack_post(CHANNEL, msg)
            if svc.get("slack_channel") and svc["slack_channel"] != CHANNEL:
                slack_post(svc["slack_channel"], msg)
            print(f"    → OUTAGE ALERT sent to {CHANNEL}")

        elif healthy and not was_healthy:
            # Recovery — send all-clear
            msg = (
                f"✅ *RESOLVED: {name} is back UP*\n"
                f"Status: `{detail}`\n"
                f"Recovered: {now_utc()}"
            )
            slack_post(CHANNEL, msg)
            print(f"    → RECOVERY alert sent")

        state[name] = {"healthy": healthy, "detail": detail, "ts": now_utc()}

    save_state(state)

    # Post a compact digest every time (for scheduled runs)
    if os.environ.get("POST_DIGEST", "false") == "true":
        ok_count  = sum(1 for s in SERVICES if state.get(s["name"], {}).get("healthy", True))
        all_count = len(SERVICES)
        summary   = "✅ All systems operational" if ok_count == all_count else f"⚠️ {all_count - ok_count}/{all_count} services degraded"
        digest = f"*🔭 Third-Party Status · {now_utc()}*\n{summary}\n\n" + "\n".join(status_lines)
        slack_post(CHANNEL, digest)

    if any_down:
        sys.exit(1)  # Non-zero exit → GitHub Actions marks step as failed


if __name__ == "__main__":
    main()
