"""
Render Live Log Fetcher

Fetches and streams the latest deploy + service logs from Render API.
Run this to see exactly what's failing on Render without needing the web UI.

Usage:
  RENDER_API_KEY=rnd_... RENDER_SERVICE_ID=srv_... python .github/scripts/render_logs.py

Or trigger via:  GitHub Actions → render-monitor → Run workflow → force_check=true

Required env vars:
  RENDER_API_KEY     — Render Dashboard → Account Settings → API Keys → Create API Key
  RENDER_SERVICE_ID  — URL of your Render service: dashboard.render.com/web/srv_XXXXXX
                       The service ID is the "srv_XXXXXX" part
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

import httpx

RENDER_API    = "https://api.render.com/v1"
API_KEY       = os.environ.get("RENDER_API_KEY", "")
SERVICE_ID    = os.environ.get("RENDER_SERVICE_ID", "")
TIMEOUT       = 20


def render_get(path: str, params: dict | None = None) -> dict | list | None:
    if not API_KEY:
        print("ERROR: RENDER_API_KEY not set")
        return None
    try:
        r = httpx.get(
            f"{RENDER_API}{path}",
            headers={"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"},
            params=params or {},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        print(f"Render API {r.status_code}: {r.text[:300]}")
    except httpx.ConnectError as e:
        print(f"Cannot reach Render API: {e}")
    except Exception as e:
        print(f"Render API error: {e}")
    return None


def get_service_info() -> dict | None:
    data = render_get(f"/services/{SERVICE_ID}")
    return data


def get_deploys(limit: int = 5) -> list[dict]:
    data = render_get(f"/services/{SERVICE_ID}/deploys", {"limit": limit})
    if not data or not isinstance(data, list):
        return []
    return [item.get("deploy", item) for item in data]


def get_deploy_logs(deploy_id: str) -> list[str]:
    data = render_get(f"/services/{SERVICE_ID}/deploys/{deploy_id}/logs")
    if not data or not isinstance(data, list):
        return []
    return [item.get("message", "") for item in data]


def get_service_events(limit: int = 20) -> list[dict]:
    data = render_get(f"/services/{SERVICE_ID}/events", {"limit": limit})
    if not data or not isinstance(data, list):
        return []
    return [item.get("event", item) for item in data]


def fmt_ts(ts_str: str | None) -> str:
    if not ts_str:
        return "—"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ts_str[:19]


def main() -> None:
    if not API_KEY:
        print("─" * 60)
        print("ERROR: RENDER_API_KEY not set")
        print()
        print("How to get it:")
        print("  1. Go to: https://dashboard.render.com/")
        print("  2. Click Account Settings (top-right avatar)")
        print("  3. API Keys → Create API Key")
        print("  4. Set env: export RENDER_API_KEY=rnd_...")
        print()
        print("How to get RENDER_SERVICE_ID:")
        print("  1. Go to your Render service page")
        print("  2. The URL is: dashboard.render.com/web/srv_XXXXXX")
        print("  3. Copy 'srv_XXXXXX' → export RENDER_SERVICE_ID=srv_...")
        print("─" * 60)
        sys.exit(1)

    if not SERVICE_ID:
        print("ERROR: RENDER_SERVICE_ID not set")
        print("It's in your Render service URL: dashboard.render.com/web/srv_XXXXXX")
        sys.exit(1)

    print(f"=== Render Service Diagnostics @ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")
    print(f"Service ID: {SERVICE_ID}")
    print()

    # Service info
    svc = get_service_info()
    if svc:
        print(f"Service Name  : {svc.get('name', '?')}")
        print(f"Status        : {svc.get('suspended', False) and 'SUSPENDED' or 'active'}")
        print(f"Type          : {svc.get('type', '?')}")
        print(f"Region        : {svc.get('serviceDetails', {}).get('region', '?')}")
        print(f"URL           : {svc.get('serviceDetails', {}).get('url', '?')}")
        print()

    # Recent deploys
    deploys = get_deploys(5)
    if not deploys:
        print("No deploys found (check SERVICE_ID)")
        return

    print(f"── Recent Deploys ──────────────────────────────────────────")
    for d in deploys:
        status     = d.get("status", "?")
        deploy_id  = d.get("id", "?")
        created    = fmt_ts(d.get("createdAt"))
        commit_msg = ""
        if d.get("commit"):
            commit_msg = d["commit"].get("message", "")[:60]
        status_icon = {"live": "✅", "build_failed": "🔴", "failed": "🔴",
                       "canceled": "⚫", "deactivated": "⚫"}.get(status, "🟡")
        print(f"  {status_icon} {status:15} {created}  {commit_msg}")

    print()

    # Latest failed deploy logs
    latest = deploys[0]
    status = latest.get("status", "")
    deploy_id = latest.get("id", "")

    FAILED = ("build_failed", "failed", "canceled", "update_failed")
    if status in FAILED:
        print(f"── Failure Logs (deploy {deploy_id[:12]}) ─────────────────────────")
        logs = get_deploy_logs(deploy_id)
        if logs:
            # Print last 80 lines
            for line in logs[-80:]:
                print(f"  {line}")
        else:
            print("  (No logs available — try again in a minute)")
        print()

        # Diagnose common patterns
        log_text = "\n".join(logs[-200:])
        print("── Diagnosis ────────────────────────────────────────────────")
        if "Network is unreachable" in log_text or "ENETUNREACH" in log_text:
            print("  🔴 IPv6 connectivity error")
            print()
            print("  If this is during alembic/database migrations:")
            print("  → Your DATABASE_URL points to a host that resolves to IPv6")
            print("  → Render free tier has NO outbound IPv6")
            print()
            print("  If using Supabase:")
            print("  → Go to: Supabase Dashboard → Settings → Database → Connection pooling")
            print("  → Copy 'Transaction mode' URL (port 6543, uses IPv4)")
            print("  → Format: postgres://postgres.PROJECT:PASS@aws-0-REGION.pooler.supabase.com:6543/postgres")
            print("  → Set this as DATABASE_URL in Render → Environment")
        elif "ModuleNotFoundError" in log_text or "ImportError" in log_text:
            import re
            m = re.search(r"(?:ModuleNotFoundError|ImportError): ([^\n]+)", log_text)
            print(f"  🔴 Missing dependency: {m.group(1) if m else 'unknown'}")
            print("  → Add the package to backend/pyproject.toml [dependencies]")
        elif "pip install" in log_text and ("error" in log_text.lower()):
            print("  🔴 pip install failed during build")
            print("  → Check backend/pyproject.toml for incompatible version constraints")
        elif "address already in use" in log_text.lower():
            print("  🔴 Port conflict — uvicorn can't bind to port")
            print("  → Check render.yaml startCommand or PORT env var")
        elif status == "build_failed":
            print("  🔴 Build failed — scroll up for the specific error")
        else:
            print("  ❓ Unknown failure — check logs above for error messages")
        print()

    elif status == "live":
        print(f"  ✅ Latest deploy is LIVE — service is healthy")
        print()

    # Events
    events = get_service_events(10)
    if events:
        print(f"── Recent Events ────────────────────────────────────────────")
        for ev in events[:10]:
            ts  = fmt_ts(ev.get("createdAt"))
            typ = ev.get("type", "?")
            print(f"  {ts}  {typ}")
        print()


if __name__ == "__main__":
    main()
