"""
Render service config auto-fixer.

Root problem: when the service was first created from the original blueprint,
its build command was `pip install ... && alembic upgrade head`. Alembic runs
at BUILD time against the Supabase IPv6 direct host → "Network is unreachable"
→ build fails. Editing render.yaml in git does NOT override a build command
already stored on the Render service.

This script uses the Render API to force the service's commands to the safe
values:
  buildCommand : pip install -e "."         (NO alembic at build time)
  startCommand : bash start.sh              (migrations run here, IPv6-skipping)

After this, the build succeeds; start.sh detects an IPv6 DATABASE_URL and skips
migrations (printing fix instructions) so the server always comes up.

Required env:
  RENDER_API_KEY     — Render API key
  RENDER_SERVICE_ID  — srv_XXXX of quantedge-api (web service)
Optional:
  RENDER_WORKER_SERVICE_ID — srv_XXXX of the worker (start cmd left as-is)
"""
from __future__ import annotations

import os
import sys

import httpx

RENDER_API = "https://api.render.com/v1"
API_KEY    = os.environ.get("RENDER_API_KEY", "")
WEB_ID     = os.environ.get("RENDER_SERVICE_ID", "").strip()
WORKER_ID  = os.environ.get("RENDER_WORKER_SERVICE_ID", "").strip()

SAFE_BUILD = 'cd backend && pip install -e "." || (pip install uv && uv pip install --system -e .)'
WEB_START  = "bash backend/start.sh"


def headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json",
            "Content-Type": "application/json"}


def get_service(sid: str) -> dict | None:
    r = httpx.get(f"{RENDER_API}/services/{sid}", headers=headers(), timeout=20)
    if r.status_code == 200:
        return r.json()
    print(f"  GET service failed: {r.status_code} {r.text[:160]}")
    return None


def patch_service(sid: str, build_cmd: str, start_cmd: str) -> bool:
    payload = {
        "serviceDetails": {
            "envSpecificDetails": {
                "buildCommand": build_cmd,
                "startCommand": start_cmd,
            }
        }
    }
    r = httpx.patch(f"{RENDER_API}/services/{sid}", headers=headers(), json=payload, timeout=20)
    if r.status_code in (200, 201):
        return True
    print(f"  PATCH failed: {r.status_code} {r.text[:200]}")
    return False


def trigger_deploy(sid: str) -> None:
    r = httpx.post(f"{RENDER_API}/services/{sid}/deploys", headers=headers(),
                   json={"clearCache": "clear"}, timeout=20)
    print(f"  Triggered redeploy (cache cleared): HTTP {r.status_code}")


def main() -> None:
    if not API_KEY or not WEB_ID:
        print("RENDER_API_KEY and RENDER_SERVICE_ID are required.")
        sys.exit(1)

    svc = get_service(WEB_ID)
    if svc:
        details = svc.get("serviceDetails", {}).get("envSpecificDetails", {})
        print(f"Web service current buildCommand: {details.get('buildCommand', '?')[:120]}")
        print(f"Web service current startCommand: {details.get('startCommand', '?')[:120]}")

    print(f"\nPatching {WEB_ID} → safe build/start commands...")
    if patch_service(WEB_ID, SAFE_BUILD, WEB_START):
        print("  ✅ buildCommand = pip install -e \".\"  (no alembic at build)")
        print("  ✅ startCommand = bash start.sh")
        trigger_deploy(WEB_ID)
    else:
        print("  ✗ Could not patch web service config")

    if WORKER_ID:
        wsvc = get_service(WORKER_ID)
        wstart = "python -m app.tasks.scheduler"
        if wsvc:
            wstart = wsvc.get("serviceDetails", {}).get("envSpecificDetails", {}).get("startCommand", wstart)
        print(f"\nPatching worker {WORKER_ID} build command...")
        if patch_service(WORKER_ID, SAFE_BUILD, wstart):
            print("  ✅ worker buildCommand fixed")
            trigger_deploy(WORKER_ID)

    print("\nDone. The build will no longer run alembic. If the app still can't")
    print("reach the DB, also run 'Render — Fix DATABASE_URL' to switch to the pooler URL.")


if __name__ == "__main__":
    main()
