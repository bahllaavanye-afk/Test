#!/usr/bin/env python3
"""Verify the live QuantEdge deployment from anywhere — no Render dashboard needed.

It does two things:
  1. Probes every candidate backend host + the frontend and decides, for each,
     whether it's the REAL app (~100 routes), the old 3-route stub, suspended,
     or down — purely over HTTP, so it works with zero credentials.
  2. If a RENDER_API_KEY is available (e.g. `doppler run -- python scripts/verify_live.py`,
     or RENDER_API_KEY exported), it also queries the Render management API for the
     authoritative service list: name, id, runtime (python vs docker), branch, and
     whether each service is suspended. This is what closes the "I can't see the real
     state from here" gap.

Exit code is non-zero if no backend currently serves the real app, so it doubles as a
CI/smoke gate.

Usage:
    python scripts/verify_live.py
    doppler run -- python scripts/verify_live.py     # adds authoritative Render state
    RENDER_API_KEY=rnd_... python scripts/verify_live.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# Keeper first. (The bare quantedge-api.onrender.com host is a known orphan stub.)
BACKEND_HOSTS = [
    "https://quantedge-api-6orc.onrender.com",
    "https://quantedge-api-9jz0.onrender.com",
    "https://quantedge-api.onrender.com",
]
FRONTEND = "https://quantedge.vercel.app"
REAL_APP_MIN_ROUTES = 30
_TIMEOUT = 20


def _get(url: str, timeout: int = _TIMEOUT):
    """Return (status_code|None, body_bytes). None status = couldn't connect."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # URLError, timeout, DNS, etc.
        return None, str(e).encode()


def probe_backend(base: str) -> tuple[str, bool]:
    """Return (human verdict, is_real_app)."""
    health, _ = _get(base + "/health")
    ostatus, body = _get(base + "/openapi.json")
    routes = title = None
    if ostatus == 200:
        try:
            spec = json.loads(body)
            routes = len(spec.get("paths", {}))
            title = spec.get("info", {}).get("title")
        except Exception:
            pass
    if routes is not None and routes >= REAL_APP_MIN_ROUTES:
        return f"REAL APP ✅  ({routes} routes, health={health})", True
    if health == 503:
        return "SUSPENDED ⛔  (503 Service Suspended)", False
    if routes is not None:
        return f"stub/other ⚠️  ({routes} routes, title={title!r})", False
    if health is not None:
        return f"responding but no openapi (health={health})", False
    return "down / unreachable ❌", False


def render_api_services():
    key = os.environ.get("RENDER_API_KEY", "").strip()
    if not key:
        return None
    req = urllib.request.Request(
        "https://api.render.com/v1/services?limit=50",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"}
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    print("=== QuantEdge live verification ===\n")
    any_real = False

    print("Backends (HTTP probe):")
    for h in BACKEND_HOSTS:
        verdict, is_real = probe_backend(h)
        any_real = any_real or is_real
        print(f"  {h:<46s} {verdict}")

    fstatus, _ = _get(FRONTEND)
    print(f"\nFrontend:\n  {FRONTEND:<46s} health={fstatus}")

    services = render_api_services()
    print("\nRender management API:")
    if services is None:
        print("  (no RENDER_API_KEY in env — run via `doppler run` or export the key")
        print("   for authoritative name/runtime/branch/suspended state)")
    elif isinstance(services, dict) and services.get("error"):
        print(f"  error: {services['error']}")
    else:
        for item in services:
            svc = item.get("service", item) if isinstance(item, dict) else {}
            details = svc.get("serviceDetails", {}) or {}
            print(
                f"  {svc.get('name','?'):<22s} id={svc.get('id','?')} "
                f"runtime={details.get('env', details.get('runtime','?'))} "
                f"branch={svc.get('branch','?')} suspended={svc.get('suspended','?')}"
            )

    print("\nVerdict:", "a live backend serves the real app ✅" if any_real
          else "NO live backend serves the real app ❌ (resume the keeper service)")
    return 0 if any_real else 1


if __name__ == "__main__":
    sys.exit(main())
