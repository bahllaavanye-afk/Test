"""
Render env-var sync — make GitHub Secrets the single source of truth.

Problem this solves: LLM/Slack keys must exist in the Render *runtime* for
the deployed backend to use them, but pasting them into the Render dashboard
by hand is tedious and easy to forget. Those same keys already live in GitHub
Secrets (the Actions LLM workflows read them). This script copies them into
the Render service via the API so you never touch the dashboard again.

It is NON-DESTRUCTIVE: it PUTs one env var at a time
(PUT /v1/services/{id}/env-vars/{key}), so existing vars (DATABASE_URL,
SECRET_KEY, ALPACA_*, ...) are left untouched. Keys whose GitHub Secret is
empty are skipped (never blanks out a value already set on Render).

Required env:
  RENDER_API_KEY     — Render API key
  RENDER_SERVICE_ID  — srv_XXXX of quantedge-api (web service)
Optional:
  RENDER_WORKER_SERVICE_ID — srv_XXXX of the worker (synced too if present)

Synced keys (whichever are present as non-empty env vars):
  GROQ_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, SLACK_BOT_TOKEN,
  ANTHROPIC_API_KEY
"""
from __future__ import annotations

import os
import sys

import httpx

RENDER_API = "https://api.render.com/v1"
API_KEY = os.environ.get("RENDER_API_KEY", "").strip()
WEB_ID = os.environ.get("RENDER_SERVICE_ID", "").strip()
WORKER_ID = os.environ.get("RENDER_WORKER_SERVICE_ID", "").strip()

# Keys we mirror from GitHub Secrets → Render runtime.
SYNC_KEYS = [
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "SLACK_BOT_TOKEN",
    "ANTHROPIC_API_KEY",
]


def headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def put_env_var(sid: str, key: str, value: str) -> bool:
    """Create-or-update a single env var without disturbing the others."""
    r = httpx.put(
        f"{RENDER_API}/services/{sid}/env-vars/{key}",
        headers=headers(),
        json={"value": value},
        timeout=20,
    )
    if r.status_code in (200, 201):
        return True
    print(f"  PUT {key} failed: {r.status_code} {r.text[:160]}")
    return False


def trigger_deploy(sid: str) -> None:
    r = httpx.post(
        f"{RENDER_API}/services/{sid}/deploys",
        headers=headers(),
        json={"clearCache": "do_not_clear"},
        timeout=20,
    )
    if r.status_code in (200, 201):
        print(f"  deploy triggered for {sid}")
    else:
        print(f"  deploy trigger failed: {r.status_code} {r.text[:160]}")


def sync_service(sid: str, label: str) -> int:
    print(f"\n→ syncing secrets to {label} ({sid})")
    synced = 0
    for key in SYNC_KEYS:
        value = os.environ.get(key, "").strip()
        if not value:
            print(f"  skip {key} (GitHub Secret empty)")
            continue
        if put_env_var(sid, key, value):
            print(f"  synced {key}")
            synced += 1
    return synced


def main() -> int:
    if not API_KEY or not WEB_ID:
        print("RENDER_API_KEY / RENDER_SERVICE_ID not set — nothing to do.")
        return 0  # soft-skip; never fail the pipeline over a missing secret

    total = sync_service(WEB_ID, "quantedge-api")
    if WORKER_ID:
        total += sync_service(WORKER_ID, "quantedge-worker")

    if total == 0:
        print("\nNo non-empty LLM/Slack secrets found in GitHub — nothing synced.")
        return 0

    # Only redeploy if we actually changed something.
    trigger_deploy(WEB_ID)
    if WORKER_ID:
        trigger_deploy(WORKER_ID)
    print(f"\nDone — {total} env var(s) pushed to Render.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
