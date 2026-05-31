"""
Render DATABASE_URL auto-fixer.

The Supabase DIRECT connection host (db.<ref>.supabase.co:5432) resolves to
IPv6, which Render's free tier cannot reach → every deploy/migration fails
with "Network is unreachable".

This script uses the Render API to:
  1. Read the service's current DATABASE_URL (+ ALEMBIC_DATABASE_URL).
  2. Detect the IPv6 direct-connection pattern.
  3. Rewrite it to the IPv4 Session Pooler form:
       postgresql+asyncpg://postgres.<ref>:<pass>@aws-0-<REGION>.pooler.supabase.com:6543/postgres
     (alembic variant uses +psycopg2)
  4. PATCH the env var back to Render and trigger a redeploy.

The password + project-ref are extracted from the URL Render already holds —
no secrets live in this repo. The only thing the script can't infer is the
Supabase REGION, so that is passed in via the SUPABASE_REGION env var.

Required env:
  RENDER_API_KEY      — Render API key
  RENDER_SERVICE_ID   — srv_XXXX of quantedge-api (web service)
  SUPABASE_REGION     — e.g. us-east-1, us-west-1, eu-central-1, ap-southeast-1
Optional:
  RENDER_WORKER_SERVICE_ID — srv_XXXX of quantedge-worker (patched too if set)
  SUPABASE_POOLER_PORT     — 6543 (transaction, default) or 5432 (session)
"""
from __future__ import annotations

import os
import re
import sys
from urllib.parse import urlparse

import httpx

RENDER_API   = "https://api.render.com/v1"
API_KEY      = os.environ.get("RENDER_API_KEY", "")
REGION       = os.environ.get("SUPABASE_REGION", "").strip()
POOLER_PORT  = os.environ.get("SUPABASE_POOLER_PORT", "6543").strip()

SERVICE_IDS = [s for s in [
    os.environ.get("RENDER_SERVICE_ID", "").strip(),
    os.environ.get("RENDER_WORKER_SERVICE_ID", "").strip(),
] if s]

DIRECT_RE = re.compile(r"db\.([a-z0-9]+)\.supabase\.co", re.IGNORECASE)


def headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}


def get_env_vars(service_id: str) -> list[dict]:
    r = httpx.get(f"{RENDER_API}/services/{service_id}/env-vars?limit=100",
                  headers=headers(), timeout=20)
    r.raise_for_status()
    data = r.json()
    return [item.get("envVar", item) for item in data]


def to_pooler(url: str, driver: str) -> str | None:
    """Rewrite a Supabase direct URL to the pooler form. driver: asyncpg|psycopg2."""
    m = DIRECT_RE.search(url)
    if not m:
        return None  # not a direct URL — nothing to do
    ref = m.group(1)
    # Extract password from the existing URL: scheme://user:PASS@host...
    parsed = urlparse(url.replace("postgresql+asyncpg", "postgresql").replace("postgresql+psycopg2", "postgresql"))
    password = parsed.password or ""
    if not password:
        return None
    host = f"aws-0-{REGION}.pooler.supabase.com"
    user = f"postgres.{ref}"
    return f"postgresql+{driver}://{user}:{password}@{host}:{POOLER_PORT}/postgres"


def patch_env_var(service_id: str, key: str, value: str) -> bool:
    r = httpx.put(
        f"{RENDER_API}/services/{service_id}/env-vars/{key}",
        headers={**headers(), "Content-Type": "application/json"},
        json={"value": value},
        timeout=20,
    )
    if r.status_code in (200, 201):
        return True
    print(f"  PATCH {key} failed: {r.status_code} {r.text[:160]}")
    return False


def trigger_deploy(service_id: str) -> None:
    r = httpx.post(f"{RENDER_API}/services/{service_id}/deploys",
                   headers={**headers(), "Content-Type": "application/json"},
                   json={"clearCache": "do_not_clear"}, timeout=20)
    print(f"  Triggered redeploy: HTTP {r.status_code}")


def main() -> None:
    if not API_KEY or not SERVICE_IDS:
        print("RENDER_API_KEY and RENDER_SERVICE_ID are required.")
        sys.exit(1)
    if not REGION:
        print("SUPABASE_REGION not set. Find it in Supabase Dashboard → Settings →")
        print("Database → Connection pooling (host looks like aws-0-<REGION>.pooler.supabase.com).")
        print("Common values: us-east-1, us-west-1, eu-central-1, ap-southeast-1, ap-south-1.")
        sys.exit(1)

    any_changed = False
    for sid in SERVICE_IDS:
        print(f"\nService {sid}:")
        try:
            envs = get_env_vars(sid)
        except Exception as e:
            print(f"  Could not read env vars: {e}")
            continue

        current = {e.get("key"): e.get("value", "") for e in envs}
        for key, driver in (("DATABASE_URL", "asyncpg"), ("ALEMBIC_DATABASE_URL", "psycopg2")):
            val = current.get(key, "")
            if not val:
                continue
            if not DIRECT_RE.search(val):
                print(f"  {key}: already pooler/clean — skipping")
                continue
            new_val = to_pooler(val, driver)
            if not new_val:
                print(f"  {key}: could not rewrite (no password in URL?) — skipping")
                continue
            print(f"  {key}: rewriting direct host → pooler ({REGION}:{POOLER_PORT})")
            if patch_env_var(sid, key, new_val):
                any_changed = True

        if any_changed:
            trigger_deploy(sid)

    if any_changed:
        print("\n✅ DATABASE_URL patched to Supabase pooler. Render is redeploying.")
    else:
        print("\nNo changes made (URLs already correct or nothing to patch).")


if __name__ == "__main__":
    main()
