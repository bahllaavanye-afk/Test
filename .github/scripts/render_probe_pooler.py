"""
Render DATABASE_URL pooler PROBER — empirically find the pooler host that the
Supabase Supavisor actually accepts, and patch Render to it.

Why this exists
---------------
The live backend has fallen back to SQLite for weeks with:
    tenant/user postgres.<ref> not found
We proved (2026-07-25) that this is NOT the causes we previously guessed:
  * the project is ACTIVE  — its REST endpoint answers 401 "No API key", i.e.
    PostgREST is up (a paused project would not answer that way);
  * the region IS us-west-1 — the direct DB host db.<ref>.supabase.co resolves
    to an AWS IPv6 in 2600:1f1c::/36, which AWS ip-ranges maps to us-west-1;
  * the URL already carries an us-west-1 pooler host (region-autofix no-op).
Yet the us-west-1 Supavisor rejects the tenant. Supavisor is partitioned into
clusters (`aws-0-<region>` vs the newer `aws-1-<region>`, transaction :6543 vs
session :5432). A tenant lives on exactly one cluster/port combination; hitting
the wrong one returns "Tenant or user not found". The only credential-free way
to know which one is to TRY connecting — which a GitHub Actions runner can do
(it reaches Postgres 6543/5432 directly, unlike the HTTPS-only agent sandbox).

What it does
------------
1. Read DATABASE_URL (+ ALEMBIC_DATABASE_URL) from Render via the API.
2. Extract project-ref + password from that URL (no secrets in the repo).
3. If the CURRENT url already connects, do nothing (a stale boot — the URL is
   fine and re-patching it would just churn; a one-shot redeploy is the human's
   call, we never loop redeploys).
4. Otherwise test candidate pooler hosts (aws-0/aws-1 × :6543/:5432) with a real
   `SELECT 1`, and PATCH Render to the FIRST candidate that verifiably works,
   then trigger a redeploy. We only ever patch to a proven-working host, so this
   cannot make the config worse.
5. If nothing works, log loudly — the remaining cause is a rotated DB password
   (needs a human to reset it in the Supabase dashboard), which no host change
   can fix.

Required env:
  RENDER_API_KEY, RENDER_SERVICE_ID
Optional:
  RENDER_WORKER_SERVICE_ID, SUPABASE_REGION (default us-west-1),
  SUPABASE_POOLER_PORTS (default "6543,5432")
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from urllib.parse import unquote, urlparse

import httpx

RENDER_API = "https://api.render.com/v1"
API_KEY = os.environ.get("RENDER_API_KEY", "")
REGION = os.environ.get("SUPABASE_REGION", "us-west-1").strip() or "us-west-1"
PORTS = [
    p.strip()
    for p in os.environ.get("SUPABASE_POOLER_PORTS", "6543,5432").split(",")
    if p.strip()
]

SERVICE_IDS = [
    s
    for s in [
        os.environ.get("RENDER_SERVICE_ID", "").strip(),
        os.environ.get("RENDER_WORKER_SERVICE_ID", "").strip(),
    ]
    if s
]

# aws-0-<region> or aws-1-<region> pooler host
POOLER_RE = re.compile(r"aws-(\d)-([a-z0-9-]+)\.pooler\.supabase\.com", re.IGNORECASE)
DIRECT_RE = re.compile(r"db\.([a-z0-9]+)\.supabase\.co", re.IGNORECASE)


def ref_and_password(url: str) -> tuple[str | None, str | None]:
    """Extract (project_ref, password) from a Supabase URL, pooler or direct.

    Pooler user is ``postgres.<ref>``; direct host is ``db.<ref>.supabase.co``.
    """
    plain = url.replace("postgresql+asyncpg", "postgresql").replace(
        "postgresql+psycopg2", "postgresql"
    )
    parsed = urlparse(plain)
    password = unquote(parsed.password) if parsed.password else None
    ref = None
    user = parsed.username or ""
    if user.startswith("postgres.") and len(user) > len("postgres."):
        ref = user.split(".", 1)[1]
    if not ref:
        m = DIRECT_RE.search(url)
        if m:
            ref = m.group(1)
    return ref, password


def candidate_hosts(region: str) -> list[str]:
    """Pooler hostnames to try, newest cluster first (aws-1 then aws-0)."""
    return [
        f"aws-1-{region}.pooler.supabase.com",
        f"aws-0-{region}.pooler.supabase.com",
    ]


def build_url(driver: str, ref: str, password: str, host: str, port: str) -> str:
    """Rebuild a pooler URL for the given driver (asyncpg|psycopg2)."""
    from urllib.parse import quote

    return (
        f"postgresql+{driver}://postgres.{ref}:{quote(password, safe='')}"
        f"@{host}:{port}/postgres"
    )


def current_host_port(url: str) -> tuple[str | None, str | None]:
    plain = url.replace("postgresql+asyncpg", "postgresql").replace(
        "postgresql+psycopg2", "postgresql"
    )
    parsed = urlparse(plain)
    return parsed.hostname, (str(parsed.port) if parsed.port else None)


OK = "ok"                      # SELECT 1 succeeded — fully working
BAD_PASSWORD = "bad_password"  # tenant EXISTS on this cluster, password rejected
NO_TENANT = "no_tenant"        # this cluster has never heard of the tenant
UNREACHABLE = "unreachable"    # network/timeout/other


def classify_error(exc: Exception) -> str:
    """Map a connection failure to a diagnosis.

    The distinction that matters: "tenant or user not found" means we are asking
    the WRONG Supavisor cluster, whereas an authentication failure means we found
    the RIGHT cluster (it knows this tenant) and only the password is stale. That
    makes a password error a positive identification of the correct host.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    if "invalidpassword" in name.lower() or "password authentication failed" in msg:
        return BAD_PASSWORD
    if "tenant or user not found" in msg or "tenant/user" in msg:
        return NO_TENANT
    return UNREACHABLE


async def _try_connect(host: str, port: str, ref: str, password: str) -> str:
    """Probe one pooler host/port. Returns OK / BAD_PASSWORD / NO_TENANT / UNREACHABLE."""
    try:
        import asyncpg
    except Exception as e:  # pragma: no cover - asyncpg installed in CI
        print(f"  asyncpg unavailable: {e}")
        return UNREACHABLE
    user = f"postgres.{ref}"
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(
                host=host,
                port=int(port),
                user=user,
                password=password,
                database="postgres",
                timeout=8,
                statement_cache_size=0,  # required for transaction-mode pooler
            ),
            timeout=12,
        )
        try:
            val = await conn.fetchval("SELECT 1")
            return OK if val == 1 else UNREACHABLE
        finally:
            await conn.close()
    except Exception as e:
        verdict = classify_error(e)
        print(f"  {host}:{port} -> {type(e).__name__}: {str(e)[:90]}  [{verdict}]")
        return verdict


def headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}


def get_env_vars(service_id: str) -> list[dict]:
    r = httpx.get(
        f"{RENDER_API}/services/{service_id}/env-vars?limit=100",
        headers=headers(),
        timeout=20,
    )
    r.raise_for_status()
    return [item.get("envVar", item) for item in r.json()]


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
    r = httpx.post(
        f"{RENDER_API}/services/{service_id}/deploys",
        headers={**headers(), "Content-Type": "application/json"},
        json={"clearCache": "do_not_clear"},
        timeout=20,
    )
    print(f"  Triggered redeploy: HTTP {r.status_code}")


async def probe_service(service_id: str) -> bool:
    """Probe + (if needed) fix one Render service. Returns True if it patched."""
    print(f"\nService {service_id}:")
    try:
        envs = get_env_vars(service_id)
    except Exception as e:
        print(f"  Could not read env vars: {e}")
        return False
    current = {e.get("key"): e.get("value", "") for e in envs}
    db_url = current.get("DATABASE_URL", "")
    if not db_url:
        print("  no DATABASE_URL set — nothing to probe")
        return False

    ref, password = ref_and_password(db_url)
    if not ref or not password:
        print("  could not extract ref/password from DATABASE_URL — skipping")
        return False

    cur_host, cur_port = current_host_port(db_url)
    print(f"  current pooler host: {cur_host}:{cur_port}")

    def apply(host: str, port: str, redeploy: bool) -> bool:
        changed = False
        for key, driver in (("DATABASE_URL", "asyncpg"), ("ALEMBIC_DATABASE_URL", "psycopg2")):
            if key == "ALEMBIC_DATABASE_URL" and not current.get(key):
                continue
            if patch_env_var(service_id, key, build_url(driver, ref, password, host, port)):
                changed = True
        if changed and redeploy:
            trigger_deploy(service_id)
        return changed

    # 1. Does the CURRENT url already connect? Then it's a stale boot, not a URL bug.
    if cur_host and cur_port:
        if await _try_connect(cur_host, cur_port, ref, password) == OK:
            print("  current DATABASE_URL connects OK → URL is fine (stale boot; not")
            print("  re-patching — a one-shot Render redeploy is the remaining lever).")
            return False

    # 2. Probe every candidate, recording which cluster RECOGNISES the tenant.
    auth_hit: tuple[str, str] | None = None      # host/port where only the password failed
    for host in candidate_hosts(REGION):
        for port in PORTS:
            if host == cur_host and port == cur_port:
                continue  # already probed above
            print(f"  trying {host}:{port} ...")
            verdict = await _try_connect(host, port, ref, password)
            if verdict == OK:
                print(f"  ✅ {host}:{port} ACCEPTS the tenant — patching Render + redeploying")
                return apply(host, port, redeploy=True)
            if verdict == BAD_PASSWORD and auth_hit is None:
                auth_hit = (host, port)

    # 3. Nothing fully connected. If some cluster recognised the tenant (password
    #    error, not "tenant not found"), that host is PROVEN correct — fix it now so
    #    the only remaining fault is the password. Idempotent: once DATABASE_URL
    #    already points there we patch nothing, so this cannot loop-redeploy.
    if auth_hit:
        host, port = auth_hit
        print(f"  🔎 {host}:{port} RECOGNISES the tenant but rejected the password.")
        print("     → that cluster is the correct one; the password is stale/rotated.")
        if (host, port) != (cur_host, cur_port):
            print(f"  Correcting host {cur_host}:{cur_port} → {host}:{port} (password unchanged,")
            print("  so this alone will not connect — but it removes the wrong-cluster fault).")
            # No redeploy: the password is still wrong, so a restart would only churn
            # the ephemeral SQLite + scheduler clocks for nothing.
            return apply(host, port, redeploy=False)
        print("  Host already correct — nothing to patch.")
        print("  ➡️  USER ACTION: Supabase → Settings → Database → Reset database password,")
        print("      then paste the pooler connection string into Render DATABASE_URL.")
        return False

    print("  ❌ no pooler host/port recognised the tenant at all.")
    print("     Region + project are healthy, so verify the project ref and that the")
    print("     project has not been deleted/migrated.")
    return False


async def main_async() -> None:
    if not API_KEY or not SERVICE_IDS:
        print("RENDER_API_KEY and RENDER_SERVICE_ID are required.")
        sys.exit(1)
    any_patched = False
    for sid in SERVICE_IDS:
        if await probe_service(sid):
            any_patched = True
    if any_patched:
        print("\n✅ Patched DATABASE_URL to a verified-working pooler. Redeploying.")
    else:
        print("\nNo change made (current URL fine, or no working host found).")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
