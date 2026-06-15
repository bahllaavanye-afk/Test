"""
Render env-var sync — make GitHub Secrets the single source of truth.

Problem this solves: LLM/Slack keys must exist in the Render *runtime* for
the deployed backend to use them, but pasting them into the Render dashboard
by hand is tedious and easy to forget. Those same keys already live in GitHub
Secrets (the Actions LLM workflows read them). This script copies them into
the Render service via the API so you never touch the dashboard again.

ROBUSTNESS (why this is not a naive copy):
  1. Numbered secrets. The org stores keys numbered — GEMINI_API_KEY_1,
     GROQ_API_KEY_1, DEEPSEEK_API_KEY_1/2/3, etc. The backend gateway's
     _resolve_key() reads either GROQ_API_KEY or GROQ_API_KEY_1. So for each
     provider we (a) push every numbered variant verbatim AND (b) ensure the
     canonical name (GROQ_API_KEY) is populated from the first available
     variant. Either path the runtime takes now finds a key.
  2. Full free cascade. We sync all 7 free providers (Gemini, SambaNova,
     Cerebras, Groq, DeepSeek, Together, OpenRouter, plus NVIDIA NIM) so the
     production llm_common cascade has the same depth as CI.
  3. Retries. Every Render API call retries with exponential backoff — a
     transient 5xx or network blip never silently drops a key.
  4. Non-destructive. Single-key PUT (PUT /v1/services/{id}/env-vars/{key});
     existing vars (DATABASE_URL, SECRET_KEY, ALPACA_*, ...) are untouched.
     Empty secrets are skipped, so a value already on Render is never blanked.
  5. Deploy verification. After syncing we trigger a deploy and poll its
     status, exiting non-zero only if the deploy itself fails.

Required env:
  RENDER_API_KEY     — Render API key
  RENDER_SERVICE_ID  — srv_XXXX of quantedge-api (web service)
Optional:
  RENDER_WORKER_SERVICE_ID — srv_XXXX of the worker (synced + deployed too)
"""
from __future__ import annotations

import os
import sys
import time

import httpx

RENDER_API = "https://api.render.com/v1"
API_KEY = os.environ.get("RENDER_API_KEY", "").strip()
WEB_ID = os.environ.get("RENDER_SERVICE_ID", "").strip()
WORKER_ID = os.environ.get("RENDER_WORKER_SERVICE_ID", "").strip()

# Providers whose canonical name the backend gateway resolves, each with the
# numbered variants we also mirror. The canonical key is filled from the first
# non-empty variant when the bare env var itself is unset.
_PROVIDER_VARIANTS: dict[str, list[str]] = {
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"],
    "GROQ_API_KEY": ["GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3"],
    "DEEPSEEK_API_KEY": ["DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_KEY_3"],
    "SAMBANOVA_API_KEY": ["SAMBANOVA_API_KEY", "SAMBANOVA_API_KEY_1"],
    "CEREBRAS_API_KEY": ["CEREBRAS_API_KEY", "CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY_2"],
    "TOGETHER_API_KEY": ["TOGETHER_API_KEY", "TOGETHER_API_KEY_1"],
    "HYPERBOLIC_API_KEY": ["HYPERBOLIC_API_KEY", "HYPERBOLIC_API_KEY_1"],
    "OPENROUTER_API_KEY": ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"],
}

# Single-value keys mirrored verbatim (no numbered fallback logic).
_SINGLE_KEYS = [
    "SLACK_BOT_TOKEN",
    "ANTHROPIC_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "NVIDIA_AGENTS_API_KEYS",
]


def headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _request(method: str, url: str, *, json_body: dict | None = None, attempts: int = 4) -> httpx.Response | None:
    """HTTP with exponential backoff (2s, 4s, 8s). Returns the final response or None."""
    delay = 2.0
    last: httpx.Response | None = None
    for i in range(attempts):
        try:
            r = httpx.request(method, url, headers=headers(), json=json_body, timeout=25)
            last = r
            if r.status_code < 500 and r.status_code != 429:
                return r
            print(f"  {method} {url.split('/')[-1]} → {r.status_code}, retry {i + 1}/{attempts}")
        except Exception as e:  # noqa: BLE001
            print(f"  {method} {url.split('/')[-1]} error: {e}; retry {i + 1}/{attempts}")
        if i < attempts - 1:
            time.sleep(delay)
            delay *= 2
    return last


def _resolve_values() -> dict[str, str]:
    """Build the final {ENV_KEY: value} map to push to Render."""
    out: dict[str, str] = {}

    for canonical, variants in _PROVIDER_VARIANTS.items():
        first_value = ""
        for var in variants:
            val = os.environ.get(var, "").strip()
            if not val:
                continue
            out[var] = val                  # mirror the numbered variant verbatim
            if not first_value:
                first_value = val
        # Ensure the canonical name the backend gateway reads is populated.
        if first_value and not out.get(canonical):
            out[canonical] = first_value

    for key in _SINGLE_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            out[key] = val

    return out


def put_env_var(sid: str, key: str, value: str) -> bool:
    r = _request("PUT", f"{RENDER_API}/services/{sid}/env-vars/{key}", json_body={"value": value})
    if r is not None and r.status_code in (200, 201):
        return True
    # Never print r.text — Render's validation errors can echo the submitted
    # value, which would leak a secret into public Actions logs. Status only.
    detail = str(r.status_code) if r is not None else "no response"
    print(f"  PUT {key} failed: HTTP {detail}")
    return False


def sync_service(sid: str, label: str, values: dict[str, str]) -> int:
    print(f"\n→ syncing {len(values)} secret(s) to {label} ({sid})")
    synced = 0
    for key, value in values.items():
        if put_env_var(sid, key, value):
            print(f"  synced {key}")
            synced += 1
    return synced


def trigger_deploy(sid: str) -> str | None:
    r = _request("POST", f"{RENDER_API}/services/{sid}/deploys", json_body={"clearCache": "do_not_clear"})
    if r is not None and r.status_code in (200, 201):
        dep_id = r.json().get("id")
        print(f"  deploy {dep_id} triggered for {sid}")
        return dep_id
    print(f"  deploy trigger failed for {sid}")
    return None


def wait_for_deploy(sid: str, dep_id: str, timeout_s: int = 240) -> bool:
    """Poll a deploy until it leaves the build/update phase. True on live/succeeded."""
    deadline = time.time() + timeout_s
    terminal_ok = {"live", "succeeded"}
    terminal_bad = {"build_failed", "update_failed", "canceled", "deactivated", "pre_deploy_failed"}
    while time.time() < deadline:
        r = _request("GET", f"{RENDER_API}/services/{sid}/deploys/{dep_id}", attempts=2)
        if r is not None and r.status_code == 200:
            status = r.json().get("status", "")
            print(f"  deploy {dep_id}: {status}")
            if status in terminal_ok:
                return True
            if status in terminal_bad:
                return False
        time.sleep(10)
    print(f"  deploy {dep_id}: still in progress after {timeout_s}s (not failing the job)")
    return True  # don't fail the pipeline just because the free tier build is slow


def main() -> int:
    if not API_KEY or not WEB_ID:
        print("RENDER_API_KEY / RENDER_SERVICE_ID not set — nothing to do.")
        return 0  # soft-skip; never fail the pipeline over a missing secret

    values = _resolve_values()
    if not values:
        print("No non-empty LLM/Slack secrets found in GitHub — nothing synced.")
        return 0

    print(f"Resolved {len(values)} env var(s) to sync: {', '.join(sorted(values))}")

    total = sync_service(WEB_ID, "quantedge-api", values)
    if WORKER_ID:
        total += sync_service(WORKER_ID, "quantedge-worker", values)

    if total == 0:
        print("\nNothing synced (all PUTs failed) — not triggering deploy.")
        return 1

    ok = True
    dep_id = trigger_deploy(WEB_ID)
    if dep_id:
        ok = wait_for_deploy(WEB_ID, dep_id) and ok
    if WORKER_ID:
        wdep = trigger_deploy(WORKER_ID)
        if wdep:
            wait_for_deploy(WORKER_ID, wdep)

    print(f"\nDone — {total} env var(s) pushed to Render. Deploy ok: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
