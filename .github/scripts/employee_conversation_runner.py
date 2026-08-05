"""Run real LLM-backed conversations for every QuantEdge employee persona."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# ── Key availability check ────────────────────────────────────────────────────
#
# This guard used a HAND-MAINTAINED list of seven env vars:
#
#   GROQ, DEEPSEEK, SAMBANOVA, CEREBRAS, HYPERBOLIC, TOGETHER, GEMINI
#
# `llm_common._PROVIDERS` has eight. The missing one is NVIDIA
# (`NVIDIA_AGENTS_API_KEYS`, alt `NVIDIA_NIM_API_KEY`) — and that is the provider
# `employee-conversations.yml` actually supplies, because its six other mappings
# point at unsuffixed secrets that are empty (the repo's populated secrets are
# the `_1` variants; `company-brain.yml` and friends survive on
# `GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY_1 }}`).
#
# So the pre-flight guard exited 0 while the cascade behind it had a usable
# provider. Measured in production run 30974… (2026-08-05 04:14):
# "No LLM keys available — skipping real conversations", the workflow reported
# SUCCESS, posted "run complete / Responded: 0/47" to #engineering, and did
# nothing — hourly.
#
# The fix is not to add NVIDIA to the list. It is to stop keeping a second list:
# ask the cascade what it can reach, so the guard cannot disagree with the thing
# it guards.
def _any_llm_key() -> bool:
    try:
        from llm_common import _PROVIDERS, _has_key
        return any(_has_key(p) for p in _PROVIDERS)
    except Exception:  # noqa: BLE001 — never let the probe itself block the run
        return True


if not _any_llm_key():
    print("No LLM keys available — skipping real conversations")
    sys.exit(0)

# ── Imports from agent_team (only if keys exist) ───────────────────────
from agent_team import employee_provider_prompt, _EMPLOYEE_PERSONAS  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
STATE_DIR   = REPO_ROOT / ".github" / "state"
MEMORY_FILE = STATE_DIR / "agent_memory.json"
PROOF_FILE  = STATE_DIR / "collaboration_proof.md"

_FALLBACK_TOPICS = [
    "Review today's CI results and flag the highest-priority reliability risk.",
    "Identify the top alpha opportunity or risk signal on your desk right now.",
    "Report one concrete improvement to platform performance or code quality.",
]


def _load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    return {"conversations": {}, "daily_topics": {}, "platform_metrics": {}, "peer_learnings": []}


# Fraction of the provider's declared free-tier RPM this ONE workflow may use.
#
# The first pacing fix computed `60 / rpm_free` — exactly 15 calls/min against a
# 15/min limit, i.e. **100% of the quota, by construction**. That silently
# assumed this workflow is the only consumer. It is not: 42 workflows map
# `secrets.GEMINI_API_KEY_1` and at least a dozen of them run on schedules
# (agent-status-check, company-brain, brain-health, channel-monitor,
# collective-learning, continuous-improvement, …). The quota is fleet-wide.
#
# Any concurrent consumer pushes a 100%-utilisation plan straight back into the
# 429s it was written to prevent. 0.6 leaves 40% for everyone else; at Gemini's
# 15 rpm that is 6.7s spacing and ~5.2 min for 47 employees — still a fifth of
# the 25-minute timeout, so the headroom is nearly free.
_QUOTA_SHARE = float(os.environ.get("EMPLOYEE_QUOTA_SHARE", "0.6") or 0.6)


# Ceiling on the adaptive multiplier. At the Gemini-only base of 6.67s this tops
# out at ~53s between calls — slow, but a run that finishes 20 employees beats
# one that 429s through all 47.
_MAX_BACKOFF = float(os.environ.get("EMPLOYEE_MAX_BACKOFF", "8") or 8)


def _run_budget_seconds() -> float:
    """Wall-clock budget for the employee loop, under the job's own timeout.

    The job allows 25 minutes. Stopping at 18 leaves room to write memory, the
    proof file and the Discord report — a job killed by the runner timeout loses
    all three, so an overrun would present as a crash rather than a partial run.
    """
    try:
        return max(60.0, float(os.environ.get("EMPLOYEE_RUN_BUDGET_S", "1080") or 1080))
    except ValueError:
        return 1080.0


def _keyed_providers() -> list[tuple[str, int]]:
    """(provider name, declared rpm_free) for every provider `_has_key` accepts.

    Exists to answer a question run 30986127682 raised and could not settle: the
    pacer computed 1.0s spacing, which means it saw a provider declaring
    `rpm_free: 60`, while all 101 errors in that log were `[gemini-key]`. Some
    60-rpm key is present and never called — invalid, or not on the path
    `_llm_waterfall` takes — and the log named neither the provider nor the
    disagreement.

    Printing this makes the next run answer it without a code change. Diagnosing
    it took reading a spacing number and inferring backwards; that is exactly
    the kind of inference this repo keeps paying for.
    """
    try:
        from llm_common import _PROVIDERS, _has_key
        return [(p.get("name", "?"), p.get("rpm_free", 0)) for p in _PROVIDERS if _has_key(p)]
    except Exception:  # noqa: BLE001
        return []


def _call_spacing_seconds() -> float:
    """Seconds to wait between employees, derived from the provider's own limit.

    First real run (30983374228, 2026-08-05 07:01) fired all 47 employees in
    ~20 seconds and got **1 response**. The log is one provider repeating:

        [gemini-key] error: HTTP Error 429: Too Many Requests
        [gemini-key] error: HTTP Error 503: Service Unavailable

    The cascade is not broken — it falls through correctly, but every other tier
    is keyless, so Gemini is the whole budget. `llm_common._PROVIDERS` declares
    `"rpm_free": 15` for it. 47 calls in 20s is ~140/min against a 15/min limit;
    the 429s were arithmetic, not an outage.

    `rpm_free` is declared for all eight providers and was enforced NOWHERE — the
    data existed without the behaviour. This derives the interval from that
    table rather than hardcoding a second copy of the number: the interval is
    set by the most permissive provider that actually has a key, since that is
    the one the cascade will settle on.

    At 15 rpm that is 4s -> ~3.1 min for 47 employees, comfortably inside the
    workflow's 25-minute timeout.
    """
    override = os.environ.get("EMPLOYEE_CALL_SPACING_S", "").strip()
    if override:
        try:
            return max(0.0, float(override))
        except ValueError:
            pass
    try:
        from llm_common import _PROVIDERS, _has_key
        rpms = [p.get("rpm_free", 0) for p in _PROVIDERS if _has_key(p)]
        rpms = [r for r in rpms if r and r > 0]
        if rpms:
            return 60.0 / (max(rpms) * _QUOTA_SHARE)
    except Exception:  # noqa: BLE001 — pacing must never break the run
        pass
    return 60.0 / (15 * _QUOTA_SHARE)   # Gemini's free tier, the observed floor


def _save_memory(mem: dict) -> None:
    # `conversations` is unbounded across all three writers of this file; see
    # shared_context.trim_conversations for the measurement.
    try:
        from shared_context import trim_conversations
        trim_conversations(mem)
    except Exception:  # noqa: BLE001 — a size trim must never fail a save
        pass
    MEMORY_FILE.write_text(json.dumps(mem, indent=2, default=str))


def _get_topics(mem: dict) -> list[str]:
    raw = mem.get("daily_topics", {})
    topics: list[str] = []
    if isinstance(raw, dict):
        topics = list(raw.values())
    elif isinstance(raw, list):
        topics = raw
    if not topics:
        topics = _FALLBACK_TOPICS
    return (topics * 10)[:3]  # exactly 3, cycling if needed


def _write_proof(
    conversations: dict,
    provider_dist: dict[str, int],
    date_str: str,
    responded: int,
) -> None:
    lines = [
        f"# QuantEdge Employee Conversations — {date_str}",
        "",
        f"**Employee count:** {responded}/47",
        "",
        "## Provider Distribution",
    ]
    for prov, cnt in sorted(provider_dist.items(), key=lambda x: -x[1]):
        lines.append(f"- {prov}: {cnt}")
    lines += ["", "## Sample Responses"]
    shown = 0
    for _ts, entry in sorted(conversations.items())[-20:]:
        if shown >= 47:
            break
        msg = (entry.get("message") or "")[:150]
        speaker = entry.get("speaker", "?")
        prov = entry.get("provider", "?")
        lines.append(f"\n**{speaker}** _(via {prov})_: {msg}")
        shown += 1
    lines += ["", f"RESPONDED_COUNT={responded}"]
    PROOF_FILE.write_text("\n".join(lines) + "\n")


def main() -> None:
    mem = _load_memory()
    topics = _get_topics(mem)
    emp_keys = list(_EMPLOYEE_PERSONAS.keys())

    # Optional cap (CI smoke / rate-limit budgets): EMPLOYEE_RUNNER_LIMIT=N runs
    # only the first N employees. Unset/<=0 runs the full roster.
    try:
        _limit = int(os.environ.get("EMPLOYEE_RUNNER_LIMIT", "0"))
    except ValueError:
        _limit = 0
    if _limit > 0:
        emp_keys = emp_keys[:_limit]

    state: dict = {}
    responded: list[str] = []
    failed: list[str] = []
    provider_dist: dict[str, int] = {}

    conversations: dict = mem.setdefault("conversations", {})

    spacing = _call_spacing_seconds()
    budget_s = _run_budget_seconds()
    started = time.monotonic()
    backoff = 1.0
    skipped: list[str] = []
    keyed = _keyed_providers()
    print(f"[employee_runner] {len(emp_keys)} employees, {spacing:.1f}s base spacing, "
          f"{budget_s/60:.0f} min budget", flush=True)
    print(f"[employee_runner] providers with a key: "
          f"{', '.join(f'{n}({r}rpm)' for n, r in keyed) or 'NONE'}", flush=True)

    for i, emp_key in enumerate(emp_keys):
        task = topics[i % len(topics)]
        elapsed = time.monotonic() - started
        if elapsed > budget_s:
            # Stop deliberately and SAY SO, rather than running into the job
            # timeout — a killed job loses the memory write and the proof file,
            # so a budget overrun would look like a crash.
            skipped = list(emp_keys[i:])
            print(f"[employee_runner] budget exhausted after {elapsed/60:.1f} min — "
                  f"skipping {len(skipped)} employees: {','.join(skipped[:5])}…",
                  flush=True)
            break
        if i:
            time.sleep(spacing * backoff)   # between calls, not before the first
        try:
            answer, provider = employee_provider_prompt(emp_key, task, state)
            if not answer:
                failed.append(emp_key)
                # ADAPTIVE. Measured 2026-08-05 07:43 (run 30986127682): 47
                # employees, 1.0s spacing, 101 consecutive `[gemini-key] HTTP
                # 429`, 5 responses. The declared rpm_free said there was
                # headroom; the provider said otherwise. Declarations are a
                # starting guess — the 429s are the ground truth, so widen on
                # failure and snap back on success.
                backoff = min(backoff * 2, _MAX_BACKOFF)
                continue
            backoff = 1.0
            ts = datetime.now(timezone.utc).isoformat()
            quality_log = state.get("quality_log", [])
            quality_score = quality_log[-1]["score"] if quality_log else None
            conversations[ts] = {
                "speaker": emp_key,
                "message": answer[:500],
                "provider": provider or "unknown",
                "round": i,
                "quality_score": quality_score,
            }
            provider_dist[provider or "unknown"] = provider_dist.get(provider or "unknown", 0) + 1
            responded.append(emp_key)
        except Exception as exc:
            print(f"[employee_runner] {emp_key} failed: {exc}", flush=True)
            failed.append(emp_key)

    # Persist updated memory
    mem["conversations"] = conversations
    _save_memory(mem)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _write_proof(conversations, provider_dist, date_str, len(responded))

    print(f"RESPONDED_COUNT={len(responded)}")
    # Skipped-for-budget is NOT the same as failed, and folding the two together
    # would misreport a healthy-but-throttled run as 42 broken employees.
    print(f"SKIPPED_COUNT={len(skipped)}")
    if skipped:
        print(f"SKIPPED_EMPLOYEES={','.join(skipped)}")
    if failed:
        print(f"FAILED_EMPLOYEES={','.join(failed)}")
    else:
        print("FAILED_EMPLOYEES=")


if __name__ == "__main__":
    main()
