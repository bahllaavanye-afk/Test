"""Run real LLM-backed conversations for every QuantEdge employee persona."""
from __future__ import annotations

import json
import os
import sys
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

    for i, emp_key in enumerate(emp_keys):
        task = topics[i % len(topics)]
        try:
            answer, provider = employee_provider_prompt(emp_key, task, state)
            if not answer:
                failed.append(emp_key)
                continue
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
    if failed:
        print(f"FAILED_EMPLOYEES={','.join(failed)}")
    else:
        print("FAILED_EMPLOYEES=")


if __name__ == "__main__":
    main()
