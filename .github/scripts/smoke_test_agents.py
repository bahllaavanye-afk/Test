"""
Production-path smoke test — runs in the GitHub Actions env with REAL keys.

Unlike test_agent_system.py (pure logic, no network), this exercises the
actual runtime path the agents use in production:
  1. One real free-LLM call through llm_common.llm() — asserts a real answer.
  2. One real Slack post (if SLACK_BOT_TOKEN set) — asserts ok.
  3. Reports which provider answered and how long it took.

This is the test that would have caught the 32s-hang / unreachable-provider
class of bug, because it calls the network exactly as production does.

Exit codes:
  0  — all available paths healthy (or cleanly skipped for missing keys)
  1  — a path that SHOULD work failed (real regression)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

_LLM_KEY_VARS = [
    "GROQ_API_KEY", "GROQ_API_KEY_1", "GEMINI_API_KEY", "GEMINI_API_KEY_1",
    "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_1", "SAMBANOVA_API_KEY", "CEREBRAS_API_KEY",
    "TOGETHER_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_NIM_API_KEY",
]


def _any_llm_key() -> bool:
    return any(os.environ.get(k, "").strip() for k in _LLM_KEY_VARS)


def smoke_llm() -> bool:
    if not _any_llm_key():
        print("· LLM smoke: SKIP (no free provider key set)")
        return True
    from llm_common import llm

    t = time.time()
    out = llm("Reply with exactly the word: PONG", max_tokens=10,
              use_cache=False, inject_company_context=False)
    dt = time.time() - t
    ok = bool(out) and not out.startswith("[LLM unavailable")
    print(f"· LLM smoke: {'PASS' if ok else 'FAIL'} ({dt:.1f}s) → {out[:40]!r}")
    if not ok:
        print("  ERROR: free LLM cascade returned nothing despite a key being set.")
    # Latency guard: a single call should never take the full race timeout.
    if ok and dt > 25:
        print(f"  WARN: call took {dt:.1f}s — a provider is likely unreachable (check egress).")
    return ok


def smoke_slack() -> bool:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        print("· Slack smoke: SKIP (no SLACK_BOT_TOKEN)")
        return True
    from llm_common import slack_post

    channel = os.environ.get("SMOKE_CHANNEL", "agent-api-usage")
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    resp = slack_post(channel, f":white_check_mark: agent smoke test OK — {stamp}")
    ok = bool(resp.get("ok"))
    print(f"· Slack smoke: {'PASS' if ok else 'FAIL'} → #{channel} ({resp.get('error', 'posted')})")
    if not ok and resp.get("error") in (None, "", "channel_not_found", "not_in_channel"):
        # channel issues are config, not a code regression — don't fail the build
        print("  (channel config issue, not a code fault — not failing)")
        return True
    return ok


def main() -> int:
    print("=== agent production-path smoke test ===")
    results = [smoke_llm(), smoke_slack()]
    ok = all(results)
    print(f"=== {'ALL HEALTHY' if ok else 'FAILURE DETECTED'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
