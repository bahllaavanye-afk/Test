"""Report which free-LLM providers actually answer, where it can be seen.

`llm_common._record_metric()` already logs every LLM call — 7 call sites — to
`.github/state/llm_metrics.jsonl`. That path is **gitignored**, and no workflow
reads it. So the metrics are written inside the ephemeral runner and thrown
away when the job ends: full observability, collected every run, visible to
nobody.

This publishes the live picture instead of the discarded file:

  * probes each configured provider with a tiny request
  * reports which have keys, which actually answer, and how fast
  * escalates when the cascade is one-deep or dead — a single working provider
    means one rate-limit away from every agent silently degrading

Writes to the GitHub step summary and posts to Discord. No git churn, and no
dependence on a file that does not survive the job.

Run:
    python .github/scripts/llm_cascade_report.py
    python .github/scripts/llm_cascade_report.py --no-probe   # keys only, no calls
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A cascade with fewer than this many working providers is one outage from dark.
HEALTHY_MIN_PROVIDERS = 2


def build_report(status: dict) -> tuple[str, bool]:
    """(markdown report, is_healthy)."""
    providers = status.get("providers", {})
    working = status.get("working", [])

    with_keys = [n for n, v in providers.items() if v.get("has_key")]
    no_keys = [n for n, v in providers.items() if not v.get("has_key")]

    lines = [
        f"### Free-LLM cascade — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"- **Working:** {len(working)} / {len(providers)} — "
        + (", ".join(working) if working else "**NONE**"),
        f"- **Keys configured:** {len(with_keys)} — "
        + (", ".join(with_keys) if with_keys else "none"),
        f"- **No key:** {', '.join(no_keys) if no_keys else 'none'}",
        "",
        "| provider | key | answers | ms | error |",
        "|---|---|---|---|---|",
    ]
    for name, v in sorted(providers.items()):
        lines.append(
            f"| {name} | {'yes' if v.get('has_key') else 'no'} "
            f"| {'yes' if v.get('ok') else 'no'} "
            f"| {v.get('ms', '') or ''} "
            f"| {str(v.get('error', ''))[:60]} |"
        )

    healthy = len(working) >= HEALTHY_MIN_PROVIDERS
    if not working:
        lines += ["", "🚨 **The cascade is DEAD.** No provider answers — every agent "
                      "that needs an LLM is silently degraded."]
    elif not healthy:
        lines += ["", f"⚠️ **The cascade is {len(working)} deep.** One rate-limit or "
                      "outage away from every agent going dark. Add a second key."]
    return "\n".join(lines), healthy


def main() -> int:
    probe = "--no-probe" not in sys.argv
    try:
        from llm_common import cascade_status
    except Exception as exc:  # pragma: no cover - import guard
        print(f"llm_common unavailable: {exc}", flush=True)
        return 0

    try:
        status = cascade_status(probe=probe)
    except Exception as exc:  # noqa: BLE001 — a reporter must never break a run
        print(f"cascade probe failed: {exc}", flush=True)
        return 0

    report, healthy = build_report(status)
    print(report, flush=True)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(report + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"step-summary write failed: {exc}", flush=True)

    working = status.get("working", [])
    if not healthy:
        try:
            import notify
            headline = (
                "🚨 **Free-LLM cascade is DEAD** — no provider answers."
                if not working else
                f"⚠️ **Free-LLM cascade is {len(working)} deep** "
                f"({', '.join(working)}). One rate-limit from every agent going dark."
            )
            notify.post_dedup("infra-alerts", headline, username="QuantEdge LLM Health")
        except Exception as exc:  # noqa: BLE001
            print(f"alert post failed: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
