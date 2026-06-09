"""
Deep Code Review Agent Pool
============================
Dispatches 8 specialist free-LLM agents to review different codebase areas.

Token-efficient v2: uses git diff (not full files) + llm_common shared infrastructure.
Sends only what CHANGED since last review — typically 500-2000 tokens vs 10,000+.
Falls back to file reading only when no diff is available.

Each agent:
  1. Gets git diff for its files since last review
  2. Calls llm_common.llm() (shared cascade + cache + company context)
  3. Writes findings to docs/agent-reviews/ AND shared company brain
  4. Posts summary to Slack

All runs committed. Zero Anthropic tokens.
"""
from __future__ import annotations

import glob as globlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Use shared LLM infrastructure — no more copy-paste cascade
sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm, slack_post, memory_write, core_get

REPO_ROOT = Path(__file__).parent.parent
BRANCH = "claude/advanced-trading-bot-d5Lmw"
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
ALLOW_PAID = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID.lower() == "true":
    sys.exit(1)

REVIEW_DIR = REPO_ROOT / "docs" / "agent-reviews"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

DATE_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Each agent: (name, slack_channel, path_prefixes)
AGENTS = [
    ("strategies", "desk-research",
     ["backend/app/strategies/"]),
    ("ml-models", "ml-research",
     ["backend/app/ml/"]),
    ("execution", "desk-equities",
     ["backend/app/execution/"]),
    ("risk", "risk",
     ["backend/app/risk/"]),
    ("api-backend", "engineering",
     ["backend/app/api/", "backend/app/main.py", "backend/app/config.py"]),
    ("tasks-scheduler", "engineering",
     ["backend/app/tasks/"]),
    ("frontend", "frontend",
     ["frontend/src/"]),
    ("infrastructure", "engineering",
     ["render.yaml", "backend/pyproject.toml", ".github/workflows/"]),
]


def slack(channel: str, msg: str) -> None:
    """Post to Slack — delegates to shared llm_common helper."""
    result = slack_post(f"#{channel}", msg)
    if not result:
        print(f"[Slack #{channel}] {msg[:120]}")


def get_diff_for_paths(paths: list[str], max_chars: int = 6000) -> str:
    """
    Get git diff for specific paths since last review tag or 7 days ago.
    Sending diffs (not full files) cuts token usage by 60-80%.
    """
    # Try diff since last review commit
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--grep=agent-reviews", "-1", "--format=%H"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        base_ref = result.stdout.strip() or "HEAD~50"
    except Exception:
        base_ref = "HEAD~50"

    diff_parts = []
    for path in paths:
        full_path = str(REPO_ROOT / path) if not Path(path).is_absolute() else path
        try:
            result = subprocess.run(
                ["git", "diff", base_ref, "--", full_path],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
            )
            diff = result.stdout.strip()
            if diff:
                diff_parts.append(f"=== diff: {path} ===\n{diff}")
        except Exception:
            pass

    diff_text = "\n\n".join(diff_parts)

    # If no diff (no changes), fall back to reading key files at reduced size
    if not diff_text.strip():
        file_parts = []
        total = 0
        for path in paths:
            pattern = str(REPO_ROOT / path)
            if "*" not in pattern:
                pattern = pattern.rstrip("/") + "/*.py"
            for fpath in sorted(globlib.glob(pattern))[:4]:
                p = Path(fpath)
                if p.exists():
                    content = p.read_text(errors="replace")[:800]  # small slice for no-diff case
                    rel = str(p.relative_to(REPO_ROOT))
                    file_parts.append(f"=== {rel} (no changes) ===\n{content}")
                    total += len(content)
                    if total > 3000:
                        break
        diff_text = "\n\n".join(file_parts)

    return diff_text[:max_chars]


# ── Review prompt ─────────────────────────────────────────────────────────────

REVIEW_PROMPT = """You are a world-class quantitative software engineer at a Two Sigma / Citadel-level firm.
Review the following changes for: {agent_name}

{code}

Generate a focused improvement document:

## Critical Issues (P0/P1)
- Up to 3 bugs/security issues with exact file:line and fix. Skip if none.

## Performance Wins
- Up to 3 concrete speedups or reliability improvements.

## Alpha/Signal Improvements
- Up to 3 ways to improve trading signal quality.

## Implementation Priority
Top 3 changes to implement NOW, one line each.

Be specific. Cite actual symbols/files from the diff above. No generic advice."""


# ── Git ───────────────────────────────────────────────────────────────────────

def git_config() -> None:
    subprocess.run(["git", "config", "user.email", "code-review-agent@quantedge.ai"], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "config", "user.name", "Code Review Agent Pool"], cwd=REPO_ROOT, check=True)


def commit_docs() -> None:
    result = subprocess.run(["git", "add", "docs/agent-reviews/"], cwd=REPO_ROOT)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if r.returncode == 0:
        print("[review] No new docs to commit")
        return
    msg = f"docs(agent-reviews): {DATE_STR} deep review by 8-agent pool"
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
    for delay in [2, 4, 8, 16]:
        result = subprocess.run(["git", "push", "-u", "origin", BRANCH], cwd=REPO_ROOT)
        if result.returncode == 0:
            break
        time.sleep(delay)
    print("[review] Committed and pushed review docs")


# ── Main ──────────────────────────────────────────────────────────────────────

def git_config() -> None:
    subprocess.run(["git", "config", "user.email", "code-review-agent@quantedge.ai"], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "config", "user.name", "Code Review Agent Pool"], cwd=REPO_ROOT, check=True)


def commit_docs() -> None:
    subprocess.run(["git", "add", "docs/agent-reviews/"], cwd=REPO_ROOT)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if r.returncode == 0:
        print("[review] No new docs to commit")
        return
    msg = f"docs(agent-reviews): {DATE_STR} deep review by 8-agent pool"
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
    for delay in [2, 4, 8, 16]:
        result = subprocess.run(["git", "push", "-u", "origin", BRANCH], cwd=REPO_ROOT)
        if result.returncode == 0:
            break
        time.sleep(delay)


def main() -> None:
    slack("engineering",
        f"*Code Review Agent Pool* ({DATE_STR}): "
        f"8 agents reviewing diffs (token-efficient v2). "
        f"Results → `docs/agent-reviews/`. Zero Anthropic tokens.")

    git_config()
    all_findings = []

    for agent_name, channel, paths in AGENTS:
        print(f"\n[review] Agent: {agent_name}")
        code = get_diff_for_paths(paths)

        if not code.strip():
            print(f"[review] {agent_name}: no diff or files found, skipping")
            continue

        prompt = REVIEW_PROMPT.format(agent_name=agent_name, code=code)
        review = llm(
            prompt,
            system=f"You are a senior quant engineer reviewing the {agent_name} module at QuantEdge.",
            max_tokens=800,
            temperature=0.1,
            use_cache=False,  # always fresh review
        )

        if not review or "unavailable" in review:
            print(f"[review] {agent_name}: LLM failed")
            continue

        # Save doc
        doc_path = REVIEW_DIR / f"{agent_name}-{DATE_STR}.md"
        header = (
            f"# {agent_name.replace('-', ' ').title()} Review — {DATE_STR}\n\n"
            f"*Token-efficient diff review. Zero Anthropic tokens.*\n\n---\n\n"
        )
        doc_path.write_text(header + review)
        print(f"[review] {agent_name}: wrote {doc_path.name}")

        # Write to shared company brain so ALL agents learn from this review
        priority_match = re.search(r"## Implementation Priority\s*(.*?)(?=\n##|$)", review, re.DOTALL)
        priority_text = priority_match.group(1).strip()[:400] if priority_match else ""

        memory_write("episodic", {
            "lesson": f"Code review [{agent_name}]: {priority_text[:200]}",
            "category": "code",
            "source": "deep_code_review",
            "agent": agent_name,
        })

        if priority_text:
            all_findings.append(f"*{agent_name}*: {priority_text.splitlines()[0][:100]}")

        critical_count = len(re.findall(r"P0|P1|Critical|CRITICAL", review))
        slack(channel,
            f"*Code Review: {agent_name}* — {critical_count} critical issues\n"
            f"{priority_text[:300] if priority_text else 'See full doc.'}\n"
            f"_Full doc: `docs/agent-reviews/{doc_path.name}`_")

        time.sleep(2)

    # Summary to #engineering with all findings
    if all_findings:
        slack("engineering",
            f"*Daily Review Summary ({DATE_STR})*\n" +
            "\n".join(f"• {f}" for f in all_findings[:8]))

    commit_docs()

    slack("engineering",
        f"✅ *Code Review Agent Pool:* {len(completed)}/{len(AGENTS)} agents completed\n"
        f"Agents: {', '.join(f'`{a}`' for a in completed)}\n"
        f"All docs committed → `docs/agent-reviews/` on `{BRANCH}`\n"
        f"_Zero Anthropic tokens used — powered by free LLM cascade_")


if __name__ == "__main__":
    main()
