"""
Deep Code Review — Employee-Led, Provider-Pinned
=================================================
Each domain is reviewed by the responsible employee at QuantEdge.
Every employee uses a DIFFERENT pinned LLM provider for full independence.

Employee → Provider assignment:
  1. alpha_dir      (Alpha Research Director) → gemini        → strategies
  2. ml_lead        (ML Modeling Lead)         → sambanova     → ml-models
  3. exec_eng       (Execution Engineer)        → cerebras      → execution
  4. risk_eng       (Risk Engineer)             → groq          → risk
  5. backend_lead   (Backend Lead)              → deepseek      → api-backend
  6. devops_dir     (Director of DevOps)        → together      → tasks-scheduler
  7. frontend_lead  (VP Frontend)               → hyperbolic    → frontend
  8. vp_eng         (VP of Engineering)         → nvidia_nim    → infrastructure

Each employee:
  1. Gets git diff for their domain paths since last review
  2. Reviews through their own expertise lens (persona from slack_agent_team.py)
  3. Uses a pinned independent LLM (different per employee → zero cross-contamination)
  4. Writes their own report to docs/agent-reviews/<domain>-<date>-<employee>.md
  5. Posts findings to their Slack channel
  6. Writes priority actions to shared company brain

CRO (Chief Risk Officer) synthesizes all 8 reports into system health verdict.
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

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm_with_provider, slack_post, memory_write, core_get

# Employee personas — imported for review system identity
try:
    from slack_agent_team import _EMPLOYEE_PERSONAS as _EMP_PERSONAS  # type: ignore
except Exception:
    _EMP_PERSONAS: dict = {}

REPO_ROOT = Path(__file__).parent.parent
BRANCH = "main"
ALLOW_PAID = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID.lower() == "true":
    sys.exit(1)

REVIEW_DIR = REPO_ROOT / "docs" / "agent-reviews"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

DATE_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TIME_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# Each agent: (domain, slack_channel, path_prefixes, provider_name, emp_key, focus_instruction)
# emp_key = the QuantEdge employee responsible for this domain (from slack_agent_team.py personas)
AGENTS: list[tuple[str, str, list[str], str, str, str]] = [
    (
        "strategies",
        "desk-research",
        ["backend/app/strategies/"],
        "gemini",
        "alpha_dir",
        "Focus on: signal correctness, lookahead bias, entry/exit logic, strategy edge validity, "
        "cross-symbol interaction bugs. Check that momentum/reversion/RSI signals don't peek at future data.",
    ),
    (
        "ml-models",
        "ml-research",
        ["backend/app/ml/"],
        "sambanova",
        "ml_lead",
        "Focus on: data leakage between train/val/test splits, model architecture correctness, "
        "feature engineering for temporal validity, SSM/LSTM/ensemble weight bugs, overfitting indicators.",
    ),
    (
        "execution",
        "desk-equities",
        ["backend/app/execution/"],
        "cerebras",
        "exec_eng",
        "Focus on: TWAP/VWAP slice timing, limit-first fallback races, RL execution policy correctness, "
        "slippage tracking accuracy, order cancellation edge cases, broker API error handling.",
    ),
    (
        "risk",
        "risk",
        ["backend/app/risk/"],
        "groq",
        "risk_eng",
        "Focus on: Kelly fraction calculation, CVaR tail risk accuracy, HRP weight sum = 1, "
        "circuit breaker trigger conditions, drawdown accounting correctness, correlation cluster logic.",
    ),
    (
        "api-backend",
        "engineering",
        ["backend/app/api/", "backend/app/main.py", "backend/app/config.py"],
        "deepseek",
        "backend_lead",
        "Focus on: JWT auth coverage on all endpoints, SQLAlchemy ORM-only (no raw SQL), "
        "Pydantic v2 validation gaps, CORS misconfiguration, rate limiting bypasses, input sanitization.",
    ),
    (
        "tasks-scheduler",
        "engineering",
        ["backend/app/tasks/"],
        "together",
        "devops_dir",
        "Focus on: APScheduler job overlaps, asyncio race conditions in strategy_runner, "
        "Redis write contention, brain file locking, price feed reconnection logic, "
        "strategy gate enforcement correctness.",
    ),
    (
        "frontend",
        "frontend",
        ["frontend/src/"],
        "hyperbolic",
        "frontend_lead",
        "Focus on: TypeScript type safety, TanStack Query stale data races, WebSocket reconnect handling, "
        "React key prop issues in lists, TradingView widget lifecycle, "
        "tearsheet calculation correctness, missing loading/error states.",
    ),
    (
        "infrastructure",
        "engineering",
        [".github/workflows/", "backend/pyproject.toml", "render.yaml"],
        "nvidia_nim",
        "vp_eng",
        "Focus on: GitHub Actions secret injection security, workflow trigger overlaps, "
        "dependency version pinning, Docker layer caching, missing ALLOW_PAID_APIS guards, "
        "ANTHROPIC_API_KEY disabled enforcement, TRADING_MODE=paper enforcement.",
    ),
]

REVIEW_PROMPT = """\
You are {emp_role} at QuantEdge. This is YOUR domain — you own it.
Write your independent deep-dive code review. No other reviewer sees what you write.

Domain: {domain}
{focus}

## Code / Changes Under Review
{code}

---

Write your full independent review with these sections:

### Critical Issues (P0/P1)
For each issue: exact file:line, what the bug is, why it matters, exact fix.
Max 4 items. Write "None found" if clean.

### Performance & Reliability Improvements
Concrete, actionable. Cite actual code. Max 3 items.

### Alpha / Signal Quality
Ways the trading logic could be improved. Max 3 items.

### Security & Safety
Auth gaps, injection risks, accidental live trading exposure. Max 2 items.

### Implementation Priority Queue
Ordered list of top 5 actions to take RIGHT NOW, with estimated impact.

### Overall Grade
Letter grade A-F with one sentence rationale.

Be surgical. Quote actual code. No generic advice. You are the best reviewer in finance."""


def get_diff_for_paths(paths: list[str], max_chars: int = 7000) -> str:
    """Get git diff since last review commit. Falls back to file snippets."""
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

    # No diff → read current files (small slice to stay within token budget)
    if not diff_text.strip():
        file_parts = []
        total = 0
        for path in paths:
            pattern = str(REPO_ROOT / path)
            if "*" not in pattern and not pattern.endswith(".py") and not pattern.endswith(".toml"):
                pattern = pattern.rstrip("/") + "/*.py"
            for fpath in sorted(globlib.glob(pattern))[:5]:
                p = Path(fpath)
                if p.exists() and p.suffix in (".py", ".ts", ".tsx", ".yml", ".toml"):
                    content = p.read_text(errors="replace")[:1200]
                    rel = str(p.relative_to(REPO_ROOT))
                    file_parts.append(f"=== {rel} (full, no recent changes) ===\n{content}")
                    total += len(content)
                    if total > 5000:
                        break
        diff_text = "\n\n".join(file_parts)

    return diff_text[:max_chars]


def git_setup() -> None:
    subprocess.run(
        ["git", "config", "user.email", "code-review-agent@quantedge.ai"],
        cwd=REPO_ROOT, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Code Review Agent Pool"],
        cwd=REPO_ROOT, check=True,
    )


def commit_and_push() -> None:
    subprocess.run(["git", "add", "docs/agent-reviews/"], cwd=REPO_ROOT)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if r.returncode == 0:
        print("[review] No new docs to commit")
        return
    msg = f"docs(agent-reviews): {DATE_STR} independent deep review — 8 agents"
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
    for delay in [2, 4, 8, 16]:
        result = subprocess.run(["git", "push", "-u", "origin", BRANCH], cwd=REPO_ROOT)
        if result.returncode == 0:
            print("[review] Pushed.")
            break
        print(f"[review] Push failed, retrying in {delay}s…")
        time.sleep(delay)


def main() -> None:
    print(f"[review] Employee Deep Code Review — {TIME_STR}", flush=True)
    print(f"[review] {len(AGENTS)} employees, each reviewing their domain via an independent LLM", flush=True)

    slack_post("#engineering",
        f"*Employee Deep Code Review* ({DATE_STR})\n"
        f"8 employees reviewing their own domains (each pinned to a different LLM):\n"
        + "\n".join(
            f"  • `{emp}` reviews `{d}` via _{p}_"
            for d, _, _, p, emp, _ in AGENTS
        ))

    git_setup()

    completed: list[str] = []
    all_priority_lines: list[str] = []
    all_grades: dict[str, str] = {}

    for domain, channel, paths, provider, emp_key, focus in AGENTS:
        # Look up employee persona — the review comes FROM this employee
        emp_persona = _EMP_PERSONAS.get(emp_key, "")
        emp_role = emp_key.replace("_", " ").title()
        if emp_persona:
            # Extract the "You are the X at QuantEdge" part for the prompt header
            role_match = re.match(r"You are ([^.]+)\.", emp_persona)
            emp_role = role_match.group(1) if role_match else emp_role

        print(f"\n[review] {domain} → {emp_key} → {provider}", flush=True)
        code = get_diff_for_paths(paths)

        if not code.strip():
            print(f"[review] {domain}: no code found, skipping")
            continue

        prompt = REVIEW_PROMPT.format(
            domain=domain,
            emp_role=emp_role,
            focus=f"Special focus for this review:\n{focus}",
            code=code,
        )

        # System prompt = employee persona (their expertise lens) + code reviewer identity
        system_prompt = (
            emp_persona
            + f"\n\nFor this task, you are performing a deep independent code review of the "
            f"{domain} domain. Cite exact file paths and line numbers. Be surgical."
            if emp_persona
            else (
                f"You are a senior quant engineer specializing in {domain}. "
                f"Write an independent deep-dive review. Cite exact code."
            )
        )

        review, used_provider = llm_with_provider(
            prompt=prompt,
            provider_name=provider,
            system=system_prompt,
            max_tokens=900,
            temperature=0.15,
            inject_company_context=True,   # all employees receive shared brain/memory context
        )

        if not review or "unavailable" in review.lower():
            print(f"[review] {domain}: LLM failed (provider={used_provider})")
            continue

        # Extract grade
        grade_match = re.search(r"###\s*Overall Grade\s*\n+([A-F][+-]?)", review)
        grade = grade_match.group(1) if grade_match else "?"
        all_grades[domain] = grade

        # Extract priority queue
        priority_match = re.search(
            r"###\s*Implementation Priority Queue\s*(.*?)(?=\n###|$)", review, re.DOTALL
        )
        priority_text = priority_match.group(1).strip()[:500] if priority_match else ""

        # Save independent report — filename includes employee key so it's clear whose report
        doc_path = REVIEW_DIR / f"{domain}-{DATE_STR}-{emp_key}.md"
        header = (
            f"# {domain.replace('-', ' ').title()} — Employee Deep Review\n"
            f"**Date:** {DATE_STR}  |  **Employee:** `{emp_key}` ({emp_role})  |  "
            f"**LLM:** {used_provider}  |  **Grade:** {grade}\n\n"
            f"_This review was written by {emp_key} ({emp_role}) using {used_provider}, "
            f"independently of all other employees' reports._\n\n---\n\n"
        )
        doc_path.write_text(header + review)
        print(f"[review] {domain}: wrote {doc_path.name} | employee={emp_key} | grade={grade}")

        # Store in brain for all employees to learn from
        memory_write("episodic", {
            "lesson": f"Code review [{domain}] by {emp_key} (grade:{grade}): {priority_text[:200]}",
            "category": "code",
            "source": "deep_code_review",
            "domain": domain,
            "employee": emp_key,
            "provider": used_provider,
            "grade": grade,
        })

        if priority_text:
            first_line = priority_text.splitlines()[0].lstrip("0123456789.-) ").strip()
            all_priority_lines.append(
                f"*{domain}* [{emp_key}, grade {grade}]: {first_line[:100]}"
            )

        # Count critical issues
        p0_count = len(re.findall(r"P0|P1|Critical|CRITICAL", review))
        slack_post(f"#{channel}",
            f"*Code Review: {domain}* [Grade: {grade}]\n"
            f"Reviewed by: `{emp_key}` ({emp_role}) via _{used_provider}_\n"
            f"{p0_count} critical issue(s) flagged.\n"
            + (f"Top priority:\n{priority_text[:350]}\n" if priority_text else "")
            + f"_Full report: `docs/agent-reviews/{doc_path.name}`_")

        completed.append(f"{domain}({emp_key},{grade})")
        time.sleep(3)   # brief pause between providers

    # CRO synthesizes all 8 employee reports into system health verdict
    if all_grades:
        grades_str = " | ".join(f"{d}[{g}]" for d, g in all_grades.items())
        cro_persona = _EMP_PERSONAS.get("cro", "You are the Chief Risk Officer at QuantEdge.")
        synthesis_prompt = (
            f"8 QuantEdge employees independently reviewed their domains and assigned grades:\n"
            f"{grades_str}\n\n"
            "As CRO, summarize in 3 sentences: overall system health, the domain needing most urgent attention, "
            "and the single highest-risk item across all domains."
        )
        synthesis, _ = llm_with_provider(
            synthesis_prompt, "gemini",
            system=cro_persona,
            max_tokens=250, temperature=0.2,
            inject_company_context=True,
        )
        if synthesis and "unavailable" not in synthesis.lower():
            slack_post("#engineering",
                f"*CRO System Health Synthesis* ({DATE_STR})\n"
                f"Grades: {grades_str}\n\n{synthesis}")
            memory_write("episodic", {
                "lesson": f"CRO system health: {synthesis[:300]}",
                "category": "code",
                "source": "deep_code_review_cro_synthesis",
            })

    # Summary of all priority items
    if all_priority_lines:
        slack_post("#engineering",
            f"*Daily Deep Review Summary ({DATE_STR}) — Top Priorities*\n"
            + "\n".join(f"• {line}" for line in all_priority_lines[:8]))

    commit_and_push()

    slack_post("#engineering",
        f"✅ *Employee Deep Code Review complete* — {len(completed)}/{len(AGENTS)} employees\n"
        f"Completed: {', '.join(f'`{a}`' for a in completed)}\n"
        f"All reports committed → `docs/agent-reviews/` on `{BRANCH}`\n"
        f"_Each employee used a different LLM. Zero Anthropic tokens. Full company brain context injected._")

    print(f"\n[review] Done. {len(completed)}/{len(AGENTS)} completed.", flush=True)


if __name__ == "__main__":
    main()
