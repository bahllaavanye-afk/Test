"""
Team Lead Issue Generator — Principal engineers open GitHub issues for their teams.

Each team lead (VP Eng, Alpha Director, ML Lead, Risk, Backend, etc.) uses Gemini
to analyse the current codebase state and file actionable GitHub issues for their team.

Runs on schedule (daily 07:00 UTC) and on workflow_dispatch.
Issues are labelled by team and priority, then picked up by free-agent-engineer.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GH_TOKEN = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
GH_REPO = os.environ.get("GH_REPO", "bahllaavanye-afk/Test")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

# ── Team lead definitions ──────────────────────────────────────────────────────

TEAM_LEADS: list[dict] = [
    {
        "role": "vp_eng",
        "name": "VP Engineering",
        "slack_name": "VP-Eng · Alex Chen",
        "team": "backend-team",
        "labels": ["engineering", "agent-fix-needed"],
        "channel": "engineering",
        "focus": (
            "You own backend reliability, CI/CD health, and platform architecture. "
            "Your job is to find and file issues for things that are broken, missing, "
            "or need improvement in: FastAPI endpoints, SQLAlchemy models, APScheduler jobs, "
            "GitHub Actions workflows, test coverage, and deployment reliability."
        ),
    },
    {
        "role": "alpha_dir",
        "name": "Alpha Research Director",
        "slack_name": "Alpha-Dir · Sofia Karlsson",
        "team": "equities-desk",
        "labels": ["strategy", "alpha-research", "agent-fix-needed"],
        "channel": "desk-equities",
        "focus": (
            "You own the equities alpha research desk. File issues for: "
            "missing backtests, strategies without walk-forward validation, "
            "signals that need regime-awareness, missing factor exposures, "
            "and any strategies that lack DEFAULT_PARAMS or unit tests."
        ),
    },
    {
        "role": "ml_lead",
        "name": "ML Lead",
        "slack_name": "ML-Lead · Kai Zhang",
        "team": "ml-team",
        "labels": ["ml", "crypto", "agent-fix-needed"],
        "channel": "ml-research",
        "focus": (
            "You own the ML pipeline and crypto desk. File issues for: "
            "models that lack OOS validation, experiment configs that need sweeps, "
            "missing feature engineering for crypto signals (funding rates, OI), "
            "ensemble weights that need reoptimization, and training scripts that "
            "need GPU notebook versions."
        ),
    },
    {
        "role": "cro",
        "name": "Chief Risk Officer",
        "slack_name": "CRO · Marcus Olufemi",
        "team": "risk-team",
        "labels": ["risk", "compliance", "agent-fix-needed"],
        "channel": "risk",
        "focus": (
            "You own firm-wide risk. File issues for: "
            "missing position size caps, strategies without stop-loss defaults, "
            "missing correlation checks between strategies, "
            "circuit breaker gaps, and any bot that doesn't create Trade records at TP/SL."
        ),
    },
    {
        "role": "frontend",
        "name": "Frontend Lead",
        "slack_name": "Frontend-Lead · Priya Subramanian",
        "team": "frontend-team",
        "labels": ["frontend", "ui", "agent-fix-needed"],
        "channel": "engineering",
        "focus": (
            "You own the Bloomberg-dark trading dashboard. File issues for: "
            "pages that render mock/static data, missing loading skeletons, "
            "TypeScript errors, missing TanStack Query hooks, WebSocket disconnections "
            "not handled gracefully, and missing mobile responsiveness."
        ),
    },
    {
        "role": "poly_desk",
        "name": "Polymarket Desk Lead",
        "slack_name": "Poly-Desk · Lior Avraham",
        "team": "polymarket-desk",
        "labels": ["polymarket", "strategy", "agent-fix-needed"],
        "channel": "desk-polymarket",
        "focus": (
            "You own the Polymarket and prediction-market desk. File issues for: "
            "markets not being scanned for YES+NO < $0.97, missing calibration arb vs Metaculus, "
            "late-resolution opportunities not tracked, Kelly fraction not applied to bets, "
            "and missing cross-platform arb vs Kalshi/Manifold."
        ),
    },
    {
        "role": "exec_eng",
        "name": "Execution Engineer",
        "slack_name": "Exec-Eng · Diego Ramirez",
        "team": "execution-team",
        "labels": ["execution", "slippage", "agent-fix-needed"],
        "channel": "desk-equities",
        "focus": (
            "You own order routing, TWAP/VWAP execution, and slippage minimization. "
            "File issues for: missing slippage tracking in Trade records, "
            "RL execution agent not trained, TWAP not slicing orders above $10k threshold, "
            "and fill quality not being measured and reported."
        ),
    },
    {
        "role": "devops_dir",
        "name": "DevOps Director",
        "slack_name": "DevOps-Dir · Kenji Watanabe",
        "team": "devops-team",
        "labels": ["devops", "ci-cd", "agent-fix-needed"],
        "channel": "incidents",
        "focus": (
            "You own GitHub Actions, deployment pipelines, and infrastructure. "
            "File issues for: workflows missing continue-on-error, missing health checks, "
            "Render deployment not auto-restarting on failure, missing UptimeRobot keep-alive, "
            "and any workflow that hasn't run successfully in the last 24h."
        ),
    },
]


# ── Codebase context gathering ─────────────────────────────────────────────────

def gather_codebase_context() -> str:
    """Collect key facts about the codebase for the team lead to reason about."""
    ctx_parts: list[str] = []

    # Git log
    try:
        log = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        ).stdout.strip()
        ctx_parts.append(f"Recent commits:\n{log}")
    except Exception:
        pass

    # Open GitHub issues
    try:
        issues = _github_api("GET", f"/repos/{GH_REPO}/issues?state=open&per_page=20")
        if isinstance(issues, list):
            issue_lines = [f"  #{i['number']}: {i['title'][:70]}" for i in issues[:10]]
            ctx_parts.append(f"Open issues ({len(issues)}):\n" + "\n".join(issue_lines))
    except Exception:
        pass

    # Strategy count
    try:
        strategies = list((REPO_ROOT / "backend" / "app" / "strategies" / "manual").glob("*.py"))
        ctx_parts.append(f"Manual strategies: {len(strategies)}")
    except Exception:
        pass

    # Test count
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "--collect-only", "-q", "--no-header"],
            capture_output=True, text=True, cwd=REPO_ROOT / "backend", timeout=30,
        )
        lines = [l for l in result.stdout.splitlines() if "test session" not in l and l.strip()]
        ctx_parts.append(f"Tests collected: {len(lines)} items")
    except Exception:
        pass

    # Workflow health
    try:
        wf_dir = REPO_ROOT / ".github" / "workflows"
        wf_count = len(list(wf_dir.glob("*.yml")))
        ctx_parts.append(f"GitHub Actions workflows: {wf_count}")
    except Exception:
        pass

    # Experiment results
    try:
        results = list((REPO_ROOT / "experiments" / "results").glob("*.json"))
        ctx_parts.append(f"Experiment result files: {len(results)}")
    except Exception:
        pass

    return "\n\n".join(ctx_parts)


# ── LLM calls ─────────────────────────────────────────────────────────────────

def call_gemini(system_prompt: str, user_message: str) -> str | None:
    if not GEMINI_API_KEY:
        return None
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"maxOutputTokens": 600, "temperature": 0.7},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[gemini] {e}")
        return None


def call_groq(system_prompt: str, user_message: str) -> str | None:
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 600, "temperature": 0.7,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[groq] {e}")
        return None


# ── GitHub API ────────────────────────────────────────────────────────────────

def _github_api(method: str, path: str, body: dict | None = None):
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()[:300]
        print(f"GitHub API error {e.code}: {body_text}")
        return None
    except Exception as e:
        print(f"GitHub API exception: {e}")
        return None


def get_existing_issue_titles() -> set[str]:
    """Get titles of open issues to avoid duplicates."""
    issues = _github_api("GET", f"/repos/{GH_REPO}/issues?state=open&per_page=100")
    if not isinstance(issues, list):
        return set()
    return {i["title"].lower() for i in issues}


def create_issue(title: str, body: str, labels: list[str]) -> dict | None:
    """Create a GitHub issue with the given title, body, and labels."""
    return _github_api("POST", f"/repos/{GH_REPO}/issues", {
        "title": title,
        "body": body,
        "labels": labels,
    })


# ── Slack notification ─────────────────────────────────────────────────────────

def slack_post(channel: str, text: str, username: str) -> None:
    if not SLACK_TOKEN:
        return
    url = "https://slack.com/api/chat.postMessage"
    payload = {
        "channel": channel,
        "text": text,
        "username": username,
        "mrkdwn": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                print(f"[slack] {result.get('error')}")
    except Exception as e:
        print(f"[slack] {e}")


# ── Issue generation per team lead ─────────────────────────────────────────────

ISSUE_SYSTEM_PROMPT = """\
You are a principal engineer / team lead at QuantEdge, an institutional-grade \
quantitative trading platform (FastAPI + React + PyTorch + Alpaca/Binance/Polymarket).

Your job: analyse the codebase context below and generate EXACTLY 3 GitHub issues \
for your team to work on. Each issue must be:
1. Specific and actionable — cite exact file names, function names, or endpoints
2. Implementable by an AI agent reading only the issue text
3. Not a duplicate of any existing open issue
4. Labelled with your team and 'agent-fix-needed'

Output ONLY valid JSON, no prose:
[
  {
    "title": "concise imperative title, max 80 chars",
    "body": "## Problem\\n...\\n\\n## Fix\\n...\\n\\n## Files\\n- `path/to/file.py`",
    "priority": "P1|P2|P3"
  },
  ...
]
"""


def generate_issues_for_lead(lead: dict, context: str, existing_titles: set[str]) -> list[dict]:
    """Call Gemini to generate 3 issues for this team lead's domain."""
    user_msg = (
        f"You are {lead['name']} at QuantEdge. {lead['focus']}\n\n"
        f"Codebase state:\n{context}\n\n"
        f"Existing open issues (don't duplicate):\n"
        + "\n".join(f"- {t}" for t in list(existing_titles)[:20])
        + "\n\nGenerate 3 actionable GitHub issues for your team."
    )

    raw = call_gemini(ISSUE_SYSTEM_PROMPT, user_msg)
    if not raw:
        raw = call_groq(ISSUE_SYSTEM_PROMPT, user_msg)
    if not raw:
        return []

    # Extract JSON array from response
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        print(f"[{lead['name']}] No JSON array found in response")
        return []
    try:
        issues = json.loads(match.group())
        return [i for i in issues if isinstance(i, dict) and "title" in i and "body" in i]
    except json.JSONDecodeError as e:
        print(f"[{lead['name']}] JSON parse error: {e}")
        return []


def run_team_lead(lead: dict, context: str, existing_titles: set[str], dry_run: bool = False) -> list[str]:
    """Run a single team lead, create their issues, post to Slack. Returns issue URLs."""
    print(f"\n{'='*55}")
    print(f"Team Lead: {lead['name']} → #{lead['channel']}")
    print(f"{'='*55}")

    proposals = generate_issues_for_lead(lead, context, existing_titles)
    if not proposals:
        print(f"  ⚠️  No issues generated (LLM returned empty)")
        return []

    print(f"  Generated {len(proposals)} issue proposals")
    created_urls: list[str] = []
    created_titles: list[str] = []

    for prop in proposals:
        title = prop.get("title", "").strip()
        body = prop.get("body", "").strip()
        priority = prop.get("priority", "P2")

        if not title or len(title) < 10:
            continue

        # Dedup check
        if title.lower() in existing_titles:
            print(f"  ⏭  Skipped (duplicate): {title[:60]}")
            continue

        print(f"  📋 [{priority}] {title[:65]}")

        if not dry_run:
            labels = lead["labels"] + [priority.lower()]
            result = create_issue(
                title=f"[{lead['team']}] {title}",
                body=(
                    f"*Filed by:* {lead['name']} (AI team lead)\n"
                    f"*Team:* {lead['team']}\n"
                    f"*Priority:* {priority}\n"
                    f"*Model:* Gemini 2.0 Flash\n\n"
                    f"---\n\n{body}\n\n"
                    f"---\n"
                    f"_Auto-created by `team_lead_issues.py`. "
                    f"Label `agent-fix-needed` triggers Free-Agent Engineer to auto-fix._"
                ),
                labels=labels,
            )
            if result and result.get("html_url"):
                url = result["html_url"]
                print(f"  ✅ Created: {url}")
                created_urls.append(url)
                created_titles.append(title)
                existing_titles.add(title.lower())
                time.sleep(0.5)  # rate limit

    # Post summary to Slack
    if created_titles:
        issue_list = "\n".join(f"• {t}" for t in created_titles)
        slack_text = (
            f"*{lead['slack_name']}* filed {len(created_titles)} new issues for the {lead['team']}:\n"
            f"{issue_list}\n\n"
            f"_These are queued for agent auto-fix (label: agent-fix-needed)_"
        )
        slack_post(lead["channel"], slack_text, username=lead["slack_name"])
        print(f"  📣 Posted to #{lead['channel']}")

    return created_urls


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="", help="Single team role to run (e.g. vp_eng)")
    parser.add_argument("--dry-run", action="store_true", help="Generate issues but don't create them")
    args = parser.parse_args()

    if not GEMINI_API_KEY and not GROQ_API_KEY:
        print("ERROR: Set GEMINI_API_KEY or GROQ_API_KEY")
        sys.exit(1)

    if not GH_TOKEN:
        print("ERROR: Set GH_TOKEN / GITHUB_TOKEN")
        sys.exit(1)

    print(f"{'='*55}")
    print(f"Team Lead Issue Generator — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*55}")

    context = gather_codebase_context()
    print(f"\nContext gathered ({len(context)} chars)")

    existing_titles = get_existing_issue_titles()
    print(f"Existing open issues: {len(existing_titles)}")

    leads = [l for l in TEAM_LEADS if not args.team or l["role"] == args.team]
    total_created = 0

    for lead in leads:
        urls = run_team_lead(lead, context, existing_titles, dry_run=args.dry_run)
        total_created += len(urls)
        time.sleep(2)  # be gentle with Gemini rate limits

    print(f"\n{'='*55}")
    print(f"Total issues created: {total_created}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
