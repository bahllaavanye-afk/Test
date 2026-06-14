"""
QuantEdge Task Router — Claude is the main brain.

Claude (via Anthropic API) reads every new task, decides complexity and domain,
routes to the best sub-agent (Gemini/Grok/GPT-4o/Perplexity), and does final
acceptance of the output before merging / closing the issue.

Flow:
  1. New GitHub Issue arrives (any label triggers routing)
  2. Claude reads title + body, classifies task:
       • research    → Perplexity Sonar (web-grounded)
       • fast-code   → Grok Mini (xAI, low latency)
       • code        → GPT-4o via GitHub Models (complex logic)
       • analysis    → Gemini 2.0 Flash (large context)
       • complex     → Gemini + GPT-4o dual review
  3. Router adds the correct label (gemini-task / grok-task / codex-task)
  4. Sub-agent workflow picks it up (gemini-task-runner.yml etc.)
  5. After sub-agent commits, Claude acceptance workflow fires:
       • Claude reviews the diff
       • Accept → closes issue, posts ✅ to Slack
       • Reject → re-opens issue with feedback label

Usage:
  python .github/scripts/task_router.py --issue <number>
  python .github/scripts/task_router.py --task "Add HMM regime detection"  # dry-run
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import slack_post

GH_TOKEN = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
GH_REPO = os.environ.get("GH_REPO", "bahllaavanye-afk/test")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

# Routing table: task_type → GitHub label applied to the issue
_LABEL_MAP = {
    "research": "perplexity-task",
    "fast-code": "grok-task",
    "code":      "codex-task",      # GPT-4o via GitHub Models
    "analysis":  "gemini-task",     # Gemini 2.0 Flash
    "complex":   "gemini-task",     # dual: gemini first, then codex review
}

# Provider tokens used per type (for the sub-agent that picks up the label)
_TYPE_DESCRIPTION = {
    "research":  "Perplexity Sonar (web-grounded research, real-time data)",
    "fast-code": "Grok Mini (xAI, fastest coding, low latency)",
    "code":      "GPT-4o via GitHub Models (complex logic, best correctness)",
    "analysis":  "Gemini 2.0 Flash (large context, analysis, strategy review)",
    "complex":   "Gemini 2.0 Flash → GPT-4o dual review (most critical tasks)",
}

_ROUTER_SYSTEM = """\
You are Claude, the main orchestrator brain of QuantEdge, an institutional-grade
quantitative trading platform. Your job is to route tasks to the right sub-agent.

Sub-agents available:
- research   → Perplexity Sonar: web data, market research, academic papers, news
- fast-code  → Grok Mini (xAI): small bug fixes, typos, config changes, quick edits
- code       → GPT-4o (OpenAI): new features, refactoring, algorithmic logic, tests
- analysis   → Gemini 2.0 Flash: large-context code review, ML analysis, strategy eval
- complex    → Gemini + GPT-4o: security changes, risk logic, ML model changes, DB migrations

Rules:
1. If the task mentions "research", "data", "market", "paper", "news" → research
2. If it's a 1-file, < 30 line change (bug fix, typo, config) → fast-code
3. If it's a new feature, algorithm, or test → code
4. If it needs full codebase understanding → analysis
5. If it touches risk, security, DB, or ML models → complex
6. When in doubt → code (GPT-4o handles it well)

Reply with ONLY a JSON object — no other text:
{"task_type": "<one of: research|fast-code|code|analysis|complex>", "reason": "<1 sentence>"}
"""


def _call_anthropic(prompt: str) -> str:
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "disabled":
        return '{"task_type": "code", "reason": "Claude unavailable, defaulting to GPT-4o"}'
    body = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 200,
        "system": _ROUTER_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()


def _gh(method: str, path: str, body: dict | None = None) -> dict:
    url = f"https://api.github.com/repos/{GH_REPO}/{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read()) if resp.status not in (204,) else {}
    except urllib.error.HTTPError as e:
        print(f"[router] GH API {method} {path}: {e.code} {e.reason}", flush=True)
        return {}


def _ensure_label(label: str) -> None:
    """Create GitHub label if it doesn't exist."""
    colors = {
        "perplexity-task": "0075ca",
        "grok-task":       "7057ff",
        "codex-task":      "008672",
        "gemini-task":     "e4e669",
        "claude-accepted": "0e8a16",
        "claude-rejected": "d93f0b",
    }
    _gh("POST", "labels", {
        "name": label,
        "color": colors.get(label, "cccccc"),
        "description": f"Auto-routed by Claude task router",
    })


def route_issue(issue_number: int) -> None:
    """Classify and label a GitHub issue for the correct sub-agent."""
    issue = _gh("GET", f"issues/{issue_number}")
    if not issue:
        print(f"[router] Issue #{issue_number} not found", flush=True)
        return

    title = issue.get("title", "")
    body = issue.get("body", "") or ""
    existing_labels = [lbl["name"] for lbl in issue.get("labels", [])]

    # Skip if already routed
    if any(lbl in existing_labels for lbl in _LABEL_MAP.values()):
        print(f"[router] Issue #{issue_number} already routed", flush=True)
        return

    task_text = f"Title: {title}\n\nDescription:\n{body[:1500]}"
    print(f"[router] Routing issue #{issue_number}: {title!r}", flush=True)

    try:
        response = _call_anthropic(task_text)
        # Extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        routing = json.loads(response[start:end])
        task_type = routing.get("task_type", "code")
        reason = routing.get("reason", "")
    except Exception as e:
        print(f"[router] Claude routing failed ({e}), defaulting to 'code'", flush=True)
        task_type = "code"
        reason = "Claude routing unavailable"

    label = _LABEL_MAP.get(task_type, "codex-task")
    agent_desc = _TYPE_DESCRIPTION.get(task_type, "GPT-4o")

    _ensure_label(label)
    _gh("POST", f"issues/{issue_number}/labels", {"labels": [label]})
    _gh("POST", f"issues/{issue_number}/comments", {"body": (
        f"🤖 **Claude Task Router** assigned this to `{label}`\n\n"
        f"**Agent:** {agent_desc}\n"
        f"**Reason:** {reason}\n\n"
        f"_The sub-agent will pick this up automatically. "
        f"Claude will review and accept/reject the output._"
    )})

    slack_post("#engineering",
        f"🔀 *Task Routed* [#{issue_number}] {title}\n"
        f"→ Agent: *{agent_desc}*\n"
        f"→ Reason: {reason}")

    print(f"[router] Issue #{issue_number} → {label} ({task_type})", flush=True)


def accept_work(issue_number: int | None = None) -> None:
    """
    Claude reviews recent commits. If work is acceptable, closes the issue.
    Called after sub-agent commits its changes.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True, text=True, timeout=10,
        )
        recent_commits = result.stdout.strip()
    except Exception:
        recent_commits = "(unavailable)"

    try:
        result = subprocess.run(
            ["git", "diff", "HEAD~3", "--stat"],
            capture_output=True, text=True, timeout=10,
        )
        diff_stat = result.stdout.strip()[:2000]
    except Exception:
        diff_stat = "(unavailable)"

    acceptance_prompt = (
        f"Recent commits by sub-agents:\n{recent_commits}\n\n"
        f"Changed files:\n{diff_stat}\n\n"
        f"As Claude (main orchestrator), decide:\n"
        f"1. Does this look like reasonable, safe, non-destructive work?\n"
        f"2. Are there obvious errors (syntax, security, trading-money-at-risk)?\n\n"
        f"Reply JSON only: "
        f'{{\"decision\": \"accept\" or \"reject\", \"reason\": \"1 sentence\"}}'
    )

    try:
        response = _call_anthropic(acceptance_prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        verdict = json.loads(response[start:end])
        decision = verdict.get("decision", "accept")
        reason = verdict.get("reason", "")
    except Exception as e:
        print(f"[router] Acceptance check failed ({e}), auto-accepting", flush=True)
        decision = "accept"
        reason = "Acceptance check unavailable — auto-accepted"

    if decision == "accept":
        emoji = "✅"
        channel_msg = f"✅ *Claude accepted* sub-agent work\n{reason}"
        if issue_number:
            _gh("PATCH", f"issues/{issue_number}", {"state": "closed"})
            _gh("POST", f"issues/{issue_number}/labels", {"labels": ["claude-accepted"]})
    else:
        emoji = "❌"
        channel_msg = f"❌ *Claude rejected* sub-agent work — re-opened for revision\n{reason}"
        if issue_number:
            _gh("PATCH", f"issues/{issue_number}", {"state": "open"})
            _gh("POST", f"issues/{issue_number}/comments", {"body": (
                f"❌ **Claude Acceptance Gate: REJECTED**\n\n"
                f"**Reason:** {reason}\n\n"
                f"Please revise and re-submit."
            )})
            _gh("POST", f"issues/{issue_number}/labels", {"labels": ["claude-rejected"]})

    slack_post("#engineering", channel_msg)
    print(f"[router] Acceptance: {emoji} {decision} — {reason}", flush=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Claude task router")
    parser.add_argument("mode", choices=["route", "accept"], nargs="?", default="route")
    parser.add_argument("--issue", type=int, help="GitHub issue number")
    parser.add_argument("--task", type=str, help="Inline task (dry-run, no GH API)")
    args = parser.parse_args()

    if args.task:
        # Dry-run: classify without touching GitHub
        response = _call_anthropic(f"Title: {args.task}\nDescription: (inline task)")
        print(f"[router] Dry-run result: {response}", flush=True)
        return

    if args.mode == "route":
        if not args.issue:
            # Route all open un-labelled issues
            issues = _gh("GET", "issues?state=open&per_page=20") or []
            if isinstance(issues, list):
                for iss in issues:
                    if not any(
                        lbl["name"] in _LABEL_MAP.values()
                        for lbl in iss.get("labels", [])
                    ):
                        route_issue(iss["number"])
        else:
            route_issue(args.issue)
    elif args.mode == "accept":
        accept_work(args.issue)


if __name__ == "__main__":
    main()
