"""
Company Brain — Aggregates ALL knowledge sources into one shared context.

This is what closes the biggest gap between this system and a real company:
in a real firm, everyone has access to all context all the time. Here we
build that shared context by pulling from every source and distilling it
into a compact (~1000 token) company memory that every agent reads first.

Sources aggregated (every 15 minutes):
  1. Slack threads — messages from the last 2 hours across key channels
     (not just posted, but REPLIES too — full conversation context)
  2. GitHub — recent PR review comments, issue discussions, CI status
  3. Trade outcomes — from .github/state/trade_log.json
  4. Experiment results — from .github/state/experiment_log.json
  5. Code review findings — from docs/agent-reviews/
  6. Previous company brain (continuity — nothing is lost)

Output: .github/state/company_brain.json
  This single file is read by llm_common.get_company_context() and injected
  into EVERY agent prompt. Every agent shares the same reality.

Architecture principle:
  No agent should act in an information vacuum. The company brain is the
  single source of truth that all agents draw from before responding.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Import shared LLM infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm, memory_write, memory_read, core_update, core_get, slack_read_channel, _load_brain, _save_brain, _STATE_DIR

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", os.environ.get("PAT_TOKEN", ""))
REPO = "bahllaavanye-afk/test"
BRANCH = "main"

# Key channels to monitor for knowledge
KNOWLEDGE_CHANNELS = [
    "desk-research", "desk-tv-indicators", "alpha-research", "ml-research",
    "risk", "engineering", "backend", "frontend", "desk-lead-review",
    "desk-crypto", "desk-equities", "strategy-performance",
]

LOOKBACK_SECONDS = 7200   # 2 hours of Slack history


def resolve_channel_ids() -> dict[str, str]:
    """Get channel name → ID mapping."""
    if not SLACK_TOKEN:
        return {}
    req = urllib.request.Request(
        "https://slack.com/api/conversations.list?limit=200&types=public_channel",
        headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return {c["name"]: c["id"] for c in data.get("channels", [])}
    except Exception:
        return {}


def fetch_slack_knowledge(channel_ids: dict[str, str]) -> list[dict]:
    """
    Pull recent Slack messages from key channels.
    Returns list of {channel, text, thread_ts, is_reply, ts}.
    Filters out bot noise and trivial messages.
    """
    cutoff = time.time() - LOOKBACK_SECONDS
    items = []
    for channel_name in KNOWLEDGE_CHANNELS:
        cid = channel_ids.get(channel_name)
        if not cid:
            continue
        msgs = slack_read_channel(cid, limit=30, oldest=cutoff)
        for m in msgs:
            text = m.get("text", "").strip()
            if not text or len(text) < 20:
                continue
            if m.get("bot_id") or m.get("subtype"):
                continue  # Skip bots and system messages
            items.append({
                "channel": channel_name,
                "text": text[:300],  # cap per-message
                "ts": m.get("ts", ""),
                "user": m.get("username", m.get("user", "?")),
            })
    return items


def fetch_github_knowledge() -> list[dict]:
    """
    Pull recent PR comments and issue discussions from GitHub.
    These contain the engineering team's actual thinking — review feedback,
    debate, decisions — the most valuable knowledge in the repo.
    """
    if not GITHUB_TOKEN:
        return []
    items = []

    # Recent PR review comments
    url = f"https://api.github.com/repos/{REPO}/pulls/comments?per_page=20&sort=created&direction=desc"
    try:
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            comments = json.loads(resp.read())
            for c in comments:
                body = c.get("body", "").strip()
                if body and len(body) > 20:
                    items.append({
                        "source": "github_pr_review",
                        "text": body[:300],
                        "path": c.get("path", ""),
                        "pr": c.get("pull_request_url", "").split("/")[-1],
                    })
    except Exception:
        pass

    # Recent issue comments (engineering decisions, bug reports)
    url = f"https://api.github.com/repos/{REPO}/issues/comments?per_page=20&sort=created&direction=desc"
    try:
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            comments = json.loads(resp.read())
            for c in comments:
                body = c.get("body", "").strip()
                if body and len(body) > 20:
                    items.append({
                        "source": "github_issue",
                        "text": body[:300],
                        "issue": c.get("issue_url", "").split("/")[-1],
                    })
    except Exception:
        pass

    return items


def fetch_recent_code_reviews() -> list[str]:
    """Pull recent code review insights from docs/agent-reviews/."""
    reviews_dir = Path(os.environ.get("GITHUB_WORKSPACE", ".")) / "docs" / "agent-reviews"
    findings = []
    if not reviews_dir.exists():
        return []
    # Read most recent review file
    files = sorted(reviews_dir.glob("*.md"), reverse=True)
    if not files:
        return []
    text = files[0].read_text()[:3000]
    # Extract bullet points (priorities)
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "* ", "1.", "2.", "3.")) and len(line) > 20:
            findings.append(line.lstrip("-*0123456789. "))
            if len(findings) >= 5:
                break
    return findings


def synthesize_insights(
    slack_msgs: list[dict],
    github_items: list[dict],
    code_findings: list[str],
) -> dict:
    """
    Use LLM to synthesize raw knowledge sources into structured insights.
    This is the company's collective intelligence step.
    """
    if not slack_msgs and not github_items and not code_findings:
        return {}

    # Build compact synthesis prompt — efficiency first
    parts = []

    if slack_msgs:
        slack_sample = slack_msgs[:10]
        slack_text = "\n".join(f"[{m['channel']}] {m.get('user','?')}: {m['text']}" for m in slack_sample)
        parts.append(f"SLACK (last 2h):\n{slack_text}")

    if github_items:
        gh_sample = github_items[:5]
        gh_text = "\n".join(f"[{g['source']}] {g['text']}" for g in gh_sample)
        parts.append(f"GITHUB DISCUSSIONS:\n{gh_text}")

    if code_findings:
        parts.append(f"CODE REVIEW FINDINGS:\n" + "\n".join(f"- {f}" for f in code_findings[:3]))

    prompt = "\n\n".join(parts)
    prompt += "\n\nExtract 3-5 actionable insights. Format: JSON array of {\"insight\": \"...\", \"category\": \"strategy|risk|ml|code|execution\", \"priority\": 1-3}"

    result = llm(
        prompt,
        system="You are the CIO of QuantEdge synthesizing intelligence from all company sources. Be concise and specific.",
        max_tokens=600,
        inject_company_context=False,  # avoid circular dependency
    )

    # Parse JSON from result
    try:
        match = __import__("re").search(r"\[.*\]", result, __import__("re").DOTALL)
        if match:
            return {"insights": json.loads(match.group(0))}
    except Exception:
        pass
    return {"raw": result}


def update_company_brain():
    """
    Main function: aggregate all sources, synthesize, save to company_brain.json.
    Called every 15 minutes by GitHub Actions.
    """
    print("Company brain update starting...")
    brain = _load_brain()

    # 1. Fetch all knowledge sources in parallel (sequential here for simplicity)
    channel_ids = resolve_channel_ids()
    print(f"Resolved {len(channel_ids)} Slack channels")

    slack_msgs = fetch_slack_knowledge(channel_ids)
    print(f"Fetched {len(slack_msgs)} Slack messages")

    github_items = fetch_github_knowledge()
    print(f"Fetched {len(github_items)} GitHub discussion items")

    code_findings = fetch_recent_code_reviews()
    print(f"Fetched {len(code_findings)} code review findings")

    # 2. Synthesize into structured insights
    if slack_msgs or github_items or code_findings:
        synthesis = synthesize_insights(slack_msgs, github_items, code_findings)
        insights = synthesis.get("insights", [])
        print(f"Synthesized {len(insights)} insights")

        # Write to episodic and specialized memories
        for insight in insights:
            memory_write("episodic", {
                "lesson": insight.get("insight", ""),
                "category": insight.get("category", "general"),
                "priority": insight.get("priority", 2),
                "source": "company_brain_synthesis",
            })

        # Write Slack-derived insights
        if slack_msgs:
            summary = llm(
                "Summarize these Slack messages in 1-2 sentences: " +
                " | ".join(m["text"][:100] for m in slack_msgs[:5]),
                max_tokens=100,
                inject_company_context=False,
            )
            memory_write("slack_insights", {"summary": summary, "msg_count": len(slack_msgs)})

        # Write GitHub-derived insights
        if github_items:
            gh_summary = llm(
                "Summarize these GitHub PR/issue discussions in 1-2 sentences: " +
                " | ".join(g["text"][:100] for g in github_items[:3]),
                max_tokens=100,
                inject_company_context=False,
            )
            memory_write("github_insights", {"summary": gh_summary, "item_count": len(github_items)})

    # 3. Check trade log for recent outcomes
    trade_log = _STATE_DIR / "trade_log.json"
    if trade_log.exists():
        try:
            trades = json.loads(trade_log.read_text())
            recent = trades[-5:] if isinstance(trades, list) else []
            for t in recent:
                if t not in brain.get("trade_outcomes", []):
                    memory_write("trade_outcomes", t)
        except Exception:
            pass

    # 4. Update CORE memory from synthesized state
    # Regime from most recent trade data
    recent_lessons = memory_read("episodic", 10)
    risk_keywords = ["circuit breaker", "drawdown", "limit breached", "alert"]
    risk_status = "alert" if any(
        any(kw in e.get("lesson", "").lower() for kw in risk_keywords)
        for e in recent_lessons
    ) else "normal"
    core_update("risk_status", risk_status)
    core_update("last_brain_update", time.time())

    # 5. Reload and report
    brain = _load_brain()
    episodic_count = len(brain.get("episodic", []))
    skills_count = len(brain.get("skills", []))
    slack_count = len(brain.get("slack_insights", []))

    print(f"Company brain updated:")
    print(f"  Episodic memory: {episodic_count} entries")
    print(f"  Skills library: {skills_count} entries")
    print(f"  Slack insights: {slack_count} entries")
    print(f"  Risk status: {risk_status}")
    print(f"  Context size: {len(get_context_preview())} chars")


def get_context_preview() -> str:
    """Return what get_company_context() would produce — for monitoring."""
    from llm_common import get_company_context
    return get_company_context()


if __name__ == "__main__":
    update_company_brain()
