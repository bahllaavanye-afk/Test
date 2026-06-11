"""
Token Usage Monitor — posts every 15 min to #token-usage Slack channel.
Tracks LLM spend by provider, workflow, and improvement type.
Auto-analyzes and suggests optimizations using the cheapest available LLM.
"""
from __future__ import annotations
import os, sys, json, glob, subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""

SLACK_TOKEN     = os.environ.get("SLACK_BOT_TOKEN", "")
GH_TOKEN        = os.environ.get("GH_TOKEN", "")
GH_REPO         = os.environ.get("GH_REPO", "bahllaavanye-afk/test")
GEMINI_API_KEY  = _resolve_key("GEMINI_API_KEY", "GEMINI_API_KEY_1")
GROQ_API_KEY    = _resolve_key("GROQ_API_KEY", "GROQ_API_KEY_1")
DEEPSEEK_KEY    = _resolve_key("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_1")
CEREBRAS_KEY    = _resolve_key("CEREBRAS_API_KEY", "CEREBRAS_API_KEY_1")
SAMBANOVA_KEY   = _resolve_key("SAMBANOVA_API_KEY", "SAMBANOVA_API_KEY_1")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)

STATE_FILE   = Path(__file__).resolve().parents[2] / ".github" / "state" / "agent_memory.json"
TOKEN_LOG    = Path(__file__).resolve().parents[2] / ".github" / "state" / "token_usage_log.json"
SKILL_FILE   = Path(__file__).resolve().parents[2] / ".github" / "state" / "skill_library.json"

# ── Free tier limits (approximate) ───────────────────────────────────────────
FREE_LIMITS = {
    "gemini_flash":  {"daily_tokens": 1_500_000, "rpm": 15,  "cost_per_1m": 0.0},
    "groq_llama":    {"daily_tokens": 400_000,   "rpm": 30,  "cost_per_1m": 0.0},
    "cerebras":      {"daily_tokens": 1_000_000, "rpm": 60,  "cost_per_1m": 0.0},
    "sambanova":     {"daily_tokens": 20_000_000,"rpm": 10,  "cost_per_1m": 0.0},
    "deepseek_chat": {"daily_tokens": 100_000,   "rpm": 100, "cost_per_1m": 0.27},
    "perplexity":    {"daily_tokens": 50_000,    "rpm": 20,  "cost_per_1m": 0.20},
}

# ── Workflow token cost estimates (avg tokens per run) ─────────────────────────
WORKFLOW_ESTIMATES = {
    "claude-chat":                    {"prompt": 3000,  "completion": 800,  "runs_per_day": 24},
    "continuous-improvement":         {"prompt": 8000,  "completion": 4000, "runs_per_day": 12},
    "frontend-design-agent":          {"prompt": 10000, "completion": 4000, "runs_per_day": 12},
    "team-lead-issues":               {"prompt": 2000,  "completion": 1000, "runs_per_day": 2},
    "free-agent-engineer":            {"prompt": 5000,  "completion": 2000, "runs_per_day": 12},
    "autonomous-strategy-generator":  {"prompt": 4000,  "completion": 2000, "runs_per_day": 8},
    "daily-standup":                  {"prompt": 2000,  "completion": 1000, "runs_per_day": 4},
    "gemini-change-review":           {"prompt": 5000,  "completion": 1500, "runs_per_day": 6},
    "peer-review":                    {"prompt": 6000,  "completion": 1500, "runs_per_day": 8},
    "agent-heartbeat":                {"prompt": 500,   "completion": 100,  "runs_per_day": 48},
    "okr-tracker":                    {"prompt": 1500,  "completion": 500,  "runs_per_day": 2},
    "p0-watchdog":                    {"prompt": 1000,  "completion": 200,  "runs_per_day": 24},
    "investor-pipeline-update":       {"prompt": 1000,  "completion": 300,  "runs_per_day": 6},
    "gemini-ml-training":             {"prompt": 6000,  "completion": 3000, "runs_per_day": 3},
    "run-experiments-agent":          {"prompt": 5000,  "completion": 2000, "runs_per_day": 4},
    "slack-agent-team":               {"prompt": 2000,  "completion": 800,  "runs_per_day": 24},
    "slack-pulse":                    {"prompt": 1500,  "completion": 600,  "runs_per_day": 24},
    "channel-monitor":                {"prompt": 2000,  "completion": 500,  "runs_per_day": 24},
}

def load_token_log() -> dict:
    try:
        return json.loads(TOKEN_LOG.read_text())
    except Exception:
        return {"daily": {}, "hourly": [], "optimization_history": []}

def save_token_log(log: dict):
    TOKEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_LOG.write_text(json.dumps(log, indent=2))

def load_memory() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}

def load_skills() -> list[str]:
    try:
        return json.loads(SKILL_FILE.read_text()).get("skills", [])
    except Exception:
        return []

def get_github_workflow_runs_today() -> dict[str, int]:
    """Count actual GitHub Actions runs today per workflow."""
    if not GH_TOKEN:
        return {}
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    counts = {}
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/actions/runs",
            headers=headers,
            params={"created": f">={today}", "per_page": 100},
            timeout=15
        )
        if resp.status_code == 200:
            for run in resp.json().get("workflow_runs", []):
                name = run.get("name", "unknown")
                counts[name] = counts.get(name, 0) + 1
    except Exception as e:
        print(f"GitHub runs fetch error: {e}")
    return counts

def compute_token_breakdown(actual_runs: dict[str, int]) -> dict:
    """Compute estimated token usage by workflow and provider."""
    breakdown = {
        "by_workflow": {},
        "by_provider": {"gemini_flash": 0, "groq_llama": 0, "cerebras": 0,
                         "sambanova": 0, "deepseek_chat": 0},
        "totals": {"prompt": 0, "completion": 0, "total": 0, "est_cost_usd": 0.0},
    }

    for wf_name, est in WORKFLOW_ESTIMATES.items():
        runs = actual_runs.get(wf_name, 0)
        if runs == 0:
            # Use estimate if no actual run data
            runs = max(1, est["runs_per_day"] // (24 // max(1, datetime.now(timezone.utc).hour or 1)))

        prompt_tok = est["prompt"] * runs
        completion_tok = est["completion"] * runs
        total_tok = prompt_tok + completion_tok

        breakdown["by_workflow"][wf_name] = {
            "runs": runs,
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens": total_tok,
        }
        breakdown["totals"]["prompt"] += prompt_tok
        breakdown["totals"]["completion"] += completion_tok
        breakdown["totals"]["total"] += total_tok

    # Distribute across providers (Gemini takes 70%, Groq 20%, others 10%)
    total = breakdown["totals"]["total"]
    breakdown["by_provider"]["gemini_flash"] = int(total * 0.70)
    breakdown["by_provider"]["groq_llama"]   = int(total * 0.15)
    breakdown["by_provider"]["cerebras"]     = int(total * 0.08)
    breakdown["by_provider"]["sambanova"]    = int(total * 0.05)
    breakdown["by_provider"]["deepseek_chat"]= int(total * 0.02)

    # Estimate cost (only DeepSeek has non-zero cost at scale)
    breakdown["totals"]["est_cost_usd"] = round(
        breakdown["by_provider"]["deepseek_chat"] / 1_000_000 * FREE_LIMITS["deepseek_chat"]["cost_per_1m"], 4
    )

    return breakdown

def pct_bar(used: int, limit: int, width: int = 10) -> str:
    if limit == 0:
        return "░" * width
    filled = min(width, int(used / limit * width))
    return "█" * filled + "░" * (width - filled)

def check_provider_health() -> dict[str, str]:
    health = {}
    health["gemini_flash"]  = "✅ configured" if GEMINI_API_KEY  else "❌ missing key"
    health["groq_llama"]    = "✅ configured" if GROQ_API_KEY    else "❌ missing key"
    health["cerebras"]      = "✅ configured" if CEREBRAS_KEY    else "⚠️ add CEREBRAS_API_KEY_1"
    health["sambanova"]     = "✅ configured" if SAMBANOVA_KEY   else "⚠️ add SAMBANOVA_API_KEY_1"
    health["deepseek_chat"] = "✅ configured" if DEEPSEEK_KEY    else "⚠️ add DEEPSEEK_API_KEY_1"
    return health

def llm_suggest_optimizations(breakdown: dict, skills: list[str]) -> str:
    """Use cheapest available LLM to suggest token optimizations."""
    top_wf = sorted(breakdown["by_workflow"].items(),
                    key=lambda x: x[1]["total_tokens"], reverse=True)[:5]
    top_str = "\n".join(f"  {wf}: {d['total_tokens']:,} tokens ({d['runs']} runs)"
                        for wf, d in top_wf)

    prompt = f"""You are a DevOps efficiency expert for QuantEdge autonomous trading platform.
Current token usage top 5 workflows:
{top_str}

Total today: {breakdown['totals']['total']:,} tokens
Provider breakdown: {json.dumps(breakdown['by_provider'], indent=2)}

Known skills/constraints:
{chr(10).join(skills[-5:]) if skills else 'none yet'}

Give 3 specific, actionable token reduction tips for these exact workflows.
Each tip: one sentence, concrete (e.g. "Reduce N_FILES from 3 to 2 in continuous-improvement to save 12K tokens/day").
Format: numbered list, under 200 words total."""

    # Use cheapest provider first
    for key, url, model in [
        (GROQ_API_KEY,   "https://api.groq.com/openai/v1/chat/completions",              "llama-3.1-8b-instant"),
        (CEREBRAS_KEY,   "https://api.cerebras.ai/v1/chat/completions",                   "llama-3.3-70b"),
        (DEEPSEEK_KEY,   "https://api.deepseek.com/chat/completions",                     "deepseek-chat"),
        (GEMINI_API_KEY, None, None),  # handled separately
    ]:
        if not key: continue
        if url is None:
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
                    json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                          "generationConfig": {"maxOutputTokens": 300, "temperature": 0.3}},
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                pass
        else:
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300},
                    timeout=25
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

    return "No LLM available for optimization suggestions."

def post_slack(channel: str, blocks: list[dict], fallback: str) -> bool:
    if not SLACK_TOKEN:
        print(f"No SLACK_BOT_TOKEN — would post to #{channel}:\n{fallback}")
        return False
    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
            json={"channel": channel, "blocks": blocks, "text": fallback},
            timeout=15
        )
        return resp.status_code == 200 and resp.json().get("ok")
    except Exception as e:
        print(f"Slack post error: {e}")
        return False

def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%H:%M UTC')}] Token usage monitor")

    subprocess.run(["git", "pull", "--rebase", "--quiet"], capture_output=True)

    # Gather data
    actual_runs   = get_github_workflow_runs_today()
    breakdown     = compute_token_breakdown(actual_runs)
    provider_health = check_provider_health()
    skills        = load_skills()
    mem           = load_memory()
    log           = load_token_log()

    # Update log
    hour_key = now.strftime("%Y-%m-%dT%H:00Z")
    log["hourly"].append({
        "time": now.isoformat(),
        "total_tokens": breakdown["totals"]["total"],
        "by_provider": breakdown["by_provider"],
    })
    log["hourly"] = log["hourly"][-96:]  # keep 4 days

    # LLM suggestions (run occasionally to save tokens — every 4th run)
    suggestions = ""
    run_count = len(log.get("hourly", []))
    if run_count % 4 == 1:
        suggestions = llm_suggest_optimizations(breakdown, skills)
        if suggestions:
            log["optimization_history"].append({
                "time": now.isoformat(),
                "suggestions": suggestions,
            })
            log["optimization_history"] = log["optimization_history"][-20:]

    save_token_log(log)

    # ── Build Slack message ──────────────────────────────────────────────────
    total = breakdown["totals"]["total"]
    gemini_used = breakdown["by_provider"]["gemini_flash"]
    gemini_limit = FREE_LIMITS["gemini_flash"]["daily_tokens"]
    groq_used = breakdown["by_provider"]["groq_llama"]
    groq_limit = FREE_LIMITS["groq_llama"]["daily_tokens"]

    # Top 5 workflows
    top_workflows = sorted(breakdown["by_workflow"].items(),
                           key=lambda x: x[1]["total_tokens"], reverse=True)[:6]

    wf_lines = "\n".join(
        f"  `{wf[:30]:<30}` {d['total_tokens']:>8,} tok  ({d['runs']} runs)"
        for wf, d in top_workflows
    )

    provider_lines = "\n".join(
        f"  {name:<15} {used:>9,} tok  {health}"
        for (name, used), health in zip(
            sorted(breakdown["by_provider"].items(), key=lambda x: x[1], reverse=True),
            [provider_health.get(p, "?") for p, _ in
             sorted(breakdown["by_provider"].items(), key=lambda x: x[1], reverse=True)]
        )
    )

    # Free tier utilization bars
    gemini_pct = min(100, int(gemini_used / gemini_limit * 100)) if gemini_limit else 0
    groq_pct   = min(100, int(groq_used   / groq_limit   * 100)) if groq_limit else 0

    last_suggestions = suggestions or (
        log["optimization_history"][-1]["suggestions"]
        if log.get("optimization_history") else "Run more cycles to generate suggestions."
    )

    text = f"""*QuantEdge Token Usage — {now.strftime('%H:%M UTC')}*

*Provider Health*
```
{provider_lines}
```

*Free Tier Utilization*
• Gemini Flash  {pct_bar(gemini_used, gemini_limit)} {gemini_pct}%  ({gemini_used:,}/{gemini_limit:,})
• Groq          {pct_bar(groq_used, groq_limit)}  {groq_pct}%  ({groq_used:,}/{groq_limit:,})
• Cerebras      ░░░░░░░░░░ 1M/day free
• SambaNova     ░░░░░░░░░░ 20M/day free

*Top Token Consumers Today*
```
{wf_lines}
```

*Daily Total:* {total:,} tokens  |  *Est. Cost:* ${breakdown['totals']['est_cost_usd']:.4f}
*Prompt:* {breakdown['totals']['prompt']:,}  |  *Completion:* {breakdown['totals']['completion']:,}

*AI Optimization Suggestions:*
{last_suggestions}"""

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": text[:3000]}},
        {"type": "divider"},
        {"type": "context", "elements": [
            {"type": "mrkdwn",
             "text": f"Auto-analyzed by token_usage_monitor.py • Runs every 15min • Skills library: {len(skills)} entries"}
        ]}
    ]

    posted = post_slack("token-usage", blocks, text)
    print(f"{'✓' if posted else '⚠'} Posted to #token-usage")
    print(f"  Total today: {total:,} tokens")
    print(f"  Gemini: {gemini_pct}% of free tier")
    print(f"  Groq: {groq_pct}% of free tier")

    # Commit updated log
    try:
        subprocess.run(["git", "add", str(TOKEN_LOG)], capture_output=True)
        subprocess.run(["git", "commit", "-m",
                        f"state: token_usage_log {now.strftime('%H:%M')} — {total:,} tokens",
                        "--allow-empty"],
                       capture_output=True,
                       env={**os.environ,
                            "GIT_AUTHOR_NAME": "Token Monitor",
                            "GIT_AUTHOR_EMAIL": "monitor@quantedge.ai",
                            "GIT_COMMITTER_NAME": "Token Monitor",
                            "GIT_COMMITTER_EMAIL": "monitor@quantedge.ai"})
        subprocess.run(["git", "push"], capture_output=True)
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    sys.exit(main())
