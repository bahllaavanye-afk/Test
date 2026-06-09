"""
Deep Code Review Agent Pool
============================
Dispatches 8 specialist free-LLM agents to review different codebases areas:

  Agent 1 — Strategies Desk       (backend/app/strategies/)
  Agent 2 — ML/Models Desk        (backend/app/ml/)
  Agent 3 — Execution Desk        (backend/app/execution/)
  Agent 4 — Risk Desk             (backend/app/risk/)
  Agent 5 — API/Auth Desk         (backend/app/api/)
  Agent 6 — Tasks/Scheduler Desk  (backend/app/tasks/)
  Agent 7 — Frontend Desk         (frontend/src/)
  Agent 8 — Infrastructure Desk   (render.yaml, pyproject.toml, workflows)

Each agent:
  1. Reads its assigned files
  2. Uses free LLM to generate a structured improvement doc
  3. Saves doc to docs/agent-reviews/<agent>-YYYY-MM-DD.md
  4. Posts a summary to Slack

All runs committed to the branch. Zero Anthropic tokens.
"""
from __future__ import annotations

import json, os, re, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
BRANCH      = "claude/advanced-trading-bot-d5Lmw"
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
ALLOW_PAID  = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID.lower() == "true":
    sys.exit(1)

REVIEW_DIR = REPO_ROOT / "docs" / "agent-reviews"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

DATE_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Each agent: (name, slack_channel, file_globs, max_chars_per_file)
AGENTS = [
    ("strategies", "desk-research",
     ["backend/app/strategies/manual/*.py", "backend/app/strategies/ml_enhanced/*.py",
      "backend/app/strategies/base.py"],
     2000),
    ("ml-models", "desk-ml",
     ["backend/app/ml/models/*.py", "backend/app/ml/features/*.py"],
     1500),
    ("execution", "desk-equity",
     ["backend/app/execution/*.py"],
     2000),
    ("risk", "desk-risk",
     ["backend/app/risk/*.py"],
     2000),
    ("api-backend", "engineering",
     ["backend/app/api/v1/*.py", "backend/app/main.py", "backend/app/config.py"],
     1500),
    ("tasks-scheduler", "engineering",
     ["backend/app/tasks/*.py"],
     1500),
    ("frontend", "engineering",
     ["frontend/src/pages/*.tsx", "frontend/src/components/layout/*.tsx"],
     1500),
    ("infrastructure", "engineering",
     ["render.yaml", "backend/pyproject.toml", ".github/workflows/*.yml"],
     1000),
]


# ── Free LLM cascade ──────────────────────────────────────────────────────────

def _llm(prompt: str, max_tokens: int = 1500) -> str | None:
    providers = [
        ("gemini",    os.environ.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY_1", "")),
         "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini-2.0-flash"),
        ("groq",      os.environ.get("GROQ_API_KEY", ""),
         "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
        ("deepseek",  os.environ.get("DEEPSEEK_API_KEY", ""),
         "https://api.deepseek.com/v1/chat/completions", "deepseek-chat"),
        ("together",  os.environ.get("TOGETHER_API_KEY", ""),
         "https://api.together.xyz/v1/chat/completions", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("cerebras",  os.environ.get("CEREBRAS_API_KEY", ""),
         "https://api.cerebras.ai/v1/chat/completions", "llama-3.3-70b"),
        ("sambanova", os.environ.get("SAMBANOVA_API_KEY", ""),
         "https://api.sambanova.ai/v1/chat/completions", "Meta-Llama-3.3-70B-Instruct"),
        ("hyperbolic", os.environ.get("HYPERBOLIC_API_KEY", ""),
         "https://api.hyperbolic.xyz/v1/chat/completions", "meta-llama/Llama-3.3-70B-Instruct"),
    ]
    for name, key, url, model in providers:
        if not key or key in ("disabled", ""):
            continue
        try:
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.1,
            }).encode()
            req = urllib.request.Request(url, data=payload, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            print(f"[review] {name}: {len(text)} chars")
            return text
        except Exception as e:
            print(f"[review] {name}: {e}")
    return None


# ── Slack helpers ─────────────────────────────────────────────────────────────

def slack(channel: str, msg: str) -> None:
    if not SLACK_TOKEN:
        print(f"[Slack #{channel}] {msg[:120]}")
        return
    try:
        payload = json.dumps({"channel": f"#{channel}", "text": msg, "mrkdwn": True}).encode()
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage", data=payload,
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Slack error: {e}")


# ── File reading ──────────────────────────────────────────────────────────────

def read_files_for_agent(globs: list[str], max_chars: int) -> str:
    import glob as globlib
    parts = []
    total = 0
    for pattern in globs:
        for fpath in sorted(globlib.glob(str(REPO_ROOT / pattern)))[:6]:  # max 6 files per pattern
            p = Path(fpath)
            if not p.exists():
                continue
            content = p.read_text(errors="replace")[:max_chars]
            rel = str(p.relative_to(REPO_ROOT))
            parts.append(f"=== {rel} ===\n{content}")
            total += len(content)
            if total > 10000:  # cap total context
                break
        if total > 10000:
            break
    return "\n\n".join(parts)


# ── Review prompt ─────────────────────────────────────────────────────────────

REVIEW_PROMPT = """You are a world-class quantitative software engineer at a Two Sigma / Citadel-level firm.
Review the following codebase section: {agent_name}

Your job: deep review, learn the architecture, identify improvements.

Files:
{code}

Generate a structured improvement document with these exact sections:

## Architecture Assessment
(2-3 sentences on the current design quality)

## Critical Issues (P0/P1)
- List up to 3 bugs or security issues with file:line reference and exact fix

## Performance Improvements
- List up to 3 performance wins (latency, throughput, memory)

## Alpha/Signal Quality Improvements
- List up to 3 ways to improve trading signal quality or alpha generation

## Code Quality
- List up to 3 refactoring suggestions that reduce complexity or improve reliability

## Implementation Priority
Ranked list of top 5 improvements to implement first, with 1-line rationale each.

Be specific: cite file names, function names, line patterns. No generic advice."""


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

def main() -> None:
    slack("engineering",
        f"🔍 *Code Review Agent Pool* ({DATE_STR}): "
        f"8 specialist agents starting deep review of entire codebase. "
        f"Results will be committed to `docs/agent-reviews/`. "
        f"Zero Anthropic tokens used.")

    git_config()
    completed = []

    for agent_name, channel, globs, max_chars in AGENTS:
        print(f"\n[review] Agent: {agent_name}")
        code = read_files_for_agent(globs, max_chars)

        if not code.strip():
            print(f"[review] {agent_name}: no files found, skipping")
            continue

        prompt = REVIEW_PROMPT.format(agent_name=agent_name, code=code[:8000])
        review = _llm(prompt, max_tokens=1500)

        if not review:
            print(f"[review] {agent_name}: LLM failed")
            continue

        # Save doc
        doc_path = REVIEW_DIR / f"{agent_name}-{DATE_STR}.md"
        header = (
            f"# {agent_name.replace('-', ' ').title()} Review — {DATE_STR}\n\n"
            f"*Generated by: Free LLM Agent Pool (0 Anthropic tokens)*\n\n"
            f"---\n\n"
        )
        doc_path.write_text(header + review)
        print(f"[review] {agent_name}: wrote {doc_path.name}")

        # Extract P0/P1 count for Slack summary
        critical_count = len(re.findall(r"P0|P1|critical|Critical|CRITICAL", review))

        # Post summary to desk channel
        # Extract just the "Implementation Priority" section for the Slack message
        priority_match = re.search(r"## Implementation Priority\s*(.*?)(?=\n##|$)", review, re.DOTALL)
        priority_text = priority_match.group(1).strip()[:400] if priority_match else "See full doc."

        slack(channel,
            f"📋 *Code Review Agent: {agent_name}*\n"
            f"_{critical_count} critical issues found_\n\n"
            f"*Top improvements:*\n{priority_text}\n\n"
            f"_Full doc: `docs/agent-reviews/{doc_path.name}`_")

        completed.append(agent_name)
        time.sleep(2)  # rate limit between agents

    # Commit all docs
    commit_docs()

    slack("engineering",
        f"✅ *Code Review Agent Pool:* {len(completed)}/{len(AGENTS)} agents completed\n"
        f"Agents: {', '.join(f'`{a}`' for a in completed)}\n"
        f"All docs committed → `docs/agent-reviews/` on `{BRANCH}`\n"
        f"_Zero Anthropic tokens used — powered by free LLM cascade_")


if __name__ == "__main__":
    main()
