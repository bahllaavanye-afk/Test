"""
Peer Reviewer Agent — multi-agent code review.
Reviews the last N commits made by autonomous agents.
Uses a separate Gemini key for independent review (Agent B reviews Agent A's work).
Posts findings as GitHub issues if critical bugs found.
"""
from __future__ import annotations
import os, sys, json, subprocess
from datetime import datetime, timezone
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

# Use key _2 or _3 for independent review — different quota/key from improver
GEMINI_KEY = _resolve_key("GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_1")
GROQ_KEY   = _resolve_key("GROQ_API_KEY", "GROQ_API_KEY_1")
GH_TOKEN   = os.environ.get("GH_TOKEN", "")
GH_REPO    = os.environ.get("GH_REPO", "bahllaavanye-afk/test")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)

STATE_FILE = Path(__file__).resolve().parents[2] / ".github" / "state" / "agent_memory.json"

def llm(prompt: str) -> str:
    if GEMINI_KEY:
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.1}},
                timeout=45
            )
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"Gemini error: {e}")
    if GROQ_KEY:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant",
                      "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500},
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
    return ""

def get_recent_agent_commits(n: int = 5) -> list[dict]:
    # Review ALL recent commits on the branch — no author filter.
    # The peer reviewer should catch issues regardless of who committed.
    result = subprocess.run(
        ["git", "log", f"-{n}", "--format=%H|%s|%ai|%an"],
        capture_output=True, text=True
    )
    commits = []
    for line in result.stdout.strip().split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|", 3)
        if len(parts) >= 3:
            commits.append({
                "sha": parts[0].strip(),
                "subject": parts[1].strip(),
                "date": parts[2].strip(),
                "author": parts[3].strip() if len(parts) > 3 else "unknown",
            })
    return commits

def get_diff(sha: str) -> str:
    result = subprocess.run(
        ["git", "show", "--stat", "--diff-filter=M", "-U3", sha],
        capture_output=True, text=True
    )
    diff = result.stdout
    return diff[:6000] if len(diff) > 6000 else diff

def review_diff(sha: str, subject: str, diff: str) -> dict | None:
    if not diff.strip():
        return None

    prompt = f"""You are a senior code reviewer. Review this autonomous agent commit.
Commit: {subject}

Diff:
{diff}

Check for:
1. Syntax errors or broken imports
2. Logic bugs that could cause exceptions in production
3. Security issues (hardcoded secrets, SQL injection, etc.)
4. Breaking changes to existing behavior
5. Missing error handling on IO operations

Respond in JSON only:
{{"severity": "ok|warning|critical", "issues": ["issue1", "issue2"], "summary": "one sentence"}}

If no issues: {{"severity": "ok", "issues": [], "summary": "looks good"}}"""

    raw = llm(prompt)
    if not raw:
        return None
    try:
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
        return json.loads(raw)
    except Exception:
        return None

def open_github_issue(title: str, body: str) -> bool:
    if not GH_TOKEN:
        return False
    try:
        resp = requests.post(
            f"https://api.github.com/repos/{GH_REPO}/issues",
            headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"},
            json={"title": title, "body": body, "labels": ["bug", "agent-review"]},
            timeout=15
        )
        return resp.status_code == 201
    except Exception:
        return False

def load_memory() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}

def save_memory(mem: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    mem["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(mem, indent=2))

def main():
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Peer review agent starting")
    subprocess.run(["git", "pull", "--rebase", "--quiet"], capture_output=True)

    commits = get_recent_agent_commits(5)
    if not commits:
        print("No recent agent commits to review")
        return 0

    mem = load_memory()
    reviewed = mem.setdefault("peer_reviewed_shas", [])
    issues_opened = 0
    reviewed_count = 0

    for commit in commits:
        sha = commit["sha"]
        if sha in reviewed:
            continue

        diff = get_diff(sha)
        result = review_diff(sha, commit["subject"], diff)
        reviewed.append(sha)
        reviewed_count += 1

        if not result:
            continue

        severity = result.get("severity", "ok")
        issues = result.get("issues", [])
        print(f"  {sha[:8]} [{severity}]: {result.get('summary', '')}")

        if severity == "critical" and issues:
            title = f"[Agent Review] Critical issue in: {commit['subject'][:60]}"
            body = f"""## Automated Peer Review Finding

**Commit**: `{sha}`
**Subject**: {commit['subject']}
**Severity**: {severity}

### Issues Found
{chr(10).join(f'- {i}' for i in issues)}

### Diff
```
{diff[:3000]}
```

*Posted by peer_reviewer.py — multi-agent review loop*
"""
            if open_github_issue(title, body):
                issues_opened += 1
                print(f"  ✓ Opened GitHub issue for critical finding")

    mem["peer_reviewed_shas"] = reviewed[-100:]  # keep last 100
    save_memory(mem)

    # Commit updated memory
    try:
        subprocess.run(["git", "add", str(STATE_FILE)], capture_output=True)
        subprocess.run(["git", "commit", "-m",
                        f"state: peer_reviewer — {reviewed_count} commits reviewed, {issues_opened} issues opened",
                        "--allow-empty"],
                       capture_output=True,
                       env={**os.environ, "GIT_AUTHOR_NAME": "Peer Reviewer",
                            "GIT_AUTHOR_EMAIL": "reviewer@quantedge.ai",
                            "GIT_COMMITTER_NAME": "Peer Reviewer",
                            "GIT_COMMITTER_EMAIL": "reviewer@quantedge.ai"})
        subprocess.run(["git", "push"], capture_output=True)
    except Exception:
        pass

    print(f"\n✓ Peer review: {reviewed_count} commits, {issues_opened} issues opened")
    return 0

if __name__ == "__main__":
    sys.exit(main())
