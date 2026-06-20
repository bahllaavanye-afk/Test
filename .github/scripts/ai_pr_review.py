"""AI PR Reviewer — a "Cursor-style" automated review on every pull request.

Runs from `ai-pr-review.yml` on pull_request open/synchronize. Pulls the PR diff,
reviews it with the free-LLM cascade (which auto-escalates to OpenRouter→Claude when the
free tier is down, so a review always lands), and posts ONE upserted review comment on
the PR (updated in place on each push — no repeated-message noise).

Advisory by design: it never fails the PR's checks (exit 0), it just leaves a review.

Env: GH_TOKEN (PR write), GH_REPO (owner/repo), PR_NUMBER, plus LLM provider keys.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm  # noqa: E402

GH_TOKEN = os.environ.get("GH_TOKEN", "")
GH_REPO = os.environ.get("GH_REPO", "bahllaavanye-afk/test")
PR_NUMBER = os.environ.get("PR_NUMBER", "")
_API = "https://api.github.com"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
_MARKER = "<!-- ai-pr-review -->"
_MAX_DIFF = int(os.environ.get("AI_REVIEW_MAX_DIFF_CHARS", "28000"))


def _headers(accept: str = "application/vnd.github+json") -> dict:
    return {"Authorization": f"token {GH_TOKEN}", "Accept": accept, "User-Agent": _UA}


def fetch_pr_diff(repo: str, number: str) -> tuple[str, str]:
    """Return (title, unified_diff) for a PR. Diff is truncated to _MAX_DIFF chars."""
    meta = requests.get(f"{_API}/repos/{repo}/pulls/{number}", headers=_headers(), timeout=20)
    title = meta.json().get("title", "") if meta.ok else ""
    diff = requests.get(
        f"{_API}/repos/{repo}/pulls/{number}",
        headers=_headers("application/vnd.github.v3.diff"), timeout=20,
    ).text
    if len(diff) > _MAX_DIFF:
        diff = diff[:_MAX_DIFF] + "\n\n[...diff truncated for review...]"
    return title, diff


def build_review_prompt(title: str, diff: str) -> str:
    return f"""You are a meticulous senior engineer doing a pull-request review (like Cursor's
bot). Review the diff for THIS PR and report only what matters.

PR title: {title}

Diff:
{diff}

Focus, in priority order:
1. Correctness bugs that would break at runtime (logic errors, bad imports, None/edge cases)
2. Security (hardcoded secrets, injection, auth/permission gaps, unsafe input)
3. Missing error handling on IO/network/DB
4. Breaking changes to existing behavior or APIs
5. Tests: are the changes covered? call out untested risk
6. Quick wins: clear simplifications or reuse

Be concise and specific — reference file/areas. Skip nits and style. If it's clean, say so.

End with EXACTLY one line:
VERDICT: APPROVE   (no blocking issues)
or
VERDICT: COMMENT   (non-blocking suggestions)
or
VERDICT: REQUEST_CHANGES   (a real bug/security issue a human should look at)
"""


def parse_verdict(review: str) -> str:
    for line in reversed(review.strip().splitlines()):
        s = line.strip().upper()
        if s.startswith("VERDICT:"):
            v = s.split(":", 1)[1].strip()
            if "REQUEST_CHANGES" in v:
                return "REQUEST_CHANGES"
            if "APPROVE" in v:
                return "APPROVE"
            return "COMMENT"
    return "COMMENT"


def format_comment(review: str, verdict: str) -> str:
    badge = {"APPROVE": "✅ Approve", "COMMENT": "💬 Comment",
             "REQUEST_CHANGES": "🛠️ Changes suggested"}.get(verdict, "💬 Comment")
    return (
        f"{_MARKER}\n## 🤖 AI PR Review — {badge}\n\n{review.strip()}\n\n"
        f"---\n*Automated review by `ai_pr_review.py` (free-LLM cascade → OpenRouter → "
        f"Claude). Advisory only — never blocks merge. Re-runs update this comment.*"
    )


def find_existing_comment(repo: str, number: str) -> int | None:
    r = requests.get(f"{_API}/repos/{repo}/issues/{number}/comments?per_page=100",
                     headers=_headers(), timeout=20)
    if not r.ok:
        return None
    for c in r.json():
        if _MARKER in (c.get("body") or ""):
            return c.get("id")
    return None


def upsert_comment(repo: str, number: str, body: str) -> bool:
    """Update the existing AI-review comment if present, else create one."""
    cid = find_existing_comment(repo, number)
    if cid:
        r = requests.patch(f"{_API}/repos/{repo}/issues/comments/{cid}",
                           headers=_headers(), json={"body": body}, timeout=20)
    else:
        r = requests.post(f"{_API}/repos/{repo}/issues/{number}/comments",
                          headers=_headers(), json={"body": body}, timeout=20)
    return r.ok


def main() -> int:
    if not (GH_TOKEN and PR_NUMBER):
        print("Missing GH_TOKEN or PR_NUMBER — skipping AI review")
        return 0
    title, diff = fetch_pr_diff(GH_REPO, PR_NUMBER)
    if not diff.strip():
        print("Empty diff — nothing to review")
        return 0
    review = llm(build_review_prompt(title, diff), max_tokens=900,
                 inject_company_context=False)
    if not review or review.startswith("[LLM unavailable"):
        print("LLM unavailable — posting a soft note")
        upsert_comment(GH_REPO, PR_NUMBER,
                       f"{_MARKER}\n## 🤖 AI PR Review\n\n_Review skipped: the LLM cascade "
                       f"was unavailable. The brain-health canary tracks this._")
        return 0
    verdict = parse_verdict(review)
    ok = upsert_comment(GH_REPO, PR_NUMBER, format_comment(review, verdict))
    print(f"AI review posted ({verdict}); upsert_ok={ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
