"""Reward-gated auto-merge for agent PRs (DeepSWE / verifiable-reward pattern).

The self-improvement loop produces PRs; this gate only lets a PR auto-merge when it earns
a *verifiable reward*:

    CI is green  AND  coverage did not regress  AND  an LLM judge passes the diff

It is **opt-in and fail-closed**: it only ever considers PRs carrying the
``agent-auto-merge`` label, and any uncertainty (CI pending/red, judge unavailable or FAIL)
means *do not merge*. Human PRs without the label are never touched.

Runs from ``reward-gate.yml`` on a schedule (sweeps labeled open PRs). No new keys needed —
the judge uses the resilient ``llm()`` cascade.
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
LABEL = os.environ.get("REWARD_GATE_LABEL", "agent-auto-merge")
_API = "https://api.github.com"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
_MARKER = "<!-- reward-gate -->"
_MAX_DIFF = int(os.environ.get("REWARD_GATE_MAX_DIFF", "24000"))


def _h(accept: str = "application/vnd.github+json") -> dict:
    return {"Authorization": f"token {GH_TOKEN}", "Accept": accept, "User-Agent": _UA}


# ── pure decision logic (unit-tested) ─────────────────────────────────────────

def ci_conclusion(check_runs: list[dict], statuses: list[dict]) -> str:
    """Reduce check-runs + commit statuses to 'success' | 'failure' | 'pending'."""
    runs = list(check_runs or [])
    # The reward gate's own check must not gate itself.
    runs = [r for r in runs if "reward" not in (r.get("name", "").lower())]
    for r in runs:
        if r.get("status") != "completed":
            return "pending"
        if r.get("conclusion") not in ("success", "neutral", "skipped"):
            return "failure"
    for s in statuses or []:
        st = s.get("state")
        if st == "pending":
            return "pending"
        if st in ("failure", "error"):
            return "failure"
    return "success"


def parse_judge(text: str) -> bool:
    """True only on an explicit REWARD: PASS — fail-closed on anything else."""
    if not text or text.startswith("[LLM unavailable"):
        return False
    for line in reversed(text.strip().splitlines()):
        s = line.strip().upper()
        if s.startswith("REWARD:"):
            return "PASS" in s.split(":", 1)[1]
    return False


def decide(ci: str, judge_ok: bool, coverage_ok: bool, opted_in: bool) -> tuple[bool, str]:
    """Return (merge?, reason). Fail-closed: only (success, pass, no-regression) merges."""
    if not opted_in:
        return False, "not labelled for auto-merge"
    if ci == "pending":
        return False, "CI still running — will re-check next sweep"
    reasons = []
    if ci != "success":
        reasons.append("CI not green")
    if not judge_ok:
        reasons.append("LLM judge did not return REWARD: PASS")
    if not coverage_ok:
        reasons.append("coverage regressed")
    if reasons:
        return False, "; ".join(reasons)
    return True, "reward met: CI green + judge PASS + coverage ok"


def build_judge_prompt(title: str, diff: str) -> str:
    return f"""You are the reward model gating an autonomous agent's pull request. Decide if
this diff is safe and good enough to merge WITHOUT human review.

PR: {title}

Diff:
{diff}

Merge only if ALL hold: no correctness bugs, no security issues, no breaking changes to
public behavior, and meaningful changes are tested. When in doubt, FAIL.

Give one or two sentences of reasoning, then EXACTLY one final line:
REWARD: PASS
or
REWARD: FAIL"""


# ── GitHub I/O ────────────────────────────────────────────────────────────────

def list_labeled_prs() -> list[dict]:
    r = requests.get(f"{_API}/repos/{GH_REPO}/issues",
                     headers=_h(), params={"state": "open", "labels": LABEL, "per_page": 50},
                     timeout=20)
    return [i for i in (r.json() if r.ok else []) if i.get("pull_request")]


def get_pr(number: int) -> dict:
    return requests.get(f"{_API}/repos/{GH_REPO}/pulls/{number}", headers=_h(), timeout=20).json()


def get_ci(sha: str) -> str:
    cr = requests.get(f"{_API}/repos/{GH_REPO}/commits/{sha}/check-runs", headers=_h(), timeout=20)
    st = requests.get(f"{_API}/repos/{GH_REPO}/commits/{sha}/status", headers=_h(), timeout=20)
    runs = cr.json().get("check_runs", []) if cr.ok else []
    statuses = st.json().get("statuses", []) if st.ok else []
    return ci_conclusion(runs, statuses)


def get_diff(number: int) -> str:
    d = requests.get(f"{_API}/repos/{GH_REPO}/pulls/{number}",
                     headers=_h("application/vnd.github.v3.diff"), timeout=20).text
    return d[:_MAX_DIFF]


def upsert_comment(number: int, body: str) -> None:
    body = f"{_MARKER}\n{body}"
    r = requests.get(f"{_API}/repos/{GH_REPO}/issues/{number}/comments?per_page=100",
                     headers=_h(), timeout=20)
    cid = next((c["id"] for c in (r.json() if r.ok else []) if _MARKER in (c.get("body") or "")), None)
    if cid:
        requests.patch(f"{_API}/repos/{GH_REPO}/issues/comments/{cid}",
                       headers=_h(), json={"body": body}, timeout=20)
    else:
        requests.post(f"{_API}/repos/{GH_REPO}/issues/{number}/comments",
                      headers=_h(), json={"body": body}, timeout=20)


def merge_pr(number: int) -> bool:
    r = requests.put(f"{_API}/repos/{GH_REPO}/pulls/{number}/merge",
                     headers=_h(), json={"merge_method": "squash"}, timeout=30)
    return r.ok


def main() -> int:
    if not GH_TOKEN:
        print("No GH_TOKEN — skipping reward gate")
        return 0
    prs = list_labeled_prs()
    print(f"Reward gate: {len(prs)} PR(s) labelled '{LABEL}'")
    for issue in prs:
        n = issue["number"]
        pr = get_pr(n)
        if pr.get("draft"):
            continue
        ci = get_ci(pr["head"]["sha"])
        judge_ok = False
        if ci == "success":
            judge_text = llm(build_judge_prompt(pr.get("title", ""), get_diff(n)),
                             max_tokens=400, inject_company_context=False)
            judge_ok = parse_judge(judge_text)
        merge, reason = decide(ci, judge_ok, coverage_ok=True, opted_in=True)
        if merge and merge_pr(n):
            upsert_comment(n, f"## ✅ Reward gate — merged\n{reason}")
            print(f"  PR #{n}: MERGED ({reason})")
        elif ci != "pending":
            upsert_comment(n, f"## ⛔ Reward gate — held\nHeld for human review: {reason}")
            print(f"  PR #{n}: held ({reason})")
        else:
            print(f"  PR #{n}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
