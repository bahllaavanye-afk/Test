"""
GitHub issue escalation bridge.

The QA Monitor finds problems every 5 minutes and auto-fixes what it can
(deprecated APIs). Whatever it *cannot* fix automatically — a real test
failure, a critical/high security finding, an import error — needs a human
or Claude to look at it. This module turns those un-fixable findings into
labelled GitHub issues so a team lead (or Claude) can review and resolve them.

Design notes:
  - Gated on GITHUB_TOKEN + GITHUB_REPO. When unset, escalation is skipped
    silently (no crash) — same convention as notion_sync.py.
  - Deduplicated: every issue carries a stable fingerprint in an HTML comment
    in its body. Before opening, we list open `auto:qa` issues and skip any
    fingerprint that already has an open issue. This prevents the 5-minute
    loop from spamming the tracker.
  - Labelled for routing: `auto:qa`, `needs-review`, `priority:<p>`,
    `role:<team>` so the right desk lead picks it up.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from app.utils.logging import logger

if TYPE_CHECKING:
    from app.tasks.qa_monitor import QAReport

GITHUB_API = "https://api.github.com"

# Max issues opened per cycle — a safety valve against a sudden flood of
# failures (e.g. a broken import that fails 200 tests) turning into 200 issues.
MAX_ISSUES_PER_CYCLE = 8

# Map a source path to the desk/team lead that owns it.
_ROLE_BY_PREFIX = [
    ("app/strategies", "strategy"),
    ("app/ml", "ml"),
    ("app/risk", "risk"),
    ("app/execution", "execution"),
    ("app/brokers", "broker"),
    ("app/tasks", "data"),
    ("app/api", "backend"),
    ("app/ws", "backend"),
    ("frontend", "frontend"),
    ("tests", "backend"),
]


def _role_for_path(path: str) -> str:
    norm = path.replace("\\", "/")
    for prefix, role in _ROLE_BY_PREFIX:
        if prefix in norm:
            return role
    return "backend"


def _fingerprint(kind: str, key: str, detail: str = "") -> str:
    """Stable short hash identifying a recurring problem across cycles."""
    raw = f"{kind}|{key}|{detail}".encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:12]


class IssueEscalator:
    def __init__(self, github_token: str | None = None, github_repo: str | None = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN", "")
        self.github_repo = github_repo or os.getenv("GITHUB_REPO", "")
        # Allow turning escalation off without removing the token.
        self._opt_out = os.getenv("QA_GITHUB_ESCALATION", "1").lower() in ("0", "false", "no")
        self.enabled = bool(self.github_token and self.github_repo and not self._opt_out)
        if not self.enabled and not self._opt_out:
            logger.info(
                "Issue escalation disabled — set GITHUB_TOKEN and GITHUB_REPO to enable "
                "(QA findings will still be written to the health report)."
            )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    # ------------------------------------------------------------------
    # Build the escalation list from a QA report
    # ------------------------------------------------------------------
    @staticmethod
    def build_escalations(report: "QAReport") -> list[dict]:
        """
        Turn a QAReport into a list of escalation dicts. Only includes findings
        that the monitor could NOT auto-fix and that genuinely need review.

        Returns dicts with: fingerprint, title, body, priority, role.
        """
        escalations: list[dict] = []

        # 1. Import errors — almost always a real breakage (P0).
        for err in report.import_errors:
            module = err.split(":", 1)[0].strip()
            fp = _fingerprint("import_error", module)
            escalations.append({
                "fingerprint": fp,
                "title": f"[QA] Import error in {module}",
                "body": (
                    f"The autonomous QA Monitor could not import `{module}`.\n\n"
                    f"```\n{err}\n```\n\n"
                    "This breaks startup and any code path that depends on the module. "
                    "Assigned for review by Claude / the owning team lead."
                ),
                "priority": "P0",
                "role": _role_for_path(module.replace(".", "/")),
            })

        # 2. Critical / high security findings that are not auto-fixable.
        for issue in report.security_issues:
            if issue.severity in ("critical", "high") and not issue.auto_fixable:
                fp = _fingerprint("security", f"{issue.file_path}:{issue.issue_type}")
                priority = "P0" if issue.severity == "critical" else "P1"
                escalations.append({
                    "fingerprint": fp,
                    "title": f"[QA] {issue.severity.title()} security: {issue.issue_type} in {issue.file_path}",
                    "body": (
                        f"**Severity:** {issue.severity}\n"
                        f"**Type:** {issue.issue_type}\n"
                        f"**Location:** `{issue.file_path}:{issue.line_number}`\n\n"
                        f"{issue.description}\n\n"
                        "Detected by the autonomous QA Monitor; needs manual review "
                        "(not auto-fixable). Assigned for Claude / team-lead review."
                    ),
                    "priority": priority,
                    "role": _role_for_path(issue.file_path),
                })

        # 3. Test failures the monitor could not auto-fix (P1).
        for fail in report.test_failures:
            if fail.fixable:
                continue  # the monitor will handle these itself
            fp = _fingerprint("test_failure", fail.test_id, fail.error_type)
            escalations.append({
                "fingerprint": fp,
                "title": f"[QA] Test failing: {fail.test_id}",
                "body": (
                    f"**Test:** `{fail.test_id}`\n"
                    f"**Error:** {fail.error_type}\n"
                    f"**File:** `{fail.file_path}`"
                    + (f":{fail.line_number}" if fail.line_number else "")
                    + "\n\n"
                    f"```\n{fail.error_msg}\n```\n\n"
                    "Surfaced by the autonomous QA Monitor. Assigned for Claude / "
                    "team-lead review."
                ),
                "priority": "P1",
                "role": _role_for_path(fail.file_path),
            })

        return escalations

    # ------------------------------------------------------------------
    # GitHub interaction
    # ------------------------------------------------------------------
    async def _open_fingerprints(self, client: httpx.AsyncClient) -> set[str]:
        """Return fingerprints that already have an OPEN auto:qa issue."""
        seen: set[str] = set()
        resp = await client.get(
            f"{GITHUB_API}/repos/{self.github_repo}/issues",
            headers=self._headers(),
            params={"state": "open", "labels": "auto:qa", "per_page": 100},
        )
        resp.raise_for_status()
        for issue in resp.json():
            body = issue.get("body") or ""
            marker = "<!-- qa-fingerprint:"
            if marker in body:
                fp = body.split(marker, 1)[1].split("-->", 1)[0].strip()
                seen.add(fp)
        return seen

    async def escalate(self, report: "QAReport") -> dict:
        """
        Open GitHub issues for un-fixable QA findings, skipping ones already
        tracked. Returns a summary dict.
        """
        summary = {"opened": 0, "skipped_existing": 0, "candidates": 0, "errors": []}
        if not self.enabled:
            summary["errors"].append("escalation_disabled")
            return summary

        candidates = self.build_escalations(report)
        summary["candidates"] = len(candidates)
        if not candidates:
            return summary

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                existing = await self._open_fingerprints(client)

                opened = 0
                for esc in candidates:
                    if opened >= MAX_ISSUES_PER_CYCLE:
                        break
                    fp = esc["fingerprint"]
                    if fp in existing:
                        summary["skipped_existing"] += 1
                        continue

                    body = (
                        f"{esc['body']}\n\n"
                        f"---\n"
                        f"_Auto-filed by QA Monitor at "
                        f"{datetime.now(timezone.utc).isoformat()}_\n"
                        f"<!-- qa-fingerprint: {fp} -->"
                    )
                    labels = [
                        "auto:qa",
                        "needs-review",
                        f"priority:{esc['priority'].lower()}",
                        f"role:{esc['role']}",
                    ]
                    create = await client.post(
                        f"{GITHUB_API}/repos/{self.github_repo}/issues",
                        headers=self._headers(),
                        json={"title": esc["title"], "body": body, "labels": labels},
                    )
                    create.raise_for_status()
                    existing.add(fp)  # guard against dup candidates in same batch
                    opened += 1
                    logger.info(
                        "QA escalation: opened issue",
                        title=esc["title"],
                        url=create.json().get("html_url"),
                    )

                summary["opened"] = opened
        except httpx.HTTPError as e:
            summary["errors"].append(str(e))
            logger.warning("QA escalation: GitHub API error", error=str(e))
        except Exception as e:  # noqa: BLE001 — never let escalation crash the QA loop
            summary["errors"].append(str(e))
            logger.warning("QA escalation: unexpected error", error=str(e))

        return summary


_singleton: IssueEscalator | None = None


def get_escalator() -> IssueEscalator:
    global _singleton
    if _singleton is None:
        _singleton = IssueEscalator()
    return _singleton
