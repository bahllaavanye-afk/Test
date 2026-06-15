"""
Notion ↔ GitHub Issues bidirectional sync.

Configure via env vars:
  NOTION_TOKEN          — Internal integration token (https://www.notion.so/my-integrations)
  NOTION_TASKS_DB_ID    — Database ID for the Engineering Tasks DB
  GITHUB_TOKEN          — PAT with `repo` scope
  GITHUB_REPO           — "owner/name", e.g. "bahllaavanye-afk/QuantEdge"

Activate by setting all four. When NOTION_TOKEN is unset, sync is skipped
silently so non-Notion users see no errors.

Notion DB schema expected (create once in the Notion UI):
  - Title        (Title)
  - Status       (Select: Backlog | In Progress | Blocked | Done)
  - Priority     (Select: P0 | P1 | P2 | P3)
  - Role         (Select: strategy | ml | risk | execution | data | broker |
                          frontend | backend | research | cto)
  - GitHub Issue (URL)
  - Sprint       (Select: v1.0 | v1.1 | v2.0)
  - Updated      (Last edited time)
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx

from app.utils.logging import logger

NOTION_API = "https://api.notion.com/v1"
GITHUB_API = "https://api.github.com"


class NotionSync:
    def __init__(
        self,
        notion_token: str | None = None,
        notion_db_id: str | None = None,
        github_token: str | None = None,
        github_repo: str | None = None,
    ):
        self.notion_token = notion_token or os.getenv("NOTION_TOKEN", "")
        self.notion_db_id = notion_db_id or os.getenv("NOTION_TASKS_DB_ID", "")
        self.github_token = github_token or os.getenv("GITHUB_TOKEN", "")
        self.github_repo = github_repo or os.getenv("GITHUB_REPO", "")
        self.enabled = bool(self.notion_token and self.notion_db_id and self.github_token and self.github_repo)
        if not self.enabled:
            logger.info("Notion sync disabled — missing one of NOTION_TOKEN, NOTION_TASKS_DB_ID, GITHUB_TOKEN, GITHUB_REPO")

    def _notion_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.notion_token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

    def _github_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def push_github_issue_to_notion(self, issue: dict) -> dict | None:
        """Create or update a Notion page from a GitHub issue payload."""
        if not self.enabled:
            return None

        title = issue["title"]
        url = issue["html_url"]
        state = "Done" if issue.get("state") == "closed" else "In Progress"
        labels = [lbl["name"] for lbl in issue.get("labels", [])]

        priority = next((p for p in ("P0", "P1", "P2", "P3") if any(f"priority:{p.lower()}" in lbl for lbl in labels)), "P2")
        role = next((r for r in ("strategy", "ml", "risk", "execution", "data", "broker", "frontend", "backend", "research", "cto")
                     if any(f"role:{r}" in lbl for lbl in labels)), "backend")

        async with httpx.AsyncClient(timeout=15) as client:
            existing = await client.post(
                f"{NOTION_API}/databases/{self.notion_db_id}/query",
                headers=self._notion_headers(),
                json={"filter": {"property": "GitHub Issue", "url": {"equals": url}}},
            )
            existing.raise_for_status()
            results = existing.json().get("results", [])

            payload = {
                "properties": {
                    "Title": {"title": [{"text": {"content": title}}]},
                    "Status": {"select": {"name": state}},
                    "Priority": {"select": {"name": priority}},
                    "Role": {"select": {"name": role}},
                    "GitHub Issue": {"url": url},
                }
            }

            if results:
                page_id = results[0]["id"]
                resp = await client.patch(
                    f"{NOTION_API}/pages/{page_id}",
                    headers=self._notion_headers(),
                    json=payload,
                )
            else:
                payload["parent"] = {"database_id": self.notion_db_id}
                resp = await client.post(
                    f"{NOTION_API}/pages",
                    headers=self._notion_headers(),
                    json=payload,
                )
            resp.raise_for_status()
            return resp.json()

    async def pull_notion_to_github(self) -> list[dict]:
        """Read Notion DB, create or update GitHub issues to match."""
        if not self.enabled:
            return []

        async with httpx.AsyncClient(timeout=30) as client:
            db = await client.post(
                f"{NOTION_API}/databases/{self.notion_db_id}/query",
                headers=self._notion_headers(),
            )
            db.raise_for_status()
            pages = db.json().get("results", [])

            synced = []
            for page in pages:
                props = page.get("properties", {})
                title = props.get("Title", {}).get("title", [{}])[0].get("text", {}).get("content", "")
                if not title:
                    continue
                gh_url = props.get("GitHub Issue", {}).get("url")
                status = props.get("Status", {}).get("select", {}).get("name", "Backlog")
                priority = props.get("Priority", {}).get("select", {}).get("name", "P2")
                role = props.get("Role", {}).get("select", {}).get("name", "backend")

                labels = [f"priority:{priority.lower()}", f"role:{role}"]
                state = "closed" if status == "Done" else "open"

                if gh_url:
                    # existing issue — update labels/state
                    issue_num = int(gh_url.rstrip("/").split("/")[-1])
                    resp = await client.patch(
                        f"{GITHUB_API}/repos/{self.github_repo}/issues/{issue_num}",
                        headers=self._github_headers(),
                        json={"labels": labels, "state": state, "title": title},
                    )
                else:
                    resp = await client.post(
                        f"{GITHUB_API}/repos/{self.github_repo}/issues",
                        headers=self._github_headers(),
                        json={"title": title, "labels": labels},
                    )
                resp.raise_for_status()
                synced.append({"title": title, "url": resp.json().get("html_url")})
            return synced

    async def sync_all(self) -> dict:
        """Bidirectional sweep: GitHub → Notion, then Notion → GitHub."""
        result = {"github_to_notion": 0, "notion_to_github": 0, "errors": []}
        if not self.enabled:
            result["errors"].append("notion_sync_disabled")
            return result

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                issues = await client.get(
                    f"{GITHUB_API}/repos/{self.github_repo}/issues",
                    headers=self._github_headers(),
                    params={"state": "all", "per_page": 100},
                )
                issues.raise_for_status()
                for issue in issues.json():
                    if "pull_request" in issue:
                        continue
                    await self.push_github_issue_to_notion(issue)
                    result["github_to_notion"] += 1

            synced_back = await self.pull_notion_to_github()
            result["notion_to_github"] = len(synced_back)
        except Exception as e:
            result["errors"].append(str(e))
            logger.error("notion_sync failed", error=str(e))

        result["synced_at"] = datetime.now(UTC).isoformat()
        return result


_singleton: NotionSync | None = None


def get_notion_sync() -> NotionSync:
    global _singleton
    if _singleton is None:
        _singleton = NotionSync()
    return _singleton
