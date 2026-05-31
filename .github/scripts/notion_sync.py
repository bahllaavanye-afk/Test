"""
One-shot Notion sync run by .github/workflows/notion-sync.yml.

Steps:
1. Discover (or create) the "QuantEdge Engineering Tasks" database in the user's
   Notion workspace. Requires either an existing DB id, OR a parent page id
   under which we'll create the DB fresh.
2. Pull all GitHub issues for the repo.
3. Upsert each issue as a Notion page in the database, keyed by GitHub URL.
"""
from __future__ import annotations

import os
import sys
import time

import httpx


NOTION_API = "https://api.notion.com/v1"
GITHUB_API = "https://api.github.com"
NOTION_VERSION = "2022-06-28"


def notion_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def whoami(token: str) -> dict:
    r = httpx.get(f"{NOTION_API}/users/me", headers=notion_headers(token), timeout=10)
    r.raise_for_status()
    return r.json()


def search_existing_db(token: str, title_match: str) -> str | None:
    """Search Notion for a database whose title contains title_match."""
    r = httpx.post(
        f"{NOTION_API}/search",
        headers=notion_headers(token),
        json={
            "query": title_match,
            "filter": {"value": "database", "property": "object"},
        },
        timeout=15,
    )
    r.raise_for_status()
    for result in r.json().get("results", []):
        title_parts = result.get("title", [])
        title_text = "".join(p.get("plain_text", "") for p in title_parts)
        if title_match.lower() in title_text.lower():
            return result["id"]
    return None


def list_accessible_pages(token: str) -> list[dict]:
    """List top-level pages the integration can access — useful for picking a parent."""
    r = httpx.post(
        f"{NOTION_API}/search",
        headers=notion_headers(token),
        json={"filter": {"value": "page", "property": "object"}, "page_size": 25},
        timeout=15,
    )
    r.raise_for_status()
    pages = []
    for item in r.json().get("results", []):
        title_parts = item.get("properties", {}).get("title", {}).get("title", []) \
            or item.get("properties", {}).get("Name", {}).get("title", [])
        title_text = "".join(p.get("plain_text", "") for p in title_parts) or "(untitled)"
        pages.append({"id": item["id"], "title": title_text})
    return pages


def create_database(token: str, parent_page_id: str) -> str:
    """Create the Engineering Tasks DB under a parent page."""
    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "icon": {"type": "emoji", "emoji": "⚙️"},
        "title": [{"type": "text", "text": {"content": "QuantEdge Engineering Tasks"}}],
        "properties": {
            "Title": {"title": {}},
            "Status": {
                "select": {
                    "options": [
                        {"name": "Backlog", "color": "gray"},
                        {"name": "In Progress", "color": "blue"},
                        {"name": "Blocked", "color": "red"},
                        {"name": "Done", "color": "green"},
                    ]
                }
            },
            "Priority": {
                "select": {
                    "options": [
                        {"name": "P0", "color": "red"},
                        {"name": "P1", "color": "orange"},
                        {"name": "P2", "color": "yellow"},
                        {"name": "P3", "color": "default"},
                    ]
                }
            },
            "Role": {
                "select": {
                    "options": [
                        {"name": "strategy", "color": "purple"},
                        {"name": "ml", "color": "blue"},
                        {"name": "risk", "color": "red"},
                        {"name": "execution", "color": "orange"},
                        {"name": "data", "color": "yellow"},
                        {"name": "broker", "color": "green"},
                        {"name": "frontend", "color": "pink"},
                        {"name": "backend", "color": "brown"},
                        {"name": "research", "color": "blue"},
                        {"name": "platform", "color": "gray"},
                        {"name": "cto", "color": "red"},
                    ]
                }
            },
            "GitHub Issue": {"url": {}},
            "GitHub #": {"number": {"format": "number"}},
            "Sprint": {
                "select": {
                    "options": [
                        {"name": "v1.0", "color": "blue"},
                        {"name": "v1.1", "color": "purple"},
                        {"name": "v2.0", "color": "green"},
                    ]
                }
            },
        },
    }
    r = httpx.post(f"{NOTION_API}/databases", headers=notion_headers(token), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def fetch_github_issues(repo: str, token: str) -> list[dict]:
    out = []
    page = 1
    while True:
        r = httpx.get(
            f"{GITHUB_API}/repos/{repo}/issues",
            headers=github_headers(token),
            params={"state": "all", "per_page": 100, "page": page},
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend([i for i in batch if "pull_request" not in i])
        page += 1
        if page > 5:  # safety cap
            break
    return out


def upsert_notion_page(token: str, db_id: str, issue: dict) -> dict:
    title = issue["title"]
    url = issue["html_url"]
    state = "Done" if issue.get("state") == "closed" else ("Blocked" if "blocked" in [l["name"] for l in issue.get("labels", [])] else "In Progress")
    labels = [lbl["name"] for lbl in issue.get("labels", [])]

    priority = next(
        (p for p in ("P0", "P1", "P2", "P3") if f"priority:{p.lower()}" in labels),
        "P2",
    )
    role_map = ("strategy", "ml", "risk", "execution", "data", "broker", "frontend", "backend", "research", "platform", "cto")
    role = next(
        (r for r in role_map if f"role:{r}" in labels),
        "backend",
    )

    properties = {
        "Title": {"title": [{"text": {"content": title}}]},
        "Status": {"select": {"name": state}},
        "Priority": {"select": {"name": priority}},
        "Role": {"select": {"name": role}},
        "GitHub Issue": {"url": url},
        "GitHub #": {"number": issue["number"]},
        "Sprint": {"select": {"name": "v1.0"}},
    }

    # Check if a page with this GitHub URL already exists
    query = httpx.post(
        f"{NOTION_API}/databases/{db_id}/query",
        headers=notion_headers(token),
        json={"filter": {"property": "GitHub Issue", "url": {"equals": url}}},
        timeout=15,
    )
    query.raise_for_status()
    existing = query.json().get("results", [])

    if existing:
        page_id = existing[0]["id"]
        r = httpx.patch(
            f"{NOTION_API}/pages/{page_id}",
            headers=notion_headers(token),
            json={"properties": properties},
            timeout=15,
        )
        r.raise_for_status()
        return {"action": "updated", "title": title, "url": url}
    else:
        r = httpx.post(
            f"{NOTION_API}/pages",
            headers=notion_headers(token),
            json={"parent": {"database_id": db_id}, "properties": properties},
            timeout=15,
        )
        r.raise_for_status()
        return {"action": "created", "title": title, "url": url}


def main() -> int:
    token = os.environ["NOTION_TOKEN"]
    parent_page = os.environ.get("NOTION_PARENT_PAGE", "").strip()
    db_id = os.environ.get("NOTION_DB_ID", "").strip()
    gh_token = os.environ["GITHUB_TOKEN"]
    gh_repo = os.environ["GITHUB_REPO"]

    print("🔑 Verifying Notion token...")
    me = whoami(token)
    bot_name = me.get("name") or "(unknown)"
    bot_workspace = me.get("bot", {}).get("workspace_name") or "(unknown)"
    print(f"   ✓ Authenticated as bot '{bot_name}' in workspace '{bot_workspace}'")

    if not db_id:
        print("\n🔍 Looking for existing 'QuantEdge Engineering Tasks' database...")
        db_id = search_existing_db(token, "QuantEdge Engineering Tasks")
        if db_id:
            print(f"   ✓ Found existing DB: {db_id}")

    if not db_id and parent_page:
        print(f"\n🆕 Creating new database under page {parent_page}...")
        db_id = create_database(token, parent_page)
        print(f"   ✓ Created DB: {db_id}")

    if not db_id:
        print("\n⚠️  No database available. Options:")
        print("  A) Re-run this workflow with `parent_page_id` set to a Notion page ID")
        print("     where the integration has Edit access.")
        print("  B) Re-run with `existing_db_id` set to an existing DB ID.")
        print("\nPages the integration can currently see:")
        pages = list_accessible_pages(token)
        if not pages:
            print("  (none — you need to share at least one page with this integration:")
            print("   open a Notion page → ⋮ → Connections → add the 'QuantEdge connection')")
        for p in pages:
            print(f"  - {p['title']:50s} {p['id']}")
        return 2

    print(f"\n📥 Fetching GitHub issues from {gh_repo}...")
    issues = fetch_github_issues(gh_repo, gh_token)
    print(f"   ✓ {len(issues)} issues fetched")

    print(f"\n📤 Upserting into Notion DB {db_id}...")
    created = updated = 0
    for issue in issues:
        try:
            result = upsert_notion_page(token, db_id, issue)
            if result["action"] == "created":
                created += 1
            else:
                updated += 1
            print(f"   {result['action']:8s} #{issue['number']}: {result['title']}")
            time.sleep(0.35)  # Notion rate limit: ~3 req/s
        except Exception as e:
            print(f"   ✗ failed #{issue['number']}: {e}")

    print(f"\n✅ Sync complete: {created} created, {updated} updated")
    print(f"   Notion DB: https://www.notion.so/{db_id.replace('-', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
