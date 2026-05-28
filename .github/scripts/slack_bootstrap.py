#!/usr/bin/env python3
"""
Slack channel bootstrap, callable from GitHub Actions.
Reads SLACK_BOT_TOKEN, GH_TOKEN, GH_REPO from env.
Always posts the result as a comment on GitHub issue #2.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from urllib.request import Request, urlopen
from urllib.error import HTTPError


SLACK_API = "https://slack.com/api"
GH_API = "https://api.github.com"


ENGINEERING_CHANNELS: list[tuple[str, bool, str]] = [
    ("engineering-standup",  False, "Daily standups from each squad (13:00 UTC)"),
    ("alpha-research",       False, "New strategy proposals + paper reviews"),
    ("pnl-daily",            False, "EOD P&L attribution by strategy"),
    ("risk-alerts",          False, "VaR breaches, circuit breakers"),
    ("incidents",            False, "P0/P1 incidents and postmortems"),
    ("deploys",              False, "Deploy notifications"),
    ("ci-failures",          False, "CI test failures (auto-routed)"),
    ("ml-experiments",       False, "Training run results, model leaderboard"),
    ("engineering",          False, "All engineers"),
    ("announcements",        False, "Company-wide announcements (CEO only posts)"),
    ("wins",                 False, "Celebrate shipped features and winning strategies"),
    ("help",                 False, "Anyone can ask, anyone answers"),
    ("squad-alpha-research", True,  "Alpha Research squad"),
    ("squad-microstructure", True,  "Microstructure squad"),
    ("squad-ml-modeling",    True,  "ML Modeling squad"),
    ("squad-ml-infra",       True,  "ML Infrastructure squad"),
    ("squad-backend",        True,  "Backend Platform squad"),
    ("squad-frontend",       True,  "Frontend squad"),
    ("squad-data",           True,  "Data Engineering squad"),
    ("squad-execution",      True,  "Execution squad"),
    ("squad-risk",           True,  "Risk Engineering squad"),
    ("squad-security",       True,  "Security squad"),
    ("squad-devops",         True,  "DevOps / SRE squad"),
    ("squad-qa",             True,  "QA / Test Automation squad"),
    ("squad-compliance",     True,  "Compliance Engineering"),
    ("squad-finance-eng",    True,  "Finance Engineering"),
    ("leadership",           True,  "VP+ only"),
    ("leadership-summary",   True,  "Daily auto-summaries from each VP"),
    ("board",                True,  "CEO + CFO + CTO + board observers"),
    ("pm-coordination",      True,  "All PMs cross-coordinate"),
]


def http_post(url: str, headers: dict, body: dict) -> tuple[int, dict]:
    """POST JSON; return (status, parsed body or {raw: ...})."""
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urlopen(req, timeout=20)
        raw = resp.read().decode("utf-8")
        return resp.getcode(), (json.loads(raw) if raw else {})
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw}


def http_get(url: str, headers: dict) -> tuple[int, dict]:
    req = Request(url, headers=headers, method="GET")
    try:
        resp = urlopen(req, timeout=20)
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        return e.code, {"raw": e.read().decode("utf-8", errors="replace")}


def slack_call(token: str, method: str, body: dict | None = None) -> dict:
    """POST to Slack API; raise on slack error."""
    code, data = http_post(
        f"{SLACK_API}/{method}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        body=body or {},
    )
    if code != 200:
        raise RuntimeError(f"slack.{method} HTTP {code}: {data}")
    if not data.get("ok"):
        # Surface full Slack response including 'needed' scope hints
        raise RuntimeError(
            f"slack.{method} error: '{data.get('error')}' "
            f"needed={data.get('needed')} provided={data.get('provided')} "
            f"full={data}"
        )
    return data


def post_github_comment(gh_token: str, repo: str, issue: int, body_md: str) -> None:
    url = f"{GH_API}/repos/{repo}/issues/{issue}/comments"
    code, data = http_post(
        url,
        headers={
            "Authorization": f"Bearer {gh_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        body={"body": body_md},
    )
    print(f"[gh comment] POST {url} → HTTP {code}")
    if code >= 300:
        print(f"[gh comment] response: {data}")


def main() -> int:
    log: list[str] = []
    def L(msg: str) -> None:
        print(msg, flush=True)
        log.append(msg)

    gh_token = os.environ.get("GH_TOKEN", "").strip()
    gh_repo = os.environ.get("GH_REPO", "").strip()
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()

    L(f"Repo: {gh_repo}")
    L(f"GH token: {'set' if gh_token else 'MISSING'}")
    L(f"Slack token prefix: {token[:11] if token else 'EMPTY'}..., length={len(token)}")

    success = False
    try:
        if not token:
            raise SystemExit("SLACK_BOT_TOKEN env var is empty")
        if not token.startswith("xoxb-"):
            raise SystemExit(f"Token must start with 'xoxb-' (got '{token[:10]}...')")

        # 1) auth.test
        L("\n=== auth.test ===")
        auth = slack_call(token, "auth.test")
        L(f"✅ Authenticated as bot: {auth.get('user')} (id={auth.get('user_id')})")
        L(f"   Team: {auth.get('team')} (id={auth.get('team_id')})")
        L(f"   URL: {auth.get('url')}")

        # 2) list existing
        L("\n=== conversations.list (check existing) ===")
        try:
            existing = []
            cursor = ""
            while True:
                d = slack_call(token, "conversations.list", {
                    "types": "public_channel,private_channel",
                    "limit": 200,
                    "cursor": cursor,
                })
                existing.extend(d.get("channels", []))
                cursor = d.get("response_metadata", {}).get("next_cursor", "")
                if not cursor:
                    break
            existing_names = {c.get("name") for c in existing}
            L(f"📋 {len(existing)} channels already in workspace")
        except Exception as e:
            L(f"⚠️ conversations.list failed (probably missing scope channels:read or groups:read): {e}")
            existing_names = set()

        # 3) create each
        L(f"\n=== create {len(ENGINEERING_CHANNELS)} channels ===")
        created, skipped, errors = [], [], []
        for name, is_private, topic in ENGINEERING_CHANNELS:
            if name in existing_names:
                skipped.append(name)
                L(f"  = #{name} (exists)")
                continue
            try:
                d = slack_call(token, "conversations.create",
                               {"name": name, "is_private": is_private})
                cid = d.get("channel", {}).get("id")
                vis = "🔒 private" if is_private else "🌐 public"
                L(f"  + #{name} {vis} (id={cid})")
                created.append(name)
                if topic and cid:
                    try:
                        slack_call(token, "conversations.setTopic",
                                   {"channel": cid, "topic": topic})
                    except Exception as te:
                        L(f"    ! setTopic on #{name} failed: {te}")
            except Exception as ce:
                errors.append((name, str(ce)))
                L(f"  ✗ #{name} FAILED: {ce}")

        L(f"\n========== SUMMARY ==========")
        L(f"✅ Created:        {len(created)}")
        L(f"⏭  Already existed: {len(skipped)}")
        L(f"❌ Errors:         {len(errors)}")

        success = len(errors) == 0
        if errors:
            L("\nERROR DETAILS:")
            for name, err in errors:
                L(f"  - {name}: {err}")

    except SystemExit as e:
        L(f"\n❌ FATAL: {e}")
    except Exception as e:
        L(f"\n❌ EXCEPTION: {e}")
        L(traceback.format_exc())

    # Always post to GitHub issue #2
    if gh_token and gh_repo:
        body = (
            f"## Slack bootstrap result\n\n"
            f"_Status: {'✅ SUCCESS' if success else '❌ FAILED'}_\n\n"
            f"```\n{chr(10).join(log)}\n```"
        )
        try:
            post_github_comment(gh_token, gh_repo, 2, body)
            L("✓ Posted result to issue #2")
        except Exception as e:
            L(f"⚠️ Could not post to issue #2: {e}")
    else:
        L("(skipping issue comment — GH_TOKEN or GH_REPO not set)")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
