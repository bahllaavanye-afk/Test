"""
Standalone Slack bootstrap script — no external dependencies, stdlib only.
Creates all 31 engineering-org channels in the configured Slack workspace.

Usage (run from repo root):
    SLACK_BOT_TOKEN=xoxb-... python .github/scripts/slack_bootstrap.py

Optional env vars:
    GH_TOKEN   — GitHub PAT to post result comment to issue #2
    GH_REPO    — owner/repo (e.g. bahllaavanye-afk/Test)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from io import StringIO

# ── Channel definitions (mirrors backend/app/integrations/slack_bot.py) ──────

CHANNELS = [
    # Public — engineering ops
    {"name": "engineering-standup",  "is_private": False, "topic": "Daily standups from each squad (13:00 UTC)"},
    {"name": "alpha-research",       "is_private": False, "topic": "New strategy proposals + paper reviews"},
    {"name": "pnl-daily",            "is_private": False, "topic": "EOD P&L attribution by strategy"},
    {"name": "risk-alerts",          "is_private": False, "topic": "VaR breaches, circuit breakers"},
    {"name": "incidents",            "is_private": False, "topic": "P0/P1 incidents and postmortems"},
    {"name": "deploys",              "is_private": False, "topic": "Deploy notifications"},
    {"name": "ci-failures",          "is_private": False, "topic": "CI test failures (auto-routed)"},
    {"name": "ml-experiments",       "is_private": False, "topic": "Training run results, model leaderboard"},
    # Public — general
    {"name": "engineering",          "is_private": False, "topic": "All engineers"},
    {"name": "announcements",        "is_private": False, "topic": "Company-wide announcements (CEO only posts)"},
    {"name": "wins",                 "is_private": False, "topic": "Celebrate shipped features and winning strategies"},
    {"name": "help",                 "is_private": False, "topic": "Anyone can ask, anyone answers"},
    # Private — squads
    {"name": "squad-alpha-research", "is_private": True,  "topic": "Alpha Research squad"},
    {"name": "squad-microstructure", "is_private": True,  "topic": "Microstructure squad"},
    {"name": "squad-ml-modeling",    "is_private": True,  "topic": "ML Modeling squad"},
    {"name": "squad-ml-infra",       "is_private": True,  "topic": "ML Infrastructure squad"},
    {"name": "squad-backend",        "is_private": True,  "topic": "Backend Platform squad"},
    {"name": "squad-frontend",       "is_private": True,  "topic": "Frontend squad"},
    {"name": "squad-data",           "is_private": True,  "topic": "Data Engineering squad"},
    {"name": "squad-execution",      "is_private": True,  "topic": "Execution & Microstructure squad"},
    {"name": "squad-risk",           "is_private": True,  "topic": "Risk Engineering squad"},
    {"name": "squad-security",       "is_private": True,  "topic": "Security squad"},
    {"name": "squad-devops",         "is_private": True,  "topic": "DevOps / SRE squad"},
    {"name": "squad-qa",             "is_private": True,  "topic": "QA / Test Automation squad"},
    {"name": "squad-compliance",     "is_private": True,  "topic": "Compliance Engineering"},
    {"name": "squad-finance-eng",    "is_private": True,  "topic": "Finance Engineering"},
    # Private — leadership
    {"name": "leadership",           "is_private": True,  "topic": "VP+ only"},
    {"name": "leadership-summary",   "is_private": True,  "topic": "Daily auto-summaries from each VP"},
    {"name": "board",                "is_private": True,  "topic": "CEO + CFO + CTO + board observers"},
    {"name": "pm-coordination",      "is_private": True,  "topic": "All PMs cross-coordinate"},
]


def slack_call(token: str, method: str, payload: dict) -> dict:
    url = f"https://slack.com/api/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
            # Capture the X-OAuth-Scopes header so we can show actual scopes
            scopes = resp.headers.get("X-OAuth-Scopes", "")
            if scopes:
                body["_scopes"] = scopes
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from slack.{method}: {e.read().decode()}")
    if not body.get("ok"):
        raise RuntimeError(f"slack.{method} error={body.get('error')} needed={body.get('needed')} body={body}")
    return body


def list_all_channels(token: str) -> dict[str, str]:
    """Return {name: id} for every channel the bot can see."""
    names: dict[str, str] = {}
    cursor = ""
    while True:
        payload: dict = {"types": "public_channel,private_channel", "limit": 200}
        if cursor:
            payload["cursor"] = cursor
        data = slack_call(token, "conversations.list", payload)
        for ch in data.get("channels", []):
            names[ch["name"]] = ch["id"]
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break
    return names


def post_github_comment(repo: str, token: str, issue_num: int, body: str) -> None:
    url = f"https://api.github.com/repos/{repo}/issues/{issue_num}/comments"
    data = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        print("✓ Posted diagnostic comment to GitHub issue #2")
    except Exception as e:
        print(f"⚠ Could not post GitHub comment: {e}")


def main() -> int:
    log = StringIO()

    def out(msg: str) -> None:
        print(msg)
        log.write(msg + "\n")

    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        out("❌ SLACK_BOT_TOKEN is not set")
        return 1
    if not token.startswith("xoxb-"):
        out(f"❌ Token does not start with 'xoxb-' — got prefix: {token[:6]!r}")
        return 1

    # Auth test
    try:
        info = slack_call(token, "auth.test", {})
        out(f"\n✅ Authenticated as bot '{info.get('user')}' in team '{info.get('team')}'")
        out(f"   Bot user ID: {info.get('user_id')}")
        out(f"   App ID:      {info.get('bot_id') or info.get('app_id', 'n/a')}")
        out(f"   Token type:  {token[:5]}…{token[-4:]}  (length={len(token)})")
        actual_scopes = info.get("_scopes", "(no X-OAuth-Scopes header)")
        out(f"\n📋 Scopes Slack reports this token has:")
        out(f"   {actual_scopes}\n")
        # Hard fail if scopes are wrong — this gives clearer error than letting
        # conversations.list fail later
        required = {"channels:read", "channels:manage", "groups:read", "groups:write"}
        have = set(s.strip() for s in actual_scopes.split(","))
        missing = required - have
        if missing:
            out(f"❌ Token is missing required scopes: {sorted(missing)}")
            out(f"   Token currently has: {sorted(have)}")
            out("")
            out("   → This means EITHER:")
            out("     (a) You added scopes in the Slack app dashboard but didn't")
            out("         click 'Reinstall to QuantEdge' (yellow banner at top)")
            out("     (b) You did reinstall but pasted the OLD token. After reinstall,")
            out("         go back to OAuth & Permissions and copy the *new* xoxb- token")
            out("         that appears (it's a different string than before).")
            out("     (c) You have two Slack apps in the workspace named similarly;")
            out("         the one you added scopes to is not the one this token is from.")
            _maybe_post(log.getvalue(), exit_code=1)
            return 1
    except RuntimeError as e:
        out(f"❌ auth.test failed: {e}")
        _maybe_post(log.getvalue(), exit_code=1)
        return 1

    # Fetch existing channels once
    try:
        existing = list_all_channels(token)
        out(f"ℹ  Found {len(existing)} existing channels in workspace\n")
    except RuntimeError as e:
        out(f"❌ conversations.list failed: {e}")
        _maybe_post(log.getvalue(), exit_code=1)
        return 1

    created, skipped, errors = [], [], []

    out("🚀 Creating channels...\n")
    for spec in CHANNELS:
        name = spec["name"]
        if name in existing:
            skipped.append(name)
            continue
        try:
            data = slack_call(token, "conversations.create", {
                "name": name,
                "is_private": spec["is_private"],
            })
            ch = data.get("channel", {})
            ch_id = ch.get("id")
            # Set topic
            if spec.get("topic") and ch_id:
                try:
                    slack_call(token, "conversations.setTopic", {
                        "channel": ch_id,
                        "topic": spec["topic"],
                    })
                except RuntimeError as te:
                    out(f"   ⚠ setTopic failed for #{name}: {te}")
            created.append(name)
            out(f"   + #{name}")
        except RuntimeError as e:
            err_str = str(e)
            if "name_taken" in err_str:
                skipped.append(name)
                out(f"   = #{name} (already existed)")
            else:
                errors.append({"channel": name, "error": err_str})
                out(f"   ✗ #{name}: {err_str}")

    out(f"\n{'='*60}")
    out(f"✅ Created:          {len(created)}")
    out(f"⏭  Already existed:  {len(skipped)}")
    out(f"❌ Errors:           {len(errors)}")

    exit_code = 1 if errors else 0
    _maybe_post(log.getvalue(), exit_code=exit_code)
    return exit_code


def _maybe_post(body: str, exit_code: int) -> None:
    gh_token = os.environ.get("GH_TOKEN", "").strip()
    gh_repo = os.environ.get("GH_REPO", "").strip()
    if not gh_token or not gh_repo:
        return
    status = "✅ SUCCESS" if exit_code == 0 else "❌ FAILED"
    comment = f"## Slack Bootstrap — {status}\n\n```\n{body}\n```"
    post_github_comment(gh_repo, gh_token, issue_num=2, body=comment)


if __name__ == "__main__":
    sys.exit(main())
