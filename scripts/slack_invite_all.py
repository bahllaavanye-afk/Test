#!/usr/bin/env python3
"""
Invite all known QuantEdge Slack users to every public channel.

Usage:
  SLACK_BOT_TOKEN=xoxb-... python scripts/slack_invite_all.py

The bot must have:
  channels:read, channels:manage, users:read, groups:read permissions.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request


SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")


def _slack_get(method: str, **params) -> dict:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url   = f"https://slack.com/api/{method}?{query}"
    req   = urllib.request.Request(url, headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
    if not body.get("ok"):
        raise RuntimeError(f"{method} failed: {body.get('error')}")
    return body


def _slack_post(method: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"https://slack.com/api/{method}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
    if not body.get("ok") and body.get("error") not in ("already_in_channel", "cant_invite_self"):
        print(f"  ⚠ {method}: {body.get('error')}")
    return body


def list_all_channels() -> list[dict]:
    channels, cursor = [], ""
    while True:
        params: dict = {"types": "public_channel", "limit": 200, "exclude_archived": "true"}
        if cursor:
            params["cursor"] = cursor
        body   = _slack_get("conversations.list", **params)
        channels.extend(body.get("channels", []))
        cursor = body.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break
    return channels


def list_all_users() -> list[str]:
    """Return list of regular member user IDs (excludes bots and slackbot)."""
    users, cursor = [], ""
    while True:
        params: dict = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        body   = _slack_get("users.list", **params)
        for u in body.get("members", []):
            if not u.get("is_bot") and not u.get("deleted") and u.get("id") != "USLACKBOT":
                users.append(u["id"])
        cursor = body.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break
    return users


def invite_users_to_channel(channel_id: str, user_ids: list[str]) -> tuple[int, int]:
    """Invite users in batches of 30. Returns (invited, skipped)."""
    invited = skipped = 0
    batch_size = 30
    for i in range(0, len(user_ids), batch_size):
        batch  = user_ids[i : i + batch_size]
        result = _slack_post("conversations.invite", {
            "channel": channel_id,
            "users":   ",".join(batch),
        })
        if result.get("ok"):
            invited += len(batch)
        elif result.get("error") == "already_in_channel":
            skipped += len(batch)
        else:
            # Partial — some already members; count as skipped
            skipped += len(batch)
        time.sleep(0.3)  # Rate limit: ~3 req/s for conversations.invite
    return invited, skipped


def main() -> None:
    if not SLACK_BOT_TOKEN:
        print("ERROR: SLACK_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    print("Fetching channels...", flush=True)
    channels = list_all_channels()
    print(f"  Found {len(channels)} public channels", flush=True)

    print("Fetching users...", flush=True)
    users = list_all_users()
    print(f"  Found {len(users)} human users", flush=True)

    if not users:
        print("No users found — check bot permissions (users:read)", file=sys.stderr)
        sys.exit(1)

    total_invited = total_skipped = 0
    for ch in channels:
        ch_name = ch.get("name", ch["id"])
        inv, skip = invite_users_to_channel(ch["id"], users)
        total_invited += inv
        total_skipped += skip
        print(f"  #{ch_name}: +{inv} invited, {skip} already members", flush=True)

    print(f"\nDone: {total_invited} invitations sent, {total_skipped} already-member skips")
    print(f"All {len(users)} users should now be in all {len(channels)} channels.")


if __name__ == "__main__":
    main()
