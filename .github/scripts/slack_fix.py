"""
Slack Emergency Fix Script

1. Tests the bot token
2. Lists all channels the bot can see
3. Joins/invites bot to every public channel
4. Posts a test message to each channel
5. Reports what worked

Run: SLACK_BOT_TOKEN=xoxb-... python .github/scripts/slack_fix.py
"""
from __future__ import annotations

import os
import sys
import httpx

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
API = "https://slack.com/api"


def call(method: str, payload: dict = {}) -> dict:
    r = httpx.post(
        f"{API}/{method}",
        headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
        json=payload,
        timeout=15,
    )
    return r.json()


def get(method: str, params: dict = {}) -> dict:
    r = httpx.get(
        f"{API}/{method}",
        headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
        params=params,
        timeout=15,
    )
    return r.json()


def main():
    if not SLACK_TOKEN:
        print("ERROR: SLACK_BOT_TOKEN not set")
        sys.exit(1)

    # 1. Test auth
    auth = call("auth.test")
    if not auth.get("ok"):
        print(f"AUTH FAILED: {auth.get('error')}")
        print("Check your SLACK_BOT_TOKEN in GitHub Secrets → it may have expired or been revoked")
        sys.exit(1)

    bot_id    = auth.get("user_id")
    team_name = auth.get("team")
    bot_name  = auth.get("user")
    print(f"✅ Token valid — Bot: @{bot_name} ({bot_id}) in workspace: {team_name}")

    # 2. List public channels
    channels_resp = get("conversations.list", {"types": "public_channel", "limit": 200})
    if not channels_resp.get("ok"):
        print(f"Cannot list channels: {channels_resp.get('error')}")
        print("Bot needs 'channels:read' scope — re-install the Slack App with updated permissions")
        sys.exit(1)

    channels = channels_resp.get("channels", [])
    print(f"\nFound {len(channels)} public channels:")

    # Key channels we must be in
    KEY_CHANNELS = [
        "general", "engineering", "ml-experiments", "risk-alerts",
        "desk-equities", "desk-crypto", "desk-options", "desk-polymarket",
        "infra-alerts", "deploys", "wins", "help",
    ]

    joined = []
    failed = []

    for ch in channels:
        name     = ch.get("name", "")
        ch_id    = ch.get("id", "")
        is_member = ch.get("is_member", False)
        icon     = "✓" if is_member else "○"
        print(f"  {icon} #{name} ({ch_id})")

        if name in KEY_CHANNELS and not is_member:
            # Join the channel
            join = call("conversations.join", {"channel": ch_id})
            if join.get("ok"):
                print(f"    → Joined #{name}")
                joined.append(name)
            else:
                err = join.get("error", "unknown")
                print(f"    → Cannot join #{name}: {err}")
                failed.append(f"#{name}: {err}")

    # 3. Post test message to key channels
    print("\nPosting test messages...")
    MESSAGE_SENT = False
    for ch in channels:
        name = ch.get("name", "")
        if name not in KEY_CHANNELS:
            continue
        ch_id = ch.get("id", "")
        msg = call("chat.postMessage", {
            "channel": ch_id,
            "text": (
                f"🤖 *QuantEdge-AI is live!*\n"
                f"Bot @{bot_name} is connected and monitoring all systems.\n"
                f"Post errors here or in <#risk-alerts> for auto-diagnosis.\n"
                f"_To trigger insights: run the 'Claude AI Slack Bot' workflow from GitHub Actions._"
            ),
            "mrkdwn": True,
        })
        if msg.get("ok"):
            print(f"  ✅ Posted to #{name}")
            MESSAGE_SENT = True
        else:
            err = msg.get("error", "unknown")
            print(f"  ❌ #{name}: {err}")
            if err == "not_in_channel":
                print(f"     → Bot needs 'chat:write.public' scope, or must be invited to #{name}")
            failed.append(f"#{name}: {err}")

    # 4. Summary
    print("\n" + "="*50)
    if MESSAGE_SENT:
        print("✅ Slack bot is working! Messages posted to key channels.")
    else:
        print("❌ No messages posted. Check bot scopes:")
        print("   Required: chat:write  chat:write.public  channels:read  channels:join")
        print("   Go to api.slack.com → Your App → OAuth & Permissions → Scopes")
        print("   Add scopes, then reinstall the app to workspace")

    if failed:
        print(f"\nFailed channels: {', '.join(failed)}")

    # 5. Check scopes
    scope_check = auth.get("response_metadata", {})
    print(f"\nBot scopes: {auth.get('headers', {})}")


if __name__ == "__main__":
    main()
