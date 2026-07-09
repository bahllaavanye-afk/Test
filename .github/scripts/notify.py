"""Unified notification delivery for all CI agents: Slack first, Discord always.

47 of 80 scripts in this directory grew their own chat.postMessage helper with
no failover — when Slack's free-tier quota died, their output went nowhere.
This is the single shared chokepoint: `post()` delivers via Slack when a
working token exists, and via Discord otherwise (or when Slack rejects).
Discord side supports per-channel webhooks (DISCORD_WEBHOOK_URL_<SLUG>) and
per-employee bot profiles via the `username` field.

Stdlib-only, safe to import from any sibling script:

    import notify
    notify.post("#infra-alerts", "something broke", username="Health Monitor")
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

_SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "").strip()
_DEFAULT_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
_GUILD_ID = os.environ.get("DISCORD_GUILD_ID", "").strip()
_DISCORD_CAP = int(os.environ.get("DISCORD_MAX_POSTS_PER_RUN", "20"))
_DISCORD_API = "https://discord.com/api/v10"
_UA = "Mozilla/5.0 (X11; Linux x86_64) QuantEdge-Notify/1.0"
# Discord's REST API REQUIRES a User-Agent of the form "DiscordBot (url, version)".
# A browser UA gets 403 Forbidden on bot endpoints — that is exactly why the
# channel-map load 403'd (while webhooks, which are lenient, accepted the browser
# UA). Bot API calls use this UA; webhook posts keep the browser UA above.
_BOT_UA = "DiscordBot (https://github.com/quantedge/quantedge, 1.0)"

# Bot-token channel routing: resolve #channel-name → channel id ONCE, so every
# message lands in its real channel instead of dumping into #general via a single
# webhook (the reason "all messages come to #general"). Cached per process.
_channel_ids: dict[str, str] | None = None


def _bot_req(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        _DISCORD_API + path, data=data, method=method,
        headers={"Authorization": f"Bot {_BOT_TOKEN}", "Content-Type": "application/json",
                 "User-Agent": _BOT_UA},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or "{}")


def _guild_ids() -> list[str]:
    """Guild ids to scan. A pinned DISCORD_GUILD_ID skips /users/@me/guilds
    (which some bot installs can't call); otherwise enumerate the bot's guilds."""
    if _GUILD_ID:
        return [_GUILD_ID]
    try:
        return [g["id"] for g in _bot_req("GET", "/users/@me/guilds")]
    except Exception as e:  # noqa: BLE001
        print(f"[notify] guild list failed ({str(e)[:70]}); "
              f"set DISCORD_GUILD_ID to skip /users/@me/guilds")
        return []


def _load_channel_ids() -> dict[str, str]:
    """{normalized-channel-name: id} across every guild the bot is in."""
    global _channel_ids
    if _channel_ids is not None:
        return _channel_ids
    _channel_ids = {}
    if not _BOT_TOKEN:
        return _channel_ids
    for gid in _guild_ids():
        try:
            for c in _bot_req("GET", f"/guilds/{gid}/channels"):
                if c.get("type") == 0:  # text channel
                    _channel_ids.setdefault(c["name"].lower().lstrip("#"), c["id"])
        except Exception as e:  # noqa: BLE001
            print(f"[notify] channels for guild {gid} failed: {str(e)[:80]}")
    return _channel_ids


def _post_to_channel_id(cid: str, text: str, username: str) -> bool:
    """Post via bot API to a specific channel id. Bots can't set per-message
    usernames, so the employee identity rides a bold name prefix."""
    prefix = f"**{str(username).strip()[:60]}** " if username and username != "QuantEdge" else ""
    try:
        _bot_req("POST", f"/channels/{cid}/messages", {"content": (prefix + text)[:2000]})
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[notify] bot post failed: {str(e)[:80]}")
        return False

stats = {"slack_ok": 0, "discord_ok": 0, "failed": 0, "discord_skipped": 0}
_slack_dead: str = ""  # first fatal Slack error; short-circuits later attempts

_FATAL_SLACK_ERRORS = {
    "message_limit_exceeded", "invalid_auth", "account_inactive",
    "token_revoked", "not_authed",
}


def _discord_webhook_for(channel: str) -> str:
    slug = str(channel).lstrip("#").upper().replace("-", "_")
    return os.environ.get(f"DISCORD_WEBHOOK_URL_{slug}", "") or _DEFAULT_WEBHOOK


def discord_post(channel: str, text: str, username: str = "QuantEdge") -> bool:
    """Deliver one message to Discord, into its REAL channel. Never raises.

    Order of preference:
      1. Bot token → resolve #channel → post to that channel id (needs the bot
         in the server with Send Messages; this is what stops the #general dump).
      2. Per-channel webhook (DISCORD_WEBHOOK_URL_<SLUG>) if one is configured.
      3. Default webhook → #general with a [#channel] prefix (last resort).
    """
    if not text:
        return False
    if stats["discord_ok"] >= _DISCORD_CAP:
        stats["discord_skipped"] += 1
        return False

    name = str(channel).lower().lstrip("#")
    # 1) bot-token routing to the actual channel
    if _BOT_TOKEN:
        cid = _load_channel_ids().get(name)
        if cid and _post_to_channel_id(cid, text, username):
            stats["discord_ok"] += 1
            time.sleep(0.4)
            return True

    # 2) / 3) webhook fallback (per-channel override, else default → #general)
    webhook = _discord_webhook_for(channel)
    if not webhook:
        return False
    # Only prefix with [#channel] when falling back to the shared default webhook.
    using_default = webhook == _DEFAULT_WEBHOOK
    content = (f"**[#{name}]** {text}" if using_default else text)[:2000]
    body = json.dumps({
        "content": content,
        "username": (str(username).strip()[:80] or "QuantEdge"),
    }).encode()
    req = urllib.request.Request(
        webhook, data=body,
        headers={"Content-Type": "application/json", "User-Agent": _UA},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
        stats["discord_ok"] += 1
        time.sleep(0.5)
        return True
    except Exception as e:  # noqa: BLE001
        stats["failed"] += 1
        print(f"[notify] discord failed: {str(e)[:80]}")
        return False


def _discord_post_LEGACY(channel: str, text: str, username: str = "QuantEdge") -> bool:
    """Superseded by the bot-token-routing discord_post above; kept for reference."""
    webhook = _discord_webhook_for(channel)
    if not webhook or not text:
        return False
    if stats["discord_ok"] >= _DISCORD_CAP:
        stats["discord_skipped"] += 1
        return False
    body = json.dumps({
        "content": f"**[#{str(channel).lstrip('#')}]** {text}"[:2000],
        "username": (str(username).strip()[:80] or "QuantEdge"),
    }).encode()
    req = urllib.request.Request(
        webhook, data=body,
        headers={
            "Content-Type": "application/json",
            # Discord/Cloudflare 403s the default Python-urllib UA (same
            # gotcha as the LLM cascade's error 1010) — send a browser UA.
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) QuantEdge-Notify/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
        stats["discord_ok"] += 1
        time.sleep(0.5)  # webhook rate bucket (~5 req / 2 s)
        return True
    except Exception as e:  # noqa: BLE001
        stats["failed"] += 1
        print(f"[notify] discord failed: {str(e)[:80]}")
        return False


def slack_post(channel: str, text: str) -> bool:
    """One Slack chat.postMessage attempt. Never raises."""
    global _slack_dead
    if _slack_dead or not _SLACK_TOKEN.startswith("xoxb-"):
        return False
    body = json.dumps({"channel": channel, "text": text, "mrkdwn": True}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=body,
        headers={"Authorization": f"Bearer {_SLACK_TOKEN}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read())
        if resp.get("ok"):
            stats["slack_ok"] += 1
            return True
        err = resp.get("error", "")
        if err in _FATAL_SLACK_ERRORS:
            _slack_dead = err
            print(f"[notify] Slack fatally dead ({err}) — Discord takes over this run")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[notify] slack failed: {str(e)[:80]}")
        return False


def post(channel: str, text: str, username: str = "QuantEdge") -> bool:
    """Deliver anywhere: Slack when healthy, Discord otherwise.

    Returns True if the message landed on at least one platform.
    """
    if not text:
        return False
    if slack_post(channel, text):
        return True
    return discord_post(channel, text, username=username)
