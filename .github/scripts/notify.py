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
_DISCORD_CAP = int(os.environ.get("DISCORD_MAX_POSTS_PER_RUN", "20"))

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
    """Deliver one message to Discord. Never raises."""
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
