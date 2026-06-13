"""
Slack bot integration — full Web API client, not just webhooks.

Authenticates with a Bot Token (xoxb-...). Required OAuth scopes:
  channels:manage   create + archive public channels
  groups:write      create + archive private channels
  chat:write        post messages (any channel the bot is in)
  channels:read     list channels (for lookups)
  groups:read       list private channels
  channels:history  read public channel messages (for audit_channels)
  groups:history    read private channel messages (for audit_channels)
  users:read        list workspace users (for invites)
  channels:join     auto-join created public channels
  im:write          DM users

Env vars:
  SLACK_BOT_TOKEN   xoxb-...
  SLACK_TEAM_ID     (optional) limit operations to one team if multi-team

The full 27-channel bootstrap function `bootstrap_engineering_org()` creates
the entire team's channel structure in one call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from app.utils.logging import logger


SLACK_API = "https://slack.com/api"


# Full channel spec — created by bootstrap_engineering_org()
ENGINEERING_CHANNELS: list[dict] = [
    # Tier 1 — engineering operations (public)
    {"name": "engineering-standup",  "is_private": False, "topic": "Daily standups from each squad (13:00 UTC)"},
    {"name": "alpha-research",       "is_private": False, "topic": "New strategy proposals + paper reviews"},
    {"name": "pnl-daily",            "is_private": False, "topic": "EOD P&L attribution by strategy"},
    {"name": "risk-alerts",          "is_private": False, "topic": "VaR breaches, circuit breakers"},
    {"name": "incidents",            "is_private": False, "topic": "P0/P1 incidents and postmortems"},
    {"name": "deploys",              "is_private": False, "topic": "Deploy notifications"},
    {"name": "ci-failures",          "is_private": False, "topic": "CI test failures (auto-routed)"},
    {"name": "ml-experiments",       "is_private": False, "topic": "Training run results, model leaderboard"},

    # Tier 1.5 — public general
    {"name": "engineering",          "is_private": False, "topic": "All engineers"},
    {"name": "announcements",        "is_private": False, "topic": "Company-wide announcements (CEO only posts)"},
    {"name": "wins",                 "is_private": False, "topic": "Celebrate shipped features and winning strategies"},
    {"name": "help",                 "is_private": False, "topic": "Anyone can ask, anyone answers"},

    # Tier 2 — squad channels (private)
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

    # Tier 3 — leadership (private)
    {"name": "leadership",           "is_private": True,  "topic": "VP+ only"},
    {"name": "leadership-summary",   "is_private": True,  "topic": "Daily auto-summaries from each VP"},
    {"name": "board",                "is_private": True,  "topic": "CEO + CFO + CTO + board observers"},
    {"name": "pm-coordination",      "is_private": True,  "topic": "All PMs cross-coordinate"},
]


@dataclass
class SlackBot:
    token: str
    _client: httpx.Client | None = None

    def __post_init__(self):
        if not self.token.startswith("xoxb-"):
            raise ValueError("SLACK_BOT_TOKEN must start with 'xoxb-' (this is a bot token)")
        self._client = httpx.Client(
            base_url=SLACK_API,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            timeout=15,
        )

    def _call(self, method: str, payload: dict | None = None) -> dict:
        r = self._client.post(f"/{method}", json=payload or {})
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"slack.{method} failed: {data.get('error', 'unknown')} — {data}")
        return data

    # ── Auth / introspection ────────────────────────────────────────────────

    def auth_test(self) -> dict:
        return self._call("auth.test")

    def list_channels(self, include_private: bool = True) -> list[dict]:
        types = "public_channel,private_channel" if include_private else "public_channel"
        out, cursor = [], ""
        while True:
            data = self._call("conversations.list", {"types": types, "limit": 200, "cursor": cursor})
            out.extend(data.get("channels", []))
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
        return out

    def find_channel_id(self, name: str, channels: list[dict] | None = None) -> str | None:
        name = name.lstrip("#")
        chans = channels or self.list_channels()
        for c in chans:
            if c.get("name") == name:
                return c.get("id")
        return None

    # ── Channel management ─────────────────────────────────────────────────

    def create_channel(self, name: str, is_private: bool = False, topic: str = "") -> dict:
        """Create a channel. Returns existing channel if name already taken."""
        existing = self.find_channel_id(name)
        if existing:
            logger.info("slack: channel exists, skipping create", name=name, id=existing)
            return {"id": existing, "name": name, "already_existed": True}

        try:
            data = self._call("conversations.create", {"name": name, "is_private": is_private})
        except RuntimeError as e:
            if "name_taken" in str(e):
                cid = self.find_channel_id(name)
                return {"id": cid, "name": name, "already_existed": True}
            raise

        ch = data.get("channel", {})
        if topic and ch.get("id"):
            try:
                self._call("conversations.setTopic", {"channel": ch["id"], "topic": topic})
            except Exception as e:
                logger.warning("slack: setTopic failed", channel=name, error=str(e))
        return {"id": ch.get("id"), "name": ch.get("name"), "already_existed": False}

    def post(self, channel: str, text: str, *, blocks: list[dict] | None = None) -> dict:
        """Post by channel name or ID."""
        cid = channel if channel.startswith(("C", "G")) else self.find_channel_id(channel)
        if not cid:
            raise ValueError(f"channel '{channel}' not found")
        payload: dict[str, Any] = {"channel": cid, "text": text}
        if blocks:
            payload["blocks"] = blocks
        return self._call("chat.postMessage", payload)

    def invite_users(self, channel: str, user_ids: list[str]) -> dict:
        cid = channel if channel.startswith(("C", "G")) else self.find_channel_id(channel)
        if not cid:
            raise ValueError(f"channel '{channel}' not found")
        return self._call("conversations.invite", {"channel": cid, "users": ",".join(user_ids)})

    # ── Reading / audit ─────────────────────────────────────────────────────

    def channel_history(self, channel: str, limit: int = 50) -> list[dict]:
        """
        Return recent messages for a channel (newest first). Empty list if the
        bot is not a member (Slack returns not_in_channel) or the channel is
        unknown — callers treat that as 'unreadable', not a crash.
        """
        cid = channel if channel.startswith(("C", "G")) else self.find_channel_id(channel)
        if not cid:
            return []
        try:
            data = self._call("conversations.history", {"channel": cid, "limit": limit})
        except RuntimeError as e:
            # not_in_channel / channel_not_found are expected, surfaced by audit.
            logger.debug("slack: history unavailable", channel=channel, error=str(e))
            return []
        return data.get("messages", [])

    def audit_channels(
        self,
        names: list[str] | None = None,
        stale_after_hours: float = 24.0,
        history_limit: int = 50,
    ) -> dict:
        """
        Walk each channel and report what's not working. For every channel:
          - is_member:     can the bot read it at all?
          - last_activity: hours since the most recent human/bot message
          - status:        'healthy' | 'silent' | 'stale' | 'unreadable' | 'flagged'
          - flagged:       count of messages mentioning failure keywords

        'silent' = no messages ever; 'stale' = nothing in stale_after_hours;
        'flagged' = recent error/blocker chatter that may need attention.
        """
        import time as _time

        FLAG_WORDS = ("error", "failed", "failing", "blocker", "blocked",
                      "broken", "stuck", "exception", "traceback", "down",
                      "timeout", "cannot", "can't", "crash")

        channels = self.list_channels(include_private=True)
        by_name = {c.get("name"): c for c in channels}
        target_names = names or [c["name"] for c in ENGINEERING_CHANNELS]
        now = _time.time()

        results: list[dict] = []
        for name in target_names:
            ch = by_name.get(name)
            if ch is None:
                results.append({"channel": name, "status": "missing",
                                "detail": "channel does not exist"})
                continue

            is_member = bool(ch.get("is_member", False))
            msgs = self.channel_history(ch["id"], limit=history_limit) if is_member else []

            if not is_member:
                results.append({"channel": name, "status": "unreadable",
                                "is_member": False,
                                "detail": "bot is not in this channel — invite it to monitor"})
                continue

            if not msgs:
                results.append({"channel": name, "status": "silent", "is_member": True,
                                "message_count": 0, "detail": "no messages ever"})
                continue

            newest_ts = max((float(m.get("ts", 0)) for m in msgs), default=0.0)
            age_hours = round((now - newest_ts) / 3600.0, 1) if newest_ts else None
            flagged = sum(
                1 for m in msgs
                if any(w in (m.get("text", "") or "").lower() for w in FLAG_WORDS)
            )

            if flagged:
                status = "flagged"
            elif age_hours is not None and age_hours > stale_after_hours:
                status = "stale"
            else:
                status = "healthy"

            results.append({
                "channel": name, "status": status, "is_member": True,
                "message_count": len(msgs), "last_activity_hours": age_hours,
                "flagged_messages": flagged,
            })

        summary = {
            "total": len(results),
            "healthy": sum(1 for r in results if r["status"] == "healthy"),
            "flagged": sum(1 for r in results if r["status"] == "flagged"),
            "stale": sum(1 for r in results if r["status"] == "stale"),
            "silent": sum(1 for r in results if r["status"] == "silent"),
            "unreadable": sum(1 for r in results if r["status"] == "unreadable"),
            "missing": sum(1 for r in results if r["status"] == "missing"),
        }
        return {"summary": summary, "channels": results}

    # ── Bootstrap ──────────────────────────────────────────────────────────

    def bootstrap_engineering_org(self) -> dict:
        """Create all 31 channels in ENGINEERING_CHANNELS. Idempotent."""
        # Cache existing channels once to avoid N lookups
        existing = self.list_channels(include_private=True)
        existing_names = {c.get("name") for c in existing}

        created, skipped, errors = [], [], []
        for spec in ENGINEERING_CHANNELS:
            try:
                if spec["name"] in existing_names:
                    skipped.append(spec["name"])
                    continue
                result = self.create_channel(spec["name"], spec["is_private"], spec.get("topic", ""))
                created.append(result["name"])
            except Exception as e:
                errors.append({"channel": spec["name"], "error": str(e)})
                logger.warning("slack bootstrap: channel create failed", name=spec["name"], error=str(e))

        return {
            "created": created,
            "skipped_existing": skipped,
            "errors": errors,
            "total_attempted": len(ENGINEERING_CHANNELS),
        }


def from_env() -> SlackBot | None:
    token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    if not token:
        return None
    return SlackBot(token=token)
