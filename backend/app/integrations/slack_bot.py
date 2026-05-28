"""
Slack bot integration — full Web API client, not just webhooks.

Authenticates with a Bot Token (xoxb-...). Required OAuth scopes:
  channels:manage   create + archive public channels
  groups:write      create + archive private channels
  chat:write        post messages (any channel the bot is in)
  channels:read     list channels (for lookups)
  groups:read       list private channels
  users:read        list workspace users (for invites)
  channels:join     auto-join created public channels

Env vars:
  SLACK_BOT_TOKEN   xoxb-...
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


SLACK_API = "https://slack.com/api"


ENGINEERING_CHANNELS: list[dict] = [
    {"name": "engineering-standup",  "is_private": False, "topic": "Daily standups from each squad (13:00 UTC)"},
    {"name": "alpha-research",       "is_private": False, "topic": "New strategy proposals + paper reviews"},
    {"name": "pnl-daily",            "is_private": False, "topic": "EOD P&L attribution by strategy"},
    {"name": "risk-alerts",          "is_private": False, "topic": "VaR breaches, circuit breakers"},
    {"name": "incidents",            "is_private": False, "topic": "P0/P1 incidents and postmortems"},
    {"name": "deploys",              "is_private": False, "topic": "Deploy notifications"},
    {"name": "ci-failures",          "is_private": False, "topic": "CI test failures (auto-routed)"},
    {"name": "ml-experiments",       "is_private": False, "topic": "Training run results, model leaderboard"},
    {"name": "engineering",          "is_private": False, "topic": "All engineers"},
    {"name": "announcements",        "is_private": False, "topic": "Company-wide announcements (CEO only posts)"},
    {"name": "wins",                 "is_private": False, "topic": "Celebrate shipped features and winning strategies"},
    {"name": "help",                 "is_private": False, "topic": "Anyone can ask, anyone answers"},
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
    {"name": "leadership",           "is_private": True,  "topic": "VP+ only"},
    {"name": "leadership-summary",   "is_private": True,  "topic": "Daily auto-summaries from each VP"},
    {"name": "board",                "is_private": True,  "topic": "CEO + CFO + CTO + board observers"},
    {"name": "pm-coordination",      "is_private": True,  "topic": "All PMs cross-coordinate"},
]


@dataclass
class SlackBot:
    token: str

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
            raise RuntimeError(f"slack.{method} failed: {data.get('error', 'unknown')} - {data}")
        return data

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

    def create_channel(self, name: str, is_private: bool = False, topic: str = "") -> dict:
        existing = self.find_channel_id(name)
        if existing:
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
            except Exception:
                pass
        return {"id": ch.get("id"), "name": ch.get("name"), "already_existed": False}

    def post(self, channel: str, text: str, *, blocks: list[dict] | None = None) -> dict:
        cid = channel if channel.startswith(("C", "G")) else self.find_channel_id(channel)
        if not cid:
            raise ValueError(f"channel '{channel}' not found")
        payload: dict[str, Any] = {"channel": cid, "text": text}
        if blocks:
            payload["blocks"] = blocks
        return self._call("chat.postMessage", payload)

    def bootstrap_engineering_org(self) -> dict:
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

        return {
            "created": created,
            "skipped_existing": skipped,
            "errors": errors,
            "total_attempted": len(ENGINEERING_CHANNELS),
        }
