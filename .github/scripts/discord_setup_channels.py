"""Create the QuantEdge channel structure in Discord — fully automated.

Webhooks can't create channels, so a single webhook dumps everything into
#general with a [#channel] prefix. This uses the BOT token (already a repo
secret) to create the desk/ops channels the platform posts to, grouped under
category headers. Idempotent: skips channels that already exist. The bot needs
the "Manage Channels" permission in the server.

Run by discord-commands-sync.yml. No-ops cleanly without DISCORD_BOT_TOKEN.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
API = "https://discord.com/api/v10"

# category -> channels (matches the Slack channel map the code posts to)
STRUCTURE: dict[str, list[str]] = {
    "TRADING DESKS": [
        "desk-equities", "desk-crypto", "desk-options",
        "desk-polymarket", "desk-fx-rates", "desk-stat-arb",
    ],
    "OPS & ALERTS": [
        "infra-alerts", "risk-alerts", "pnl-daily", "ci-failures",
    ],
    "COMPANY": [
        "engineering", "alpha-research", "leadership-summary", "okrs",
    ],
}


def _req(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={
            "Authorization": f"Bot {TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "QuantEdge-Setup (https://quantedge, 1.0)",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read() or "{}")


def main() -> int:
    if not TOKEN:
        print("DISCORD_BOT_TOKEN not set — skipping channel setup")
        return 0
    try:
        guilds = _req("GET", "/users/@me/guilds")
    except Exception as e:  # noqa: BLE001
        print(f"Could not list guilds (bot not in any server / bad token): {e}")
        return 0
    if not guilds:
        print("Bot is in no servers — invite it first, then re-run.")
        return 0

    for g in guilds:
        gid = g["id"]
        print(f"\nServer: {g.get('name')} ({gid})")
        try:
            existing = _req("GET", f"/guilds/{gid}/channels")
        except Exception as e:  # noqa: BLE001
            print(f"  cannot read channels (missing Manage Channels?): {e}")
            continue
        have = {c["name"].lower(): c for c in existing}

        for category, channels in STRUCTURE.items():
            cat_id = None
            cat_key = category.lower()
            match = next((c for c in existing if c["name"].lower() == cat_key and c["type"] == 4), None)
            if match:
                cat_id = match["id"]
            else:
                try:
                    cat = _req("POST", f"/guilds/{gid}/channels", {"name": category, "type": 4})
                    cat_id = cat["id"]
                    print(f"  + category {category}")
                    time.sleep(0.4)
                except Exception as e:  # noqa: BLE001
                    print(f"  category {category} failed: {e}")

            for ch in channels:
                if ch in have:
                    print(f"    = #{ch} (exists)")
                    continue
                try:
                    _req("POST", f"/guilds/{gid}/channels",
                         {"name": ch, "type": 0, "parent_id": cat_id})
                    print(f"    + #{ch}")
                    time.sleep(0.4)  # stay under the create rate limit
                except Exception as e:  # noqa: BLE001
                    print(f"    #{ch} failed: {e}")
    print("\nChannel setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
