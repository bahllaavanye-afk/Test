"""Brain-health canary — probe the free-LLM cascade and SCREAM if it's down.

The cascade silently died once (Cloudflare 1010) and nobody noticed for days
because the agent workflows degrade to green. This canary probes every provider,
prints a status report, posts a Slack alert to #infra-alerts when the brain is
unhealthy, and exits non-zero so the workflow goes red.

Run:  python .github/scripts/brain_health.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common as L  # noqa: E402

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def _alert_slack(text: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return
    channel = os.environ.get("ALERT_CHANNEL", "infra-alerts")
    body = json.dumps({"channel": channel, "text": text}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "User-Agent": _UA},
    )
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] slack alert failed: {e}", file=sys.stderr)


def main() -> int:
    st = L.cascade_status(probe=True)
    print(json.dumps(st, indent=2))

    if st["healthy"]:
        print(f"BRAIN OK — working providers: {st['working']}")
        return 0

    keyed = [n for n, v in st["providers"].items() if v.get("has_key")]
    msg = (
        ":brain::red_circle: *QuantEdge LLM cascade is DOWN* — every free provider is "
        f"failing. Agents are running blind.\nKeyed providers tried: {keyed or 'none'}.\n"
        "Likely causes: expired/quota keys, or a request-layer block (e.g. Cloudflare 1010 "
        "needs a browser User-Agent)."
    )
    print("BRAIN UNHEALTHY — alerting", file=sys.stderr)
    _alert_slack(msg)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
