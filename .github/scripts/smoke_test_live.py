"""Post-deploy live smoke test — the hard gate that catches broken/fake data.

Runs against the DEPLOYED backend (not the test DB). It logs in as the demo
user exactly like the website does, then asserts every key data endpoint is:
  • reachable (no 5xx, no unexpected 404 from a missing trailing slash),
  • shaped correctly, and
  • HONEST — e.g. /analytics/live-stats must NOT serve the old hardcoded
    Sharpe 2.1 / win 68 / drawdown 14.7 constants, and /analytics/tearsheet
    must never 500 (it returns a clean 404 when there are no trades).

Any failure exits non-zero (fails the workflow) and prints a summary the
notify layer forwards to Discord. This is the "features always work as soon as
they ship" guard: it would have caught both bugs found on 2026-07-06.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("SMOKE_BASE_URL", "https://quantedge-api-agb8.onrender.com").rstrip("/")
API = f"{BASE}/api/v1"
UA = "QuantEdge-SmokeTest/1.0 (+https://quantedge)"
TIMEOUT = 30

failures: list[str] = []
notes: list[str] = []


def _req(method: str, path: str, token: str | None = None, body: dict | None = None):
    url = path if path.startswith("http") else f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read().decode()
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode()[:300]
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def check(name: str, ok: bool, detail: str = "") -> None:
    (notes if ok else failures).append(f"{'✅' if ok else '❌'} {name}{f' — {detail}' if detail else ''}")


def main() -> int:
    # 1) Health
    status, body = _req("GET", f"{BASE}/health")
    check("health", status == 200 and isinstance(body, dict) and body.get("status") == "ok", f"HTTP {status}")

    # 2) Demo login (the website's guest path)
    status, body = _req("POST", "/auth/demo")
    token = body.get("access_token") if isinstance(body, dict) else None
    check("auth/demo", bool(token), f"HTTP {status}")
    if not token:
        _summary()
        return 1

    # 3) Core data endpoints must not 5xx (trailing slashes matter — a missing
    #    one 404s and silently blanks the UI).
    for path in ("/accounts/", "/positions/", "/trades/", "/bots/", "/strategies/",
                 "/orders/?limit=5", "/leaderboard/summary", "/analytics/performance",
                 "/analytics/system-status"):
        status, _ = _req("GET", path, token)
        check(f"GET {path}", status == 200, f"HTTP {status}")

    # 4) HONESTY: live-stats must not serve the retired fake constants.
    status, ls = _req("GET", "/analytics/live-stats", token)
    if status != 200 or not isinstance(ls, dict):
        check("analytics/live-stats", False, f"HTTP {status}")
    else:
        total = ls.get("total_trades", 0) or 0
        sharpe, win, dd = ls.get("sharpe_ratio"), ls.get("win_rate_pct"), ls.get("max_drawdown_pct")
        # With zero trades, perf metrics MUST be null — not fabricated numbers.
        fake = total == 0 and (sharpe == 2.1 or win == 68.0 or dd == 14.7)
        check("live-stats honest (no 2.1/68/14.7 with 0 trades)", not fake,
              f"trades={total} sharpe={sharpe} win={win} dd={dd}")

    # 5) tearsheet must never 500 (clean 404 when empty is fine).
    status, _ = _req("GET", "/analytics/tearsheet?days=90", token)
    check("analytics/tearsheet no 500", status != 500, f"HTTP {status}")

    return _summary()


def _summary() -> int:
    print("── QuantEdge live smoke test ──", flush=True)
    for line in notes + failures:
        print(" ", line, flush=True)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as fh:
            fh.write("### QuantEdge live smoke test\n\n")
            for line in notes + failures:
                fh.write(f"- {line}\n")
    if failures:
        print(f"\n{len(failures)} check(s) FAILED", flush=True)
        return 1
    print(f"\nAll {len(notes)} checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
