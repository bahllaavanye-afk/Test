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

# `or` not a get() default: an env var set to "" returns "" from get(), which
# yielded a bare "/api/v1/health" and a ValueError instead of falling back.
BASE = (os.environ.get("SMOKE_BASE_URL") or "https://quantedge-api-9jz0.onrender.com").rstrip("/")
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


def health_detailed_checks(status: int, det, strict_db: bool) -> None:
    """Assert on `/health/detailed`. Pure in its inputs so it can be tested.

    `/health` returns `{"status": "ok"}` unconditionally — no DB, by design, so
    the Render keep-alive ping stays cheap. The smoke test only ever hit that,
    which is why it stayed green for over a week while the primary database was
    paused and every trade was landing in an ephemeral SQLite file.

    The field to key on is `database_primary.ok`, NOT `database.ok`.
    `app/main.py:487` reports `database.ok = True` whenever `SELECT 1` succeeds,
    and on the SQLite fallback it does — the fallback is functional. Only
    `database_primary` (emitted at :502, and only when `db_fallback_active`)
    reports the outage. A guard written against `database.ok` would be green
    right now, during the exact condition it exists to catch.
    """
    if status != 200 or not isinstance(det, dict):
        check("health/detailed reachable", False, f"HTTP {status}")
        return

    checks = det.get("checks") or {}
    degraded = sorted(k for k, v in checks.items()
                      if isinstance(v, dict) and v.get("ok") is False)
    detail = f"status={det.get('status')} degraded={','.join(degraded) or 'none'}"

    # TRADING_MODE must never drift off paper. Cheap to assert, and this is the
    # only automated place that would see it.
    mode = det.get("mode")
    check("trading mode is paper", mode == "paper", f"mode={mode!r}")

    durable = (checks.get("database_primary") or {}).get("ok", True) is not False
    if strict_db:
        check("durable database (not the ephemeral SQLite fallback)", durable, detail)
    else:
        check("health/detailed reachable", True, detail)
        if not durable:
            # Non-failing on the 30-min schedule: a known, operator-blocked
            # condition paging #ci-failures 48×/day trains people to ignore it.
            # The deploy-time run (strict_db) is the one that fails.
            notes.append(
                "⚠️ database_primary DOWN — serving from the ephemeral SQLite "
                "fallback; state resets on every redeploy. Unpause Supabase."
            )


def main() -> int:
    # 1) Health
    status, body = _req("GET", f"{BASE}/health")
    check("health", status == 200 and isinstance(body, dict) and body.get("status") == "ok", f"HTTP {status}")

    # 1b) Subsystem health. See health_detailed_checks for why this is a
    #     separate probe and why it keys on database_primary.
    status, det = _req("GET", f"{BASE}/health/detailed")
    health_detailed_checks(status, det, strict_db=os.environ.get("SMOKE_FAIL_ON_DEGRADED_DB") == "1")

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

    # ── Staleness + liveness (the 2026-07-18 lessons) ────────────────────────
    # 1. DEPLOY PARITY: the live template count must match the repo's. For two
    #    weeks the site served a stale build (29/57 templates) while CI was
    #    green — this makes a stale deploy fail the smoke within 30 minutes.
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[2]
        _sys.path.insert(0, str(_repo / "backend"))
        import os as _os
        _os.environ.setdefault("SECRET_KEY", "s" * 64)
        _os.environ.setdefault("DATABASE_URL", "")
        from app.bots.templates import BOT_TEMPLATES as _REPO_TEMPLATES
        status, live_t = _req("GET", "/bots/templates")
        live_n = len(live_t) if isinstance(live_t, dict) else -1
        check("deploy parity: live templates == repo templates",
              status == 200 and live_n == len(_REPO_TEMPLATES),
              f"live={live_n} repo={len(_REPO_TEMPLATES)} (mismatch = STALE DEPLOY)")
    except Exception as exc:  # noqa: BLE001 — parity check must not hide other failures
        check("deploy parity: live templates == repo templates", False, f"check errored: {exc}")

    # 2. SCHEDULER LIVENESS: enabled bots must actually tick. The scheduler was
    #    dead from Jul 5-18 (2 of 29 bots ever ran) and nothing alerted.
    status, bots = _req("GET", "/bots/", token)
    if status == 200 and isinstance(bots, list) and any(b.get("is_enabled") for b in bots):
        from datetime import datetime, timedelta, timezone
        newest = max((b.get("last_run_at") or "" for b in bots), default="")
        fresh = False
        if newest:
            try:
                ts = datetime.fromisoformat(newest.replace("Z", "+00:00"))
                # The API returns last_run_at with NO timezone ("2026-07-27T09:33:21.698324"),
                # so the Z-replace is a no-op and fromisoformat yields a NAIVE datetime.
                # Subtracting that from an aware now() raises TypeError — which the
                # `except ValueError` below does not catch, so it crashed the whole
                # smoke test. This check could only ever pass while last_run_at was
                # empty: it was guaranteed to break the moment bots actually ran.
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                fresh = datetime.now(timezone.utc) - ts < timedelta(hours=24)
            except (ValueError, TypeError):
                pass
        check("bot scheduler alive (a bot ran in the last 24h)", fresh,
              f"newest last_run_at={newest or 'NEVER'}")
    else:
        check("bot scheduler alive (a bot ran in the last 24h)", False,
              f"bots endpoint HTTP {status}")

    return _summary()


DURABLE_DB_CHECK = "durable database"


def _emit_outputs() -> None:
    """Hand the failure detail to the workflow BEFORE exiting non-zero.

    The Discord page was a fixed string — "a deployed endpoint is broken or
    serving fake data" — which describes one failure mode out of nine. With the
    durable-database gate added it would actively misreport the most likely
    failure. A page that names the wrong cause is worse than the generic one:
    it sends the reader to the wrong place.

    `only_known_degraded` marks the case where the ONLY failure is the paused
    primary database. That is operator decision #2, unfixable from code, and it
    recurs on every push to main (~10/day measured over 2026-08-03/04). The run
    still goes RED — nothing is hidden, the step summary still lists it — but it
    does not re-page a channel that already knows. A second, real failure in the
    same run clears the flag and the page fires normally.
    """
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    only_known = bool(failures) and all(DURABLE_DB_CHECK in f for f in failures)
    detail = " · ".join(f.lstrip("❌ ").strip() for f in failures)[:1200]
    with open(out, "a") as fh:
        fh.write(f"only_known_degraded={'1' if only_known else '0'}\n")
        fh.write("failed_checks<<SMOKE_EOF\n" + detail + "\nSMOKE_EOF\n")


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
    _emit_outputs()
    if failures:
        print(f"\n{len(failures)} check(s) FAILED", flush=True)
        return 1
    print(f"\nAll {len(notes)} checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
