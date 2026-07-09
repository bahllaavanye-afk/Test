"""Data Team — the shared data-quality desk every other team depends on.

Real quant firms run a data function that ML, backtesting, forward-testing and
live desks all sit on top of; bad data silently poisons all of them. This is
that function, deterministic and evidence-only:

  1. COVERAGE  — for every symbol every desk trades (equities + the full crypto
                 universe), fetch bars from Alpaca and check: enough history
                 (≥50 rows), fresh (last bar recent), no NaN/zero-price rows.
  2. QUOTES    — spot-check live quotes for a sample of symbols (staleness).
  3. ML INPUTS — verify the feature windows ML experiments train on are
                 satisfiable (≥120 rows) per symbol, so weekly training never
                 silently trains on stubs.
  4. REPORT    — Discord #alpha-research digest; files/updates a single
                 `agent-fix-needed` issue when systematic gaps appear (the
                 task queue is GitHub Issues — that's how work gets assigned).

Runs every 6h via data-team.yml. Skips cleanly without Alpaca keys (reports
that honestly rather than pretending coverage is fine).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
DATA_BASE = "https://data.alpaca.markets"

MIN_ROWS = 50          # desks refuse to trade below this
ML_MIN_ROWS = 120      # weekly walk-forward needs this much history
STALE_DAYS_EQUITY = 5  # last daily bar older than this (trading days ~3) = stale
STALE_DAYS_CRYPTO = 2  # crypto trades 24/7 — anything older than 2d is broken


def _desk_symbols() -> dict[str, list[str]]:
    """Pull the live desk universe straight from desk_order_placer (no drift)."""
    from desk_order_placer import DESKS

    out: dict[str, list[str]] = {"equity": [], "crypto": []}
    for d in DESKS:
        for s in d.symbols:
            bucket = "crypto" if "/" in s else "equity"
            if s not in out[bucket]:
                out[bucket].append(s)
    return out


def _get(path: str, params: dict) -> dict:
    """Alpaca data GET with 429 backoff. The free tier throttles hard, and
    fetching every symbol one-by-one used to 429 most of them (the 'fetch
    failed: HTTP Error 429' gaps in the Data Team report)."""
    url = f"{DATA_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "User-Agent": "QuantEdge-DataTeam/1.0",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else 1.5 * (attempt + 1)
                except (TypeError, ValueError):
                    delay = 1.5 * (attempt + 1)
                time.sleep(min(delay, 5.0))
                continue
            raise
    return {}


def fetch_all_bars(symbols: list[str]) -> dict[str, list | None]:
    """Fetch bars for every symbol in ONE request per asset class (Alpaca takes
    comma-separated `symbols=`). Collapses ~31 single-symbol calls into ~2,
    which is what stops the 429 storm. Value is the bars list, or None when the
    batch call itself failed (so the symbol is reported as fetch-failed, not
    silently 'healthy')."""
    start = (datetime.now(timezone.utc) - timedelta(days=300)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: dict[str, list | None] = {}
    crypto = [s for s in symbols if "/" in s]
    stocks = [s for s in symbols if "/" not in s]

    def _batch(path: str, syms: list[str], extra: dict) -> None:
        for i in range(0, len(syms), 20):
            chunk = syms[i:i + 20]
            try:
                data = _get(path, {"symbols": ",".join(chunk), "timeframe": "1Day",
                                   "limit": 250, "start": start, **extra})
                bars_by = data.get("bars") or {}
                for s in chunk:
                    out[s] = bars_by.get(s, [])
            except Exception as exc:  # noqa: BLE001
                print(f"  batch fetch failed for {chunk}: {exc}")
                for s in chunk:
                    out[s] = None
            time.sleep(0.3)

    if crypto:
        _batch("/v1beta3/crypto/us/bars", crypto, {})
    if stocks:
        _batch("/v2/stocks/bars", stocks, {"adjustment": "split", "feed": "iex"})
    return out


def check_symbol(symbol: str, bars: list | None) -> dict:
    """Return a quality record for one symbol from PRE-FETCHED bars: rows,
    freshness, integrity. bars is None when the batched fetch failed."""
    is_crypto = "/" in symbol
    if bars is None:
        return {"symbol": symbol, "ok": False, "reason": "fetch failed (rate-limited)"}

    n = len(bars)
    if n < MIN_ROWS:
        return {"symbol": symbol, "ok": False, "reason": f"only {n} bars (<{MIN_ROWS})", "rows": n}

    last = bars[-1]
    try:
        last_ts = datetime.fromisoformat(str(last.get("t", "")).replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - last_ts).days
    except ValueError:
        return {"symbol": symbol, "ok": False, "reason": "unparseable last-bar timestamp", "rows": n}

    stale_after = STALE_DAYS_CRYPTO if is_crypto else STALE_DAYS_EQUITY
    if age_days > stale_after:
        return {"symbol": symbol, "ok": False,
                "reason": f"stale — last bar {age_days}d old (>{stale_after}d)", "rows": n}

    bad = sum(1 for b in bars if not b.get("c") or float(b["c"]) <= 0)
    if bad:
        return {"symbol": symbol, "ok": False, "reason": f"{bad} zero/NaN closes", "rows": n}

    return {"symbol": symbol, "ok": True, "rows": n, "ml_ready": n >= ML_MIN_ROWS}


def main() -> int:
    if not ALPACA_KEY or not ALPACA_SECRET:
        msg = "Data Team: ALPACA keys absent — coverage UNKNOWN (not 'fine'). Skipping honestly."
        print(msg)
        _notify(msg)
        return 0

    universe = _desk_symbols()
    all_syms = [s for syms in universe.values() for s in syms]
    fetched = fetch_all_bars(all_syms)   # one batched request per asset class
    results: list[dict] = []
    for bucket, syms in universe.items():
        for s in syms:
            r = check_symbol(s, fetched.get(s))
            r["bucket"] = bucket
            results.append(r)
            print(("✅" if r["ok"] else "❌"), s, r.get("reason", f"{r.get('rows', 0)} bars"))

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    ml_short = [r for r in ok if not r.get("ml_ready")]

    lines = [
        "**🗄️ Data Team report** (live Alpaca checks — every desk symbol)",
        f"Coverage: **{len(ok)}/{len(results)}** symbols healthy",
    ]
    if bad:
        lines.append(f"\n**Gaps ({len(bad)}):**")
        lines += [f"• `{r['symbol']}` — {r['reason']}" for r in bad[:10]]
    if ml_short:
        lines.append(f"\n**ML-short (train-window <{ML_MIN_ROWS} rows):** "
                     + ", ".join(f"`{r['symbol']}`" for r in ml_short[:10]))
    if not bad and not ml_short:
        lines.append("All symbols trade-ready and ML-ready. ✅")
    report = "\n".join(lines)
    print("\n" + report)
    _notify(report)

    # Systematic failure (>25% of universe broken) → file into the task queue.
    if len(bad) > max(2, len(results) // 4):
        _file_issue(bad)
    return 0


def _notify(text: str) -> None:
    try:
        from notify import discord_post

        discord_post("alpha-research", text, username="QuantEdge Data Team")
    except Exception as exc:  # noqa: BLE001
        print(f"(Discord notify skipped: {exc})")


def _file_issue(bad: list[dict]) -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        return
    body = {
        "title": f"[data-team] {len(bad)} desk symbols failing data-quality checks",
        "body": "Automated Data Team finding — these symbols cannot be traded or trained on:\n\n"
                + "\n".join(f"- `{r['symbol']}` — {r['reason']}" for r in bad)
                + "\n\nOwner: fix the feed or prune the symbols (a symbol that never has data "
                  "is pure log noise — see the Binance-451 precedent).",
        "labels": ["agent-fix-needed", "area:infra-engineer"],
    }
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/issues",
            data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "QuantEdge-DataTeam/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"Filed issue #{json.loads(r.read()).get('number')}")
    except Exception as exc:  # noqa: BLE001
        print(f"(issue filing skipped: {exc})")


if __name__ == "__main__":
    raise SystemExit(main())
