#!/usr/bin/env python3
"""Symbol Scout — keeps desk universes valid and growing.

Sibling of strategy_scout.py, for symbols instead of strategies:

  1. VALIDATE: every symbol every desk trades must be an active, tradable
     Alpaca asset. An unlisted symbol fails soft at the desk (no bars, skip),
     which silently shrinks coverage — this makes it loud instead.
  2. PROPOSE: active tradable Alpaca crypto pairs (vs USD) that no desk
     trades, plus liquid equity/ETF candidates from a curated watchlist.

Digest goes to Discord #desk-research; when validation finds dead symbols or
new proposals appear, an item is appended to IMPROVEMENTS.md (same growing-
queue pattern as Strategy Scout). State in .github/state/symbol_scout.json.
Honest guards: no Alpaca keys -> clean exit 0 with a message.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).parent
REPO_ROOT = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

STATE_FILE = REPO_ROOT / ".github" / "state" / "symbol_scout.json"
IMPROVEMENTS = REPO_ROOT / "IMPROVEMENTS.md"

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Liquid candidates worth proposing when absent from every desk. Curated, not
# exhaustive — the point is a steady drip of vetted ideas, not noise.
EQUITY_WATCHLIST = [
    "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE", "XBI", "SMH", "XOP",
    "KRE", "ITB", "JETS", "TAN", "ARKK", "SOXL", "GDXJ", "SIL", "REMX", "MOO",
]


def fetch_assets(asset_class: str) -> list[dict]:
    """Active assets from Alpaca. [] on failure (validation then skips)."""
    url = f"{ALPACA_BASE}/v2/assets?status=active&asset_class={asset_class}"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) QuantEdge-SymbolScout/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            assets = json.loads(r.read())
        return assets if isinstance(assets, list) else []
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ assets fetch failed for {asset_class}: {str(exc)[:100]}", flush=True)
        return []


def validate_universe(desk_symbols: set[str], tradable: set[str]) -> list[str]:
    """Desk symbols NOT in the tradable set (pure; unit-tested)."""
    return sorted(s for s in desk_symbols if s not in tradable)


def propose_crypto(tradable_pairs: set[str], wired: set[str]) -> list[str]:
    """Tradable /USD pairs no desk trades (pure; unit-tested)."""
    return sorted(p for p in tradable_pairs if p.endswith("/USD") and p not in wired)


def propose_equities(watchlist: list[str], tradable: set[str], wired: set[str]) -> list[str]:
    """Watchlist symbols that are tradable but absent from every desk (pure)."""
    return [s for s in watchlist if s in tradable and s not in wired]


def build_digest(dead: list[str], crypto_new: list[str], equity_new: list[str]) -> str:
    lines = ["🧭 **Symbol Scout** — universe validation & proposals", ""]
    if dead:
        lines.append(f"⚠️ **{len(dead)} desk symbols NOT tradable on Alpaca (silently skipped every run — prune or fix):**")
        lines.append("  " + ", ".join(f"`{s}`" for s in dead))
    else:
        lines.append("✅ Every desk symbol is an active, tradable Alpaca asset.")
    if crypto_new:
        lines.append(f"➕ Tradable crypto pairs on no desk: " + ", ".join(f"`{s}`" for s in crypto_new))
    if equity_new:
        lines.append(f"➕ Liquid watchlist ETFs on no desk: " + ", ".join(f"`{s}`" for s in equity_new))
    if not crypto_new and not equity_new:
        lines.append("No new symbol proposals this run.")
    return "\n".join(lines)


def main() -> int:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Symbol Scout: Alpaca keys absent — skipping honestly.")
        return 0

    spec = importlib.util.spec_from_file_location("dop_symscout", SCRIPTS / "desk_order_placer.py")
    dop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dop)  # type: ignore[union-attr]

    wired: set[str] = set()
    for d in dop.DESKS:
        wired |= set(d.symbols)
    stocks_wired = {s for s in wired if "/" not in s}
    crypto_wired = {s for s in wired if "/" in s}

    equities = fetch_assets("us_equity")
    crypto = fetch_assets("crypto")
    tradable_stocks = {a["symbol"] for a in equities if a.get("tradable")}
    tradable_pairs = {a["symbol"] for a in crypto if a.get("tradable")}

    dead: list[str] = []
    if tradable_stocks:
        dead += validate_universe(stocks_wired, tradable_stocks)
    if tradable_pairs:
        dead += validate_universe(crypto_wired, tradable_pairs)

    crypto_new = propose_crypto(tradable_pairs, crypto_wired) if tradable_pairs else []
    equity_new = propose_equities(EQUITY_WATCHLIST, tradable_stocks, stocks_wired) if tradable_stocks else []

    digest = build_digest(dead, crypto_new, equity_new)
    print(digest)

    prev: dict = {}
    try:
        prev = json.loads(STATE_FILE.read_text())
    except Exception:  # noqa: BLE001
        pass
    changed = (sorted(dead) != sorted(prev.get("dead", []))
               or sorted(crypto_new) != sorted(prev.get("crypto_new", [])))
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "dead": dead, "crypto_new": crypto_new, "equity_new": equity_new,
        "last_run": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    if changed and dead and IMPROVEMENTS.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = (f"- [ ] **[P1] Prune/fix {len(dead)} untradable desk symbols** "
                 f"(symbol-scout {stamp}): " + ", ".join(f"`{s}`" for s in dead) + "\n")
        src = IMPROVEMENTS.read_text()
        marker = "# QuantEdge — Improvements & Task Tracker\n"
        if marker in src and entry not in src:
            IMPROVEMENTS.write_text(src.replace(marker, marker + "\n" + entry, 1))
            print(f"[symbol-scout] appended dead-symbol item to IMPROVEMENTS.md")

    try:
        import notify
        notify.post("#desk-research", digest, username="QuantEdge Symbol Scout")
    except Exception as exc:  # noqa: BLE001
        print(f"[symbol-scout] notify skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
