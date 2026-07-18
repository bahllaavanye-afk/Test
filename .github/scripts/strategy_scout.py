#!/usr/bin/env python3
"""Strategy Scout — keeps every desk supplied with new strategies, forever.

The 2026-07-15 audit found 60 of 113 registry strategies wired to NO desk —
strategies were being written, contract-tested, even SOTA-upgraded on a
schedule, then never traded. This scout closes that loop permanently:

  1. UNWIRED: diff STRATEGY_REGISTRY against every desk (incl. FX) and report
     strategies that trade nowhere, with a suggested desk per name.
  2. COVERAGE: per-desk strategy/symbol counts, so shrinkage is visible.
  3. IDEAS: rotate 3 candidate strategies per run from a research backlog of
     documented premia not yet in the registry — a standing prompt for the
     research team / strategy generator to build next.

Output goes to Discord #desk-research and (when something is NEW since the
last run) appends to IMPROVEMENTS.md so the queue always grows. State lives
in .github/state/strategy_scout.json for idempotence.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).parent
REPO_ROOT = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT / "backend"))

STATE_FILE = REPO_ROOT / ".github" / "state" / "strategy_scout.json"
IMPROVEMENTS = REPO_ROOT / "IMPROVEMENTS.md"

# Research backlog: documented, implementable premia NOT in the registry yet.
# The scout surfaces 3 per run (rotating) as build-next prompts. Add freely —
# this list is the "always keep finding new strategies" hopper.
RESEARCH_BACKLOG: list[tuple[str, str, str]] = [
    ("meta_labeling_gate", "Equities", "Lopez de Prado triple-barrier meta-model: P(win) multiplier on every desk signal"),
    ("cross_sectional_ls_portfolio", "Equities", "rank whole universe, long top decile / short bottom — profits in flat tape"),
    ("johansen_pairs_refresh", "StatArb", "weekly Johansen cointegration scan to auto-refresh the pairs universe"),
    ("tsmom_overlay", "Commodities", "Moskowitz-Ooi-Pedersen TSMOM overlay gating all commodity entries"),
    ("betting_against_beta", "Equities", "Frazzini-Pedersen BAB: long low-beta levered, short high-beta"),
    ("quality_minus_junk", "Equities", "AQR QMJ factor via fundamental screens (profitability, growth, safety)"),
    ("fx_carry_basket", "Macro/FX", "G10 carry basket: long top-3 yielders, short bottom-3, vol-targeted"),
    ("vol_risk_premium_fx", "Macro/FX", "sell FX vol when implied >> realized (OANDA has no options — use proxy ETFs)"),
    ("crypto_perp_basis", "Crypto", "cash-and-carry vs perp funding via a geo-permitted venue or proxy"),
    ("seasonality_calendar", "Commodities", "documented seasonal windows (natgas winter, grains harvest)"),
    ("earnings_drift_pead2", "Equities", "modern PEAD with revenue surprise + guidance direction"),
    ("etf_flow_momentum", "International", "country-ETF creation/redemption flow momentum"),
    ("term_structure_carry", "Commodities", "backwardation/contango carry via front-vs-deferred ETF pairs"),
    ("dispersion_v2", "Options", "index-vs-components correlation trade with real option legs (mleg is live)"),
    ("skew_harvest", "Options", "sell rich put skew via defined-risk verticals when skew percentile > 80"),
]

# Known exclusions: unwired ON PURPOSE, with the blocker. The scout reports
# them separately so they read as "needs unblocking", not free coverage wins.
KNOWN_EXCLUSIONS: dict[str, str] = {
    "covered_call": "needs existing share inventory the desks don't track",
    "funding_rate_arb": "Binance funding data geo-blocked (451) from US runners",
    "dex_cex_arb": "needs DEX orderbook feed (none wired)",
    "crypto_basis_roll": "needs dated-futures curve (no venue wired)",
    "token_unlock_fade": "needs a token-unlock calendar feed",
    "news_momentum": "needs a news/headline feed (none wired)",
    "earnings_accruals": "needs fundamentals (yfinance too slow/flaky in CI)",
    "micro_cap_momentum": "universe mismatch — desks trade large-caps/ETFs",
    "moc_auction_imbalance": "needs intraday auction-imbalance data",
    "order_flow_imbalance": "needs L2/order-flow data (daily bars can't feed it)",
}

_DESK_HINTS = [
    (("poly_",), "Polymarket"),
    (("crypto_", "funding_", "mvrv", "token_", "dex_", "liquidation", "onchain", "on_chain"), "Crypto"),
    (("options_", "put_call", "iv_", "gamma", "skew", "vol_", "wheel", "condor", "credit_spread",
      "covered_call", "cash_secured", "long_call", "earnings_iv", "straddle"), "Options"),
    (("yield_", "duration", "breakeven", "pmi_", "macro_", "central_bank", "bond_", "tlt_",
      "dollar_", "fx_", "interest_rate", "cross_asset"), "Macro/FX"),
    (("commodity_", "basis_carry",), "Commodities"),
    (("pairs", "stat_arb", "pca", "kalman", "cointegration", "lorentzian"), "StatArb"),
    (("_tv",), "TV Indicators"),
]


def suggest_desk(name: str) -> str:
    """Best-guess desk for a strategy name (pure; unit-tested)."""
    n = name.lower()
    for keys, desk in _DESK_HINTS:
        if any(k in n for k in keys):
            return desk
    return "Equities"


def build_digest(unwired: dict[str, str], coverage: list[tuple[str, int, int]],
                 ideas: list[tuple[str, str, str]]) -> str:
    """Discord digest (pure; unit-tested)."""
    lines = ["📡 **Strategy Scout** — desk coverage report", ""]
    fresh = {n: d for n, d in unwired.items() if n not in KNOWN_EXCLUSIONS}
    blocked = {n: KNOWN_EXCLUSIONS[n] for n in unwired if n in KNOWN_EXCLUSIONS}
    if fresh:
        lines.append(f"**{len(fresh)} registry strategies trade on NO desk (wire these):**")
        for name, desk in sorted(fresh.items()):
            lines.append(f"  · `{name}` → suggest **{desk}**")
    else:
        lines.append("✅ Every wirable registry strategy is on a desk.")
    if blocked:
        lines.append(f"⛔ {len(blocked)} excluded pending a data source:")
        for name, why in sorted(blocked.items()):
            lines.append(f"  · `{name}` — {why}")
    lines.append("")
    lines.append("**Desk coverage** (strategies × symbols):")
    for name, n_strats, n_syms in coverage:
        lines.append(f"  · {name}: {n_strats} × {n_syms}")
    if ideas:
        lines.append("")
        lines.append("**Build next** (research backlog, rotating):")
        for key, desk, desc in ideas:
            lines.append(f"  · `{key}` [{desk}] — {desc}")
    return "\n".join(lines)


def _load_desks():
    spec = importlib.util.spec_from_file_location("dop_scout", SCRIPTS / "desk_order_placer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def _load_fx_strategies() -> list[str]:
    try:
        spec = importlib.util.spec_from_file_location("fx_scout", SCRIPTS / "fx_desk.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # type: ignore[union-attr]
        return list(m.STRATEGIES)
    except Exception:  # noqa: BLE001 — FX desk optional
        return []


def main() -> int:
    os.environ.setdefault("SECRET_KEY", "s" * 64)
    os.environ.setdefault("DATABASE_URL", "")

    from app.strategies import STRATEGY_REGISTRY

    dop = _load_desks()
    wired: set[str] = set(_load_fx_strategies())
    coverage: list[tuple[str, int, int]] = []
    for d in dop.DESKS:
        wired |= set(d.strategy_names)
        coverage.append((d.name, len(d.strategy_names), len(d.symbols)))

    loaded = {n for n, c in STRATEGY_REGISTRY.items() if c is not None}
    unwired = {n: suggest_desk(n) for n in sorted(loaded - wired)}

    # rotate 3 research ideas by ISO week so every run isn't identical
    week = datetime.now(timezone.utc).isocalendar()[1]
    start = (week * 3) % max(len(RESEARCH_BACKLOG), 1)
    ideas = [RESEARCH_BACKLOG[(start + i) % len(RESEARCH_BACKLOG)] for i in range(3)]

    digest = build_digest(unwired, coverage, ideas)
    print(digest)

    # Research → registry pipeline: hand the top rotating idea to the strategy
    # generator as its priority direction (state/research_seed.json).
    seed_file = REPO_ROOT / ".github" / "state" / "research_seed.json"
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    key, desk, desc = ideas[0]
    seed_file.write_text(json.dumps({
        "key": key, "desk": desk, "description": desc,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    # state / idempotence: only touch IMPROVEMENTS.md when the unwired set CHANGED
    prev: dict = {}
    try:
        prev = json.loads(STATE_FILE.read_text())
    except Exception:  # noqa: BLE001
        pass
    changed = sorted(unwired) != sorted(prev.get("unwired", []))
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "unwired": sorted(unwired),
        "coverage": coverage,
        "last_run": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    fresh_unwired = {n: d for n, d in unwired.items() if n not in KNOWN_EXCLUSIONS}
    if changed and fresh_unwired and IMPROVEMENTS.exists():
        unwired = fresh_unwired  # only queue actionable ones
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = (f"- [ ] **[P1] Wire {len(unwired)} unwired strategies to desks** "
                 f"(strategy-scout {stamp}): " +
                 ", ".join(f"`{n}`→{d}" for n, d in sorted(unwired.items())[:12]) +
                 (" …" if len(unwired) > 12 else "") + "\n")
        src = IMPROVEMENTS.read_text()
        marker = "# QuantEdge — Improvements & Task Tracker\n"
        if marker in src and entry not in src:
            src = src.replace(marker, marker + "\n" + entry, 1)
            IMPROVEMENTS.write_text(src)
            print(f"[scout] appended {len(unwired)}-strategy item to IMPROVEMENTS.md")

    try:
        import notify
        notify.post("#desk-research", digest, username="QuantEdge Strategy Scout")
    except Exception as exc:  # noqa: BLE001
        print(f"[scout] notify skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
