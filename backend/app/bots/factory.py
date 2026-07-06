"""Bot variant factory — systematic search over the options-template space.

Options Alpha can't be searched programmatically (no API, plan caps at ~50
bots). Here the space IS enumerable: structure × short-delta × DTE × profit
target. The factory generates bounded variant sets as ordinary bot templates;
the existing machinery does the rest — lifecycle instantiates them, they trade
paper at small size, desk→Trades records results, the leaderboard ranks them,
and the lifecycle manager kills losers / keeps winners. Evolutionary selection
over real (paper) fills instead of backtest overfitting.

Pure functions, deterministic output (variant ids are stable), fully unit-
testable. Generated variants are capped and sized small so the fleet can't
explode: the factory is a research budget, not a firehose.
"""
from __future__ import annotations

from itertools import product

# The grid that matters (see docs/research/OPTIONS_ALPHA_BOT_ANALYSIS_2026.md §3).
STRUCTURES = ("iron_condor", "put_credit_spread", "call_credit_spread", "iron_butterfly")
SHORT_DELTAS = (0.10, 0.16, 0.20, 0.30)
DTES = (0, 7, 14, 30)
PROFIT_TARGETS = (25, 50)

# Bounded research budget: the full grid is 4×4×4×2 = 128; we cap what a single
# generation emits so paper capital stays meaningful per variant.
MAX_VARIANTS = 16
VARIANT_SIZE_PCT = 1.0   # $1k of the $100k paper account per research variant
WING_DELTA = 0.05        # long wing for defined risk


def _legs(structure: str, short_delta: float, dte: int) -> list[dict]:
    if structure == "put_credit_spread":
        return [
            {"side": "sell", "option_type": "put", "delta": short_delta, "dte": dte, "ratio": 1},
            {"side": "buy", "option_type": "put", "delta": WING_DELTA, "dte": dte, "ratio": 1},
        ]
    if structure == "call_credit_spread":
        return [
            {"side": "sell", "option_type": "call", "delta": short_delta, "dte": dte, "ratio": 1},
            {"side": "buy", "option_type": "call", "delta": WING_DELTA, "dte": dte, "ratio": 1},
        ]
    if structure == "iron_butterfly":
        return [
            {"side": "sell", "option_type": "call", "delta": 0.50, "dte": dte, "ratio": 1},
            {"side": "sell", "option_type": "put", "delta": 0.50, "dte": dte, "ratio": 1},
            {"side": "buy", "option_type": "call", "delta": short_delta, "dte": dte, "ratio": 1},
            {"side": "buy", "option_type": "put", "delta": short_delta, "dte": dte, "ratio": 1},
        ]
    # iron_condor (default)
    return [
        {"side": "sell", "option_type": "call", "delta": short_delta, "dte": dte, "ratio": 1},
        {"side": "buy", "option_type": "call", "delta": WING_DELTA, "dte": dte, "ratio": 1},
        {"side": "sell", "option_type": "put", "delta": short_delta, "dte": dte, "ratio": 1},
        {"side": "buy", "option_type": "put", "delta": WING_DELTA, "dte": dte, "ratio": 1},
    ]


def generate_variants(
    symbol: str = "SPY",
    max_variants: int = MAX_VARIANTS,
    generation: int = 0,
) -> dict[str, dict]:
    """Emit a bounded, deterministic slice of the grid as bot templates.

    ``generation`` walks the grid in stable order across calls: generation 0
    emits variants [0, max), generation 1 emits [max, 2*max), etc. — so
    successive research cycles explore fresh territory without duplication.
    """
    grid = list(product(STRUCTURES, SHORT_DELTAS, DTES, PROFIT_TARGETS))
    # Skip degenerate combos: an iron butterfly's wings must be OTM relative
    # to its 50Δ body, so a 0.30 "short delta" wing is fine but 0.50 isn't;
    # 0DTE at 0.10Δ collects too little premium to survive fees.
    grid = [g for g in grid if not (g[0] == "iron_butterfly" and g[1] >= 0.50)
            and not (g[2] == 0 and g[1] <= 0.10)]

    start = generation * max_variants
    chunk = grid[start:start + max_variants]

    out: dict[str, dict] = {}
    for structure, delta, dte, tp in chunk:
        vid = f"gen_{structure}_{int(delta * 100)}d_{dte}dte_tp{tp}"
        sells = any(l["side"] == "sell" for l in _legs(structure, delta, dte))
        exit_rules = [{"type": "take_profit", "value": tp}]
        if sells:
            exit_rules.append({"type": "stop_loss", "value": 2 * tp})  # 2:1 stop-to-target
        if dte == 0:
            exit_rules.append({"type": "time_exit", "hours": 7})  # 0DTE never holds overnight
        out[vid] = {
            "name": f"[gen] {structure.replace('_', ' ')} {int(delta * 100)}Δ {dte}DTE tp{tp}",
            "description": (
                f"Factory variant (research budget, generation {generation}): {structure} on "
                f"{symbol}, short {delta:.2f}Δ, {dte} DTE, TP {tp}% / stop {2 * tp}%. "
                "Small size; lives or dies by its live paper record via the lifecycle manager."
            ),
            "symbol": symbol,
            "market_type": "options",
            "trigger": {"type": "schedule", "interval": "1m"},
            "conditions": [
                {"type": "time_window", "start_time": "14:05", "end_time": "19:30"},
                {"type": "no_position"},
            ],
            "condition_logic": "ALL",
            "action": {
                "type": "open_option_spread",
                "size_pct": VARIANT_SIZE_PCT,
                "take_profit_pct": tp,
                "legs": _legs(structure, delta, dte),
            },
            "exit_rules": exit_rules,
        }
    return out


# ── ML-guided generation ─────────────────────────────────────────────────────
# Pure grid walking treats every combo as equally promising. Once [gen] variants
# have live records, learn which grid dimensions carry the P&L signal and bias
# the next generation toward winning neighborhoods (a contextual bandit over a
# 4-dim discrete space — the right-sized "ML" for n≈dozens of observations;
# a neural net here would be overfitting theater).

def _parse_variant_name(name: str) -> dict | None:
    """'[gen] iron condor 16Δ 7DTE tp50' → {structure, delta, dte, tp}."""
    import re

    m = re.match(r"\[gen\] (.+) (\d+)Δ (\d+)DTE tp(\d+)$", name.strip())
    if not m:
        return None
    return {
        "structure": m.group(1).replace(" ", "_"),
        "delta": int(m.group(2)) / 100.0,
        "dte": int(m.group(3)),
        "tp": int(m.group(4)),
    }


def score_dimensions(results: list[dict]) -> dict[str, dict]:
    """Per-dimension average P&L from live variant results.

    ``results``: [{"name": bot_name, "total_pnl": float, "trades": int}, ...]
    (the exact shape /leaderboard/live emits). Variants with <3 trades are
    ignored — no learning from noise. Returns {dim: {value: avg_pnl}}.
    """
    sums: dict[str, dict] = {"structure": {}, "delta": {}, "dte": {}, "tp": {}}
    counts: dict[str, dict] = {"structure": {}, "delta": {}, "dte": {}, "tp": {}}
    for r in results:
        parsed = _parse_variant_name(r.get("name") or r.get("strategy") or "")
        if not parsed or int(r.get("trades") or 0) < 3:
            continue
        pnl = float(r.get("total_pnl") or 0)
        for dim, val in parsed.items():
            sums[dim][val] = sums[dim].get(val, 0.0) + pnl
            counts[dim][val] = counts[dim].get(val, 0) + 1
    return {
        dim: {val: sums[dim][val] / counts[dim][val] for val in sums[dim]}
        for dim in sums
    }


def generate_variants_ml(
    results: list[dict],
    symbol: str = "SPY",
    max_variants: int = MAX_VARIANTS,
    explore_frac: float = 0.25,
) -> dict[str, dict]:
    """Exploit + explore: rank the untried grid by learned dimension scores.

    75% of the batch = highest-scoring untried combos (exploit); 25% = the
    least-explored dimension values (explore), so the fleet can't tunnel-vision
    on one month's regime. With no usable results yet, falls back to the plain
    grid (generation 0) — cold start is honest, not random.
    """
    scores = score_dimensions(results)
    if not any(scores[d] for d in scores):
        return generate_variants(symbol=symbol, max_variants=max_variants, generation=0)

    from itertools import product as _product

    tried = set()
    for r in results:
        p = _parse_variant_name(r.get("name") or r.get("strategy") or "")
        if p:
            tried.add((p["structure"], p["delta"], p["dte"], p["tp"]))

    grid = [g for g in _product(STRUCTURES, SHORT_DELTAS, DTES, PROFIT_TARGETS)
            if g not in tried
            and not (g[0] == "iron_butterfly" and g[1] >= 0.50)
            and not (g[2] == 0 and g[1] <= 0.10)]

    def combo_score(g) -> float:
        s = 0.0
        for dim, val in zip(("structure", "delta", "dte", "tp"), g):
            s += scores[dim].get(val, 0.0)
        return s

    def novelty(g) -> int:  # fewer observations across dims = more novel
        n = 0
        for dim, val in zip(("structure", "delta", "dte", "tp"), g):
            n += 1 if val in scores[dim] else 0
        return n

    n_explore = max(1, int(max_variants * explore_frac))
    exploit = sorted(grid, key=combo_score, reverse=True)[: max_variants - n_explore]
    explore = sorted((g for g in grid if g not in exploit), key=novelty)[:n_explore]

    # Reuse the canonical builder (legs, exits, guards) — one full-grid build.
    all_templates = generate_variants(symbol=symbol, max_variants=10**6, generation=0)
    out: dict[str, dict] = {}
    for structure, delta, dte, tp in exploit + explore:
        vid = f"gen_{structure}_{int(delta * 100)}d_{dte}dte_tp{tp}"
        if vid in all_templates:
            out[vid] = all_templates[vid]
        if len(out) >= max_variants:
            break
    return out
