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
