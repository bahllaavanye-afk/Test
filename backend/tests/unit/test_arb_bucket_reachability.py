"""The arbitrage bucket claims 32 strategies. Four of them cannot trade at all.

IMPROVEMENTS.md carried this as an open question: *"32 strategies in the arb
bucket but near-zero desk fills attributed to them; verify their signals reach
a desk with an order path and aren't all filtered at the confidence gate."*

Audited 2026-07-29. Of the 32 registry strategies whose `risk_bucket` is
`arbitrage`, **28 are wired to a desk and 4 are not**:

    covered_call        needs share inventory (already documented inline in the
                        Options desk roster — a deliberate exclusion)
    crypto_basis_roll   short perp + long spot; Alpaca paper has no perpetuals
    funding_rate_arb    trades perpetual funding rates; same missing venue
    dex_cex_arb         Uniswap v3 vs CEX; no DEX connectivity exists here

So none of the four is a defect — but all four are counted in the bucket while
being structurally unable to produce an order on the current broker. That
inflates the bucket's apparent capacity by ~12% and is part of why fills
attributed to it look sparse.

This test converts that from an unexamined gap into a maintained invariant: an
arb-bucket strategy must be EITHER desk-wired OR listed below with a reason.
A newly added one that is neither fails here rather than quietly joining the
count — the same "honestly unwired" idiom as
`test_factor_exposure_is_still_honestly_unwired`.

WHAT THIS DOES NOT ANSWER: whether the 28 wired strategies actually fill, or
are filtered at the confidence gate. That needs per-strategy attribution, and
`strategy_performance.json` has never existed in the repo — the producer was
only fixed on 2026-07-29 (see CONTINUITY 03:40). Deliberately left open rather
than guessed at.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_DESK = _REPO / ".github" / "scripts" / "desk_order_placer.py"

# Arb-bucket strategies with no desk, each with the reason it cannot have one.
# Adding to this list must be a deliberate act with a stated cause.
DORMANT_BY_DESIGN = {
    "covered_call":      "needs share inventory (excluded inline in the Options desk roster)",
    "crypto_basis_roll": "short perp + long spot; Alpaca paper has no perpetual futures",
    "funding_rate_arb":  "trades perpetual funding rates; no perp venue configured",
    "dex_cex_arb":       "Uniswap v3 vs CEX; no DEX connectivity in this deployment",
}


def _desk_strategy_names() -> set[str]:
    spec = importlib.util.spec_from_file_location("dop_arb_audit", _DESK)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dop_arb_audit"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return {n for d in mod.DESKS for n in d.strategy_names}


def _arb_bucket_strategies() -> set[str]:
    from app.strategies import STRATEGY_REGISTRY

    out = set()
    for name, cls in STRATEGY_REGISTRY.items():
        try:
            bucket = getattr(cls(), "risk_bucket", None)
        except Exception:  # noqa: BLE001 — construction may need args
            bucket = getattr(cls, "risk_bucket", None)
        if bucket == "arbitrage":
            out.add(name)
    return out


@pytest.fixture(scope="module")
def arb() -> set[str]:
    names = _arb_bucket_strategies()
    assert names, "no arbitrage-bucket strategies resolved — the audit would be vacuous"
    return names


@pytest.fixture(scope="module")
def wired() -> set[str]:
    names = _desk_strategy_names()
    assert len(names) > 50, f"only {len(names)} desk strategies — desk config did not load"
    return names


def test_the_desk_file_is_where_we_think():
    assert _DESK.is_file(), f"missing {_DESK}"


def test_every_arb_strategy_is_wired_or_explicitly_dormant(arb, wired):
    """The invariant. A new arb strategy with no desk and no reason fails here."""
    orphans = sorted(arb - wired - set(DORMANT_BY_DESIGN))
    assert not orphans, (
        "These arbitrage-bucket strategies are wired to NO desk, so they can "
        "never produce an order, and no reason is recorded for it:\n  "
        + "\n  ".join(orphans)
        + "\n\nEither add them to a desk in .github/scripts/desk_order_placer.py, "
          "or add them to DORMANT_BY_DESIGN with the reason they cannot trade."
    )


def test_the_dormant_list_has_not_gone_stale(arb, wired):
    """A dormant entry that got wired (or deleted) must not linger as a lie."""
    stale = sorted(n for n in DORMANT_BY_DESIGN if n in wired)
    assert not stale, (
        f"these are listed as dormant but ARE desk-wired now: {stale} — "
        f"remove them from DORMANT_BY_DESIGN"
    )
    gone = sorted(n for n in DORMANT_BY_DESIGN if n not in arb)
    assert not gone, (
        f"these are listed as dormant but are no longer arbitrage-bucket "
        f"strategies: {gone} — remove them from DORMANT_BY_DESIGN"
    )


def test_every_dormant_entry_states_a_reason():
    for name, reason in DORMANT_BY_DESIGN.items():
        assert reason and len(reason) > 15, f"{name}: reason is too thin to audit later"


def test_the_bucket_is_mostly_reachable(arb, wired):
    """Guards against silent erosion — if this ratio collapses, the bucket has
    become a label rather than a set of tradeable strategies."""
    reachable = arb & wired
    assert len(reachable) / len(arb) >= 0.75, (
        f"only {len(reachable)}/{len(arb)} arbitrage strategies can reach an "
        f"order path"
    )
