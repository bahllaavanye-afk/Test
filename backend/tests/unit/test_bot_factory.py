"""Bot variant factory — the systematic search Options Alpha can't do."""
from app.bots.factory import MAX_VARIANTS, generate_variants
from app.schemas.bot import BotCreate
from typing import Optional


def _validate_generate_variants_params(
    generation: Optional[int] = None, max_variants: Optional[int] = None
) -> None:
    """Validate parameters for ``generate_variants``.

    Args:
        generation: Optional generation index; must be a non‑negative integer if provided.
        max_variants: Optional upper bound for variants; must be a positive integer if provided.

    Raises:
        ValueError: If any argument fails validation.
    """
    if generation is not None:
        if not isinstance(generation, int):
            raise ValueError(
                f"`generation` must be an integer, got {type(generation).__name__}"
            )
        if generation < 0:
            raise ValueError("`generation` must be non‑negative")
    if max_variants is not None:
        if not isinstance(max_variants, int):
            raise ValueError(
                f"`max_variants` must be an integer, got {type(max_variants).__name__}"
            )
        if max_variants <= 0:
            raise ValueError("`max_variants` must be a positive integer")


def test_variants_validate_and_are_bounded():
    _validate_generate_variants_params()
    v = generate_variants()
    assert 0 < len(v) <= MAX_VARIANTS
    for tid, t in v.items():
        BotCreate(**t)  # schema‑valid → instantiable by the lifecycle manager


def test_generations_are_deterministic_and_disjoint():
    _validate_generate_variants_params(generation=0)
    g0a = generate_variants(generation=0)
    _validate_generate_variants_params(generation=0)
    g0b = generate_variants(generation=0)
    _validate_generate_variants_params(generation=1)
    g1 = generate_variants(generation=1)
    assert g0a.keys() == g0b.keys()            # deterministic
    assert not (g0a.keys() & g1.keys())        # successive generations explore fresh grid


def test_research_variants_are_small_and_guarded():
    _validate_generate_variants_params()
    for t in generate_variants().values():
        assert t["action"]["size_pct"] == 1.0          # research budget, not a firehose
        assert {"type": "no_position"} in t["conditions"]
        kinds = {r["type"] for r in t["exit_rules"]}
        sells = any(l["side"] == "sell" for l in t["action"]["legs"])
        assert not sells or "stop_loss" in kinds       # premium sellers always have stops


def test_zero_dte_variants_never_hold_overnight():
    _validate_generate_variants_params(generation=0)
    for t in generate_variants(generation=0).values():
        if any(l["dte"] == 0 for l in t["action"]["legs"]):
            assert any(r["type"] == "time_exit" for r in t["exit_rules"])


def test_ml_generation_biases_toward_winners():
    from app.bots.factory import generate_variants_ml

    # iron_condor 16Δ 7DTE dominated; call spreads lost. The next generation
    # must lean toward condor/7DTE neighborhoods it hasn't tried yet.
    results = [
        {"name": "[gen] iron condor 16Δ 7DTE tp50", "total_pnl": 500.0, "trades": 10},
        {"name": "[gen] call credit spread 30Δ 30DTE tp25", "total_pnl": -300.0, "trades": 8},
    ]
    _validate_generate_variants_params(max_variants=8)
    v = generate_variants_ml(results, max_variants=8)
    assert len(v) == 8
    assert "gen_iron_condor_16d_7dte_tp50" not in v          # never re‑emit tried combos
    condorish = sum(1 for t in v.values() if "iron condor" in t["name"] or "7DTE" in t["name"])
    ccs = sum(1 for t in v.values() if "call credit" in t["name"])
    assert condorish > ccs                                    # exploit tilts to winners


def test_ml_generation_cold_start_falls_back_to_grid():
    from app.bots.factory import generate_variants, generate_variants_ml

    _validate_generate_variants_params(generation=0)
    assert generate_variants_ml([]).keys() == generate_variants(generation=0).keys()