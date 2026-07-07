"""Bot variant factory — the systematic search Options Alpha can't do."""
from app.bots.factory import MAX_VARIANTS, generate_variants, generate_variants_ml
from app.schemas.bot import BotCreate


def _validate_generation(generation: int) -> None:
    """Validate that generation is a non‑negative integer."""
    if not isinstance(generation, int):
        raise ValueError(f"`generation` must be an integer, got {type(generation).__name__}")
    if generation < 0:
        raise ValueError(f"`generation` must be non‑negative, got {generation}")


def _validate_max_variants(max_variants: int) -> None:
    """Validate that max_variants is a positive integer."""
    if not isinstance(max_variants, int):
        raise ValueError(f"`max_variants` must be an integer, got {type(max_variants).__name__}")
    if max_variants <= 0:
        raise ValueError(f"`max_variants` must be greater than zero, got {max_variants}")


def _validate_results(results) -> None:
    """Validate that results is a list of dicts with required keys."""
    if not isinstance(results, list):
        raise ValueError(f"`results` must be a list, got {type(results).__name__}")
    required_keys = {"name", "total_pnl", "trades"}
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"`results[{idx}]` must be a dict, got {type(item).__name__}")
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"`results[{idx}]` missing required keys: {missing}")


def test_variants_validate_and_are_bounded():
    v = generate_variants()
    assert 0 < len(v) <= MAX_VARIANTS
    for tid, t in v.items():
        BotCreate(**t)  # schema‑valid → instantiable by the lifecycle manager


def test_generations_are_deterministic_and_disjoint():
    _validate_generation(0)
    g0a, g0b, g1 = (
        generate_variants(generation=0),
        generate_variants(generation=0),
        generate_variants(generation=1),
    )
    assert g0a.keys() == g0b.keys()            # deterministic
    assert not (g0a.keys() & g1.keys())        # successive generations explore fresh grid


def test_research_variants_are_small_and_guarded():
    for t in generate_variants().values():
        assert t["action"]["size_pct"] == 1.0          # research budget, not a firehose
        assert {"type": "no_position"} in t["conditions"]
        kinds = {r["type"] for r in t["exit_rules"]}
        sells = any(l["side"] == "sell" for l in t["action"]["legs"])
        assert not sells or "stop_loss" in kinds       # premium sellers always have stops


def test_zero_dte_variants_never_hold_overnight():
    _validate_generation(0)
    for t in generate_variants(generation=0).values():
        if any(l["dte"] == 0 for l in t["action"]["legs"]):
            assert any(r["type"] == "time_exit" for r in t["exit_rules"])


def test_ml_generation_biases_toward_winners():
    # iron_condor 16Δ 7DTE dominated; call spreads lost. The next generation
    # must lean toward condor/7DTE neighborhoods it hasn't tried yet.
    results = [
        {"name": "[gen] iron condor 16Δ 7DTE tp50", "total_pnl": 500.0, "trades": 10},
        {"name": "[gen] call credit spread 30Δ 30DTE tp25", "total_pnl": -300.0, "trades": 8},
    ]
    _validate_results(results)
    _validate_max_variants(8)
    v = generate_variants_ml(results, max_variants=8)
    assert len(v) == 8
    assert "gen_iron_condor_16d_7dte_tp50" not in v          # never re‑emit tried combos
    condorish = sum(1 for t in v.values() if "iron condor" in t["name"] or "7DTE" in t["name"])
    ccs = sum(1 for t in v.values() if "call credit" in t["name"])
    assert condorish > ccs                                    # exploit tilts to winners


def test_ml_generation_cold_start_falls_back_to_grid():
    _validate_results([])
    keys_ml = generate_variants_ml([]).keys()
    keys_grid = generate_variants(generation=0).keys()
    assert keys_ml == keys_grid