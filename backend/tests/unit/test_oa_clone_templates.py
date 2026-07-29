"""The 11 Options Alpha clone bots (user's OA account, 2026-07-06 screenshot).

Every template must validate against BotCreate (so the UI, seeding, and the
lifecycle manager can instantiate them), run at 1‑minute cadence with the
$2,500 (2.5%) allocation, and carry real exit rules — no fire‑and‑forget bots.
"""
from app.bots.templates import BOT_TEMPLATES
from app.schemas.bot import BotCreate

OA_IDS = [
    "oa_friday_14dte_bwb",
    "oa_ib_0945_10d",
    "oa_ncr_0dte_spx_a",
    "oa_vix_range",
    "oa_ncr_spx_0dte_b",
    "oa_iron_butter_clone",
    "oa_short_put_spread",
    "oa_ccs_friday_15d",
    "oa_long_calls_confirm",
    "oa_iron_boi",
    "oa_delta_adjusting_strangle",
    "oa_steamrolled_spy",
    "oa_steamrolled_qqq",
    "oa_td_npdt_s1_trend",
    "oa_trendy_short_put",
    "hybrid_regime_gamma",
    "hybrid_regime_theta",
    "hybrid_put_ladder_trend",
    "hybrid_wide_0dte_tp25",
    "hybrid_morning_momentum",
]


def _validate_oa_ids(oa_ids):
    """Validate that OA_IDS is a non‑empty list of strings."""
    if not isinstance(oa_ids, list):
        raise ValueError("OA_IDS must be a list.")
    if not oa_ids:
        raise ValueError("OA_IDS cannot be empty.")
    for idx, val in enumerate(oa_ids):
        if not isinstance(val, str):
            raise ValueError(f"OA_IDS element at index {idx} is not a string.")


def _validate_template_id(tid):
    """Validate that a template identifier exists and has the required structure."""
    if not isinstance(tid, str):
        raise ValueError("Template identifier must be a string.")
    if tid not in BOT_TEMPLATES:
        raise ValueError(f"Template identifier '{tid}' not found in BOT_TEMPLATES.")
    template = BOT_TEMPLATES[tid]
    if not isinstance(template, dict):
        raise ValueError(f"Template '{tid}' must be a dictionary.")
    required_keys = {"trigger", "action", "market_type", "exit_rules", "conditions"}
    missing = required_keys - template.keys()
    if missing:
        raise ValueError(f"Template '{tid}' is missing required keys: {missing}.")
    return template


def test_all_11_oa_clones_exist():
    _validate_oa_ids(OA_IDS)
    missing = [i for i in OA_IDS if i not in BOT_TEMPLATES]
    if missing:
        raise ValueError(f"The following OA IDs are missing from BOT_TEMPLATES: {missing}.")
    assert [i for i in OA_IDS if i in BOT_TEMPLATES] == OA_IDS


def test_oa_clones_validate_against_bot_schema():
    _validate_oa_ids(OA_IDS)
    for tid in OA_IDS:
        template = _validate_template_id(tid)
        BotCreate(**template)  # raises on any schema violation


def test_oa_clones_one_minute_and_2500_allocation():
    _validate_oa_ids(OA_IDS)
    for tid in OA_IDS:
        t = _validate_template_id(tid)
        if t["trigger"]["interval"] != "1m":
            raise ValueError(f"{tid} trigger interval is not '1m'.")
        if t["action"]["size_pct"] != 2.5:
            raise ValueError(f"{tid} action size_pct is not 2.5.")
        if t["market_type"] != "options":
            raise ValueError(f"{tid} market_type is not 'options'.")


def test_oa_clones_always_have_exits_and_entry_guard():
    _validate_oa_ids(OA_IDS)
    for tid in OA_IDS:
        t = _validate_template_id(tid)
        if not t["exit_rules"]:
            raise ValueError(f"{tid} has no exit rules.")
        cond_types = {c["type"] for c in t["conditions"] if isinstance(c, dict)}
        if "no_position" not in cond_types:
            raise ValueError(f"{tid} missing no_position guard.")


def test_short_premium_clones_have_stops():
    _validate_oa_ids(OA_IDS)
    for tid in OA_IDS:
        t = _validate_template_id(tid)
        sells = any(l.get("side") == "sell" for l in t["action"]["legs"])
        if sells:
            kinds = {r.get("type") for r in t["exit_rules"] if isinstance(r, dict)}
            if "stop_loss" not in kinds:
                raise ValueError(f"{tid} sells premium without a stop loss.")