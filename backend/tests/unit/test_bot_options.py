"""Options productization: templates parse, engine emits a spread plan, and the
TradeStation request-builders produce correct multi-leg order bodies.

All pure/offline — no DB, no network, no broker credentials.
"""
from datetime import date

import pytest

from app.bots.templates import BOT_TEMPLATES
from app.brokers.tradestation import TradeStationBroker
from app.schemas.bot import ActionConfig, BotCreate, OptionLeg


# --------------------------------------------------------------------------- #
# Templates                                                                    #
# --------------------------------------------------------------------------- #
def test_every_template_parses_as_botcreate():
    """Catch malformed templates at build time, not at seed/runtime."""
    for tid, t in BOT_TEMPLATES.items():
        BotCreate(**{k: v for k, v in t.items() if k != "template_id"})


OPTION_TEMPLATE_IDS = [
    tid for tid, t in BOT_TEMPLATES.items() if t.get("market_type") == "options"
]


def test_options_templates_exist():
    # bull put, iron condor, covered call, long straddle
    assert len(OPTION_TEMPLATE_IDS) >= 4


@pytest.mark.parametrize("tid", OPTION_TEMPLATE_IDS)
def test_options_templates_have_valid_legs(tid):
    t = BOT_TEMPLATES[tid]
    action = ActionConfig(**t["action"])
    assert action.type == "open_option_spread"
    assert action.legs, f"{tid} must declare legs"
    for leg in action.legs:
        assert leg.side in ("buy", "sell")
        assert leg.option_type in ("call", "put")
        assert leg.dte > 0
        assert leg.ratio >= 1


def test_iron_condor_is_four_legged_and_balanced():
    action = ActionConfig(**BOT_TEMPLATES["opt_iron_condor"]["action"])
    assert len(action.legs) == 4
    sells = [lg for lg in action.legs if lg.side == "sell"]
    buys = [lg for lg in action.legs if lg.side == "buy"]
    assert len(sells) == 2 and len(buys) == 2  # defined risk: every short is hedged


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #
def test_option_leg_defaults():
    leg = OptionLeg(side="sell", option_type="put")
    assert leg.dte == 30 and leg.ratio == 1 and leg.delta is None and leg.strike is None


def test_open_option_spread_is_valid_action_type():
    a = ActionConfig(type="open_option_spread", legs=[{"side": "buy", "option_type": "call"}])
    assert a.type == "open_option_spread" and len(a.legs) == 1


# --------------------------------------------------------------------------- #
# TradeStation request-builders (pure, no creds/network)                       #
# --------------------------------------------------------------------------- #
def test_build_option_symbol_call_and_put():
    assert (
        TradeStationBroker.build_option_symbol("spy", date(2024, 1, 19), 447.5, "call")
        == "SPY 240119C447.5"
    )
    # whole-number strike drops the trailing .0
    assert (
        TradeStationBroker.build_option_symbol("AAPL", date(2024, 3, 15), 150.0, "put")
        == "AAPL 240315P150"
    )


def test_build_option_order_body_bull_put_spread():
    legs = [
        {"symbol": "SPY 240119P440", "side": "sell", "ratio": 1},
        {"symbol": "SPY 240119P430", "side": "buy", "ratio": 1},
    ]
    body = TradeStationBroker.build_option_order_body("ACC123", legs, quantity=2)
    assert body["AccountID"] == "ACC123"
    assert body["Quantity"] == "2"
    assert body["OrderType"] == "Market"
    assert len(body["Legs"]) == 2
    assert body["Legs"][0]["TradeAction"] == "SELLTOOPEN"
    assert body["Legs"][1]["TradeAction"] == "BUYTOOPEN"
    # leg quantity = ratio * spread quantity
    assert body["Legs"][0]["Quantity"] == "2"


def test_build_option_order_body_limit_and_closing():
    legs = [{"symbol": "QQQ 240119C400", "side": "buy", "ratio": 2}]
    body = TradeStationBroker.build_option_order_body(
        "ACC", legs, quantity=1, order_type="limit", limit_price=1.25, opening=False
    )
    assert body["OrderType"] == "Limit"
    assert body["LimitPrice"] == "1.25"
    assert body["Legs"][0]["TradeAction"] == "BUYTOCLOSE"
    assert body["Legs"][0]["Quantity"] == "2"


def test_build_option_order_body_rejects_empty_legs():
    with pytest.raises(ValueError):
        TradeStationBroker.build_option_order_body("ACC", [])
