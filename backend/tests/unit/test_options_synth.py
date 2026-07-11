"""Synthetic options backtester — pricer sanity + end-to-end template scoring."""
import math

from app.backtest.options_synth import (
    backtest_template,
    bs_delta,
    bs_price,
    norm_ppf,
    strike_from_delta,
)


def _compute_net_credit(template, spot, sigma):
    """Calculate the net premium of the option spread defined in ``template``."""
    net = 0.0
    for leg in template["action"]["legs"]:
        # Resolve strike from delta for the given side
        delta_target = leg["delta"]
        opt_type = leg["option_type"]
        dte = leg["dte"]
        T = dte / 365.0
        K = strike_from_delta(spot, delta_target, T, sigma, opt_type)
        price = bs_price(spot, K, T, sigma, opt_type)
        # Buy legs pay premium, sell legs receive premium
        sign = -1 if leg["side"] == "buy" else 1
        net += sign * price * leg.get("ratio", 1)
    return net


def test_put_call_parity():
    S, K, T, sig = 100.0, 105.0, 30 / 365, 0.25
    c = bs_price(S, K, T, sig, "call")
    p = bs_price(S, K, T, sig, "put")
    assert abs((c - p) - (S - K * math.exp(-0.04 * T))) < 1e-6


def test_norm_ppf_roundtrip():
    from app.backtest.options_synth import norm_cdf

    for p in (0.05, 0.16, 0.5, 0.84, 0.99):
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-6


def test_strike_from_delta_inverts_bs_delta():
    S, T, sig = 500.0, 14 / 365, 0.20
    for target, ot in ((0.30, "call"), (0.16, "put")):
        K = strike_from_delta(S, target, T, sig, ot)
        assert abs(abs(bs_delta(S, K, T, sig, ot)) - target) < 0.01


def test_short_condor_profits_on_flat_tape():
    """Theta check: a short 16Δ condor on a dead‑flat tape must win."""
    template = {
        "action": {
            "type": "open_option_spread",
            "legs": [
                {"side": "sell", "option_type": "call", "delta": 0.16, "dte": 14, "ratio": 1},
                {"side": "buy", "option_type": "call", "delta": 0.05, "dte": 14, "ratio": 1},
                {"side": "sell", "option_type": "put", "delta": 0.16, "dte": 14, "ratio": 1},
                {"side": "buy", "option_type": "put", "delta": 0.05, "dte": 14, "ratio": 1},
            ],
        },
        "exit_rules": [
            {"type": "take_profit", "value": 50},
            {"type": "stop_loss", "value": 100},
        ],
    }

    # gently oscillating tape — no trend, realized vol > 0
    closes = [500 + 0.8 * math.sin(i / 3) for i in range(140)]

    # Entry filter: verify that the spread is a net credit (tight entry condition)
    spot = closes[0]
    implied_vol = 0.20  # reasonable proxy for the test environment
    net_credit = _compute_net_credit(template, spot, implied_vol)
    assert net_credit > 0, "Short condor should open with a net credit"

    r = backtest_template(template, closes)

    # Core expectations
    assert r["trades"] >= 2
    assert r["total_pnl"] > 0
    assert r["win_rate"] and r["win_rate"] >= 0.5
    assert "synthetic" in r["method"]

    # Exit logic verification: at least one trade should have hit the take‑profit target
    exit_counts = r.get("exit_counts", {})
    assert exit_counts.get("take_profit", 0) >= 1


def test_long_call_loses_on_flat_tape():
    """Debit check: long calls bleed theta when nothing moves."""
    template = {
        "action": {
            "type": "open_option_spread",
            "legs": [
                {"side": "buy", "option_type": "call", "delta": 0.60, "dte": 14, "ratio": 1},
            ],
        },
        "exit_rules": [
            {"type": "take_profit", "value": 100},
            {"type": "stop_loss", "value": 50},
        ],
    }

    closes = [500 + 0.8 * math.sin(i / 3) for i in range(140)]

    # Entry filter: ensure the long call is a net debit (tight entry condition)
    spot = closes[0]
    implied_vol = 0.20
    net_debit = _compute_net_credit(template, spot, implied_vol)
    assert net_debit < 0, "Long call should open with a net debit"

    r = backtest_template(template, closes)

    assert r["trades"] >= 2 and r["total_pnl"] < 0

    # Exit logic verification: stop‑loss should have been triggered at least once
    exit_counts = r.get("exit_counts", {})
    assert exit_counts.get("stop_loss", 0) >= 1