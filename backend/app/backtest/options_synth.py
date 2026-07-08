"""Synthetic options backtester — score option-spread bot templates on history.

We have years of underlying OHLCV but no historical option chains, so spreads
are repriced with Black-Scholes using realized vol as the IV proxy (HV20 ×
IV_PREMIUM, the variance-risk-premium markup). This is the standard research
approximation; it captures theta/delta/vega mechanics and regime behavior but
NOT skew dynamics or bid/ask — results are for RANKING templates against each
other, not for promising returns. Every consumer must carry that caveat.

Pure numpy/math (no scipy): norm CDF via math.erf, inverse CDF via the
Acklam approximation. Deterministic; fully unit-testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any

IV_PREMIUM = 1.10       # implied ≈ 1.1 × realized (documented VRP assumption)
RISK_FREE = 0.04
MULTIPLIER = 100        # options contract multiplier
MIN_T = 6.5 / 24 / 365  # 0DTE priced as one trading session


def _validate_numeric(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric type, got {type(value).__name__}")


def _validate_positive(name: str, value: Any) -> None:
    _validate_numeric(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(name: str, value: Any) -> None:
    _validate_numeric(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non‑negative, got {value}")


def _validate_option_type(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    if not (value.startswith("c") or value.startswith("p")):
        raise ValueError(f"{name} must start with 'c' (call) or 'p' (put), got '{value}'")


def norm_cdf(x: float) -> float:
    _validate_numeric("x", x)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's inverse-normal approximation (|err| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str,
    r: float = RISK_FREE,
) -> float:
    """Black‑Scholes price for a European option."""
    _validate_positive("S (spot price)", S)
    _validate_positive("K (strike price)", K)
    _validate_positive("T (time to expiry)", T)
    _validate_positive("sigma (volatility)", sigma)
    _validate_option_type("option_type", option_type)
    _validate_non_negative("r (risk‑free rate)", r)

    T = max(T, MIN_T)
    sigma = max(sigma, 1e-4)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type.startswith("c"):
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str,
    r: float = RISK_FREE,
) -> float:
    """Black‑Scholes delta."""
    _validate_positive("S (spot price)", S)
    _validate_positive("K (strike price)", K)
    _validate_positive("T (time to expiry)", T)
    _validate_positive("sigma (volatility)", sigma)
    _validate_option_type("option_type", option_type)
    _validate_non_negative("r (risk‑free rate)", r)

    T = max(T, MIN_T)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if option_type.startswith("c") else norm_cdf(d1) - 1.0


def strike_from_delta(
    S: float,
    target_delta: float,
    T: float,
    sigma: float,
    option_type: str,
    r: float = RISK_FREE,
) -> float:
    """Invert BS delta → strike (calls: Δ=N(d1); puts: |Δ|=N(-d1))."""
    _validate_positive("S (spot price)", S)
    _validate_positive("T (time to expiry)", T)
    _validate_positive("sigma (volatility)", sigma)
    _validate_option_type("option_type", option_type)
    _validate_non_negative("r (risk‑free rate)", r)

    if not isinstance(target_delta, (int, float)):
        raise ValueError(f"target_delta must be numeric, got {type(target_delta).__name__}")
    if not 0 < target_delta < 1:
        raise ValueError(f"target_delta must be in (0,1), got {target_delta}")

    T = max(T, MIN_T)
    p = target_delta if option_type.startswith("c") else 1.0 - target_delta
    d1 = norm_ppf(p)
    return S * math.exp((r + sigma ** 2 / 2) * T - d1 * sigma * math.sqrt(T))


@dataclass
class _Leg:
    sign: int          # +1 buy, -1 sell
    option_type: str
    strike: float
    ratio: int


def _net_value(legs: List[_Leg], S: float, T: float, sigma: float) -> float:
    return sum(l.sign * l.ratio * bs_price(S, l.strike, T, sigma, l.option_type) for l in legs)


def backtest_template(
    template: Dict[str, Any],
    closes: List[float],
    trading_days_per_entry: int = 1,
) -> Dict[str, Any]:
    """Walk daily closes; open the template's spread whenever flat, manage exits.

    Entry uses HV20×IV_PREMIUM as sigma and resolves strikes from leg deltas.
    Exits: take_profit/stop_loss as % of entry premium (both credit and debit),
    plus expiry settlement. Returns ranking metrics — see module caveat.
    """
    # Basic input validation
    if not isinstance(template, dict):
        raise ValueError("template must be a dict")
    if "action" not in template or not isinstance(template["action"], dict):
        raise ValueError("template must contain an 'action' dict")
    action = template["action"]
    if "legs" not in action or not isinstance(action["legs"], list) or not action["legs"]:
        raise ValueError("action must contain a non‑empty list of 'legs'")
    if not isinstance(closes, list) or len(closes) < 22:
        raise ValueError("closes must be a list with at least 22 price entries")
    for i, price in enumerate(closes):
        if not isinstance(price, (int, float)):
            raise ValueError(f"closes[{i}] must be numeric, got {type(price).__name__}")
        if price <= 0:
            raise ValueError(f"closes[{i}] must be positive, got {price}")

    if not isinstance(trading_days_per_entry, int) or trading_days_per_entry < 1:
        raise ValueError("trading_days_per_entry must be a positive integer")

    tp_pct = next(
        (r["value"] for r in template.get("exit_rules", []) if r.get("type") == "take_profit"),
        50,
    )
    if not isinstance(tp_pct, (int, float)):
        raise ValueError("take_profit value must be numeric")
    sl_pct = next(
        (r["value"] for r in template.get("exit_rules", []) if r.get("type") == "stop_loss"),
        None,
    )
    if sl_pct is not None and not isinstance(sl_pct, (int, float)):
        raise ValueError("stop_loss value must be numeric")

    trades: List[float] = []
    pos: List[_Leg] | None = None
    entry_net = 0.0
    days_held = 0
    dte = max(int(action["legs"][0].get("dte", 30)), 0)

    for i in range(21, len(closes)):
        S = closes[i]
        # Historical volatility estimation
        rets = [math.log(closes[j] / closes[j - 1]) for j in range(i - 19, i + 1)]
        mean = sum(rets) / len(rets)
        hv = math.sqrt(sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)) * math.sqrt(252)
        sigma = max(hv * IV_PREMIUM, 0.05)

        if pos is None:
            T0 = max(dte, 1) / 365.0
            pos = []
            for lg in action["legs"]:
                if lg.get("strike"):
                    K = float(lg["strike"])
                    _validate_positive("leg strike", K)
                else:
                    delta_val = float(lg.get("delta") or 0.5)
                    _validate_positive("leg delta (as absolute)", delta_val)
                    K = strike_from_delta(
                        S,
                        delta_val,
                        T0,
                        sigma,
                        lg["option_type"],
                    )
                _validate_option_type("leg option_type", lg["option_type"])
                ratio = int(lg.get("ratio", 1))
                if ratio <= 0:
                    raise ValueError("leg ratio must be positive")
                sign = +1 if lg["side"] == "buy" else -1
                pos.append(_Leg(sign, lg["option_type"], K, ratio))
            entry_net = _net_value(pos, S, T0, sigma)
            days_held = 0
            continue

        days_held += 1
        T_rem = max(dte - days_held, 0) / 365.0
        cur = (
            _net_value(pos, S, T_rem, sigma)
            if T_rem > 0
            else sum(
                l.sign
                * l.ratio
                * max((S - l.strike) if l.option_type.startswith("c") else (l.strike - S), 0.0)
                for l in pos
            )
        )
        pnl = (cur - entry_net) * MULTIPLIER
        base = max(abs(entry_net) * MULTIPLIER, 1.0)

        expired = days_held >= max(dte, 1)
        hit_tp = pnl >= (tp_pct / 100.0) * base
        hit_sl = sl_pct is not None and pnl <= -(sl_pct / 100.0) * base
        if hit_tp or hit_sl or expired:
            trades.append(pnl)
            pos = None

    n = len(trades)
    wins = sum(1 for t in trades if t > 0)
    total = sum(trades)
    cum, peak, mdd = 0.0, 0.0, 0.0
    for t in trades:
        cum += t
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return {
        "trades": n,
        "win_rate": round(wins / n, 4) if n else None,
        "total_pnl": round(total, 2),
        "avg_pnl": round(total / n, 2) if n else None,
        "max_drawdown": round(mdd, 2),
        "method": "synthetic-BS (HV20×1.1 IV proxy) — ranking signal, not a return promise",
    }