"""
Multi-criteria strategy promotion gate.
=======================================
A single out-of-sample Sharpe is NOT enough to promote a strategy — when you
screen hundreds of strategies, the best Sharpe is mostly luck (selection bias).
This gate combines the metrics a real quant desk would require before risking a
paper-candidate slot, including the **Deflated Sharpe Ratio** (Bailey & López de
Prado, 2014), which haircuts the Sharpe for the number of trials run.

Pure/offline — no network, no LLM. Used by strategy_promotion.py Gate 1 and
covered directly by tests/test_strategy_gate.py.
"""
from __future__ import annotations

import math
from statistics import NormalDist, pstdev

_NORM = NormalDist()
_EULER = 0.5772156649015329  # Euler–Mascheroni constant

# ── Thresholds (a strategy must clear ALL hard gates) ──────────────────────────
SHARPE_MIN        = 1.0    # out-of-sample (test) Sharpe
VAL_SHARPE_MIN    = 0.5    # must still work on the validation split
MAXDD_MAX_PCT     = 20.0   # max drawdown ceiling (percent, positive)
MIN_TRADES        = 20     # statistical-significance floor — no 3-trade flukes
DSR_MIN           = 0.90   # Deflated Sharpe Ratio (prob. true SR > 0 after N trials)
OOS_CONSISTENCY   = 0.50   # test_sharpe must be >= 50% of val_sharpe (anti-overfit)
# Soft gates — only enforced when the metric is present in the backtest output
SORTINO_MIN       = 1.0
CALMAR_MIN        = 0.5
WIN_RATE_MIN      = 0.40
PROFIT_FACTOR_MIN = 1.2
OMEGA_MIN         = 1.3    # prob-weighted gains/losses above threshold
RECOVERY_MIN      = 1.0    # net return / max drawdown
TAIL_RATIO_MIN    = 1.0    # right-tail vs left-tail magnitude


def compute_advanced_metrics(returns: list[float], periods_per_year: int = 252) -> dict:
    """SOTA risk-adjusted metrics from a per-period returns series.

    Returns a dict with omega, tail_ratio, ulcer_index, cvar_95, recovery_factor,
    gain_to_pain, common_sense_ratio, cagr — the metrics a real desk looks at
    beyond Sharpe. All defensive: returns {} on too-little data.
    """
    r = [float(x) for x in returns if x is not None]
    n = len(r)
    if n < 5:
        return {}
    gains = [x for x in r if x > 0]
    losses = [x for x in r if x < 0]
    sum_gain = sum(gains)
    sum_loss = abs(sum(losses)) or 1e-9
    # Omega(0): prob-weighted gains / losses about a 0 threshold
    omega = sum_gain / sum_loss
    # Tail ratio: 95th pct / |5th pct|
    s = sorted(r)
    p95 = s[min(n - 1, int(0.95 * n))]
    p05 = s[max(0, int(0.05 * n))]
    tail_ratio = (p95 / abs(p05)) if p05 != 0 else float("inf")
    # Equity curve → drawdowns for Ulcer + recovery
    eq, peak, dd2, max_dd = 1.0, 1.0, [], 0.0
    for x in r:
        eq *= (1.0 + x)
        peak = max(peak, eq)
        d = (eq / peak) - 1.0
        dd2.append(d * d)
        max_dd = min(max_dd, d)
    ulcer = (sum(dd2) / n) ** 0.5
    total_return = eq - 1.0
    recovery_factor = (total_return / abs(max_dd)) if max_dd < 0 else float("inf")
    # CVaR 95% (expected shortfall of the worst 5%)
    k = max(1, int(0.05 * n))
    cvar_95 = sum(s[:k]) / k
    gain_to_pain = sum_gain / sum_loss
    common_sense = tail_ratio * (sum_gain / sum_loss)
    cagr = (eq ** (periods_per_year / n)) - 1.0 if eq > 0 else -1.0
    return {
        "omega": round(omega, 3),
        "tail_ratio": round(tail_ratio, 3) if tail_ratio != float("inf") else None,
        "ulcer_index": round(ulcer, 4),
        "cvar_95": round(cvar_95, 4),
        "recovery_factor": round(recovery_factor, 3) if recovery_factor != float("inf") else None,
        "gain_to_pain": round(gain_to_pain, 3),
        "common_sense_ratio": round(common_sense, 3) if common_sense == common_sense else None,
        "cagr": round(cagr, 4),
    }


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    sharpe_variance_across_trials: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR) — probability the true Sharpe > 0 after
    accounting for `n_trials` strategies having been tried.

    Returns a probability in [0, 1]; promote only when it clears DSR_MIN.

    Method (Bailey & López de Prado 2014):
      1. Benchmark SR0 = expected max Sharpe under the null across N trials.
      2. DSR = PSR(SR0): the probabilistic Sharpe ratio vs that benchmark,
         using sample length, skew and kurtosis of the returns.
    """
    if n_obs < 2 or n_trials < 1:
        return 0.0
    var_trials = max(sharpe_variance_across_trials, 1e-9)
    sigma_sr = math.sqrt(var_trials)
    if n_trials == 1:
        sr0 = 0.0
    else:
        # Expected maximum of N iid standard normals (Gumbel approximation).
        z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
        z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = sigma_sr * ((1.0 - _EULER) * z1 + _EULER * z2)
    # Probabilistic Sharpe Ratio vs SR0, adjusted for non-normal returns.
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    if denom <= 0:
        return 0.0
    psr = _NORM.cdf((sharpe - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))
    return float(psr)


def passes_promotion_gate(
    metrics: dict,
    n_trials: int = 1,
    sharpe_variance_across_trials: float = 0.25,
) -> tuple[bool, dict]:
    """Evaluate a strategy against every promotion criterion.

    `metrics` keys (test_sharpe required; the rest optional):
      test_sharpe, val_sharpe, max_dd (pct), num_trades, sortino, calmar,
      win_rate (0-1), profit_factor, skew, kurtosis.

    Returns (passed, scorecard) where scorecard maps each criterion to
    {value, threshold, ok}. A criterion that can't be evaluated (missing optional
    metric) is recorded as ok=True with value=None so it never silently fails.
    """
    sc: dict[str, dict] = {}

    def check(name, value, threshold, ok):
        sc[name] = {"value": value, "threshold": threshold, "ok": bool(ok)}

    test_sharpe = float(metrics.get("test_sharpe", 0.0) or 0.0)
    val_sharpe = float(metrics.get("val_sharpe", 0.0) or 0.0)
    max_dd = abs(float(metrics.get("max_dd", 0.0) or 0.0))
    # num_trades is optional: only enforce when the backtest actually recorded it,
    # so existing results (which don't yet emit it) aren't auto-rejected.
    raw_trades = metrics.get("num_trades")
    num_trades = int(raw_trades) if raw_trades not in (None, "") else None

    # ── Hard gates ──
    check("test_sharpe", test_sharpe, SHARPE_MIN, test_sharpe >= SHARPE_MIN)
    check("val_sharpe", val_sharpe, VAL_SHARPE_MIN, val_sharpe >= VAL_SHARPE_MIN)
    check("max_dd_pct", max_dd, MAXDD_MAX_PCT, max_dd < MAXDD_MAX_PCT)
    if num_trades is None:
        check("num_trades", None, MIN_TRADES, True)  # not recorded → don't fail
    else:
        check("num_trades", num_trades, MIN_TRADES, num_trades >= MIN_TRADES)
    # Out-of-sample consistency: test must not collapse vs validation (overfit tell)
    consistent = val_sharpe <= 0 or test_sharpe >= OOS_CONSISTENCY * val_sharpe
    check("oos_consistency", round(test_sharpe / val_sharpe, 2) if val_sharpe > 0 else None,
          OOS_CONSISTENCY, consistent)
    # Deflated Sharpe — the multiple-testing haircut. When trade count is unknown,
    # assume ~252 observations (≈1y daily) rather than collapsing the sample.
    n_obs = num_trades if (num_trades and num_trades >= 2) else 252
    dsr = deflated_sharpe_ratio(
        test_sharpe,
        n_obs=n_obs,
        n_trials=max(n_trials, 1),
        sharpe_variance_across_trials=sharpe_variance_across_trials,
        skew=float(metrics.get("skew", 0.0) or 0.0),
        kurtosis=float(metrics.get("kurtosis", 3.0) or 3.0),
    )
    check("deflated_sharpe", round(dsr, 3), DSR_MIN, dsr >= DSR_MIN)

    # ── Soft gates (only when present) ──
    for key, thr, name in (
        ("sortino", SORTINO_MIN, "sortino"),
        ("calmar", CALMAR_MIN, "calmar"),
        ("win_rate", WIN_RATE_MIN, "win_rate"),
        ("profit_factor", PROFIT_FACTOR_MIN, "profit_factor"),
        ("omega", OMEGA_MIN, "omega"),
        ("recovery_factor", RECOVERY_MIN, "recovery_factor"),
        ("tail_ratio", TAIL_RATIO_MIN, "tail_ratio"),
    ):
        if metrics.get(key) is None:
            check(name, None, thr, True)  # not evaluated → don't fail on it
        else:
            v = float(metrics[key])
            check(name, v, thr, v >= thr)

    passed = all(c["ok"] for c in sc.values())
    return passed, sc
