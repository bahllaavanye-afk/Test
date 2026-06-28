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
    ):
        if metrics.get(key) is None:
            check(name, None, thr, True)  # not evaluated → don't fail on it
        else:
            v = float(metrics[key])
            check(name, v, thr, v >= thr)

    passed = all(c["ok"] for c in sc.values())
    return passed, sc
