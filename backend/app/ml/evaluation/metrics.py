"""
QuantEdge — Advanced Strategy Evaluation Metrics

All functions accept pd.Series of per-period strategy returns (e.g. daily).
Benchmark series is optional; when provided, relative metrics are computed.

No scipy. Pure numpy / pandas throughout.

Research references:
  Sharpe (1966), Sortino & Price (1994), Young (1991) Calmar, Kaplan & Knowles
  (2004) Omega, Martin (2012) Serenity/Ulcer, Lo (2002) IR, Black-Scholes CAPM.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    # ── Core ──────────────────────────────────────────────────────────────────
    total_return:       float = 0.0   # compounded return over period
    ann_return:         float = 0.0   # CAGR
    ann_volatility:     float = 0.0   # annualised std dev
    # ── Risk-adjusted ratios ──────────────────────────────────────────────────
    sharpe:             float = 0.0   # (ann_return - rf) / ann_vol
    sortino:            float = 0.0   # (ann_return - rf) / downside_vol
    calmar:             float = 0.0   # ann_return / |max_drawdown|
    omega:              float = 0.0   # prob-weighted gains / losses above threshold
    serenity:           float = 0.0   # ann_return / ulcer_index
    sterling:           float = 0.0   # ann_return / avg_top3_drawdowns
    pain_ratio:         float = 0.0   # ann_return / pain_index
    recovery_factor:    float = 0.0   # total_return / |max_drawdown|
    gain_to_pain:       float = 0.0   # sum(returns) / sum(|negative_returns|)
    tail_ratio:         float = 0.0   # 95th pct return / |5th pct return|
    # ── Drawdown ──────────────────────────────────────────────────────────────
    max_drawdown:       float = 0.0   # peak-to-trough, negative number
    avg_drawdown:       float = 0.0
    ulcer_index:        float = 0.0   # RMS of drawdown series
    pain_index:         float = 0.0   # mean of drawdown series
    max_dd_duration:    int   = 0     # bars from peak to recovery
    avg_dd_duration:    float = 0.0
    n_drawdown_periods: int   = 0
    # ── Trade statistics ──────────────────────────────────────────────────────
    n_trades:           int   = 0
    win_rate:           float = 0.0   # fraction of winning trades
    profit_factor:      float = 0.0   # gross_profit / gross_loss
    payoff_ratio:       float = 0.0   # avg_win / avg_loss
    expected_value:     float = 0.0   # win_rate * avg_win - loss_rate * avg_loss
    avg_trade_return:   float = 0.0
    avg_win:            float = 0.0
    avg_loss:           float = 0.0
    max_consec_wins:    int   = 0
    max_consec_losses:  int   = 0
    # ── Risk ──────────────────────────────────────────────────────────────────
    var_95:             float = 0.0   # 1-period 95% VaR (historical)
    var_99:             float = 0.0
    cvar_95:            float = 0.0   # Expected Shortfall at 95%
    skewness:           float = 0.0   # return distribution skewness
    excess_kurtosis:    float = 0.0   # excess kurtosis (normal = 0)
    # ── Benchmark-relative (populated only when benchmark provided) ───────────
    alpha:              float = 0.0   # CAPM alpha (annualised)
    beta:               float = 0.0   # CAPM beta
    information_ratio:  float = 0.0   # active_return / tracking_error
    tracking_error:     float = 0.0   # annualised std(active_returns)
    active_return:      float = 0.0   # ann_return - benchmark_ann_return
    up_capture:         float = 0.0   # return in bull periods / benchmark return
    down_capture:       float = 0.0   # return in bear periods / benchmark return
    hit_rate_vs_bench:  float = 0.0   # % periods outperforming benchmark
    # ── Monthly stats ─────────────────────────────────────────────────────────
    hit_rate_monthly:   float = 0.0   # % months with positive return
    best_month:         float = 0.0
    worst_month:        float = 0.0
    avg_monthly_return: float = 0.0
    # ── ML signal quality (populated when signal series provided) ─────────────
    ic:                 float = 0.0   # Pearson IC: corr(signal, next_return)
    rank_ic:            float = 0.0   # Spearman Rank IC
    icir:               float = 0.0   # IC / std(IC) over rolling windows
    factor_turnover:    float = 0.0   # mean |signal_t - signal_{t-1}| / std(signal)

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


# ─── Helpers ───────────────────────────────────────────────────────────────────

ANN_FACTORS = {252: "daily", 52: "weekly", 12: "monthly"}


def _ann_factor(returns: pd.Series) -> int:
    """Guess annualisation factor from average time between observations."""
    if len(returns) < 2:
        return 252
    if isinstance(returns.index, pd.DatetimeIndex):
        delta = (returns.index[-1] - returns.index[0]).days / len(returns)
        if delta < 2:
            return 252
        if delta < 10:
            return 52
        return 12
    return 252


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def _drawdown_series(cum: pd.Series) -> pd.Series:
    peak = cum.cummax()
    return (cum - peak) / peak


# ─── Core computation ──────────────────────────────────────────────────────────

def compute_metrics(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    signal: pd.Series | None = None,
    entries: pd.Series | None = None,
    exits: pd.Series | None = None,
    rf: float = 0.0,
    ann_factor: int | None = None,
) -> PerformanceMetrics:
    """
    Compute the full suite of performance metrics.

    Args:
        returns:   per-period strategy returns (not cumulative)
        benchmark: per-period benchmark returns (e.g. SPY daily returns)
        signal:    raw ML signal series aligned with returns (for IC metrics)
        entries:   boolean entry series (for trade-level stats)
        exits:     boolean exit series  (for trade-level stats)
        rf:        risk-free rate per period (default 0 for paper trading)
        ann_factor: override annualisation factor (252 daily, 52 weekly, 12 monthly)
    """
    m  = PerformanceMetrics()
    r  = returns.dropna().astype(float)
    af = ann_factor or _ann_factor(r)

    if len(r) < 5:
        return m

    # ── Core ──────────────────────────────────────────────────────────────────
    cum     = _equity_curve(r)
    m.total_return   = float(cum.iloc[-1] - 1)
    n_years          = len(r) / af
    m.ann_return     = float((cum.iloc[-1]) ** (1 / max(n_years, 1e-9)) - 1) if n_years > 0 else 0.0
    m.ann_volatility = float(r.std() * math.sqrt(af))

    excess = r - rf
    m.sharpe  = float((excess.mean() * af) / (r.std() * math.sqrt(af))) if r.std() > 0 else 0.0

    neg = r[r < rf]
    down_vol = float(neg.std() * math.sqrt(af)) if len(neg) > 1 else 1e-9
    m.sortino = float((m.ann_return - rf * af) / down_vol)

    # ── Drawdown series ────────────────────────────────────────────────────────
    dd = _drawdown_series(cum)
    m.max_drawdown = float(dd.min())
    m.avg_drawdown = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    m.ulcer_index  = float(math.sqrt((dd ** 2).mean()))
    m.pain_index   = float(dd.abs().mean())

    # Drawdown periods (contiguous sequences of negative DD)
    in_dd = (dd < -1e-6).astype(int)
    transitions = in_dd.diff().fillna(0)
    starts = r.index[transitions == 1].tolist()
    ends   = r.index[transitions == -1].tolist()
    m.n_drawdown_periods = len(starts)

    if starts:
        # Ensure paired starts/ends
        if ends and ends[0] < starts[0]:
            ends = ends[1:]
        durations = []
        for i, s in enumerate(starts):
            e = ends[i] if i < len(ends) else r.index[-1]
            s_loc = r.index.get_loc(s) if s in r.index else 0
            e_loc = r.index.get_loc(e) if e in r.index else len(r) - 1
            durations.append(e_loc - s_loc)
        m.max_dd_duration = int(max(durations)) if durations else 0
        m.avg_dd_duration = float(np.mean(durations)) if durations else 0.0

    # ── Risk-adjusted ratios ───────────────────────────────────────────────────
    m.calmar = float(m.ann_return / abs(m.max_drawdown)) if m.max_drawdown < 0 else 0.0
    m.serenity = float(m.ann_return / m.ulcer_index) if m.ulcer_index > 0 else 0.0
    m.pain_ratio = float(m.ann_return / m.pain_index) if m.pain_index > 0 else 0.0
    m.recovery_factor = float(m.total_return / abs(m.max_drawdown)) if m.max_drawdown < 0 else 0.0

    # Gain to Pain
    pos_sum = float(r[r > 0].sum())
    neg_sum = float(r[r < 0].abs().sum())
    m.gain_to_pain = pos_sum / neg_sum if neg_sum > 0 else pos_sum * 100.0

    # Omega Ratio (threshold = rf per period)
    above = (r - rf)[r > rf].sum()
    below = (rf - r)[r < rf].sum()
    m.omega = float(above / below) if below > 0 else 100.0

    # Sterling: ann_return / mean of 3 worst drawdowns
    dd_periods = []
    if starts:
        for i, s in enumerate(starts):
            e = ends[i] if i < len(ends) else r.index[-1]
            period_dd = float(dd[s:e].min()) if s in dd.index else 0.0
            dd_periods.append(abs(period_dd))
    top3 = sorted(dd_periods, reverse=True)[:3]
    m.sterling = float(m.ann_return / np.mean(top3)) if top3 and np.mean(top3) > 0 else 0.0

    # Tail Ratio
    p95 = float(np.percentile(r, 95))
    p5  = float(np.percentile(r, 5))
    m.tail_ratio = float(p95 / abs(p5)) if abs(p5) > 1e-9 else 0.0

    # ── Risk ──────────────────────────────────────────────────────────────────
    m.var_95 = float(np.percentile(r, 5))
    m.var_99 = float(np.percentile(r, 1))
    m.cvar_95 = float(r[r <= m.var_95].mean()) if (r <= m.var_95).any() else m.var_95

    n = len(r)
    mean_r = float(r.mean())
    std_r  = float(r.std())
    if std_r > 0 and n > 3:
        m.skewness = float(((r - mean_r) ** 3).mean() / std_r ** 3)
        m.excess_kurtosis = float(((r - mean_r) ** 4).mean() / std_r ** 4) - 3.0

    # ── Monthly stats ─────────────────────────────────────────────────────────
    if isinstance(r.index, pd.DatetimeIndex):
        monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        if len(monthly) > 0:
            m.hit_rate_monthly   = float((monthly > 0).mean())
            m.best_month         = float(monthly.max())
            m.worst_month        = float(monthly.min())
            m.avg_monthly_return = float(monthly.mean())

    # ── Trade-level stats ─────────────────────────────────────────────────────
    if entries is not None and exits is not None:
        _fill_trade_stats(m, r, entries, exits)

    # ── Benchmark-relative ────────────────────────────────────────────────────
    if benchmark is not None:
        b = benchmark.reindex(r.index).fillna(0).astype(float)
        _fill_benchmark_stats(m, r, b, af, rf)

    # ── ML signal quality ─────────────────────────────────────────────────────
    if signal is not None:
        _fill_signal_stats(m, r, signal)

    return m


# ─── Trade-level helper ────────────────────────────────────────────────────────

def _fill_trade_stats(m: PerformanceMetrics, r: pd.Series,
                      entries: pd.Series, exits: pd.Series) -> None:
    """Reconstruct trade P&L from entry/exit signals and fill trade stats."""
    trade_returns: list[float] = []
    in_trade = False
    trade_start_return = 0.0
    cumulative = 0.0

    entries_b = entries.reindex(r.index).fillna(False).astype(bool)
    exits_b   = exits.reindex(r.index).fillna(False).astype(bool)

    for i in range(len(r)):
        ret = float(r.iloc[i])
        if not in_trade and entries_b.iloc[i]:
            in_trade = True
            cumulative = 0.0
        if in_trade:
            cumulative += ret
            if exits_b.iloc[i]:
                trade_returns.append(cumulative)
                in_trade = False

    if not trade_returns:
        return

    arr = np.array(trade_returns)
    wins  = arr[arr > 0]
    losses = arr[arr < 0]

    m.n_trades     = len(arr)
    m.win_rate     = float(len(wins) / len(arr)) if len(arr) > 0 else 0.0
    m.avg_win      = float(wins.mean())  if len(wins) > 0  else 0.0
    m.avg_loss     = float(losses.mean()) if len(losses) > 0 else 0.0
    m.avg_trade_return = float(arr.mean())

    gross_profit = float(wins.sum())           if len(wins) > 0   else 0.0
    gross_loss   = float(np.abs(losses).sum()) if len(losses) > 0 else 1e-9
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit * 100.0

    m.payoff_ratio = float(abs(m.avg_win / m.avg_loss)) if m.avg_loss != 0 else 0.0
    loss_rate      = 1.0 - m.win_rate
    m.expected_value = m.win_rate * m.avg_win + loss_rate * m.avg_loss

    # Max consecutive wins / losses
    consec_w = consec_l = max_w = max_l = 0
    for tr in arr:
        if tr > 0:
            consec_w += 1; consec_l = 0
        else:
            consec_l += 1; consec_w = 0
        max_w = max(max_w, consec_w)
        max_l = max(max_l, consec_l)
    m.max_consec_wins   = max_w
    m.max_consec_losses = max_l


# ─── Benchmark-relative helper ────────────────────────────────────────────────

def _fill_benchmark_stats(m: PerformanceMetrics, r: pd.Series, b: pd.Series,
                          af: int, rf: float) -> None:
    active = r - b
    m.tracking_error = float(active.std() * math.sqrt(af))
    m.active_return  = m.ann_return - float((b + 1).prod() ** (af / max(len(b), 1)) - 1)
    m.information_ratio = m.active_return / m.tracking_error if m.tracking_error > 0 else 0.0
    m.hit_rate_vs_bench = float((active > 0).mean())

    # CAPM beta / alpha
    b_var = float(b.var())
    if b_var > 0:
        m.beta  = float(np.cov(r, b)[0, 1] / b_var)
        m.alpha = float((m.ann_return - rf * af) - m.beta * (
            float((b + 1).prod() ** (af / max(len(b), 1)) - 1) - rf * af
        ))
    else:
        m.beta = 1.0; m.alpha = 0.0

    # Up / Down capture
    up_mask   = b > 0
    down_mask = b < 0
    if up_mask.any():
        bench_up = float((b[up_mask] + 1).prod() - 1)
        strat_up = float((r[up_mask] + 1).prod() - 1)
        m.up_capture = strat_up / bench_up if abs(bench_up) > 1e-9 else 0.0
    if down_mask.any():
        bench_dn = float((b[down_mask] + 1).prod() - 1)
        strat_dn = float((r[down_mask] + 1).prod() - 1)
        m.down_capture = strat_dn / bench_dn if abs(bench_dn) > 1e-9 else 0.0


# ─── ML signal quality helper ─────────────────────────────────────────────────

def _fill_signal_stats(m: PerformanceMetrics, r: pd.Series,
                       signal: pd.Series) -> None:
    """Information Coefficient and related signal quality metrics."""
    sig = signal.reindex(r.index).shift(1).dropna()  # signal predicts NEXT return
    ret = r.reindex(sig.index).dropna()
    aligned_sig = sig.reindex(ret.index).dropna()
    aligned_ret = ret.reindex(aligned_sig.index)

    if len(aligned_sig) < 10:
        return

    # Pearson IC
    s_arr = aligned_sig.values.astype(float)
    r_arr = aligned_ret.values.astype(float)
    if np.std(s_arr) > 0 and np.std(r_arr) > 0:
        m.ic = float(np.corrcoef(s_arr, r_arr)[0, 1])

    # Spearman Rank IC
    rank_s = np.argsort(np.argsort(s_arr)).astype(float)
    rank_r = np.argsort(np.argsort(r_arr)).astype(float)
    if np.std(rank_s) > 0 and np.std(rank_r) > 0:
        m.rank_ic = float(np.corrcoef(rank_s, rank_r)[0, 1])

    # Rolling IC for ICIR (20-period rolling windows)
    window = min(20, len(aligned_sig) // 3)
    if window >= 5:
        rolling_ics = []
        for i in range(window, len(s_arr)):
            w_s = s_arr[i - window : i]
            w_r = r_arr[i - window : i]
            if np.std(w_s) > 0 and np.std(w_r) > 0:
                rolling_ics.append(float(np.corrcoef(w_s, w_r)[0, 1]))
        if rolling_ics:
            ic_mean = float(np.mean(rolling_ics))
            ic_std  = float(np.std(rolling_ics))
            m.icir  = ic_mean / ic_std if ic_std > 0 else 0.0

    # Factor turnover
    diffs = np.abs(np.diff(s_arr))
    m.factor_turnover = float(np.mean(diffs) / np.std(s_arr)) if np.std(s_arr) > 0 else 0.0


# ─── Convenience: from position series ────────────────────────────────────────

def metrics_from_positions(
    position: pd.Series,
    prices: pd.Series,
    benchmark_prices: pd.Series | None = None,
    signal: pd.Series | None = None,
    rf: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute metrics from a {-1, 0, 1} position series and price series.

    Args:
        position:         position series (1 = long, -1 = short, 0 = flat)
        prices:           close price series
        benchmark_prices: benchmark close prices for relative metrics
        signal:           raw ML signal for IC computation
    """
    raw_returns = prices.pct_change().fillna(0.0)
    strat_r = (position.shift(1).fillna(0) * raw_returns)

    bench_r = None
    if benchmark_prices is not None:
        bench_r = benchmark_prices.pct_change().reindex(strat_r.index).fillna(0.0)

    return compute_metrics(strat_r, benchmark=bench_r, signal=signal, rf=rf)


# ─── Convenience: from entry/exit signals ─────────────────────────────────────

def metrics_from_signals(
    entries: pd.Series,
    exits: pd.Series,
    prices: pd.Series,
    benchmark_prices: pd.Series | None = None,
    signal: pd.Series | None = None,
    rf: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute metrics from boolean entry/exit signals.

    Builds a position series, then delegates to metrics_from_positions.
    """
    position = pd.Series(0, index=prices.index, dtype=float)
    in_trade = False
    entries_b = entries.reindex(prices.index).fillna(False).astype(bool)
    exits_b   = exits.reindex(prices.index).fillna(False).astype(bool)

    for i in range(len(prices)):
        if not in_trade and entries_b.iloc[i]:
            in_trade = True
        elif in_trade and exits_b.iloc[i]:
            in_trade = False
        position.iloc[i] = 1.0 if in_trade else 0.0

    return metrics_from_positions(
        position, prices, benchmark_prices, signal, rf=rf
    )


# ─── Formatting helpers ────────────────────────────────────────────────────────

def format_metrics_slack(m: PerformanceMetrics, name: str = "") -> str:
    """Return a Slack-formatted metrics summary block."""
    def pct(v: float) -> str:
        return f"{v:+.1%}"
    def f2(v: float) -> str:
        return f"{v:+.3f}"
    def f1(v: float) -> str:
        return f"{v:.1f}"

    lines = [f"*{name}* — evaluation metrics" if name else "*Evaluation metrics*", ""]

    # Traffic-light emoji by Sharpe
    sharpe_e = "🟢" if m.sharpe > 1.5 else ("🟡" if m.sharpe > 0.7 else "🔴")

    lines += [
        "*Performance*",
        f"  Total return: `{pct(m.total_return)}`  |  Ann. return: `{pct(m.ann_return)}`  |  Volatility: `{pct(m.ann_volatility)}`",
        "",
        "*Risk-adjusted ratios*",
        f"  {sharpe_e} Sharpe: `{f2(m.sharpe)}`  |  Sortino: `{f2(m.sortino)}`  |  Calmar: `{f2(m.calmar)}`",
        f"  Omega: `{f1(m.omega)}`  |  Serenity: `{f2(m.serenity)}`  |  Pain ratio: `{f2(m.pain_ratio)}`",
        f"  Gain/Pain: `{f2(m.gain_to_pain)}`  |  Tail ratio: `{f2(m.tail_ratio)}`  |  Recovery factor: `{f2(m.recovery_factor)}`",
        "",
        "*Drawdown*",
        f"  Max DD: `{pct(m.max_drawdown)}`  |  Avg DD: `{pct(m.avg_drawdown)}`  |  Ulcer index: `{f2(m.ulcer_index)}`",
        f"  Max DD duration: `{m.max_dd_duration} bars`  |  Periods: `{m.n_drawdown_periods}`",
        "",
        "*Risk*",
        f"  VaR 95%: `{pct(m.var_95)}`  |  VaR 99%: `{pct(m.var_99)}`  |  CVaR 95%: `{pct(m.cvar_95)}`",
        f"  Skew: `{f2(m.skewness)}`  |  Kurtosis: `{f2(m.excess_kurtosis)}`",
    ]

    if m.n_trades > 0:
        lines += [
            "",
            "*Trades*",
            f"  n_trades: `{m.n_trades}`  |  Win rate: `{m.win_rate:.1%}`  |  Profit factor: `{f2(m.profit_factor)}`",
            f"  Payoff: `{f2(m.payoff_ratio)}`  |  EV/trade: `{f2(m.expected_value)}`",
            f"  Consec wins: `{m.max_consec_wins}`  |  Consec losses: `{m.max_consec_losses}`",
        ]

    if m.alpha != 0 or m.beta != 0:
        lines += [
            "",
            "*vs Benchmark*",
            f"  Alpha: `{pct(m.alpha)}`  |  Beta: `{f2(m.beta)}`  |  IR: `{f2(m.information_ratio)}`",
            f"  Up capture: `{m.up_capture:.1%}`  |  Down capture: `{m.down_capture:.1%}`",
            f"  Active return: `{pct(m.active_return)}`  |  Tracking error: `{pct(m.tracking_error)}`",
        ]

    if m.ic != 0:
        lines += [
            "",
            "*ML Signal Quality*",
            f"  IC: `{f2(m.ic)}`  |  Rank IC: `{f2(m.rank_ic)}`  |  ICIR: `{f2(m.icir)}`",
            f"  Factor turnover: `{f2(m.factor_turnover)}`",
        ]

    if m.hit_rate_monthly > 0:
        lines += [
            "",
            "*Monthly*",
            f"  Hit rate: `{m.hit_rate_monthly:.1%}`  |  Avg: `{pct(m.avg_monthly_return)}`",
            f"  Best: `{pct(m.best_month)}`  |  Worst: `{pct(m.worst_month)}`",
        ]

    return "\n".join(lines)
