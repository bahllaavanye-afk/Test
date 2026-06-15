"""
Execution learner — closes the slippage feedback loop.

The smart router historically picked an execution algorithm purely from order
size. But the slippage tracker has been recording realized cost (slippage_bps,
implementation shortfall) for every fill, tagged with the algo used. This module
turns that history into a per-symbol verdict: "for AAPL, limit_first has
averaged 3.1 bps over 40 fills vs market's 11.4 bps — prefer limit_first."

How it feeds back without making the router async:
  - A scheduled task calls `refresh_scorecard()` hourly. It joins SlippageRecord
    to Order (for the symbol), aggregates mean slippage per (symbol, algo), and
    caches the winning algo per symbol in a module-level dict.
  - The router calls the synchronous `get_best_algo(symbol)` and uses the
    learned algo when there's enough evidence, otherwise falls back to its
    size-based default. No hardcoded numbers — everything is derived from fills.

An algo only wins a symbol if it has at least `min_samples` fills AND beats the
runner-up's mean slippage by `min_edge_bps` — so we don't chase noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC

from app.utils.logging import logger

# symbol -> recommended algo, refreshed by the scheduled task. Read by the
# router synchronously. Empty until the first refresh produces evidence.
_BEST_ALGO_BY_SYMBOL: dict[str, str] = {}
# symbol -> {algo -> {"n", "avg_slippage_bps", "avg_is_bps"}} for transparency.
_SCORECARD: dict[str, dict] = {}

DEFAULT_MIN_SAMPLES = 15
DEFAULT_MIN_EDGE_BPS = 1.0
DEFAULT_LOOKBACK_DAYS = 30


@dataclass
class AlgoStats:
    algo: str
    n: int
    avg_slippage_bps: float
    avg_is_bps: float


def get_best_algo(symbol: str) -> str | None:
    """Synchronous lookup for the router. None = no learned preference yet."""
    return _BEST_ALGO_BY_SYMBOL.get(symbol)


def get_scorecard() -> dict[str, dict]:
    """Full per-symbol/per-algo scorecard (for the analytics endpoint)."""
    return dict(_SCORECARD)


def _decide_best(
    per_algo: dict[str, AlgoStats],
    min_samples: int,
    min_edge_bps: float,
) -> str | None:
    """Pick the lowest-slippage algo that has enough samples and a clear edge."""
    eligible = [s for s in per_algo.values() if s.n >= min_samples]
    if len(eligible) < 2:
        # Need a comparison to justify overriding the size-based default.
        return None
    eligible.sort(key=lambda s: s.avg_slippage_bps)
    best, runner_up = eligible[0], eligible[1]
    if runner_up.avg_slippage_bps - best.avg_slippage_bps >= min_edge_bps:
        return best.algo
    return None


async def compute_scorecard(
    db,
    days: int = DEFAULT_LOOKBACK_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_edge_bps: float = DEFAULT_MIN_EDGE_BPS,
) -> dict[str, dict]:
    """
    Build the per-symbol/per-algo slippage scorecard from realized fills.
    Pure read via the ORM (joins SlippageRecord → Order for the symbol).
    Returns the scorecard; does not mutate the module cache (refresh does that).
    """
    from datetime import datetime, timedelta

    from sqlalchemy import select

    from app.models.order import Order
    from app.models.slippage import SlippageRecord

    cutoff = datetime.now(UTC) - timedelta(days=days)
    rows = await db.execute(
        select(Order.symbol, SlippageRecord.execution_algo,
               SlippageRecord.slippage_bps, SlippageRecord.is_cost_bps)
        .join(Order, SlippageRecord.order_id == Order.id)
        .where(SlippageRecord.created_at >= cutoff)
    )

    # symbol -> algo -> [slippage list], [is list]
    agg: dict[str, dict[str, dict[str, list]]] = {}
    for symbol, algo, slip, is_cost in rows.all():
        if not symbol or not algo or slip is None:
            continue
        a = agg.setdefault(symbol, {}).setdefault(algo, {"slip": [], "is": []})
        a["slip"].append(float(slip))
        if is_cost is not None:
            a["is"].append(float(is_cost))

    scorecard: dict[str, dict] = {}
    for symbol, by_algo in agg.items():
        per_algo: dict[str, AlgoStats] = {}
        entry: dict[str, dict] = {}
        for algo, vals in by_algo.items():
            n = len(vals["slip"])
            avg_slip = sum(vals["slip"]) / n if n else 0.0
            avg_is = sum(vals["is"]) / len(vals["is"]) if vals["is"] else 0.0
            per_algo[algo] = AlgoStats(algo, n, avg_slip, avg_is)
            entry[algo] = {"n": n, "avg_slippage_bps": round(avg_slip, 2),
                           "avg_is_bps": round(avg_is, 2)}
        best = _decide_best(per_algo, min_samples, min_edge_bps)
        scorecard[symbol] = {"algos": entry, "best_algo": best}
    return scorecard


async def refresh_scorecard(db_session_factory=None, **kwargs) -> dict:
    """
    Recompute the scorecard and update the in-memory caches the router reads.
    Designed to be scheduled hourly. Returns a small summary.
    """
    if db_session_factory is None:
        from app.database import AsyncSessionLocal as db_session_factory

    try:
        async with db_session_factory() as db:
            scorecard = await compute_scorecard(db, **kwargs)
    except Exception as e:  # noqa: BLE001 — never let the loop die on a bad query
        logger.warning("execution_learner: scorecard refresh failed", error=str(e))
        return {"updated": 0, "error": str(e)}

    _SCORECARD.clear()
    _SCORECARD.update(scorecard)
    _BEST_ALGO_BY_SYMBOL.clear()
    learned = 0
    for symbol, data in scorecard.items():
        best = data.get("best_algo")
        if best:
            _BEST_ALGO_BY_SYMBOL[symbol] = best
            learned += 1

    logger.info("execution_learner: scorecard refreshed",
                symbols=len(scorecard), learned_preferences=learned)
    return {"symbols": len(scorecard), "learned": learned}
