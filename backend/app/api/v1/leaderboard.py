"""Strategy leaderboard — aggregate backtest, paper, and live metrics per strategy."""
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.account import Account
from app.models.backtest import BacktestResult, BacktestRun
from app.models.strategy import Strategy
from app.models.trade import Trade
from app.models.user import User
from app.utils.logging import logger

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


# ─── Response Models ──────────────────────────────────────────────────────────


class MetricsBlock(BaseModel):
    total_return: float | None = None
    annualized_return: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    total_trades: int | None = None
    avg_trade_pnl: float | None = None
    last_updated: datetime | None = None


class LeaderboardEntry(BaseModel):
    id: str
    name: str
    display_name: str | None = None
    market_type: str
    strategy_type: str
    risk_bucket: str
    is_enabled: bool
    symbols: list[str]
    backtest: MetricsBlock | None = None
    paper: MetricsBlock | None = None
    live: MetricsBlock | None = None
    forward_test: MetricsBlock | None = None
    vs_spy_sharpe: float | None = None
    ml_improvement_pct: float | None = None
    rank: int = 0


class LeaderboardSummary(BaseModel):
    total_strategies: int
    running_count: int
    avg_sharpe: float | None
    best_strategy: str | None
    total_paper_pnl: float
    total_live_pnl: float


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _float(val: Any) -> float | None:
    """Safely cast a potentially Decimal ORM value to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


async def _user_account_ids(db: AsyncSession, user_id: str) -> list[str]:
    result = await db.execute(
        select(Account.id).where(Account.user_id == user_id, Account.is_active == True)  # noqa: E712
    )
    return [row[0] for row in result.all()]


async def _account_mode_map(db: AsyncSession, account_ids: list[str]) -> dict[str, str]:
    """Return {account_id: mode} for all given account IDs."""
    if not account_ids:
        return {}
    result = await db.execute(
        select(Account.id, Account.mode).where(Account.id.in_(account_ids))
    )
    return {row.id: row.mode for row in result.all()}


async def _best_backtest_result(
    db: AsyncSession, strategy_name: str, user_id: str
) -> BacktestResult | None:
    """Return the completed backtest result with the highest Sharpe ratio."""
    q = (
        select(BacktestResult)
        .join(BacktestRun, BacktestResult.run_id == BacktestRun.id)
        .where(
            BacktestRun.strategy_name == strategy_name,
            BacktestRun.user_id == user_id,
            BacktestRun.status == "done",
        )
        .order_by(BacktestResult.sharpe_ratio.desc().nullslast())
        .limit(1)
    )
    res = await db.execute(q)
    return res.scalar_one_or_none()


async def _best_forward_result(
    db: AsyncSession, strategy_name: str, user_id: str
) -> BacktestResult | None:
    """Return the best completed walk-forward backtest result."""
    q = (
        select(BacktestResult)
        .join(BacktestRun, BacktestResult.run_id == BacktestRun.id)
        .where(
            BacktestRun.strategy_name == strategy_name,
            BacktestRun.user_id == user_id,
            BacktestRun.status == "done",
            # Walk-forward runs set params.walk_forward=true or interval contains wf marker
            BacktestRun.params["walk_forward"].as_boolean() == True,  # noqa: E712
        )
        .order_by(BacktestResult.sharpe_ratio.desc().nullslast())
        .limit(1)
    )
    try:
        res = await db.execute(q)
        row = res.scalar_one_or_none()
        if row is not None:
            return row
    except Exception as exc:
        logger.debug("walk_forward backtest lookup failed", strategy=strategy_name, error=str(exc))

    # Fallback: check interval field for walk_forward marker
    q2 = (
        select(BacktestResult)
        .join(BacktestRun, BacktestResult.run_id == BacktestRun.id)
        .where(
            BacktestRun.strategy_name == strategy_name,
            BacktestRun.user_id == user_id,
            BacktestRun.status == "done",
            BacktestRun.interval.contains("walk_forward"),
        )
        .order_by(BacktestResult.sharpe_ratio.desc().nullslast())
        .limit(1)
    )
    try:
        res2 = await db.execute(q2)
        return res2.scalar_one_or_none()
    except Exception:
        return None


def _backtest_result_to_block(result: BacktestResult, run: BacktestRun | None = None) -> MetricsBlock:
    last_updated = None
    if run and run.completed_at:
        last_updated = run.completed_at.replace(tzinfo=UTC) if run.completed_at.tzinfo is None else run.completed_at

    total_trades = result.total_trades
    total_return = _float(result.total_return)
    avg_trade_pnl: float | None = None
    if total_trades and total_trades > 0 and total_return is not None:
        # Approximate avg trade pnl from trades_log if available
        trades_log = result.trades_log
        if isinstance(trades_log, list) and len(trades_log) > 0:
            try:
                pnls = [float(t.get("pnl", 0)) for t in trades_log if isinstance(t, dict)]
                avg_trade_pnl = sum(pnls) / len(pnls) if pnls else None
            except Exception:
                avg_trade_pnl = None

    return MetricsBlock(
        total_return=total_return,
        annualized_return=_float(result.annualized_return),
        sharpe_ratio=_float(result.sharpe_ratio),
        sortino_ratio=_float(result.sortino_ratio),
        calmar_ratio=_float(result.calmar_ratio),
        max_drawdown=_float(result.max_drawdown),
        win_rate=_float(result.win_rate),
        profit_factor=_float(result.profit_factor),
        total_trades=total_trades,
        avg_trade_pnl=avg_trade_pnl,
        last_updated=last_updated,
    )


async def _aggregate_trade_metrics(
    db: AsyncSession,
    strategy_name: str,
    account_ids: list[str],
) -> MetricsBlock | None:
    """Aggregate trade-level metrics for a strategy across given accounts."""
    if not account_ids:
        return None

    result = await db.execute(
        select(
            func.count(Trade.id).label("total_trades"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
            func.sum(case((Trade.realized_pnl > 0, Trade.realized_pnl), else_=0)).label("gross_profit"),
            func.sum(case((Trade.realized_pnl < 0, Trade.realized_pnl), else_=0)).label("gross_loss"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
            func.max(Trade.closed_at).label("last_updated"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.strategy_name == strategy_name,
        )
    )
    row = result.one_or_none()
    if row is None or (row.total_trades or 0) == 0:
        return None

    total_trades = int(row.total_trades)
    wins = int(row.wins or 0)
    win_rate = wins / total_trades if total_trades > 0 else None
    gross_profit = float(row.gross_profit or 0)
    gross_loss = abs(float(row.gross_loss or 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    last_updated = row.last_updated
    if last_updated and last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=UTC)

    return MetricsBlock(
        total_return=float(row.total_pnl or 0),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        avg_trade_pnl=float(row.avg_pnl or 0),
        last_updated=last_updated,
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[LeaderboardEntry]:
    """
    Return all strategies with aggregated backtest, paper, and live metrics.
    Sorted by backtest Sharpe ratio descending.
    """
    # Load strategies for this user's accounts
    account_ids = await _user_account_ids(db, current_user.id)

    if account_ids:
        strat_q = select(Strategy).where(
            Strategy.account_id.in_(account_ids)
        )
    else:
        strat_q = select(Strategy)

    strat_result = await db.execute(strat_q)
    strategies: list[Strategy] = strat_result.scalars().all()

    if not strategies:
        return []

    # Separate account IDs by mode
    mode_map = await _account_mode_map(db, account_ids)
    paper_account_ids = [aid for aid, mode in mode_map.items() if mode == "paper"]
    live_account_ids = [aid for aid, mode in mode_map.items() if mode == "live"]

    entries: list[LeaderboardEntry] = []

    for strategy in strategies:
        # ── Backtest block ──────────────────────────────────────────────────
        best_bt = await _best_backtest_result(db, strategy.name, current_user.id)
        backtest_block: MetricsBlock | None = None
        if best_bt is not None:
            # Fetch associated run for timestamp
            run_result = await db.execute(
                select(BacktestRun).where(BacktestRun.id == best_bt.run_id)
            )
            run = run_result.scalar_one_or_none()
            backtest_block = _backtest_result_to_block(best_bt, run)

        # ── Forward-test block ──────────────────────────────────────────────
        best_ft = await _best_forward_result(db, strategy.name, current_user.id)
        forward_block: MetricsBlock | None = None
        if best_ft is not None and best_ft.run_id != (best_bt.run_id if best_bt else None):
            ft_run_result = await db.execute(
                select(BacktestRun).where(BacktestRun.id == best_ft.run_id)
            )
            ft_run = ft_run_result.scalar_one_or_none()
            forward_block = _backtest_result_to_block(best_ft, ft_run)

        # ── Paper trades block ──────────────────────────────────────────────
        paper_block = await _aggregate_trade_metrics(db, strategy.name, paper_account_ids)

        # ── Live trades block ───────────────────────────────────────────────
        live_block = await _aggregate_trade_metrics(db, strategy.name, live_account_ids)

        # ── ML improvement % (vs manual baseline if available) ─────────────
        ml_improvement_pct: float | None = None
        if strategy.strategy_type == "ml_enhanced" and backtest_block and backtest_block.sharpe_ratio is not None:
            # Compare against best non-ML run for same strategy name prefix (best-effort)
            # Return None if we can't find a clean baseline
            ml_improvement_pct = None

        symbols = strategy.symbols if isinstance(strategy.symbols, list) else []

        entry = LeaderboardEntry(
            id=strategy.id,
            name=strategy.name,
            display_name=strategy.display_name,
            market_type=strategy.market_type,
            strategy_type=strategy.strategy_type,
            risk_bucket=strategy.risk_bucket,
            is_enabled=strategy.is_enabled,
            symbols=symbols,
            backtest=backtest_block,
            paper=paper_block,
            live=live_block,
            forward_test=forward_block,
            vs_spy_sharpe=None,  # populated by comparison service if needed
            ml_improvement_pct=ml_improvement_pct,
            rank=0,
        )
        entries.append(entry)

    # Sort by backtest sharpe descending (None sorts last)
    entries.sort(
        key=lambda e: (e.backtest.sharpe_ratio is not None, e.backtest.sharpe_ratio or 0),
        reverse=True,
    )

    # Assign rank (1-indexed)
    for idx, entry in enumerate(entries):
        entry.rank = idx + 1

    return entries


@router.get("/summary", response_model=LeaderboardSummary)
async def get_leaderboard_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> LeaderboardSummary:
    """
    Return aggregate leaderboard statistics:
    total strategies, running count, average Sharpe, best strategy name,
    total paper P&L, total live P&L.
    """
    account_ids = await _user_account_ids(db, current_user.id)

    # Strategy counts
    if account_ids:
        total_q = await db.execute(
            select(func.count(Strategy.id)).where(Strategy.account_id.in_(account_ids))
        )
        running_q = await db.execute(
            select(func.count(Strategy.id)).where(
                Strategy.account_id.in_(account_ids),
                Strategy.is_enabled == True,  # noqa: E712
            )
        )
    else:
        total_q = await db.execute(select(func.count(Strategy.id)))
        running_q = await db.execute(
            select(func.count(Strategy.id)).where(Strategy.is_enabled == True)  # noqa: E712
        )

    total_strategies = int(total_q.scalar_one() or 0)
    running_count = int(running_q.scalar_one() or 0)

    # Average Sharpe from best completed backtests per strategy
    # Subquery: best sharpe per strategy_name
    sharpe_q = (
        select(func.avg(BacktestResult.sharpe_ratio))
        .join(BacktestRun, BacktestResult.run_id == BacktestRun.id)
        .where(
            BacktestRun.user_id == current_user.id,
            BacktestRun.status == "done",
        )
    )
    sharpe_result = await db.execute(sharpe_q)
    avg_sharpe_raw = sharpe_result.scalar_one_or_none()
    avg_sharpe = _float(avg_sharpe_raw)

    # Best strategy by sharpe
    best_q = (
        select(BacktestRun.strategy_name)
        .join(BacktestResult, BacktestResult.run_id == BacktestRun.id)
        .where(
            BacktestRun.user_id == current_user.id,
            BacktestRun.status == "done",
        )
        .order_by(BacktestResult.sharpe_ratio.desc().nullslast())
        .limit(1)
    )
    best_result = await db.execute(best_q)
    best_strategy_row = best_result.one_or_none()
    best_strategy = best_strategy_row[0] if best_strategy_row else None

    # P&L totals split by account mode
    mode_map = await _account_mode_map(db, account_ids)
    paper_ids = [aid for aid, mode in mode_map.items() if mode == "paper"]
    live_ids = [aid for aid, mode in mode_map.items() if mode == "live"]

    async def _sum_pnl(ids: list[str]) -> float:
        if not ids:
            return 0.0
        r = await db.execute(
            select(func.coalesce(func.sum(Trade.realized_pnl), 0.0)).where(
                Trade.account_id.in_(ids)
            )
        )
        return float(r.scalar_one() or 0.0)

    total_paper_pnl = await _sum_pnl(paper_ids)
    total_live_pnl = await _sum_pnl(live_ids)

    return LeaderboardSummary(
        total_strategies=total_strategies,
        running_count=running_count,
        avg_sharpe=avg_sharpe,
        best_strategy=best_strategy,
        total_paper_pnl=round(total_paper_pnl, 2),
        total_live_pnl=round(total_live_pnl, 2),
    )


@router.get("/entries", response_model=list[LeaderboardEntry])
async def get_leaderboard_entries(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[LeaderboardEntry]:
    """Alias for GET /leaderboard/ — returns all strategy leaderboard entries."""
    return await get_leaderboard(db=db, current_user=current_user)
