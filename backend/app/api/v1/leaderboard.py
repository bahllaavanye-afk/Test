"""Strategy leaderboard — aggregate backtest, paper, and live metrics per strategy."""
from datetime import datetime, timezone
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
        select(Account.id).where(
            Account.user_id == user_id,
            Account.is_active == True,  # noqa: E712
        )
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
        logger.debug(
            "walk_forward backtest lookup failed",
            strategy=strategy_name,
            error=str(exc),
        )

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


def _backtest_result_to_block(
    result: BacktestResult, run: BacktestRun | None = None
) -> MetricsBlock:
    last_updated = None
    if run and run.completed_at:
        last_updated = (
            run.completed_at.replace(tzinfo=timezone.utc)
            if run.completed_at.tzinfo is None
            else run.completed_at
        )

    total_trades = result.total_trades
    total_return = _float(result.total_return)
    avg_trade_pnl: float | None = None
    if total_trades and total_trades > 0 and total_return is not None:
        trades_log = result.trades_log
        if isinstance(trades_log, list) and len(trades_log) > 0:
            try:
                pnls = [
                    float(t.get("pnl", 0))
                    for t in trades_log
                    if isinstance(t, dict)
                ]
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
            func.sum(case((Trade.realized_pnl > 0, Trade.realized_pnl), else_=0)).label(
                "gross_profit"
            ),
            func.sum(case((Trade.realized_pnl < 0, Trade.realized_pnl), else_=0)).label(
                "gross_loss"
            ),
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
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else None

    return MetricsBlock(
        total_return=_float(row.total_pnl),
        total_trades=total_trades,
        avg_trade_pnl=_float(row.avg_pnl),
        win_rate=_float(win_rate),
        profit_factor=_float(profit_factor),
        last_updated=row.last_updated,
    )


# ─── Endpoint ─────────────────────────────────────────────────────────────────


@router.get("/", response_model=dict)
async def get_leaderboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the leaderboard with aggregated metrics for each strategy."""
    # Gather user accounts and their modes
    account_ids = await _user_account_ids(db, current_user.id)
    mode_map = await _account_mode_map(db, account_ids)

    paper_account_ids = [aid for aid, mode in mode_map.items() if mode == "paper"]
    live_account_ids = [aid for aid, mode in mode_map.items() if mode == "live"]

    # Fetch all strategies (could be filtered based on user permissions if needed)
    strategy_rows = await db.execute(select(Strategy))
    strategies: list[Strategy] = strategy_rows.scalars().all()

    entries: list[LeaderboardEntry] = []
    sharpe_sum = 0.0
    sharpe_count = 0
    best_sharpe = -float("inf")
    best_strategy_name: str | None = None
    total_paper_pnl = 0.0
    total_live_pnl = 0.0
    running_count = 0

    for strat in strategies:
        # Backtest best result
        backtest_result = await _best_backtest_result(db, strat.name, current_user.id)
        backtest_block = None
        if backtest_result:
            # Retrieve associated run to get completed_at timestamp
            run = await db.get(BacktestRun, backtest_result.run_id)
            backtest_block = _backtest_result_to_block(backtest_result, run)

            # Accumulate Sharpe for summary
            if backtest_block.sharpe_ratio is not None:
                sharpe_sum += backtest_block.sharpe_ratio
                sharpe_count += 1
                if backtest_block.sharpe_ratio > best_sharpe:
                    best_sharpe = backtest_block.sharpe_ratio
                    best_strategy_name = strat.name

        # Paper and Live trade metrics
        paper_block = await _aggregate_trade_metrics(db, strat.name, paper_account_ids)
        live_block = await _aggregate_trade_metrics(db, strat.name, live_account_ids)

        if paper_block and paper_block.total_return is not None:
            total_paper_pnl += paper_block.total_return
        if live_block and live_block.total_return is not None:
            total_live_pnl += live_block.total_return

        # Forward test (walk‑forward) result
        forward_result = await _best_forward_result(db, strat.name, current_user.id)
        forward_block = None
        if forward_result:
            forward_run = await db.get(BacktestRun, forward_result.run_id)
            forward_block = _backtest_result_to_block(forward_result, forward_run)

        # Determine if strategy is currently running (live trades exist)
        if live_block and (live_block.total_trades or 0) > 0:
            running_count += 1

        entry = LeaderboardEntry(
            id=str(strat.id),
            name=strat.name,
            display_name=strat.display_name,
            market_type=strat.market_type,
            strategy_type=strat.strategy_type,
            risk_bucket=strat.risk_bucket,
            is_enabled=strat.is_enabled,
            symbols=strat.symbols or [],
            backtest=backtest_block,
            paper=paper_block,
            live=live_block,
            forward_test=forward_block,
            vs_spy_sharpe=None,
            ml_improvement_pct=None,
            rank=0,  # will be filled after sorting
        )
        entries.append(entry)

    # Rank entries by backtest Sharpe (descending)
    entries.sort(
        key=lambda e: e.backtest.sharpe_ratio if e.backtest and e.backtest.sharpe_ratio is not None else -float("inf"),
        reverse=True,
    )
    for idx, entry in enumerate(entries, start=1):
        entry.rank = idx

    avg_sharpe = sharpe_sum / sharpe_count if sharpe_count > 0 else None

    summary = LeaderboardSummary(
        total_strategies=len(entries),
        running_count=running_count,
        avg_sharpe=avg_sharpe,
        best_strategy=best_strategy_name,
        total_paper_pnl=total_paper_pnl,
        total_live_pnl=total_live_pnl,
    )

    return {"entries": entries, "summary": summary}


# The `/entries` and `/summary` routes were dropped by an unvalidated change (everything
# 404'd). Restore them as thin views over get_leaderboard so the frontend + tests work.
@router.get("/entries", response_model=list[LeaderboardEntry])
async def list_leaderboard_entries(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[LeaderboardEntry]:
    """Per-strategy leaderboard entries, ranked by Sharpe."""
    data = await get_leaderboard(current_user=current_user, db=db)
    return data["entries"]


@router.get("/summary", response_model=LeaderboardSummary)
async def get_leaderboard_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LeaderboardSummary:
    """Aggregate leaderboard roll-up (totals, avg Sharpe, best strategy)."""
    data = await get_leaderboard(current_user=current_user, db=db)
    return data["summary"]