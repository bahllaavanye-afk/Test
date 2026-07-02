"""Strategy leaderboard — aggregate backtest, paper, and live metrics per strategy."""
from datetime import datetime, timezone
from typing import Any, List

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
        win_rate=win_rate,
        profit_factor=profit_factor,
        last_updated=row.last_updated,
    )


# ─── Endpoints ─────────────────────────────────────────────────────────────


@router.get("/", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[LeaderboardEntry]:
    """Return a list of strategies with aggregated metrics."""
    account_ids = await _user_account_ids(db, current_user.id)
    account_mode = await _account_mode_map(db, account_ids)

    # Retrieve all enabled strategies
    strategies_res = await db.execute(select(Strategy).where(Strategy.is_enabled == True))  # noqa: E712
    strategies = strategies_res.scalars().all()

    entries: List[LeaderboardEntry] = []

    # Separate account ids by mode for later aggregation
    paper_ids = [aid for aid, mode in account_mode.items() if mode == "paper"]
    live_ids = [aid for aid, mode in account_mode.items() if mode == "live"]

    for strat in strategies:
        backtest_res = await _best_backtest_result(db, strat.name, current_user.id)
        backtest_block = None
        if backtest_res:
            run = await db.get(BacktestRun, backtest_res.run_id)
            backtest_block = _backtest_result_to_block(backtest_res, run)

        forward_res = await _best_forward_result(db, strat.name, current_user.id)
        forward_block = None
        if forward_res:
            run = await db.get(BacktestRun, forward_res.run_id)
            forward_block = _backtest_result_to_block(forward_res, run)

        paper_block = await _aggregate_trade_metrics(db, strat.name, paper_ids)
        live_block = await _aggregate_trade_metrics(db, strat.name, live_ids)

        entry = LeaderboardEntry(
            id=str(strat.id),
            name=strat.name,
            display_name=strat.display_name,
            market_type=strat.market_type,
            strategy_type=strat.strategy_type,
            risk_bucket=strat.risk_bucket,
            is_enabled=strat.is_enabled,
            symbols=strat.symbols,
            backtest=backtest_block,
            forward_test=forward_block,
            paper=paper_block,
            live=live_block,
            vs_spy_sharpe=None,
            ml_improvement_pct=None,
            rank=0,
        )
        entries.append(entry)

    # Rank by backtest Sharpe ratio (higher is better)
    entries.sort(
        key=lambda e: (
            e.backtest.sharpe_ratio
            if e.backtest and e.backtest.sharpe_ratio is not None
            else -float("inf")
        ),
        reverse=True,
    )
    for idx, entry in enumerate(entries, start=1):
        entry.rank = idx

    return entries


@router.get("/summary", response_model=LeaderboardSummary)
async def get_leaderboard_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LeaderboardSummary:
    """Return aggregated summary statistics for the leaderboard."""
    leaderboard = await get_leaderboard(current_user, db)

    total_strategies = len(leaderboard)
    running_count = sum(1 for e in leaderboard if e.live is not None)

    sharpe_vals = [
        e.backtest.sharpe_ratio
        for e in leaderboard
        if e.backtest and e.backtest.sharpe_ratio is not None
    ]
    avg_sharpe = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else None

    best_strategy = None
    if sharpe_vals:
        best_entry = max(
            leaderboard,
            key=lambda e: e.backtest.sharpe_ratio
            if e.backtest and e.backtest.sharpe_ratio is not None
            else -float("inf"),
        )
        best_strategy = best_entry.name

    total_paper_pnl