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
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    last_updated = row.last_updated
    if last_updated and last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)

    return MetricsBlock(
        total_return=_float(row.total_pnl),
        annualized_return=None,
        sharpe_ratio=None,
        sortino_ratio=None,
        calmar_ratio=None,
        max_drawdown=None,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        avg_trade_pnl=_float(row.avg_pnl),
        last_updated=last_updated,
    )


# ─── Builders ────────────────────────────────────────────────────────────────


async def _build_entry(
    db: AsyncSession,
    strategy: Strategy,
    user_id: str,
    paper_ids: list[str],
    live_ids: list[str],
) -> LeaderboardEntry:
    """Assemble a single leaderboard entry for one strategy."""
    backtest_block: MetricsBlock | None = None
    bt = await _best_backtest_result(db, strategy.name, user_id)
    if bt is not None:
        backtest_block = _backtest_result_to_block(bt)

    forward_block: MetricsBlock | None = None
    fwd = await _best_forward_result(db, strategy.name, user_id)
    if fwd is not None:
        forward_block = _backtest_result_to_block(fwd)

    paper_block = await _aggregate_trade_metrics(db, strategy.name, paper_ids)
    live_block = await _aggregate_trade_metrics(db, strategy.name, live_ids)

    return LeaderboardEntry(
        id=strategy.id,
        name=strategy.name,
        display_name=strategy.display_name,
        market_type=strategy.market_type,
        strategy_type=strategy.strategy_type,
        risk_bucket=strategy.risk_bucket,
        is_enabled=strategy.is_enabled,
        symbols=strategy.symbols if isinstance(strategy.symbols, list) else [],
        backtest=backtest_block,
        paper=paper_block,
        live=live_block,
        forward_test=forward_block,
    )


def _entry_sort_key(entry: LeaderboardEntry) -> float:
    """Rank by best available Sharpe (live > paper > backtest); unranked last."""
    for block in (entry.live, entry.paper, entry.backtest):
        if block is not None and block.sharpe_ratio is not None:
            return block.sharpe_ratio
    return -float("inf")


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.get("/entries", response_model=list[LeaderboardEntry])
async def list_leaderboard_entries(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[LeaderboardEntry]:
    """Per-strategy leaderboard: backtest + forward + paper + live metrics, ranked by Sharpe."""
    account_ids = await _user_account_ids(db, user.id)
    mode_map = await _account_mode_map(db, account_ids)
    live_ids = [aid for aid, mode in mode_map.items() if mode == "live"]
    paper_ids = [aid for aid in account_ids if aid not in live_ids]

    result = await db.execute(select(Strategy))
    strategies = result.scalars().all()

    entries = [
        await _build_entry(db, s, user.id, paper_ids, live_ids) for s in strategies
    ]
    entries.sort(key=_entry_sort_key, reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry.rank = rank
    return entries


@router.get("/summary", response_model=LeaderboardSummary)
async def leaderboard_summary(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LeaderboardSummary:
    """Aggregate roll-up across all strategies for the leaderboard header."""
    account_ids = await _user_account_ids(db, user.id)
    mode_map = await _account_mode_map(db, account_ids)
    live_ids = [aid for aid, mode in mode_map.items() if mode == "live"]
    paper_ids = [aid for aid in account_ids if aid not in live_ids]

    result = await db.execute(select(Strategy))
    strategies = result.scalars().all()

    running_count = 0
    sharpes: list[float] = []
    best_strategy: str | None = None
    best_sharpe = -float("inf")
    total_paper_pnl = 0.0
    total_live_pnl = 0.0

    for s in strategies:
        if s.is_enabled:
            running_count += 1
        bt = await _best_backtest_result(db, s.name, user.id)
        if bt is not None:
            sr = _float(bt.sharpe_ratio)
            if sr is not None:
                sharpes.append(sr)
                if sr > best_sharpe:
                    best_sharpe = sr
                    best_strategy = s.name
        paper_block = await _aggregate_trade_metrics(db, s.name, paper_ids)
        if paper_block is not None and paper_block.total_return is not None:
            total_paper_pnl += paper_block.total_return
        live_block = await _aggregate_trade_metrics(db, s.name, live_ids)
        if live_block is not None and live_block.total_return is not None:
            total_live_pnl += live_block.total_return

    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else None

    return LeaderboardSummary(
        total_strategies=len(strategies),
        running_count=running_count,
        avg_sharpe=avg_sharpe,
        best_strategy=best_strategy,
        total_paper_pnl=total_paper_pnl,
        total_live_pnl=total_live_pnl,
    )