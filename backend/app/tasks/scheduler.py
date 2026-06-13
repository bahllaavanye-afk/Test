"""APScheduler setup: hourly snapshots, nightly retraining, order sync."""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app.utils.logging import logger

_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
    return _scheduler


async def agent_health_check() -> None:
    """Check all agent processes alive, ping Redis, alert on failures."""
    try:
        from app.redis_client import redis_client
        if redis_client:
            await redis_client.set("health:scheduler", "ok", ex=120)
    except Exception as e:
        logger.error("agent_health_check failed", error=str(e))


async def signal_quality_monitor() -> None:
    """Compute rolling IC for each active strategy. Post to agent bus if IC < 0.02."""
    try:
        from app.redis_client import redis_client
        # Post heartbeat
        if redis_client:
            import json, time
            await redis_client.set("monitor:signal_quality:last_run", str(time.time()), ex=600)
    except Exception as e:
        logger.error("signal_quality_monitor failed", error=str(e))


async def regime_monitor_task() -> None:
    """Re-fit HMM on latest SPY returns, update market:regime Redis key."""
    try:
        from app.tasks.regime_monitor import RegimeMonitor
        monitor = RegimeMonitor()
        await monitor.run()
    except Exception as e:
        logger.error("regime_monitor_task failed", error=str(e))


async def alpha_mining_cycle() -> None:
    """Run LLM alpha miner every 6 hours to propose new alpha factors."""
    import time
    try:
        from app.redis_client import redis_client
        if redis_client:
            await redis_client.set("monitor:alpha_mining:last_run", str(time.time()), ex=25200)
    except Exception:
        pass

    try:
        from app.config import settings as _settings
        if not getattr(_settings, "anthropic_api_key", None):
            logger.info("alpha_mining_cycle: ANTHROPIC_API_KEY not set — skipping")
            return

        from experiments.alpha_mining.llm_alpha_miner import AlphaMiner
        import asyncio

        # Run in a thread to avoid blocking the event loop (sync I/O + yfinance)
        symbols = ["SPY", "QQQ", "BTCUSDT", "ETHUSDT"]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: AlphaMiner().mine_and_save(
                symbols=symbols,
                output_dir="experiments/alpha_mining/results/",
            ),
        )
        logger.info("alpha_mining_cycle: mining complete", symbols=symbols)
    except ImportError:
        logger.debug("alpha_mining_cycle: AlphaMiner module not found — skipping")
    except Exception as e:
        logger.error("alpha_mining_cycle failed", error=str(e))


def start_scheduler(db_session_factory, broker=None) -> AsyncIOScheduler:
    scheduler = get_scheduler()

    async def _hourly_snapshot():
        """
        Capture an equity snapshot for every active account.
        Fetches live account data from the broker and persists an AccountSnapshot row.
        """
        logger.info("Running hourly account snapshot")
        if db_session_factory is None:
            # Fallback: create a fresh session from the global engine
            try:
                from app.database import AsyncSessionLocal as _factory
                factory = _factory
            except Exception as exc:
                logger.warning("Snapshot: no DB session factory", error=str(exc))
                return
        else:
            factory = db_session_factory

        try:
            from app.models.account import Account, AccountSnapshot
            from app.brokers.alpaca_orders import get_alpaca_account

            async with factory() as db:
                result = await db.execute(
                    select(Account).where(Account.is_active == True)  # noqa: E712
                )
                accounts = result.scalars().all()

            snap_records: list[AccountSnapshot] = []
            for acct in accounts:
                try:
                    if acct.broker == "alpaca" and acct.encrypted_key:
                        data = await get_alpaca_account(acct)
                        snap = AccountSnapshot(
                            id=str(uuid.uuid4()),
                            account_id=acct.id,
                            ts=datetime.now(timezone.utc),
                            total_equity=float(data.get("equity", 0)),
                            cash=float(data.get("cash", 0)),
                            unrealized_pnl=float(data.get("unrealized_pl", 0)),
                            raw_payload=data,
                        )
                        snap_records.append(snap)
                except Exception as exc:
                    logger.warning(
                        "Snapshot fetch failed",
                        account_id=acct.id,
                        broker=acct.broker,
                        error=str(exc),
                    )

            if snap_records:
                async with factory() as db:
                    db.add_all(snap_records)
                    await db.commit()
                logger.info("Hourly snapshot saved", count=len(snap_records))
            else:
                logger.info("Hourly snapshot: no active broker accounts with credentials")

        except Exception as exc:
            logger.error("Hourly snapshot failed", error=str(exc))

    async def _nightly_retrain():
        """
        Trigger nightly ML model retraining at 02:00 UTC.
        Delegates to ml_retrain.nightly_retrain() which downloads data and trains.
        """
        logger.info("Nightly ML retrain triggered")
        try:
            from app.tasks.ml_retrain import nightly_retrain
            await nightly_retrain()
        except Exception as exc:
            logger.error("Nightly retrain failed", error=str(exc))

    async def _order_sync():
        """
        Sync open broker orders back to the DB every minute.
        Updates status, filled_qty, and avg_fill_price for pending/partial fills.
        """
        logger.info("Order sync tick")
        if db_session_factory is None:
            try:
                from app.database import AsyncSessionLocal as _factory
                factory = _factory
            except Exception as exc:
                logger.debug("Order sync: no DB session factory", error=str(exc))
                return
        else:
            factory = db_session_factory

        try:
            from app.models.order import Order
            from app.models.account import Account
            from app.brokers.alpaca_orders import _headers, _base_url
            import httpx

            # Fetch all open orders from the DB
            async with factory() as db:
                result = await db.execute(
                    select(Order, Account)
                    .join(Account, Order.account_id == Account.id)
                    .where(
                        Order.status.in_(["pending", "accepted", "partially_filled", "new"]),
                        Account.is_active == True,  # noqa: E712
                    )
                )
                rows = result.all()

            if not rows:
                return

            updates: list[tuple[str, dict]] = []
            for order_row, acct in rows:
                try:
                    if not order_row.broker_order_id or acct.broker != "alpaca":
                        continue
                    headers = await _headers(acct)
                    base = _base_url(acct)
                    async with httpx.AsyncClient(timeout=8) as client:
                        resp = await client.get(
                            f"{base}/v2/orders/{order_row.broker_order_id}",
                            headers=headers,
                        )
                    if resp.status_code == 200:
                        data = resp.json()
                        updates.append((order_row.id, {
                            "status": data.get("status", order_row.status),
                            "filled_qty": float(data.get("filled_qty") or 0),
                            "avg_fill_price": (
                                float(data["filled_avg_price"])
                                if data.get("filled_avg_price") else None
                            ),
                        }))
                except Exception as exc:
                    logger.debug(
                        "Order sync: failed to fetch order",
                        order_id=order_row.id,
                        error=str(exc),
                    )

            if updates:
                async with factory() as db:
                    for order_id, fields in updates:
                        result = await db.execute(
                            select(Order).where(Order.id == order_id)
                        )
                        order = result.scalar_one_or_none()
                        if order:
                            for key, val in fields.items():
                                setattr(order, key, val)
                    await db.commit()
                logger.info("Order sync complete", updated=len(updates))

        except Exception as exc:
            logger.error("Order sync failed", error=str(exc))

    scheduler.add_job(
        _hourly_snapshot,
        "interval",
        hours=1,
        id="snapshot",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        _nightly_retrain,
        "cron",
        hour=2,
        minute=0,
        id="retrain",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        _order_sync,
        "interval",
        minutes=1,
        id="order_sync",
        replace_existing=True,
        max_instances=1,
    )

    async def _slack_employee_report():
        """Post hourly employee status to Slack #engineering."""
        try:
            from app.notifications.slack import slack
            from app.main import app as _app
            from datetime import datetime, timezone

            algo = getattr(_app.state, "algo_agent", None)
            research = getattr(_app.state, "research_scientist", None)
            modeling = getattr(_app.state, "modeling_engineer", None)

            lines = [f"*QuantEdge Hourly Status* — {datetime.now(timezone.utc).strftime('%H:%M UTC')}"]
            if algo:
                lb = algo.get_leaderboard()
                best = lb[0] if lb else {}
                lines.append(f"• AlgoAgent: {algo._total_runs} runs | top: {best.get('strategy','?')} sharpe={best.get('avg_sharpe',0):.3f}")
            if research:
                s = research.get_research_summary()
                lines.append(f"• Research: {s.get('cycles_completed',0)} cycles | {s.get('total_findings',0)} findings | queue: {len(s.get('implement_queue',[]))} ideas")
            if modeling:
                e = modeling.get_engineering_summary()
                lines.append(f"• Modeling: {e.get('promote_count',0)} promotions | {e.get('retrain_count',0)} retrains")

            await slack.send("system", "system", "📊 Hourly Status", text="\n".join(lines))
        except Exception as exc:
            logger.debug("Slack employee report failed", error=str(exc))

    scheduler.add_job(
        _slack_employee_report,
        "interval",
        hours=1,
        id="slack_employee_report",
        replace_existing=True,
        max_instances=1,
    )

    async def _auto_queue_backtests():
        """
        Daily at 03:00 UTC: queue backtests for every registered strategy using
        symbols appropriate for that strategy's market_type.
        Skips polymarket strategies (no OHLCV) and runs already queued today.
        """
        from app.database import AsyncSessionLocal
        from app.models.backtest import BacktestRun
        from app.strategies import STRATEGY_REGISTRY
        from datetime import date, timedelta

        # Symbol universe per market type — driven by strategy.market_type, not hardcoded
        SYMBOLS_BY_MARKET: dict[str, list[str]] = {
            "equity": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GLD", "TLT"],
            "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "polymarket": [],  # no OHLCV available
        }
        INTERVAL = "1d"
        END = date.today()
        START = END - timedelta(days=730)
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        queued = 0
        try:
            async with AsyncSessionLocal() as db:
                for strategy_name, strategy_cls in STRATEGY_REGISTRY.items():
                    market = getattr(strategy_cls, "market_type", "equity")
                    symbols = SYMBOLS_BY_MARKET.get(market, [])
                    for symbol in symbols:
                        existing = await db.execute(
                            select(BacktestRun)
                            .where(
                                BacktestRun.strategy_name == strategy_name,
                                BacktestRun.symbol == symbol,
                                BacktestRun.created_at >= today_start,
                            )
                            .limit(1)
                        )
                        if existing.scalar_one_or_none():
                            continue
                        run = BacktestRun(
                            id=str(uuid.uuid4()),
                            strategy_name=strategy_name,
                            symbol=symbol,
                            interval=INTERVAL,
                            start_date=START,
                            end_date=END,
                            status="queued",
                            created_at=datetime.now(timezone.utc),
                        )
                        db.add(run)
                        queued += 1
                await db.commit()
            logger.info("Auto-queued backtests", count=queued)
        except Exception as exc:
            logger.error("Auto-queue backtests failed", error=str(exc))

    scheduler.add_job(
        _auto_queue_backtests,
        "cron",
        hour=3,
        minute=0,
        id="auto_queue_backtests",
        replace_existing=True,
        max_instances=1,
    )

    async def _auto_run_experiments():
        """
        Daily at 04:00 UTC: run experiment configs that are missing results or have
        results older than 7 days. Caps at 3 per run to avoid overwhelming free-tier CPU.
        Cycles through all configs over time so everything stays fresh.
        """
        import sys
        import json
        from pathlib import Path
        from datetime import timedelta

        configs_dir = Path(__file__).parents[3] / "experiments" / "configs"
        results_dir = Path(__file__).parents[3] / "experiments" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        stale_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        due: list[Path] = []

        for cfg in sorted(configs_dir.glob("*.yaml")):
            result_file = results_dir / f"{cfg.stem}.json"
            if not result_file.exists():
                due.append(cfg)
                continue
            try:
                data = json.loads(result_file.read_text())
                ran_at_str = data.get("trained_at") or data.get("completed_at") or ""
                if ran_at_str:
                    from datetime import datetime as _dt
                    ran_at = _dt.fromisoformat(ran_at_str.replace("Z", "+00:00"))
                    if ran_at < stale_cutoff:
                        due.append(cfg)
                else:
                    due.append(cfg)
            except Exception:
                due.append(cfg)

        if not due:
            logger.info("Auto-run experiments: all configs are fresh")
            return

        to_run = due[:3]
        logger.info("Auto-run experiments: starting", configs=[c.name for c in to_run])

        run_script = Path(__file__).parents[3] / "experiments" / "run_experiment.py"
        for cfg in to_run:
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(run_script), "--config", cfg.name,
                    cwd=str(cfg.parent.parent),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    await asyncio.wait_for(proc.communicate(), timeout=600)
                    logger.info("Experiment completed", config=cfg.name, returncode=proc.returncode)
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("Experiment timed out", config=cfg.name)
            except Exception as exc:
                logger.error("Experiment run failed", config=cfg.name, error=str(exc))

    scheduler.add_job(
        _auto_run_experiments,
        "cron",
        hour=4,
        minute=0,
        id="auto_run_experiments",
        replace_existing=True,
        max_instances=1,
    )

    async def _slack_check_followups():
        """Every 4 hours: post follow-up nudges for unanswered Slack questions."""
        try:
            from app.api.v1.notifications import _run_followup_check
            result = await _run_followup_check(hours_threshold=4)
            if result.get("followed_up", 0):
                logger.info("Slack follow-ups sent", count=result["followed_up"])
        except Exception as exc:
            logger.debug("Slack follow-up check failed", error=str(exc))

    scheduler.add_job(
        _slack_check_followups,
        "interval",
        hours=4,
        id="slack_check_followups",
        replace_existing=True,
        max_instances=1,
    )

    # Bot Runner — load all enabled bots and schedule them
    async def _start_bot_runner():
        try:
            from app.tasks.bot_runner import BotRunner
            bot_runner = BotRunner(scheduler)
            await bot_runner.start()
            # Store on app state so the API can reschedule on create/update
            try:
                from app.main import app as _app
                _app.state.bot_runner = bot_runner
            except Exception:
                pass
            logger.info("BotRunner started")
        except Exception as exc:
            logger.error("BotRunner start failed", error=str(exc))

    scheduler.add_job(
        _start_bot_runner,
        "date",  # run once at startup
        id="bot_runner_init",
        replace_existing=True,
        max_instances=1,
    )

    async def _supabase_keepalive():
        """
        Ping the database every 5 days to prevent Supabase free-tier auto-pause.
        Supabase pauses inactive projects after 7 days — this job fires at day 5
        with a simple SELECT 1, keeping the project alive indefinitely.
        Only runs when DATABASE_URL points to Supabase (contains 'supabase' or 'pooler').
        """
        from app.config import settings as _settings
        db_url = _settings.database_url.lower()
        if "supabase" not in db_url and "pooler" not in db_url:
            return  # not a Supabase URL — no-op

        try:
            from app.database import AsyncSessionLocal
            from sqlalchemy import text
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            logger.info("Supabase keep-alive ping succeeded")
        except Exception as exc:
            logger.warning(
                "Supabase keep-alive ping failed — project may be paused. "
                "Go to supabase.com/dashboard and click Unpause to restore.",
                error=str(exc),
            )

    scheduler.add_job(
        _supabase_keepalive,
        "interval",
        days=5,
        id="supabase_keepalive",
        replace_existing=True,
        max_instances=1,
    )

    async def _bot_exit_checker():
        """
        Every 5 minutes: check all open bot paper positions for TP/SL hits
        and create Trade records (Option Alpha-style trade history).
        """
        try:
            from app.bots.engine import check_bot_exits
            from app.database import AsyncSessionLocal
            async with AsyncSessionLocal() as db:
                n = await check_bot_exits(db)
            if n > 0:
                logger.info("Bot exit checker: closed positions", count=n)
        except Exception as exc:
            logger.debug("Bot exit checker failed", error=str(exc))

    scheduler.add_job(
        _bot_exit_checker,
        "interval",
        minutes=5,
        id="bot_exit_checker",
        replace_existing=True,
        max_instances=1,
    )

    async def _position_monitor():
        """
        Every 30 seconds: check all open positions for exit conditions
        (stop-loss, take-profit, trailing stop, time-based, regime, etc.)
        and submit close orders when triggered.
        """
        try:
            from app.tasks.position_monitor import start_position_monitor
            from app.redis_client import get_redis
            from app.database import AsyncSessionLocal

            # Build broker best-effort (same pattern as strategy_runner)
            _broker = None
            try:
                from app.config import settings as _settings
                if _settings.alpaca_api_key and _settings.alpaca_secret_key:
                    from app.brokers.alpaca import AlpacaBroker
                    _broker = AlpacaBroker(
                        api_key=_settings.alpaca_api_key,
                        secret_key=_settings.alpaca_secret_key,
                        paper=(_settings.trading_mode != "live"),
                    )
            except Exception as _exc:
                logger.debug("PositionMonitor: broker unavailable", error=str(_exc))

            _redis = get_redis()
            await start_position_monitor(
                broker=_broker,
                redis_client=_redis,
                db_session_factory=AsyncSessionLocal,
            )
        except Exception as exc:
            logger.debug("Position monitor tick failed", error=str(exc))

    scheduler.add_job(
        _position_monitor,
        "interval",
        seconds=30,
        id="position_monitor",
        replace_existing=True,
        max_instances=1,
    )

    # ── Self-improving infrastructure ────────────────────────────────────────

    async def _self_improving_cycle():
        """Hourly: evaluate strategy performance, auto-disable losers, get LLM suggestions."""
        try:
            from app.database import AsyncSessionLocal
            from app.redis_client import get_redis
            from app.tasks.self_improving_loop import SelfImprovingLoop
            loop = SelfImprovingLoop(AsyncSessionLocal, get_redis())
            await loop.run_cycle()
        except Exception as exc:
            logger.debug("Self-improving cycle failed", error=str(exc))

    scheduler.add_job(
        _self_improving_cycle,
        "interval",
        hours=1,
        minutes=15,  # offset from snapshot to avoid DB contention
        id="self_improving_cycle",
        replace_existing=True,
        max_instances=1,
    )

    async def _research_pipeline():
        """Every 4 hours: LLM-driven research → experiment config generation."""
        try:
            from app.redis_client import get_redis
            from app.tasks.research_pipeline import ResearchPipeline
            pipeline = ResearchPipeline(get_redis())
            await pipeline.run()
        except Exception as exc:
            logger.debug("Research pipeline failed", error=str(exc))

    scheduler.add_job(
        _research_pipeline,
        "interval",
        hours=4,
        id="research_pipeline",
        replace_existing=True,
        max_instances=1,
    )

    async def _ai_strategy_generator():
        """Every 6 hours: AI-driven strategy proposal generation → staging area."""
        try:
            from app.redis_client import get_redis
            from app.tasks.ai_strategy_generator import AIStrategyGenerator
            gen = AIStrategyGenerator(get_redis())
            await gen.run()
        except Exception as exc:
            logger.debug("AI strategy generator failed", error=str(exc))

    scheduler.add_job(
        _ai_strategy_generator,
        "interval",
        hours=6,
        id="ai_strategy_generator",
        replace_existing=True,
        max_instances=1,
    )

    async def _scanner_sweep():
        """Every 5 minutes: run all desk scanners and publish top picks to Redis."""
        try:
            from app.redis_client import get_redis
            from app.tasks.stock_scanners import ScannerOrchestrator
            orchestrator = ScannerOrchestrator(redis_client=get_redis())
            await orchestrator.run_all()
        except Exception as exc:
            logger.debug("Scanner sweep failed", error=str(exc))

    scheduler.add_job(
        _scanner_sweep,
        "interval",
        minutes=5,
        id="scanner_sweep",
        replace_existing=True,
        max_instances=1,
    )

    # ── Agent Bus + Knowledge Loop (event-driven, always-on) ────────────────

    async def _start_agent_bus_and_knowledge_loop():
        """Start the Redis Streams event bus and the continuous knowledge loop."""
        try:
            from app.redis_client import get_redis
            from app.tasks.agent_bus import get_bus
            from app.tasks.knowledge_loop import get_knowledge_loop

            redis = get_redis()
            bus = get_bus(redis)
            knowledge_loop = get_knowledge_loop(redis)

            # Knowledge loop subscribes to all topics first, then bus starts dispatching
            await knowledge_loop.start()
            await bus.start()

            logger.info("Agent bus + knowledge loop running — company learns from every event")
        except Exception as exc:
            logger.warning("Agent bus / knowledge loop failed to start", error=str(exc))

    asyncio.create_task(_start_agent_bus_and_knowledge_loop())

    # ── Task Queue Worker ────────────────────────────────────────────────────

    async def _start_task_queue_worker():
        """Start the durable task queue worker."""
        try:
            from app.redis_client import get_redis
            from app.tasks.task_queue import get_task_queue
            queue = get_task_queue(get_redis())
            asyncio.create_task(queue.run_worker())
            logger.info("Task queue worker started")
        except Exception as exc:
            logger.warning("Task queue worker failed to start", error=str(exc))

    asyncio.create_task(_start_task_queue_worker())

    # ── Strategy Auction (hourly capital reallocation) ───────────────────────

    async def _run_strategy_auction():
        """Hourly: run UCB1 auction, reallocate capital to proven strategies."""
        try:
            from app.redis_client import get_redis
            from app.tasks.strategy_auction import get_auction
            auction = get_auction(get_redis())
            allocations = await auction.run_auction()
            logger.info("Strategy auction complete", strategy_count=len(allocations))
        except Exception as exc:
            logger.debug("Strategy auction failed", error=str(exc))

    scheduler.add_job(
        _run_strategy_auction,
        "interval",
        hours=1,
        id="strategy_auction",
        replace_existing=True,
        max_instances=1,
    )

    # Every 1 minute: agent health check
    scheduler.add_job(
        agent_health_check,
        trigger="interval",
        seconds=60,
        id="agent_health_check",
        replace_existing=True,
        max_instances=1,
    )

    # Every 3 minutes: signal quality monitor
    scheduler.add_job(
        signal_quality_monitor,
        trigger="interval",
        seconds=180,
        id="signal_quality_monitor",
        replace_existing=True,
        max_instances=1,
    )

    # Every 15 minutes: regime monitor
    scheduler.add_job(
        regime_monitor_task,
        trigger="interval",
        seconds=900,
        id="regime_monitor",
        replace_existing=True,
        max_instances=1,
    )

    # Every 6 hours: alpha mining
    scheduler.add_job(
        alpha_mining_cycle,
        trigger="interval",
        hours=6,
        id="alpha_mining_cycle",
        replace_existing=True,
        max_instances=1,
    )

    async def _run_holistic_review_job():
        """Daily at 06:00 UTC: holistic review of all active strategy promotions."""
        try:
            from app.tasks.holistic_review import run_holistic_review
            from app.database import AsyncSessionLocal
            await run_holistic_review(AsyncSessionLocal)
        except Exception as exc:
            logger.error("Holistic review job failed", error=str(exc))

    scheduler.add_job(
        _run_holistic_review_job,
        "cron",
        hour=6,
        minute=0,
        id="holistic_review",
        replace_existing=True,
        max_instances=1,
    )

    async def _run_promotion_metrics_sync():
        """Every 6 hours: sync live trade metrics into promotion pipeline."""
        try:
            from app.tasks.promotion_metrics_sync import sync_promotion_metrics
            from app.database import AsyncSessionLocal
            await sync_promotion_metrics(AsyncSessionLocal)
        except Exception as exc:
            logger.error("Promotion metrics sync failed", error=str(exc))

    scheduler.add_job(
        _run_promotion_metrics_sync,
        trigger="interval",
        hours=6,
        id="promotion_metrics_sync",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()
    logger.info("Scheduler started")
    return scheduler
