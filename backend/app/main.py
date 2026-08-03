"""FastAPI app factory with lifespan, CORS, routers, and background tasks."""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.api.limiter import limiter

from app.config import settings
from app.database import engine, Base
import app.models  # noqa: F401 — registers all ORM models with Base.metadata before create_all
from app.api.v1.router import api_router
from app.ws.prices import router as prices_router
from app.ws.orders import router as orders_router
from app.ws.alerts import router as alerts_router
from app.tasks.scheduler import start_scheduler
from app.utils.logging import configure_logging, logger
from app.risk.correlation_monitor import correlation_monitor

# configure_logging() had NO CALLER anywhere — one textual occurrence in the
# package, its own `def`. So structlog ran on its library defaults instead of
# this app's configuration, with two consequences in production:
#
#   * renderer was ConsoleRenderer, not JSONRenderer — Render received
#     unstructured text, so nothing downstream could parse a log line
#   * wrapper_class was BoundLoggerFilteringAtNotset, which filters NOTHING,
#     so all 105 logger.debug() call sites in app/ emitted on every run
#
# Called at import rather than inside lifespan: module-level code and anything
# imported before startup already logs, and static_server.py imports this
# module, so this is the one place that covers every entrypoint.
configure_logging()


async def _supervised(coro_factory, name: str, restart_delay: int = 30):
    """Restart a background coroutine if it crashes, with exponential backoff.
    Delay resets to restart_delay after each successful (non-crashing) run."""
    delay = restart_delay
    while True:
        try:
            await coro_factory()
            delay = restart_delay  # task exited cleanly — reset backoff
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Background task {name} crashed: {e}. Restarting in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 300)


async def _validate_alpaca(broker) -> None:
    from app.brokers.alpaca import validate_alpaca_connection
    ok = await validate_alpaca_connection(broker)
    if not ok:
        logger.warning("Alpaca broker is not connected — strategy runner will use yfinance fallback")


async def _risk_state_sync(risk_manager, broker, interval_seconds: int = 60) -> None:
    """Keep the risk manager fed with real equity and positions.

    Without this the manager runs forever on its seeded `initial_equity`: the
    drawdown circuit breaker only ever sees one data point, so it can never
    trip, and the position cap is computed against a fabricated NAV. A gate
    that is wired in but never given data is still not a gate.
    """
    while True:
        try:
            account = await broker.get_account()
            equity = account.get("equity")
            if equity is not None:
                equity = float(equity)
                if equity < 0:
                    # update_equity() rejects negatives, and a broker CAN report
                    # one (margin call). Letting that raise here would leave the
                    # manager on its seeded 100k forever — failing OPEN on the
                    # one condition that must halt trading. Clamp to 0 so
                    # check_order()'s `equity <= 0` halt actually fires.
                    logger.error(
                        "Broker reports negative equity — halting via zero-equity gate",
                        equity=equity,
                    )
                    equity = 0.0
                risk_manager.update_equity(equity)
        except Exception as exc:
            # Never let a broker hiccup kill the loop — the next tick retries.
            logger.warning("risk equity sync failed", error=str(exc))

        # Separate try: a positions failure must not also cost us the equity
        # update above, which is what feeds the drawdown breaker.
        try:
            risk_manager.update_positions(await broker.get_positions())
        except Exception as exc:
            logger.warning("risk positions sync failed", error=str(exc))

        await asyncio.sleep(interval_seconds)


async def _position_exit_monitor(broker, interval_seconds: int = 30) -> None:
    """Enforce stop-loss / take-profit / trailing exits on open positions.

    PositionMonitor was never started. `start_position_monitor()`'s docstring
    says "Factory function called from scheduler.py"; scheduler.py has no such
    job, and nothing anywhere constructs a PositionMonitor. Meanwhile the
    strategy runner faithfully writes `pos_exit:<symbol>` to Redis on every
    fill — stop_loss, take_profit, peak_price — under the comment "Store exit
    config in Redis for position_monitor.py". The producer ran; the consumer
    did not exist. Every strategy stop-loss was recorded and none was enforced,
    and the whole CompositeExit engine in execution/position_exit.py was
    reachable only from here.

    (Bot positions are separately covered by the `bot_exit_checker` scheduler
    job — this is the strategy-runner path, which had nothing.)
    """
    from app.database import AsyncSessionLocal
    from app.redis_client import get_redis
    from app.tasks.position_monitor import PositionMonitor

    monitor = PositionMonitor(broker, get_redis(), AsyncSessionLocal)
    while True:
        try:
            await monitor.start()
        except Exception as exc:
            logger.warning("position exit monitor pass failed", error=str(exc))
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("QuantEdge starting up", mode=settings.trading_mode)

    # Probe the primary DB first; if it's dead (Supabase pause), switch to the
    # SQLite fallback NOW so everything below (create_all, seeding, scheduler)
    # binds to a database that actually works instead of 500ing all session.
    import app.database as db_mod
    live_engine = await db_mod.ensure_database_alive()

    # Create tables (managed by Alembic in production; this covers dev/test and
    # is a no-op re-run when the fallback path already created the schema)
    for attempt in range(5):
        try:
            async with live_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            break
        except Exception as e:
            if attempt == 4:
                logger.error(f"DB not reachable after 5 attempts: {e}. Continuing without create_all.")
            else:
                wait_secs = 2 ** attempt
                logger.warning(f"DB connection attempt {attempt + 1} failed: {e}. Retrying in {wait_secs}s")
                await asyncio.sleep(wait_secs)

    # Seed demo content in-process (idempotent, DEMO_MODE-gated). start.sh seeds
    # pre-boot too, but that standalone step dies against a dead primary DB; seeding
    # here — after the fallback probe — guarantees the site never boots empty.
    try:
        from app.bots.seed import seed_all
        seeded = await seed_all()
        if any(seeded.values()):
            logger.info("Demo seed on boot", **seeded)
    except Exception as e:
        logger.warning(f"Demo seed skipped: {e}")

    # Start background scheduler
    scheduler = start_scheduler(db_session_factory=None)
    app.state.scheduler = scheduler

    # Start AlgoAgent (UCB1 exploration/exploitation)
    from app.tasks.algo_agent import AlgoAgent
    algo_agent = AlgoAgent(interval_seconds=300)
    app.state.algo_agent = algo_agent

    # Self-improvement autoloop
    from app.tasks.self_improver import SelfImprover
    from app.tasks.code_quality_loop import CodeQualityLoop
    self_improver = SelfImprover(algo_agent=algo_agent, interval_seconds=900)
    app.state.self_improver = self_improver

    code_quality_loop = CodeQualityLoop(interval_seconds=3600)
    app.state.code_quality_loop = code_quality_loop

    from app.tasks.qa_monitor import QAMonitor
    qa_monitor = QAMonitor(interval_seconds=300)
    app.state.qa_monitor = qa_monitor

    from app.tasks.research_scientist import ResearchScientist
    research_scientist = ResearchScientist(interval_seconds=3600)
    app.state.research_scientist = research_scientist

    from app.tasks.modeling_engineer import ModelingEngineer
    modeling_engineer = ModelingEngineer(interval_seconds=1800)
    app.state.modeling_engineer = modeling_engineer

    bg_tasks = []
    app.state.bg_tasks = bg_tasks

    bg_tasks.append(asyncio.create_task(_supervised(lambda: algo_agent.run(), "algo_agent")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: self_improver.run(), "self_improver")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: code_quality_loop.run(), "code_quality_loop")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: correlation_monitor.run_forever(), "correlation_monitor")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: qa_monitor.run(), "qa_monitor")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: research_scientist.run(), "research_scientist")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: modeling_engineer.run(), "modeling_engineer")))

    # ── Strategy runner + price feed ──────────────────────────────────────────
    # Build the Alpaca broker (returns None gracefully when API keys are absent)
    from app.brokers.alpaca import create_alpaca_broker
    alpaca_broker = create_alpaca_broker(paper=settings.is_paper)
    app.state.alpaca_broker = alpaca_broker

    if alpaca_broker is not None:
        asyncio.create_task(_validate_alpaca(alpaca_broker))

    # ── Risk manager ──────────────────────────────────────────────────────────
    # This was the only process that never built one. orders.py reads
    # `app.state.risk_manager` and skips the gate when it is absent, and nothing
    # ever assigned it — so every REST order reached the broker unchecked, and
    # the strategy runner was handed risk_manager=None outright. The sole
    # RiskManager() construction in the codebase lived in
    # strategy_runner.start_strategy_runner(), whose docstring claims main.py
    # registers it; main.py never called it. Pinned by
    # tests/unit/test_risk_gate_wiring.py.
    from app.risk.manager import RiskManager
    risk_manager = RiskManager()
    app.state.risk_manager = risk_manager
    if alpaca_broker is not None:
        # Registered in bg_tasks so shutdown cancels it with everything else.
        bg_tasks.append(asyncio.create_task(
            _supervised(lambda: _risk_state_sync(risk_manager, alpaca_broker), "risk_state_sync")
        ))
    else:
        logger.warning("Risk manager running on seeded equity — no broker to sync from")

    # Position exit monitor — the consumer for the exit configs the strategy
    # runner has been writing to Redis all along. See _position_exit_monitor.
    bg_tasks.append(asyncio.create_task(
        _supervised(lambda: _position_exit_monitor(alpaca_broker), "position_exit_monitor")
    ))

    # Load active strategies from DB; fall back to a sensible default set if DB
    # is not yet reachable at startup (e.g. first cold boot before migrations).
    active_strategies: list[dict] = []
    try:
        from app.database import AsyncSessionLocal
        from app.models.strategy import Strategy
        from sqlalchemy import select as _select
        async with AsyncSessionLocal() as _db:
            _result = await _db.execute(
                _select(Strategy).where(Strategy.is_enabled == True)  # noqa: E712
            )
            _rows = _result.scalars().all()
            active_strategies = [
                {
                    "name": s.name,
                    "symbols": s.symbols if isinstance(s.symbols, list) else [],
                    "params": {},
                    "tick_interval_seconds": int(getattr(s, "tick_interval_seconds", 3600)),
                    "confidence_threshold": float(getattr(s, "confidence_threshold", 0.6)),
                }
                for s in _rows
            ]
        logger.info("Loaded active strategies from DB", count=len(active_strategies))
    except Exception as _exc:
        logger.warning("Could not load strategies from DB at startup", error=str(_exc))

    # Default watchlist used when no strategies are enabled in DB yet.
    # Uses the shared constant rather than a second inline copy: the duplicate
    # that used to live here was equities-only, so a cold start ran no crypto
    # strategy at all even though btc_eth_stat_arb is in the shared default.
    if not active_strategies:
        from app.tasks.strategy_runner import DEFAULT_ACTIVE_STRATEGIES
        logger.info("No active DB strategies — using default paper watchlist")
        active_strategies = [dict(s) for s in DEFAULT_ACTIVE_STRATEGIES]

    app.state.active_strategies = active_strategies

    # Collect all unique symbols for the price feed
    all_symbols: list[str] = list({
        sym
        for s in active_strategies
        for sym in s.get("symbols", [])
    })

    # Price feed — polls broker quotes → Redis + WebSocket
    # Always started (incl. paper mode); gracefully skips ticks when broker is absent
    from app.tasks.price_feed import run_price_feed

    def _publish_mark(symbol: str, last: float) -> None:
        """Feed the risk manager a mark so market orders can be size-capped."""
        risk_manager.update_prices({symbol: last})

    async def _price_feed_wrapper():
        try:
            if alpaca_broker is not None and all_symbols:
                await run_price_feed(alpaca_broker, all_symbols, on_mark=_publish_mark)
            else:
                # Park the task until restart — avoids tight no-op loop
                logger.warning(
                    "Price feed idle",
                    reason="no broker" if alpaca_broker is None else "no symbols",
                )
                await asyncio.sleep(3600)
        except Exception as exc:
            logger.error(f"Price feed error: {exc}")
            raise

    bg_tasks.append(asyncio.create_task(
        _supervised(_price_feed_wrapper, "price_feed")
    ))
    logger.info("Price feed task registered", symbols=len(all_symbols))

    # Strategy runner — one asyncio loop per (strategy, symbol) pair
    # Always started so strategies run in paper mode too
    from app.tasks.strategy_runner import ContinuousStrategyRunner
    strategy_runner = ContinuousStrategyRunner(
        broker=alpaca_broker,
        risk_manager=risk_manager,
    )
    app.state.strategy_runner = strategy_runner
    bg_tasks.append(asyncio.create_task(
        _supervised(lambda: strategy_runner.start(active_strategies), "strategy_runner")
    ))
    logger.info("Strategy runner registered", strategies=len(active_strategies))

    # Backtest worker — polls for queued BacktestRun rows every 30 s, executes via yfinance
    from app.tasks.backtest_worker import backtest_worker_loop
    bg_tasks.append(asyncio.create_task(_supervised(backtest_worker_loop, "backtest_worker")))
    logger.info("Backtest worker registered")

    # Regime monitor — fits HMM every 5 min, writes 0/1/2 to Redis key 'market:regime'
    from app.tasks.regime_monitor import RegimeMonitor
    regime_monitor = RegimeMonitor()
    app.state.regime_monitor = regime_monitor
    bg_tasks.append(asyncio.create_task(
        _supervised(regime_monitor._loop, "regime_monitor")
    ))

    yield

    regime_monitor.stop()

    for task in getattr(app.state, "bg_tasks", []):
        task.cancel()
    await asyncio.gather(*getattr(app.state, "bg_tasks", []), return_exceptions=True)

    scheduler.shutdown(wait=False)
    await engine.dispose()
    logger.info("QuantEdge shutdown complete")


# The four exact top-level filenames app/ml/inference.py opens. Nothing else is
# reachable by it — not at any subdirectory depth.
INFERENCE_MODEL_FILES = (
    "lstm_latest.pt",
    "xgboost_latest.ubj",
    "lorentzian_latest.pkl",
    "scaler_latest.pkl",
)


def ml_models_check(models_dir: str | Path) -> dict:
    """Report trained artifacts and loadable models as two separate numbers.

    They are different questions, and the old check conflated them into one
    number that answered neither:

      artifacts_on_disk — what training actually produced. The previous version
        globbed only the top level, but `ci_lstm_trainer.py` writes to
        `ARTIFACTS_DIR/lstm_<symbol>_1d/model.pt` — a subdirectory — so every
        model it has ever trained was invisible here. Hence `rglob`.

      count / ok — what `app/ml/inference.py` can actually load, i.e. the four
        exact top-level names in INFERENCE_MODEL_FILES.

    `ok` keys off LOADABLE, deliberately. Switching this check to `rglob` alone
    (the literal fix IMPROVEMENTS.md called for) would have flipped
    `ok: false -> true` the moment the weekly trainer ran, reporting "models
    loaded" while inference still had nothing — the exact green-looking absence
    this repo keeps paying for. The gap is real and documented: the CI trainer
    and `app.ml.models.lstm` define two different networks under the same name,
    so even a correctly-named artifact would fail `load_state_dict()`.

    Reporting both numbers keeps the difference between them visible instead of
    letting either one imply the other.
    """
    models_path = Path(models_dir)
    artifacts: list[Path] = []
    if models_path.exists():
        for pattern in ("*.pt", "*.ubj", "*.pkl"):
            artifacts.extend(models_path.rglob(pattern))

    loadable = [n for n in INFERENCE_MODEL_FILES if (models_path / n).exists()]

    if loadable:
        note = f"{len(loadable)} model(s) loadable by inference: {', '.join(loadable)}"
    elif artifacts:
        note = (
            f"{len(artifacts)} trained artifact(s) on disk but NONE loadable — "
            f"inference.py reads only {', '.join(INFERENCE_MODEL_FILES)} from the "
            f"top level of {models_dir}. Promotion is unwired."
        )
    else:
        note = "Run experiments/run_experiment.py to train models"

    return {
        "ok": len(loadable) > 0,
        "count": len(loadable),
        "artifacts_on_disk": len(artifacts),
        "note": note,
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="QuantEdge API",
        description="Institutional-grade quantitative trading platform",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS — explicit allowlist only. Browsers reject `*` + credentials anyway,
    # so the fallback to `*` was both insecure and broken. In dev we permit
    # localhost; in any other mode the operator MUST set CORS_ORIGINS.
    if settings.cors_origins:
        allowed_origins = settings.cors_origins
    elif settings.trading_mode in ("dev", "test"):
        allowed_origins = [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ]
    else:
        logger.warning(
            "CORS_ORIGINS not configured in non-dev mode — refusing all cross-origin requests"
        )
        allowed_origins = []

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )

    # REST API
    app.include_router(api_router, prefix="/api/v1")

    # WebSocket
    app.include_router(prices_router)
    app.include_router(orders_router)
    app.include_router(alerts_router)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "platform": "QuantEdge",
            "version": "1.0.0",
            "mode": "paper",
            "live_trading": False,
            "agents": "active",
        }

    @app.get("/health/detailed")
    async def health_detailed():
        """Comprehensive system health — DB, Redis, scheduler, and background tasks."""
        import time
        import importlib.util
        from app.database import AsyncSessionLocal

        checks: dict[str, dict] = {}

        # Database
        try:
            t0 = time.perf_counter()
            async with AsyncSessionLocal() as session:
                await session.execute(__import__("sqlalchemy").text("SELECT 1"))
            checks["database"] = {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as e:
            err_str = str(e)[:200]
            hint = ""
            if "supabase" in settings.database_url.lower() or "pooler" in settings.database_url.lower():
                hint = (" | SUPABASE PROJECT MAY BE PAUSED — go to supabase.com/dashboard, "
                        "find your project and click Unpause (free tier pauses after 7d inactivity). "
                        "You have 90 days before data is lost.")
            checks["database"] = {"ok": False, "error": err_str + hint}

        # Primary DB, when the app booted onto the SQLite fallback: keep the outage
        # visible (status stays degraded, watchdogs page) even though the app works.
        import app.database as _db_mod
        if _db_mod.db_fallback_active:
            checks["database"]["fallback"] = "sqlite"
            checks["database_primary"] = {
                "ok": False,
                "error": (_db_mod.db_primary_error or "unreachable at boot")
                + " | Running on the local SQLite fallback: functional but EPHEMERAL "
                "(data resets on redeploy). Unpause the Supabase project to restore durable state.",
            }

        # Redis
        try:
            from app.redis_client import get_redis
            redis = get_redis()
            if redis is None:
                checks["redis"] = {"ok": True, "note": "disabled (REDIS_URL not set)"}
            else:
                t0 = time.perf_counter()
                await redis.ping()
                checks["redis"] = {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as e:
            checks["redis"] = {"ok": False, "error": str(e)[:120]}

        # Scheduler — include the LIVE job table (id + next run). 2026-07-22:
        # bots showed last_run_at=None for days while "scheduler: ok" lied by
        # omission; whether bot_* jobs even EXIST at runtime was invisible.
        sched = getattr(app.state, "scheduler", None)
        sched_check: dict = {"ok": sched is not None and sched.running if sched else False}
        try:
            if sched is not None:
                jobs = sched.get_jobs()
                bot_jobs = [j for j in jobs if str(j.id).startswith("bot_")]
                sched_check["jobs_total"] = len(jobs)
                sched_check["bot_jobs"] = len(bot_jobs)
                sched_check["sample"] = [
                    {"id": str(j.id)[:40],
                     "next_run": j.next_run_time.isoformat() if j.next_run_time else None}
                    for j in (bot_jobs[:3] + [j for j in jobs if not str(j.id).startswith("bot_")][:5])
                ]
        except Exception as e:  # noqa: BLE001
            sched_check["jobs_error"] = str(e)[:120]
        checks["scheduler"] = sched_check

        # AlgoAgent
        agent = getattr(app.state, "algo_agent", None)
        checks["algo_agent"] = {"ok": agent is not None}

        # Background tasks (count running)
        bg_tasks = getattr(app.state, "bg_tasks", [])
        running_tasks = sum(1 for t in bg_tasks if not t.done())
        checks["background_tasks"] = {"ok": running_tasks > 0, "running": running_tasks, "total": len(bg_tasks)}

        # ML availability
        torch_available = importlib.util.find_spec("torch") is not None
        checks["torch"] = {"ok": True, "available": torch_available, "note": "optional — ML strategies degrade gracefully if absent"}

        # Strategy registry
        try:
            from app.strategies import STRATEGY_REGISTRY
            checks["strategies"] = {"ok": True, "count": len(STRATEGY_REGISTRY)}
        except Exception as e:
            checks["strategies"] = {"ok": False, "error": str(e)[:120]}

        # Broker status
        alpaca = getattr(app.state, "alpaca_broker", None)
        checks["alpaca"] = {
            "ok": alpaca is not None,
            "note": "Set ALPACA_API_KEY + ALPACA_SECRET_KEY to enable live/paper trading" if alpaca is None else "connected",
        }

        checks["ml_models"] = ml_models_check(settings.models_dir)

        # Non-critical checks don't make status degraded
        non_critical = {"redis", "torch", "alpaca", "ml_models"}
        critical_checks = {k: v for k, v in checks.items() if k not in non_critical}
        all_ok = all(v.get("ok", False) for v in critical_checks.values())
        return {
            "status": "ok" if all_ok else "degraded",
            "version": "2.0.0",
            "mode": settings.trading_mode,
            "checks": checks,
        }

    # Security headers on every response
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self' wss:; "
            "frame-ancestors 'none';"
        )
        # Only set HSTS in production (not dev/test where HTTP is used)
        if settings.trading_mode not in ("dev", "paper"):
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

    return app


app = create_app()
