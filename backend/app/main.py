"""FastAPI app factory with lifespan, CORS, routers, and background tasks."""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import engine, Base
import app.models  # noqa: F401 — registers all ORM models with Base.metadata before create_all
from app.api.v1.router import api_router
from app.ws.prices import router as prices_router
from app.ws.orders import router as orders_router
from app.ws.alerts import router as alerts_router
from app.tasks.scheduler import start_scheduler
from app.utils.logging import logger
from app.risk.correlation_monitor import correlation_monitor


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


async def _slack_startup_catchup() -> None:
    """
    On startup: wait 30 s for the app to settle, then post CTO reviews for
    the last 3 unreviewed messages per joined Slack channel.
    Keeps the CTO agent live even without an explicit /slack/review-history call.
    """
    await asyncio.sleep(30)
    from app.config import settings
    from app.api.v1.notifications import _cto_review_message
    import httpx

    token = getattr(settings, "slack_bot_token", "") or ""
    if not token:
        return

    try:
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Get channels the bot is in
            resp = await client.get(
                "https://slack.com/api/conversations.list",
                headers=headers,
                params={"types": "public_channel,private_channel", "limit": 100},
            )
            data = resp.json()
            channels = [c["id"] for c in data.get("channels", []) if c.get("is_member")]

            for ch_id in channels[:10]:  # cap at 10 channels
                resp = await client.get(
                    "https://slack.com/api/conversations.history",
                    headers=headers,
                    params={"channel": ch_id, "limit": 3},
                )
                hist = resp.json()
                for msg in hist.get("messages", []):
                    if msg.get("bot_id") or msg.get("subtype"):
                        continue
                    text = msg.get("text", "")
                    if not text or text.startswith("🤖") or len(text) < 5:
                        continue
                    ts = msg.get("ts", "")
                    await _cto_review_message(
                        channel_id=ch_id,
                        user_id=msg.get("user", ""),
                        text=text,
                        thread_ts=ts,
                        is_reply=False,
                    )
                    await asyncio.sleep(1)  # avoid rate limiting

    except Exception as e:
        logger.debug("Slack startup catch-up failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("QuantEdge starting up", mode=settings.trading_mode)

    # Create tables (managed by Alembic in production; this covers dev/test)
    for attempt in range(5):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            break
        except Exception as e:
            if attempt == 4:
                logger.error(f"DB not reachable after 5 attempts: {e}. Continuing without create_all.")
            else:
                wait_secs = 2 ** attempt
                logger.warning(f"DB connection attempt {attempt + 1} failed: {e}. Retrying in {wait_secs}s")
                await asyncio.sleep(wait_secs)

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

    # Startup Slack catch-up: review recent messages in all joined channels
    # Runs once, 30s after startup, so it doesn't block the hot path
    if getattr(settings, "slack_bot_token", ""):
        asyncio.create_task(_slack_startup_catchup())

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

    # Default watchlist used when no strategies are enabled in DB yet
    if not active_strategies:
        logger.info("No active DB strategies — using default paper watchlist")
        active_strategies = [
            {"name": "momentum",       "symbols": ["SPY", "QQQ", "AAPL", "TSLA"], "params": {}, "tick_interval_seconds": 3600, "confidence_threshold": 0.6},
            {"name": "mean_reversion", "symbols": ["SPY", "QQQ"],                  "params": {}, "tick_interval_seconds": 3600, "confidence_threshold": 0.6},
            {"name": "rsi_macd",       "symbols": ["SPY", "AAPL"],                 "params": {}, "tick_interval_seconds": 3600, "confidence_threshold": 0.6},
        ]

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

    async def _price_feed_wrapper():
        try:
            if alpaca_broker is not None and all_symbols:
                await run_price_feed(alpaca_broker, all_symbols)
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
        risk_manager=None,
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


def create_app() -> FastAPI:
    app = FastAPI(
        title="QuantEdge API",
        description="Institutional-grade quantitative trading platform",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

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
        return {"status": "ok", "mode": settings.trading_mode}

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
            checks["database"] = {"ok": False, "error": str(e)[:120]}

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

        # Scheduler
        sched = getattr(app.state, "scheduler", None)
        checks["scheduler"] = {"ok": sched is not None and sched.running if sched else False}

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

        # ML models on disk
        from pathlib import Path as _Path
        _models_dir = _Path(settings.models_dir)
        _model_files = (list(_models_dir.glob("*.pt")) + list(_models_dir.glob("*.ubj")) + list(_models_dir.glob("*.pkl"))) if _models_dir.exists() else []
        checks["ml_models"] = {
            "ok": len(_model_files) > 0,
            "count": len(_model_files),
            "note": "Run experiments/run_experiment.py to train models" if not _model_files else "models loaded",
        }

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
