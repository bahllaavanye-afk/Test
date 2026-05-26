"""FastAPI app factory with lifespan, CORS, routers, and background tasks."""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import engine, Base
from app.api.v1.router import api_router
from app.ws.prices import router as prices_router
from app.ws.orders import router as orders_router
from app.ws.alerts import router as alerts_router
from app.tasks.scheduler import start_scheduler
from app.utils.logging import logger
from app.risk.correlation_monitor import correlation_monitor


async def _supervised(coro_factory, name: str, restart_delay: int = 30):
    """Restart a background coroutine if it crashes, with exponential backoff."""
    delay = restart_delay
    while True:
        try:
            await coro_factory()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Background task {name} crashed: {e}. Restarting in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("QuantEdge starting up", mode=settings.trading_mode)

    # Create tables (managed by Alembic in production; this covers dev/test)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Start background scheduler
    scheduler = start_scheduler(db_session_factory=None)
    app.state.scheduler = scheduler

    # Start AlgoAgent (UCB1 exploration/exploitation)
    from app.tasks.algo_agent import AlgoAgent
    import asyncio
    algo_agent = AlgoAgent(interval_seconds=300)
    app.state.algo_agent = algo_agent

    # Self-improvement autoloop
    from app.tasks.self_improver import SelfImprover
    from app.tasks.code_quality_loop import CodeQualityLoop
    self_improver = SelfImprover(algo_agent=algo_agent, interval_seconds=900)
    app.state.self_improver = self_improver

    code_quality_loop = CodeQualityLoop(interval_seconds=3600)
    app.state.code_quality_loop = code_quality_loop

    bg_tasks = []
    app.state.bg_tasks = bg_tasks

    bg_tasks.append(asyncio.create_task(_supervised(lambda: algo_agent.run(), "algo_agent")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: self_improver.run(), "self_improver")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: code_quality_loop.run(), "code_quality_loop")))
    bg_tasks.append(asyncio.create_task(_supervised(lambda: correlation_monitor.run_forever(), "correlation_monitor")))

    yield

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

    # CORS — restrict to Vercel frontend in production
    # cors_origins is a list[str] property on the settings object
    allowed_origins = settings.cors_origins if settings.cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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

    # Security headers on every response
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Only set HSTS in production (not dev/test where HTTP is used)
        if settings.trading_mode not in ("dev", "paper"):
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

    return app


app = create_app()
