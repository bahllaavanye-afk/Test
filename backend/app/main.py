"""FastAPI app factory with lifespan, CORS, routers, and background tasks."""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.api.v1.router import api_router
from app.ws.prices import router as prices_router
from app.ws.orders import router as orders_router
from app.ws.alerts import router as alerts_router
from app.tasks.scheduler import start_scheduler
from app.utils.logging import logger


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
    asyncio.create_task(algo_agent.run())

    # Self-improvement autoloop
    from app.tasks.self_improver import SelfImprover
    from app.tasks.code_quality_loop import CodeQualityLoop
    self_improver = SelfImprover(algo_agent=algo_agent, interval_seconds=900)
    app.state.self_improver = self_improver
    asyncio.create_task(self_improver.run())

    code_quality_loop = CodeQualityLoop(interval_seconds=3600)
    app.state.code_quality_loop = code_quality_loop
    asyncio.create_task(code_quality_loop.run())

    yield

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

    return app


app = create_app()
