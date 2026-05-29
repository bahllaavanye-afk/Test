"""
A/B traffic router for ML model serving.

Maintains an in-memory snapshot of release states refreshed lazily from DB.
Routes each inference request to champion or challenger based on traffic_pct.

Thread-safety: uses asyncio.Lock to prevent thundering-herd refreshes.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import NamedTuple

import structlog

logger = structlog.get_logger()


# ─── Route decision ───────────────────────────────────────────────────────────

class RouteDecision(NamedTuple):
    release_id: str
    model_name: str
    version: str
    artifact_path: str
    framework: str
    ab_group: str       # "champion" | "challenger" | "shadow"
    traffic_pct: float


# ─── Router ───────────────────────────────────────────────────────────────────

class ABRouter:
    """
    In-memory A/B router backed by the model_releases DB table.

    The snapshot is a dict mapping model_name → list of active release dicts.
    It is refreshed at most once per `refresh_interval_s` seconds using a
    lazy-refresh strategy so the hot inference path is never blocked by a DB
    query (unless the cache is completely cold).

    Usage::

        router = ABRouter(AsyncSessionLocal)
        await router.refresh()            # warm up on startup
        decision = await router.route("lstm_momentum")
        if decision:
            # load model from decision.artifact_path ...
    """

    def __init__(self, db_factory, refresh_interval_s: int = 60) -> None:
        self._db_factory = db_factory
        self._refresh_interval = refresh_interval_s
        # model_name → list of {id, model_name, version, artifact_path, framework, status, traffic_pct}
        self._snapshot: dict[str, list[dict]] = {}
        self._last_refresh: float = 0.0
        self._refresh_lock: asyncio.Lock = asyncio.Lock()

    # ── Snapshot management ────────────────────────────────────────────────────

    async def refresh(self) -> None:
        """Reload champion/challenger/shadow state from DB."""
        async with self._refresh_lock:
            # Double-checked locking: another coroutine may have refreshed while
            # we were waiting for the lock.
            if time.monotonic() - self._last_refresh < self._refresh_interval:
                return
            try:
                from sqlalchemy import select
                from app.models.model_release import ModelRelease

                async with self._db_factory() as db:
                    result = await db.execute(
                        select(
                            ModelRelease.id,
                            ModelRelease.model_name,
                            ModelRelease.version,
                            ModelRelease.artifact_path,
                            ModelRelease.framework,
                            ModelRelease.status,
                            ModelRelease.traffic_pct,
                        ).where(
                            ModelRelease.status.in_(["champion", "challenger", "shadow"])
                        )
                    )
                    rows = result.all()

                snapshot: dict[str, list[dict]] = {}
                for row in rows:
                    entry = {
                        "id": row.id,
                        "model_name": row.model_name,
                        "version": row.version,
                        "artifact_path": row.artifact_path,
                        "framework": row.framework,
                        "status": row.status,
                        "traffic_pct": float(row.traffic_pct or 0),
                    }
                    snapshot.setdefault(row.model_name, []).append(entry)

                self._snapshot = snapshot
                self._last_refresh = time.monotonic()
                logger.debug(
                    "ABRouter: snapshot refreshed",
                    n_models=len(snapshot),
                    total_releases=sum(len(v) for v in snapshot.values()),
                )
            except Exception as exc:
                logger.error("ABRouter: refresh failed", error=str(exc))

    async def _maybe_refresh(self) -> None:
        if time.monotonic() - self._last_refresh > self._refresh_interval:
            await self.refresh()

    # ── Traffic routing ────────────────────────────────────────────────────────

    async def route(self, model_name: str) -> RouteDecision | None:
        """
        Return a RouteDecision for *model_name*.

        Returns None if no champion exists for this model name.

        Traffic splitting:
        - If no challenger: champion receives 100 % of calls.
        - If challenger with traffic_pct=T: challenger receives T % of calls,
          champion receives (100-T) %.
        - Shadow releases: never routed; use :meth:`route_shadow` for logging.
        """
        await self._maybe_refresh()

        releases = self._snapshot.get(model_name, [])
        champion = next((r for r in releases if r["status"] == "champion"), None)
        challenger = next((r for r in releases if r["status"] == "challenger"), None)

        if champion is None:
            return None

        if challenger and random.random() * 100 < challenger["traffic_pct"]:
            chosen = challenger
        else:
            chosen = champion

        return RouteDecision(
            release_id=chosen["id"],
            model_name=chosen["model_name"],
            version=chosen["version"],
            artifact_path=chosen["artifact_path"],
            framework=chosen["framework"],
            ab_group=chosen["status"],
            traffic_pct=chosen["traffic_pct"],
        )

    def get_champion(self, model_name: str) -> dict | None:
        """Return the champion release dict for *model_name* from the snapshot."""
        return next(
            (r for r in self._snapshot.get(model_name, []) if r["status"] == "champion"),
            None,
        )

    def get_challenger(self, model_name: str) -> dict | None:
        """Return the challenger release dict for *model_name* from the snapshot, if any."""
        return next(
            (r for r in self._snapshot.get(model_name, []) if r["status"] == "challenger"),
            None,
        )

    def invalidate(self, model_name: str | None = None) -> None:
        """
        Force the next call to route() to re-read from DB.

        Call this after any promote/archive action to prevent stale routing.
        Pass model_name to invalidate just one model, or None for full invalidation.
        """
        if model_name:
            self._snapshot.pop(model_name, None)
        else:
            self._snapshot.clear()
        self._last_refresh = 0.0


# ── Module-level singleton ────────────────────────────────────────────────────
# Initialised in app startup (main.py lifespan or first use).
_router: ABRouter | None = None


def get_ab_router() -> ABRouter:
    global _router
    if _router is None:
        from app.database import AsyncSessionLocal
        _router = ABRouter(AsyncSessionLocal)
    return _router
