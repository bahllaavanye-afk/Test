"""
A/B traffic router for ML model serving.

Maintains an in‑memory snapshot of release states refreshed lazily from DB.
Routes each inference request to champion or challenger based on traffic_pct,
with optional health‑check confirmation filters.

Thread‑safety: uses asyncio.Lock to prevent thundering‑herd refreshes.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Callable, NamedTuple, Optional

import structlog

logger = structlog.get_logger()


# ─── Route decision ───────────────────────────────────────────────────────────


class RouteDecision(NamedTuple):
    release_id: str
    model_name: str
    version: str
    artifact_path: str
    framework: str
    ab_group: str  # "champion" | "challenger" | "shadow"
    traffic_pct: float


# ─── Router ───────────────────────────────────────────────────────────────────


class ABRouter:
    """
    In‑memory A/B router backed by the model_releases DB table.

    The snapshot is a dict mapping model_name → list of active release dicts.
    It is refreshed at most once per `refresh_interval_s` seconds using a
    lazy‑refresh strategy so the hot inference path is never blocked by a DB
    query (unless the cache is completely cold).

    An optional ``health_check`` callable can be supplied to filter releases
    based on runtime health metrics (e.g., latency, error rate).  The callable
    receives a release dict and must return ``True`` if the release is fit for
    routing.

    Usage::

        router = ABRouter(AsyncSessionLocal, health_check=my_check)
        await router.refresh()            # warm up on startup
        decision = await router.route("lstm_momentum")
        if decision:
            # load model from decision.artifact_path ...
    """

    def __init__(
        self,
        db_factory,
        refresh_interval_s: int = 60,
        health_check: Optional[Callable[[dict], bool]] = None,
    ) -> None:
        self._db_factory = db_factory
        self._refresh_interval = refresh_interval_s
        self._snapshot: dict[str, list[dict]] = {}
        self._last_refresh: float = 0.0
        self._refresh_lock: asyncio.Lock = asyncio.Lock()
        # Default health check always passes.
        self._health_check: Callable[[dict], bool] = health_check or (lambda _: True)

    # ── Snapshot management ────────────────────────────────────────────────────

    async def refresh(self) -> None:
        """Reload champion/challenger/shadow state from DB."""
        async with self._refresh_lock:
            # Double‑checked locking: another coroutine may have refreshed while
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
                    try:
                        traffic = float(row.traffic_pct or 0)
                        if not (0.0 <= traffic <= 100.0):
                            raise ValueError("traffic_pct out of bounds")
                    except Exception as exc:
                        logger.warning(
                            "ABRouter: dropping release with invalid traffic_pct",
                            release_id=row.id,
                            error=str(exc),
                        )
                        continue

                    entry = {
                        "id": row.id,
                        "model_name": row.model_name,
                        "version": row.version,
                        "artifact_path": row.artifact_path,
                        "framework": row.framework,
                        "status": row.status,
                        "traffic_pct": traffic,
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

    # ── Health‑check handling ──────────────────────────────────────────────────

    def set_health_check(self, fn: Callable[[dict], bool]) -> None:
        """
        Replace the health‑check callable used to validate releases before routing.

        The callable receives a release dict and must return ``True`` if the
        release is healthy enough to be considered for traffic.
        """
        self._health_check = fn
        logger.info("ABRouter: health‑check function updated")

    # ── Traffic routing ────────────────────────────────────────────────────────

    async def route(self, model_name: str) -> RouteDecision | None:
        """
        Return a :class:`RouteDecision` for *model_name*.

        Returns ``None`` if no valid champion exists for this model name.

        Traffic splitting:
        - If no challenger: champion receives 100 % of calls.
        - If challenger with traffic_pct=T: challenger receives T % of calls,
          champion receives (100‑T) % of calls.
        - Shadow releases are never routed; use :meth:`route_shadow` for logging.
        """
        await self._maybe_refresh()

        releases = self._snapshot.get(model_name, [])
        champion = next((r for r in releases if r["status"] == "champion"), None)
        challenger = next((r for r in releases if r["status"] == "challenger"), None)

        # Validate health of champion first; if unhealthy, fall back to challenger.
        if champion and not self._health_check(champion):
            logger.warning(
                "ABRouter: champion failed health check, attempting fallback",
                model_name=model_name,
                release_id=champion["id"],
            )
            champion = None

        # Validate health of challenger as well.
        if challenger and not self._health_check(challenger):
            logger.warning(
                "ABRouter: challenger failed health check, will be ignored",
                model_name=model_name,
                release_id=challenger["id"],
            )
            challenger = None

        if champion is None:
            # No viable champion – cannot route.
            logger.error(
                "ABRouter: no healthy champion available for routing",
                model_name=model_name,
            )
            return None

        # Decide between challenger and champion when challenger exists.
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

    async def route_shadow(self, model_name: str) -> RouteDecision | None:
        """
        Return a :class:`RouteDecision` for a shadow release, if any exist.

        Shadow releases are intended for passive monitoring; they receive no
        traffic in the normal routing path.  This method can be used to log
        shadow inference calls without affecting traffic splits.
        """
        await self._maybe_refresh()

        shadow = next(
            (r for r in self._snapshot.get(model_name, []) if r["status"] == "shadow"), None
        )
        if shadow is None:
            return None

        logger.debug(
            "ABRouter: routing to shadow release",
            model_name=model_name,
            release_id=shadow["id"],
        )
        return RouteDecision(
            release_id=shadow["id"],
            model_name=shadow["model_name"],
            version=shadow["version"],
            artifact_path=shadow["artifact_path"],
            framework=shadow["framework"],
            ab_group="shadow",
            traffic_pct=shadow["traffic_pct"],
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
        Force the next call to :meth:`route` to re‑read from DB.

        Call this after any promote/archive action to prevent stale routing.
        Pass *model_name* to invalidate just one model, or ``None`` for full
        invalidation.
        """
        if model_name:
            self._snapshot.pop(model_name, None)
        else:
            self._snapshot.clear()
        self._last_refresh = 0.0
        logger.info("ABRouter: cache invalidated", model_name=model_name)


# ── Module‑level singleton ────────────────────────────────────────────────────
# Initialised in app startup (main.py lifespan or first use).

_router: ABRouter | None = None


def get_ab_router() -> ABRouter:
    global _router
    if _router is None:
        from app.database import AsyncSessionLocal

        _router = ABRouter(AsyncSessionLocal)
    return _router