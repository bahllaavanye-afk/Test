"""
A/B traffic router for ML model serving.

Maintains an in‑memory snapshot of release states refreshed lazily from the
database. Routes each inference request to a champion or challenger based on
the configured traffic percentage.

Thread‑safety: uses ``asyncio.Lock`` to prevent thundering‑herd refreshes.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, NamedTuple, Optional

import structlog

logger = structlog.get_logger()


# ─── Route decision ───────────────────────────────────────────────────────────


class RouteDecision(NamedTuple):
    """
    Immutable description of the routing decision for a single inference request.

    Attributes
    ----------
    release_id: str
        Primary‑key identifier of the model release.
    model_name: str
        Logical name of the model (e.g. ``"lstm_momentum"``).
    version: str
        Version string of the release.
    artifact_path: str
        Filesystem or object‑store location of the model artefact.
    framework: str
        ML framework used to train the model (e.g. ``"pytorch"``).
    ab_group: str
        One of ``"champion"``, ``"challenger"``, or ``"shadow"``.
    traffic_pct: float
        Configured traffic percentage for the release (0‑100).
    """  # noqa: D401

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
    In‑memory A/B router backed by the ``model_releases`` DB table.

    The snapshot is a mapping ``model_name → list[dict]`` where each dict
    contains the fields ``id``, ``model_name``, ``version``, ``artifact_path``,
    ``framework``, ``status`` and ``traffic_pct``. The snapshot is refreshed at
    most once per ``refresh_interval_s`` seconds using a lazy‑refresh strategy
    so the hot inference path is never blocked by a DB query (unless the cache
    is completely cold).

    Parameters
    ----------
    db_factory: Callable[[], Any]
        Callable that returns an asynchronous database session (e.g.
        ``AsyncSessionLocal``). The callable is invoked each time a refresh
        needs to query the database.
    refresh_interval_s: int, optional
        Minimum number of seconds between successive refreshes. Defaults to 60.
    """

    def __init__(self, db_factory: Callable[[], Any], refresh_interval_s: int = 60) -> None:
        self._db_factory = db_factory
        self._refresh_interval = refresh_interval_s
        # model_name → list of {id, model_name, version, artifact_path, framework, status, traffic_pct}
        self._snapshot: dict[str, list[dict[str, Any]]] = {}
        self._last_refresh: float = 0.0
        self._refresh_lock: asyncio.Lock = asyncio.Lock()

    # ── Snapshot management ────────────────────────────────────────────────────

    async def refresh(self) -> None:
        """Reload champion/challenger/shadow state from the database."""
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

                snapshot: dict[str, list[dict[str, Any]]] = {}
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
        """Refresh the snapshot if the refresh interval has elapsed."""
        if time.monotonic() - self._last_refresh > self._refresh_interval:
            await self.refresh()

    # ── Traffic routing ────────────────────────────────────────────────────────

    async def route(self, model_name: str) -> Optional[RouteDecision]:
        """
        Return a :class:`RouteDecision` for *model_name*.

        The method may return ``None`` if no champion release exists for the
        requested model. Traffic splitting follows these rules:

        * If no challenger is present, the champion receives 100 % of calls.
        * If a challenger with ``traffic_pct = T`` exists, the challenger receives
          ``T`` % of calls and the champion receives ``100‑T`` %.
        * Shadow releases are never routed by this method; use
          :meth:`route_shadow` for logging or monitoring.

        Parameters
        ----------
        model_name: str
            Logical name of the model to route.

        Returns
        -------
        Optional[RouteDecision]
            Decision describing which release to use, or ``None`` when no
            champion is available.
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

    def get_champion(self, model_name: str) -> Optional[dict]:
        """
        Return the champion release dictionary for *model_name* from the snapshot.

        Parameters
        ----------
        model_name: str
            Logical name of the model.

        Returns
        -------
        Optional[dict]
            The champion release dict, or ``None`` if no champion exists.
        """
        return next(
            (r for r in self._snapshot.get(model_name, []) if r["status"] == "champion"),
            None,
        )

    def get_challenger(self, model_name: str) -> Optional[dict]:
        """
        Return the challenger release dictionary for *model_name* from the snapshot.

        Parameters
        ----------
        model_name: str
            Logical name of the model.

        Returns
        -------
        Optional[dict]
            The challenger release dict, or ``None`` if no challenger exists.
        """
        return next(
            (r for r in self._snapshot.get(model_name, []) if r["status"] == "challenger"),
            None,
        )

    def invalidate(self, model_name: str | None = None) -> None:
        """
        Force the next call to :meth:`route` to re‑read from the database.

        This should be called after any promote or archive action to avoid
        stale routing information. Providing a *model_name* limits the
        invalidation to that single model; ``None`` clears the entire snapshot.

        Parameters
        ----------
        model_name: str | None, optional
            Specific model to invalidate, or ``None`` to clear all cached data.
        """
        if model_name:
            self._snapshot.pop(model_name, None)
        else:
            self._snapshot.clear()
        self._last_refresh = 0.0


# ── Module‑level singleton ────────────────────────────────────────────────────
# Initialised in app startup (main.py lifespan or first use).

_router: ABRouter | None = None


def get_ab_router() -> ABRouter:
    """
    Retrieve the global :class:`ABRouter` instance.

    The router is instantiated lazily on first use with the application's
    asynchronous session factory. Subsequent calls return the same instance.

    Returns
    -------
    ABRouter
        The singleton router used throughout the service.
    """
    global _router
    if _router is None:
        from app.database import AsyncSessionLocal

        _router = ABRouter(AsyncSessionLocal)
    return _router