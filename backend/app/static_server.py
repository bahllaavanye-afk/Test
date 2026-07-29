"""Serve the built Vite SPA from the FastAPI backend — single-deployment hosting.

This wraps the existing app (`app.main:app`) so we do NOT modify main.py. Render
builds `frontend/dist` (see render.yaml) and `start.sh` runs
`uvicorn app.static_server:app`. The API routers are registered first on the
imported app, so they always match before the SPA catch‑all. If the build dir is
missing (e.g. backend‑only dev), the API still works and the catch‑all no‑ops.

Why this instead of Vercel: one deployment, one origin (no CORS), auto‑deploys
from main on Render — no separate frontend host or orphaned deployments.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.main import app  # the existing FastAPI app — unchanged

logger = logging.getLogger(__name__)

# frontend/dist relative to repo root (backend/app/static_server.py → ../../frontend/dist)
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
_INDEX = _DIST / "index.html"

if _INDEX.is_file():
    _assets = _DIST / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str, request: Request) -> FileResponse | JSONResponse:  # noqa: D401
        """
        Serve static assets or the SPA entry point.

        - API/WebSocket namespaces are excluded early.
        - Path traversal is prevented by resolving the candidate path and ensuring it stays within ``_DIST``.
        - Missing files fall back to the SPA shell to enable client‑side routing.
        """
        # Never shadow the API/WebSocket namespaces — let unmatched ones 404 as JSON
        if full_path.startswith(("api/", "ws/", "health")):
            logger.debug("Blocked request to reserved namespace: %s", full_path)
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        candidate = _DIST / full_path

        # Confirmation filter: resolve the path and verify it is inside the static directory
        try:
            resolved_candidate = candidate.resolve(strict=False)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to resolve path %s: %s", candidate, exc)
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        if not str(resolved_candidate).startswith(str(_DIST)):
            logger.warning("Attempted directory traversal attack: %s", full_path)
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        # Serve a real static file if it exists (favicon, manifest, robots…)
        if full_path and resolved_candidate.is_file():
            logger.debug("Serving static file: %s", resolved_candidate)
            return FileResponse(str(resolved_candidate))

        # Exit logic: fallback to SPA shell for client‑side routing
        logger.debug("Falling back to SPA index for path: %s", full_path)
        return FileResponse(str(_INDEX))