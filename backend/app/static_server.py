"""Serve the built Vite SPA from the FastAPI backend — single-deployment hosting.

This wraps the existing app (`app.main:app`) so we do NOT modify main.py. Render
builds `frontend/dist` (see render.yaml) and `start.sh` runs
`uvicorn app.static_server:app`. The API routers are registered first on the
imported app, so they always match before the SPA catch-all. If the build dir is
missing (e.g. backend-only dev), the API still works and the catch-all no-ops.

Why this instead of Vercel: one deployment, one origin (no CORS), auto-deploys
from main on Render — no separate frontend host or orphaned deployments.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.main import app  # the existing FastAPI app — unchanged

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
_logger = logging.getLogger("static_server")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

# Global request counter for metric “signal count”
_request_counter: int = 0


class _LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs request‑level metrics for the static server.

    For each request a JSON‑serialisable dict is emitted at INFO level containing:
    - ``signal_count``: total number of requests processed since process start.
    - ``execution_time_ms``: processing time for the request.
    - ``path``: request path.
    - ``pnl``: placeholder for profit & loss (not applicable for static assets).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        global _request_counter
        _request_counter += 1
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        _logger.info(
            {
                "signal_count": _request_counter,
                "execution_time_ms": round(elapsed_ms, 2),
                "path": request.url.path,
                "pnl": None,
            }
        )
        return response


# Register the middleware before mounting static assets
app.add_middleware(_LoggingMiddleware)

# --------------------------------------------------------------------------- #
# Static file serving
# --------------------------------------------------------------------------- #
# frontend/dist relative to repo root (backend/app/static_server.py → ../../frontend/dist)
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
_INDEX = _DIST / "index.html"

if _INDEX.is_file():
    _assets = _DIST / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str) -> FileResponse | JSONResponse:  # noqa: D401
        """Serve SPA assets or fallback to the SPA entry point.

        The route is only reached when no earlier API/WebSocket route matches.
        It returns a static file when it exists (e.g., favicon, manifest) or the
        main ``index.html`` otherwise, enabling client‑side routing.
        """
        # Never shadow the API/WebSocket namespaces — let unmatched ones 404 as JSON
        if full_path.startswith(("api/", "ws/", "health")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_INDEX))