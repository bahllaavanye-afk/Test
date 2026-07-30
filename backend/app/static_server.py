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

from pathlib import Path
from urllib.parse import unquote

from fastapi import Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.main import app  # the existing FastAPI app — unchanged

# frontend/dist relative to repo root (backend/app/static_server.py → ../../frontend/dist)
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
_INDEX = _DIST / "index.html"

if _INDEX.is_file():
    _assets = _DIST / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="spa-assets")

    @app.get("/health", include_in_schema=False)
    async def _health() -> JSONResponse:
        """Simple health check endpoint."""
        return JSONResponse({"status": "ok"}, status_code=200)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str) -> Response:  # noqa: D401
        """
        Serve the SPA or static assets.

        - Reject paths that target API, WebSocket, or health endpoints.
        - Prevent directory traversal attacks.
        - Serve existing static files; otherwise fall back to the SPA shell.
        """
        # Never shadow the API/WebSocket namespaces — let unmatched ones 404 as JSON
        if full_path.startswith(("api/", "ws/", "health")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        # Decode URL-encoded characters and normalize the path
        sanitized_path = unquote(full_path).lstrip("/")
        candidate = (_DIST / sanitized_path).resolve()

        # Ensure the resolved path is still within the distribution directory
        if not str(candidate).startswith(str(_DIST)):
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        # Serve a real static file if it exists (favicon, manifest, robots…),
        # otherwise return the SPA shell so client-side routing works on any path.
        if sanitized_path and candidate.is_file():
            return FileResponse(
                str(candidate),
                headers={"Cache-Control": "public, max-age=31536000"},
            )
        return FileResponse(
            str(_INDEX),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )