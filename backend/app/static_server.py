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
from typing import Optional

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

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: Optional[str] = None):  # noqa: D401
        """
        Serve static assets when they exist, otherwise fall back to the SPA index.

        Edge‑case handling:
        * `full_path` may be ``None`` – treat it as an empty string.
        * Empty or ``/`` paths are normalised to ``""``.
        * Leading slashes are stripped to avoid absolute path resolution.
        * Path components are joined safely to prevent directory‑traversal attacks.
        """
        # Normalise the incoming path
        if not full_path:
            clean_path = ""
        else:
            clean_path = full_path.lstrip("/")

        # Never shadow the API/WebSocket namespaces — let unmatched ones 404 as JSON
        # (this route only runs when no registered API route matched first).
        if clean_path.startswith(("api/", "ws/", "health")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        # Resolve the candidate file safely, avoiding ".." traversal.
        path_parts = Path(clean_path).parts
        candidate = _DIST.joinpath(*path_parts) if path_parts else _DIST

        # Serve a real static file if it exists (favicon, manifest, robots…),
        # otherwise return the SPA shell so client‑side routing works on any path.
        if clean_path and candidate.is_file():
            return FileResponse(str(candidate))

        return FileResponse(str(_INDEX))