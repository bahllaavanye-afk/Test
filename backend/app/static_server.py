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

from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.main import app  # the existing FastAPI app — unchanged

# ---- Constants --------------------------------------------------------------

FRONTEND_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend"
DIST_DIR_NAME = "dist"
ASSETS_DIR_NAME = "assets"
INDEX_FILE_NAME = "index.html"

DIST_PATH = FRONTEND_ROOT / DIST_DIR_NAME
INDEX_PATH = DIST_PATH / INDEX_FILE_NAME

ASSETS_MOUNT_PATH = "/assets"
ASSETS_MOUNT_NAME = "spa-assets"

SPA_ROUTE = "/{full_path:path}"
API_PATH_PREFIXES = ("api/", "ws/", "health")
NOT_FOUND_DETAIL = "Not Found"
NOT_FOUND_STATUS_CODE = 404

# ---- Static assets mounting --------------------------------------------------

if INDEX_PATH.is_file():
    _assets_path = DIST_PATH / ASSETS_DIR_NAME
    if _assets_path.is_dir():
        app.mount(
            ASSETS_MOUNT_PATH,
            StaticFiles(directory=str(_assets_path)),
            name=ASSETS_MOUNT_NAME,
        )

    @app.get(SPA_ROUTE, include_in_schema=False)
    async def _spa(full_path: str):  # noqa: D401
        # Never shadow the API/WebSocket namespaces — let unmatched ones 404 as JSON
        # (this route only runs when no registered API route matched first).
        if full_path.startswith(API_PATH_PREFIXES):
            return JSONResponse(
                {"detail": NOT_FOUND_DETAIL}, status_code=NOT_FOUND_STATUS_CODE
            )
        # Serve a real static file if it exists (favicon, manifest, robots…),
        # otherwise return the SPA shell so client-side routing works on any path.
        candidate = DIST_PATH / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(INDEX_PATH))