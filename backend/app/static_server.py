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
from pathlib import Path

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
    async def _spa(full_path: str):  # noqa: D401
        """Catch‑all route serving the SPA or static assets.

        Returns a JSON 404 for API/WebSocket namespace collisions, serves the
        requested static file when it exists, otherwise falls back to the SPA
        index. Unexpected errors are logged and result in a 500 response.
        """
        try:
            # Never shadow the API/WebSocket namespaces — let unmatched ones 404 as JSON
            # (this route only runs when no registered API route matched first).
            if full_path.startswith(("api/", "ws/", "health")):
                return JSONResponse({"detail": "Not Found"}, status_code=404)

            # Serve a real static file if it exists (favicon, manifest, robots…),
            # otherwise return the SPA shell so client‑side routing works on any path.
            candidate = _DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(str(candidate))

            return FileResponse(str(_INDEX))

        except FileNotFoundError as exc:
            # Specific handling for missing files; this should be rare because
            # existence checks are performed above, but we guard against race
            # conditions.
            logger.error(
                "File not found while serving SPA",
                extra={"path": full_path, "error": str(exc)},
            )
            return JSONResponse({"detail": "File Not Found"}, status_code=404)

        except PermissionError as exc:
            logger.error(
                "Permission error while accessing SPA assets",
                extra={"path": full_path, "error": str(exc)},
            )
            return JSONResponse({"detail": "Permission Denied"}, status_code=403)

        except Exception as exc:  # pylint: disable=broad-except
            # Catch‑all for unexpected runtime errors; log with stack trace.
            logger.exception(
                "Unexpected error serving SPA",
                extra={"path": full_path, "error": str(exc)},
            )
            return JSONResponse({"detail": "Internal Server Error"}, status_code=500)