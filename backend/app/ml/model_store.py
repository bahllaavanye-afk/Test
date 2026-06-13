"""
Durable model artifact store backed by Supabase Storage REST API.
Falls back to local filesystem when Supabase is not configured.

Upload:  PUT  {supabase_url}/storage/v1/object/{bucket}/{path}
Download: GET  {supabase_url}/storage/v1/object/{bucket}/{path}
"""
from __future__ import annotations

from pathlib import Path

import httpx

from app.config import settings
from app.utils.logging import logger

_ARTIFACTS_DIR = Path(__file__).parents[3] / "models_artifacts"


class ModelStore:
    """
    Two-tier artifact storage:
    1. Supabase Storage (if configured) — survives Render restarts
    2. Local filesystem — fallback for dev and when Supabase not configured
    """

    def __init__(self) -> None:
        self._base = settings.supabase_url.rstrip("/") if settings.supabase_url else ""
        self._bucket = settings.model_bucket
        self._key = settings.supabase_service_key
        self._remote_enabled = bool(
            settings.model_store_enabled and self._base and self._key
        )
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def remote_enabled(self) -> bool:
        return self._remote_enabled

    # ── Upload ────────────────────────────────────────────────────────────────

    async def upload(self, local_path: Path, remote_name: str) -> bool:
        """Upload file to Supabase Storage. Returns True on success."""
        if not self._remote_enabled:
            return False
        try:
            data = local_path.read_bytes()
            url = f"{self._base}/storage/v1/object/{self._bucket}/{remote_name}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.put(
                    url,
                    content=data,
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "application/octet-stream",
                        "x-upsert": "true",
                    },
                )
            if resp.status_code in (200, 201):
                logger.info("Model uploaded to Supabase", remote_name=remote_name, bytes=len(data))
                return True
            else:
                logger.warning("Supabase upload failed", status=resp.status_code, body=resp.text[:200])
                return False
        except Exception as e:
            logger.warning("Supabase upload exception", error=str(e), remote_name=remote_name)
            return False

    # ── Download ──────────────────────────────────────────────────────────────

    async def download(self, remote_name: str, local_path: Path) -> bool:
        """Download from Supabase Storage to local path. Returns True on success."""
        if not self._remote_enabled:
            return False
        try:
            url = f"{self._base}/storage/v1/object/{self._bucket}/{remote_name}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self._key}"},
                )
            if resp.status_code == 200:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(resp.content)
                logger.info("Model downloaded from Supabase", remote_name=remote_name, bytes=len(resp.content))
                return True
            elif resp.status_code == 404:
                logger.debug("Model not found in Supabase", remote_name=remote_name)
                return False
            else:
                logger.warning("Supabase download failed", status=resp.status_code)
                return False
        except Exception as e:
            logger.warning("Supabase download exception", error=str(e), remote_name=remote_name)
            return False

    # ── Save (local + upload) ─────────────────────────────────────────────────

    async def save_model(self, model_bytes: bytes, artifact_name: str) -> Path:
        """Save bytes locally and upload to Supabase. Returns local path."""
        local_path = _ARTIFACTS_DIR / artifact_name
        local_path.write_bytes(model_bytes)
        if self._remote_enabled:
            await self.upload(local_path, artifact_name)
        return local_path

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    async def bootstrap_local(self) -> list[str]:
        """
        On startup: for each known model artifact, if missing locally,
        attempt to download from Supabase. Returns list of successfully bootstrapped names.
        """
        KNOWN_ARTIFACTS = [
            "lstm_latest.pt",
            "xgboost_latest.ubj",
            "lorentzian_latest.pkl",
            "scaler_latest.pkl",
        ]
        bootstrapped = []
        for name in KNOWN_ARTIFACTS:
            local_path = _ARTIFACTS_DIR / name
            if not local_path.exists() and self._remote_enabled:
                ok = await self.download(name, local_path)
                if ok:
                    bootstrapped.append(name)
        return bootstrapped


# Global singleton
_store: ModelStore | None = None


def get_model_store() -> ModelStore:
    global _store
    if _store is None:
        _store = ModelStore()
    return _store
