"""
Tests for backend/app/ml/model_store.py
Mocks httpx calls — no network activity.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.ml.model_store import ModelStore, get_model_store


# ── 1. Local-only mode: no HTTP ────────────────────────────────────────────────


def test_model_store_local_only(tmp_path):
    """When remote is disabled, save_model writes locally and makes no HTTP calls."""
    with patch("app.ml.model_store.settings") as mock_settings, \
         patch("app.ml.model_store._ARTIFACTS_DIR", tmp_path):
        mock_settings.supabase_url = ""
        mock_settings.supabase_service_key = ""
        mock_settings.model_bucket = "model-artifacts"
        mock_settings.model_store_enabled = False

        store = ModelStore()
        assert store.remote_enabled is False

        artifact = b"fake model bytes"
        with patch("httpx.AsyncClient") as mock_client:
            result_path = asyncio.run(
                store.save_model(artifact, "test_model.pkl")
            )
            # HTTP client should never be called
            mock_client.assert_not_called()

        assert result_path.exists()
        assert result_path.read_bytes() == artifact


# ── 2. Upload success: PUT returns 200 ────────────────────────────────────────


def test_model_store_upload_success(tmp_path):
    """When PUT returns 200, upload() returns True."""
    artifact_file = tmp_path / "model.pkl"
    artifact_file.write_bytes(b"model data")

    with patch("app.ml.model_store.settings") as mock_settings:
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_key = "service-key-xyz"
        mock_settings.model_bucket = "model-artifacts"
        mock_settings.model_store_enabled = True

        store = ModelStore()
        assert store.remote_enabled is True

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.put = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            result = asyncio.run(
                store.upload(artifact_file, "model.pkl")
            )

    assert result is True
    mock_async_client.put.assert_called_once()
    call_kwargs = mock_async_client.put.call_args
    assert "model-artifacts/model.pkl" in call_kwargs[0][0]


# ── 3. Upload failure: PUT returns 500 ────────────────────────────────────────


def test_model_store_upload_failure(tmp_path):
    """When PUT returns 500, upload() returns False."""
    artifact_file = tmp_path / "model.pkl"
    artifact_file.write_bytes(b"model data")

    with patch("app.ml.model_store.settings") as mock_settings:
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_key = "service-key-xyz"
        mock_settings.model_bucket = "model-artifacts"
        mock_settings.model_store_enabled = True

        store = ModelStore()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.put = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            result = asyncio.run(
                store.upload(artifact_file, "model.pkl")
            )

    assert result is False


# ── 4. Bootstrap skips existing local files ───────────────────────────────────


def test_bootstrap_skips_existing(tmp_path):
    """If a known artifact already exists locally, no HTTP call should be made."""
    with patch("app.ml.model_store.settings") as mock_settings, \
         patch("app.ml.model_store._ARTIFACTS_DIR", tmp_path):
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_key = "service-key-xyz"
        mock_settings.model_bucket = "model-artifacts"
        mock_settings.model_store_enabled = True

        store = ModelStore()

        # Pre-create all known artifacts locally
        for name in ["lstm_latest.pt", "xgboost_latest.ubj", "lorentzian_latest.pkl", "scaler_latest.pkl"]:
            (tmp_path / name).write_bytes(b"existing model")

        with patch("httpx.AsyncClient") as mock_client:
            bootstrapped = asyncio.run(
                store.bootstrap_local()
            )
            mock_client.assert_not_called()

    assert bootstrapped == []


# ── 5. Singleton: two calls return same instance ──────────────────────────────


def test_get_model_store_singleton():
    """get_model_store() should return the same ModelStore instance each time."""
    import app.ml.model_store as store_module
    # Reset the global singleton
    store_module._store = None

    store1 = get_model_store()
    store2 = get_model_store()

    assert store1 is store2
    assert isinstance(store1, ModelStore)

    # Cleanup
    store_module._store = None
