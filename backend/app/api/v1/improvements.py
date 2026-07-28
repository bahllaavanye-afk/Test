"""Self-improvement history endpoint."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver:
        return improver.get_history()
    return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    from app.main import app
    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": "not_running", "message": "Code quality loop not started"}
    return loop_ref.latest()


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        return {"status": "not_running", "best_params": {}}
    return {"best_params": getattr(improver, "_best_params", {})}


# ==============================
# Unit tests for edge conditions
# ==============================
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Helper to create a minimal FastAPI app with the router attached
def _create_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(monkeypatch):
    """
    Fixture that provides a TestClient with a mocked `app.state`.
    The `get_current_user` dependency is also overridden to bypass auth.
    """
    test_app = _create_test_app()

    # Override the authentication dependency to return a dummy user
    async def _dummy_user():
        return User(id=0, username="test_user")
    test_app.dependency_overrides[get_current_user] = _dummy_user

    # Expose the test app via the import path used in the endpoint functions
    import importlib
    app_module = importlib.import_module("app.main")
    monkeypatch.setattr(app_module, "app", test_app, raising=False)

    return TestClient(test_app)


def test_get_history_without_improver_returns_empty_list(client):
    """When `self_improver` is not set, the endpoint should return an empty list."""
    response = client.get("/improvements/history")
    assert response.status_code == 200
    assert response.json() == []


def test_get_quality_when_loop_not_running_returns_status_message(client):
    """When `code_quality_loop` is absent, the endpoint must report not_running status."""
    response = client.get("/improvements/quality")
    assert response.status_code == 200
    json_body = response.json()
    assert json_body["status"] == "not_running"
    assert "Code quality loop not started" in json_body["message"]


def test_get_best_params_without_improver_returns_not_running(client):
    """When `self_improver` is missing, the best_params endpoint should indicate not running."""
    response = client.get("/improvements/best_params")
    assert response.status_code == 200
    json_body = response.json()
    assert json_body["status"] == "not_running"
    assert json_body["best_params"] == {}