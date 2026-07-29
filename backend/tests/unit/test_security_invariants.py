"""Brain-independent security invariants — structural guards that fail if a
safety-critical control is removed again.

These exist because two Codex security fixes silently regressed (the REST order risk
gate and the /slack/events signature check) and the LLM-based red team missed it (the
brain was unavailable). These are pure source-structure assertions: no LLM, no network,
so they catch regressions even when the brain is down.
"""
from pathlib import Path

from pydantic import BaseModel, Field, validator

API = Path(__file__).resolve().parents[2] / "app" / "api" / "v1"
CONFIG = Path(__file__).resolve().parents[2] / "app" / "config.py"


class PathSchema(BaseModel):
    """Schema representing critical project paths used in security invariant tests.

    Attributes
    ----------
    api_path: Path
        Path to the API v1 directory. Must exist and be a directory.
    config_path: Path
        Path to the application configuration file. Must exist and be a file.
    """
    api_path: Path = Field(
        ...,
        description="Path to the API v1 directory containing route definitions.",
        example="/path/to/project/app/api/v1",
    )
    config_path: Path = Field(
        ...,
        description="Path to the main application configuration file.",
        example="/path/to/project/app/config.py",
    )

    @validator("api_path")
    def api_path_must_be_directory(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"api_path does not exist or is not a directory: {v}")
        return v

    @validator("config_path")
    def config_path_must_be_file(cls, v: Path) -> Path:
        if not v.is_file():
            raise ValueError(f"config_path does not exist or is not a file: {v}")
        return v


# Instantiate the schema for potential external use; tests continue to use API and CONFIG directly.
PATHS = PathSchema(api_path=API, config_path=CONFIG)


def test_rest_order_submission_is_risk_gated():
    src = (API / "orders.py").read_text()
    assert "risk_manager" in src, "risk manager reference missing from orders.py"
    # submit_order AND submit_bracket must each gate through check_order before the broker.
    assert src.count("check_order") >= 2, (
        "REST order submission must call risk_manager.check_order() in BOTH submit_order "
        "and submit_bracket before reaching the broker (regression guard)"
    )
    # The gate must precede the broker call in the file.
    assert src.index("check_order") < src.rindex("submit_alpaca_order"), (
        "risk check must come before the Alpaca submission"
    )


def test_every_notifications_route_requires_auth():
    """No unauthenticated inbound endpoint on the notifications router.

    Replaces the old /slack/events signature test: that endpoint was an
    UNAUTHENTICATED inbound webhook, so it needed an HMAC signature check to be
    safe. It was removed with the rest of the Slack integration (2026-07-25), and
    the invariant that matters now is the stronger one — every route here is
    behind `get_current_user`, so there is no unauthenticated entry point at all.
    If someone re-adds a public webhook, this fails and they must add verification.
    """
    src = (API / "notifications.py").read_text()
    routes = [ln for ln in src.splitlines() if ln.strip().startswith("@router.")]
    assert routes, "expected at least one route in notifications.py"
    # Count the auth dependency once per route handler.
    assert src.count("Depends(get_current_user)") >= len(routes), (
        f"{len(routes)} routes but only {src.count('Depends(get_current_user)')} "
        "auth dependencies — every notifications route must require auth, or carry "
        "its own request-signature verification if it is a public webhook"
    )


def test_slack_integration_stays_removed():
    """Slack was removed completely on 2026-07-25 (user directive).

    The autonomous improver rewrites these files unattended, so without a guard
    it can reintroduce a Slack path — which would silently swallow notifications
    again (the dead SLACK_BOT_TOKEN made `slack_post` a no-op for weeks).
    """
    backend_app = CONFIG.parent
    offenders = []
    for path in backend_app.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(errors="ignore")
        if "slack.com/api" in text or "SLACK_BOT_TOKEN" in text:
            offenders.append(str(path.relative_to(backend_app)))
    assert not offenders, f"Slack must stay removed from the backend; found in: {offenders}"