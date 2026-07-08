"""Unit-test fixtures — does NOT load the FastAPI app (no DB/network required)."""
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, validator

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class TestConfig(BaseModel):
    """Configuration model for unit tests."""

    backend_root: Path = Field(
        ...,
        description="Root directory of the backend project.",
        example="/absolute/path/to/backend",
    )
    env: str = Field(
        "test",
        description="Name of the testing environment.",
        example="test",
    )
    enable_network: bool = Field(
        False,
        description="Whether network calls are allowed during tests.",
        example=False,
    )

    @validator("backend_root")
    def validate_backend_root(cls, v: Path) -> Path:
        """Ensure the backend root directory exists."""
        if not v.is_dir():
            raise ValueError(f"backend_root does not exist: {v}")
        return v


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Provides a TestConfig instance for tests."""
    return TestConfig(backend_root=BACKEND_ROOT)