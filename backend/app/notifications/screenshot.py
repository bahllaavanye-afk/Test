"""Screenshot capture utility. Uses Playwright if installed, falls back to dummy."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.utils.logging import logger
from pydantic import BaseModel, Field, HttpUrl, validator


SCREENSHOTS_DIR = Path(__file__).parents[3] / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


class ScreenshotRequest(BaseModel):
    """Schema for requesting a dashboard screenshot."""

    url: HttpUrl = Field(
        ...,
        description="Base URL of the dashboard to capture.",
        example="https://quantedge.vercel.app",
    )
    page: str = Field(
        "",
        description="Relative page path on the dashboard (without leading slash).",
        example="analytics",
    )

    @validator("page")
    def normalize_page(cls, v: str) -> str:
        """Strip leading slashes and ensure the page name is safe for filenames."""
        normalized = v.lstrip("/")
        if "/" in normalized:
            # Nested paths are allowed but will be flattened for the filename
            normalized = normalized.replace("/", "_")
        return normalized


class ScreenshotResponse(BaseModel):
    """Schema representing the result of a screenshot capture."""

    filepath: str = Field(
        ...,
        description="Absolute filesystem path where the screenshot image was saved.",
        example="/app/screenshots/dashboard_analytics_20240101_120000.png",
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of when the screenshot was taken.",
        example="2024-01-01T12:00:00Z",
    )

    @validator("filepath")
    def check_path_exists(cls, v: str) -> str:
        """Validate that the screenshot file exists on disk."""
        if not Path(v).exists():
            raise ValueError(f"Screenshot file does not exist at path: {v}")
        return v


class ScreenshotBatchResponse(BaseModel):
    """Schema for a batch screenshot operation returning multiple file paths."""

    files: list[ScreenshotResponse] = Field(
        ...,
        description="List of screenshot results for each requested page.",
        example=[
            {
                "filepath": "/app/screenshots/dashboard_root_20240101_120000.png",
                "timestamp": "2024-01-01T12:00:00Z",
            },
            {
                "filepath": "/app/screenshots/dashboard_analytics_20240101_120100.png",
                "timestamp": "2024-01-01T12:01:00Z",
            },
        ],
    )


async def capture_dashboard(url: str = "https://quantedge.vercel.app", page: str = "") -> str | None:
    """
    Capture a screenshot of the dashboard. Returns the filepath or None on failure.
    Requires `playwright` installed: pip install playwright && playwright install chromium
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.debug("Playwright not installed — skipping screenshot")
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    page_name = page.replace("/", "_") or "root"
    filepath = SCREENSHOTS_DIR / f"dashboard_{page_name}_{timestamp}.png"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
            pg = await ctx.new_page()
            await pg.goto(f"{url}/{page}", wait_until="networkidle", timeout=15_000)
            await pg.screenshot(path=str(filepath), full_page=True)
            await browser.close()
        logger.info("Screenshot captured", path=str(filepath))
        return str(filepath)
    except Exception as e:
        logger.warning("Screenshot failed", error=str(e))
        return None


async def capture_all_pages(base_url: str = "https://quantedge.vercel.app") -> list[str]:
    """Capture all main dashboard pages."""
    pages = ["", "equity", "crypto", "comparison", "backtest", "experiments", "analytics", "risk"]
    results = []
    for page in pages:
        path = await capture_dashboard(base_url, page)
        if path:
            results.append(path)
        await asyncio.sleep(0.5)
    return results