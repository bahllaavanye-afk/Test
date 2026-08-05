"""Screenshot capture utility.

This module provides asynchronous functions to capture screenshots of the QuantEdge dashboard.
It uses Playwright when available, otherwise it logs a debug message and returns ``None``.
Screenshots are saved under the project's ``screenshots`` directory with a timestamped
filename.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.utils.logging import logger

# Directory where screenshots are stored. Created on import if it does not exist.
SCREENSHOTS_DIR = Path(__file__).parents[3] / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


async def capture_dashboard(url: str = "https://quantedge.vercel.app", page: str = "") -> str | None:
    """Capture a screenshot of a specific dashboard page.

    Args:
        url: Base URL of the dashboard. Defaults to the public QuantEdge site.
        page: Relative path of the page to capture (e.g., ``"equity"``). An empty string
            represents the root page. Slashes are replaced with underscores for the
            filename.

    Returns:
        The absolute path to the saved screenshot as a string, or ``None`` if the
        capture could not be performed (e.g., Playwright is not installed or an
        exception occurs).
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
    """Capture screenshots for all primary dashboard pages.

    Args:
        base_url: The base URL of the dashboard to use for each page request.

    Returns:
        A list containing the file paths of successfully captured screenshots.
        Pages that fail to capture are omitted from the list.
    """
    pages: list[str] = [
        "",
        "equity",
        "crypto",
        "comparison",
        "backtest",
        "experiments",
        "analytics",
        "risk",
    ]
    results: list[str] = []
    for page in pages:
        path = await capture_dashboard(base_url, page)
        if path:
            results.append(path)
        await asyncio.sleep(0.5)
    return results