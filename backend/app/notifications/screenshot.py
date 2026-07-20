"""Screenshot capture utility. Uses Playwright if installed, falls back to dummy."""
from __future__ import annotations

import asyncio
import logging
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.utils.logging import logger

# Directory where screenshots are stored
SCREENSHOTS_DIR = Path(__file__).parents[3] / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration constants
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds, multiplied each retry
_TIMEOUT_MS = 15_000
_VIEWPORT = {"width": 1920, "height": 1080}


def _is_valid_url(url: str) -> bool:
    """Basic validation that the URL has a scheme and netloc."""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _sanitize_page_name(page: str) -> str:
    """Convert a page path to a filesystem‑safe name."""
    if not page:
        return "root"
    # Replace path separators and any characters unsafe for filenames
    safe = page.replace("/", "_")
    return safe


async def _capture_once(
    url: str,
    page: str,
    filepath: Path,
) -> bool:
    """
    Perform a single attempt to capture a screenshot.
    Returns True on success, False otherwise.
    """
    try:
        from playwright.async_api import async_playwright  # Local import to avoid hard dependency
    except ImportError:
        logger.debug("Playwright not installed — skipping screenshot")
        return False

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(viewport=_VIEWPORT)
            pg = await context.new_page()
            target = f"{url.rstrip('/')}/{page.lstrip('/')}" if page else url
            await pg.goto(target, wait_until="networkidle", timeout=_TIMEOUT_MS)
            await pg.screenshot(path=str(filepath), full_page=True)
            # Verify that the file was written
            if filepath.is_file():
                logger.info("Screenshot captured", path=str(filepath))
                return True
            else:
                logger.warning("Screenshot file missing after capture", path=str(filepath))
                return False
        finally:
            await browser.close()


async def capture_dashboard(url: str = "https://quantedge.vercel.app", page: str = "") -> Optional[str]:
    """
    Capture a screenshot of the dashboard. Returns the filepath as a string or None on failure.

    The function validates the URL, sanitizes the page name, and retries a few times
    with exponential back‑off if transient errors occur.
    """
    if not _is_valid_url(url):
        logger.warning("Invalid URL supplied for screenshot capture", url=url)
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    page_name = _sanitize_page_name(page)
    filepath = SCREENSHOTS_DIR / f"dashboard_{page_name}_{timestamp}.png"

    attempt = 0
    while attempt < _MAX_RETRIES:
        success = await _capture_once(url, page, filepath)
        if success:
            return str(filepath)
        attempt += 1
        backoff = _RETRY_BACKOFF * (2 ** (attempt - 1))
        logger.debug(
            "Retrying screenshot capture",
            url=url,
            page=page,
            attempt=attempt,
            backoff=backoff,
        )
        await asyncio.sleep(backoff)

    logger.warning(
        "All attempts to capture screenshot failed",
        url=url,
        page=page,
        attempts=_MAX_RETRIES,
    )
    return None


async def capture_all_pages(base_url: str = "https://quantedge.vercel.app") -> List[str]:
    """Capture all main dashboard pages and return a list of successfully saved file paths."""
    pages = [
        "",
        "equity",
        "crypto",
        "comparison",
        "backtest",
        "experiments",
        "analytics",
        "risk",
    ]
    results: List[str] = []
    for page in pages:
        path = await capture_dashboard(base_url, page)
        if path:
            results.append(path)
        # Small pause to avoid overwhelming the target server
        await asyncio.sleep(0.5)
    return results