"""Screenshot capture utility. Uses Playwright if installed, falls back to dummy."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from app.utils.logging import logger

SCREENSHOTS_DIR = Path(__file__).parents[3] / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


async def capture_dashboard(url: str = "http://localhost:5173", page: str = "") -> str | None:
    """
    Capture a screenshot of the dashboard. Returns the filepath or None on failure.
    Requires `playwright` installed: pip install playwright && playwright install chromium
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.debug("Playwright not installed — skipping screenshot")
        return None

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
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


async def capture_all_pages(base_url: str = "http://localhost:5173") -> list[str]:
    """Capture all main dashboard pages."""
    pages = ["", "equity", "crypto", "comparison", "backtest", "experiments", "analytics", "risk"]
    results = []
    for page in pages:
        path = await capture_dashboard(base_url, page)
        if path:
            results.append(path)
        await asyncio.sleep(0.5)
    return results
