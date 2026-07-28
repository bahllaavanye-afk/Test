"""Screenshot capture utility. Uses Playwright if installed, falls back to dummy."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.utils.logging import logger

SCREENSHOTS_DIR = Path(__file__).parents[3] / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


async def capture_dashboard(url: str = "https://quantedge.vercel.app", page: str = "") -> Optional[str]:
    """
    Capture a screenshot of the dashboard. Returns the filepath or None on failure.
    Requires `playwright` installed: pip install playwright && playwright install chromium
    """
    # Edge‑case handling for inputs
    if not isinstance(url, str) or not url:
        logger.warning("Invalid URL supplied to capture_dashboard", url=url)
        return None
    if page is None:
        page = ""
    if not isinstance(page, str):
        logger.warning("Invalid page type supplied to capture_dashboard", page=page)
        return None

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
            await pg.goto(f"{url.rstrip('/')}/{page.lstrip('/')}", wait_until="networkidle", timeout=15_000)
            await pg.screenshot(path=str(filepath), full_page=True)
            await browser.close()
        logger.info("Screenshot captured", path=str(filepath))
        return str(filepath)
    except Exception as e:
        logger.warning("Screenshot failed", error=str(e))
        return None


async def capture_all_pages(base_url: str = "https://quantedge.vercel.app") -> List[str]:
    """Capture all main dashboard pages."""
    if not isinstance(base_url, str) or not base_url:
        logger.warning("Invalid base_url supplied to capture_all_pages", base_url=base_url)
        return []

    pages = ["", "equity", "crypto", "comparison", "backtest", "experiments", "analytics", "risk"]
    results: List[str] = []

    for page in pages:
        path = await capture_dashboard(base_url, page)
        if path:
            results.append(path)
        # Guard against potential off‑by‑one timing issues by ensuring a minimal pause
        await asyncio.sleep(0.5)

    return results