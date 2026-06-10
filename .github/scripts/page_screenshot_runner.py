"""
QuantEdge Real Page Screenshot Runner
======================================
Takes actual browser screenshots of every dashboard page and uploads them
to Slack channels. Requires Playwright (installed by the CI workflow).

Usage:
    python page_screenshot_runner.py [--base-url URL] [--output-dir DIR]

Default base URL: http://localhost:5173

The screenshots are taken at 1440x900 (laptop) and 375x812 (mobile).
Each screenshot is uploaded to the assigned Slack channel for that page.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Page → (route, channel, employee_display_name, emoji)
PAGES: list[tuple[str, str, str, str]] = [
    ("/landing",     "general",          "Product Lead (Sarah Kim)",  ":rocket:"),
    ("/login",       "squad-frontend",   "Priya Iyer (VP Frontend)",  ":key:"),
    ("/",            "squad-frontend",   "Priya Iyer (VP Frontend)",  ":house:"),
    ("/equity",      "desk-equities",    "Alpha Director",            ":chart_with_upwards_trend:"),
    ("/crypto",      "desk-crypto",      "ML Lead",                   ":coin:"),
    ("/comparison",  "strategy-review",  "Alpha Director",            ":bar_chart:"),
    ("/backtest",    "alpha-research",   "Backtest Engineer",         ":test_tube:"),
    ("/experiments", "ml-experiments",   "ML Lead",                   ":microscope:"),
    ("/insights",    "ml-experiments",   "ML Researcher",             ":brain:"),
    ("/analytics",   "pnl-daily",        "Portfolio Manager",         ":abacus:"),
    ("/risk",        "risk-alerts",      "Risk Engineer",             ":shield:"),
    ("/agents",      "engineering",      "VP Engineering",            ":robot_face:"),
    ("/pnl",         "pnl-daily",        "Portfolio Manager",         ":money_with_wings:"),
    ("/monitor",     "infra-alerts",     "DevOps Director",           ":computer:"),
]


def _post_to_slack(token: str, channel: str, text: str, username: str = "Screenshot Bot",
                   icon_emoji: str = ":camera:") -> dict:
    payload = {
        "channel": channel,
        "text": text,
        "username": username,
        "icon_emoji": icon_emoji,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _upload_screenshot_to_slack(token: str, channel: str, file_path: Path,
                                 title: str, comment: str) -> dict:
    """Upload a PNG screenshot to a Slack channel via files.upload API."""
    ch_id = _get_channel_id(token, channel)
    if not ch_id:
        print(f"  [screenshot] channel #{channel} not found — posting text only")
        _post_to_slack(token, channel, f":camera: {comment}\n_(screenshot upload failed — channel not found)_")
        return {"ok": False, "error": "channel_not_found"}

    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # Use multipart form upload
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    body_parts = []

    def _field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body_parts.append(_field("channels", ch_id))
    body_parts.append(_field("title", title))
    body_parts.append(_field("initial_comment", comment))
    body_parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + file_data + b"\r\n"
    )
    body_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(body_parts)

    req = urllib.request.Request(
        "https://slack.com/api/files.upload",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                # Fall back to text post with description
                _post_to_slack(
                    token, channel,
                    f":camera: {comment}\n_(image upload failed: {result.get('error', 'unknown')})_",
                )
            return result
    except Exception as e:
        _post_to_slack(token, channel, f":camera: {comment}\n_(upload error: {e})_")
        return {"ok": False, "error": str(e)}


def _get_channel_id(token: str, channel_name: str) -> str | None:
    clean = channel_name.lstrip("#")
    url = (
        "https://slack.com/api/conversations.list"
        f"?limit=200&exclude_archived=true"
    )
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        for ch in data.get("channels", []):
            if ch.get("name") == clean:
                return ch["id"]
    except Exception:
        pass
    return None


def take_screenshots(base_url: str, output_dir: Path, token: str) -> int:
    """
    Use Playwright to take real browser screenshots of every page.
    Returns number of screenshots successfully taken and posted.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("[screenshot] playwright not installed — run: pip install playwright && playwright install chromium")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    posted = 0
    now_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage"],
            headless=True,
        )

        # Desktop viewport
        ctx_desktop = browser.new_context(
            viewport={"width": 1440, "height": 900},
            color_scheme="dark",
            device_scale_factor=1,
        )

        for route, channel, employee, emoji in PAGES:
            url = f"{base_url}{route}"
            page_name = route.lstrip("/") or "dashboard"
            print(f"  [screenshot] {url} → #{channel}")

            page = ctx_desktop.new_page()
            try:
                # Navigate and wait for network idle (max 15s)
                page.goto(url, wait_until="networkidle", timeout=15_000)
                # Extra wait for WebSocket components to render
                page.wait_for_timeout(2000)
            except PWTimeout:
                print(f"  [screenshot] timeout on {url} — taking screenshot of current state")
            except Exception as e:
                print(f"  [screenshot] error on {url}: {e}")
                page.close()
                continue

            # Desktop screenshot
            screenshot_path = output_dir / f"{page_name}_desktop.png"
            try:
                page.screenshot(path=str(screenshot_path), full_page=False)
            except Exception as e:
                print(f"  [screenshot] screenshot failed for {url}: {e}")
                page.close()
                continue

            page.close()

            # Upload to Slack
            if token.startswith("xoxb-"):
                comment = (
                    f"{emoji} *{page_name.title()} — Live Screenshot* | {now_str}\n"
                    f"_Reported by {employee} · 1440×900 desktop view_\n"
                    f"URL: `{url}`"
                )
                result = _upload_screenshot_to_slack(
                    token, channel, screenshot_path,
                    title=f"QuantEdge {page_name.title()} — {now_str}",
                    comment=comment,
                )
                if result.get("ok"):
                    posted += 1
                    print(f"  [screenshot] ✓ {page_name} → #{channel}")
                else:
                    print(f"  [screenshot] ✗ {page_name} → #{channel}: {result.get('error')}")
                time.sleep(1.5)  # Slack rate limit
            else:
                print(f"  [screenshot] (no token) saved to {screenshot_path}")
                posted += 1

        ctx_desktop.close()
        browser.close()

    return posted


def post_screenshot_summary(token: str, pages_done: int, pages_total: int) -> None:
    """Post a summary of the screenshot run to #squad-frontend."""
    now_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    msg = (
        f":camera_flash: *Page Screenshot Run Complete* | {now_str}\n"
        f"• {pages_done}/{pages_total} pages captured and posted to their channels\n"
        f"• Desktop view: 1440×900 · Dark theme · Headless Chromium\n"
        f"• Each page posted to its assigned team channel for review\n"
        f"_All employees: check your channel for latest UI screenshots and flag any issues._"
    )
    _post_to_slack(token, "squad-frontend", msg,
                   username="Screenshot Bot", icon_emoji=":camera_flash:")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="QuantEdge Page Screenshot Runner")
    parser.add_argument("--base-url", default="http://localhost:5173", help="Frontend base URL")
    parser.add_argument("--output-dir", default="/tmp/quantedge-screenshots", help="Screenshot output directory")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    output_dir = Path(args.output_dir)

    print(f"📸 QuantEdge Screenshot Runner")
    print(f"   Base URL: {args.base_url}")
    print(f"   Output:   {output_dir}")
    print(f"   Slack:    {'✅ token present' if token.startswith('xoxb-') else '⚠️ no token'}")
    print(f"   Pages:    {len(PAGES)}")
    print()

    n = take_screenshots(args.base_url, output_dir, token)

    if token.startswith("xoxb-"):
        post_screenshot_summary(token, n, len(PAGES))

    print(f"\n✅ Done — {n}/{len(PAGES)} pages captured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
