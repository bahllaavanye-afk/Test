"""
QuantEdge — Live App Screenshot Bot.

Builds the frontend, serves it (vite preview) alongside the FastAPI backend,
then drives a headless Chromium (Playwright) across every page route, taking a
full-page screenshot of each. For every page it runs a lightweight
"what's broken" heuristic:

  • blank / near-white page   → mean brightness high + low pixel stdev (PIL)
  • console errors            → page.on("console", level == "error")
  • uncaught page errors      → page.on("pageerror")
  • failed network requests   → page.on("response", status >= 400)

The screenshots are composited into a single contact-sheet PNG, and that image
plus a text analysis of which pages look broken is posted to the Slack channel
"#app-screenshots" (auto-created). Individual page PNGs are also saved so the
workflow can upload them as an artifact.

Honesty contract: if Playwright/browsers aren't installed, or the servers never
come up, we print a clear diagnosis and exit 0 — we never fabricate a screenshot.

Required env:
  SLACK_BOT_TOKEN   xoxb-... with chat:write, files:write, channels:read/manage
  GH_REPO           owner/repo (optional, used only for context in the comment)

The Slack upload helpers (slack_call / ensure_channel / upload_image /
_legacy_upload) mirror .github/scripts/employee_dashboard.py exactly.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"
BACKEND_DIR = REPO_ROOT / "backend"

CHANNEL = os.environ.get("APP_SCREENSHOTS_CHANNEL", "app-screenshots")

FRONTEND_PORT = 5173
BACKEND_PORT = 8000
FRONTEND_BASE = f"http://localhost:{FRONTEND_PORT}"
BACKEND_BASE = f"http://localhost:{BACKEND_PORT}"

VIEWPORT = {"width": 1440, "height": 900}

# Output dir for individual screenshots + contact sheet.
OUT_DIR = REPO_ROOT / "app_screenshots_out"

# Routes to capture. Derived from frontend/src/App.tsx — these are the REAL
# paths, not guesses. Landing is /landing (NOT /), "/" is the protected
# Dashboard, MLInsights is /ml-insights. Protected routes sit behind a
# RequireAuth gate that only checks sessionStorage["access_token"] is truthy,
# so we seed a dummy token (see add_init_script below) to let them render.
ROUTES: list[tuple[str, str, bool]] = [
    # (label, path, is_public)
    ("Landing", "/landing", True),
    ("Login", "/login", True),
    ("Dashboard", "/", False),
    ("EquityTrading", "/equity", False),
    ("CryptoTrading", "/crypto", False),
    ("Comparison", "/comparison", False),
    ("BacktestLab", "/backtest", False),
    ("Experiments", "/experiments", False),
    ("Analytics", "/analytics", False),
    ("RiskManager", "/risk", False),
    ("Polymarket", "/polymarket", False),
    ("MLInsights", "/ml-insights", False),
]

# A token-shaped string so the request interceptor sends an Authorization
# header (which will 401 against the real backend — that's a legitimate
# "broken" signal we want to surface, not hide). Any non-empty value gets the
# UI past the RequireAuth <Navigate to="/login"> gate.
SEED_TOKEN = "screenshot-bot-token"

# Blank-page heuristic: a real page has visual structure → high pixel stdev.
# A blank/near-white (or solid) page has very low stdev.
BLANK_STDEV_THRESHOLD = 8.0
NEAR_WHITE_MEAN_THRESHOLD = 245.0


# ─────────────────────────────────────────────────────────────────────────────
# Server lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status < 500
    except urllib.error.HTTPError as e:
        # Any HTTP response (even 404/405) means the server is up.
        return e.code < 500
    except Exception:
        return False


def wait_for_http(url: str, timeout: float = 60.0, label: str = "") -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if http_ok(url):
            print(f"  {label} reachable at {url}", flush=True)
            return True
        time.sleep(1.0)
    print(f"  {label} NOT reachable at {url} after {timeout:.0f}s", flush=True)
    return False


def start_backend() -> subprocess.Popen | None:
    """Start uvicorn for the FastAPI backend on BACKEND_PORT."""
    if not (BACKEND_DIR / "app" / "main.py").exists():
        print("  backend app/main.py not found — skipping backend", flush=True)
        return None
    env = os.environ.copy()
    env.setdefault("SECRET_KEY", "d70e4526f3a23417cf4b30afce4e5f2d827a9f33b56fc2ff665eeea2558ea1f1")
    env.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./dev.db")
    env.setdefault("TRADING_MODE", "paper")
    # Ensure the backend package is importable.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(BACKEND_DIR) + (os.pathsep + existing if existing else "")
    print("  launching uvicorn (app.main:app)…", flush=True)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "0.0.0.0", "--port", str(BACKEND_PORT), "--log-level", "warning"],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def build_frontend() -> bool:
    """Run `npm run build`. Returns True on success."""
    if (FRONTEND_DIR / "dist" / "index.html").exists():
        print("  frontend/dist already present — skipping build", flush=True)
        return True
    if not (FRONTEND_DIR / "package.json").exists():
        print("  frontend/package.json not found — cannot build", flush=True)
        return False
    print("  building frontend (npm run build)…", flush=True)
    proc = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(FRONTEND_DIR),
        env={**os.environ, "VITE_API_URL": ""},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("  npm run build FAILED:", flush=True)
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], flush=True)
        return False
    return True


def start_frontend() -> subprocess.Popen | None:
    """Serve the built frontend with `npm run preview -- --port 5173`."""
    if not (FRONTEND_DIR / "dist" / "index.html").exists():
        print("  frontend/dist/index.html missing — cannot preview", flush=True)
        return None
    print(f"  launching vite preview on :{FRONTEND_PORT}…", flush=True)
    return subprocess.Popen(
        ["npm", "run", "preview", "--", "--port", str(FRONTEND_PORT), "--host"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def drain(proc: subprocess.Popen | None, label: str, limit: int = 1500) -> None:
    """Print whatever a (likely dead) process emitted, for diagnosis."""
    if proc is None or proc.stdout is None:
        return
    try:
        out = proc.stdout.read() or b""
        if out:
            print(f"  --- {label} output ---", flush=True)
            print(out.decode(errors="replace")[-limit:], flush=True)
    except Exception:
        pass


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Screenshot capture + heuristics
# ─────────────────────────────────────────────────────────────────────────────

def analyze_blank(path: Path) -> dict:
    """Compute mean brightness + stdev of a grayscale version of the image."""
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return {"blank": False, "mean": None, "stdev": None, "note": "PIL missing"}
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
            stat = ImageStat.Stat(gray)
            mean = stat.mean[0]
            stdev = stat.stddev[0]
    except Exception as exc:
        return {"blank": False, "mean": None, "stdev": None, "note": f"read err: {exc}"}
    blank = stdev < BLANK_STDEV_THRESHOLD or (
        mean > NEAR_WHITE_MEAN_THRESHOLD and stdev < BLANK_STDEV_THRESHOLD * 2
    )
    return {"blank": blank, "mean": round(mean, 1), "stdev": round(stdev, 1), "note": ""}


def capture_all(results: list[dict]) -> bool:
    """Drive Playwright across every route. Returns True if anything captured."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("DIAGNOSIS: Playwright is not installed — "
              "`pip install playwright && python -m playwright install chromium`.", flush=True)
        return False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captured_any = False

    try:
        pw = sync_playwright().start()
    except Exception as exc:
        print(f"DIAGNOSIS: could not start Playwright: {exc}", flush=True)
        return False

    try:
        try:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        except Exception as exc:
            print("DIAGNOSIS: Chromium failed to launch — browsers likely not "
                  f"installed (`python -m playwright install --with-deps chromium`). {exc}",
                  flush=True)
            return False

        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        # Seed a token before any app script runs so RequireAuth lets us in.
        ctx.add_init_script(
            f"window.sessionStorage.setItem('access_token', '{SEED_TOKEN}');"
        )

        for label, path, is_public in ROUTES:
            entry = {
                "label": label, "path": path, "public": is_public,
                "console_errors": [], "page_errors": [], "http_errors": [],
                "screenshot": None, "blank": False, "mean": None, "stdev": None,
                "nav_error": None,
            }
            page = ctx.new_page()

            page.on("console", lambda msg, e=entry: (
                e["console_errors"].append(msg.text[:200])
                if msg.type == "error" else None
            ))
            page.on("pageerror", lambda exc, e=entry: e["page_errors"].append(str(exc)[:200]))

            def on_response(resp, e=entry):
                try:
                    if resp.status >= 400:
                        e["http_errors"].append(f"{resp.status} {resp.url[:120]}")
                except Exception:
                    pass
            page.on("response", on_response)

            url = FRONTEND_BASE + path
            try:
                page.goto(url, wait_until="networkidle", timeout=20_000)
            except Exception as exc:
                # networkidle can time out on apps with long-polling/WS; fall
                # back to a plain load and still try to screenshot.
                entry["nav_error"] = str(exc)[:160]
                try:
                    page.goto(url, wait_until="load", timeout=15_000)
                except Exception as exc2:
                    entry["nav_error"] = str(exc2)[:160]

            # Let lazy chunks / charts settle.
            page.wait_for_timeout(2500)

            shot = OUT_DIR / f"{label}.png"
            try:
                page.screenshot(path=str(shot), full_page=True)
                entry["screenshot"] = shot
                captured_any = True
                blank_info = analyze_blank(shot)
                entry.update({
                    "blank": blank_info["blank"],
                    "mean": blank_info["mean"],
                    "stdev": blank_info["stdev"],
                })
            except Exception as exc:
                entry["nav_error"] = (entry["nav_error"] or "") + f" | screenshot: {exc}"[:160]

            page.close()
            results.append(entry)
            flags = page_flags(entry)
            print(f"  [{label}] {path} → {'OK' if not flags else ', '.join(flags)}", flush=True)

        ctx.close()
        browser.close()
    finally:
        try:
            pw.stop()
        except Exception:
            pass

    return captured_any


def page_flags(entry: dict) -> list[str]:
    """Human-readable problem flags for a single page."""
    flags: list[str] = []
    if entry["screenshot"] is None:
        flags.append("NO SCREENSHOT")
    if entry["blank"]:
        flags.append(f"blank (stdev={entry['stdev']})")
    if entry["page_errors"]:
        flags.append(f"{len(entry['page_errors'])} JS error(s)")
    if entry["console_errors"]:
        flags.append(f"{len(entry['console_errors'])} console error(s)")
    if entry["http_errors"]:
        # Don't count expected auth 401s as a "break" on protected pages, but
        # still surface them.
        flags.append(f"{len(entry['http_errors'])} failed request(s)")
    if entry["nav_error"]:
        flags.append("nav error")
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Contact sheet
# ─────────────────────────────────────────────────────────────────────────────

def build_contact_sheet(results: list[dict], out_path: Path) -> bool:
    """Composite captured screenshots into a labeled grid PNG."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  PIL missing — cannot build contact sheet", flush=True)
        return False

    shots = [e for e in results if e["screenshot"] is not None]
    if not shots:
        return False

    cols = 3
    rows = (len(shots) + cols - 1) // cols
    thumb_w, thumb_h = 460, 290
    pad = 14
    label_h = 26
    cell_w = thumb_w + pad
    cell_h = thumb_h + label_h + pad
    header_h = 70

    sheet_w = cols * cell_w + pad
    sheet_h = header_h + rows * cell_h + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (11, 18, 32))
    draw = ImageDraw.Draw(sheet)

    def font(size: int):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_broken = sum(1 for e in results if page_flags(e))
    draw.text((pad, 14), "QuantEdge — App Screenshots", fill=(226, 232, 240), font=font(26))
    draw.text((pad, 46), f"{ts}   ·   {len(results)} pages   ·   {n_broken} flagged",
              fill=(148, 163, 184), font=font(14))

    for i, e in enumerate(shots):
        r, c = divmod(i, cols)
        x = pad + c * cell_w
        y = header_h + r * cell_h
        flags = page_flags(e)
        border = (220, 38, 38) if flags else (22, 163, 74)
        try:
            with Image.open(e["screenshot"]) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_w, thumb_h))
                # Center the thumbnail in its cell.
                tx = x + (thumb_w - im.width) // 2
                ty = y + label_h + (thumb_h - im.height) // 2
                draw.rectangle([x, y + label_h, x + thumb_w, y + label_h + thumb_h],
                               fill=(17, 17, 17))
                sheet.paste(im, (tx, ty))
        except Exception:
            draw.rectangle([x, y + label_h, x + thumb_w, y + label_h + thumb_h],
                           fill=(40, 10, 10))
        draw.rectangle([x - 2, y + label_h - 2, x + thumb_w + 2, y + label_h + thumb_h + 2],
                       outline=border, width=3)
        suffix = "  " + " · ".join(flags) if flags else "  OK"
        label_color = (255, 120, 120) if flags else (125, 211, 252)
        draw.text((x, y), f"{e['label']} ({e['path']}){suffix}"[:64],
                  fill=label_color, font=font(13))

    sheet.save(out_path)
    print(f"Contact sheet rendered: {out_path} ({out_path.stat().st_size} bytes)", flush=True)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Slack upload (mirrors employee_dashboard.py)
# ─────────────────────────────────────────────────────────────────────────────

def slack_call(token: str, method: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"https://slack.com/api/{method}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"http_{e.code}", "body": e.read().decode()[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def ensure_channel(token: str, name: str) -> str | None:
    """Find or create the channel, return its ID."""
    cursor = ""
    while True:
        payload = {"types": "public_channel", "limit": 200}
        if cursor:
            payload["cursor"] = cursor
        data = slack_call(token, "conversations.list", payload)
        if not data.get("ok"):
            break
        for ch in data.get("channels", []):
            if ch["name"] == name:
                return ch["id"]
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break
    created = slack_call(token, "conversations.create", {"name": name})
    if created.get("ok"):
        print(f"  created #{name}", flush=True)
        return created["channel"]["id"]
    print(f"  could not create #{name}: {created.get('error')}", flush=True)
    return None


def upload_image(token: str, channel_id: str, path: Path, title: str, comment: str) -> bool:
    """Upload via the modern files.getUploadURLExternal flow."""
    size = path.stat().st_size
    step1 = slack_call(token, "files.getUploadURLExternal",
                       {"filename": path.name, "length": size})
    if not step1.get("ok"):
        return _legacy_upload(token, channel_id, path, title, comment)
    upload_url = step1["upload_url"]
    file_id = step1["file_id"]
    try:
        req = urllib.request.Request(upload_url, data=path.read_bytes(), method="POST")
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as exc:
        print(f"  upload POST failed: {exc}", flush=True)
        return False
    done = slack_call(token, "files.completeUploadExternal", {
        "files": [{"id": file_id, "title": title}],
        "channel_id": channel_id,
        "initial_comment": comment,
    })
    if not done.get("ok"):
        print(f"  completeUpload failed: {done.get('error')}", flush=True)
        return False
    return True


def _legacy_upload(token, channel_id, path, title, comment) -> bool:
    import uuid
    boundary = uuid.uuid4().hex
    parts = []

    def add(name, value):
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(str(value).encode())
    add("channels", channel_id)
    add("title", title)
    add("initial_comment", comment)
    parts.append(f"--{boundary}".encode())
    parts.append(f'Content-Disposition: form-data; name="file"; filename="{path.name}"'.encode())
    parts.append(b"Content-Type: image/png")
    parts.append(b"")
    parts.append(path.read_bytes())
    parts.append(f"--{boundary}--".encode())
    body = b"\r\n".join(parts)
    req = urllib.request.Request("https://slack.com/api/files.upload", data=body,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        if not resp.get("ok"):
            print(f"  legacy upload failed: {resp.get('error')}", flush=True)
        return resp.get("ok", False)
    except Exception as exc:
        print(f"  legacy upload exception: {exc}", flush=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Analysis summary
# ─────────────────────────────────────────────────────────────────────────────

def build_comment(results: list[dict]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    repo = os.environ.get("GH_REPO", "")
    broken = [e for e in results if page_flags(e)]
    ok = [e for e in results if not page_flags(e)]

    lines = [
        f":camera_with_flash: *App Screenshots* — {ts}" + (f"  ·  `{repo}`" if repo else ""),
        f"{len(results)} pages captured   ·   :large_green_circle: {len(ok)} ok   "
        f":red_circle: {len(broken)} flagged",
    ]
    if broken:
        lines.append("")
        lines.append("*Pages that look broken:*")
        for e in broken:
            detail = []
            if e["screenshot"] is None:
                detail.append("no screenshot")
            if e["blank"]:
                detail.append(f"blank/near-empty (stdev {e['stdev']}, mean {e['mean']})")
            if e["page_errors"]:
                detail.append(f"{len(e['page_errors'])} JS error(s): {e['page_errors'][0]}")
            if e["console_errors"]:
                detail.append(f"{len(e['console_errors'])} console error(s)")
            if e["http_errors"]:
                detail.append(f"{len(e['http_errors'])} failed req(s): {e['http_errors'][0]}")
            if e["nav_error"]:
                detail.append(f"nav: {e['nav_error']}")
            lines.append(f"• *{e['label']}* (`{e['path']}`) — " + "; ".join(detail))
    else:
        lines.append("All captured pages rendered without blank screens, JS errors, "
                     "or failed requests. :tada:")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("QuantEdge app-screenshot bot starting…", flush=True)

    # 1. Build frontend.
    if not build_frontend():
        print("DIAGNOSIS: frontend build failed — cannot screenshot. Exiting cleanly.", flush=True)
        return 0

    backend = None
    frontend = None
    try:
        # 2. Start servers.
        backend = start_backend()
        frontend = start_frontend()

        be_up = wait_for_http(f"{BACKEND_BASE}/", timeout=60, label="backend")
        if not be_up:
            be_up = wait_for_http(f"{BACKEND_BASE}/docs", timeout=10, label="backend (/docs)")
        if not be_up:
            drain(backend, "backend")
            print("  NOTE: backend not reachable — pages will still be screenshotted; "
                  "API-driven sections may show errors (a legitimate 'broken' signal).",
                  flush=True)

        fe_up = wait_for_http(f"{FRONTEND_BASE}/login", timeout=60, label="frontend")
        if not fe_up:
            drain(frontend, "frontend")
            print("DIAGNOSIS: frontend preview server never came up — exiting cleanly.", flush=True)
            return 0

        # 3. Capture.
        results: list[dict] = []
        captured = capture_all(results)
        if not captured:
            print("DIAGNOSIS: no screenshots captured (Playwright/browser unavailable). "
                  "Exiting cleanly without fabricating images.", flush=True)
            return 0

    finally:
        terminate(frontend)
        terminate(backend)

    # 4. Contact sheet.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sheet_path = REPO_ROOT / f"app_screenshots_{sheet_ts}.png"
    have_sheet = build_contact_sheet(results, sheet_path)

    # 5. Console summary.
    broken = [e for e in results if page_flags(e)]
    print(f"\nSummary: {len(results)} pages, {len(broken)} flagged.", flush=True)
    for e in broken:
        print(f"  BROKEN {e['label']} ({e['path']}): {', '.join(page_flags(e))}", flush=True)

    # 6. Post to Slack.
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        print("⚠ No SLACK_BOT_TOKEN — screenshots saved locally but not posted.", flush=True)
        return 0

    ch_id = ensure_channel(token, CHANNEL)
    if not ch_id:
        print("⚠ Could not resolve Slack channel.", flush=True)
        return 0

    comment = build_comment(results)
    if have_sheet:
        ok = upload_image(token, ch_id, sheet_path, "QuantEdge App Screenshots", comment)
        print("✅ posted contact sheet" if ok else "❌ failed to post contact sheet", flush=True)
    else:
        # No sheet (PIL missing) — post the text analysis at least.
        posted = slack_call(token, "chat.postMessage", {"channel": ch_id, "text": comment})
        print("✅ posted text analysis" if posted.get("ok")
              else f"❌ failed to post: {posted.get('error')}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
