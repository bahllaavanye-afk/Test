"""
QuantEdge Employee Performance Dashboard.

Builds a single dashboard image that shows, for every "employee" (GitHub
workflow pipeline, trading desk, and background agent), whether they are
UP / DOWN / IDLE and how well they're performing. Posts the image to a
dedicated Slack channel (#employee-dashboard) via files.upload.

Employees & their health sources (all real, no mock data):
  • Pipelines  — GitHub Actions API: each workflow's latest run conclusion
  • Desks      — experiments/results/desk_runs.jsonl (orders + signals)
  • Algo agent — experiments/results/algo_agent_results.json
  • ICs/leads  — git log author activity (commits in last 7 days)

Required env:
  SLACK_BOT_TOKEN   xoxb-... with chat:write, files:write, channels:read/manage
  GH_TOKEN          GITHUB_TOKEN (read actions + contents)
  GH_REPO           owner/repo
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
DASH_CHANNEL = os.environ.get("DASHBOARD_CHANNEL", "employee-dashboard")

UP, DOWN, IDLE = "UP", "DOWN", "IDLE"
COLORS = {UP: "#16a34a", DOWN: "#dc2626", IDLE: "#a16207"}


# ─────────────────────────────────────────────────────────────────────────────
# GitHub API
# ─────────────────────────────────────────────────────────────────────────────

def gh_api(path: str) -> dict | list | None:
    token = os.environ.get("GH_TOKEN", "")
    repo = os.environ.get("GH_REPO", "")
    if not token or not repo:
        return None
    url = f"https://api.github.com/repos/{repo}{path}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as exc:
        print(f"  gh_api {path} failed: {exc}", flush=True)
        return None


def pipeline_health() -> list[dict]:
    """Each workflow's latest run conclusion → UP/DOWN/IDLE employee row."""
    data = gh_api("/actions/workflows")
    rows: list[dict] = []
    if not isinstance(data, dict):
        return rows
    for wf in data.get("workflows", []):
        wf_id = wf.get("id")
        name = wf.get("name", "?")
        state = wf.get("state", "")
        runs = gh_api(f"/actions/workflows/{wf_id}/runs?per_page=5")
        latest = None
        if isinstance(runs, dict) and runs.get("workflow_runs"):
            latest = runs["workflow_runs"][0]
        if state != "active":
            status, detail, score = IDLE, "disabled", 0
        elif latest is None:
            status, detail, score = IDLE, "never run", 0
        else:
            concl = latest.get("conclusion")
            created = latest.get("created_at", "")
            # recency
            try:
                age_h = (datetime.now(timezone.utc) -
                         datetime.fromisoformat(created.replace("Z", "+00:00"))).total_seconds() / 3600
            except Exception:
                age_h = 999
            if concl == "success":
                status, score = UP, 100
                detail = f"ok {age_h:.0f}h ago"
            elif concl in ("failure", "startup_failure", "timed_out"):
                status, score = DOWN, 0
                detail = f"{concl} {age_h:.0f}h ago"
            elif concl is None:
                status, score, detail = UP, 80, "running"
            else:
                status, score, detail = IDLE, 40, str(concl)
            # Recent success rate across last 5 runs
            if isinstance(runs, dict):
                last5 = runs.get("workflow_runs", [])[:5]
                concls = [r.get("conclusion") for r in last5 if r.get("conclusion")]
                if concls:
                    ok = sum(1 for c in concls if c == "success")
                    score = int(100 * ok / len(concls))
                    if ok == 0:
                        status = DOWN
                    elif ok < len(concls):
                        status = IDLE if status != DOWN else DOWN
        rows.append({
            "group": "Pipeline", "name": name[:34], "status": status,
            "detail": detail, "score": score,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Desk health (from persisted desk runs)
# ─────────────────────────────────────────────────────────────────────────────

DESKS = ["Equities", "Crypto", "Options", "Polymarket", "Macro/FX", "StatArb"]


def desk_health() -> list[dict]:
    path = REPO_ROOT / "experiments" / "results" / "desk_runs.jsonl"
    rows: list[dict] = []
    runs: list[dict] = []
    if path.exists():
        for line in path.read_text().splitlines()[-200:]:
            try:
                runs.append(json.loads(line))
            except Exception:
                continue
    # last run recency
    last_ts = None
    if runs:
        try:
            last_ts = datetime.fromisoformat(runs[-1]["ts"])
        except Exception:
            last_ts = None
    age_h = ((datetime.now(timezone.utc) - last_ts).total_seconds() / 3600) if last_ts else 999

    # aggregate per desk over recent runs
    per_desk: dict[str, dict] = {d: {"signals": 0, "orders": 0} for d in DESKS}
    for r in runs[-20:]:
        for s in r.get("signals", []):
            d = s.get("desk")
            if d in per_desk:
                per_desk[d]["signals"] += 1
        for o in r.get("orders", []):
            d = o.get("desk")
            if d in per_desk:
                per_desk[d]["orders"] += 1
        for ps in r.get("poly_signals", []):
            if "Polymarket" in per_desk:
                per_desk["Polymarket"]["signals"] += 1

    for d in DESKS:
        st = per_desk[d]
        if not runs or age_h > 6:
            status, detail, score = DOWN, "no recent run", 0
        elif st["signals"] == 0 and st["orders"] == 0:
            status, detail, score = IDLE, "no signals", 50
        else:
            status = UP
            detail = f"{st['signals']} sig, {st['orders']} ord"
            score = min(100, 60 + st["orders"] * 5 + st["signals"] * 2)
        rows.append({"group": "Desk", "name": d, "status": status,
                     "detail": detail, "score": score})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# IC / lead health (git commit activity, last 7 days)
# ─────────────────────────────────────────────────────────────────────────────

def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode()
    except Exception:
        return ""


def contributor_health() -> list[dict]:
    raw = sh(["git", "log", "--since=7 days ago", "--pretty=format:%an"])
    counts: dict[str, int] = {}
    for line in raw.splitlines():
        a = line.strip()
        if a:
            counts[a] = counts.get(a, 0) + 1
    rows = []
    for author, n in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
        status = UP if n >= 3 else (IDLE if n >= 1 else DOWN)
        rows.append({"group": "Contributor", "name": author[:30],
                     "status": status, "detail": f"{n} commits/7d",
                     "score": min(100, n * 12)})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Algo agent health
# ─────────────────────────────────────────────────────────────────────────────

def algo_agent_health() -> list[dict]:
    path = REPO_ROOT / "experiments" / "results" / "algo_agent_results.json"
    if not path.exists():
        return [{"group": "Agent", "name": "AlgoAgent", "status": DOWN,
                 "detail": "no results", "score": 0}]
    try:
        data = json.loads(path.read_text())
    except Exception:
        return [{"group": "Agent", "name": "AlgoAgent", "status": DOWN,
                 "detail": "unreadable", "score": 0}]
    if not data:
        return [{"group": "Agent", "name": "AlgoAgent", "status": IDLE,
                 "detail": "empty", "score": 30}]
    last = data[-1]
    sharpes = [r.get("sharpe", 0) for r in data[-50:] if isinstance(r.get("sharpe"), (int, float))]
    best = max(sharpes) if sharpes else 0
    n = len(data)
    status = UP if best > 0 else IDLE
    return [{"group": "Agent", "name": "AlgoAgent (UCB1)", "status": status,
             "detail": f"{n} runs, best Sharpe {best:.2f}", "score": min(100, 50 + int(best * 25))}]


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(rows: list[dict], out_path: Path) -> dict:
    n_up = sum(1 for r in rows if r["status"] == UP)
    n_down = sum(1 for r in rows if r["status"] == DOWN)
    n_idle = sum(1 for r in rows if r["status"] == IDLE)

    # Sort: DOWN first (most urgent), then IDLE, then UP; within, by group
    order = {DOWN: 0, IDLE: 1, UP: 2}
    rows = sorted(rows, key=lambda r: (order[r["status"]], r["group"], -r["score"]))

    n = len(rows)
    fig_h = max(6, 0.42 * n + 2.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, n + 3)
    ax.axis("off")

    # Header
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ax.text(0.1, n + 2.3, "QuantEdge — Employee Performance Dashboard",
            fontsize=20, fontweight="bold", color="#e2e8f0")
    ax.text(0.1, n + 1.6, f"{ts}   ·   {len(rows)} employees   ·   "
            f"🟢 {n_up} up   🔴 {n_down} down   🟡 {n_idle} idle",
            fontsize=12, color="#94a3b8")

    # Column headers
    ax.text(0.1, n + 0.7, "STATUS", fontsize=10, fontweight="bold", color="#64748b")
    ax.text(1.7, n + 0.7, "EMPLOYEE", fontsize=10, fontweight="bold", color="#64748b")
    ax.text(6.0, n + 0.7, "ROLE", fontsize=10, fontweight="bold", color="#64748b")
    ax.text(7.6, n + 0.7, "DETAIL", fontsize=10, fontweight="bold", color="#64748b")
    ax.text(10.6, n + 0.7, "SCORE", fontsize=10, fontweight="bold", color="#64748b")

    for i, r in enumerate(rows):
        y = n - i - 0.3
        c = COLORS[r["status"]]
        # status pill
        ax.add_patch(FancyBboxPatch((0.1, y - 0.18), 1.3, 0.42,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     facecolor=c, edgecolor="none", alpha=0.9))
        ax.text(0.75, y, r["status"], fontsize=9, fontweight="bold",
                color="white", ha="center", va="center")
        # name / group / detail
        ax.text(1.7, y, r["name"], fontsize=11, color="#e2e8f0", va="center")
        ax.text(6.0, y, r["group"], fontsize=9, color="#7dd3fc", va="center")
        ax.text(7.6, y, r["detail"][:34], fontsize=9, color="#cbd5e1", va="center")
        # score bar
        score = r["score"]
        ax.add_patch(plt.Rectangle((10.6, y - 0.12), 1.2, 0.24,
                     facecolor="#1e293b", edgecolor="none"))
        ax.add_patch(plt.Rectangle((10.6, y - 0.12), 1.2 * score / 100, 0.24,
                     facecolor=c, edgecolor="none"))
        ax.text(11.85, y, f"{score}", fontsize=8, color="#e2e8f0", va="center", ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {"up": n_up, "down": n_down, "idle": n_idle, "total": len(rows),
            "down_names": [r["name"] for r in rows if r["status"] == DOWN],
            "top": [(r["name"], r["score"]) for r in rows if r["status"] == UP][:5]}


# ─────────────────────────────────────────────────────────────────────────────
# Slack upload
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
    """Find or create the dashboard channel, return its ID."""
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
        # fall back to legacy multipart files.upload
        return _legacy_upload(token, channel_id, path, title, comment)
    upload_url = step1["upload_url"]
    file_id = step1["file_id"]
    # POST the raw bytes
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
    import mimetypes, uuid
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def gather_rows() -> list[dict]:
    rows: list[dict] = []
    print("Gathering pipeline health…", flush=True)
    rows += pipeline_health()
    print("Gathering desk health…", flush=True)
    rows += desk_health()
    print("Gathering algo-agent health…", flush=True)
    rows += algo_agent_health()
    print("Gathering contributor health…", flush=True)
    rows += contributor_health()
    return rows


def main() -> int:
    rows = gather_rows()
    if not rows:
        print("No employee data gathered.", flush=True)
        return 0

    out = REPO_ROOT / "employee_dashboard.png"
    summary = render_dashboard(rows, out)
    print(f"Dashboard rendered: {out} ({out.stat().st_size} bytes)", flush=True)
    print(f"  UP={summary['up']} DOWN={summary['down']} IDLE={summary['idle']}", flush=True)
    if summary["down_names"]:
        print(f"  DOWN: {', '.join(summary['down_names'])}", flush=True)

    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        print("⚠ No SLACK_BOT_TOKEN — dashboard saved locally but not posted.", flush=True)
        return 0

    ch_id = ensure_channel(token, DASH_CHANNEL)
    if not ch_id:
        print("⚠ Could not resolve dashboard channel.", flush=True)
        return 0

    down = summary["down_names"]
    top = summary["top"]
    comment_lines = [
        f":bar_chart: *Employee Performance Dashboard* — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"🟢 *{summary['up']}* up   🔴 *{summary['down']}* down   🟡 *{summary['idle']}* idle   (of {summary['total']})",
    ]
    if down:
        comment_lines.append(f"🔴 *Needs attention:* {', '.join(down[:8])}")
    if top:
        comment_lines.append("🏆 *Top performers:* " + ", ".join(f"{n} ({s})" for n, s in top))
    comment = "\n".join(comment_lines)

    ok = upload_image(token, ch_id, out, "Employee Performance Dashboard", comment)
    print("✅ posted dashboard" if ok else "❌ failed to post dashboard", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
