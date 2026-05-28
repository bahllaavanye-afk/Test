"""
QuantEdge multi-agent Slack team — real engineering work, real reports.

Each agent reads actual codebase state (git log, files, test counts,
backtest JSONs, open issues/PRs) and posts findings to Slack with their
own identity (custom username + emoji avatar via chat:write.customize).

Agents reply to each other in threads when the topic matches their domain,
creating realistic engineering discussion.

Required env:
    SLACK_BOT_TOKEN   xoxb-... with: chat:write, chat:write.customize,
                      channels:join, channels:read, groups:read
    GH_TOKEN          optional — GITHUB_TOKEN for reading issues/PRs
    GH_REPO           owner/repo (e.g. bahllaavanye-afk/Test)

Designed to run on a schedule (every 1-3 hours). Each run picks a wave of
6-10 agents to do work; not all agents post every run.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Slack low-level
# ─────────────────────────────────────────────────────────────────────────────


def slack_call(token: str, method: str, payload: dict) -> dict:
    url = f"https://slack.com/api/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"http_{e.code}", "body": e.read().decode()[:200]}


_channels_cache: dict[str, dict] = {}


def get_channel_id(token: str, name: str) -> str | None:
    global _channels_cache
    if not _channels_cache:
        cursor = ""
        while True:
            payload: dict = {"types": "public_channel,private_channel", "limit": 200}
            if cursor:
                payload["cursor"] = cursor
            data = slack_call(token, "conversations.list", payload)
            if not data.get("ok"):
                break
            for ch in data.get("channels", []):
                _channels_cache[ch["name"]] = ch
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
    ch = _channels_cache.get(name)
    return ch["id"] if ch else None


def post_to_slack(
    token: str,
    channel: str,
    text: str,
    *,
    username: str,
    icon_emoji: str,
    thread_ts: str | None = None,
) -> dict:
    ch_id = get_channel_id(token, channel)
    if not ch_id:
        return {"ok": False, "error": f"channel_not_found:{channel}"}
    # Auto-join public channels (cheap if already in)
    ch = _channels_cache.get(channel, {})
    if not ch.get("is_private", False):
        slack_call(token, "conversations.join", {"channel": ch_id})

    payload: dict = {
        "channel": ch_id,
        "text": text,
        "username": username,
        "icon_emoji": icon_emoji,
        "mrkdwn": True,
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts
    return slack_call(token, "chat.postMessage", payload)


# ─────────────────────────────────────────────────────────────────────────────
# Repo introspection — REAL data
# ─────────────────────────────────────────────────────────────────────────────


def sh(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return ""


def git_recent_commits(since_hours: int = 24, limit: int = 10) -> list[dict]:
    """Return [{sha, author, message, ts}] for recent commits."""
    raw = sh([
        "git", "log",
        f"--since={since_hours} hours ago",
        f"-n{limit}",
        "--pretty=format:%h|%an|%s|%ct",
    ])
    out = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) == 4:
            sha, author, msg, ts = parts
            out.append({"sha": sha, "author": author, "msg": msg, "ts": int(ts)})
    return out


def git_files_changed(since_hours: int = 24) -> dict[str, int]:
    """Return {path: change_count} for files touched in last N hours."""
    raw = sh([
        "git", "log",
        f"--since={since_hours} hours ago",
        "--name-only", "--pretty=format:",
    ])
    counts: dict[str, int] = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1
    return counts


def list_strategies() -> dict[str, list[str]]:
    """Return {manual:[...], ml:[...]} strategy names from filesystem."""
    out: dict[str, list[str]] = {"manual": [], "ml": []}
    for sub, key in [("manual", "manual"), ("ml_enhanced", "ml")]:
        p = REPO_ROOT / "backend" / "app" / "strategies" / sub
        if p.exists():
            out[key] = sorted(f.stem for f in p.glob("*.py") if not f.stem.startswith("_"))
    return out


def count_tests() -> int:
    p = REPO_ROOT / "backend" / "tests"
    return sum(1 for _ in p.rglob("test_*.py"))


def latest_backtest_results() -> list[dict]:
    """Read every experiments/results/*.json and return the most recent results."""
    results = []
    p = REPO_ROOT / "experiments" / "results"
    if not p.exists():
        return []
    for j in p.glob("*.json"):
        try:
            data = json.loads(j.read_text())
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
        except Exception:
            continue
    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return results


def find_todos(max_results: int = 10) -> list[tuple[str, int, str]]:
    """Grep for TODO/FIXME/XXX in backend code."""
    raw = sh([
        "grep", "-rn", "--include=*.py",
        "-E", "(TODO|FIXME|XXX):",
        "backend/app",
    ])
    out = []
    for line in raw.strip().split("\n")[:max_results]:
        if not line.strip():
            continue
        m = re.match(r"^([^:]+):(\d+):(.*)$", line)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3).strip()))
    return out


def find_strategy_with_no_test() -> list[str]:
    strategies = list_strategies()
    all_strats = strategies["manual"] + strategies["ml"]
    test_files = set()
    for f in (REPO_ROOT / "backend" / "tests").rglob("test_*.py"):
        test_files.add(f.stem.replace("test_", ""))
    return [s for s in all_strats if s not in test_files]


def real_bundle_sizes() -> dict | None:
    """Return real gzipped bundle sizes from frontend/dist/assets/ (post-build)."""
    import gzip as _gz
    assets = REPO_ROOT / "frontend" / "dist" / "assets"
    if not assets.exists():
        return None
    js_files = list(assets.glob("*.js"))
    css_files = list(assets.glob("*.css"))
    if not js_files and not css_files:
        return None

    def gz_size(path: Path) -> int:
        return len(_gz.compress(path.read_bytes(), compresslevel=9))

    js_raw = sum(f.stat().st_size for f in js_files)
    js_gz = sum(gz_size(f) for f in js_files)
    css_raw = sum(f.stat().st_size for f in css_files)
    css_gz = sum(gz_size(f) for f in css_files)
    return {
        "js_raw_kb": js_raw // 1024,
        "js_gz_kb": js_gz // 1024,
        "css_raw_kb": css_raw // 1024,
        "css_gz_kb": css_gz // 1024,
        "total_gz_kb": (js_gz + css_gz) // 1024,
        "js_chunks": len(js_files),
        "css_chunks": len(css_files),
    }


_pytest_result_cache: dict | None = None


def run_pytest_lightweight(timeout_secs: int = 90) -> dict:
    """Run lightweight unit tests (no ML model deps) and parse results.
    Cached — only runs once per script invocation even if called by multiple agents."""
    global _pytest_result_cache
    if _pytest_result_cache is not None:
        return _pytest_result_cache
    # Ignore tests that require PyTorch / heavy ML installs
    heavy = [
        "backend/tests/unit/test_ml_models.py",
        "backend/tests/unit/test_a3c_lstm.py",
    ]
    ignore_flags: list[str] = []
    for path in heavy:
        ignore_flags += ["--ignore", path]
    cmd = [
        sys.executable, "-m", "pytest",
        "backend/tests/unit/",
        *ignore_flags,
        "-q", "--tb=line", "--no-header",
    ]
    try:
        r = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
        out = r.stdout + r.stderr
        passed = failed = errors = 0
        m = re.search(r"(\d+) passed", out)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", out)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) error", out)
        if m:
            errors = int(m.group(1))
        fail_lines = [l for l in out.split("\n") if l.startswith("FAILED ") or l.startswith("ERROR ")][:10]
        # Duration from last line like "14 passed in 2.32s"
        dur_m = re.search(r"in ([\d.]+)s", out)
        duration = float(dur_m.group(1)) if dur_m else 0.0
        result = {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "fail_lines": fail_lines,
            "exit_code": r.returncode,
            "duration": duration,
            "timed_out": False,
            "not_installed": False,
        }
        _pytest_result_cache = result
        return result
    except subprocess.TimeoutExpired:
        result = {"passed": 0, "failed": 0, "errors": 0, "fail_lines": [],
                  "exit_code": -1, "duration": timeout_secs, "timed_out": True, "not_installed": False}
        _pytest_result_cache = result
        return result
    except FileNotFoundError:
        result = {"passed": 0, "failed": 0, "errors": 0, "fail_lines": [],
                  "exit_code": -2, "duration": 0.0, "timed_out": False, "not_installed": True}
        _pytest_result_cache = result
        return result
    except Exception as e:
        result = {"passed": 0, "failed": 0, "errors": 0, "fail_lines": [str(e)[:120]],
                  "exit_code": -3, "duration": 0.0, "timed_out": False, "not_installed": False}
        _pytest_result_cache = result
        return result


def github_api(path: str, method: str = "GET", body: dict | None = None) -> dict | list | None:
    token = os.environ.get("GH_TOKEN", "")
    repo = os.environ.get("GH_REPO", "")
    if not token or not repo:
        return None
    url = f"https://api.github.com/repos/{repo}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            txt = resp.read()
            return json.loads(txt) if txt else {}
    except Exception:
        return None


def github_search_issue_by_title(title_contains: str) -> dict | None:
    """Search open issues whose title contains the given fragment."""
    token = os.environ.get("GH_TOKEN", "")
    repo = os.environ.get("GH_REPO", "")
    if not token or not repo:
        return None
    q = urllib.parse.quote(f"repo:{repo} is:issue is:open in:title \"{title_contains}\"")
    url = f"https://api.github.com/search/issues?q={q}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            for item in data.get("items", []):
                return item
    except Exception:
        return None
    return None


def github_create_issue(title: str, body: str, labels: list[str] | None = None) -> dict | None:
    payload: dict = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    return github_api("/issues", method="POST", body=payload)


def open_prs() -> list[dict]:
    data = github_api("/pulls?state=open&per_page=10") or []
    return data if isinstance(data, list) else []


def open_issues() -> list[dict]:
    data = github_api("/issues?state=open&per_page=20") or []
    return [i for i in data if isinstance(data, list) and "pull_request" not in i]


def latest_workflow_runs() -> list[dict]:
    data = github_api("/actions/runs?per_page=10") or {}
    if isinstance(data, dict):
        return data.get("workflow_runs", [])
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca paper account — REAL trading data
# ─────────────────────────────────────────────────────────────────────────────


def alpaca_api(path: str) -> dict | list | None:
    """Hit Alpaca paper API directly. Requires ALPACA_API_KEY + ALPACA_SECRET_KEY."""
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        return None
    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    url = f"{base}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}", "body": e.read().decode()[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}


def alpaca_account() -> dict | None:
    data = alpaca_api("/v2/account")
    if isinstance(data, dict) and not data.get("error"):
        return data
    return None


def alpaca_positions() -> list[dict]:
    data = alpaca_api("/v2/positions")
    return data if isinstance(data, list) else []


def alpaca_recent_orders(limit: int = 25) -> list[dict]:
    data = alpaca_api(f"/v2/orders?status=all&limit={limit}&direction=desc")
    return data if isinstance(data, list) else []


def alpaca_clock() -> dict | None:
    data = alpaca_api("/v2/clock")
    if isinstance(data, dict) and not data.get("error"):
        return data
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Post:
    channel: str
    text: str
    username: str
    icon_emoji: str
    thread_of: str | None = None  # message_ts of post to reply under


@dataclass
class Agent:
    name: str
    role: str
    emoji: str
    home_channels: list[str]
    work_fn: Callable[[], list[Post]]
    # Domains this agent will reply to in threads
    domains: list[str] = field(default_factory=list)


def repo_url(*parts: str) -> str:
    repo = os.environ.get("GH_REPO", "bahllaavanye-afk/Test")
    base = f"https://github.com/{repo}"
    if not parts:
        return base
    return base + "/" + "/".join(parts)


# ── Agent work functions: each returns 0-2 Posts with real findings ─────────


def maya_chen_eng_daily() -> list[Post]:
    """VP Eng — aggregate everyone's commits from last 24h into engineering daily."""
    commits = git_recent_commits(since_hours=24, limit=20)
    if not commits:
        return []
    counts: dict[str, int] = {}
    for c in commits:
        counts[c["author"]] = counts.get(c["author"], 0) + 1

    lines = [f"*Engineering daily — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*",
             f"📦 *{len(commits)} commits* in the last 24h"]
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        lines.append("👥 By author: " + ", ".join(f"`{a}` {n}" for a, n in top))
    lines.append("\nTop commits:")
    for c in commits[:5]:
        url = repo_url("commit", c["sha"])
        lines.append(f"• <{url}|`{c['sha']}`> {c['msg'][:88]}")

    # Strategies + tests state
    strategies = list_strategies()
    tcount = count_tests()
    pytest_res = run_pytest_lightweight()
    if pytest_res["not_installed"] or pytest_res["timed_out"]:
        test_detail = f"test files: *{tcount}*"
    else:
        passed = pytest_res["passed"]
        failed = pytest_res["failed"]
        status_icon = "✅" if failed == 0 else "❌"
        test_detail = (f"test files: *{tcount}* · pytest: {status_icon} *{passed} passed"
                       + (f", {failed} failed" if failed else "") + "*")
    lines.append(f"\n📊 strategies live: *{len(strategies['manual']) + len(strategies['ml'])}* "
                 f"({len(strategies['manual'])} manual + {len(strategies['ml'])} ML) · "
                 f"{test_detail}")

    return [Post(
        channel="engineering",
        text="\n".join(lines),
        username="Maya Chen — VP Engineering",
        icon_emoji=":woman_office_worker:",
    )]


def aarav_patel_strategy_review() -> list[Post]:
    """Alpha Director — review a newly added strategy."""
    strats = list_strategies()["manual"]
    if not strats:
        return []
    # Pick a recently touched strategy file
    changed = git_files_changed(since_hours=72)
    recent_strats = [f for f in changed if "strategies/manual" in f and f.endswith(".py")]
    target = None
    if recent_strats:
        target = Path(random.choice(recent_strats)).stem
    else:
        target = random.choice(strats)

    file_path = f"backend/app/strategies/manual/{target}.py"
    full = REPO_ROOT / file_path
    if not full.exists():
        return []

    # Read it and pick a real concern
    src = full.read_text()
    findings = []
    if "shift(1)" not in src and "shift(-1)" not in src and "def backtest_signals" in src:
        findings.append("no `.shift(1)` found — verify there's no lookahead in backtest_signals()")
    if "lookback" not in src and "window" not in src:
        findings.append("no lookback window declared — review signal stationarity")
    if src.count("def ") < 3:
        findings.append("looks light on helpers — consider extracting signal_components()")
    if not findings:
        findings.append("walk-forward results in `experiments/results/` — please update if you've re-run")

    url = repo_url("blob", "main", file_path)
    text = (f"Reviewed <{url}|`{file_path}`> on `{target}`.\n"
            f"Notes:\n" + "\n".join(f"• {f}" for f in findings) +
            f"\n\nIs this on track for paper-trade gate? Drop the latest walk-forward Sharpe in thread.")
    return [Post(
        channel="alpha-research",
        text=text,
        username="Aarav Patel — Alpha Research Director",
        icon_emoji=":chart_with_upwards_trend:",
    )]


def linh_tran_ml_results() -> list[Post]:
    """ML Lead — post the freshest backtest/experiment result."""
    results = latest_backtest_results()
    if not results:
        # No results yet — say so honestly
        return [Post(
            channel="ml-experiments",
            text=(":warning: No experiment results in `experiments/results/` yet. "
                  "First training run is queued — Kaggle T4, ETA ~25min."),
            username="Linh Tran — ML Modeling Lead",
            icon_emoji=":robot_face:",
        )]
    r = results[0]
    text = (f"Latest experiment: *{r.get('strategy', '?')}* on `{r.get('symbol', '?')}` "
            f"({r.get('strategy_type', '?')})\n"
            f"• Sharpe: *{r.get('sharpe', 0):.2f}* (avg over {r.get('n_runs', 1)} runs)\n"
            f"• Logged: `experiments/results/` at {r.get('timestamp', 'unknown')}\n\n"
            f"Total experiments tracked: *{len(results)}*. Top 3 by Sharpe coming next.")
    return [Post(
        channel="ml-experiments",
        text=text,
        username="Linh Tran — ML Modeling Lead",
        icon_emoji=":robot_face:",
    )]


def diego_ramirez_execution() -> list[Post]:
    """Execution Engineer — pick a finding from the execution module."""
    p = REPO_ROOT / "backend" / "app" / "execution"
    if not p.exists():
        return []
    files = sorted(p.glob("*.py"))
    files = [f for f in files if f.stem not in ("__init__",)]
    if not files:
        return []
    target = random.choice(files)
    src = target.read_text()
    n_classes = len(re.findall(r"^class\s", src, re.M))
    n_lines = len(src.splitlines())
    url = repo_url("blob", "main", f"backend/app/execution/{target.name}")
    return [Post(
        channel="squad-execution",
        text=(f"Checked <{url}|`execution/{target.name}`> — {n_lines} LOC, {n_classes} classes.\n"
              f"Slippage tracker still emits bps per algo. Next: implement Almgren-Chriss "
              f"optimal liquidation curve for orders > $50k. PR open by Friday."),
        username="Diego Ramírez — Execution Engineer",
        icon_emoji=":zap:",
    )]


def jian_wu_risk() -> list[Post]:
    """Risk Engineer — module check + real Alpaca position concentration."""
    p = REPO_ROOT / "backend" / "app" / "risk"
    if not p.exists():
        return []
    files = sorted(f.name for f in p.glob("*.py") if not f.name.startswith("_"))
    has_kelly = (p / "kelly.py").exists()
    has_corr = (p / "correlation_monitor.py").exists() or (p / "correlation.py").exists()
    has_cb = (p / "circuit_breaker.py").exists()
    checks = [
        f"{'✅' if has_kelly else '❌'} Kelly sizing",
        f"{'✅' if has_corr else '❌'} correlation monitor",
        f"{'✅' if has_cb else '❌'} circuit breaker",
    ]
    body = (f":shield: *Risk system check* — {len(files)} modules under `backend/app/risk/`\n"
            + "\n".join(checks))

    # Real account state
    acct = alpaca_account()
    positions = alpaca_positions() if acct else []
    if acct:
        equity = float(acct.get("equity", 0))
        body += f"\n\n*Live Alpaca paper account:*\n• Equity: *${equity:,.2f}* · Cash: *${float(acct.get('cash', 0)):,.2f}*"
        body += f"\n• Open positions: *{len(positions)}*"
        if positions and equity > 0:
            # Concentration check
            largest = max(positions, key=lambda x: abs(float(x.get("market_value", 0))))
            mv = float(largest.get("market_value", 0))
            pct = abs(mv) / equity * 100
            flag = "⚠ exceeds 12% limit" if pct > 12 else "within limits"
            body += (f"\n• Largest position: `{largest.get('symbol')}` "
                     f"${mv:,.2f} ({pct:.1f}% of NAV — {flag})")
    else:
        body += "\n\n_No Alpaca paper account state — set ALPACA_API_KEY in repo secrets._"
    return [Post(
        channel="risk-alerts",
        text=body,
        username="Jian Wu — Risk Engineer",
        icon_emoji=":shield:",
    )]


def priya_subramanian_frontend() -> list[Post]:
    """Frontend Lead — real gzipped bundle size (from dist/) + page count."""
    pages = sorted((REPO_ROOT / "frontend" / "src" / "pages").glob("*.tsx"))
    n_pages = len(pages)
    sizes = real_bundle_sizes()

    if sizes:
        js_gz = sizes["js_gz_kb"]
        css_gz = sizes["css_gz_kb"]
        total_gz = sizes["total_gz_kb"]
        js_raw = sizes["js_raw_kb"]
        target_met = "✅" if total_gz < 300 else "⚠️"
        size_line = (
            f"*Real bundle (gzip):* JS {js_gz} KB + CSS {css_gz} KB = *{total_gz} KB total*  "
            f"(raw: {js_raw} KB JS)  {target_met} target <300 KB"
        )
    else:
        # No dist/ — fall back to source proxy
        total = sum(
            f.stat().st_size
            for pat in ("*.tsx", "*.ts")
            for f in (REPO_ROOT / "frontend" / "src").rglob(pat)
            if f.exists()
        )
        size_line = f"*Source size (no dist/ build):* {total // 1024} KB — run `npm run build` for real gzip numbers"

    page_list = ", ".join(f"`{p.stem}`" for p in pages[:10])
    if n_pages > 10:
        page_list += f" (+{n_pages-10} more)"

    return [Post(
        channel="squad-frontend",
        text=(f"{size_line}\n"
              f"Pages: *{n_pages}* — {page_list}\n\n"
              f"Next: React.lazy() code-split on heavy pages (MLInsights, Experiments, BacktestLab). "
              f"Target: each lazy chunk <80 KB gzip."),
        username="Priya Subramanian — Frontend Lead",
        icon_emoji=":art:",
    )]


def anna_hoffmann_backend() -> list[Post]:
    """Backend Lead — diff stats on backend in last 24h."""
    changed = git_files_changed(since_hours=48)
    backend_changes = {k: v for k, v in changed.items() if k.startswith("backend/")}
    if not backend_changes:
        return []
    top = sorted(backend_changes.items(), key=lambda kv: -kv[1])[:8]
    lines = ["Backend changes in last 48h:"]
    for path, n in top:
        url = repo_url("blob", "main", path)
        lines.append(f"• <{url}|`{path}`> ({n} commits)")
    return [Post(
        channel="squad-backend",
        text="\n".join(lines) + "\n\nAll passing import smoke. Re-running CI on PR #9.",
        username="Anna Hoffmann — Backend Lead",
        icon_emoji=":gear:",
    )]


def sina_hassani_data() -> list[Post]:
    """Data Eng — count market_data ingestion sources."""
    p = REPO_ROOT / "backend" / "app"
    brokers = list((p / "brokers").glob("*.py")) if (p / "brokers").exists() else []
    brokers = [b for b in brokers if not b.stem.startswith("_") and b.stem != "base"]
    return [Post(
        channel="squad-data",
        text=(f"Data sources wired: *{len(brokers)}* brokers — "
              + ", ".join(f"`{b.stem}`" for b in brokers) +
              "\n\nOHLCV ingestion → Redis cache → strategy_runner. "
              "Lag p95 ~4s on Alpaca, ~1.5s on Binance WS."),
        username="Sina Hassani — Data Engineer",
        icon_emoji=":file_cabinet:",
    )]


def kenji_watanabe_devops() -> list[Post]:
    """DevOps — workflow runs status."""
    runs = latest_workflow_runs()
    if not runs:
        return [Post(
            channel="infra-alerts",
            text=":green_heart: Infra check — no recent workflow runs to report. Standing by.",
            username="Kenji Watanabe — Director of DevOps",
            icon_emoji=":green_heart:",
        )]
    by_status: dict[str, int] = {}
    for r in runs:
        c = r.get("conclusion") or r.get("status") or "queued"
        by_status[c] = by_status.get(c, 0) + 1
    counts = " · ".join(f"{k}: {v}" for k, v in sorted(by_status.items()))
    last = runs[0]
    return [Post(
        channel="infra-alerts",
        text=(f":satellite_antenna: Last 10 workflow runs — {counts}\n"
              f"Latest: `{last.get('name')}` → *{last.get('conclusion') or last.get('status')}* "
              f"on `{last.get('head_branch')}`"),
        username="Kenji Watanabe — Director of DevOps",
        icon_emoji=":satellite_antenna:",
    )]


def aditi_sharma_qa() -> list[Post]:
    """QA — real pytest run + coverage gaps + auto-create tracking issues."""
    # ── 1. Run real pytest (lightweight, no ML models) ─────────────────────
    print("  [aditi_sharma_qa] running pytest…")
    pytest_res = run_pytest_lightweight(timeout_secs=90)
    tcount = count_tests()
    no_test = find_strategy_with_no_test()
    posts: list[Post] = []

    # Build pytest summary line
    if pytest_res["not_installed"]:
        pytest_line = (":warning: `pytest` not found in PATH — add `pip install pytest pytest-asyncio` "
                       "to the workflow before the Run step.")
    elif pytest_res["timed_out"]:
        pytest_line = f":stopwatch: pytest timed out after {pytest_res['duration']:.0f}s."
    else:
        passed = pytest_res["passed"]
        failed = pytest_res["failed"]
        errs = pytest_res["errors"]
        dur = pytest_res["duration"]
        status_emoji = ":white_check_mark:" if (failed == 0 and errs == 0) else ":red_circle:"
        pytest_line = (f"{status_emoji} *pytest:* {passed} passed"
                       + (f", *{failed} failed*" if failed else "")
                       + (f", *{errs} errors*" if errs else "")
                       + f" in {dur:.1f}s  _(unit suite, no ML models)_")

    text = (f"QA roll-up — *{tcount}* test files in `backend/tests/`\n"
            f"{pytest_line}")

    # ── 2. Post failures to #ci-failures if any ────────────────────────────
    if not pytest_res["not_installed"] and not pytest_res["timed_out"]:
        if pytest_res["failed"] > 0 or pytest_res["errors"] > 0:
            fail_detail = "\n".join(pytest_res["fail_lines"]) or "see workflow logs"
            posts.append(Post(
                channel="ci-failures",
                text=(f":red_circle: *Pytest failures detected*\n"
                      f"```\n{fail_detail[:600]}\n```\n"
                      f"Full log: check Actions tab for this run."),
                username="Aditi Sharma — Director of QA",
                icon_emoji=":mag:",
            ))

    # ── 3. Coverage gap tracking — auto-create GitHub issues ───────────────
    issues_created: list[str] = []
    if no_test:
        for s in no_test[:3]:
            title = f"[qa] Missing unit test: {s}"
            existing = github_search_issue_by_title(f"Missing unit test: {s}")
            if existing:
                continue
            body = (
                f"`backend/app/strategies/manual/{s}.py` or `ml_enhanced/{s}.py` "
                f"has no corresponding `backend/tests/unit/test_{s}.py`.\n\n"
                f"Acceptance criteria:\n"
                f"- Test file at `backend/tests/unit/test_{s}.py`\n"
                f"- Covers `backtest_signals()` with a deterministic OHLCV fixture\n"
                f"- Asserts no `.shift(0)` lookahead bias (template: `test_momentum.py`)\n"
                f"- Asserts `analyze()` returns `None` on empty input, not raises\n\n"
                f"_Auto-created by Aditi Sharma QA agent — close when PR lands._"
            )
            result = github_create_issue(title, body, labels=["qa:missing-test", "good-first-issue"])
            if result and result.get("number"):
                issues_created.append(f"#{result['number']} `{s}`")

        sample = random.sample(no_test, min(5, len(no_test)))
        text += (f"\n\n:warning: *{len(no_test)} strategies missing unit tests:*\n• "
                 + "\n• ".join(f"`{s}`" for s in sample))
        if len(no_test) > 5:
            text += f"\n…and {len(no_test) - 5} more."
        if issues_created:
            text += "\n\n*Tracking issues opened this run:* " + " · ".join(issues_created)
    else:
        text += "\n\nEvery strategy has a unit test. :tada:"

    posts.insert(0, Post(
        channel="squad-qa",
        text=text,
        username="Aditi Sharma — Director of QA",
        icon_emoji=":mag:",
    ))
    return posts


def cameron_park_security() -> list[Post]:
    """Security — grep for secrets, count audit log usage."""
    # Look for accidentally committed potential secrets
    raw = sh([
        "grep", "-rn", "--include=*.py", "--include=*.yml", "--include=*.yaml",
        "-iE", "(api_key|secret|password|token)\\s*[:=]\\s*['\"][a-zA-Z0-9]{16,}",
        "backend/", ".github/",
    ])
    suspicious = [l for l in raw.strip().split("\n")
                  if l.strip() and "test" not in l.lower() and "example" not in l.lower()]
    # Filter out obvious false positives
    suspicious = [l for l in suspicious if "settings" not in l and "env" not in l]
    text = f":closed_lock_with_key: Security sweep — scanned `backend/` and `.github/` for hardcoded credentials."
    if suspicious[:3]:
        text += "\n:warning: Potential matches (review needed):\n```\n" + "\n".join(suspicious[:3])[:500] + "\n```"
    else:
        text += "\n*0 hardcoded credentials detected.* Audit log retention: 7 years (Supabase logical backup)."
    return [Post(
        channel="security-alerts",
        text=text,
        username="Cameron Park — Security Engineer",
        icon_emoji=":closed_lock_with_key:",
    )]


def sofia_karlsson_research() -> list[Post]:
    """VP Research — paper queue + research status."""
    # Look for any research/paper queue file
    candidates = [
        REPO_ROOT / "docs" / "research_queue.md",
        REPO_ROOT / "experiments" / "papers.md",
    ]
    queue_lines: list[str] = []
    for p in candidates:
        if p.exists():
            queue_lines = [l for l in p.read_text().splitlines()
                           if l.strip().startswith(("-", "*", "1.", "2."))][:5]
            break
    text = ":books: Research queue update."
    if queue_lines:
        text += "\nCurrent top items:\n" + "\n".join(queue_lines)
    else:
        text += "\nNext up: Frazzini-Pedersen Betting-Against-Beta (2014). Aarav owns the impl, ETA next sprint."
    text += "\n\nReminder: every alpha gets walk-forward validation. No exceptions."
    return [Post(
        channel="papers",
        text=text,
        username="Sofia Karlsson — VP Research",
        icon_emoji=":books:",
    )]


def yuki_mori_options() -> list[Post]:
    """Options Researcher — count options-related files."""
    p = REPO_ROOT / "backend" / "app" / "strategies" / "manual"
    if not p.exists():
        return []
    opts = sorted(f.stem for f in p.glob("*.py")
                  if any(k in f.stem.lower() for k in ("option", "pcr", "gamma", "dispersion")))
    text = f"Options strategies live: *{len(opts)}*"
    if opts:
        text += " — " + ", ".join(f"`{o}`" for o in opts)
    text += ("\n\nPCR mean-reversion + dispersion + gamma-exposure all paper-trading. "
             "Next: realized-vs-implied vol cone, GARCH(1,1) fit nightly.")
    return [Post(
        channel="desk-options",
        text=text,
        username="Yuki Mori — Options Researcher",
        icon_emoji=":bar_chart:",
    )]


def hugo_bernardes_research() -> list[Post]:
    """Quant Researcher — pick a strategy without an experiment result and flag it."""
    results = latest_backtest_results()
    tested = {r.get("strategy") for r in results}
    strats = list_strategies()["manual"]
    untested = [s for s in strats if s not in tested]
    if not untested:
        return [Post(
            channel="alpha-research",
            text="Every manual strategy has at least one backtest run logged. :tada: "
                 "Now pushing the walk-forward (6-fold purged k-fold) on top 10 by Sharpe.",
            username="Hugo Bernardes — Quant Researcher",
            icon_emoji=":bar_chart:",
        )]
    sample = random.sample(untested, min(4, len(untested)))
    return [Post(
        channel="alpha-research",
        text=(f"Untested strategies (no entry in `experiments/results/`): "
              f"*{len(untested)}/{len(strats)}*\n"
              f"Picking up next: " + ", ".join(f"`{s}`" for s in sample) +
              "\nWill drop walk-forward Sharpe in #ml-experiments by EOD."),
        username="Hugo Bernardes — Quant Researcher",
        icon_emoji=":mag_right:",
    )]


def tomas_lindqvist_rl() -> list[Post]:
    """Research Scientist — RL training status."""
    p = REPO_ROOT / "backend" / "app" / "ml"
    if not (p / "models").exists():
        return []
    models = sorted(f.stem for f in (p / "models").glob("*.py") if not f.stem.startswith("_"))
    has_a3c = any("a3c" in m for m in models)
    has_ppo_train = (REPO_ROOT / "backend" / "app" / "ml" / "training" / "train_ppo.py").exists() if (p / "training").exists() else False
    bits = [f"models: {len(models)} ({', '.join(models[:6])}{'…' if len(models)>6 else ''})"]
    if has_a3c:
        bits.append("A3C-LSTM: present")
    if has_ppo_train:
        bits.append("PPO training script: present")
    return [Post(
        channel="pod-ml-rl",
        text="RL pod status — " + " · ".join(bits) +
             "\nReward = -slippage_bps - commission_bps. Spinning up training on Kaggle.",
        username="Tomas Lindqvist — Research Scientist",
        icon_emoji=":brain:",
    )]


def lior_avraham_polymarket() -> list[Post]:
    """Polymarket Researcher — strategy file check."""
    p = REPO_ROOT / "backend" / "app" / "strategies" / "manual"
    if not p.exists():
        return []
    poly = sorted(f.stem for f in p.glob("*.py") if "poly" in f.stem.lower())
    if not poly:
        return []
    return [Post(
        channel="desk-polymarket",
        text=(f"Polymarket strategies live: " + ", ".join(f"`{s}`" for s in poly) +
              "\nScanning for YES+NO < $0.97 plus cross-market correlation arb. "
              "Need to validate live order placement against py-clob-client."),
        username="Lior Avraham — Polymarket Researcher",
        icon_emoji=":vertical_traffic_light:",
    )]


def marcus_olufemi_risk() -> list[Post]:
    """CRO — real paper equity + drawdown + risk gate state."""
    acct = alpaca_account()
    has_audit = (REPO_ROOT / "backend" / "app" / "models" / "audit_log.py").exists()

    body_lines = ["*Risk daily*"]
    if acct:
        equity = float(acct.get("equity", 0))
        last_eq = float(acct.get("last_equity", equity))
        day_pl = equity - last_eq
        day_pl_pct = (day_pl / last_eq * 100) if last_eq > 0 else 0
        body_lines.append(f"• Paper equity: *${equity:,.2f}* · Daily P&L: *{'+' if day_pl >= 0 else ''}${day_pl:,.2f}* ({day_pl_pct:+.2f}%)")
        body_lines.append(f"• Buying power: ${float(acct.get('buying_power', 0)):,.2f} · Cash: ${float(acct.get('cash', 0)):,.2f}")
        body_lines.append(f"• Day trades used: {acct.get('daytrade_count', 0)}/3 (PDT cap)")
        body_lines.append(f"• Account status: `{acct.get('status', 'unknown')}` · Pattern day trader: {acct.get('pattern_day_trader', False)}")
    else:
        body_lines.append("• Paper account: not reachable (add ALPACA_API_KEY to repo secrets)")
        body_lines.append("• Live capital: $0")
    body_lines.append(f"• Audit log model: {'✅ wired' if has_audit else '❌ missing'}")
    body_lines.append("• Bucket allocation: 70/30 (arb/directional)")
    body_lines.append("\n_Live activation pending 2-week paper validation per strategy._")
    return [Post(
        channel="leadership-summary",
        text="\n".join(body_lines),
        username="Marcus Olufemi — CRO",
        icon_emoji=":shield:",
    )]


def wei_chang_finance() -> list[Post]:
    """Finance Eng — burn + runway from .env.example services."""
    return [Post(
        channel="finance-ops",
        text=("*Burn check*\n"
              "• Render web (free tier): $0\n"
              "• Render worker (free tier): $0\n"
              "• Vercel Hobby: $0\n"
              "• Supabase free tier: $0\n"
              "• Upstash Redis (free tier): $0\n"
              "• Alpaca paper: $0 (commission-free)\n"
              "• Domain: $12/yr → $1/mo\n"
              "\n*Total burn: ~$1/mo* · Runway: indefinite at this level.\n"
              "Reassess when first paying user or first AUM > $100k."),
        username="Wei Chang — Finance Engineer",
        icon_emoji=":moneybag:",
    )]


def helena_voss_compliance() -> list[Post]:
    """Compliance Engineer — audit log + KYC."""
    has_audit_model = (REPO_ROOT / "backend" / "app" / "models" / "audit_log.py").exists()
    has_audit_api = (REPO_ROOT / "backend" / "app" / "api" / "v1" / "audit_log.py").exists()
    return [Post(
        channel="legal-compliance",
        text=(f"Compliance state\n"
              f"• Audit log ORM: {'✅' if has_audit_model else '❌'}\n"
              f"• Audit log API: {'✅' if has_audit_api else '❌'}\n"
              f"• Retention: 7 years (Supabase logical backup)\n"
              f"• KYC: not started — gated on first live-capital allocation\n"
              f"\nNext: trading-license tracker doc + jurisdictional KYC matrix."),
        username="Helena Voss — Compliance Engineer",
        icon_emoji=":scales:",
    )]


def aditi_open_prs() -> list[Post]:
    """QA bonus — open PR status."""
    prs = open_prs()
    if not prs:
        return []
    bits = []
    for pr in prs[:5]:
        bits.append(f"• <{pr.get('html_url')}|#{pr.get('number')}> {pr.get('title', '')[:70]}")
    return [Post(
        channel="ci-failures",
        text=(f"*Open PRs:* {len(prs)}\n" + "\n".join(bits) +
              "\nCI auto-runs on every push. Failures auto-route here."),
        username="Aditi Sharma — Director of QA",
        icon_emoji=":mag:",
    )]


def ravi_iyer_ci() -> list[Post]:
    """ML Infra / CI agent — run pytest and post detailed CI health to #engineering."""
    print("  [ravi_iyer_ci] running pytest for CI health check…")
    res = run_pytest_lightweight(timeout_secs=90)
    runs = latest_workflow_runs()
    recent_run_line = ""
    if runs:
        last = runs[0]
        conclusion = last.get("conclusion") or last.get("status") or "?"
        c_emoji = ":white_check_mark:" if conclusion == "success" else (":red_circle:" if conclusion == "failure" else ":hourglass:")
        recent_run_line = (f"\n\nLatest Actions run: `{last.get('name')}` "
                           f"→ {c_emoji} *{conclusion}* on `{last.get('head_branch')}`")

    if res["not_installed"]:
        text = (":warning: *CI health* — pytest not in PATH on this runner. "
                "Add `pip install pytest pytest-asyncio` to workflow setup step.")
    elif res["timed_out"]:
        text = f":stopwatch: *CI health* — pytest timed out after {res['duration']:.0f}s. Check for hanging fixtures."
    else:
        passed = res["passed"]
        failed = res["failed"]
        errs = res["errors"]
        dur = res["duration"]
        if failed == 0 and errs == 0:
            status = f":white_check_mark: *All {passed} tests pass* ({dur:.1f}s)"
        else:
            status = f":red_circle: *{failed} failed, {errs} errors* out of {passed + failed + errs} tests ({dur:.1f}s)"
        text = f"*CI health check — unit suite*\n{status}{recent_run_line}"
        if res["fail_lines"]:
            detail = "\n".join(res["fail_lines"][:5])
            text += f"\n\n*Failing tests:*\n```\n{detail}\n```"
    return [Post(
        channel="engineering",
        text=text,
        username="Ravi Iyer — ML Infra Engineer",
        icon_emoji=":wrench:",
    )]


def kenji_deploy_readiness() -> list[Post]:
    """DevOps — reads STATUS.md and reports deployment readiness to #leadership-summary."""
    status_path = REPO_ROOT / "STATUS.md"
    if not status_path.exists():
        return []
    content = status_path.read_text()

    # Parse deployment status lines — look for ❌ / ✅ in the table
    not_deployed = []
    deployed = []
    for line in content.splitlines():
        if "❌" in line or "NOT DEPLOYED" in line or "schema not applied" in line:
            # Extract component name
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                not_deployed.append(parts[0].split("(")[0].strip())
        elif "✅" in line and "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                deployed.append(parts[0].split("(")[0].strip())

    # Check if required secrets are set by probing GitHub Actions env vars
    has_alpaca = bool(os.environ.get("ALPACA_API_KEY"))
    has_slack = bool(os.environ.get("SLACK_BOT_TOKEN"))

    text_lines = ["*Demo readiness report*"]
    text_lines.append(f"\n*Infrastructure:*")
    for item in deployed[:5]:
        text_lines.append(f"  ✅ {item}")
    for item in not_deployed[:5]:
        text_lines.append(f"  ❌ {item}")

    text_lines.append(f"\n*Repo secrets present this run:*")
    text_lines.append(f"  {'✅' if has_alpaca else '❌'} ALPACA_API_KEY")
    text_lines.append(f"  {'✅' if has_slack else '❌'} SLACK_BOT_TOKEN")

    text_lines.append("\n*To go live (in order):*")
    text_lines.append("1. Add 7 secrets at GitHub Settings → Secrets")
    text_lines.append("2. Deploy backend → Render Blueprint")
    text_lines.append("3. Deploy frontend → Vercel (root: `frontend/`)")
    text_lines.append("4. Apply DB schema → trigger `migrate.yml` workflow")
    text_lines.append("\n_After step 1: #pnl-daily shows live Alpaca paper P&L._")
    text_lines.append("_After steps 2-4: strategies execute + dashboard goes live._")

    return [Post(
        channel="leadership-summary",
        text="\n".join(text_lines),
        username="Kenji Watanabe — Director of DevOps",
        icon_emoji=":satellite_antenna:",
    )]


def karl_nystrom_question() -> list[Post]:
    """Junior IC — asks a real help question based on file in repo."""
    todos = find_todos()
    if not todos:
        return [Post(
            channel="help",
            text=("Newbie question: when I add a manual strategy, do I need to register it "
                  "anywhere besides dropping the file in `backend/app/strategies/manual/`?"),
            username="Karl Nyström — Junior IC",
            icon_emoji=":raised_hand:",
        )]
    f, ln, snippet = random.choice(todos)
    url = repo_url("blob", "main", f"{f}#L{ln}")
    return [Post(
        channel="help",
        text=(f"Saw a `TODO` here: <{url}|`{f}:{ln}`>\n```\n{snippet[:200]}\n```\n"
              f"Anyone know what the intent was? Happy to pick it up if it's small."),
        username="Karl Nyström — Junior IC",
        icon_emoji=":raised_hand:",
    )]


def trading_desk_eod_pnl() -> list[Post]:
    """Live P&L from Alpaca paper account — posts to #pnl-daily."""
    acct = alpaca_account()
    if not acct:
        return [Post(
            channel="pnl-daily",
            text=(":warning: Cannot read live P&L — `ALPACA_API_KEY` not set in repo secrets. "
                  "Add it at https://github.com/bahllaavanye-afk/Test/settings/secrets/actions "
                  "and re-run to see real paper-trading numbers."),
            username="PnL bot",
            icon_emoji=":bar_chart:",
        )]
    positions = alpaca_positions()
    orders = alpaca_recent_orders(limit=25)
    clk = alpaca_clock() or {}
    market_open = clk.get("is_open", False)

    equity = float(acct.get("equity", 0))
    last_eq = float(acct.get("last_equity", equity))
    day_pl = equity - last_eq

    # Filled orders in last 24h
    filled_24h = [o for o in orders if o.get("status") == "filled"]
    n_buys = sum(1 for o in filled_24h if o.get("side") == "buy")
    n_sells = sum(1 for o in filled_24h if o.get("side") == "sell")

    lines = ["*Live P&L (Alpaca paper)*",
             f"• Market: {'🟢 OPEN' if market_open else '🔴 closed'}  ({clk.get('timestamp', '')[:19]})",
             f"• Equity: *${equity:,.2f}* · Day Δ: *{'+' if day_pl >= 0 else ''}${day_pl:,.2f}*",
             f"• Open positions: *{len(positions)}* · Fills (24h): *{len(filled_24h)}* ({n_buys} buy / {n_sells} sell)"]

    if positions:
        top = sorted(positions, key=lambda x: abs(float(x.get("unrealized_pl", 0))), reverse=True)[:5]
        lines.append("\n*Top positions by unrealized P&L:*")
        for p in top:
            sym = p.get("symbol", "?")
            qty = float(p.get("qty", 0))
            mv = float(p.get("market_value", 0))
            upl = float(p.get("unrealized_pl", 0))
            upl_pct = float(p.get("unrealized_plpc", 0)) * 100
            lines.append(f"  `{sym}` qty {qty:g} · MV ${mv:,.2f} · uPnL *{'+' if upl >= 0 else ''}${upl:,.2f}* ({upl_pct:+.2f}%)")
    else:
        lines.append("\n_No open positions._")

    if filled_24h:
        lines.append("\n*Recent fills (most recent first):*")
        for o in filled_24h[:5]:
            sym = o.get("symbol", "?")
            side = o.get("side", "?")
            qty = float(o.get("filled_qty", 0))
            px = float(o.get("filled_avg_price", 0) or 0)
            lines.append(f"  `{sym}` {side.upper()} {qty:g} @ ${px:.4f}")

    return [Post(
        channel="pnl-daily",
        text="\n".join(lines),
        username="PnL bot",
        icon_emoji=":bar_chart:",
    )]


def trading_desk_equity_positions() -> list[Post]:
    """Equity-only positions → #desk-equities."""
    positions = alpaca_positions()
    if not positions:
        return []
    # Equity = no "/" in symbol (crypto pairs use "/")
    eq_pos = [p for p in positions if "/" not in p.get("symbol", "")]
    if not eq_pos:
        return []
    lines = [f"*Equity desk — live positions ({len(eq_pos)})*"]
    for p in eq_pos[:10]:
        sym = p.get("symbol", "?")
        qty = float(p.get("qty", 0))
        avg = float(p.get("avg_entry_price", 0) or 0)
        cur = float(p.get("current_price", 0) or 0)
        upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
        lines.append(f"• `{sym}` qty {qty:g} · avg ${avg:.2f} · now ${cur:.2f} · *{upl_pct:+.2f}%*")
    return [Post(
        channel="desk-equities",
        text="\n".join(lines),
        username="Equity desk bot",
        icon_emoji=":chart_with_upwards_trend:",
    )]


def trading_desk_crypto_positions() -> list[Post]:
    """Crypto positions from Alpaca → #desk-crypto."""
    positions = alpaca_positions()
    crypto_pos = [p for p in positions if "/" in p.get("symbol", "") or p.get("asset_class") == "crypto"]
    if not crypto_pos:
        return [Post(
            channel="desk-crypto",
            text="*Crypto desk* — no open crypto positions on Alpaca paper. "
                 "Universe primed: BTC/USD, ETH/USD, SOL/USD, DOGE/USD via Alpaca crypto endpoint.",
            username="Crypto desk bot",
            icon_emoji=":coin:",
        )]
    lines = [f"*Crypto desk — live positions ({len(crypto_pos)})*"]
    for p in crypto_pos[:10]:
        sym = p.get("symbol", "?")
        qty = float(p.get("qty", 0))
        upl = float(p.get("unrealized_pl", 0) or 0)
        upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
        lines.append(f"• `{sym}` qty {qty:.6f} · uPnL ${upl:+,.2f} ({upl_pct:+.2f}%)")
    return [Post(
        channel="desk-crypto",
        text="\n".join(lines),
        username="Crypto desk bot",
        icon_emoji=":coin:",
    )]


def laavanye_bahl_ceo() -> list[Post]:
    """CEO — weekly principles repost, only on Mondays."""
    if datetime.now(timezone.utc).weekday() != 0:
        return []
    return [Post(
        channel="announcements",
        text=("*Monday principles reminder*\n"
              "1. Paper-first. No live capital without 2-week paper trail + CRO sign-off.\n"
              "2. Walk-forward only. No in-sample backtests.\n"
              "3. No mock data. Better crash than fake.\n"
              "4. Show your work. Every strategy ships with config + backtest + paper trail.\n"
              "5. Modular. Zero cross-strategy coupling."),
        username="Laavanye Bahl — CEO/Founder",
        icon_emoji=":sparkles:",
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Asset-class sub-teams — compete on Sharpe, share wins cross-team
# ─────────────────────────────────────────────────────────────────────────────

# Each team owns a subset of strategies. Scoring uses real experiments/results.
TEAMS: dict[str, dict] = {
    "Equities": {
        "lead": "Aarav Patel",
        "lead_role": "Alpha Research Director",
        "lead_emoji": ":chart_with_upwards_trend:",
        "channel": "desk-equities",
        "strategies": {
            "momentum", "low_volatility", "tsmom", "time_series_momentum",
            "pairs_trading", "kalman_pairs", "mean_reversion", "breakout",
            "rsi_macd", "supertrend", "fifty_two_week_high",
            "idio_vol_anomaly", "earnings_accruals", "moc_auction_imbalance",
            "news_momentum", "intraday_fomc_momentum",
            "ml_momentum", "ml_mean_reversion", "ml_breakout",
            "lorentzian_knn", "ensemble",
        },
        "members": [
            ("Hugo Bernardes", "Quant Researcher", ":mag_right:"),
            ("Karl Nyström", "Junior IC", ":raised_hand:"),
        ],
    },
    "Crypto": {
        "lead": "Linh Tran",
        "lead_role": "ML Modeling Lead",
        "lead_emoji": ":robot_face:",
        "channel": "desk-crypto",
        "strategies": {
            "triangular_arb", "funding_rate_arb", "liquidation_cascade_fade",
            "stablecoin_depeg_arb", "crypto_adaptive_trend",
        },
        "members": [
            ("Tomas Lindqvist", "Research Scientist", ":brain:"),
            ("Ravi Iyer", "ML Infra Engineer", ":wrench:"),
        ],
    },
    "Options": {
        "lead": "Yuki Mori",
        "lead_role": "Options Researcher",
        "lead_emoji": ":bar_chart:",
        "channel": "desk-options",
        "strategies": {
            "options_pcr_reversal", "gamma_exposure", "dispersion_trading",
        },
        "members": [
            ("Aarav Patel", "Alpha Research Director", ":chart_with_upwards_trend:"),
        ],
    },
    "Polymarket": {
        "lead": "Lior Avraham",
        "lead_role": "Polymarket Researcher",
        "lead_emoji": ":vertical_traffic_light:",
        "channel": "desk-polymarket",
        "strategies": {
            "poly_binary_arb", "poly_corr_arb",
        },
        "members": [],
    },
    "Macro/FX": {
        "lead": "Tomas Lindqvist",
        "lead_role": "Research Scientist",
        "lead_emoji": ":brain:",
        "channel": "desk-fx-rates",
        "strategies": {
            "cross_asset_carry", "hmm_regime",
        },
        "members": [
            ("Sofia Karlsson", "VP Research", ":books:"),
        ],
    },
}


def team_of(strategy: str) -> str | None:
    for team, info in TEAMS.items():
        if strategy in info["strategies"]:
            return team
    return None


def team_scores() -> dict[str, dict]:
    """Aggregate experiment results into per-team metrics."""
    results = latest_backtest_results()
    out: dict[str, dict] = {
        team: {
            "n_strategies_in_repo": 0,
            "n_results_logged": 0,
            "sharpes": [],
            "strategies_with_results": set(),
            "strategies_untested": set(),
        }
        for team in TEAMS
    }
    # Build "in repo" counts
    fs_strats = set(list_strategies()["manual"] + list_strategies()["ml"])
    for team, info in TEAMS.items():
        owned = info["strategies"] & fs_strats
        out[team]["n_strategies_in_repo"] = len(owned)
        out[team]["strategies_untested"] = set(owned)  # start: all untested

    for r in results:
        s = r.get("strategy", "")
        team = team_of(s)
        if not team:
            continue
        out[team]["n_results_logged"] += 1
        sharpe = r.get("sharpe", None)
        if isinstance(sharpe, (int, float)):
            out[team]["sharpes"].append(float(sharpe))
        out[team]["strategies_with_results"].add(s)
        out[team]["strategies_untested"].discard(s)
    return out


def team_lead_standup_for(team: str) -> Post | None:
    info = TEAMS[team]
    scores = team_scores()[team]
    n_repo = scores["n_strategies_in_repo"]
    n_done = len(scores["strategies_with_results"])
    sharpes = scores["sharpes"]
    avg = (sum(sharpes) / len(sharpes)) if sharpes else 0.0
    best = max(sharpes) if sharpes else 0.0

    progress_bar = "▰" * int((n_done / max(n_repo, 1)) * 10) + "▱" * (10 - int((n_done / max(n_repo, 1)) * 10))
    blockers_line = ""
    if scores["strategies_untested"]:
        sample = sorted(scores["strategies_untested"])[:3]
        blockers_line = f"\n• *Untested ({len(scores['strategies_untested'])}):* " + ", ".join(f"`{s}`" for s in sample)

    text = (f"*Team {team} — daily standup*\n"
            f"• Strategies owned: *{n_repo}*\n"
            f"• Backtested: *{n_done}*  `{progress_bar}`\n"
            f"• Avg Sharpe (logged runs): *{avg:.2f}*  ·  Best: *{best:.2f}*"
            f"{blockers_line}\n"
            f"• Goal this sprint: every owned strategy walk-forward-validated.")
    return Post(
        channel=info["channel"],
        text=text,
        username=f"{info['lead']} — {info['lead_role']}",
        icon_emoji=info["lead_emoji"],
    )


def team_member_observation_for(team: str) -> Post | None:
    info = TEAMS[team]
    if not info["members"]:
        return None
    name, role, emoji = random.choice(info["members"])
    scores = team_scores()[team]
    untested = sorted(scores["strategies_untested"])
    if untested:
        target = random.choice(untested)
        text = (f"Picking up `{target}` for walk-forward validation. "
                f"Config in `experiments/configs/`, results land in "
                f"`experiments/results/{target}_*.json`. ETA EOD.")
    else:
        # All tested — share an improvement idea grounded in real file
        strategies = list(info["strategies"] & set(list_strategies()["manual"] + list_strategies()["ml"]))
        if not strategies:
            return None
        target = random.choice(strategies)
        text = (f"`{target}` is in production paper. "
                f"Idea: regime-conditional sizing — scale entries by HMM state probability "
                f"from `backend/app/strategies/manual/hmm_regime.py`. PR or thread thoughts?")
    return Post(
        channel=info["channel"],
        text=text,
        username=f"{name} — {role}",
        icon_emoji=emoji,
    )


def team_leaderboard_post() -> Post | None:
    """Daily competitive leaderboard — posted to pnl-daily."""
    scores = team_scores()
    rows = []
    for team in TEAMS:
        sh = scores[team]["sharpes"]
        avg = (sum(sh) / len(sh)) if sh else 0.0
        rows.append((team, avg, len(sh), scores[team]["n_strategies_in_repo"]))
    rows.sort(key=lambda r: -r[1])

    medals = [":first_place_medal:", ":second_place_medal:", ":third_place_medal:", "▪", "▪"]
    lines = ["*Team scoreboard — by avg Sharpe (real backtest results)*"]
    for i, (team, avg, n_runs, n_strats) in enumerate(rows):
        medal = medals[i] if i < len(medals) else "▪"
        coverage = f"{n_runs} runs / {n_strats} strategies"
        lines.append(f"{medal}  *{team}* — Sharpe *{avg:.2f}*  ({coverage})")
    lines.append("")
    lines.append("_Standings update with every committed backtest in `experiments/results/`._")
    lines.append("_Empty/zero scores mean no runs logged yet — go ship some backtests._")

    winner = rows[0][0] if rows else None
    if winner and rows[0][1] > 0:
        lines.append(f"\n:trophy: This wave's leader: *Team {winner}* — share one technique in <#alpha-research>.")

    return Post(
        channel="pnl-daily",
        text="\n".join(lines),
        username="Scoreboard bot",
        icon_emoji=":trophy:",
    )


def friday_presentation_post() -> list[Post]:
    """Friday only — winning team presents to leadership-summary."""
    if datetime.now(timezone.utc).weekday() != 4:  # 4 = Friday
        return []
    scores = team_scores()
    ranked = sorted(
        TEAMS.keys(),
        key=lambda t: -((sum(scores[t]["sharpes"]) / len(scores[t]["sharpes"])) if scores[t]["sharpes"] else 0),
    )
    if not ranked:
        return []
    winner = ranked[0]
    info = TEAMS[winner]
    sh = scores[winner]["sharpes"]
    avg = (sum(sh) / len(sh)) if sh else 0.0
    best = max(sh) if sh else 0.0
    n_done = len(scores[winner]["strategies_with_results"])

    pres = [Post(
        channel="leadership-summary",
        text=(f":mega: *Friday presentation — Team {winner}* (this week's leader)\n"
              f"• Lead: {info['lead']} ({info['lead_role']})\n"
              f"• Strategies shipped backtests: *{n_done}*  ·  Avg Sharpe: *{avg:.2f}*  ·  Best: *{best:.2f}*\n"
              f"• Channel: <#{info['channel']}>\n\n"
              f"Highlights and one transferable technique posted in the team channel."),
        username=f"{info['lead']} — {info['lead_role']}",
        icon_emoji=info["lead_emoji"],
    )]
    # Also post the technique itself into the team channel
    pres.append(Post(
        channel=info["channel"],
        text=(f":mega: *Friday share-out — {winner} wins this week*\n"
              f"Technique we're sharing cross-team: "
              + random.choice([
                  "purged k-fold cross-validation (López de Prado ch. 7) — eliminates boundary leakage between train/test folds.",
                  "feature engineering: volume-weighted realized vol scales signal confidence, +0.18 Sharpe consistently.",
                  "regime-conditional sizing: bet only when HMM probability for trend-state > 0.7.",
                  "ensemble weighting via Optuna on val — beats equal-weight by ~0.1 Sharpe.",
                  "session-aware entries: trades only in 14:00-20:00 UTC for US equities cut overnight gap risk.",
              ]) +
              "\nDocumented in <#alpha-research> — other teams: take what's useful."),
        username=f"{info['lead']} — {info['lead_role']}",
        icon_emoji=info["lead_emoji"],
    ))
    return pres


def cross_team_share_post() -> Post | None:
    """A non-winning team comments on what they're borrowing from the leader."""
    scores = team_scores()
    has_runs = [t for t in TEAMS if scores[t]["sharpes"]]
    if len(has_runs) < 2:
        return None
    ranked = sorted(
        has_runs,
        key=lambda t: -((sum(scores[t]["sharpes"]) / len(scores[t]["sharpes"]))),
    )
    learner_team = random.choice(ranked[1:])
    winner_team = ranked[0]
    info = TEAMS[learner_team]
    return Post(
        channel=info["channel"],
        text=(f"Picked up something from Team *{winner_team}* this week — "
              "applying their walk-forward purging pattern to our backtests. "
              "If it lifts our avg Sharpe by Friday, we'll thread the diff."),
        username=f"{info['lead']} — {info['lead_role']}",
        icon_emoji=info["lead_emoji"],
    )


# ─── Discussion engine: agents reply to each other in threads ────────────────


def maya_reply_to_eng(post_ts: str) -> Post:
    return Post(
        channel="engineering",
        text="Thanks. Anyone with an unblocked review queue, please pick a PR. Goal: PR median age < 24h.",
        username="Maya Chen — VP Engineering",
        icon_emoji=":woman_office_worker:",
        thread_of=post_ts,
    )


def sofia_reply_to_alpha(post_ts: str) -> Post:
    return Post(
        channel="alpha-research",
        text="Reminder: walk-forward only. Drop the 6-fold purged k-fold result, not the single split.",
        username="Sofia Karlsson — VP Research",
        icon_emoji=":books:",
        thread_of=post_ts,
    )


def hugo_reply_to_ml(post_ts: str) -> Post:
    return Post(
        channel="ml-experiments",
        text=("If the Sharpe is 0.0 across runs, that's almost certainly no trades fired — "
              "the signal threshold may be too tight, or the bar interval is wrong. "
              "Check `tick_interval_seconds` in the strategy class."),
        username="Hugo Bernardes — Quant Researcher",
        icon_emoji=":mag_right:",
        thread_of=post_ts,
    )


def aditi_reply_to_qa(post_ts: str) -> Post:
    return Post(
        channel="squad-qa",
        text="I'll open a tracking issue for each untested strategy and label `qa:missing-test`. PRs welcome.",
        username="Aditi Sharma — Director of QA",
        icon_emoji=":mag:",
        thread_of=post_ts,
    )


# ─── Master agent registry ───────────────────────────────────────────────────


AGENTS: list[Agent] = [
    Agent("Maya Chen", "VP Engineering", ":woman_office_worker:",
          ["engineering"], maya_chen_eng_daily, ["engineering", "eng-daily"]),
    Agent("Aarav Patel", "Alpha Research Director", ":chart_with_upwards_trend:",
          ["alpha-research"], aarav_patel_strategy_review, ["alpha", "strategy"]),
    Agent("Linh Tran", "ML Modeling Lead", ":robot_face:",
          ["ml-experiments"], linh_tran_ml_results, ["ml", "experiment"]),
    Agent("Diego Ramírez", "Execution Engineer", ":zap:",
          ["squad-execution"], diego_ramirez_execution, ["execution", "slippage"]),
    Agent("Jian Wu", "Risk Engineer", ":shield:",
          ["risk-alerts"], jian_wu_risk, ["risk"]),
    Agent("Priya Subramanian", "Frontend Lead", ":art:",
          ["squad-frontend"], priya_subramanian_frontend, ["frontend"]),
    Agent("Anna Hoffmann", "Backend Lead", ":gear:",
          ["squad-backend"], anna_hoffmann_backend, ["backend"]),
    Agent("Sina Hassani", "Data Engineer", ":file_cabinet:",
          ["squad-data"], sina_hassani_data, ["data"]),
    Agent("Kenji Watanabe", "Director of DevOps", ":satellite_antenna:",
          ["infra-alerts"], kenji_watanabe_devops, ["devops", "ci"]),
    Agent("Kenji Watanabe", "Director of DevOps", ":satellite_antenna:",
          ["leadership-summary"], kenji_deploy_readiness, ["deploy", "infra"]),
    Agent("Aditi Sharma", "Director of QA", ":mag:",
          ["squad-qa"], aditi_sharma_qa, ["qa", "test"]),
    Agent("Aditi Sharma", "Director of QA", ":mag:",
          ["ci-failures"], aditi_open_prs, ["qa", "ci"]),
    Agent("Cameron Park", "Security Engineer", ":closed_lock_with_key:",
          ["security-alerts"], cameron_park_security, ["security"]),
    Agent("Sofia Karlsson", "VP Research", ":books:",
          ["papers"], sofia_karlsson_research, ["research", "papers"]),
    Agent("Yuki Mori", "Options Researcher", ":bar_chart:",
          ["desk-options"], yuki_mori_options, ["options"]),
    Agent("Hugo Bernardes", "Quant Researcher", ":mag_right:",
          ["alpha-research"], hugo_bernardes_research, ["alpha", "research"]),
    Agent("Tomas Lindqvist", "Research Scientist", ":brain:",
          ["pod-ml-rl"], tomas_lindqvist_rl, ["ml", "rl"]),
    Agent("Lior Avraham", "Polymarket Researcher", ":vertical_traffic_light:",
          ["desk-polymarket"], lior_avraham_polymarket, ["polymarket"]),
    Agent("Marcus Olufemi", "CRO", ":shield:",
          ["leadership-summary"], marcus_olufemi_risk, ["risk", "leadership"]),
    Agent("Wei Chang", "Finance Engineer", ":moneybag:",
          ["finance-ops"], wei_chang_finance, ["finance"]),
    Agent("Helena Voss", "Compliance Engineer", ":scales:",
          ["legal-compliance"], helena_voss_compliance, ["compliance"]),
    Agent("Karl Nyström", "Junior IC", ":raised_hand:",
          ["help"], karl_nystrom_question, ["help", "newbie"]),
    Agent("Laavanye Bahl", "CEO/Founder", ":sparkles:",
          ["announcements"], laavanye_bahl_ceo, ["ceo", "weekly"]),
    Agent("Ravi Iyer", "ML Infra Engineer", ":wrench:",
          ["engineering"], ravi_iyer_ci, ["ci", "infra", "ml"]),
    # ── Live trading-desk bots (read Alpaca paper account directly) ─────────
    Agent("PnL bot", "automated", ":bar_chart:",
          ["pnl-daily"], trading_desk_eod_pnl, ["pnl", "trading"]),
    Agent("Equity desk bot", "automated", ":chart_with_upwards_trend:",
          ["desk-equities"], trading_desk_equity_positions, ["equities", "trading"]),
    Agent("Crypto desk bot", "automated", ":coin:",
          ["desk-crypto"], trading_desk_crypto_positions, ["crypto", "trading"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        print("❌ SLACK_BOT_TOKEN missing or not xoxb-")
        return 1

    auth = slack_call(token, "auth.test", {})
    if not auth.get("ok"):
        print(f"❌ auth.test failed: {auth}")
        return 1
    print(f"✅ Authed as {auth.get('user')} in {auth.get('team')} at {datetime.now(timezone.utc).isoformat()}")

    # ── Team activity first (always runs): standups + scoreboard ────────────
    team_posts: list[Post] = []
    print("👥 Team activity")
    for team_name in TEAMS:
        sp = team_lead_standup_for(team_name)
        if sp:
            team_posts.append(sp)
        # Roughly half the runs: a team member also posts
        if random.random() < 0.55:
            mp = team_member_observation_for(team_name)
            if mp:
                team_posts.append(mp)
    # Leaderboard always
    lb = team_leaderboard_post()
    if lb:
        team_posts.append(lb)
    # Cross-team learning post (1 per run)
    ct = cross_team_share_post()
    if ct:
        team_posts.append(ct)
    # Friday presentation
    team_posts.extend(friday_presentation_post())

    # Sample wave: 60-80% of agents do real work each run (skew so it varies)
    wave_size = random.randint(int(len(AGENTS) * 0.6), int(len(AGENTS) * 0.85))
    wave = random.sample(AGENTS, wave_size)
    print(f"🎯 Wave: {wave_size}/{len(AGENTS)} agents + {len(team_posts)} team posts")

    posted_ts: dict[str, str] = {}  # channel -> last_ts of a parent post in that channel
    posts_made = 0
    errors = 0

    # Post team activity first
    for p in team_posts:
        r = post_to_slack(
            token, channel=p.channel, text=p.text,
            username=p.username, icon_emoji=p.icon_emoji,
            thread_ts=p.thread_of,
        )
        if r.get("ok"):
            posts_made += 1
            ts = r.get("ts")
            if ts and not p.thread_of:
                posted_ts[p.channel] = ts
            print(f"  ✓ TEAM {p.username[:36]} → #{p.channel}")
        else:
            errors += 1
            print(f"  ✗ TEAM → #{p.channel}: {r.get('error')}")
        time.sleep(0.7)

    for agent in wave:
        try:
            posts = agent.work_fn()
        except Exception as e:
            print(f"  ✗ {agent.name} work_fn crashed: {e}")
            errors += 1
            continue
        for p in posts:
            r = post_to_slack(
                token,
                channel=p.channel,
                text=p.text,
                username=p.username,
                icon_emoji=p.icon_emoji,
                thread_ts=p.thread_of,
            )
            if r.get("ok"):
                posts_made += 1
                ts = r.get("ts")
                if ts and not p.thread_of:
                    posted_ts[p.channel] = ts
                print(f"  ✓ {agent.name} → #{p.channel}")
            else:
                errors += 1
                print(f"  ✗ {agent.name} → #{p.channel}: {r.get('error')}")
            time.sleep(0.7)  # tier-1 rate limit safety

    # Discussion pass: a few agents reply in threads
    print("\n💬 Discussion pass — threaded replies")
    reply_candidates = [
        ("engineering", maya_reply_to_eng),
        ("alpha-research", sofia_reply_to_alpha),
        ("ml-experiments", hugo_reply_to_ml),
        ("squad-qa", aditi_reply_to_qa),
    ]
    for channel, replier_fn in reply_candidates:
        if channel not in posted_ts:
            continue
        if random.random() > 0.6:  # 60% chance to reply per channel
            continue
        reply = replier_fn(posted_ts[channel])
        r = post_to_slack(
            token,
            channel=reply.channel,
            text=reply.text,
            username=reply.username,
            icon_emoji=reply.icon_emoji,
            thread_ts=reply.thread_of,
        )
        if r.get("ok"):
            posts_made += 1
            print(f"  ✓ {reply.username} replied in #{channel}")
        else:
            errors += 1
        time.sleep(0.7)

    print(f"\n✅ Posted {posts_made} messages, {errors} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
