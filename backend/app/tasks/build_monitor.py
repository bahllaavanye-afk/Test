"""
Hourly build monitor with auto-fix.

Every hour it answers one question per surface — "does this still build?" — and
fixes what it safely can before anyone notices:

  Backend:  import smoke (app.main et al.) + `ruff check`.
  Frontend: `tsc --noEmit` typecheck (the meaningful "does it compile" gate;
            a full `vite build` is heavier and adds little signal hourly).

When a build is red, it runs a SAFE auto-fix pass — `ruff check --fix`,
`eslint --fix`, and the QA monitor's deprecated-API rewriter — then rebuilds.
Whatever is still red after that is escalated to a GitHub issue (deduplicated)
for Claude / a team lead, and broadcast on the agent bus so other desks see it.

Auto-fixes are applied in-place. Committing them is opt-in via
BUILD_AUTOFIX_COMMIT=1 so a 24/7 loop never surprise-pushes to the branch.
Everything is guarded: missing tools (node, ruff) or timeouts degrade to a
skipped/!ok result, never a crash.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from app.utils.logging import logger

BACKEND_DIR = Path(__file__).parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
BUILD_REPORT_PATH = PROJECT_ROOT / "build_report.json"

_CMD_TIMEOUT = 300

# Hard gates decide whether the build is "broken": code that won't import or
# won't typecheck. Lint (ruff) is a soft signal — auto-fixed, but a lint nit
# never marks the build red or opens an issue.
HARD_GATES = {"imports", "tsc"}


@dataclass
class BuildResult:
    component: str          # "backend" | "frontend"
    check: str              # "ruff" | "imports" | "tsc"
    ok: bool
    skipped: bool = False
    detail: str = ""


@dataclass
class BuildReport:
    timestamp: str
    overall_ok: bool
    results: list[BuildResult] = field(default_factory=list)
    autofixes_applied: list[str] = field(default_factory=list)
    issues_escalated: int = 0
    duration_seconds: float = 0.0


def _run(cmd: list[str], cwd: Path, timeout: int = _CMD_TIMEOUT) -> tuple[int, str]:
    """Run a command, returning (returncode, combined_output). Never raises."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(cwd), timeout=timeout
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(cmd)}"
    except Exception as e:  # noqa: BLE001
        return 1, f"error running {' '.join(cmd)}: {e}"


# ---------------------------------------------------------------------------
# Individual build checks
# ---------------------------------------------------------------------------
def backend_imports() -> BuildResult:
    """Reuse the QA monitor's import smoke test."""
    try:
        from app.tasks.qa_monitor import check_imports
        errors = check_imports()
        return BuildResult(
            component="backend", check="imports", ok=not errors,
            detail="\n".join(errors[:5]) if errors else "all modules import",
        )
    except Exception as e:  # noqa: BLE001
        return BuildResult(component="backend", check="imports", ok=False, detail=str(e))


def backend_ruff(fix: bool = False) -> BuildResult:
    cmd = ["ruff", "check", "app"] + (["--fix"] if fix else [])
    code, out = _run(cmd, BACKEND_DIR)
    if code == 127:
        return BuildResult(component="backend", check="ruff", ok=True, skipped=True,
                           detail="ruff not installed")
    return BuildResult(component="backend", check="ruff", ok=(code == 0),
                       detail=out[-800:])


def frontend_typecheck() -> BuildResult:
    tsc = FRONTEND_DIR / "node_modules" / ".bin" / "tsc"
    if not tsc.exists():
        return BuildResult(component="frontend", check="tsc", ok=True, skipped=True,
                           detail="node_modules/tsc absent — frontend not installed")
    code, out = _run([str(tsc), "--noEmit"], FRONTEND_DIR)
    return BuildResult(component="frontend", check="tsc", ok=(code == 0),
                       detail=out[-800:])


# ---------------------------------------------------------------------------
# Auto-fix pass
# ---------------------------------------------------------------------------
def apply_autofixes() -> list[str]:
    """Run safe in-place fixers. Returns a list of human-readable fix summaries."""
    applied: list[str] = []

    # 1. ruff --fix (import sorting, unused imports, simple lint fixes)
    res = backend_ruff(fix=True)
    if not res.skipped and "fixed" in res.detail.lower():
        applied.append("ruff --fix applied to backend/app")

    # 2. Deprecated-API rewriter from the QA monitor.
    try:
        from app.tasks.qa_monitor import scan_security_issues, auto_fix_deprecated_apis
        n = auto_fix_deprecated_apis(scan_security_issues())
        if n:
            applied.append(f"deprecated-API rewrite in {n} file(s)")
    except Exception as e:  # noqa: BLE001
        logger.debug("build_monitor: deprecated-API fix failed", error=str(e))

    # 3. eslint --fix (frontend), only when the toolchain is present.
    eslint = FRONTEND_DIR / "node_modules" / ".bin" / "eslint"
    if eslint.exists():
        code, _out = _run([str(eslint), ".", "--ext", "ts,tsx", "--fix"], FRONTEND_DIR)
        if code == 0:
            applied.append("eslint --fix applied to frontend")

    return applied


def _maybe_commit(applied: list[str]) -> bool:
    """Opt-in commit of auto-fixes. Off unless BUILD_AUTOFIX_COMMIT is truthy."""
    if not applied:
        return False
    if os.getenv("BUILD_AUTOFIX_COMMIT", "0").lower() not in ("1", "true", "yes"):
        return False
    code_add, _ = _run(["git", "add", "-A"], PROJECT_ROOT, timeout=60)
    msg = "build: hourly auto-fix — " + "; ".join(applied) + " [skip ci]"
    code_commit, out = _run(["git", "commit", "-m", msg], PROJECT_ROOT, timeout=60)
    if code_commit == 0:
        logger.info("build_monitor: committed auto-fixes", summary=applied)
        return True
    logger.debug("build_monitor: nothing to commit or commit failed", out=out[-200:])
    return False


def _escalation_candidates(failed: list[BuildResult]) -> list[dict]:
    import hashlib
    candidates = []
    role_by_component = {"backend": "backend", "frontend": "frontend"}
    for r in failed:
        fp = hashlib.sha1(f"build|{r.component}|{r.check}".encode()).hexdigest()[:12]
        candidates.append({
            "fingerprint": fp,
            "title": f"[Build] {r.component} {r.check} failing",
            "body": (
                f"The hourly build monitor found `{r.component}` failing its "
                f"`{r.check}` check, and the auto-fix pass did not resolve it.\n\n"
                f"```\n{r.detail}\n```\n\n"
                "Assigned for Claude / team-lead review."
            ),
            "priority": "P1",
            "role": role_by_component.get(r.component, "backend"),
        })
    return candidates


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
class BuildMonitor:
    def __init__(self, interval_seconds: int = 3600):
        self.interval_seconds = interval_seconds
        self._running = False
        self.last_report: BuildReport | None = None

    def _build_all(self) -> list[BuildResult]:
        return [backend_imports(), backend_ruff(fix=False), frontend_typecheck()]

    async def run_cycle(self) -> BuildReport:
        import time
        start = time.time()
        loop = asyncio.get_running_loop()

        results = await loop.run_in_executor(None, self._build_all)
        autofixes: list[str] = []

        # Trigger an auto-fix pass if anything is red (including soft lint), but
        # only HARD-gate failures count as a broken build / get escalated.
        any_red = [r for r in results if not r.ok and not r.skipped]
        red = [r for r in any_red if r.check in HARD_GATES]
        if any_red:
            autofixes = await loop.run_in_executor(None, apply_autofixes)
            if autofixes:
                await loop.run_in_executor(None, _maybe_commit, autofixes)
                results = await loop.run_in_executor(None, self._build_all)
                red = [r for r in results
                       if not r.ok and not r.skipped and r.check in HARD_GATES]

        report = BuildReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_ok=not red,
            results=results,
            autofixes_applied=autofixes,
            duration_seconds=round(time.time() - start, 2),
        )

        # Escalate whatever is still red after the fix pass.
        if red:
            try:
                from app.tasks.issue_escalation import get_escalator
                escalator = get_escalator()
                if escalator.enabled:
                    summ = await escalator.escalate_items(_escalation_candidates(red))
                    report.issues_escalated = summ.get("opened", 0)
            except Exception as e:  # noqa: BLE001
                logger.warning("build_monitor: escalation failed", error=str(e))

        self.last_report = report
        await self._persist_and_broadcast(report)
        logger.info("build_monitor: cycle complete", overall_ok=report.overall_ok,
                    autofixes=len(autofixes), escalated=report.issues_escalated,
                    duration_s=report.duration_seconds)
        return report

    async def _persist_and_broadcast(self, report: BuildReport) -> None:
        try:
            BUILD_REPORT_PATH.write_text(json.dumps(asdict(report), indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            logger.debug("build_monitor: report write failed", error=str(e))
        try:
            from app.tasks.agent_bus import get_bus
            await get_bus().broadcast_signal(
                {"type": "build_cycle", "overall_ok": report.overall_ok,
                 "autofixes": report.autofixes_applied,
                 "escalated": report.issues_escalated},
                from_agent="build_monitor",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("build_monitor: broadcast failed", error=str(e))

    async def run(self) -> None:
        self._running = True
        logger.info("BuildMonitor started", interval=self.interval_seconds)
        while self._running:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                logger.info("BuildMonitor cancelled — shutting down")
                break
            except Exception as e:  # noqa: BLE001
                logger.error("BuildMonitor cycle crashed", error=str(e))
            if self._running:
                await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False


_monitor: BuildMonitor | None = None


def get_build_monitor() -> BuildMonitor:
    global _monitor
    if _monitor is None:
        _monitor = BuildMonitor()
    return _monitor
