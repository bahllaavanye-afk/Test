"""
Red-team agent — an independent security auditor that hunts for mistakes in the
codebase 24/7 and routes what it finds to the security desk for review/fix.

It runs a deeper static audit than the QA monitor's quick scan: dangerous calls
(eval/exec, pickle.loads on untrusted input, shell=True), disabled safety
(verify=False, debug=True, JWT decode without verification), and likely secret
leakage. Confirmed high/critical findings are filed as deduplicated GitHub
issues labelled role:security via the escalation bridge, broadcast on the agent
bus, and posted to the security channel — so the "hackers" plug into every
workflow and the LLM agents see their findings.

Findings are advisory: the agent never edits code itself. It surfaces precise
file:line locations so a human or Claude fixes them deliberately.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from app.utils.logging import logger

BACKEND_DIR = Path(__file__).parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

# (regex, severity, id, description). Severity drives escalation priority.
AUDIT_PATTERNS: list[tuple[str, str, str, str]] = [
    # Negative lookbehind for `.` and word chars so we match the dangerous
    # builtin eval(/exec( but NOT benign method calls like model.eval() (a
    # ubiquitous PyTorch idiom) or names like safe_eval(.
    (r"(?<![.\w])eval\s*\(",                 "high",     "eval_use",
     "eval() executes arbitrary code — avoid on any external input"),
    (r"(?<![.\w])exec\s*\(",                 "high",     "exec_use",
     "exec() executes arbitrary code — avoid on any external input"),
    (r"pickle\.loads?\s*\(",                 "high",     "insecure_pickle",
     "pickle on untrusted data enables RCE — use json or a vetted format"),
    (r"subprocess\.[a-z]+\([^)]*shell\s*=\s*True", "high", "shell_true",
     "subprocess with shell=True invites shell injection"),
    (r"verify\s*=\s*False",                  "high",     "tls_verify_off",
     "TLS verification disabled — vulnerable to MITM"),
    (r"jwt\.decode\([^)]*verify\s*=\s*False", "critical", "jwt_no_verify",
     "JWT decoded without signature verification — auth bypass"),
    (r"\bdebug\s*=\s*True",                  "medium",   "debug_true",
     "debug=True leaks stack traces in production"),
    (r"(?:password|secret|token|api_key)\s*=\s*[\"'][^\"']{8,}[\"']", "critical", "hardcoded_secret",
     "Possible hardcoded credential in source"),
    (r"yaml\.load\s*\((?![^)]*Loader)",      "high",     "unsafe_yaml",
     "yaml.load without SafeLoader can construct arbitrary objects"),
    (r"md5\s*\(",                            "low",      "weak_hash",
     "MD5 is cryptographically broken — use sha256 for security contexts"),
]


@dataclass
class Finding:
    severity: str
    rule_id: str
    file_path: str
    line_number: int
    description: str
    snippet: str


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for py in BACKEND_DIR.rglob("app/**/*.py"):
        s = str(py)
        if "__pycache__" in s:
            continue
        if py.name in ("red_team.py", "qa_monitor.py"):
            continue  # these files contain the patterns as regex literals
        files.append(py)
    return files


def scan_codebase() -> list[Finding]:
    """Run the static audit across backend source. Pure function — no I/O escape."""
    compiled = [(re.compile(p), sev, rid, desc) for p, sev, rid, desc in AUDIT_PATTERNS]
    findings: list[Finding] = []
    for py in _iter_source_files():
        try:
            text = py.read_text(errors="replace")
        except Exception:
            continue
        lines = text.splitlines()
        for rx, sev, rid, desc in compiled:
            for m in rx.finditer(text):
                line_no = text[:m.start()].count("\n") + 1
                snippet = lines[line_no - 1].strip()[:160] if line_no - 1 < len(lines) else ""
                findings.append(Finding(
                    severity=sev, rule_id=rid,
                    file_path=str(py.relative_to(PROJECT_ROOT)),
                    line_number=line_no, description=desc, snippet=snippet,
                ))
    return findings


def to_escalations(findings: list[Finding]) -> list[dict]:
    """Convert high/critical findings into escalation candidates (role:security)."""
    import hashlib
    candidates = []
    for f in findings:
        if f.severity not in ("critical", "high"):
            continue
        fp = hashlib.sha1(f"redteam|{f.rule_id}|{f.file_path}|{f.line_number}".encode()).hexdigest()[:12]
        priority = "P0" if f.severity == "critical" else "P1"
        candidates.append({
            "fingerprint": fp,
            "title": f"[Security] {f.rule_id} in {f.file_path}:{f.line_number}",
            "body": (
                f"**Severity:** {f.severity}\n"
                f"**Rule:** {f.rule_id}\n"
                f"**Location:** `{f.file_path}:{f.line_number}`\n\n"
                f"{f.description}\n\n"
                f"```python\n{f.snippet}\n```\n\n"
                "Found by the autonomous red-team agent. Assigned to the security "
                "desk for review — the agent does not edit code itself."
            ),
            "priority": priority,
            "role": "security",
        })
    return candidates


def summarize(findings: list[Finding]) -> dict:
    by_sev: dict[str, int] = {}
    for f in findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    return {
        "total": len(findings),
        "critical": by_sev.get("critical", 0),
        "high": by_sev.get("high", 0),
        "medium": by_sev.get("medium", 0),
        "low": by_sev.get("low", 0),
    }


class RedTeamAgent:
    def __init__(self, interval_seconds: int = 3600):
        self.interval_seconds = interval_seconds
        self._running = False
        self.last_summary: dict | None = None

    async def run_cycle(self) -> dict:
        loop = asyncio.get_running_loop()
        findings = await loop.run_in_executor(None, scan_codebase)
        summary = summarize(findings)
        summary["scanned_at"] = datetime.now(timezone.utc).isoformat()
        self.last_summary = summary

        candidates = to_escalations(findings)
        escalated = 0
        if candidates:
            try:
                from app.tasks.issue_escalation import get_escalator
                escalator = get_escalator()
                if escalator.enabled:
                    res = await escalator.escalate_items(candidates)
                    escalated = res.get("opened", 0)
            except Exception as e:  # noqa: BLE001
                logger.warning("red_team: escalation failed", error=str(e))
        summary["escalated"] = escalated

        try:
            from app.tasks.agent_bus import get_bus
            await get_bus().broadcast_signal(
                {"type": "red_team_scan", **summary}, from_agent="red_team")
        except Exception as e:  # noqa: BLE001
            logger.debug("red_team: broadcast failed", error=str(e))

        logger.info("red_team: scan complete", **{k: summary[k] for k in
                    ("total", "critical", "high", "escalated")})
        return summary

    async def run(self) -> None:
        self._running = True
        logger.info("RedTeamAgent started", interval_s=self.interval_seconds)
        while self._running:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                logger.info("RedTeamAgent cancelled — shutting down")
                break
            except Exception as e:  # noqa: BLE001
                logger.error("RedTeamAgent cycle crashed", error=str(e))
            if self._running:
                await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False


_agent: RedTeamAgent | None = None


def get_red_team() -> RedTeamAgent:
    global _agent
    if _agent is None:
        _agent = RedTeamAgent()
    return _agent
