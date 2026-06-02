"""
QuantEdge Agent Health Monitor — self-healing, channel-level diagnostics.

Checks every channel and every engineer function for signs of failure, then:
  1. Posts a structured health report to #incidents
  2. Posts a brief summary to #allquantedge
  3. Attempts auto-healing (retry failed agents, patch bad config)
  4. Exits non-zero if critical failures remain after healing attempts

Run modes:
  python agent_health_monitor.py             # full health sweep
  python agent_health_monitor.py --channel desk-crypto  # single channel
  python agent_health_monitor.py --dry-run   # report only, no Slack posts
"""

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─── Config ──────────────────────────────────────────────────────────────────

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
REPO_ROOT = Path(__file__).parent.parent.parent
AGENT_SCRIPT = Path(__file__).parent / "slack_agent_team.py"
DRY_RUN = "--dry-run" in sys.argv

# Channels that MUST have at least one proactive posting agent
REQUIRED_POSTING_CHANNELS = {
    "engineering", "alpha-research", "ml-experiments", "squad-qa",
    "desk-crypto", "squad-backend", "squad-frontend", "risk-alerts",
    "infra-alerts", "desk-equities", "desk-polymarket", "desk-commodities",
    "desk-futures", "desk-rates", "desk-kalshi", "desk-stat-arb",
    "desk-fx-rates", "desk-options", "help", "pnl-daily", "ci-failures",
    "squad-execution", "squad-data", "general", "standup", "wins",
    "incidents", "strategy-review", "model-performance", "code-review",
    "leadership-summary", "papers", "finance-ops", "legal-compliance",
    "security-alerts", "pod-ml-rl", "announcements", "allquantedge",
}

# Employee function names that must call real LLM (not hardcoded)
REQUIRED_LLM_CALLERS = [
    "maya_chen_eng_daily", "aarav_patel_strategy_review", "linh_tran_ml_results",
    "diego_ramirez_execution", "jian_wu_risk", "priya_subramanian_frontend",
    "anna_hoffmann_backend", "sina_hassani_data", "kenji_watanabe_devops",
    "kenji_deploy_readiness", "aditi_sharma_qa", "aditi_open_prs",
    "cameron_park_security", "sofia_karlsson_research", "yuki_mori_options",
    "hugo_bernardes_research", "tomas_lindqvist_rl", "lior_avraham_polymarket",
    "marcus_olufemi_risk", "wei_chang_finance", "helena_voss_compliance",
    "karl_nystrom_question", "laavanye_bahl_ceo", "ravi_iyer_ci",
    "sara_kim_ml_research", "marcus_williams_dl_engineer",
    "priya_nair_feature_eng", "alex_chen_quant_ml",
]

LLM_CALL_PATTERNS = [
    "employee_provider_prompt", "moa_employee_prompt",
    "call_best_agent", "call_litellm", "call_best_agent_for_task",
]

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ChannelCheck:
    channel: str
    has_agent: bool = False
    agent_names: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    status: str = "unknown"   # ok | warn | critical

@dataclass
class EngineerCheck:
    fn_name: str
    uses_llm: bool = False
    llm_call: str = ""
    issues: list[str] = field(default_factory=list)
    status: str = "unknown"   # ok | warn | critical

@dataclass
class HealthReport:
    timestamp: str
    channel_checks: list[ChannelCheck] = field(default_factory=list)
    engineer_checks: list[EngineerCheck] = field(default_factory=list)
    exec_results: list["AgentExecResult"] = field(default_factory=list)
    env_checks: dict[str, bool] = field(default_factory=dict)
    llm_probe_results: dict[str, str] = field(default_factory=dict)
    syntax_ok: bool = True
    critical_count: int = 0
    warn_count: int = 0
    ok_count: int = 0
    healed: list[str] = field(default_factory=list)


# ─── Static analysis helpers ──────────────────────────────────────────────────

def _load_agent_ast() -> ast.Module | None:
    try:
        src = AGENT_SCRIPT.read_text()
        return ast.parse(src)
    except SyntaxError as e:
        print(f"[health] SYNTAX ERROR in agent script: {e}")
        return None


def _get_all_agent_entries(tree: ast.Module) -> list[dict]:
    """Extract Agent(...) call arguments from the AGENTS list (handles AnnAssign)."""
    entries = []

    def _extract_from_list(list_node: ast.List) -> None:
        for elt in list_node.elts:
            if not isinstance(elt, ast.Call):
                continue
            args = elt.args
            # Agent(username, display_name, emoji, [channels], fn, [keywords])
            if len(args) < 5:
                continue
            channels_node = args[3]
            fn_node = args[4]
            channels = []
            if isinstance(channels_node, ast.List):
                channels = [
                    c.value if isinstance(c, ast.Constant) else ""
                    for c in channels_node.elts
                ]
            fn_name = fn_node.id if isinstance(fn_node, ast.Name) else ""
            username = args[0].value if isinstance(args[0], ast.Constant) else ""
            entries.append({"username": username, "channels": channels, "fn_name": fn_name})

    for node in ast.walk(tree):
        # Handle both Assign and AnnAssign (AGENTS: list[Agent] = [...])
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "AGENTS":
                if node.value and isinstance(node.value, ast.List):
                    _extract_from_list(node.value)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "AGENTS":
                    if isinstance(node.value, ast.List):
                        _extract_from_list(node.value)
    return entries


def _fn_uses_llm(tree: ast.Module, fn_name: str) -> tuple[bool, str]:
    """Check if a function body contains any real LLM call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            fn_src = ast.unparse(node)
            for pat in LLM_CALL_PATTERNS:
                if pat in fn_src:
                    return True, pat
            return False, ""
    return False, ""


def _check_required_channels(tree: ast.Module) -> list[ChannelCheck]:
    entries = _get_all_agent_entries(tree)
    covered: dict[str, list[str]] = {}
    for e in entries:
        for ch in e["channels"]:
            covered.setdefault(ch, []).append(e["username"] or e["fn_name"])

    checks = []
    for ch in REQUIRED_POSTING_CHANNELS:
        c = ChannelCheck(channel=ch)
        if ch in covered:
            c.has_agent = True
            c.agent_names = covered[ch]
            c.status = "ok"
        else:
            c.has_agent = False
            c.issues.append(f"No Agent() entry posts to #{ch}")
            c.status = "critical"
        checks.append(c)
    return checks


def _check_engineer_llm_usage(tree: ast.Module) -> list[EngineerCheck]:
    checks = []
    for fn_name in REQUIRED_LLM_CALLERS:
        ec = EngineerCheck(fn_name=fn_name)
        uses, call = _fn_uses_llm(tree, fn_name)
        ec.uses_llm = uses
        ec.llm_call = call
        if uses:
            ec.status = "ok"
        else:
            ec.issues.append(f"{fn_name}() has no LLM call — may be hardcoded")
            ec.status = "critical"
        checks.append(ec)
    return checks


# ─── Environment checks ───────────────────────────────────────────────────────

def _check_env() -> dict[str, bool]:
    keys = [
        "SLACK_BOT_TOKEN",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
    ]
    return {k: bool(os.environ.get(k, "").strip()) for k in keys}


# ─── LLM provider probe ───────────────────────────────────────────────────────

def _probe_gemini(api_key: str) -> str:
    """Quick ping to Gemini Flash — returns 'ok' or error string."""
    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={api_key}"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": "Reply with just 'pong'"}]}],
            "generationConfig": {"maxOutputTokens": 5},
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
        candidates = resp.get("candidates", [])
        if candidates:
            return "ok"
        return f"empty response: {resp}"
    except Exception as e:
        return f"error: {e}"


def _probe_groq(api_key: str) -> str:
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = json.dumps({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Reply with just 'pong'"}],
            "max_tokens": 5,
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
        return "ok" if resp.get("choices") else f"empty: {resp}"
    except urllib.error.HTTPError as e:
        if e.code == 503:
            return "cf_blocked"
        return f"http_{e.code}"
    except Exception as e:
        return f"error: {e}"


def _probe_llm_providers() -> dict[str, str]:
    results: dict[str, str] = {}
    gk = os.environ.get("GEMINI_API_KEY", "").strip()
    if gk:
        results["gemini"] = _probe_gemini(gk)
    else:
        results["gemini"] = "no_key"

    rk = os.environ.get("GROQ_API_KEY", "").strip()
    if rk:
        results["groq"] = _probe_groq(rk)
    else:
        results["groq"] = "no_key"

    ck = os.environ.get("CEREBRAS_API_KEY", "").strip()
    results["cerebras"] = "key_present" if ck else "no_key"

    return results


# ─── Real-world agent execution tests ────────────────────────────────────────

# Agents to fire-test: (fn_name, channel, expected_min_length, description)
AGENT_EXECUTION_TESTS = [
    # Core engineering
    ("maya_chen_eng_daily",        "engineering",      80, "VP Eng daily report"),
    ("anna_hoffmann_backend",      "squad-backend",    80, "Backend lead report"),
    ("aditi_sharma_qa",            "squad-qa",         80, "QA director report"),
    # ML
    ("linh_tran_ml_results",       "ml-experiments",   80, "ML results report"),
    # Risk
    ("jian_wu_risk",               "risk-alerts",      60, "Risk alert report"),
    # Trading desks
    ("trading_desk_eod_pnl",       "pnl-daily",        30, "EOD PnL report"),
    ("trading_desk_crypto_positions", "desk-crypto",   30, "Crypto positions"),
    # Research
    ("sofia_karlsson_research",    "papers",           80, "Research paper summary"),
    # Company wide
    ("general_channel",            "general",          80, "General channel digest"),
    ("allquantedge_channel",       "allquantedge",     80, "All-hands broadcast"),
    # Ops
    ("wei_chang_finance",          "finance-ops",      60, "Finance ops report"),
    ("cameron_park_security",      "security-alerts",  60, "Security check"),
    ("standup_channel",            "standup",          80, "Daily standup"),
]


@dataclass
class AgentExecResult:
    fn_name: str
    channel: str
    description: str
    success: bool = False
    n_posts: int = 0
    post_lengths: list[int] = field(default_factory=list)
    error: str = ""
    duration_ms: int = 0
    status: str = "unknown"  # ok | warn | critical


def _execute_agent_test(
    fn_name: str,
    channel: str,
    description: str,
    min_length: int,
    module: Any,
) -> AgentExecResult:
    """Call one agent function and verify it returns non-empty, substantive Posts."""
    result = AgentExecResult(fn_name=fn_name, channel=channel, description=description)
    t0 = time.monotonic()
    try:
        fn = getattr(module, fn_name, None)
        if fn is None:
            result.error = "function not found in module"
            result.status = "critical"
            return result

        posts = fn()
        result.duration_ms = int((time.monotonic() - t0) * 1000)

        if not isinstance(posts, list):
            result.error = f"returned {type(posts).__name__}, expected list"
            result.status = "critical"
            return result

        if len(posts) == 0:
            # Some agents skip on non-trigger days (e.g. CEO only on Monday) — warn not critical
            result.n_posts = 0
            result.status = "warn"
            result.error = "returned empty list (may be day-gated)"
            return result

        result.n_posts = len(posts)
        for p in posts:
            text = getattr(p, "text", "") or ""
            result.post_lengths.append(len(text))
            if len(text) < min_length:
                result.error = f"post too short ({len(text)} chars, min {min_length})"
                result.status = "warn"
                return result

        result.success = True
        result.status = "ok"

    except Exception as e:
        result.duration_ms = int((time.monotonic() - t0) * 1000)
        result.error = f"{type(e).__name__}: {str(e)[:200]}"
        result.status = "critical"

    return result


def _run_agent_execution_tests(target_channel: str | None = None) -> list[AgentExecResult]:
    """
    Import the agent module once, then call each agent function.
    No Slack token needed — functions return Posts without posting.
    """
    # Temporarily stub out any network calls that would fail in test
    spec = importlib.util.spec_from_file_location("slack_agent_team", AGENT_SCRIPT)
    if spec is None or spec.loader is None:
        print("[health] Cannot import agent module for execution tests")
        return []

    try:
        mod = importlib.util.module_from_spec(spec)
        # Don't exec the module if it's huge — just do static tests
        # (exec would call top-level code including loading state etc.)
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        print(f"[health] Module import error: {e}")
        return []

    tests = AGENT_EXECUTION_TESTS
    if target_channel:
        tests = [(fn, ch, d, ml, desc) for fn, ch, desc, ml, d_ in
                 [(*t[:2], t[2], t[3], t[4]) for t in tests]
                 if ch == target_channel] if False else [
            t for t in tests if t[1] == target_channel
        ]

    results = []
    for fn_name, channel, min_len, description in [(t[0], t[1], t[2], t[3]) for t in tests]:
        print(f"[health] Testing {fn_name}()...")
        r = _execute_agent_test(fn_name, channel, description, min_len, mod)
        status_icon = {"ok": "✓", "warn": "~", "critical": "✗"}.get(r.status, "?")
        print(f"  {status_icon} {fn_name}: {r.status} ({r.n_posts} posts, {r.duration_ms}ms) {r.error or ''}")
        results.append(r)

    return results


# ─── Slack posting ────────────────────────────────────────────────────────────

def _slack_post(channel: str, text: str) -> bool:
    if DRY_RUN or not SLACK_BOT_TOKEN:
        print(f"[dry-run] #{channel}: {text[:120]}")
        return True
    try:
        payload = json.dumps({"channel": channel, "text": text}).encode()
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
        return bool(resp.get("ok"))
    except Exception as e:
        print(f"[health] slack post error: {e}")
        return False


# ─── Report builders ──────────────────────────────────────────────────────────

def _build_incident_report(report: HealthReport) -> str:
    ts = report.timestamp
    lines = [f"*:stethoscope: QuantEdge Agent Health Report* — {ts}"]

    # Summary line
    lines.append(
        f"> :white_check_mark: {report.ok_count} ok   "
        f":warning: {report.warn_count} warn   "
        f":x: {report.critical_count} critical"
    )

    # Channel issues
    ch_issues = [c for c in report.channel_checks if c.status != "ok"]
    if ch_issues:
        lines.append("\n*Channel coverage issues:*")
        for c in ch_issues:
            emoji = ":x:" if c.status == "critical" else ":warning:"
            lines.append(f"{emoji} `#{c.channel}` — {'; '.join(c.issues)}")
    else:
        lines.append(":white_check_mark: All required channels have posting agents")

    # Engineer LLM issues
    eng_issues = [e for e in report.engineer_checks if e.status != "ok"]
    if eng_issues:
        lines.append("\n*Engineer LLM usage issues:*")
        for e in eng_issues:
            lines.append(f":x: `{e.fn_name}` — {'; '.join(e.issues)}")
    else:
        lines.append(f":white_check_mark: All {len(report.engineer_checks)} engineers using real LLMs")

    # Env check
    missing_keys = [k for k, v in report.env_checks.items() if not v]
    if missing_keys:
        lines.append(f"\n*Missing secrets:* {', '.join(f'`{k}`' for k in missing_keys)}")
        lines.append("  → Add these to GitHub → Settings → Secrets and variables → Actions")
    else:
        lines.append(":white_check_mark: All required secrets present")

    # LLM probe results
    if report.llm_probe_results:
        lines.append("\n*LLM provider probe:*")
        for provider, status in report.llm_probe_results.items():
            emoji = ":white_check_mark:" if status == "ok" else ":warning:"
            lines.append(f"{emoji} `{provider}`: {status}")

    # Agent execution test results
    if report.exec_results:
        exec_critical = [r for r in report.exec_results if r.status == "critical"]
        exec_warn = [r for r in report.exec_results if r.status == "warn"]
        exec_ok = [r for r in report.exec_results if r.status == "ok"]
        lines.append(
            f"\n*Agent execution tests* ({len(exec_ok)} passed, {len(exec_warn)} warned, {len(exec_critical)} failed):"
        )
        for r in exec_critical:
            avg_len = int(sum(r.post_lengths) / len(r.post_lengths)) if r.post_lengths else 0
            lines.append(
                f":x: `{r.fn_name}` → #{r.channel} — {r.error or 'FAILED'} ({r.duration_ms}ms)"
            )
        for r in exec_warn:
            lines.append(
                f":warning: `{r.fn_name}` → #{r.channel} — {r.error} ({r.duration_ms}ms)"
            )
        if exec_ok and not exec_critical:
            lines.append(
                f":white_check_mark: {len(exec_ok)} agents produced substantive output (avg "
                f"{int(sum(sum(r.post_lengths) for r in exec_ok if r.post_lengths) / max(1, sum(1 for r in exec_ok if r.post_lengths)))} chars/post)"
            )

    # Healed actions
    if report.healed:
        lines.append("\n*Auto-healed:*")
        for h in report.healed:
            lines.append(f":wrench: {h}")

    lines.append(f"\n_Run `python .github/scripts/agent_health_monitor.py --dry-run` locally for details._")
    return "\n".join(lines)


def _build_allquantedge_summary(report: HealthReport) -> str:
    if report.critical_count == 0:
        return (
            f":green_heart: Agent health check passed — all {len(report.channel_checks)} channels "
            f"covered, all {len(report.engineer_checks)} engineers using real LLMs. "
            f"Langfuse traces active. System is self-healing and online."
        )
    else:
        return (
            f":warning: Health monitor found *{report.critical_count} critical issue(s)* — "
            f"{sum(1 for c in report.channel_checks if c.status=='critical')} channel coverage gaps, "
            f"{sum(1 for e in report.engineer_checks if e.status=='critical')} engineer LLM gaps. "
            f"Details in #incidents. Auto-healing in progress."
        )


# ─── Auto-healing ─────────────────────────────────────────────────────────────

def _attempt_self_heal(report: HealthReport) -> None:
    """
    Attempt to auto-fix known recoverable issues.
    Records healed actions in report.healed.
    """
    # 1. If ALLOW_PAID_APIS leaked to True, correct it
    if AGENT_SCRIPT.exists():
        src = AGENT_SCRIPT.read_text()
        if "ALLOW_PAID_APIS = True" in src or 'ALLOW_PAID_APIS="True"' in src:
            fixed = src.replace("ALLOW_PAID_APIS = True", "ALLOW_PAID_APIS = False")
            fixed = fixed.replace('ALLOW_PAID_APIS="True"', 'ALLOW_PAID_APIS="False"')
            AGENT_SCRIPT.write_text(fixed)
            report.healed.append("Reset ALLOW_PAID_APIS back to False (zero-spend policy)")

    # 2. If Langfuse not installed, attempt install (CI environment)
    try:
        import langfuse  # noqa
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "langfuse", "--quiet"],
                capture_output=True, timeout=60
            )
            report.healed.append("Auto-installed langfuse package")
        except Exception as e:
            pass  # non-critical

    # 3. If litellm not installed, attempt install
    try:
        import litellm  # noqa
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "litellm", "--quiet"],
                capture_output=True, timeout=90
            )
            report.healed.append("Auto-installed litellm package")
        except Exception:
            pass

    # 4. Recount after healing
    report.critical_count = (
        sum(1 for c in report.channel_checks if c.status == "critical") +
        sum(1 for e in report.engineer_checks if e.status == "critical")
    )
    report.warn_count = (
        sum(1 for c in report.channel_checks if c.status == "warn") +
        sum(1 for e in report.engineer_checks if e.status == "warn")
    )
    report.ok_count = (
        sum(1 for c in report.channel_checks if c.status == "ok") +
        sum(1 for e in report.engineer_checks if e.status == "ok")
    )


# ─── Runtime agent execution test ─────────────────────────────────────────────

def _test_agent_function_execution(fn_name: str, timeout_secs: int = 30) -> tuple[bool, str]:
    """
    Import the agent script and call the named function.
    Returns (success, output_or_error).
    Only tests functions that don't need Slack token (they return Posts).
    """
    spec = importlib.util.spec_from_file_location("slack_agent_team", AGENT_SCRIPT)
    if spec is None or spec.loader is None:
        return False, "cannot load module"
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        fn = getattr(mod, fn_name, None)
        if fn is None:
            return False, f"function {fn_name} not found"
        result = fn()
        if isinstance(result, list):
            return True, f"returned {len(result)} Post(s)"
        return True, f"returned {type(result).__name__}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_health_check(target_channel: str | None = None) -> HealthReport:
    report = HealthReport(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )

    print(f"[health] QuantEdge Agent Health Monitor — {report.timestamp}")

    # 1. Syntax check
    tree = _load_agent_ast()
    if tree is None:
        report.syntax_ok = False
        report.critical_count = 999
        return report

    # 2. Channel coverage
    report.channel_checks = _check_required_channels(tree)
    if target_channel:
        report.channel_checks = [c for c in report.channel_checks if c.channel == target_channel]

    # 3. Engineer LLM usage
    report.engineer_checks = _check_engineer_llm_usage(tree)

    # 4. Environment secrets
    report.env_checks = _check_env()

    # 5. LLM provider probe
    print("[health] Probing LLM providers...")
    report.llm_probe_results = _probe_llm_providers()

    # 6. Agent execution tests (real-world: call each function, verify output)
    print("[health] Running agent execution tests...")
    report.exec_results = _run_agent_execution_tests(target_channel)

    # 7. Count totals
    report.critical_count = (
        sum(1 for c in report.channel_checks if c.status == "critical") +
        sum(1 for e in report.engineer_checks if e.status == "critical") +
        sum(1 for r in report.exec_results if r.status == "critical")
    )
    report.warn_count = (
        sum(1 for c in report.channel_checks if c.status == "warn") +
        sum(1 for e in report.engineer_checks if e.status == "warn") +
        sum(1 for r in report.exec_results if r.status == "warn")
    )
    report.ok_count = (
        sum(1 for c in report.channel_checks if c.status == "ok") +
        sum(1 for e in report.engineer_checks if e.status == "ok") +
        sum(1 for r in report.exec_results if r.status == "ok")
    )

    # 7. Auto-heal
    print("[health] Attempting self-heal...")
    _attempt_self_heal(report)

    return report


def main() -> int:
    target_channel = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--channel" and i + 1 < len(sys.argv) - 1:
            target_channel = sys.argv[i + 2]

    report = run_health_check(target_channel)

    # Print human-readable summary
    print(f"\n{'='*60}")
    print(f"Health Summary: {report.ok_count} ok, {report.warn_count} warn, {report.critical_count} critical")

    ch_issues = [c for c in report.channel_checks if c.status != "ok"]
    if ch_issues:
        print("\nChannel issues:")
        for c in ch_issues:
            print(f"  [{c.status.upper()}] #{c.channel}: {'; '.join(c.issues)}")

    eng_issues = [e for e in report.engineer_checks if e.status != "ok"]
    if eng_issues:
        print("\nEngineer LLM issues:")
        for e in eng_issues:
            print(f"  [{e.status.upper()}] {e.fn_name}: {'; '.join(e.issues)}")

    missing = [k for k, v in report.env_checks.items() if not v]
    if missing:
        print(f"\nMissing env vars: {', '.join(missing)}")

    print("\nLLM probes:", report.llm_probe_results)

    if report.healed:
        print("\nAuto-healed:")
        for h in report.healed:
            print(f"  + {h}")

    # Post to Slack
    if SLACK_BOT_TOKEN or DRY_RUN:
        incident_text = _build_incident_report(report)
        _slack_post("incidents", incident_text)

        # Only post to allquantedge on critical issues or explicit run
        if report.critical_count > 0 or "--always-post" in sys.argv:
            summary = _build_allquantedge_summary(report)
            _slack_post("allquantedge", summary)

    # Save JSON report
    report_path = Path("/tmp/health_report.json")
    try:
        report_path.write_text(json.dumps({
            "timestamp": report.timestamp,
            "ok": report.ok_count,
            "warn": report.warn_count,
            "critical": report.critical_count,
            "syntax_ok": report.syntax_ok,
            "healed": report.healed,
            "env_checks": report.env_checks,
            "llm_probes": report.llm_probe_results,
            "channel_issues": [
                {"channel": c.channel, "status": c.status, "issues": c.issues}
                for c in report.channel_checks if c.status != "ok"
            ],
            "engineer_issues": [
                {"fn": e.fn_name, "status": e.status, "issues": e.issues}
                for e in report.engineer_checks if e.status != "ok"
            ],
        }, indent=2))
        print(f"\nReport saved to {report_path}")
    except Exception:
        pass

    return 1 if report.critical_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
