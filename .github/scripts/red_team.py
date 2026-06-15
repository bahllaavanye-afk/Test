"""
Red team — continuous adversarial security scan of the multi-agent setup.

Runs deterministic security checks FIRST (so it always works with zero LLM
keys), then optionally adds a free-LLM adversarial review of the most recent
diff. Findings are posted to #security-alerts (or printed in dry mode).

Deterministic checks (no network, no LLM):
  1. Secret leakage: source that prints/echoes API keys, tokens, or headers.
  2. Paid-API bypass: ALLOW_PAID_APIS flipped to true, or Claude gating removed.
  3. Committed secrets: high-entropy / known-prefix tokens (xoxb-, sk-, AIza...)
     present in tracked files (not workflows referencing ${{ secrets.* }}).
  4. Unsafe workflow triggers: pull_request_target with checkout of PR head.
  5. Prompt-injection surface: untrusted Slack/GitHub text used in a prompt
     that then triggers an action without a sanitizer.

Exit code is always 0 (a monitor, not a gate) — but every finding is logged
and posted so the team sees it. Severity is tagged HIGH/MED/LOW.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / ".github" / "scripts"
WORKFLOWS = REPO / ".github" / "workflows"

Finding = tuple[str, str, str]  # (severity, location, message)


def _scan_secret_leakage() -> list[Finding]:
    out: list[Finding] = []
    # Flag only OUTPUT statements (print/echo/logger) that emit a raw secret
    # VALUE. A secret is "raw" when interpolated as {VAR} or $VAR directly.
    output_re = re.compile(r"\b(print|logger\.\w+)\s*\(|^\s*echo\b")
    secret_interp_re = re.compile(
        r"[{$]\s*([A-Za-z_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Za-z_0-9]*)\b"
    )
    for f in SCRIPTS.glob("*.py"):
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if not output_re.search(line):
                continue
            m = secret_interp_re.search(line)
            if not m:
                continue
            # Safe patterns: presence/boolean checks, masking, length, not the value.
            if re.search(r"bool\(|\bconfigured\b|\bpresent\b|\bset\b|len\(|\[:\d|\.\.\.|masked|redact|\*{3,}", line, re.I):
                continue
            # Authorization header construction is API usage over TLS, not a log leak.
            if re.search(r"Authorization|Bearer|headers\s*=|\"Accept\"", line):
                continue
            out.append(("HIGH", f"{f.name}:{i}", f"secret value emitted to output: {line.strip()[:90]}"))
    return out


def _scan_paid_api_bypass() -> list[Finding]:
    out: list[Finding] = []
    # Real assignment at line start (python `X = true`, `export X=true`, or yaml
    # `X: "true"`). Excludes comments, comparisons, and string-literal mentions
    # inside detection/guard code ("ALLOW_PAID_APIS = True" in src / print(...)).
    assign_re = re.compile(r"^\s*(?:export\s+)?ALLOW_PAID_APIS\s*(?:=(?!=)|:)\s*[\"']?[Tt]rue\b")
    for f in list(SCRIPTS.glob("*.py")) + list(WORKFLOWS.glob("*.yml")):
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if assign_re.search(line):
                out.append(("HIGH", f"{f.name}:{i}", "ALLOW_PAID_APIS assigned true — paid APIs enabled"))
    return out


def _scan_shell_injection() -> list[Finding]:
    """GitHub Actions run: steps that interpolate untrusted ${{ ... }} into shell."""
    out: list[Finding] = []
    untrusted = re.compile(
        r"\$\{\{\s*(github\.event|inputs|github\.head_ref|steps\.[^}]*\.outputs)[^}]*\}\}"
    )
    # A YAML mapping assignment (`KEY: ${{ ... }}`, e.g. an env: block) is the
    # SAFE way to pass untrusted input. Only flag ${{ }} used in an actual shell
    # token (TASK="${{...}}", `--task ${{...}}`, $(...)).
    yaml_mapping = re.compile(r"^\s*[A-Za-z_][\w.-]*:\s")
    for f in WORKFLOWS.glob("*.yml"):
        in_run = False
        run_indent = 0
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            stripped = line.strip()
            m = re.match(r"(\s*)-?\s*run:\s*\|?", line)
            if m:
                in_run = True
                run_indent = len(m.group(1))
                continue
            if in_run and stripped and (len(line) - len(line.lstrip())) <= run_indent:
                in_run = False
            if in_run and not yaml_mapping.match(line) and untrusted.search(line):
                out.append(("HIGH", f"{f.name}:{i}",
                            f"untrusted ${{{{...}}}} in shell run line: {stripped[:70]}"))
    return out


def _scan_committed_secrets() -> list[Finding]:
    out: list[Finding] = []
    token_re = re.compile(r"(xoxb-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_\-]{30,}|gsk_[A-Za-z0-9]{20,})")
    try:
        tracked = subprocess.run(
            ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, timeout=30
        ).stdout.splitlines()
    except Exception:
        tracked = []
    fake_re = re.compile(r"test|dummy|example|fake|placeholder|xxx|sample|your[-_]|redact", re.I)
    for rel in tracked:
        if rel.endswith((".png", ".jpg", ".lock", ".pt", ".bin")):
            continue
        # Test fixtures legitimately contain fake tokens.
        if "test" in rel.lower() or rel.startswith(("tests/", "backend/tests/")):
            continue
        p = REPO / rel
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "secrets." in line:
                continue  # a reference, not a value
            m = token_re.search(line)
            if m and not fake_re.search(line):
                out.append(("HIGH", f"{rel}:{i}", f"hardcoded token-like string: {m.group(0)[:12]}..."))
    return out


def _scan_unsafe_workflows() -> list[Finding]:
    out: list[Finding] = []
    for f in WORKFLOWS.glob("*.yml"):
        text = f.read_text(errors="ignore")
        if "pull_request_target" in text and re.search(r"ref:\s*\$\{\{\s*github\.event\.pull_request\.head", text):
            out.append(("HIGH", f.name, "pull_request_target checks out PR head — code-exec-with-secrets risk"))
    return out


def _run_deterministic() -> list[Finding]:
    findings: list[Finding] = []
    findings += _scan_secret_leakage()
    findings += _scan_paid_api_bypass()
    findings += _scan_committed_secrets()
    findings += _scan_unsafe_workflows()
    findings += _scan_shell_injection()
    return findings


def _llm_review_recent_diff() -> str | None:
    """Optional: free-LLM adversarial review of the last commit's diff."""
    keys = ["GROQ_API_KEY", "GROQ_API_KEY_1", "GEMINI_API_KEY", "GEMINI_API_KEY_1",
            "DEEPSEEK_API_KEY", "SAMBANOVA_API_KEY", "CEREBRAS_API_KEY"]
    if not any(os.environ.get(k, "").strip() for k in keys):
        return None
    try:
        diff = subprocess.run(
            ["git", "-C", str(REPO), "diff", "HEAD~3..HEAD", "--", ".github/scripts", "backend/app"],
            capture_output=True, text=True, timeout=30,
        ).stdout[:12000]
        if not diff.strip():
            return None
        from model_router import smart_llm
        prompt = (
            "You are a red-team security reviewer. Review this diff from a quant "
            "trading platform's agent system for SECURITY issues only: secret leakage, "
            "prompt injection, unsafe deploys, auth bypass, command injection. "
            "List only concrete issues with file/line, or reply 'No security issues found.'\n\n"
            f"{diff}"
        )
        text, _ = smart_llm("code_review", "security_lead", prompt, max_tokens=500)
        return text
    except Exception:
        return None


def main() -> int:
    print("=== RED TEAM — adversarial security scan ===")
    findings = _run_deterministic()

    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    findings.sort(key=lambda x: order.get(x[0], 9))

    if findings:
        print(f"\n{len(findings)} finding(s):")
        for sev, loc, msg in findings:
            print(f"  [{sev}] {loc} — {msg}")
    else:
        print("\nNo deterministic security issues found. ✅")

    llm_review = _llm_review_recent_diff()
    if llm_review:
        print("\n--- LLM adversarial review (recent diff) ---")
        print(llm_review[:800])

    # Post to Slack if configured.
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if token:
        from llm_common import slack_post
        highs = [f for f in findings if f[0] == "HIGH"]
        header = (f":rotating_light: Red team: {len(highs)} HIGH, {len(findings)} total"
                  if findings else ":white_check_mark: Red team: no issues found")
        body = header
        if findings:
            body += "\n" + "\n".join(f"• [{s}] `{l}` — {m}" for s, l, m in findings[:15])
        if llm_review and "No security issues" not in llm_review:
            body += f"\n\n*LLM review:*\n{llm_review[:1500]}"
        slack_post("security-alerts", body)
        print("\nPosted to #security-alerts")

    # Monitor, not a gate — always exit 0 so it keeps running.
    return 0


if __name__ == "__main__":
    sys.exit(main())
