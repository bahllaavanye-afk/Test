#!/usr/bin/env python3
"""
Autonomous quality review for the Slack agent team.
Checks slack_agent_team.py for hardcoded content and quality violations.
Exits 0 if clean, exits 1 if critical issues found (blocks merge).

Checks performed:
  CRITICAL    - ALLOW_PAID_APIS is True
  SECRET      - Hardcoded API key patterns (sk-*, ghp_*, xoxb-*)
  HARDCODED   - Employee function returns Post() without any LLM call
  SHORT_PROMPT - employee_provider_prompt called with task string < 50 chars
"""
import ast
import json
import re
import sys
from pathlib import Path

SCRIPT = Path(".github/scripts/slack_agent_team.py")
REPORT = Path("review_report.json")

# Sentinel: patterns that look like real secrets (long enough to be real tokens).
SECRET_PATTERNS = [
    r"\bsk-[A-Za-z0-9]{20,}",
    r"\bghp_[A-Za-z0-9]{20,}",
    r"\bxoxb-[0-9]+-[A-Za-z0-9]+",
]

# Name fragments that identify agent / employee functions worth checking.
EMPLOYEE_NAME_FRAGMENTS = [
    "_data", "_finance", "_compliance", "_security",
    "_options", "_rl", "_deploy", "_eng", "_ceo",
    "_ml", "_research", "_backend", "_results",
]

issues: list[dict] = []
warnings: list[dict] = []


def add_issue(kind: str, msg: str, line: int) -> None:
    issues.append({"type": kind, "msg": msg, "line": line})


def add_warning(kind: str, msg: str, line: int) -> None:
    warnings.append({"type": kind, "msg": msg, "line": line})


# ---------------------------------------------------------------------------
# Load source
# ---------------------------------------------------------------------------
if not SCRIPT.exists():
    print(f"ERROR: {SCRIPT} not found — nothing to review")
    REPORT.write_text(json.dumps({"issues": [], "warnings": [], "total_issues": 0, "status": "SKIP"}, indent=2))
    sys.exit(0)

src = SCRIPT.read_text()
lines = src.splitlines()

try:
    tree = ast.parse(src)
except SyntaxError as exc:
    add_issue("SYNTAX", f"Syntax error in {SCRIPT}: {exc}", exc.lineno or 0)
    report = {"issues": issues, "warnings": warnings, "total_issues": len(issues), "status": "FAIL"}
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nCRITICAL: cannot parse {SCRIPT} — syntax error")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Check 1: ALLOW_PAID_APIS must not be True
# ---------------------------------------------------------------------------
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "ALLOW_PAID_APIS":
                if isinstance(node.value, ast.Constant) and node.value.value is True:
                    add_issue(
                        "CRITICAL",
                        "ALLOW_PAID_APIS is True — must be False to prevent unexpected spend",
                        node.lineno,
                    )


# ---------------------------------------------------------------------------
# Check 2: Employee functions — hardcoded Post() and short task prompts
# ---------------------------------------------------------------------------
employee_fns = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef)
    and any(frag in node.name for frag in EMPLOYEE_NAME_FRAGMENTS)
]

for fn in employee_fns:
    fn_src = "\n".join(lines[fn.lineno - 1 : fn.end_lineno])

    has_llm_call = (
        "employee_provider_prompt" in fn_src
        or "moa_employee_prompt" in fn_src
    )
    has_post_literal = bool(re.search(r"Post\s*\(", fn_src))

    # HARDCODED: builds Post() but never calls an LLM
    if has_post_literal and not has_llm_call:
        add_issue(
            "HARDCODED",
            f"{fn.name}() returns Post() with no LLM call — responses will be static",
            fn.lineno,
        )

    # SHORT_PROMPT: task string passed to LLM provider is suspiciously short
    for match in re.finditer(
        r"employee_provider_prompt\([^,]+,\s*[\"'](.{0,49})[\"']", fn_src
    ):
        task_text = match.group(1)
        if len(task_text) < 50:
            add_issue(
                "SHORT_PROMPT",
                f"{fn.name}(): task prompt is only {len(task_text)} chars — "
                f"likely too vague for quality output: '{task_text}'",
                fn.lineno,
            )


# ---------------------------------------------------------------------------
# Check 3: Hardcoded secret patterns (skip commented-out lines)
# ---------------------------------------------------------------------------
for lineno, line in enumerate(lines, start=1):
    stripped = line.lstrip()
    if stripped.startswith("#"):
        # Whole line is a comment — skip
        continue

    for pattern in SECRET_PATTERNS:
        match = re.search(pattern, line)
        if not match:
            continue

        # Determine whether the match sits after an inline comment marker
        match_start = match.start()
        comment_pos = line.find("#")
        if comment_pos != -1 and comment_pos < match_start:
            # Secret appears in a comment — skip
            continue

        add_issue(
            "SECRET",
            f"Potential hardcoded secret at line {lineno}: '{match.group()[:12]}...'",
            lineno,
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
status = "FAIL" if issues else "PASS"
report = {
    "issues": issues,
    "warnings": warnings,
    "total_issues": len(issues),
    "total_warnings": len(warnings),
    "status": status,
    "checked_file": str(SCRIPT),
}
REPORT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

if not issues:
    print("\nAll quality checks passed")
    sys.exit(0)

critical = [i for i in issues if i["type"] in ("CRITICAL", "SECRET", "SYNTAX")]
non_critical = [i for i in issues if i["type"] not in ("CRITICAL", "SECRET", "SYNTAX")]

if non_critical and not critical:
    print(f"\n{len(non_critical)} issue(s) found (non-blocking) — fix before next deploy")
    sys.exit(0)

print(f"\n{len(critical)} CRITICAL issue(s) found — blocking push")
sys.exit(1)
