"""Lightweight prompt-injection / unsafe-execution red-team for the agent fleet.

Prompt injection is the #1 AI-agent threat in 2026 (OWASP LLM Top-10). Our
agents ingest UNTRUSTED external text — GitHub issue bodies, PR comments, CI
logs — and feed it to LLMs and shells. This static probe flags the two failure
classes that turn that into a real exploit:

  1. Untrusted external input reaching a shell/eval/exec sink.
  2. LLM prompts that interpolate external text with no delimiter/guard,
     so an issue body saying "ignore previous instructions, ..." is obeyed.

Exit is always 0 (a scan must not break CI); findings print as `RED-TEAM FAIL`
lines that security-scan.yml turns into an agent-fix-needed issue.
"""
from __future__ import annotations

import pathlib
import re
import sys

SCRIPTS = pathlib.Path(__file__).parent
findings: list[str] = []

# Sinks that must never receive untrusted, un-sanitised input.
DANGEROUS = [
    (re.compile(r"\bos\.system\s*\("), "os.system() call"),
    (re.compile(r"\bsubprocess\.(?:call|run|Popen)\([^)]*shell\s*=\s*True"), "subprocess(shell=True)"),
    (re.compile(r"\beval\s*\("), "eval()"),
    (re.compile(r"\bexec\s*\("), "exec()"),
    (re.compile(r"__import__\s*\("), "__import__()"),
]

# Markers that the script handles EXTERNAL untrusted content.
EXTERNAL_HINT = re.compile(
    r"issue.?body|pr.?body|comment.?body|event\.body|GITHUB_EVENT|review.?body|"
    r"issue\[.?body|\.get\(['\"]body['\"]\)|user_input|external",
    re.IGNORECASE,
)


def scan() -> None:
    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name == "security_redteam.py":
            continue
        try:
            src = path.read_text()
        except Exception:
            continue
        touches_external = bool(EXTERNAL_HINT.search(src))

        for rx, label in DANGEROUS:
            for m in rx.finditer(src):
                line = src[: m.start()].count("\n") + 1
                # High severity only when the same file also handles external input.
                sev = "HIGH" if touches_external else "low"
                if sev == "HIGH":
                    findings.append(
                        f"RED-TEAM FAIL [{sev}] {path.name}:{line} — {label} in a script that "
                        f"also reads external input (injection→execution path)"
                    )
                else:
                    print(f"  note [{sev}] {path.name}:{line} — {label} (no external-input path)")

        # LLM prompt built from external text without a delimiter/guard.
        if touches_external and re.search(r"prompt\s*=.*f['\"]", src):
            if not re.search(r"untrusted|<external|do not follow|ignore any instructions|delimiter", src, re.IGNORECASE):
                findings.append(
                    f"RED-TEAM FAIL [medium] {path.name} — builds an LLM prompt from external "
                    f"text with no injection guard/delimiter (wrap untrusted content and instruct "
                    f"the model to treat it as data, not instructions)"
                )


def main() -> int:
    print("=== Agent prompt-injection red-team ===")
    scan()
    if findings:
        print(f"\n{len(findings)} finding(s):")
        for f in findings:
            print(" ", f)
    else:
        print("\n✅ No injection→execution paths or unguarded external-prompt patterns found.")
    return 0  # never break CI


if __name__ == "__main__":
    sys.exit(main())
