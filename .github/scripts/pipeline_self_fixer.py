"""
Pipeline Self-Fixer — diagnoses and repairs broken CI for the whole app.

Triggered by the `pipeline-self-fixer` workflow on ANY workflow_run that
concludes in failure. It:

  1. Fetches the failed run's job logs via the GitHub API.
  2. Runs a battery of static repairs on the repo that catch the failure
     classes that actually took this project down:
       • collapsed YAML `with:` blocks (uses: action@v4   key: val)
       • empty / truncated workflow files
       • hardcoded `ref: <feature-branch>` checkout pins
       • Python scripts that no longer compile
       • workflow YAML that no longer parses
  3. Classifies the log against known runtime-failure signatures and emits
     a diagnosis (missing dep, bad secret, import error, rate limit, …).
  4. If it made repairs, writes them to disk so the workflow can open a PR.
  5. Posts a summary to Slack #ci-failures.

No mock data: if it cannot determine a cause it says so and leaves a precise
diagnosis for a human rather than inventing a fix.

Env:
  GH_TOKEN, GH_REPO        — read run logs, used by the workflow to open the PR
  RUN_ID                   — the failed run id (the workflow passes this)
  SLACK_BOT_TOKEN          — optional, posts diagnosis to #ci-failures
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WF_DIR = REPO_ROOT / ".github" / "workflows"
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


# ─────────────────────────────────────────────────────────────────────────────
# GitHub API
# ─────────────────────────────────────────────────────────────────────────────

def _gh(path: str, raw: bool = False):
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
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read() if raw else json.loads(r.read())
    except Exception as exc:
        print(f"  gh {path} failed: {exc}", flush=True)
        return None


def fetch_failed_log(run_id: str) -> str:
    """Download the run's logs zip and concatenate the text."""
    blob = _gh(f"/actions/runs/{run_id}/logs", raw=True)
    if not blob:
        return ""
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
        chunks = []
        for name in zf.namelist():
            if name.endswith(".txt"):
                chunks.append(zf.read(name).decode("utf-8", "replace"))
        return "\n".join(chunks)
    except Exception as exc:
        print(f"  could not unzip logs: {exc}", flush=True)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Static repairs (the failure classes that hit THIS app)
# ─────────────────────────────────────────────────────────────────────────────

_COLLAPSED = re.compile(r"(uses:\s*\S+@\S+)\s{2,}(\w[\w.-]*:)")


def fix_collapsed_with_blocks() -> list[str]:
    """`- uses: actions/checkout@v4   fetch-depth: 1` → proper with: block."""
    fixed = []
    for f in WF_DIR.glob("*.yml"):
        text = f.read_text()
        new_lines = []
        changed = False
        for line in text.split("\n"):
            m = _COLLAPSED.search(line)
            if m:
                indent = len(line) - len(line.lstrip())
                lead = line[:indent]
                # everything after the action ref is collapsed key:val pairs
                uses_part = line[:line.index(m.group(2))].rstrip()
                rest = line[line.index(m.group(2)):].strip()
                new_lines.append(uses_part)
                new_lines.append(f"{lead}  with:")
                # split multiple "k: v   k2: v2" pairs heuristically (usually one)
                new_lines.append(f"{lead}    {rest}")
                changed = True
            else:
                new_lines.append(line)
        if changed:
            f.write_text("\n".join(new_lines))
            fixed.append(f.name)
    return fixed


def fix_empty_workflows() -> list[str]:
    """Restore empty/truncated workflow files from the newest non-empty git blob."""
    fixed = []
    for f in WF_DIR.glob("*.yml"):
        if f.stat().st_size >= 80:
            continue
        rel = f.relative_to(REPO_ROOT)
        # walk history for a non-trivial version
        try:
            shas = subprocess.check_output(
                ["git", "log", "--format=%h", "--", str(rel)],
                cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().split()
        except Exception:
            shas = []
        for sha in shas:
            try:
                blob = subprocess.check_output(
                    ["git", "show", f"{sha}:{rel}"],
                    cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode()
            except Exception:
                continue
            if len(blob) >= 200:
                f.write_text(blob)
                fixed.append(f.name)
                break
    return fixed


def fix_branch_ref_pins() -> list[str]:
    """Remove hardcoded `ref: claude/...` checkout pins so scheduled runs work."""
    fixed = []
    for f in WF_DIR.glob("*.yml"):
        lines = f.read_text().split("\n")
        out = [l for l in lines
               if not (l.strip().startswith("ref:") and "claude/advanced-trading" in l)]
        if len(out) != len(lines):
            f.write_text("\n".join(out))
            fixed.append(f.name)
    return fixed


def verify_yaml_parses() -> list[str]:
    """Return list of workflow files that still fail to parse."""
    bad = []
    try:
        import yaml
    except ImportError:
        return []
    for f in WF_DIR.glob("*.yml"):
        try:
            yaml.safe_load(f.read_text())
        except Exception:
            bad.append(f.name)
    return bad


def verify_scripts_compile() -> list[str]:
    """Return list of referenced scripts that fail to compile."""
    import py_compile
    bad = []
    referenced = set()
    for f in WF_DIR.glob("*.yml"):
        referenced |= set(re.findall(r"\.github/scripts/([a-zA-Z_]+\.py)", f.read_text()))
    for name in sorted(referenced):
        p = SCRIPTS_DIR / name
        if not p.exists():
            bad.append(f"{name} (MISSING)")
            continue
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as exc:
            bad.append(f"{name} ({str(exc)[:60]})")
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# Runtime-log classification
# ─────────────────────────────────────────────────────────────────────────────

SIGNATURES = [
    (r"ModuleNotFoundError: No module named '([^']+)'",
     "Missing Python dependency: {0}. Add it to the workflow's pip install step."),
    (r"ImportError: ([^\n]+)",
     "Import error: {0}. Check dependency versions / circular imports."),
    (r"NameError: name ['\"]([A-Za-z_][\w]*)['\"]",
     "Undefined name: {0}. A refactor likely left a dangling reference."),
    (r"KeyError: ['\"]?([A-Za-z_][\w]*)",
     "Missing key/env var: {0}."),
    (r"401|invalid_auth|not_authed",
     "Auth failure — a secret (SLACK_BOT_TOKEN / ALPACA_*) is missing, expired, or wrong."),
    (r"403\b|Forbidden",
     "403 Forbidden — credential lacks scope/permission, or the endpoint is blocked."),
    (r"429|rate.?limit|Too Many Requests",
     "Rate limited — add backoff or reduce call frequency."),
    (r"yaml(\.|\s).*(could not|mapping values|scanner|parser)",
     "Workflow YAML is malformed — see verify_yaml_parses repairs above."),
    (r"SECRET_KEY|SecretKey.*at least|must be 32",
     "SECRET_KEY invalid/too short — workflow must export a 32-byte hex key."),
    (r"No module named 'app'|PYTHONPATH",
     "PYTHONPATH not pointing at backend/ — add PYTHONPATH: ${{ github.workspace }}/backend."),
    (r"could not connect|Connection refused|getaddrinfo|Temporary failure in name resolution",
     "Network/DNS failure reaching an external API — usually transient; add a retry."),
    (r"Traceback \(most recent call last\)",
     "Unhandled Python exception — see the traceback in the captured log."),
]


def classify(log: str) -> list[str]:
    diagnoses = []
    for pat, msg in SIGNATURES:
        m = re.search(pat, log, re.IGNORECASE)
        if m:
            try:
                diagnoses.append(msg.format(*m.groups()))
            except Exception:
                diagnoses.append(msg)
    # dedupe, keep order
    seen, out = set(), []
    for d in diagnoses:
        if d not in seen:
            seen.add(d); out.append(d)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Slack
# ─────────────────────────────────────────────────────────────────────────────

def post_slack(text: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        print("  (no SLACK_BOT_TOKEN — skipping Slack post)", flush=True)
        return
    try:
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=json.dumps({"channel": "#ci-failures", "text": text}).encode(),
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
            if not body.get("ok"):
                print(f"  slack error: {body.get('error')}", flush=True)
    except Exception as exc:
        print(f"  slack post failed: {exc}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    run_id = os.environ.get("RUN_ID", "").strip()
    wf_name = os.environ.get("FAILED_WORKFLOW", "unknown")
    print(f"Pipeline self-fixer — failed workflow '{wf_name}' run {run_id or '(none)'}", flush=True)

    log = fetch_failed_log(run_id) if run_id else ""
    if log:
        print(f"  fetched {len(log)} bytes of log", flush=True)
    else:
        print("  no log available (running static repairs only)", flush=True)

    # ── Static repairs ───────────────────────────────────────────────────────
    repairs: dict[str, list[str]] = {
        "collapsed_with_blocks": fix_collapsed_with_blocks(),
        "empty_workflows": fix_empty_workflows(),
        "branch_ref_pins": fix_branch_ref_pins(),
    }
    repairs = {k: v for k, v in repairs.items() if v}

    still_broken_yaml = verify_yaml_parses()
    still_broken_scripts = verify_scripts_compile()

    # ── Log classification ───────────────────────────────────────────────────
    diagnoses = classify(log) if log else []

    # ── Build report ─────────────────────────────────────────────────────────
    lines = [f":rotating_light: *Pipeline Self-Fixer* — workflow `{wf_name}` failed"]
    if run_id:
        repo = os.environ.get("GH_REPO", "")
        lines.append(f"<https://github.com/{repo}/actions/runs/{run_id}|view failed run →>")

    if repairs:
        lines.append("\n*Auto-repairs applied:*")
        for kind, files in repairs.items():
            lines.append(f"  • {kind}: {', '.join(files)}")
    else:
        lines.append("\n_No static repairs were needed._")

    if diagnoses:
        lines.append("\n*Log diagnosis:*")
        for d in diagnoses[:6]:
            lines.append(f"  • {d}")

    if still_broken_yaml:
        lines.append(f"\n:warning: Still-unparseable YAML: {', '.join(still_broken_yaml)}")
    if still_broken_scripts:
        lines.append(f":warning: Scripts not compiling: {', '.join(still_broken_scripts)}")

    made_changes = bool(repairs)
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    post_slack(report)

    # Tell the workflow whether to open a PR (writes to GITHUB_OUTPUT)
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"made_changes={'true' if made_changes else 'false'}\n")
            fh.write(f"diagnosis={'; '.join(diagnoses)[:300] or 'no signature matched'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
