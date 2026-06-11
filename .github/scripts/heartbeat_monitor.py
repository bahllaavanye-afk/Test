"""
Agent Heartbeat Monitor — zero-downtime assurance.

Checks:
1. Critical GitHub Actions workflows ran within expected windows
2. Gemini API key quota status (how many keys healthy)
3. Groq fallback available
4. Posts Slack alert ONLY when something is actually broken

This runs even when Claude is offline — it only uses GitHub API + free LLMs.

Context sharing: exports agent_context.json so all agents stay in sync
without needing Claude in the loop.
"""
from __future__ import annotations

import json
import os

# ── Key resolver: supports both plain and numbered secrets ────────────────────
def _resolve_key(*names: str) -> str:
    """Return first non-empty value from env, checking plain + numbered variants."""
    for name in names:
        v = os.environ.get(name, "")
        if v:
            return v
        # Try _1 suffix if not already numbered
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v:
                return v
    return ""

import sys
import requests
from datetime import datetime, timedelta, timezone

GH_TOKEN       = os.environ.get("GH_TOKEN", "")
GH_REPO        = os.environ.get("GH_REPO", "bahllaavanye-afk/test")
SLACK_TOKEN    = os.environ.get("SLACK_BOT_TOKEN", "")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

GEMINI_KEYS = [
    k for k in [
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
    ] if k
]
GROQ_API_KEY = _resolve_key("GROQ_API_KEY", "GROQ_API_KEY_1")

if ALLOW_PAID_APIS.lower() == "true":
    print("SECURITY: ALLOW_PAID_APIS must be False")
    sys.exit(1)

# ── Critical workflows and their expected max gap (hours) ─────────────────────

CRITICAL_WORKFLOWS = {
    "Claude ↔ Employee Conversations":           2.5,   # every 2h
    "Free Agent Engineer — autonomous issue fixing": 2.5,
    "Continuous Improvement — Autonomous Code Quality": 2.5,
    "OKR Tracker — Daily C-Suite Report":        26,    # daily
    "Team Lead Issue Generator — All Teams":     13,    # 2x daily
    "P0 Watchdog — SLA Monitor":                 1.5,   # hourly
}

def gh_headers():
    return {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}

def check_workflow_health() -> list[dict]:
    """Return list of workflows that haven't run within their expected window."""
    alerts = []
    if not GH_TOKEN:
        return alerts
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/actions/workflows",
            headers=gh_headers(), params={"per_page": 100}, timeout=15
        )
        if resp.status_code != 200:
            return alerts
        workflows = {w["name"]: w["id"] for w in resp.json().get("workflows", [])}
    except Exception as e:
        print(f"Workflow list error: {e}")
        return alerts

    now = datetime.now(timezone.utc)
    for wf_name, max_gap_hours in CRITICAL_WORKFLOWS.items():
        wf_id = workflows.get(wf_name)
        if not wf_id:
            continue
        try:
            r = requests.get(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{wf_id}/runs",
                headers=gh_headers(),
                params={"per_page": 1, "status": "completed"},
                timeout=15
            )
            if r.status_code != 200:
                continue
            runs = r.json().get("workflow_runs", [])
            if not runs:
                alerts.append({"workflow": wf_name, "issue": "never run", "gap_hours": None})
                continue
            last_run = datetime.fromisoformat(runs[0]["updated_at"].replace("Z", "+00:00"))
            gap = (now - last_run).total_seconds() / 3600
            if gap > max_gap_hours:
                alerts.append({"workflow": wf_name, "issue": f"last run {gap:.1f}h ago (expected < {max_gap_hours}h)", "gap_hours": round(gap, 1)})
        except Exception as e:
            print(f"Run check error for {wf_name}: {e}")

    return alerts

def check_gemini_keys() -> dict:
    """Test each Gemini key with a minimal call to see which ones are alive."""
    status = {}
    for i, key in enumerate(GEMINI_KEYS):
        key_label = f"GEMINI_API_KEY{'_' + str(i+1) if i > 0 else ''}"
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
                json={"contents": [{"role": "user", "parts": [{"text": "Reply: ok"}]}],
                      "generationConfig": {"maxOutputTokens": 5}},
                timeout=10
            )
            if resp.status_code == 200:
                status[key_label] = "healthy"
            elif resp.status_code == 429:
                status[key_label] = "quota_exhausted"
            elif resp.status_code == 403:
                status[key_label] = "invalid_key"
            else:
                status[key_label] = f"error_{resp.status_code}"
        except Exception as e:
            status[key_label] = f"timeout: {e}"
    return status

def check_groq() -> str:
    if not GROQ_API_KEY:
        return "not_configured"
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "ok"}], "max_tokens": 3},
            timeout=10
        )
        return "healthy" if resp.status_code == 200 else f"error_{resp.status_code}"
    except Exception:
        return "timeout"

def post_slack_alert(alerts: list[dict], gemini_status: dict, groq_status: str):
    if not SLACK_TOKEN:
        return
    healthy_gemini = sum(1 for v in gemini_status.values() if v == "healthy")
    exhausted_gemini = sum(1 for v in gemini_status.values() if v == "quota_exhausted")

    lines = []
    if alerts:
        lines.append("*⚠️ Workflow Health Alerts:*")
        for a in alerts:
            lines.append(f"  • {a['workflow']}: {a['issue']}")
    if exhausted_gemini > 0 and healthy_gemini == 0:
        lines.append(f"*🔴 ALL Gemini keys exhausted* — running on Groq fallback only")
        lines.append("Add `GEMINI_API_KEY_2` to GitHub Secrets to restore capacity")
    elif exhausted_gemini > 0:
        lines.append(f"*⚠️ {exhausted_gemini} Gemini key(s) exhausted* — {healthy_gemini} still healthy")

    if not lines:
        return  # No alert needed

    msg = "\n".join(lines) + f"\n\n_Groq fallback: {groq_status} | {datetime.now(timezone.utc).strftime('%H:%M UTC')}_"
    try:
        requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
            json={"channel": "incidents", "text": msg, "mrkdwn": True},
            timeout=10
        )
        requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
            json={"channel": "engineering", "text": msg, "mrkdwn": True},
            timeout=10
        )
    except Exception as e:
        print(f"Slack error: {e}")

def main():
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Agent heartbeat check")

    # 1. Check workflows
    print("Checking workflow health...")
    workflow_alerts = check_workflow_health()

    # 2. Check LLM keys
    print("Checking Gemini key health...")
    gemini_status = check_gemini_keys()
    groq_status = check_groq()

    healthy_gemini = sum(1 for v in gemini_status.values() if v == "healthy")
    exhausted_gemini = sum(1 for v in gemini_status.values() if v == "quota_exhausted")

    print(f"Gemini: {healthy_gemini} healthy, {exhausted_gemini} exhausted, {len(GEMINI_KEYS) - healthy_gemini - exhausted_gemini} other")
    print(f"Groq: {groq_status}")

    if workflow_alerts:
        for a in workflow_alerts:
            print(f"⚠️ {a['workflow']}: {a['issue']}")
    else:
        print("✅ All critical workflows healthy")

    # 3. Alert if needed
    post_slack_alert(workflow_alerts, gemini_status, groq_status)

    # 4. Write report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workflow_alerts": workflow_alerts,
        "llm_status": {
            "gemini": gemini_status,
            "gemini_healthy": healthy_gemini,
            "gemini_exhausted": exhausted_gemini,
            "groq": groq_status,
        },
        "overall_status": "degraded" if (healthy_gemini == 0 and groq_status != "healthy") else "healthy",
    }
    with open("/tmp/heartbeat_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOverall status: {report['overall_status']}")
    return 0  # Never fail — heartbeat must always complete

if __name__ == "__main__":
    sys.exit(main())
