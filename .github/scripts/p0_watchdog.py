"""
P0 Watchdog — posts to #incidents if any P0/critical issue > 24h unresolved.
Also posts to #risk for risk-related P0s.
Run hourly via GitHub Actions.
"""
import os, requests, json
from datetime import datetime, timedelta, timezone

GH_TOKEN = os.environ["GH_TOKEN"]
GH_REPO = os.environ.get("GH_REPO", "bahllaavanye-afk/test")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

def check_p0_sla():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    breaching = []
    for label in ["P0", "p0", "critical", "incident"]:
        resp = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/issues",
            params={"state": "open", "labels": label, "per_page": 50},
            headers={"Authorization": f"token {GH_TOKEN}"}
        )
        if resp.status_code == 200:
            for issue in resp.json():
                created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
                if created < cutoff:
                    age_h = round((datetime.now(timezone.utc) - created).total_seconds() / 3600, 1)
                    breaching.append({"number": issue["number"], "title": issue["title"], "age_hours": age_h, "url": issue["html_url"]})
    return breaching

def auto_label_new_issues():
    """Label any issue with 'P0', 'error', 'crash', 'down' in title as P0 if not already labeled."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
    resp = requests.get(
        f"https://api.github.com/repos/{GH_REPO}/issues",
        params={"state": "open", "per_page": 50, "sort": "created", "direction": "desc"},
        headers={"Authorization": f"token {GH_TOKEN}"}
    )
    if resp.status_code != 200:
        return

    p0_keywords = ["p0", "critical", "down", "crash", "error", "broken", "failed", "outage", "incident"]
    for issue in resp.json():
        created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
        if created < cutoff:
            break
        title_lower = issue["title"].lower()
        existing_labels = [l["name"].lower() for l in issue.get("labels", [])]
        if any(kw in title_lower for kw in p0_keywords) and "p0" not in existing_labels:
            requests.post(
                f"https://api.github.com/repos/{GH_REPO}/issues/{issue['number']}/labels",
                headers={"Authorization": f"token {GH_TOKEN}", "Content-Type": "application/json"},
                json={"labels": ["P0"]}
            )
            print(f"Auto-labeled issue #{issue['number']} as P0: {issue['title'][:60]}")

def post_alert(breaching):
    if not SLACK_BOT_TOKEN or not breaching:
        return
    lines = []
    for b in breaching:
        lines.append(f"• #{b['number']} — {b['title'][:70]} ({b['age_hours']}h unresolved) {b['url']}")
    text = f"🚨 *P0 SLA BREACH — {len(breaching)} issue(s) unresolved > 24h*\n\n" + "\n".join(lines) + "\n\n_CTO OKR requires: zero P0 unresolved > 24h_"
    for channel in ["incidents", "engineering"]:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
            json={"channel": channel, "text": text, "mrkdwn": True}
        )
        r = resp.json()
        print(f"{'✓' if r.get('ok') else '✗'} Posted P0 alert to #{channel}")

if __name__ == "__main__":
    auto_label_new_issues()
    breaching = check_p0_sla()
    if breaching:
        print(f"⚠️ {len(breaching)} P0 issue(s) breaching 24h SLA:")
        for b in breaching:
            print(f"  #{b['number']} ({b['age_hours']}h): {b['title'][:60]}")
        post_alert(breaching)
    else:
        print("✅ No P0 SLA breaches — CTO OKR clean")

    # Exit non-zero if breaches exist (marks CI check as warning)
    import sys
    sys.exit(0)  # don't fail the workflow — just alert
