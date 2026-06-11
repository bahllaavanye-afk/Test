import os, json, requests
from datetime import datetime, timedelta, timezone

GH_TOKEN = os.environ["GH_TOKEN"]
GH_REPO = os.environ.get("GH_REPO", "bahllaavanye-afk/test")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

def get_commits_last_24h():
    """Count commits across all branches in last 24h"""
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    resp = requests.get(
        f"https://api.github.com/repos/{GH_REPO}/commits",
        params={"since": since, "per_page": 100, "sha": "main"},
        headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    )
    if resp.status_code != 200:
        return 0
    return len(resp.json())

def get_p0_issues_breaching():
    """Find open issues labeled P0 or critical that are older than 24h"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    breaching = []
    for label in ["P0", "p0", "critical", "incident"]:
        resp = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/issues",
            params={"state": "open", "labels": label, "per_page": 50},
            headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        )
        if resp.status_code == 200:
            for issue in resp.json():
                created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
                if created < cutoff:
                    breaching.append({"number": issue["number"], "title": issue["title"], "age_hours": round((datetime.now(timezone.utc) - created).total_seconds() / 3600, 1)})
    return breaching

def get_open_issues_count():
    resp = requests.get(f"https://api.github.com/repos/{GH_REPO}", headers={"Authorization": f"token {GH_TOKEN}"})
    if resp.status_code == 200:
        return resp.json().get("open_issues_count", 0)
    return 0

def get_agent_fix_issues():
    """Count issues labeled agent-fix-needed (auto-resolution queue)"""
    resp = requests.get(
        f"https://api.github.com/repos/{GH_REPO}/issues",
        params={"state": "open", "labels": "agent-fix-needed", "per_page": 100},
        headers={"Authorization": f"token {GH_TOKEN}"}
    )
    return len(resp.json()) if resp.status_code == 200 else 0

def get_investor_pipeline():
    """Read investor pipeline from data/investor_pipeline.json"""
    try:
        with open("data/investor_pipeline.json") as f:
            data = json.load(f)
        stages = {}
        for inv in data["pipeline"]:
            s = inv["stage"]
            stages[s] = stages.get(s, 0) + 1
        return {
            "total": len(data["pipeline"]),
            "target": data["target"],
            "stages": stages,
            "series_a_target_date": data["series_a_target_date"],
            "days_to_target": (datetime.strptime(data["series_a_target_date"], "%Y-%m-%d") - datetime.now()).days
        }
    except Exception:
        return {"total": 0, "target": 10, "stages": {}, "days_to_target": 90}

def get_strategy_count():
    """Count strategy files in backend"""
    import glob
    manual = glob.glob("backend/app/strategies/manual/*.py")
    ml = glob.glob("backend/app/strategies/ml_enhanced/*.py")
    return len([f for f in manual if not f.endswith("__init__.py")]) + len([f for f in ml if not f.endswith("__init__.py")])

def get_workflow_count():
    import glob
    return len(glob.glob(".github/workflows/*.yml"))

def build_okr_report(commits, p0_breaching, pipeline, strategy_count, workflow_count, open_issues, agent_issues, report_type="daily"):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # CEO OKR status
    pipeline_ok = pipeline["total"] >= pipeline["target"]
    ceo_emoji = "✅" if pipeline_ok else "⚠️"
    days_left = pipeline.get("days_to_target", 90)

    # CTO OKR status
    commits_ok = commits >= 50
    p0_ok = len(p0_breaching) == 0
    cto_emoji = "✅" if (commits_ok and p0_ok) else ("⚠️" if commits_ok else "🔴")

    # Pipeline stages
    stage_text = " → ".join([f"{k}: {v}" for k, v in sorted(pipeline.get("stages", {}).items())])

    # P0 breach detail
    p0_text = ""
    if p0_breaching:
        p0_lines = [f"  • #{i['number']} ({i['age_hours']}h) {i['title'][:60]}" for i in p0_breaching[:3]]
        p0_text = f"\n*P0 Breaches:*\n" + "\n".join(p0_lines)

    title = "📊 QuantEdge Daily OKR Report" if report_type == "daily" else "📊 QuantEdge Weekly C-Suite OKR Sync"

    text = f"""*{title}*  _as of {now}_

*OKR 1 — CEO (Investor Pipeline)*  {ceo_emoji}
> Active pipeline: *{pipeline['total']} / {pipeline['target']} target*  |  Series A target: {pipeline['series_a_target_date']} ({days_left}d away)
> Stages: {stage_text or 'No investors tracked'}

*OKR 1 — CTO (Engineering Velocity)*  {cto_emoji}
> Commits last 24h: *{commits}* {'✅' if commits_ok else '🔴 (target: ≥50)'}  |  P0 breaches: *{len(p0_breaching)}* {'✅' if p0_ok else '🔴'}
{p0_text}
*Platform Health*
> Strategies in repo: *{strategy_count}*  |  Workflows: *{workflow_count}*
> Open issues: *{open_issues}*  |  Agent-fix queue: *{agent_issues}*

_OKR targets: pipeline ≥10, Series A by D90, ≥50 commits/day, zero P0 >24h_"""
    return text

def post_to_slack(text, channel="okr-updates"):
    if not SLACK_BOT_TOKEN:
        print("No SLACK_BOT_TOKEN — printing report instead:")
        print(text)
        return
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        json={"channel": channel, "text": text, "mrkdwn": True}
    )
    result = resp.json()
    if result.get("ok"):
        print(f"✓ Posted OKR report to #{channel}")
    else:
        print(f"✗ Slack error: {result.get('error')} — printing report:\n{text}")

if __name__ == "__main__":
    import sys
    report_type = "weekly" if "--weekly" in sys.argv else "daily"

    print("Collecting OKR metrics...")
    commits = get_commits_last_24h()
    p0_breaching = get_p0_issues_breaching()
    pipeline = get_investor_pipeline()
    strategy_count = get_strategy_count()
    workflow_count = get_workflow_count()
    open_issues = get_open_issues_count()
    agent_issues = get_agent_fix_issues()

    report = build_okr_report(commits, p0_breaching, pipeline, strategy_count, workflow_count, open_issues, agent_issues, report_type)
    print(report)

    # Post to multiple channels
    post_to_slack(report, "okr-updates")
    if report_type == "weekly":
        post_to_slack(report, "general")

    # Write JSON summary for GitHub step summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ceo_okr1": {"pipeline_total": pipeline["total"], "target": pipeline["target"], "achieved": pipeline["total"] >= pipeline["target"]},
        "cto_okr1": {"commits_24h": commits, "p0_breaches": len(p0_breaching), "achieved": commits >= 50 and len(p0_breaching) == 0},
        "p0_breaching": p0_breaching,
        "platform": {"strategies": strategy_count, "workflows": workflow_count, "open_issues": open_issues}
    }
    with open("/tmp/okr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary written to /tmp/okr_summary.json")
