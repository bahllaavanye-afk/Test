# Agent Workflow System — Operating Manual

> Engineering org = 92 autonomous Claude agents.
> Each agent reads its CLAUDE.md role file, picks a task from Notion or GitHub Issues,
> writes code, commits, and reports out to Slack + Google Docs.

---

## Communication stack (per artifact type)

| Artifact | Primary tool | Backup / mirror | Cadence | Owner |
|----------|--------------|-----------------|---------|-------|
| **Tasks / tickets** | GitHub Issues (source of truth) | Notion DB (Engineering Tasks) | Continuous, bi-dir sync | All ICs |
| **Daily standups** | Slack #engineering-standup | Google Doc rolling minutes per squad | 13:00 UTC daily | Squad leads |
| **Alpha proposals & reviews** | Slack #alpha-research | Google Doc Alpha Reviews + Notion Strategy Reviews DB | 17:00 UTC daily | Alpha squad |
| **Daily P&L** | Slack #pnl-daily | Google Sheet PnL Daily + Notion P&L DB | 21:00 UTC daily | CTO + PM Alpha |
| **Code reviews** | GitHub PR comments | Slack notification on @mention | Continuous | All ICs |
| **Deploy notifications** | Slack #deploys | GitHub Actions logs | On every deploy | DevOps |
| **Incidents & postmortems** | Slack #incidents | Google Doc Postmortems (per incident) + Notion Incidents DB | Within 1hr of detection | On-call lead |
| **CI failures** | Slack #ci-failures | GitHub Actions | Within 30s of failure | CTO (auto) |
| **ML training results** | Slack #ml-experiments | Google Sheet Experiments + MLflow UI | On every training run completion | ML squad |
| **Research papers** | Notion Research Queue | Google Drive folder "Research Library" | Continuous | VP Research |
| **C-suite weekly sync** | Google Doc Weekly Minutes | Slack #leadership-summary | Monday 14:00 UTC | CEO + CTO + CFO |
| **Monthly board meeting** | Google Slides board deck | Google Doc minutes | First Tuesday | CEO + CFO |
| **Quarterly OKRs** | Google Doc OKR Tracker | Notion OKRs DB | Quarterly | Each VP |
| **Onboarding** | Google Doc Employee Handbook | Notion Onboarding DB | On hire | VP Eng |
| **Architecture decisions (ADRs)** | Google Doc per ADR | GitHub `docs/adr/` folder | Per decision | Architects |

---

## Per-squad Notion + Slack + Google Docs setup

Each of the 16 squads has:

```
Slack:
  • #squad-<name>          (private channel for the squad)
  • #squad-<name>-standup  (daily standup posts)

Notion:
  • Squad page with:
    - Active sprint board (linked to global Engineering Tasks DB)
    - OKR tracker
    - Skills matrix
    - On-call rotation

Google Drive (folder per squad):
  • Squad/
    ├── Standup minutes (rolling doc, 1 per squad)
    ├── Postmortems/      (one doc per incident)
    ├── ADRs/             (one doc per architectural decision)
    ├── Onboarding/       (per new hire to squad)
    └── Show-and-tell/    (weekly slides)
```

---

## Agent operating loop (every agent runs this 24×7)

```python
while True:
    # 1. Read role
    role = read_claude_md(my_role_path)

    # 2. Pull next task
    task = notion.get_next_open_task(filter={"role": role.name, "status": "Backlog"})
    if not task:
        task = github.get_next_open_issue(labels=[f"role:{role.name}", "priority:p0"])
    if not task:
        # No assigned work — do background investigation
        task = pick_proactive_task(role)

    # 3. Mark in progress
    notion.update(task, status="In Progress", assignee=my_name)
    github.add_comment(task, f"@{my_name} picked up — ETA {role.eta_default}")

    # 4. Implement
    changes = implement(task, constraints=role.safe_files)
    if changes.violates_safe_files:
        notion.update(task, status="Blocked", note="Out of role scope")
        slack.post(role.channel, f"⚠️ Task {task.id} needs different role")
        continue

    # 5. Test locally
    if not run_tests(changes.touched_modules):
        fix_iteratively(max_attempts=3)

    # 6. Commit + push
    commit(changes, message=f"[{role.name}] {task.title}\n\nRefs #{task.github_number}")
    push()

    # 7. Open PR or push direct if trusted
    if role.can_push_direct:
        wait_for_ci()
        if ci_passed():
            merge()
        else:
            autofix_ci()
    else:
        pr = open_pr()
        request_review(reviewers=role.required_reviewers)

    # 8. Report out
    slack.post(role.report_channel, task.summary)
    google_docs.log_standup(role.squad, shipped=[task.title])
    notion.update(task, status="Done")
```

---

## Slack channel structure (production)

```
PUBLIC CHANNELS
├── #announcements          — company-wide announcements (CEO only)
├── #engineering            — all engineers
├── #engineering-standup    — daily standup posts
├── #pnl-daily              — EOD P&L for everyone
├── #wins                   — celebrate shipped features and winning strategies
└── #help                   — anyone can ask, anyone answers

ENGINEERING SUB-CHANNELS
├── #alpha-research         — new strategy proposals
├── #ml-experiments         — training runs, model leaderboard
├── #microstructure         — order book, OFI discussion
├── #risk-alerts            — VaR breaches, circuit breakers
├── #incidents              — P0/P1 + postmortems
├── #deploys                — deploy notifications
├── #ci-failures            — CI alerts
└── #security-alerts        — secret rotation, sec scans

SQUAD CHANNELS (private, one per squad)
├── #squad-alpha-research
├── #squad-microstructure
├── #squad-ml-modeling
├── #squad-ml-infra
├── #squad-backend
├── #squad-frontend
├── #squad-data
├── #squad-execution
├── #squad-risk
├── #squad-security
├── #squad-devops
├── #squad-qa
├── #squad-compliance
└── #squad-finance-eng

LEADERSHIP
├── #leadership             — VP+ only
├── #leadership-summary     — daily auto-summaries from each VP
├── #board                  — CEO + CFO + CTO + board observers
└── #pm-coordination        — all PMs

SOCIAL (optional)
├── #random
├── #book-club              — quant book of the month
└── #papers                 — share interesting papers
```

---

## Google Drive folder structure

```
QuantEdge/
├── 01_Company/
│   ├── Vision & Strategy.gdoc
│   ├── Employee Handbook.gdoc
│   ├── OKRs/
│   │   ├── 2026-Q3-OKRs.gdoc
│   │   └── 2026-Q4-OKRs.gdoc
│   └── Board/
│       ├── 2026-06-Board-Deck.gslides
│       └── 2026-07-Board-Deck.gslides
│
├── 02_Engineering/
│   ├── Architecture/
│   │   └── ADRs/
│   ├── Runbooks/
│   ├── Postmortems/
│   └── Show-and-tell/
│
├── 03_Research/
│   ├── Paper Library/             (PDFs + summary doc per paper)
│   ├── Alpha Reviews.gdoc         (rolling, every strategy)
│   ├── Backtest Results/
│   └── Walk-Forward Reports/
│
├── 04_PnL_and_Metrics/
│   ├── PnL Daily.gsheet           (one row per day per strategy)
│   ├── Slippage Tracking.gsheet
│   ├── ML Experiments.gsheet
│   └── Capacity Estimates.gsheet
│
├── 05_People/
│   ├── Skills Matrix.gsheet       (one row per IC, columns = skills)
│   ├── Promotion Calibration.gdoc
│   └── 1-1-Notes/                 (per IC, per manager)
│
├── 06_Finance/
│   ├── Cash Burn.gsheet
│   ├── Vendor Costs.gsheet
│   └── Investor Updates.gdoc
│
└── 07_Legal_and_Compliance/
    ├── Trading Licenses.gdoc
    ├── KYC Procedures.gdoc
    └── Audit Trail Spec.gdoc
```

---

## Automation features for agents (workflow improvements)

### 1. **AI standup auto-summary**
At 12:55 UTC, a Notion automation aggregates each IC's commits + PRs + issue updates since yesterday's standup and pre-fills their squad standup template. ICs only edit if they disagree.

### 2. **Strategy auto-pickup**
The picker (`scripts/agents/pick_next_paper.py`) runs every 30 min. When an alpha-research IC's work queue drops below 2 active strategies, picker auto-assigns the highest-expected-Sharpe pending paper from the research queue.

### 3. **PR auto-assignment**
GitHub Action triggers on PR open → reads the `role:` label → assigns the squad lead as reviewer + 1 random IC from the same squad for a 2nd review.

### 4. **Continuous benchmarking**
Every PR diff against `backend/app/strategies/` triggers a backtest job. Comment posts on the PR with Sharpe / MaxDD / IC delta vs the strategy's last green build. Block merge if regression > 5%.

### 5. **Skill calibration via PR reviews**
Each PR review counts toward the reviewer's bi-weekly review. The system tracks which reviewers catch real bugs (correlated with post-merge incidents on the same code). High-signal reviewers get higher weight in calibration committee.

### 6. **Idea board (Notion-only)**
Any IC can drop an idea into the "Idea Board" DB. Other ICs upvote. Top 5 weekly ideas get a 1-day spike from any squad with capacity. Spike result either becomes a tracked task or gets archived with reasoning.

### 7. **Live dashboard for everyone**
A Notion-embedded iframe shows the live frontend at https://quantedge.vercel.app for any squad member who wants the visual state.

### 8. **Auto-postmortem template**
When an incident is closed, a GitHub Action generates a postmortem doc skeleton in Google Docs from a template, pre-populated with the incident timeline from #incidents, deploy log entries, and impacted PR diffs. Owner only fills "root cause" and "lessons learned".

### 9. **Weekly research digest auto-generated**
Every Friday 16:00 UTC, an automation pulls all alpha reviews + ML experiments + winning strategies of the week into a single Google Doc digest. Auto-shared to investors mailing list (CFO approves before send).

### 10. **Compute budget enforcement**
CFO bot watches Render + Vercel + Kaggle hours daily. Anyone exceeding their squad's monthly budget gets a Slack DM + the squad's PM gets cc'd. Hard cap at 200% triggers VP DevOps to throttle.

---

## How to scale further (when ready)

| Trigger | Action |
|---------|--------|
| Active strategies > 100 | Hire 2nd Alpha Director, split into Equities Alpha + Cross-Asset Alpha squads |
| Trades/day > 10,000 | Hire dedicated FIX protocol engineer + redundant broker connection |
| Latency p99 > 100ms | Hire Performance Engineering Director + Rust microservice for hot paths |
| Daily Slack messages > 1,000 | Hire Knowledge Manager to maintain Notion + curate the Slack→Notion summary bot |
| Engineers > 100 | Add Director of Engineering Productivity to own dev tooling |
| AUM > $10M | Add Compliance Director, scale legal team to 3 |
| AUM > $100M | Add Head of Trading Operations + 24×7 NOC team |
| Trading on 5+ exchanges | Add Exchange Relations Manager + dedicated VP Markets |
