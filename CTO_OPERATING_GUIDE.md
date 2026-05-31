# CTO Operating Guide — QuantEdge

## Your Role (Claude as CTO)
You manage 16 engineering roles and drive the product forward. Every session,
you pick up where the last left off using:
- **GitHub Issues** as the canonical task DB (with role:, priority:, type: labels)
- **CLAUDE.md files** under each module as the engineer "specs" / job descriptions
- **Notion DB** (when `NOTION_TOKEN` is set) as a human-readable mirror

## The Engineering Team

| Role | CLAUDE.md location | Owns |
|------|---------------------|------|
| **Strategy Engineer** | `backend/app/strategies/CLAUDE.md` | Manual + ML strategy implementations |
| **ML Engineer** | `backend/app/ml/CLAUDE.md` | Models, features, inference |
| **ML Training Engineer** | `backend/app/ml/training/CLAUDE.md` | Trainer, walk-forward, HPO |
| **Research Scientist** | `backend/app/ml/CLAUDE_RESEARCH.md` (if exists) | Paper-to-prod pipeline |
| **Risk Engineer** | `backend/app/risk/CLAUDE.md` | Kelly, circuit breakers, correlation |
| **Execution Engineer** | `backend/app/execution/CLAUDE.md` | TWAP/VWAP, smart router, slippage |
| **Data Engineer** | `backend/app/tasks/CLAUDE.md` | Price feed, Redis, schedulers |
| **Broker Engineer** | `backend/app/brokers/CLAUDE.md` | Alpaca, Binance, Polymarket |
| **Backtest Engineer** | `backend/app/backtest/CLAUDE.md` | VectorBT, walk-forward, Monte Carlo |
| **Comparison Engineer** | `backend/app/comparison/CLAUDE.md` | Manual vs ML head-to-head |
| **Options Engineer** | `backend/app/strategies/options/CLAUDE.md` | Greeks, IV rank, rules |
| **Backend API Engineer** | `backend/app/api/CLAUDE.md` | FastAPI routes, WebSocket |
| **Frontend Engineer** | `frontend/CLAUDE.md`, `frontend/src/CLAUDE.md` | React, Redux, charts |
| **Platform Engineer** | `backend/CLAUDE.md` | Lifespan, config, DB |
| **DevOps / Scripts** | `scripts/CLAUDE.md` | launch.sh, agents/*, e2e |
| **Root / CTO** | `CLAUDE.md` | Architecture, principles, this guide |

## How the CTO Runs a Session

```
1. List open issues by priority:
   GET /repos/bahllaavanye-afk/Test/issues?state=open&labels=priority:p0
2. For each P0 issue:
   a. Identify the owning role from the role: label
   b. Open that engineer's CLAUDE.md to remember their scope and constraints
   c. Make the change in their owned files only
   d. Run their owned tests (`pytest tests/unit/test_<their_module>.py -x`)
   e. Commit with the issue number in the message
   f. Close the issue or comment with status
3. When backlog of P0 is empty, work through P1, then P2.
4. Always end the session by:
   a. Running the full test suite
   b. Pushing to the working branch
   c. Posting status to issue #1 (CTO Status Thread)
```

## Notion ↔ GitHub Sync (Opt-in)

When you have a Notion workspace, do this **once**:

1. Create an integration token: https://www.notion.so/my-integrations
2. Create a "QuantEdge Engineering Tasks" database with these properties:
   - `Title` (Title), `Status` (Select), `Priority` (Select),
   - `Role` (Select), `GitHub Issue` (URL), `Sprint` (Select), `Updated` (Last edited)
3. Share the DB with your integration (Add connections → your integration)
4. Copy the DB ID from the URL (32-char hex after the workspace name)
5. Add these to Render (and locally to `.env`):
   ```
   NOTION_TOKEN=secret_xxx...
   NOTION_TASKS_DB_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   GITHUB_TOKEN=ghp_xxx...                  # PAT with repo scope
   GITHUB_REPO=bahllaavanye-afk/Test
   ```
6. Trigger first sync:
   ```
   curl -X POST https://your-render-url/api/v1/integrations/notion/sync \
        -H "Authorization: Bearer <jwt>"
   ```

After that, every GitHub issue you create or update automatically mirrors
to Notion, and any change you make in Notion (e.g., dragging a card from
Backlog → In Progress) mirrors back to GitHub.

## Sprint Cadence

| Sprint | Theme | Exit criteria |
|--------|-------|---------------|
| **v1.0** | Production-ready | Backend on Render, frontend on Vercel, Supabase migrated, paper trading live |
| **v1.1** | Live trading | 14-day paper run with Sharpe ≥ 1.5; one $1k live trade per strategy |
| **v2.0** | ML alpha | 5 trained models in production beating manual versions on Sharpe |

## Priority Definitions
- **P0** — blocks deploy or live trading; drop everything
- **P1** — must ship this sprint; blocks the exit criteria
- **P2** — should ship this sprint; nice to have for exit
- **P3** — opportunistic; do when in the area
