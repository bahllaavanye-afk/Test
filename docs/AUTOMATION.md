# QuantEdge — Full Automation Map (what runs with zero manual work)

_Goal: no manual steps from the operator. This documents every hands-off loop and
the only irreducible manual bits (secrets a human must paste once)._

## The end-to-end loop (no human in the path)

```
Scouts append ideas ─┐
                     ▼
IMPROVEMENTS.md ──► improvements-worker.yml (event-chained, ~24h gate)
                     │   dedups → files `agent-fix-needed` issues
                     ▼
Free-Agent Engineer (continuous-improvement.yml) implements → pushes agent branch
                     ▼
auto-pr.yml (on push claude/**): opens PR + `automerge` label + dispatches CI
                     + ARMS GitHub native auto-merge (gh pr merge --auto)
                     ▼
CI = test.yml: frontend-build · test · test-agents  (required status checks)
                     ▼
GitHub NATIVE auto-merge: when the required checks pass, GitHub merges the PR
                     itself, protected by the branch-protection rule you set
                     ▼
deploy-on-main.yml: POST Render deploy → poll to live → page Discord on failure
                     ▼
smoke-test.yml (every 30 min): deploy-parity + scheduler-liveness + Discord self-test
```

### Why GitHub NATIVE auto-merge (not a self-merging CI job)
`auto-merge.yml` watches `workflow_run:[CI]` / `check_suite` completion. But
agent-branch CI is **dispatched with `GITHUB_TOKEN`** (auto-pr.yml), and GitHub's
recursion guard **suppresses the completion events of any GITHUB_TOKEN-triggered
run**. So that gate never re-fired for agent PRs and a human had to click merge
every time. Native auto-merge sidesteps this: it is a GitHub *internal* mechanism
that watches the required status checks on the PR head and merges when they're
green — no event cascade, no recursion guard, no PAT, and **the merge decision
stays in your branch-protection gate**, not in a CI job that grants itself write
access. `auto-pr.yml` only *arms* it (`gh pr merge --auto`).

**One-time setup (operator, ~2 min, then never again):**
1. Repo → **Settings → General → Pull Requests → ✓ Allow auto-merge**.
2. Repo → **Settings → Branches → Add branch protection rule** for `main`:
   **✓ Require status checks to pass**, and add `test`, `frontend-build`,
   `test-agents` as required checks. (Leave "require approvals" OFF for full
   hands-off; turn it ON if you want to keep a human review gate.)

Until that's done, `gh pr merge --auto` is a harmless no-op and PRs just wait for a
human — nothing breaks. `auto-merge.yml` stays as a belt-and-suspenders.

## Standing autonomous workflows

| Workflow | Trigger | What it does hands-off |
|---|---|---|
| `improvements-worker.yml` | event-chain + ~24h gate | IMPROVEMENTS.md items → issues → engineer |
| `continuous-improvement.yml` | event-chain | Free-Agent Engineer implements queued issues |
| `auto-pr.yml` | push `claude/**` | opens labelled PR + dispatches CI + **arms native auto-merge** |
| GitHub native auto-merge | required checks green | **merges the PR** (protected by your branch rule) |
| `auto-merge.yml` | human PR events | label-gated fallback merger for human PRs |
| `deploy-on-main.yml` | push main / dispatch | Render deploy + poll + page on failure |
| `smoke-test.yml` | every 30 min | deploy parity + scheduler liveness + Discord self-test |
| `strategy-scout` / `symbol-scout` | daily + CI-chain | find new strategies/symbols → append IMPROVEMENTS |
| `oa_scout` | daily + CI-chain | watch Options Alpha for new bots (needs cookie secret) |
| `strategy-autopilot` / `auto-tune` / `promotion` | chained | tune + promote proven strategies |
| `oa-backtests.yml` | weekly | refresh OA clone backtest scores |

## The ONLY manual bits (secrets — I have no secret-write API)

These are pasted **once** into GitHub → Settings → Secrets → Actions, then every
loop above uses them forever. Everything else is automated.

| Secret | Unlocks | Status |
|---|---|---|
| `OA_SESSION_COOKIE` | OA Scout harvesting your bots/templates | optional; expires, re-paste when stale |
| `TRADIER_SANDBOX_TOKEN` | real option chains + ORATS greeks/IV (free, no card) | **highest-value unlock** |
| `WORKFLOW_PAT` | would let `auto-merge.yml` fire on agent PRs too (now redundant — the CI `gate` job covers it PAT-free) | no longer required |
| Alpaca / OANDA / Polymarket / Kalshi / Discord / Render keys | live desks, deploy, alerts | already provided |

If a secret is absent, the dependent loop **skips cleanly** (never blocks the
pipeline) and says so in its logs — so missing unlocks degrade gracefully rather
than stalling everything.

## What "no manual work" now means concretely
- You never merge a PR: green agent PRs self-merge via the CI `gate` job.
- You never deploy: merges dispatch Render deploys automatically.
- You never hand-file work: scouts + IMPROVEMENTS.md + the worker generate it.
- You only ever (optionally) paste a secret to unlock a new data source.
