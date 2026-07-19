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
auto-pr.yml (on push claude/**): opens draft PR + `automerge` label + dispatches CI
                     ▼
CI = test.yml: frontend-build · test · test-agents  (real correctness gates)
                     ▼
CI `gate` job (NEW): green + `automerge` label → squash-merges the PR itself,
                     then dispatches deploy-on-main.yml
                     ▼
deploy-on-main.yml: POST Render deploy → poll to live → page Discord on failure
                     ▼
smoke-test.yml (every 30 min): deploy-parity + scheduler-liveness + Discord self-test
```

### Why the `gate` job had to live *inside* CI
`auto-merge.yml` watches `workflow_run:[CI]` / `check_suite` completion. But
agent-branch CI is **dispatched with `GITHUB_TOKEN`** (auto-pr.yml), and GitHub's
recursion guard **suppresses the completion events of any GITHUB_TOKEN-triggered
run**. So that gate never re-fired for agent PRs and a human had to click merge
every time. The fix: merge from *inside* the same CI run that already went green
(`test.yml` → `gate` job, `needs: [test, test-agents, frontend-build]`), which
needs no PAT and no cross-workflow event. The GITHUB_TOKEN merge's push to main
also can't trigger `deploy-on-main` (same guard), so the gate dispatches it.

`auto-merge.yml` still handles **human** PRs (real `pull_request` events cascade
normally); the `gate` job handles the **agent** PRs it couldn't reach. No overlap.

## Standing autonomous workflows

| Workflow | Trigger | What it does hands-off |
|---|---|---|
| `improvements-worker.yml` | event-chain + ~24h gate | IMPROVEMENTS.md items → issues → engineer |
| `continuous-improvement.yml` | event-chain | Free-Agent Engineer implements queued issues |
| `auto-pr.yml` | push `claude/**` | opens labelled draft PR + dispatches CI |
| `test.yml` `gate` job | dispatched CI, green | **auto-merges** agent PR + dispatches deploy |
| `auto-merge.yml` | human PR events | auto-merges labelled human PRs |
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
