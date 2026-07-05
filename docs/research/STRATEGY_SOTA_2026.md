# Strategy & Autonomy Audit + SOTA Research — 2026-07-05

Deep review of the strategy stack, the autonomous "company," and what the
state of the art (mid-2026) says we should build next. Companion to
`docs/research/AI_COMPANY_SOTA.md` and `docs/OPTIONALPHA_PARITY.md`.

## Part 1 — What we actually have (audited, not aspirational)

### Inventory
- **93 manual strategies** in `backend/app/strategies/manual/` spanning 6 CI
  trading desks (Equities, Crypto, Options, Polymarket, Macro/FX, StatArb)
  plus commodity/rates/FX families not yet mapped to any desk.
- **29 bots** (indicator-triggered, Option Alpha-style) in the app across
  equity/crypto/options; runner + 5-min exit checker live since today.
- **4 deep models** (LSTM, PatchTST, Mamba, SSM) that cannot load in prod
  (512MB dyno, no torch) — training now lives in `ml-experiments.yml`.

### Scrutiny findings (green ≠ working)
| Finding | Impact | Status |
|---|---|---|
| `desk-trading.yml` ran with `continue-on-error: true` | Placer crashes showed ✅ for weeks | **Fixed** — removed |
| One global equity clock gated ALL desks | "Crypto 24/7" never traded nights/weekends (`market_closed`, 0 orders) | **Fixed** — `DeskConfig.always_open`, per-desk gate |
| `backend_team.py` free-LLM path: `NameError: context` | Backend AI team crashed every run since key was disabled | **Fixed** — builds its own file bundle |
| `secrets-check.yml` never installed httpx | Red every 6h for the wrong reason | **Fixed** — pip install step |
| auto-merge `workflow_run: ["test"]` ≠ CI's name `"CI"` | Gate never re-fired after checks finished; green PRs sat unmerged | **Fixed** — watches `"CI"` |
| Desk orders bypass the app DB (`DATABASE_URL: ""`) | Orders exist only at Alpaca; app trades/positions stay empty; two disconnected P&L worlds | **Open — next build** |
| No persisted strategy performance (leaderboard empty, backtests `[]` in prod) | "Scale up winners" has no data to act on | **Open — next build** |
| `signals_generated=0` on recent desk runs | Thresholds (0.62–0.75 conf) + regime gate may be over-tight, or bars fetch degrades silently | **Open — needs per-stage logging** |

### Test-integrity audit (reward-hacking check)
Audited every autonomous-agent commit touching `backend/tests/` since
2026-06-20, including `test_reward_gate.py` (the gate's own test — the classic
self-serving edit). **Verdict: clean.** All five agent edits were cosmetic or
strengthening (assertion messages added, duplicate-revision check added,
+98 lines of new TWAP/execution cases). One process smell: the improver
mislabels commits (e.g. "improve(strategy_logic)" on a test file) because it
picks files blindly — worth constraining its file selector, but no assertions
were weakened anywhere.

### Is it "really a company"?
- **Real and working:** 30 scheduled workflows ran in the last 3h, 28 green.
  The reward-gate loop (agents open draft PRs → CI → auto-merge) is now
  end-to-end after today's draft + workflow-name fixes. The queue worker
  (Mondays) + model audit (Saturdays) + ml-experiments (Sundays) close the
  plan→build→verify→land loop without a human.
- **Was theater until today:** desk trading (masked failures, equity clock on
  crypto, orders invisible to the app), employee Slack chatter (quota-dead
  since 06-29; Discord failover shipped but `DISCORD_WEBHOOK_URL` still unset
  — chatter currently goes nowhere), deep-ML "employees" (can't load torch).
- **Bottom line:** the machinery is genuinely autonomous; the *trading desk
  outputs* were the weakest link, and the P&L feedback loop is still missing.

## Part 2 — SOTA (mid-2026) and where we stand

Key findings from current literature and frameworks:

- **Multi-agent LLM trading** ([TradingAgents](https://github.com/TauricResearch/TradingAgents),
  [paper](https://arxiv.org/abs/2412.20138), v0.3.0 June 2026): specialized
  analyst/researcher/trader/risk roles debating before a trade. One documented
  run: ~7% in 30 days vs SPX 4.5% — with 22% drawdown and no repeatability
  guarantee. Our hourly agents already mirror the *role* structure; what we
  lack is their **structured debate → single decision → tracked outcome** loop.
- **LLM alpha mining** ([Chain-of-Alpha](https://arxiv.org/pdf/2508.06312),
  [Automate Strategy Finding](https://arxiv.org/html/2409.06289v2)): LLMs
  generate interpretable formulaic alphas, then a *deterministic* backtest
  ranks them. This matches our reward-gate philosophy: LLM proposes, code
  verifies. **This is the highest-leverage SOTA idea for us** — an
  `alpha-miner.yml` that proposes factor formulas weekly and walk-forward
  tests them is cheap and fits the existing gate.
- **Hybrid > pure LLM** ([survey](https://lunefi.com/blog/machine-learning-trading-strategies-2026-trends-stats-insights),
  [regime-adaptive hybrid](https://arxiv.org/html/2601.19504v1)): rules +
  narrow ML beat end-to-end LLM traders; realistic live win rates are 50–60%
  and Sharpe > 1.5 is the honest bar. Our GBC walk-forward baseline (51.8% on
  synthetic noise) is calibrated exactly right.
- **Look-ahead bias in LLMs** ([Look-Ahead-Bench](https://arxiv.org/pdf/2601.13770)):
  point-in-time discipline matters even for LLM features. Our
  `test_all_strategies_contract.py` no-lookahead guard already enforces this
  for code; LLM-generated alphas must pass the same guard.
- **Agentic RL** ([AlphaQuanter](https://arxiv.org/html/2510.14264v1),
  [FinPos](https://arxiv.org/pdf/2510.27251)): tool-orchestrated RL traders are
  research-grade, compute-heavy, and unproven live — **skip for now**.

## Part 3 — Scaling plan (priority order)

1. **Close the P&L feedback loop** (prereq for everything): desk placer posts
   fills to the backend (`POST /orders/` with its JWT) OR a 15-min
   `order_sync` pull from Alpaca into the app DB; then the leaderboard ranks
   strategies on *live paper* Sharpe, not vibes.
2. **Auto-allocation from the leaderboard**: weekly job reads 30-day live
   Sharpe per strategy → writes `tuned_thresholds.json` + per-desk notional
   weights (winners scale, losers throttle to zero). The auto-tuner file
   plumbing already exists.
3. **LLM alpha miner** (Chain-of-Alpha pattern): weekly, free-cascade LLM
   proposes 5 formulaic alphas → walk-forward harness scores OOS → top alpha
   opens a draft PR as a new strategy into the reward gate. LLM proposes,
   backtest disposes.
4. **Signal-drought telemetry**: per-stage counts (bars fetched, signals,
   regime-blocked, threshold-blocked) posted to the digest so
   `signals_generated=0` is a visible alarm, not a silent norm.
5. **Structured debate for size** (TradingAgents-lite): before any order >
   $500 notional, one bull + one bear free-LLM pass; disagreement halves size.
   Cheap, bounded, logged.
6. **Options desk real routing**: OCC symbols through Alpaca's options order
   API (paper) so the 4 options bots trade real chains instead of the generic
   engine.

Sources: [TradingAgents](https://github.com/TauricResearch/TradingAgents) ·
[TradingAgents paper](https://arxiv.org/abs/2412.20138) ·
[Chain-of-Alpha](https://arxiv.org/pdf/2508.06312) ·
[LLM strategy finding](https://arxiv.org/html/2409.06289v2) ·
[Look-Ahead-Bench](https://arxiv.org/pdf/2601.13770) ·
[AlphaQuanter](https://arxiv.org/html/2510.14264v1) ·
[FinPos](https://arxiv.org/pdf/2510.27251) ·
[2026 ML trading survey](https://lunefi.com/blog/machine-learning-trading-strategies-2026-trends-stats-insights) ·
[Hybrid regime-adaptive system](https://arxiv.org/html/2601.19504v1)
