# CONTINUITY — read me first, every session

> **Purpose:** chat sessions are ephemeral and context resets when tokens run out. This
> file (committed to the repo) + the `SessionStart` hook in `.claude/settings.json` make
> every new/resumed session **auto-load the current state** so no memory or progress is
> lost. Keep it current: when you finish or start something material, update this file in
> the same commit.

_Last updated: 2026-08-06._

## 🔴 LIVE ACCOUNT STATE — 2026-08-06 06:00 (read before trusting any P&L)

    equity $21,801.52   cash -$48,471.29   buying_power $0.00   non_marginable_bp $0.00
    17 positions queued to flatten at the open · desk placed 0 orders (reason=account_unavailable)

**The account cannot trade.** `recover_negative_cash` fires correctly (cash < 0, no buying power left) and
issues a close-all — but the market is shut, so the closes sit `accepted` and free nothing. Runs
`31070310789` (04:06) and `31072118909` (04:45) flattened **the same 17 positions** and reported cash
identical **to the cent**, which is how the loop was identified. It re-fires every run until the open.

**What happens at the open is the operator decision (item #17):** 17 positions flatten at once, realising
losses into the first prints. That is the documented "buy on margin, get liquidated, get frozen" path at
`desk_order_placer.py:583` — the realised loss can trip the daily cap, which with an empty book allows only
risk-reducing orders, which blocks everything until the next session rollover. Doing nothing is a choice to
let that run.

## 🤖 THE IMPROVER'S TARGETING WAS HALF DECORATIVE — fixed 2026-08-06 07:30 (#1549)

Relevant to the standing "improver PRs bypass CI so main silently breaks" concern, because it explains where
a good share of the low-value diffs came from.

`continuous_improver` chose the improvement TYPE by `hour % 12` (fixed per run) and the target PATTERN by
`((hour + attempts) % 24) % 12` (rotating per attempt). Both lists were length 12, so **attempt 0 paired type
*i* with pattern *i* permanently**, recurring every 12 hours:

    test_cases     -> backend/app/ml/models/*.py    unit tests written INTO model source
    strategy_logic -> backend/app/api/v1/*.py       route handlers have no entry/exit logic
    monitoring     -> backend/tests/unit/*.py       P&L logging added to unit tests

**And 6 of the 12 patterns yielded ZERO usable files** once `PROTECTED_PREFIXES` and the 8000-char guard
applied (`strategies/manual`, `strategies/ml_enhanced`, `ml/models`, `ml/features`, `execution`, `risk`). Those
runs fell through to a glob over the whole backend, so on **half of all hours the targeting did nothing** —
silently. That is the mechanism behind `strategy_logic` landing on `models/account.py` (#1510).

Now: `TYPE_TARGETS` maps each type to locations where the change is meaningful, every pattern is verified to
yield at least one usable file, `test_every_improvement_type_has_a_live_target` fails CI if that stops being
true, and the fallback announces itself. **`strategy_logic` is removed, not re-pointed** — its prompt only
means something inside `backend/app/strategies/`, which is protected precisely because money-path behavior is
not the improver's to touch. A type whose only legal target is off-limits is not a type.

**This does not close the CI-bypass concern.** The improver still opens PRs that skip the gate; this only
stops it aiming good prompts at wrong files.

## 🌐 CI FAILS ON PyPI OUTAGES, AND THAT IS NOT A REGRESSION — seen twice, 2026-08-05 and 2026-08-06

`test` died in the dependency step, before any test ran:

    error: Failed to fetch: `https://pypi.org/simple/httpcore/`
      Caused by: stream closed because of a broken pipe

Earlier the same day it was `https://pypi.org/simple/rsa/`. Read the FAILING STEP before diagnosing: a red
`test` job here means nothing about the code. `rerun_failed_jobs` is **not available** to the MCP token
(403 "Resource not accessible by integration"), so the retrigger is a fresh push.

## 🧠 THE LLM CASCADE IS ONE PROVIDER DEEP — partly fixed 2026-08-06 09:30

Brain-health run `31078219842` (06:42) reported `healthy: true` with **one of eight providers answering**, and
the 03:32 run had already failed outright — so the cascade has been fully dark once today.

**Fixed in code:** `_has_key()` checked only `key_env`/`key_env_alt` while `_provider_keys()` also collects
`_1.._3`. With `GROQ_API_KEY` empty and `GROQ_API_KEY_1/2/3` set, Groq reported `has_key: false` and — because
`_call_parallel_race` builds the live cascade from `_has_key` — was **excluded from every actual LLM call**.
Three keys at 30 rpm each, unused, while the platform ran on `nvidia_nim` alone at 18.5 s per call.

**VERIFIED LIVE 09:49, run `31090685392`:** `groq` now answers with 3 keys in **171 ms** (nvidia_nim was
18,550 ms). In that same probe **`nvidia_nim` timed out** — so without this fix the run would have had zero
working providers and the cascade would have gone dark for the second time today.

**Fix recovered TWO providers (corrected 10:45):** `gemini` also went `has_key: false` → `keys: 3`. Six keys
across two providers were ignored; providers with keys went 1 → 3. Gemini currently returns **429** (free quota
exhausted), so adding more Gemini keys buys little — a distinct provider is the useful move.

**Still open, both operator calls:**
1. **Partly addressed 11:15:** the canary now prints `⚠ BRAIN AT RISK — only 1 of N keyed provider(s)
   answering` instead of `BRAIN OK`, and `cascade_status()` exposes `single_point_of_failure`. **The gate is
   unchanged** — still exit 0, still no page. Whether a one-deep cascade should PAGE is the open decision.
   `healthy = bool(working)` still means "at least one provider answers". That is not health for a platform whose
   agents, improver and desk commentary all need the cascade. A floor of 2 would make one-away-from-dark
   visible. Not changed unattended — it changes when the infra alarm fires.
2. **Seven providers have no key at all**, all free tiers. Adding one (`GEMINI_API_KEY` is easiest) removes the
   single point of failure with no code change.

## 🎯 OPERATOR DECISIONS — everything now blocked on a human, ranked
Nothing below can be done from code. Each line says what it unblocks, so it can be
triaged without reading the 2,600 lines under it. Detail for every item is further
down this file; this index exists because findings were being **rediscovered** —
the torch-free ML path, the Polymarket no-order-path and the CI-bypass item were
each independently "found" after already being documented.

| # | Action | Effort | Unblocks |
|---|--------|--------|----------|
| 1 | **Suspend/delete the `quantedge-api-agb8` Render service** | ~30s | A second backend (paper mode, Alpaca-connected, 11 background tasks, dead DB) is trading the SAME account as `9jz0`. Contaminates equity/buying-power/position reads the desk uses for Kelly sizing, the loss cap and `is_risk_reducing`, plus the new slippage attribution — it can also place duplicate orders. ~~Prime suspect for the `< $25 available cash` block.~~ **That justification is withdrawn 2026-08-04 13:50** — the cash block cleared on its own at the open (11 orders placed), which a draining second backend would not do. Still worth shutting down for the duplicate-order and attribution reasons; the cash argument does not support it. | **RE-VERIFIED LIVE 2026-08-05 19:00** — `background_tasks: running 11`, `alpaca: connected`, `strategies: 113`, own DB dead. ⚠️ **The evidence I twice cited FOR this item is withdrawn 2026-08-06 06:00.** `audit_order_origins()` reported "5 of 50" and then "50 of 50 orders NOT placed by this desk — a second writer is on this account", and I relayed both. Those orders were **our own** `recover_negative_cash` flatten: five of them share a `submitted_at` to within **2.1 milliseconds**, which is one `DELETE /v2/positions` call, not five decisions. The audit was reading "untagged" as "foreign" while `grep -c client_order_id backend/app/brokers/*.py` returns **0 for all eight broker files** — nothing we run tags anything except the desk placer. Fixed same day (burst detection); the alarm no longer fires on our own flattens. **This item now rests only on its original duplicate-order and attribution arguments — there is still no direct evidence agb8 has placed an order.**
| 2 | **Unpause Supabase `vexzwnfbmznvxoxxktax`** — ⚠️ **NOT SUFFICIENT ON ITS OWN, found 2026-08-06 01:00.** `alembic/versions/d4e5f6a7b8c9_create_bots_table.py:32` declares `is_enabled` as `sa.Boolean()` with `server_default=sa.text("1")`. That is SQLite idiom; Postgres rejects it outright (`DatatypeMismatch: column "is_enabled" is of type boolean but default expression is of type integer`), so **the migration chain cannot apply to Postgres**. The app has only ever run on the SQLite fallback, which is precisely the engine that tolerates it. Fix is one line — `server_default=sa.true()`, which SQLAlchemy renders per dialect — but `alembic/versions/` is Do-Not-Modify, so it needs you. Do this BEFORE or WITH the unpause, or the change window is wasted. | ~1min + 1 line | Nothing persists. Trades land in ephemeral sqlite and are wiped on every redeploy → empty leaderboard, inert attribution pruning, no per-strategy TCA history. Also the last open link of the post-deploy revival P0. |
| 3 | **Vercel: fix `ignoreCommand` + clear the stale-failure PRs** | ~5min | ~100 PRs frozen. Two separable actions: prepend `case "$VERCEL_GIT_COMMIT_REF" in improver/*) exit 0;; esac` to stop NEW PRs being stamped, and push/redeploy/close the existing ones — a commit status is immutable per sha, so they never self-clear. (`frontend/vercel.json` is under "Do NOT Modify", so this is yours by policy too.) |
| 4 | **Crypto confidence recalibration** (needs a walk-forward backtest) | hours | The only always-open desk. `confidence = |raw| · (target_vol/rv) / 2` caps at 0.40 @50% vol vs a 0.60 gate, so overnight trading is rare. The naive fix scores 0.83–0.94 on a ZERO-DRIFT random walk — it trades noise — so this is a strategy decision, not a patch. `xfail(strict=True)` tests flip to failure when it is done. |
| 5 | **ML: pick a path** | decision | Either add an XGBoost/LightGBM trainer (torch-free — runtime, model class and loader all already exist on Render; only the trainer is missing) **or** host inference where torch fits. Until then every LSTM item is unreachable work: torch is deliberately excluded from Render (`pyproject.toml:54-56`). |
| 6 | **Agent-wave cadence** | decision | `auto-launch.yml` has no schedule, so the 48-agent company essentially never runs autonomously. Per-agent metrics already exist (`agent_tracking`) and are posted-then-discarded; persisting them is ~10 lines but pointless until the wave runs. It is a Discord-noise decision. |
| 7 | **Polymarket: wire py-clob-client, or retire the desk** | decision | 8 strategies producing up to conf=1.00 with no venue route. Now reported honestly as `NO ORDER PATH` rather than "closed" (#1385), but still unexecutable. |
| 8 | **Set `OA_SESSION_COOKIE`** | ~1min | The OA scout is auth-walled and has found 0 bots in 56 runs. It no longer spams commits (#1394), but stays a no-op without this. |
| 9 | **Cap the `/notifications/discord/cto-review` endpoint** | small | Only uncapped paid-Claude path: calls `claude-haiku-4-5` directly, no free-tier attempt, no cascade, no budget cap. |
| 10 | **Commit signing / merge path** | decision | Silences the recurring "Unverified commits" stop-hook. Those commits are GitHub squash-merges and repo state-bots, not mine — I have declined to rewrite them every time, since amending would reattribute other authors' merged work to me. |
| 11 | **Add a second free-tier LLM key** (`GROQ_API_KEY_1` / `DEEPSEEK_API_KEY_1` / `TOGETHER_API_KEY_1`) | ~2min | **Measured ceiling: 5 of 47 employees respond.** One Gemini key is the whole fleet's capacity. Key presence is not capacity — `_has_key()` reported a 60-rpm provider the cascade never actually called. |
| 12 | **Reset the Alpaca paper account** | ~1min | Cash is −$22,179.98 (margin debit); the book is levered ~1.6×. Not currently blocking equity (bp $34,713, 13 fills at 17:43 today), but it is what keeps **crypto at exactly zero orders** — see 13. `MARGIN_FLOOR_PCT` is now in place so the freeze does not recur after the reset. |
| 13 | **Crypto cash reserve — an allocation decision, deliberately not shipped** | decision | Crypto sizes against `non_marginable_buying_power` = settled cash, and cannot use margin at all. Cash is negative, so crypto is at **$0.00 capacity and has placed nothing**. The only fix caps the equity desks at `cash − X`, i.e. **stops equity using margin** near the reserve. Crypto is the sole `always_open=True` desk, so today the platform's overnight coverage is zero orders, not "quiet". Your call on the trade-off; I would not make it unattended. |
| 15 | **Create `#desk-india-mf` in the guild** | ~30s | Both India workflows post to it and it does not exist, so `notify` falls back to the default webhook and the NAV ranking + NSE read land in whichever channel that targets. The notifier says so on every run — the messages arrive, just not where addressed. Full channel inventory (17) is in IMPROVEMENTS 21:40. |
| 17 | **Decide what happens to the 17 queued flattens at the next open** | decision | ⚠️ **NEW 2026-08-06 06:00, and the most consequential item here.** The account is at cash **-$48,471.29** with **$0.00 buying power** and cannot place an order; `recover_negative_cash` has queued a close-all of 17 positions that cannot fill until the market opens. At the open they all fill at once and realise losses into the first prints — the documented "buy on margin, get liquidated, get frozen" path (`desk_order_placer.py:583`), which can trip the daily loss cap and freeze the desk to risk-reducing orders until the next session rollover. Three options, all yours: **(a)** let it flatten and accept the realised loss + likely cap trip; **(b)** set `AUTO_FLATTEN_ON_NEGATIVE_CASH=0` and unwind by hand at the open; **(c)** reset the Alpaca paper account, which zeroes the debit and the position book together. I have not chosen — unwinding a book unattended is not a tick-hours action, and the code change for (b) is a single env var you control. |
| 16 | **Decide whether the International desk needs `#desk-international`** | decision | It currently posts to `#desk-equities`. Since the India expansion put INDA/EPI/SMIN/INDY on that desk, **India ETF activity now reads as Equities activity**. Defensible, but worth knowing before reading either channel. Not repointed in code on purpose: aiming a desk at a channel that does not exist is strictly worse than an honest shared one. |
| 14 | **Set `ANGELONE_API_KEY` / `_CLIENT_ID` / `_PASSWORD` / `_TOTP_SECRET`** | ~5min | Turns the India work from research into execution. AngelOne SmartAPI is free and TOTP-derivable, so it is the only broker that can log in unattended — **Zerodha cannot** (browser redirect, daily token, ₹2,000/mo). `india_broker.py` reports what is missing; nothing places an Indian order until these exist. |


## 🌏 2026-08-06 05:10 — THE OVERNIGHT READ COVERS SEVEN MARKETS (Asia-Pacific added; Europe cannot work)

Same machinery as the NSE tilt — foreign close → state file → bounded nudge at the desk's confidence gate.
Membership is decided by **close time vs the 13:30 UTC US open**, checked before extending:

    Taiwan ^TWII 05:30 (-8h00)   Japan ^N225 06:00 (-7h30)   Australia ^AXJO 06:00 (-7h30)
    Korea  ^KS11 06:30 (-7h00)   HK    ^HSI  08:00 (-5h30)   Singapore ^STI  09:00 (-4h30)
    India  ^NSEI 10:00 (-3h30)

**EUROPE IS EXCLUDED AND A TEST ENFORCES IT.** DAX/CAC close 15:30 UTC, FTSE 16:30 — *after* the US open, so a
European close cannot inform an order at it. The desks trade EWG/EWQ/EWU, so this will look like an oversight
to the next reader. It is not.

Weights: 0.9 country index → MSCI tracker; **^HSI → FXI is 0.7** (a Hong Kong index against China H-shares).

**Two bugs fell out, both more valuable than the feature:**
1. The session close was ONE hardcoded constant (NSE's 10:00) — 4.5h wrong for Taiwan. Now per-market, with
   half-hours (Korea 06:30) and ADRs inheriting their market by exchange suffix.
2. **The live run produced a tilt from a session still trading.** `EWT -0.0107 ← ^TWII -0.60%` at 05:15, with
   Taiwan 15 minutes from its close. The future guard tolerated ±1h — harmless when NSE was the only source and
   the run happened at 10:20. Now clock-skew only (0.1h). **An intraday snapshot reported as a close is worse
   than no read: it is indistinguishable from a real one.**

**Tooling trap:** `__pycache__` made a mutation test lie — source read `0.1` while the import reported `1.0`,
and a test passed alone but failed in the full suite. Clear the cache or set `PYTHONDONTWRITEBYTECODE=1`
between mutation rounds.

## ✂️ 2026-08-06 04:40 — STRATEGY RETIREMENT WAS UNAUDITABLE **AND** IRREVERSIBLE

The noise-floor lesson from the ML work applies to the path that retires live strategies. Each retirement's
win count tested against a coin flip:

    avellaneda  6/10  P=0.828 (chance)   vol_of_vol 2/10 P=0.055   realized_vol_asymmetry 3/10 P=0.172
    options_pcr_reversal 2/11 P=0.033    stat_arb_e 0/11 P<0.001

`avellaneda` won 6 of 10 and was retired **permanently**. Its rule was a magnitude one, for which win rate is
admittedly the wrong test — but **`fill_tracker` recorded no dispersion at all**, so nothing could distinguish
"bleeds -0.79% every trade" from "flat nine times, lost 8.4% once". Both give the same `total_return_pct` at
the same n. And nothing ever removes an entry from `strategy_trims.json` while a retired strategy places no
orders, so the decision can never generate the evidence that would overturn it.

**Shipped:** `fill_tracker` carries a running `sum_sq_return_pct` and `worst_trade_pct`; the trim reason now
reads `[worst single trade -8.40% EXCEEDS the net loss — the other 9 were net positive, stdev 2.61%]` versus
`[worst single trade -1.20% = 15% of the loss, stdev 0.31%]`. **Thresholds unchanged, with a test pinning
that** — what gets retired is capital-allocation policy, not a reporting detail.

**Operator question, now askable with evidence:** should a magnitude retirement require a dispersion bar, and
should retirement carry a TTL or re-audition path? Both move real money, so neither shipped.

## 🚨 2026-08-05 19:10 — 10% OF THE BOOK HAS NO OWNER, AND NOTHING CAN SAY WHOSE IT IS

`audit_order_origins()` shipped at 19:00 and fired on its first live desk run (`31037485632`):

    ⚠️ ORDER-ORIGIN AUDIT: 5 of 50 recent orders were NOT placed by this desk
       19:02:24 USO  buy [filled]   19:01:51 NVDA buy [filled]   19:01:42 QQQ buy [filled]
       18:38:52 NVDA buy [filled]   17:58:46 AAPL buy [filled]

All five are FILLED BUYS inside 65 minutes, all carrying bare UUIDs — the id Alpaca generates when the caller
supplies none. Broker auto-liquidation is ruled out: that sells.

**The root cause: NO backend order path sets a `client_order_id` — zero occurrences in all four files**
(`brokers/alpaca.py`, `brokers/alpaca_orders.py`, `bots/engine.py`, `api/v1/orders.py`). A first pass blamed
`submit_alpaca_order` alone; that is the *least* likely source, since it is reached only from the HTTP endpoint.
The background-task path is `AlpacaBroker.place_order` (`brokers/alpaca.py:157`), equally untagged.

**Narrowing that does hold, from the code:** `position_monitor` is the only order-placing background task and
it places **exits** — these are all buys, so not it. `bot_runner._run_bot()` does a `select(Bot)` before every
run and **agb8's DB is `ok: false` with no sqlite fallback**, so its bot path cannot complete one. That points
at `9jz0`'s own bots (legitimate) rather than agb8 — an inference from code, not proof.

**Both candidates produce identical evidence:**
- `9jz0` runs `bot_jobs: 64`, firing every 1-2 min → the orders would be **legitimate**.
- `agb8` runs 11 background tasks, Alpaca connected, dead DB → the orders would be **rogue duplicates**.

That ambiguity matters more than the count: **nobody can currently answer "did our platform place this
trade?" about 10% of its own book**, and operator item #1 cannot be settled either way without it.

**The fix is a `qb-` tag at BOTH submission points** — `AlpacaBroker.place_order` and
`submit_alpaca_order` — mirroring the desk's working `qe-` scheme, which `desk_trade_sync` already parses.
`backend/app/brokers/*.py` is Do-Not-Modify and this is the live order-submission path, so it needs a human.
Once tagged the audit names the writer, operator item #1 settles either way, and desk_trade_sync gains
attribution for the 10% of the book it currently skips.

## ⏱️ 2026-08-05 19:40 — CRON LAG RE-MEASURED: 2.5–4.7 HOURS, NOT ~2.5 (correcting my own figure)

`india-mf.yml` had 0 runs at 67 minutes past its 18:30 cron, which looked like breakage. It is not.

    daily-standup       nominal 13:30   actual 18:13   lag 4h43m   (235 scheduled runs — healthy)
    strategy-auto-tune  nominal 00:30   actual 03:41   lag 3h11m   (42 runs)
    ml-experiments      nominal 04:17   actual 06:45   lag 2h28m   (5 of 5 Sundays)

**Ruled out: the minute is not the cause.** 41 of this repo's crons sit on `:00`, so collision is the tempting
story, but other `:30` workflows fire fine. `india-mf`'s first slot simply is not due yet at this lag —
**expect 21:00–23:15 UTC**.

**Consequence I got wrong earlier and am correcting:** I wrote that `india-nse-signal.yml` (`20 10`) has ~3h of
margin and "still lands first" before the 13:30 open. At the real lag it can arrive **12:50–15:00 — i.e. it can
miss the open**. Moving the cron earlier is impossible: NSE closes at 10:00 UTC.

**No code change needed, because the design already absorbs it.** The desk re-checks the file's age at read
time and accepts the previous session inside its 30h window, so a late producer costs *yesterday's* Indian read
rather than no read, and logs which it used. Desks running after the late arrival get today's. Worst case is
absorbed; best case is just not guaranteed.

## 🚨 2026-08-06 00:05 — A GUARD THAT HAS NEVER RUN IN CI (and the violation it should have caught)

`test_no_datetime_utcnow_in_source` fails locally and passes in CI. I recorded that as a path discrepancy
earlier today. It is a **dead guard**: it hardcodes `Path("/home/user/Test/backend/app")`, which does not exist
on a runner, so `rglob` yields nothing and `assert violations == []` is vacuously true. Both
`TestDeprecatedAPIRegression` tests do this.

`backend/app/models/backtest.py` used `datetime.utcnow()` three times the whole time.

**The rule, because this is the second instance today in mirror image:** a test that locates source by path
must derive it from `__file__`. A *relative* path broke the improver tests outside the repo root this morning;
an *absolute* path silently disabled these inside CI tonight.

Fixed: paths derived from `__file__` with `assert backend_dir.is_dir()` so a future move fails loudly, and the
three `utcnow()` calls replaced. Injecting a violation into `models/ml_model.py` now turns the test red, so the
guard is real. The sibling `get_event_loop` scan comes back clean on 0 files.

## ✅ 2026-08-05 19:00 — THE STRATEGY TRIMMER IS LIVE (pending verification since 07-29, now closed)

Both of today's desk runs log the line the item asked for:
```
✂ 8 strategy(ies) retired by the trimmer will not trade: avellaneda, avellaneda_stoikov_mm,
  options_pcr_reversal, realized_vol_asymmetry, stat_arb_e, stat_arb_etf, vol_of_vol, vol_of_vol_timing
```
`strategy_trims.json` carries the reasons (`avellaneda` −7.9% over 10 trades, 07-29; `vol_of_vol` win_rate 20%,
07-31). **`avellaneda` is the legacy truncated key the desk could never have matched**, and it is retired next
to the full `avellaneda_stoikov_mm` — so the truncated-key expansion works, and the whole four-link chain
(producer commits the file → cadence aligned → full names emitted → truncated keys expanded) is confirmed live.
The *other* pruning path — attribution weights from `/leaderboard/live` — remains inert while Supabase is
paused, so the trimmer is currently the only one working, not the redundant one.

## 🔬 2026-08-05 18:50 — "THE ML MODEL LOSES TO BUY-AND-HOLD" WAS AN ARTEFACT OF AN UNSTABLE WINDOW

**Correcting an earlier claim in this file.** The model's record depends entirely on which data window a run
happened to get, and until this morning nothing persisted enough to notice.

Alpaca's free **IEX** feed returns ~940 usable rows; yfinance returns **1399**. `main()` used Alpaca and fell
back only on a *total* failure, so a transient error silently swapped the evaluation period — per symbol.
Four runs in one day alternated `oos_days` 688 / 1147 on identical params.

The strategy Sharpe was stable across runs. **The benchmark was not** — SPY buy-and-hold read 1.482 on the
short window and 0.789 on the long one, because the long one reaches into the 2022 bear market.

| | short (≈940 rows) | long (1399 rows, incl. 2022) |
|---|---|---|
| QQQ | 1.118 vs 1.325 loses | **1.128 vs 0.736 beats** |
| NVDA | 0.949 vs 1.487 loses | **1.227 vs 1.119 beats** |
| SPY | 0.932 vs 1.482 loses | 0.645 vs 0.789 loses |

**✅ VERIFIED 19:22 — the fix works and the numbers are now reproducible.** First run after the merge:
all three symbols on `yfinance`, 1399 rows, oos 1147, window `2021-01-07 → 2026-08-04`. One source, one
window, and `first_date`/`last_date` in the payload. On that pinned window:

    SPY   0.634 vs 0.791  loses      QQQ  1.201 vs 0.734  BEATS      NVDA  1.184 vs 1.130  BEATS

Expected shape for a defensive model (`time_in_market` ≈ 0.5): gives up upside in a bull run, earns its keep in
a drawdown. **✅ 01:20 — THE SUB-WINDOWS ARRIVED AND THEY FLIP THE RECOMMENDATION.** Run `2026-08-06T00:13`:

    QQQ  overall 1.360 vs 0.729 BEATS   but 2022-01→2023-07: 2.057 vs 0.176 beats
                                            2025-01→2026-08: 0.390 vs 1.089 LOSES
    NVDA overall 1.196 vs 1.125 BEATS   recent window still ahead (1.508 vs 1.153)
    SPY  overall 0.606 vs 0.788 loses   loses both recent windows

QQQ's headline is carried entirely by the 2022 bear market. ~~In the most recent 18 months the model is behind
on 2 of 3.~~ **CORRECTED 02:00** — under the noise floor shipped since (`2·sqrt(2/n)`, ≈0.145 over a 382-day
window), the recent window reads **QQQ decisively loses, NVDA decisively beats, SPY inconclusive** (its −0.100
margin is inside the floor). So: **QQQ's edge has clearly decayed; the three-symbol picture is mixed, not
adverse.** The overall figures were never wrong — they were the wrong statistic to decide on. This argues
**against** routing live orders through ML, not for it.

**Do not replace "it loses" with "it wins"** — the honest claim is that the window decides, and
`ml_experiment.py` now pins one: longest-history source, both ends clamped to what every symbol covers, and
`first_date`/`last_date` in the payload. Confirmation is the next scheduled run, **Sundays 04:17 UTC**.

Still true and unchanged: **no ML output reaches a trade.** `ml_signal` / `ml_enhanced` remain 0 references in
`desk_order_placer.py`. Wiring it is a decision, not an oversight — and it should wait for one reproducible run.

## 🇮🇳 2026-08-05 18:10 — NSE NOW FEEDS THE US-LISTED INDIA ORDERS (the `[P2]` play, shipped)

PR #1473 merged (the time-bomb fix; `test` green). This is the next item off IMPROVEMENTS.md.

**The shape of it, so it doesn't have to be re-derived.** NSE closes 10:00 UTC. US desks run from 13:30 UTC.
That gap is the entire opportunity: Mumbai has already priced a day of India news that INDA/INFY/HDB will react
to, and unlike a `.NS` desk, those symbols can actually be routed by Alpaca.

    NSE close 10:00 UTC ──► india_nse_signal.py 10:20 ──► .github/state/india_nse_signal.json ──► desk run 13:30+

- **Producer**: `.github/scripts/india_nse_signal.py` + `.github/workflows/india-nse-signal.yml` (cron
  `20 10 * * 1-5`, `contents: write`, posts to `#desk-india-mf`).
- **Consumer**: `desk_order_placer.py` — `india_tilt(symbol, side)` applied at the confidence gate, bounded to
  **±0.06**, written back to `item["confidence"]` so top-K ranks on the tilted number.
- **Map**: ADRs at weight 1.0 (`INFY.NS→INFY`, `HDFCBANK.NS→HDB`, `ICICIBANK.NS→IBN`, `WIPRO.NS→WIT`,
  `DRREDDY.NS→RDY`); `^NSEI →` INDA 1.0, INDY 1.0, EPI 0.9, SMIN 0.6. MMYT unmapped on purpose (no NSE listing).

**The two things most likely to be broken by a future edit**, both test-pinned:
1. The desk must *call* `india_tilt` — `test_the_desk_actually_applies_the_tilt`. Fourth call-site guard this
   week; three features already shipped green and dead.
2. The desk must **re-check the file's age itself**. If `india-nse-signal.yml` ever stops running, the state
   file stays in the repo looking valid forever. The producer's freshness check ran once, when it wrote it;
   only the consumer's check can notice the workflow died.

**✅ 18:40 — THE TILT FIRED IN A LIVE DESK RUN.** Run `31035271962`:
```
🇮🇳 India overnight tilt: 5 symbol(s), 0.6h old
   supertrend_rsi_tv/INFY conf 0.73 → 0.72 (-0.011 from INFY.NS +0.56%)
   ► ichimoku_cloud_tv/INDA signal=BUY conf=0.97 — placing $1033 limit-first order
```
A **down**-tilt from an **up** session is correct — that was a SELL signal — but the line did not say so, and it
reads as a sign error until you dig for the side. Fixed: the line now prints the side and whether India
`agrees`/`disagrees`. INDA got no tilt because the Nifty moved +0.04%, under the noise floor; it traded on its
own signal. Both behaviours are the designed ones, observed in production.

**VERIFIED LIVE** on the real 2026-08-05 Indian session (Nifty 24,624.65 +0.04%, HDFCBANK −0.94%, INFY +0.56%).
Five real tilts written and read back by the desk. The committed `india_nse_signal.json` is that run's output,
so the tilt is active now rather than from tomorrow's cron.

**yfinance does not work from this container and the failure is silent.** It ships `curl_cffi`, which dies
behind the egress proxy (`curl: (35) Recv failure`) on all six symbols, while plain `requests` to the identical
Yahoo URL succeeds. An empty frame from a dead transport looks exactly like a flat Indian day. `_via_chart_api`
is the fallback and the log names which path each symbol took. **If you are debugging this module from the dev
container, that is why** — it is not a broken ticker.

## 💸 2026-08-05 17:50 — THE DESKS ARE TRADING AGAIN. Buying power went $206 → $34,713.

Run `31031318516` (17:43 UTC), the first healthy run since the margin freeze:

| | 08:15 today | 17:43 today |
|---|---|---|
| buying power | **$206.86** | **$34,713.48** |
| cash | −$33,401.86 | −$22,179.98 |
| equity | $22,013.89 | $21,903.38 |
| orders placed / filled | 0 | **14 / 13** |

**The margin floor is not firing, and that is the correct behaviour** — 0 "margin floor" drops because bp
$34,713 is far above the floor (10% of equity = $2,190). It is a backstop against re-exhaustion, not a
throttle; it only speaks when the account is nearly out. The 2 `insufficient cash` drops are crypto, which
sizes against `non_marginable_bp = $0.00` and is still starved.

**An India symbol traded live**: `ichimoku_cloud_tv/INDA signal=BUY`, filled. All 10 India symbols are in the
104-symbol bars request (105 minus `MKR/USD`, skipped as non-tradable). The expansion works end to end, and
the NSE tilt above now feeds exactly this order path.

Cash is still a −$22k margin debit and the book is levered ~1.6×, so **"reset the paper account" stays on the
operator list** — but it is no longer blocking, and the floor is in place to stop the freeze recurring.

**Crypto is still at zero and it is structural, not a bug.** `non_marginable_bp = $0.00` because cash is a
margin debit, and Alpaca crypto cannot use margin at all. The margin floor cannot help — it reserves buying
power, which crypto cannot spend. The only fix is a *cash* reserve, which caps the equity desks at `cash − X`
and therefore stops them using margin; that is an allocation trade-off, so it is on the operator list rather
than shipped unattended. Consequence worth stating plainly: crypto is the only `always_open=True` desk, so
while cash is negative the platform's **overnight coverage is zero orders, not "quiet"**.

## ✅ 2026-08-05 09:15 — BACKTESTS PERSIST FOR THE FIRST TIME, and a margin floor now stops the freeze recurring

**Backtest persistence VERIFIED in production.** Run `30992181279`:
```
✓ 12 backtests | top Sharpe: 1.29 (rsi/AAPL)
[main ed92ec8] backtest: update results 09:11 [skip ci]
```
`git log --all -- .github/state/last_backtest.json` went from **0 commits to 1**. This workflow had run every
15 minutes since inception, green every time, and had never saved a result. Two causes, both required:
no `permissions:` block (push 403'd) and staging only `last_backtest.json` while the runner also writes
`agent_memory.json`, so `git pull --rebase` aborted on all four retries. `continue-on-error` on both steps is
why it stayed green throughout.

**ML experiments now persist too** — `.github/state/ml_experiments.json`, 60-run rolling. Previously the whole
output was a `print`; the workflow's fallback appends to an issue titled `ml-experiments-log` that **does not
exist** (500 issues paged, step still reports success).

**ML answered, definitively: working, NOT used in trades.** Real walk-forward GBC on real bars, torch-free (the
Render torch exclusion never applied to this path — that is LSTM only). `ml_signal`/`ml_enhanced` appear **zero**
times in `desk_order_placer.py`. And the model **loses to buy-and-hold**: SPY Sharpe 0.56 vs 0.789, QQQ 1.12 vs
1.325. Wiring it into live sizing is a strategy decision, deliberately not taken.

**`MARGIN_FLOOR_PCT` (0.10) added.** Nothing prevented the book levering to a standstill because
`cash_capped_notional` only asked "can we pay for this one?", never "should we spend the last of it?". Against
the real 07:56 account state a $500 SPY order now sizes to **0** instead of ~$196. Crypto exempt (cannot use
margin; already starved). Skip reason reported as `margin floor`, distinct from `insufficient cash`. **Frees
nothing already committed — it stops the state recurring after a paper-account reset.**

## 💣 2026-08-05 16:40 — A SECOND TIME BOMB, and it went off at 15:00 UTC today

`test_desk_trade_sync_env.py::test_env_desk_fills_attributed_to_keyless_account` began failing on main with
`assert written2 == 0` → `assert 1 == 0`. **Not a code regression** — the test and `desk_trade_sync.py` last
changed in #1143 / #764.

**The fixture had an expiry date.** Orders were hardcoded at `2026-07-06T14:00/15:00Z`, and
`sync_desk_trades` dedups against `Trade.closed_at >= now - lookback_days` (30). Measured at 16:39 UTC:
```
lookback_start   2026-07-06T16:39
trade closed_at  2026-07-06T15:00   ← outside the window by 1.7 hours
```
Once the close time aged past 30 days, the dedup query could no longer see the row it had just written, so
the second sync re-inserted it. The test passed until ~15:00 UTC today and has failed since — **exactly 30
days after the hardcoded date, to the hour.** Fixed with relative timestamps (`_ago(hours=25/24)`), which
cannot expire.

**Second instance of this class in two days** (`DENYLIST_TTL_DAYS`, 08-04). Both were tests asserting the
absence of an expiry they were written to live inside. **When a test hardcodes a date and the code under it
has a time window, the test has a shelf life.**

**Swept for a third — none.** `test_desk_trade_sync.py` has 18 hardcoded dates at the same 30-day boundary
but passes: it exercises `reconstruct_closed_trades`, a pure pairing function with no time window
(`grep -c "lookback|closed_at"` → 0). Its dates are relative to each other, not to now. Checked before
"fixing" 18 timestamps that are fine.

**Noted, not acted on:** `test_no_datetime_utcnow_in_source` fails locally on `backend/app/models/backtest.py`
(3 occurrences, present on main, file unmodified) but **passes in CI** — 2034 passed, 1 failed, the time
bomb only. Path resolution differs between environments; CI is the authority for whether main is broken,
and `backend/app/**` is outside the modify-safe set in `scripts/CLAUDE.md`.

## ✅ 2026-08-05 14:44 — THE BACKTEST NOW COVERS BOTH DESKS, verified in production

Run `31016650918`, committed as `98fdaf96`:
```
Binance BTCUSDT: HTTP 451 (geo-blocked from this runner) — falling back to yfinance
  momentum/BTCUSDT: Sharpe=-0.15      mean_reversion/BTCUSDT: Sharpe=-0.19
Binance ETHUSDT: HTTP 451 (geo-blocked from this runner) — falling back to yfinance
  momentum/ETHUSDT: Sharpe=-0.18      mean_reversion/ETHUSDT: Sharpe=+0.30
✓ 16 backtests | top Sharpe: 1.47 (rsi/AAPL)
```
Persisted state: **`total_runs: 16`, `desks: ['equity', 'crypto']`** — up from 12 / equity-only.

**Both open questions answered.** The 451 is real in the Actions runner, not just the dev container — so
Binance is geo-blocked from GitHub's infrastructure. And **yfinance does resolve `BTC-USD` / `ETH-USD`
there**, which could not be checked locally because both hosts are blocked from this container. The fix
needed no third source.

**The whole chain is now verified end to end**, each link having failed silently before:
`contents: write` + staging the whole state dir (0 commits ever → 6) → crypto 451 logged instead of a bare
`None` → yfinance fallback → crypto results tagged and persisted.

**Note the numbers are real and mostly negative** — BTC momentum −0.15, ETH momentum −0.18, ETH
mean-reversion +0.30. That is the point: a backtest that only ever reported equity results was not a
kinder view of the strategies, it was no view at all.

## 🔴 2026-08-05 08:15 — THE ACCOUNT IS OUT OF MARGIN. Nothing has traded for ~7 hours.

**This is now the binding constraint on the entire platform**, ahead of everything else in this file. The desks
are fine. There is no capacity.

Buying power across the last ten desk runs:
```
00:41 bp=$0.00     01:31 bp=$0.00     04:26 bp=$116.98
00:58 bp=$0.00     02:22 bp=$46.35    05:15 bp=$101.23
01:03 bp=$27.61    07:05 bp=$115.79   07:56 bp=$206.86
```
**`cash` is pinned at exactly −$33,401.86 from 00:58 through 07:56.** An unchanging cash balance across eight
consecutive runs means **zero fills** in that window. Equity drifts $21,892 → $22,013 on mark-to-market alone.

**The desks are healthy — this is not a signal or code problem.** Run `30986611287` (07:56):
```
funnel: 51 generated → 17 survived gate+topK (3 exploration) → 0 placed
⚠️ 17 dropped — 13 market closed · 3 no order path · 1 insufficient cash
```
Generation, gating and top-K all work. `MIN_ORDER_USD = 25` against `0.95 × buying_power` means a $0–$200 book
places nothing.

**The safety logic is correctly refusing to fix this, and must keep refusing.** `recover_negative_cash` declines
to flatten while `bp > 0` ("MARGIN DEBIT, not orphaned notional"). That restraint is load-bearing: the
2026-07-27 incident shows flattening a levered book realises losses, trips the daily loss cap and freezes
trading until session rollover. **Do not make it flatten.**

**[USER] The decision is capital, not code.**
1. Reset the Alpaca paper account — fastest, restores buying power, loses position history.
2. Let the book run and accept trading only as positions close.
3. Add a margin-utilisation cap so desks stop sizing up before 100% — prevents recurrence, frees nothing now.

**This supersedes the 2026-08-04 17:45 crypto-starvation entry.** That described equity margin consuming the
non-marginable cash crypto needs. Equity margin is now exhausted as well, so both sides are stalled and the
allocation-policy question has become urgent rather than theoretical.

**Verified simultaneously, so it is not confused with the above:** 54 of 55 frontend GET endpoints return HTTP
200 — no 404s, no 5xx. The web app's plumbing is sound; 18 endpoints simply have nothing to serve, ~10 pages
render blank, and almost all of it traces to the paused durable DB. Full breakdown in `IMPROVEMENTS.md`
under "DEEP REVIEW".

## 📉 2026-08-05 08:00 — KEY PRESENCE IS NOT CAPACITY. 5/47, and the declared limit was fiction.

Run `30986127682` (07:43), the first with pacing:
```
[employee_runner] 47 employees, 1.0s spacing (~0.8 min)
101 × [gemini-key] error: HTTP Error 429: Too Many Requests
RESPONDED_COUNT=5
```
Better than 1/47, still 5/47. **And the spacing was 1.0s, not the 4.0s I expected** — which is the finding.

`1.0s = 60/60`, so `max(rpms)` saw a provider declaring `rpm_free: 60` (sambanova / deepseek / together /
hyperbolic). One of those secrets is non-empty. **But the cascade never called it: 101 errors, all
`[gemini-key]`, and no other provider appears in the log at all.**

So the pacer computed headroom from a key that exists and is never used. **`_has_key()` tests that an env
var is non-empty — not that the provider is reachable, valid, or in the path the cascade actually takes.**
Presence is not capacity. The pacing was correct arithmetic over a number that described nothing real.

**Fixed by measuring instead of declaring.** Declared `rpm_free` is now only the starting guess; the 429s
are ground truth:
- **adaptive backoff** — double the interval on a failed employee (cap ×8 ≈ 53s at the Gemini-only base),
  snap back to base on success. Responds to the provider that is actually answering, whichever it is.
- **an 18-minute loop budget** inside the 25-minute job timeout. A job killed by the runner timeout loses
  the memory write, the proof file and the Discord report — so an overrun would present as a crash rather
  than a partial run. It now stops deliberately and says how many it skipped.
- **`SKIPPED_COUNT` reported separately from `FAILED_EMPLOYEES`.** They are different outcomes, and the
  workflow opens an issue when failures exceed 5 names — folding a budget stop into that would file
  spurious bugs about 42 healthy employees.

**Unresolved, and it is a secret, not code:** some 60-rpm provider has a key that the cascade never
reaches. Either the key is invalid, or `_llm_waterfall` (which runs before the manual cascade and keeps its
own provider order) never gets to it. Worth one measurement next session — the backoff makes the system
survive it either way, but it does not explain it.

## ✅ 2026-08-05 07:45 — PACEMAKER FLEET DISPATCH **VERIFIED**, and the pacing margin that was missing

**Correction to the 07:15 and 07:00 entries, which called this unverified.** It is verified. Run
`30983374228` — the 1/47 run analysed below — was itself the proof, and I misread it as cron:
```
employee-conversations   07:00:45  event=workflow_dispatch  actor=github-actions[bot]
multi-agent-discussion   07:00:47  event=workflow_dispatch  actor=github-actions[bot]
```
Both fleet workflows, two seconds apart, dispatched by the pacemaker. The CI-cascade fix works.

**And that exposed a flaw in the pacing I shipped 30 minutes earlier.** `60 / rpm_free` = `60/15` = 4.0s is
**exactly 100% of Gemini's quota, by construction** — it assumed this workflow is the only consumer.
It is not: **42 workflows map `secrets.GEMINI_API_KEY_1`**, and at least a dozen run on schedules
(agent-status-check, company-brain, brain-health, channel-monitor, collective-learning,
continuous-improvement, …). The quota is fleet-wide, so a 100%-utilisation plan walks straight back into
the 429s it was written to prevent — and the pacemaker now fires two of those workflows *simultaneously*.

`_QUOTA_SHARE = 0.6` reserves 40% for everyone else:
```
gemini only (15 rpm)  ->  6.67s  ->  47 employees ≈ 5.2 min
+ together  (60 rpm)  ->  1.67s  ->  ≈ 1.3 min
```
5.2 min is a fifth of the 25-minute timeout, so the headroom is nearly free.

`multi_agent_discussion` needs no pacing — checked, not assumed: at most **3 LLM calls per run** (one round
of ≤3 speakers), negligible against any limit.

**The lesson worth keeping:** the first pacing fix was correct arithmetic against the wrong denominator.
*Whose budget is it?* is a different question from *what is the limit?*, and only the second one is written
in the provider table.

## 🚦 2026-08-05 07:15 — THE FLEET RAN FOR THE FIRST TIME AND RETURNED 1/47 (paced; next layer down)

Run `30983374228` (07:01) — **the key-guard fix worked**: the runner no longer exits at "No LLM keys
available", it executed. And then:
```
RESPONDED_COUNT=1
FAILED_EMPLOYEES=alpha_dir,ml_lead,risk_eng,backend_lead,qa_dir,devops_dir,… (46 of 47)
```
Whole run: **28 seconds**. Log is one provider repeating:
```
[gemini-key] error: HTTP Error 429: Too Many Requests
[gemini-key] error: HTTP Error 503: Service Unavailable
```

**This is arithmetic, not an outage.** The cascade is fine — it falls through correctly, but every other
tier is keyless, so **Gemini is the entire budget**. `llm_common._PROVIDERS` declares `"rpm_free": 15` for
it. 47 calls in ~20s is **~140/min against a 15/min limit**.

**`rpm_free` is declared for all eight providers and was enforced NOWHERE.** The data existed without the
behaviour — the same shape as `desk_outcomes` being listed but not searched, and `strategy_performance.json`
being written but not read. Spacing is now derived from that table (not a second hardcoded copy) using the
**most permissive provider that actually has a key**, since that is the one the cascade settles on:
```
gemini only (15 rpm)  -> 4.0s   -> 47 employees ≈ 3.1 min
+ groq      (30 rpm)  -> 2.0s
+ together  (60 rpm)  -> 1.0s
```
3.1 minutes of sleep sits well inside the 25-minute timeout set an hour earlier; a test asserts that
relationship holds rather than leaving the two numbers to drift apart.

**What this does NOT fix:** one provider is still the whole budget. Populating `GROQ_API_KEY_1` /
`DEEPSEEK_API_KEY_1` / `TOGETHER_API_KEY_1` would both raise the ceiling and shorten the run — the pacing
math above adapts automatically. That is an operator action, not a code change.

**Fourth layer in one night.** cap → blocked by the key guard → blocked by the pacemaker cascade → blocked
by rate limits. Each was only visible once the one above it was fixed, and each looked green from outside.

## ⏱️ 2026-08-05 07:00 — A RISK I CREATED: the fleet workflow had no timeout, and was about to do real work

`employee-conversations.yml` had **no `timeout-minutes`**, so it inherited GitHub's **6-hour** default.
That was invisible for as long as it mattered least: the key guard exited at line ~24 and every run
finished in **0.3–0.8 min** having done nothing.

Fixing the guard (05:00) and adding the pacemaker dispatch (05:40) changed both sides of that at once. The
job now makes **47 sequential LLM calls**, each with a possible quality-gate retry, and fires every ~50
minutes on top of `cron: '5 * * * *'`. A hung provider would have burned six hours of free-tier budget per
run — a hazard created by my own two fixes, in a workflow whose duration history says 0.8 min.

Set to **25 minutes**: generous for 47 calls (`multi-agent-discussion` allows 10 for a smaller roster,
`desk-trading` 15) and bounded well under the dispatch interval. If a real run approaches it, cap the
roster with `EMPLOYEE_RUNNER_LIMIT` rather than raising the timeout.

**Generalised, not patched:** a test now asserts every workflow the pacemaker dispatches declares a
`timeout-minutes` that is under the ~50-minute dispatch interval. **Waking something on a schedule makes
its runtime a budget question**, and the next workflow added to that list gets the check for free.

Related, already safe: GitHub allows only one *pending* run per concurrency group, so queued runs cannot
pile up unboundedly — `cancel-in-progress: false` bounds it at one running plus one pending.

## ✅ 2026-08-05 06:30 — THREE FIXES VERIFIED IN PRODUCTION (one still unverified — say which)

Run `multi-agent-discussion` at 06:26, commit `bb2a9092`. All three landed together because they ride the
same producer.

**1. `conversations` cap executed.** `915 → exactly 300` entries; `agent_memory.json` **933,178 → 523,990
bytes, 44% smaller**. It had been merged for over three hours without running — nothing was wrong with it;
no producer had written.

**2. Attribution injection (#814) executed.** Six lines, and the construction held:
```
[attribution @ 2026-08-05T06:27] avellaneda_stoikov_mm: 10 trades, 90% win rate, +53.48% ...
[attribution @ 2026-08-05T06:27] stat_arb_e: 21 trades, 0% win rate, -37.17% total return ...
```
Five best plus **the worst performer**, which is the whole point — `stat_arb_e` ranks last by total return
and would be cropped by any top-N.

**3. Echo filter executed.** 13 entries written by that run, **0 echoes**. The newest surviving echo is
timestamped `2026-08-05T02:28`, before the filter merged. 51 historical echoes remain in the rolling
200-window and will age out; they are not evidence of a leak.

**NOT verified: the pacemaker fleet dispatch.** The 06:26 run was `event=schedule` — free-tier cron, not
the new dispatch. That fix still needs a pacemaker cycle carrying the new step. **Do not read this entry as
verifying it.**

**The pattern worth keeping.** Each of the last three fixes was blocked from verification by a deeper one
underneath: the `conversations` cap could not run because `employee-conversations` exited at its key guard;
that workflow barely ran because the pacemaker's cascade never reached it. Three layers, and the top only
surfaced because "the unit tests pass" was refused as proof of execution.

## 💔 2026-08-05 05:40 — THE PACEMAKER'S CI CASCADE HAS NEVER REACHED THE FLEET (fixed)

**This corrects the central claim of `pacemaker.yml` and of the 08-03 entries below.** The design is: 36
workflows chain off `workflow_run: workflows: ["CI"]`, so dispatching CI every 50 minutes revives all of
them. Measured 2026-08-05, dispatching CI revives **none** of them.

```
CI on main, dispatched by the pacemaker — 01:26, 02:16, 04:21, 05:11, all success,
  triggering_actor = github-actions[bot]
Downstream workflow_run events from those four completions:  ZERO

Last real cascade: 01:01 — Peer Review + Desk Trading + Fill Tracker +
  employee-conversations + multi-agent-discussion, all at once.
  triggering_actor = bahllaavanye-afk (User)  ← a PR merge
```

**Cause: GitHub's recursion guard.** "Events triggered by the GITHUB_TOKEN will not create a new workflow
run." `workflow_dispatch` is the documented exception that lets the pacemaker *start* CI — and that part
demonstrably works. But the CI run it starts is itself GITHUB_TOKEN-triggered, so **its** `completed` event
cascades to nothing. **The exception covers the dispatch, not the descendants.**

**So the pacemaker only keeps alive what it dispatches DIRECTLY** — CI, the merge gate, desk-trading. The
agent fleet has been running on free-tier cron alone the whole time: `employee-conversations`
(`cron: '5 * * * *'`, i.e. hourly) actually ran at 20:36, 22:16, 00:09, 04:14 — **four times in eight
hours**. That is exactly the symptom the pacemaker was built to cure: *"Discord is only active when this
chat resumes"* — because a human merging a PR is the only thing that cascades.

**Fixed** by dispatching `employee-conversations.yml` and `multi-agent-discussion.yml` directly, the way
desk-trading already is. Deliberately a short list, not all 36: dispatching every chained workflow every 50
minutes is a large, unmeasured change in free-tier minutes. Both set `cancel-in-progress: false`, so a
dispatch overlapping a cron run queues instead of clobbering it. The crypto desk stays excluded — see the
measured collision in the desk-trading comment.

**The step that claimed otherwise was renamed.** It was "Dispatch CI to drive the 36 downstream workflows";
it is now "Dispatch CI (heartbeat — does NOT cascade to the fleet)". Three findings this session were
independently re-derived because a correction lived somewhere other than the claim it corrected. **Fix the
label, not just the docs.**

## 👥 2026-08-05 04:40 — THE HOURLY EMPLOYEE-CONVERSATIONS WORKFLOW HAS NEVER RUN (fixed)

47 employee personas, hourly, green every time, `Responded: 0/47` posted to `#engineering`. Production log,
run at 04:14 UTC on sha `071df8db`:
```
No LLM keys available — skipping real conversations
```
`sys.exit(0)` — a clean exit, so the step passed and the workflow was never red.

**Two independent causes, and the second is the interesting one.**

**(a) The workflow mapped the wrong secret names.** The repo's populated free-tier secrets are the **`_1`
variants**. The three workflows that demonstrably get real replies — `agent-status-check`,
`multi-agent-discussion`, `claude-chat` — all map `GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY_1 }}`.
`employee-conversations` mapped the **unsuffixed** `secrets.GEMINI_API_KEY`, which is empty. Now mapped as
`secrets.X_1 || secrets.X`, so it cannot regress if the unsuffixed name is later populated.

**(b) The guard kept its own copy of the provider list, and it disagreed with the cascade.**
`_LLM_KEY_VARS` listed seven env vars. `llm_common._PROVIDERS` has **eight** — the missing one is NVIDIA
(`NVIDIA_AGENTS_API_KEYS`, alt `NVIDIA_NIM_API_KEY`), **which this workflow does supply.** So a pre-flight
check vetoed a cascade that had a usable provider. The fix is not to add NVIDIA to the list; it is to stop
keeping a second list — `_any_llm_key()` now asks `llm_common` what it can reach. Same failure family as
the smoke gate keying on `database.ok`: *a guard that maintains its own model of the thing it guards will
drift from it.*

**The other 18 workflows mapping unsuffixed secrets are NOT broken — measured, don't "fix" them.** Every
one of them also maps at least one populated secret (usually `GEMINI_API_KEY_1`), so they have a working
provider; the empty mappings are dead weight, not breakage. Adding `_1` fallbacks there would buy free-tier
headroom, not correctness.

**A test-isolation leak found on the way, and fixed.** `test_frontend_design_guard.py` replaced
`sys.modules["llm_common"]` with a three-function stub and never restored it. Every later test that
imported `llm_common` got the shim. It made the new key-guard tests **fail in the full suite while passing
alone** — the guard's fail-open `except` branch was firing on the stub's missing attributes, not on any
real condition. Now restored in a `finally`. **A leaked module stub can invent or mask failures anywhere
downstream of it**; that this surfaced as "my new tests fail" rather than "an old test lies" was luck.

## 🎭 2026-08-05 03:50 — THE AGENTS DISCUSSED STATUS BECAUSE NOBODY GAVE THEM RESULTS (fixed, #814)

Two halves of one loop, found together.

**1. The attribution artifact reached no agent.** `backend/performance_log/strategy_performance.json` is
written by `fill_tracker.py` from real filled orders, attributed via the `client_order_id` encoding, and
it **is** committed — 22 strategies, 247 tracked order ids, regenerated at 00:55. Nothing read it into
agent context. So the daily discussion ran on self-reported status while the actual results sat in a file:
```
avellaneda_stoikov_mm  10 trades   90% win   +53.48%
stat_arb_etf           30 trades   93% win   +31.89%
options_pc             14 trades  100% win   +24.03%
stat_arb_e             21 trades    0% win   −37.17%   <- never discussed
```
`shared_context.outcome_learnings()` now injects these into `peer_learnings` through
`multi_agent_discussion`. **The worst performer is always included**, outside `top_n`, by construction:
reporting only winners is exactly the status theater the item names, and a 0%-win strategy is the most
useful thing in the list.

**2. 28% of `peer_learnings` was the model restating its prompt.** 56 of 200 entries:
```
[investor_pipeline @ …] The user asks: "Give a one-sentence status update: what are you…"
[self_improver @ …]     We need to respond as self_improver agent, autonomous, 2 sentences max…
```
`agent_status_checker.py:247` and `multi_agent_discussion.py:360` append the raw reply with no quality
check, so a **failed generation is stored as a learning**. That was merely wasteful until the retrieval
fix landed hours earlier — these entries are now *retrieved into other agents' prompts*, so the noise
compounds. `is_low_quality_learning()` filters at the write boundary. On live data it rejects 65 of 200.

**The patterns are narrow on purpose.** A false positive silently deletes a real finding, which is worse
than keeping one echo. `test_ordinary_learnings_are_never_dropped` pins realistic entries that must
survive.

**Testing note worth keeping:** a mutation reversing the attribution sort **survived** the first test
pass. `stat_arb_e` was still present — it just landed in the leading slice instead of the trailing one —
so a membership assertion could not see that "top N winners + the worst" had become "worst N + the best".
**Assert composition and position, not membership**, whenever a ranking decides what a human or an agent
reads first.

**Noted, not fixed:** attribution keys are still truncated to 10 chars in places (`time_serie`,
`stat_arb_e`). `desk_order_placer._expand_truncated` handles this for trims; the performance file itself
carries the short form.

## 🗜️ 2026-08-05 03:00 — 47% OF THE GIT REPOSITORY IS ONE UNBOUNDED STATE FILE (growth stopped)

`agent_memory.json` has a `conversations` dict that three writers append to and **none** trimmed.

Measured 2026-08-05:
```
conversations      576 KB   915 entries   <- no cap
employee_context   231 KB    11 entries   <- capped (_HISTORY_CAP = 60/employee)
peer_learnings      38 KB   200 entries   <- capped [-100:] / [-200:]
failure_traces       7 KB    41 entries   <- capped [-200:]
                   ------
agent_memory.json  933 KB
```
Growth ~7 KB per commit (917,453 B at 00:15 → 933,178 B at 02:29). **200 of the last 200 commits touching
the repo rewrote the whole blob**, so git holds 2340 copies: **59.1 MB of a 125 MB `.git`.**

The retention bought nothing. The only reader is `context_sync.py`:
`recent = sorted(convs.items())[-20:]`, then it displays 10. ~900 entries retained to serve a consumer
that reads 20. The file already capped its other three structures; the discipline never reached the
largest one.

Capped at 300 in `shared_context.trim_conversations`. Applied against the live file: **915 → 300 entries,
933,178 → 521,231 bytes, 44% smaller**, newest entry retained.

**Only the three PRODUCERS trim, and that is sufficient — do not "finish the job" on the other eleven.**
`agent_memory.json` has ~14 writers (`heartbeat`, `signal_runner`, `peer_reviewer`, `system_watchdog`,
`agent_chat_handler`, …). Each is a separate short-lived process: load → mutate → rewrite. A writer that
adds no conversation entries can only preserve what it read, so **trimmed entries are never resurrected**
and the dict cannot pass the cap. Patching all 14 would be 11 no-op edits and 11 chances to regress. The
producers are `claude_conversations.py`, `employee_conversation_runner.py`, `multi_agent_discussion.py`;
a test fails if a fourth appears.

**The trimmed file is deliberately NOT committed here.** Shipping a state snapshot from a branch reverts
whatever the bots wrote to main in the meantime — the exact defect recorded at `IMPROVEMENTS.md:857`. The
bots shrink it themselves on their next write.

**This does not shrink the existing 59 MB of history.** That needs a history rewrite, which is an operator
decision and is not done here. It stops the growth.

**Two false alarms on the way, both from my own detectors:**
- An AST scan flagged 11 "uncovered writers". Most were real writers but irrelevant (see above), and two
  (`context_sync`, `token_usage_monitor`) were pure false positives — the scan matched any file that
  *mentioned* the path and wrote *something*, including a different file.
- A substring match on `"conversation"` flagged `agent_chat_handler`, which writes `agent_conversations` —
  a different structure, already capped at `[-100:]`. Match the exact key.

## 🧠 2026-08-05 02:10 — THE RETRIEVAL UPGRADE DROPPED WHAT THE DESKS ACTUALLY DID (fixed)

`memory_manager.SemanticRetriever` — pure-python TF-IDF, live for a while (`_MEMORY_MANAGER_OK` is True,
so every `inject_company_context=True` prompt routes through it) — searched a hardcoded category list:
```python
["episodic", "skills", "chat_insights", "github_insights", "trade_outcomes", "experiment_results"]
```
Three of those six have **never existed** in `company_brain.json`. And `desk_outcomes`, which does exist
and held **100 of the brain's 403 entries**, was not in it. Each entry is one desk run:
```json
{"channel": "desk-commodities", "source": "desk_run_summary",
 "summary": "*Commodities Desk* — 3 order(s) placed 🟢 `time_series_momentum/SLV` BUY $200 conf=100% …"}
```
Side, notional, confidence, per order — the most decision-relevant memory a trading firm has, reaching no
agent prompt.

**It is a regression, not an oversight.** The recency path it replaced (`llm_common.get_company_context`,
~line 951) *did* include them: `brain.get("desk_outcomes", [])[-3:]`. So moving to semantic retrieval made
agent context strictly worse for trading outcomes while looking like an upgrade.

**Why nothing caught it: the failure mode is a NARROWER result set, not an empty one.**
`search("crypto desk orders placed")` still returned four plausible `episodic` hits. A silent narrowing is
indistinguishable from a specific query. Fixed by `DEFAULT_SEARCH_CATEGORIES` + `_unsearched_categories()`,
which reports populated brain keys outside the search set.

**That reporter is a RUNTIME stderr line, not a CI assertion — deliberately.** The brain is written by
background bots. A test asserting on the live file lets a 3am bot commit turn the agent suite red and block
every PR under `pytest -x` — the denylist TTL failure of 2026-08-04, which this repo has now paid for once.

**Two traps found in passing, both worth not re-learning:**
- **Tiny corpora score zero.** IDF is `log(N / (1 + df))`, so on a small or homogeneous fixture every term
  scores ≤ 0 and the `score > 0` filter empties the result. Two of my first tests failed on this, not on the
  code. Fixtures need enough *varied* documents. Do not "fix" it by changing the formula — that re-ranks the
  whole brain and needs its own justification. The live brain (~400 docs) is unaffected.
- **`_BRAIN_FILE` is relative** (`Path(os.environ.get("GITHUB_WORKSPACE", ".")) / ".github" / "state"`).
  Run from anywhere but the repo root it silently resolves to nothing and every search returns 0 hits. I
  read that as total breakage for several minutes. Same shape as `_too_large` in `continuous_improver.py`.
  **When an agent script returns suspiciously empty, check cwd before diagnosing.**

## 🫀 2026-08-05 01:00 — "17 of 25 PACEMAKER RUNS CANCELLED" IS HEALTHY. Do not re-fix it.

I nearly reported this as a regression of the `cancel-in-progress` bug fixed on 08-03. It is not.

Last 25 pacemaker runs: **8 success, 17 cancelled.** Every success ran 51.4 / 59.9 / 63.6 / 63.6 / 51.9 /
86.9 / 76.4 min — all past the 50-minute sleep, all dispatching. Desk-trading dispatches landed at 22:05,
22:55, 23:46, 00:36 — ~50 min apart, no drift.

**The cancelled runs never allocated a runner.** `GET /actions/runs/<id>/jobs` returns `{"jobs": []}` for
every one checked (30962371134, 30960506332, 30959430157). They are **pending-queue evictions**: with
`concurrency: {group: pacemaker, cancel-in-progress: false}` GitHub permits at most ONE pending run per
group and cancels the pending one when a newer arrives. The pacemaker fans in from many `workflow_run`
sources, so several queue per cycle and all but the newest are evicted.

**The reported duration of a cancelled run is QUEUE time, not run time** — 0.0 to 34.7 min of waiting
before being superseded. That is what makes this look like the old bug, where runs died *mid-sleep* at up
to 47.9 min. Duration does not distinguish them. **Whether a runner was ever allocated does.**

`cancel-in-progress: true` is the actual bug: it cancels the RUNNING sleeper, which killed 25 of 30 runs
before 08-03.

## ✅ 2026-08-05 01:20 — the post-deploy smoke can finally see a dead database (IMPROVEMENTS 843, shipped)

The smoke test asserted on `/health` and nothing else. `/health` returns `{"status": "ok"}`
unconditionally and does no DB work — deliberately, so the Render keep-alive ping stays cheap. So every
subsystem the backend already computes was invisible to the only automated post-deploy gate, and smoke
stayed **green through more than a week of paused database**.

**The item's own field cannot fire.** `IMPROVEMENTS.md:843` specified "fail deploy on `database.ok=false`".
`main.py:487` sets `database.ok = True` whenever `SELECT 1` succeeds, and on the SQLite fallback it does —
the fallback is functional, just ephemeral. Live payload with Supabase paused:
```
"database":         {"ok": true,  "latency_ms": 5.2, "fallback": "sqlite"}
"database_primary": {"ok": false, "error": "(ENOTFOUND) tenant/user postgres.vexzwnfbmznvxoxxktax ..."}
```
`database.ok` is **true during the exact outage the guard exists to catch**. `database_primary`
(`main.py:502`, emitted only when `db_fallback_active`) is the field that reports it. Shipped keyed on
`database_primary`; the test that distinguishes the two is
`test_a_functional_sqlite_fallback_is_still_a_failure`.

**Strictness is scoped to deploys.** `SMOKE_FAIL_ON_DEGRADED_DB` is set only for `push`, so a degraded
primary fails the post-deploy run while the 30-min schedule reports it as a warning. Also asserts
`mode == "paper"` from the same payload — the only automated check positioned to see TRADING_MODE drift.

**Expect main to be RED on smoke until Supabase is unpaused.** That is the gate working. The Discord page
is suppressed while the durable-DB check is the *only* failure (~10 push-triggered runs/day would
otherwise re-page operator decision #2); the run still fails and the step summary still lists it. Any
second failure clears the suppression and the page fires naming both.

## 🔬 VERIFICATION DISCIPLINE — 2026-08-04/05. Read this before trusting any check you just wrote.

Two findings from the same day, same root cause: **a check that reports nothing is
indistinguishable from a check that found nothing.** This is the inverse of the
"green-looking absence" bug family this repo keeps hitting in production code —
here it happened in the tools used to *verify* production code, which is worse,
because it manufactures both false alarms and false all-clears.

### The rule
**Validate the probe on a known-positive before trusting a negative.** A scan that
returns "clean" is only evidence if you have first watched it return "dirty" on a
case you already know is dirty. Cost: one extra run. Without it, "no results" and
"the scan never executed" produce identical output.

### Case 1 — four broken checks in one tick produced a false "work lost" alarm
Chasing whether session work had been lost to the reset/force-push pattern, three
of my own verification scripts were silently wrong:
- an **orphan scan that scanned 0 shas** — its input list came back empty, the loop
  body never ran, and the empty output read as "no orphans found";
- an **ancestry scan that flagged ~199 commits** — it used
  `git merge-base --is-ancestor`, but a **squash-merged commit is never an ancestor
  of `main`**. Ancestry is the wrong test for "did this land". Content is;
- `grep -qF "$line"` erroring `invalid option` on every probe line beginning with
  `-`. Fix: `grep -qF -- "$line"`.

**Resolved by a content audit whose method was first proven on a known-merged
commit**, then applied to 36 doc commits: **0 lost.** The alarm was entirely an
artifact of the instruments.

### Case 2 — a test that was a time bomb, and the sweep for its siblings
`test_the_shipped_denylist_parses_and_contains_the_confirmed_asset` asserted
`"MKR/USD" in dop._denylisted_assets()`. That is a **time-dependent** fact: the
shipped entry carries `since=2026-07-28T21:52:00Z` and `DENYLIST_TTL_DAYS = 7`, so
the assertion was true when written and false exactly seven days later. Measured at
2026-08-04 23:39 UTC the entry was 7 days 1 hour old — expired about an hour
earlier. Nothing was broken; the TTL did exactly what its own docstring says it
exists to do, and the test was asserting the **absence** of the expiry it was
written to protect. Under `pytest -x` this took out a required check on **every**
open PR. Fixed in #1413 by asserting against the raw shipped JSON (time-independent)
plus `isinstance(dop._denylisted_assets(), set)` — the evidence stays pinned,
removing `MKR/USD` from the file still fails, and the clock no longer participates.

**Sweep for other time bombs — none found**, and the search surfaced the
known-positive (`DENYLIST_TTL_DAYS`, lines 1673/1707) before returning the negative,
which is what makes the negative worth recording:
- `DENYLIST_TTL_DAYS` is the **only** TTL applied to shipped state in
  `desk_order_placer.py`;
- `_trimmed_strategies()` (`.github/state/strategy_trims.json`, line 1782) has **no**
  expiry — trims persist until explicitly recovered;
- the remaining `timedelta` uses are **API query windows**, not state filters: a
  420-day bars lookback (line 577) and the options DTE window (lines 1310-1312);
- of the six test files that reference `.github/state/`, none assert a decaying fact
  about a shipped file's contents — the references are docstrings or workflow-source
  assertions.


## 🔴 WHY THERE ARE NO TRADES — answered 2026-08-03, full writeup in `docs/REVIEW_2026-08-03_WHY_NO_TRADES.md`

The desks are **not** broken and the strategies are **not** silent. Signals fire on
nearly every run and are then discarded, for two unrelated mechanical reasons.

**Equity-side desks (8 of 9) — right signals, wrong time.** Run `30673525449`
logged `conf=1.00` on SLV, `1.00` on EPOL, `0.98` on EIDO, `0.80` on IWM and placed
nothing, because it ran at 23:44 UTC and `desk_open = is_open or desk.always_open`
was false. Only **12 of desk-trading's last 30 runs landed inside RTH** (5 Wed / 4
Thu / 3 Fri) against a nominal 26 *per day* — ~15% of intended in-window cadence.
Two causes:
- the pacemaker was cancelling itself — `cancel-in-progress: true` on a
  sleep-3000s-then-dispatch job, **25 of 30 runs cancelled**, all dying short of
  the 50-minute sleep (max 47.9 min), while the only 4 successes took 50.4 min.
  **Fixed** (`cancel-in-progress: false`) and the pacemaker now dispatches
  `desk-trading.yml` directly every ~50 min.
- ~~the `workflow_run: ["CI"]` trigger on both desks has never fired~~ —
  **CORRECTED 2026-08-03 08:50.** The 30-run sample containing zero
  workflow_run events was accurate; the inference from it was wrong.
  `workflow_run` only fires for upstream runs on the DEFAULT branch, and CI runs
  almost entirely on `pull_request` (head_branch = the PR branch), so the trigger
  was **dormant, not dead**. The moment CI actually ran on main it fired on both
  desks — 06:43 and 07:45, both `success` on desk-trading.

### ✅ TRADING IS WORKING — verified 2026-08-03 13:43
Run `30818846913` (13:37 UTC, `workflow_dispatch` from the pacemaker, first one to
land inside RTH): **`Done. 7 orders placed across 9 desks.`** The chain is closed:
pacemaker survives its sleep → dispatches desk-trading → lands in market hours →
real paper orders. Dispatch cadence is exact: 10:16, 11:07, 11:57, 12:47, 13:37,
14:28 — every 50 minutes, no drift. Cron contributed **one** run all day (12:17).

### 🧠 ML IS AIMED AT THE ONE MODEL FAMILY THAT CANNOT RUN IN PRODUCTION (2026-08-03 21:45)
Before doing any more LSTM promotion work, read this. It cannot pay off on the
current hosting, and the reason is deliberate.

**torch is excluded from Render on purpose.** `backend/pyproject.toml:54-56`:
> `# ML inference — in [ml] optional group so Render free tier skips the 800MB torch wheel`
> `# Render installs: pip install -e "."   (no torch = ML strategies degrade gracefully)`

`render.yaml:19` confirms it: `pip install -e "."` — no `[ml]` extra. So
`torch: available: false` in `/health/detailed` is **correct behaviour, not a bug**,
and LSTM / SSM / Mamba / PatchTST inference is impossible there at any artifact
quality.

**But Render DOES get xgboost, lightgbm and scikit-learn** — they are base
dependencies (`pyproject.toml:40,45,46`). And `app/ml/inference.py` loads four
exact filenames, **three of which need no torch**:
`lstm_latest.pt` (torch), `xgboost_latest.ubj`, `lorentzian_latest.pkl`,
`scaler_latest.pkl`.

**The mismatch:** the only CI trainer is `.github/scripts/ci_lstm_trainer.py` —
LSTM, torch-only. A repo-wide grep for `xgboost_latest` / `lorentzian_latest`
returns only `app/main.py` (the health check) and `app/ml/inference.py` (the
loader). **Nothing anywhere produces them.** The backend trainers are
`train_lstm`/`train_ssm` (torch) and `train_ppo_exec`/`train_rl`
(stable-baselines3 → torch). `app/ml/models/xgboost_model.py` exists as a class
with no pipeline behind it.

So: six successful weekly LSTM training runs produced artifacts for the one
runtime that is not installed, while the two runtimes that ARE installed have no
trainer at all.

**Consequence for the IMPROVEMENTS ML items** (unify the two `LSTMPredictor`
architectures, promote `lstm_latest.pt`, verify the round trip): all real work,
none of it can change production behaviour while torch is absent. Do not treat
them as the next ML step.

**The two genuine options, both operator calls:**
- **Torch-free path (no hosting change):** add an XGBoost/LightGBM trainer to CI
  and promote to `xgboost_latest.ubj`. Works on Render *today* — the runtime,
  the model class and the loader all already exist; only the trainer is missing.
- **Torch path:** host inference where torch fits (paid Render tier or a separate
  worker), which then makes the LSTM unification work worth doing.

### 🧩 WHY Vercel blocks improver PRs but not mine — resolved 2026-08-03 18:40
My PRs merge while ~100 improver PRs sit frozen on a Vercel `failure`. That is not
gate inconsistency. `frontend/vercel.json` has:
```json
"ignoreCommand": "git diff --quiet HEAD^ HEAD -- ."
```
Vercel skips the build when this exits 0 (no diff). My PRs touch only backend and
workflow files, so the build is skipped and Vercel posts **`success` —
"Canceled by Ignored Build Step"** (verified on `#1366`, sha `290522ee`). Improver
PRs attempt a real build, hit the 100/day free-tier cap, and get `failure`.

**The command only inspects the TIP COMMIT** (`HEAD^ HEAD`), while every improver
PR has **5 commits** (verified on `#1352` and `#1358`). So whether a preview builds
is decided by whether the *last* commit happened to touch `frontend/` — not by
whether the PR does. Quota gets burned on PRs whose net frontend diff may be
nothing, and previews are skipped for PRs that genuinely change the frontend.

To be precise about severity: build **correctness** is NOT at risk — `frontend-build`
is a REQUIRED CI check and runs regardless. A false skip loses only the preview
deployment. The real cost is the quota burn, and the `failure` status it produces
once the cap is hit, which is what freezes the merge gate.

**`frontend/vercel.json` is under "Do NOT Modify" (`frontend/src/CLAUDE.md:40`), so
this was NOT changed here.** Operator options, both one-liners in that file:
- Skip previews for bot branches outright — prepend
  `case "$VERCEL_GIT_COMMIT_REF" in improver/*) exit 0;; esac` to the ignoreCommand.
  Directly ends the cap burn and unfreezes the backlog.
- Or compare the whole PR range instead of the tip commit, so the decision reflects
  the PR's actual frontend diff.

### ✅ RESOLVED: the cash "blocker" was transient (2026-08-04 13:50) — and a correction to the "27% of the clock" claim
**The cash block cleared by itself.** First in-hours run after the open, `30915083852` (13:41):
`Done. 11 orders placed across 9 desks.` — more than the 7-9 of the previous session.
So `insufficient available cash (< $25; frees as pending closes fill)` meant exactly what it said: a
temporarily fully-deployed book overnight, not a locked account. **Two things I got wrong and am correcting
rather than quietly dropping:** (a) I flagged it as a possible new *binding* constraint — it was not;
(b) I named agb8 the prime suspect for consuming the buying power — if a second backend were draining the
account, cash would not recover cleanly at the open, so that specific argument is withdrawn (agb8 still
warrants shutdown on duplicate-order and attribution grounds).
Original entry follows.

### 💵 NEW BLOCKER + a correction to the "27% of the clock" claim (2026-08-04 07:40)
Run `30885806270` (06:56, on `f5800e2e`) shows the crypto desk reaching order placement and being stopped by
something I had not seen:
```
[stage] ✓ Apply confidence threshold + top-K filter — passed=12  filtered=39  explored=1
· crypto_adaptive_trend/LTC/USD skipped — insufficient available cash (< $25; frees as pending closes fill)
```
**Two things follow.**

1. **CORRECTION to the 23:40 entry.** I wrote that outside RTH "the whole fleet is structurally idle" because the
   crypto confidence ceiling sits under the 0.60 gate. Too absolute. That was inferred from ONE run (`30860193311`,
   which showed `Place orders — 0.01s` and nothing reaching placement). Here **12 signals passed the gate** and a
   crypto signal reached placement. The ceiling analysis still holds as arithmetic — high realized vol caps the score —
   but vol varies, so crypto signals DO clear the gate some of the time. "Structurally idle" overstates it; "idle most
   of the time, and the gate is the usual reason" is accurate.

2. **A different blocker is now binding: the paper account is out of free cash.** `< $25 available` means no new
   position can be opened regardless of signal quality. Yesterday the same desk placed 7-9 orders per run totalling
   ~$3.5k, so this is recent — the book is now fully deployed.
   **This plausibly connects to the agb8 hazard below:** a second backend trading the same account consumes the same
   buying power. Not proven, and worth not asserting — but it is the first candidate to check.
   **Next step:** read the desk's `Account equity=$… cash=$… buying_power=$…` line (stage 2 of any run, early in the
   log) during market hours to see the actual cash position.

### 🔁 The improver watchdog is tautological — it can never fire (found 2026-08-04 10:45)
`system_watchdog.py:133-137`:
```python
age_hours = (now - commits[0].committer.date) / 3600      # LATEST COMMIT FROM ANYONE
if age_hours > 2:
    return False, f"last commit {age_hours:.1f}h ago — improver may be down"
```
It claims to detect a dead improver, but it reads the newest commit **by any author**. Measured over the last 24h,
`main` took **104 commits, ~82 of them state-bots**:
```
12  watchdog: auto-heal state files      6  status: agent roll call
11  chore(health): heartbeat             6  chore(status): refresh SYSTEM_STATUS.md
10  learn: distill skills                5  chore: company brain sync
 9  discuss: peer learnings updated      5  autopilot: strategy exploration run
 7  scan: market scanner state update    ... and 1 (one) actual improve() commit
```
So the 2-hour window is satisfied permanently — **including by the watchdog's own commit**. Its diff is literally
`"detail": "last commit 40m ago"` → `"last commit 30m ago"`: it resets the metric by measuring it.

Two of those bots are pure bookkeeping, same shape as the OA scout fixed in #1394:
- `chore(health): heartbeat` — increments `health_check_count: 381 → 382` and bumps `last_health_check`. Nothing else.
- `watchdog: auto-heal state files` — bumps `last_updated` / `last_watchdog_run` and the self-referential detail above.

**Why this matters now:** exactly one `improve()` commit landed in 24h, because improver PRs are frozen behind the stale
Vercel `failure` status. The improver IS effectively down, and the watchdog built to say so has been reporting healthy
the entire time.

**NOT fixed here, deliberately.** The correct fix is to count only improver commits (message prefix `improve(`) rather
than any commit — but that would immediately and repeatedly alert Discord about something already known and already
tracked (the Vercel freeze), and the 2h threshold was calibrated against "any commit" so it needs re-picking too.
Adding a loud recurring alarm for a known, decided-upon condition is noise, not signal. Sequence: unfreeze the PR
backlog first, then narrow this check and choose a threshold against observed improver cadence.

### 🚨 LIVE: a SECOND backend is running against the same Alpaca paper account (verified 2026-08-04 02:45)
`IMPROVEMENTS.md` has carried this as `[P1] agb8 double-execution hazard` for days. **It is still live.**
`GET https://quantedge-api-agb8.onrender.com/health/detailed` → HTTP 200:
```
mode: paper   version: 2.0.0
alpaca:           {"ok": true, "note": "connected"}      <-- same paper account as 9jz0
background_tasks: {"ok": true, "running": 11, "total": 11}
strategies:       {"ok": true, "count": 113}
scheduler:        {"ok": true}
database:         {"ok": false, "error": "[Errno -2] Name or service not known"}
```
So an OLD build, with a **dead database**, is running 11 background tasks and a live scheduler while **connected to the
same Alpaca paper account** as the keeper `9jz0`. Per `backend/CLAUDE.md`, those tasks include `StrategyRunner` — "one
asyncio task per (strategy, symbol), runs 24/7".

**Why this matters beyond duplicate orders.** The desk reads LIVE Alpaca state for three separate decisions:
`_kelly_notional` (equity), `daily_loss_cap_hit` (equity vs last_equity), and `is_risk_reducing` (position map). If a
second service is trading the same account, all three are computed against a book this desk does not fully own — and
the slippage measurements added 2026-08-03 attribute fills to desk decisions that another actor may have moved.

**Proven:** the service is up, in paper mode, Alpaca-connected, with live background tasks and 113 strategies.
**Not proven:** that it has actually placed an order today. Its DB is dead, so DB-backed bot definitions likely fail to
load; registry-driven `StrategyRunner` loops are the exposure. Do not overstate it as confirmed duplicate execution.

**USER ACTION (~30s):** suspend or delete the `quantedge-api-agb8` service in the Render dashboard. Nothing in the repo
can reach it — this cannot be fixed from code.

### 💡 WHY CRYPTO CANNOT TRADE: the equity desks spend the cash it needs (found 2026-08-04 17:45)
A second, independent blocker on the crypto desk — one that operates **even when signals clear the 0.60 gate**.
This is a capital-allocation interaction, not a bug: every component is individually correct.

Live evidence, run `30930093709` (16:45, 12 orders):
```
*QuantEdge Desk Run* (16:45 UTC)  equity=$21,968.66  cash=$-2,554.56  buying_power=$49,855.63  regime=bull/calm
funnel: 48 generated → 18 survived gate+topK (3 exploration) → 12 placed
execution: avg slippage +1.2 bps · worst +10.0 bps · 10/12 measured (2 unmeasured)
⚠️ 5 dropped before placement — 3 no order path · 2 insufficient cash
⚠️ *Crypto*: 2 signal(s) fired, **0 placed** — 2 insufficient cash
```
**The mechanism.** `cash_capped_notional` (line ~791) sizes against a DIFFERENT field per asset class, correctly,
because Alpaca crypto is cash-only and cannot use margin:
```python
field = "non_marginable_buying_power" if is_crypto else "buying_power"
avail = float(account.get(field, 0) or 0) * 0.95
if avail < MIN_ORDER_USD:   # $25
    return 0.0              # -> caller logs "insufficient available cash"
```
So: equity desks buy **marginable** equities → cash goes negative and
`non_marginable_buying_power` goes to ~0 **by construction** → crypto's `avail` falls under $25 → every crypto order
is sized to 0 and skipped. Equities meanwhile keep trading happily on $49.8k of buying power.

**Crypto is therefore starved whenever the equity book is levered, independent of signal quality.** Today 2 crypto
signals passed the confidence gate and still could not fill. That is a materially better explanation of the crypto
desk's idleness than the confidence-ceiling story alone, which I have been leaning on since 08-03.

**~~Stated as inference~~ NOW DIRECTLY MEASURED (2026-08-04 19:41).** The field is reported since #1407, and the
last in-hours run of the day, `30943960438`, reads it out:
```
equity=$21,964.00  cash=$-29,912.20  buying_power=$6,010.55  crypto_bp=$0.00  regime=bull/calm
⚠️ *Crypto*: 2 signal(s) fired, **0 placed** — 2 insufficient cash
```
`crypto_bp=$0.00` — exactly zero, no longer deduced. The interaction also visibly TIGHTENS through a session: cash
went from −$2,554.56 at 16:45 to −$29,912.20 at 19:41 as the equity desks levered further, while equities still had
$6,010.55 of margin buying power to trade on and crypto had none. Equities placed 12 orders in both runs; crypto
placed zero in both, with signals that had already cleared the 0.60 gate.

**Nothing here is malfunctioning.** `recover_negative_cash` correctly declines to flatten (`bp > 0` → "MARGIN DEBIT,
not orphaned notional"), and that restraint is load-bearing: the 2026-07-27 incident recorded in that function shows
flattening a levered book realised losses, tripped the daily loss cap, and froze all trading until session rollover.

**[USER] The fix is an allocation policy, not a patch.** `CLAUDE.md` specifies risk buckets (70% arbitrage / 30%
ML-directional) but nothing reserves NON-MARGINABLE cash for the only always-open desk. Options: hold a cash floor
(e.g. keep `non_marginable_buying_power` above some multiple of MIN_ORDER_USD before equity desks size up), or cap
aggregate equity margin usage, or accept that crypto trades only when the equity book is flat. This is a
capital-allocation decision and is deliberately not made here.

### ⏰ MEASURED 2026-08-03 23:40 — the platform trades ~27% of the clock, not 24/7
Today's full day, from run logs:
- **Inside US market hours (13:30-20:00 UTC):** 11 runs, **7-9 orders each**. Working well.
- **Outside them:** run `30860193311` (22:51) logged `[stage] ✓ Place orders — 0.01s  orders_placed=0`.
  The **0.01s** is the tell — nothing reached the placement stage at all, so this is not the loss cap and not a
  broker error. No signal cleared its gate.

Cause is already documented above: Crypto is the ONLY `always_open=True` desk, its only strategy still producing
signals is `crypto_adaptive_trend`, and that strategy's confidence ceiling is `0.40/(2·rv_21)` — **0.40 at 50% vol,
0.26 at 65%** — against a 0.60 order gate and a 0.45 exploration floor. So outside RTH the whole fleet is
structurally idle. US RTH is 6.5h of 24, so **~73% of every day has zero trading capability**, and 100% of
Saturday and Sunday.

This reframes the crypto recalibration from a nice-to-have into the single change that would unlock two thirds of
the clock. It is still an operator decision — the naive fix trades noise (see the xfail tests in
`backend/tests/unit/test_desk_confidence_gate_is_reachable.py`) and a real one needs a walk-forward backtest — but
the value of making it is now measured rather than assumed.

### ⚠️ CORRECTION 2026-08-04 00:45 — the Vercel block is a TIME lottery, and the status is never refreshed
The 18:40 note below said backend-only PRs are exempt because the ignore step skips them. **That is wrong.** Two PRs,
same `automerge` label, 22 minutes apart:
```
#1376  mine, docs-only, 23:39  ->  failure  "Deployment rate limited — retry in 24 hours."
#1377  improver, 5 commits, 00:11  ->  success  "Canceled by Ignored Build Step"   -> merged normally
```
My docs-only PR was rate-limited while a 5-commit improver PR sailed through the ignore step. So the discriminator is
**not** what the PR touches — it is whether the account was over its rolling 100/day cap at the instant Vercel stamped
the sha. When over, Vercel rejects before the ignore step can even run.

**The consequence that matters: a commit status is immutable per sha.** Once a PR is stamped `failure`, nothing
re-posts it — Vercel only publishes a new status on a new deployment trigger, i.e. a new commit. So every PR stamped
during a saturated window is blocked **permanently**, long after the cap frees. They will never self-clear; they need a
push (or a manual redeploy) to get a fresh status.

That is why the backlog sits at 100 while individual PRs still merge: new PRs stamped during a free window go through,
old ones stamped `failure` are frozen forever. Fixing the ignoreCommand stops NEW PRs being stamped, but does nothing
for the ~100 already carrying a stale `failure`.

### 🔴 The merge gate is fine; VERCEL is blocking all 100 open PRs (found 14:37)
`auto-merge.yml` now fires on both events — 3 `schedule` + 6 `workflow_dispatch`
runs today, all `success`, all sweeping every open PR. **The backlog still did not
move; it grew to 100.** The cause is not the triggers, the label rule, or CI.

`#1358`: `automerge` label, not a draft, base `main`, all three REQUIRED_CHECKS
green (`test`, `test-agents`, `frontend-build`), `mergeable_state: "unstable"`.
Its combined commit status:
```json
{"state": "failure", "context": "Vercel",
 "description": "Deployment rate limited — retry in 24 hours."}
```
The gate ends with `if (combined.state === 'failure' || combined.state === 'error') continue;`
so **every bot PR is refused because a preview deploy could not run.** Closed loop:
bot PR → Vercel preview → free-tier cap (100/day) exhausted → `failure` status →
gate refuses → PRs pile up → more previews attempted.

The gate is internally inconsistent about this: it already ignores Vercel for
*check-runs* (`IGNORE = [..., 'Vercel Preview Comments']`, "Vercel comments are
decorative. Neither gates correctness — only real CI jobs do") and then blocks on
the Vercel *commit status*. Excluding the `Vercel` context from the combined-status
check is consistent and safe — `frontend-build` is REQUIRED and is the real
frontend gate.

**That one-liner is deliberately NOT shipped.** It would auto-merge ~50 stale PRs
in one sweep, which is the operator's open decision, and several are known-defective
(`#1246`: false invariant in a SHARED test, a `position[idx-1]` on a `DatetimeIndex`
that cannot run, an audit log that silently drops records). Cheaper fix is on the
Vercel side: disable previews for `improver/**`.

### ✅ VERIFIED 2026-08-03 10:16 — the pacemaker dispatch works
First-ever `workflow_dispatch` run on `desk-trading.yml`: run `30804944671`,
triggered by `github-actions[bot]`, `desks_run=9`, conclusion `success`. So
`cancel-in-progress: false` let the sleeper reach its dispatch, and the dispatch
returned 204. `orders_placed=0` is CORRECT for that run — 10:21 UTC is pre-market
(US RTH = 13:30-20:00 UTC). Note there were **no `schedule` runs at all** on
Monday between 09:00 and 10:16 despite cron `*/15 9-22 * * 1-5`, which is the
starvation this dispatch exists to route around.

### 🔴 The Polymarket desk has NO execution path (found 2026-08-03 10:30)
Run `30804944671` logged three `poly_cross_market_hedge` signals at **conf=1.00**
on `PM:` symbols, all dropped as "(Polymarket closed)".

**Do not "fix" this by setting `always_open=True`.** That was the instinct and it
is wrong for the second time in two ticks (cf. the crypto-24x7 gate below).
`desk_order_placer.py` references `brokers` **zero times** — every order is a POST
to Alpaca `/v2/orders` (line 1101). `backend/app/brokers/polymarket.py` exists but
the desk script never imports it. Flipping the flag would send
`PM:Will Tucker Carlson win the 2028 Republi…` to Alpaca and fail.

The clock gate is accidentally masking a deeper gap: **the Polymarket desk is
decorative in the desk runner** — 8 strategies, conf up to 1.00, no route to a
venue. Fixing it means wiring `brokers/polymarket.py` into the desk (or removing
the desk), not touching the market-hours flag. Not attempted; needs a decision.

**Do NOT dispatch `desk-trading-crypto-24x7.yml` from the pacemaker.** The first
version of that step dispatched both desk workflows and would have reintroduced a
measured bug. `desk-trading.yml` already runs ALL NINE desks and crypto is
`always_open=True`, so one dispatch covers 24/7. The crypto-only workflow gates
its job on `schedule || workflow_dispatch` precisely to CEDE shared triggers: the
two use different concurrency groups, so on a shared trigger they run in PARALLEL
and race for Alpaca's free-tier data. Measured over 60 runs (2026-07-28): 22
collided; one pair on sha 49e46ded had desk-trading fetch 70 bars while the
crypto-only run got 5 and 429s. Pinned by
`test_the_crypto_desk_is_not_dispatched_too`.

**Crypto desk — right time, unreachable gate.** `always_open`, so the clock is
never its issue. `confidence = |raw_signal| · (target_vol/rv_21) / 2` is a
*position size*, not a conviction, and falls as vol rises. Ceiling is
`0.40/(2·rv_21)`: **0.40 at 50% vol, 0.26 at 65%** — under the 0.60 order gate AND
the 0.45 exploration floor. Run `30782697088`: 16 signals, confidences 0.23–0.44,
`passed=0 filtered=16 explored=0`. Best signal 0.44 against a 0.45 floor.
Also: 16 of 16 signals came from `crypto_adaptive_trend` alone — the desk's other
13 strategies die on CoinGecko 429 / Binance-OI HTTP 451 from US runners.

**NOT a bug:** desk-trading's Fri→Mon gap is `cron * * 1-5` correctly skipping the
weekend. Checked before reporting.

**Fixed in this pass (mechanical only):** pacemaker `cancel-in-progress: false`;
pacemaker now dispatches both desks directly via `workflow_dispatch` (the proven
recursion-guard exception, independent of cron and of the dead workflow_run path);
`test_pacemaker_actually_delivers.py` (12 tests, mutation-checked).

**NOT fixed, deliberately — operator decision needed:** the crypto recalibration.
The obvious fix (`confidence = |raw_signal|`) was tried and **reverted**: it scores
0.83/0.92/0.94 on a *zero-drift* random walk because `tanh(composite_raw*5)`
saturates, i.e. it trades noise. A real fix is a backtested strategy change under
the repo's walk-forward standard. Recorded as `xfail(strict=True)` in
`backend/tests/unit/test_desk_confidence_gate_is_reachable.py` so it reports XPASS
when someone fixes it. Related: `analyze()` (tanh on raw returns) and
`backtest_signals()` (`.rank(pct=True)` percentiles) are **different functions** —
backtests of this strategy do not describe live behaviour.

**Discord is fine, and had been reporting all of this.** `#pnl-daily` carries
per-desk funnel telemetry (`⚠️ *Commodities*: 3 signal(s) fired, 0 placed — 3
market closed`) every run. Per-desk channels are silent only because
`_post_chat(desk.chat_channel, …)` sits inside `if desk_order_list:`.

## 🛑 SUPABASE IS **PAUSED** — read this before the 07-25 section below, which is now WRONG
**Measured 2026-07-29 14:00 via Supabase MCP `list_projects`:**
```
ref: vexzwnfbmznvxoxxktax   name: Trade   region: us-west-1   status: "INACTIVE"
```
`INACTIVE` = the free-tier project has **auto-paused**. And `/health/detailed` has reverted to
```
(ENOTFOUND) tenant/user postgres.vexzwnfbmznvxoxxktax not found
```
after reading `password authentication failed` for the whole of 2026-07-28.

### Why the section below is wrong, and how it misleads
It concludes *"Fault #1 is closed. Fault #2 (the password) is the single remaining blocker"*, and
treats `tenant not found` as **proof of the wrong cluster**. That inference was reasonable when
the project was running, but it is **not the only cause of that message**: a *paused* project has
no tenant on ANY pooler, so it returns exactly the same error from the correct host. The
"aws-0 vs aws-1" table below cannot distinguish "wrong cluster" from "project paused" — it was
taken while the project was still awake.

**Do NOT start with the password reset.** Resetting the credential on a paused project changes
nothing, and the reset itself may not apply until the project is restored.

### The two faults COMPOUND, which is why this got stuck
auth failure → no successful connections → Supabase counts the project as inactive → auto-pause →
now the tenant does not resolve at all. The `DB Keep-Alive` workflow exists to prevent exactly
this, but it cannot ping a database it was never able to authenticate to.

### ACTION (user — I deliberately did not do this)
1. **Unpause/restore** project `vexzwnfbmznvxoxxktax` in the Supabase dashboard (or tell me to
   call `restore_project`; I held off because direct Supabase actions were previously declined
   and restoring is a bigger step than the SQL that was refused).
2. **Then** reset the DB password and update Render's `DATABASE_URL` exactly as the 07-25 section
   describes — that part is still correct, and the host/port there is still the right guess.
3. **Verify:** `/health/detailed` → `database.fallback` gone, `database_primary.ok` true.
   `database_primary` is a BOOT-time value, so it only changes after a restart.

### What this one thing is costing, measured today
Everything durable is downstream of it: ephemeral sqlite → **trade history wiped on every
redeploy** (the 13:12 desk run logged `✓ Performance weights active for 11 strategies`; after the
13:48 deploy `/api/v1/trades/` and `/leaderboard/live` both returned **0**) → attribution-weight
pruning (`✂ pruned by attribution`) goes **inert**, leaving the file-based trimmer as the only
working pruning path. Positions survive only because `/api/v1/positions/` falls back to live
Alpaca.

⚠️ Supabase MCP was disconnected at the 14:37 tick, so `INACTIVE` could not be re-confirmed then;
the unchanged `tenant/user … not found` error corroborates it.

## ⚠️ SUPERSEDED — DURABLE POSTGRES — CLUSTER FIX 2026-07-25 07:36 UTC (see the PAUSED section above)
**CONFIRMED IN PRODUCTION 07:36 UTC:** `/health/detailed` now reports
`password authentication failed for user "postgres"` — it previously said
`(ENOTFOUND) tenant/user ... not found`. That change is hard proof the cluster fix reached the
live boot: the backend now REACHES the correct `aws-1` Supavisor, which recognises the tenant
and rejects only the credential. **Fault #1 is closed. Fault #2 (the password) is the single
remaining blocker and needs the user.**

The pooler prober ran (via the `workflow_run` chain off Agent Heartbeat) and tested all four
cluster/port combinations. Verdict is unambiguous — there were **TWO** faults stacked:
```
aws-0-us-west-1...:6543  -> tenant/user postgres.<ref> not found   [no_tenant]   <- URL pointed here
aws-1-us-west-1...:6543  -> password authentication failed          [bad_password] <- tenant LIVES here
aws-1-us-west-1...:5432  -> password authentication failed          [bad_password]
aws-0-us-west-1...:5432  -> tenant/user not found                   [no_tenant]
```
1. **WRONG SUPAVISOR CLUSTER (fixed autonomously).** The URL used `aws-0`; this project's tenant
   lives on `aws-1`. The prober patched Render `DATABASE_URL` to `aws-1-us-west-1...:6543` and
   triggered a redeploy. This is why the error was "tenant not found" for weeks — that message
   means "wrong cluster", NOT "paused project" and NOT "wrong region".
2. **STALE DB PASSWORD (needs the user — cannot be automated).** `aws-1` RECOGNISES the tenant
   and rejects the credential. An auth failure is a POSITIVE identification of the right cluster;
   no host change can fix it.
   ⚠️ **STILL TRUE AS AN OBSERVATION, BUT NO LONGER THE FIRST STEP** — the project has since
   auto-paused, so the tenant no longer resolves at all. Unpause first (see the top section).
   **ACTION (~2 min): Supabase dashboard → project `vexzwnfbmznvxoxxktax` → Settings → Database →
   Reset database password → copy it → Render → quantedge-api → Environment → set `DATABASE_URL`
   to `postgresql+asyncpg://postgres.vexzwnfbmznvxoxxktax:<NEW_PASSWORD>@aws-1-us-west-1.pooler.supabase.com:6543/postgres`
   (URL-encode any special characters) → Save (auto-redeploys).**
Once that lands, the boot reaches Postgres → `alembic upgrade head` provisions the 22-table schema
(incl. catch-up `k6f7a8b9c0d1`) → durable state, and bot P&L stops resetting on every deploy.
**Verify:** `/health/detailed` → `database.fallback` gone, `database_primary.ok` true.
NOTE `database_primary` is a BOOT-time value, so it only changes after a restart.

## 🚨 2026-07-27 — NOTHING ENFORCED A STOP-LOSS (P0, fixed)
Second pass of the principal review. After the risk gate I swept `app/` for the *shape* of
that bug — public functions nothing references — since finding it twice by hand meant more.
- **`PositionMonitor` was never started.** `start_position_monitor()` claims scheduler.py
  calls it; scheduler.py has no such job and nothing constructs a PositionMonitor. Third
  instance of the same lying-docstring pattern. Meanwhile `strategy_runner.py:308` writes
  `pos_exit:<symbol>` (stop_loss, take_profit, peak_price) on every fill "for
  position_monitor.py". **The producer ran for months; the consumer did not exist.** The whole
  `CompositeExit` engine was reachable only from that dead module. Now a supervised 30s loop.
- **Every Redis price read used a key nothing writes.** `set_price()` writes
  `price:<exchange>:<symbol>`; all three readers built `prices:<symbol>` — the **WebSocket
  topic**, not a Redis key. A miss looks exactly like a cold cache, so each silently took its
  fallback. Worst: the **live** `bot_exit_checker` fell through to a yfinance **daily** bar,
  so intraday stop-losses were priced off a daily close. Root cause was
  `tasks/CLAUDE.md` listing the WS topic in its "Redis Key Schema" table — the code matched
  the docs, not reality. Added `redis_client.price_key()`; fixed the table.

**With the inert risk manager, that is the full account of the −$8,287.81 paper account:**
nothing capped size, nothing halted on drawdown, nothing enforced a stop, and the one exit
job that did run priced stops off yesterday's close.

Pinned by `backend/tests/unit/test_exit_path_wiring.py` (8 tests; 5 fail on pre-fix code).
**Still unwired, confirmed repo-wide** — with one correction to my own first draft. I wrote
that root CLAUDE.md principle #5 ("Walk-forward only") is "enforced nowhere". **Wrong.**
There are TWO walk-forward implementations and I had found the dead one:
- `app/backtest/walk_forward.walk_forward()` — **live**, called from `api/v1/backtests.py`
  and `.github/scripts/ml_experiment.py`, with a `deflated_sharpe_ratio` overfit gate.
- `app/ml/training/walk_forward.walk_forward_validate()` (torch) — dead.
So the principle IS enforced for strategy backtests, and is NOT for ML model training.

Was dead, each referenced only by its own test: `probability_of_backtest_overfitting`,
`probabilistic_sharpe_ratio`, `monte_carlo_simulation`. `run_stress_tests` has no reference
at all.

**PSR + Monte Carlo now run on every walk-forward — REPORTED, NOT GATED** (`is_robust`
untouched; changing the promotion bar is a risk decision, not a wiring fix). Two bugs found
while wiring: (1) my first draft passed an **annualised** Sharpe to PSR, whose contract needs
the same frequency as the moments — that inflates it ~16x and wrecks the z-score;
(2) **`p95_max_dd` is the LUCKY tail** — `max_dd = dd.min()` so drawdowns are negative and the
95th percentile is the *mildest* drawdown, so quoting it as risk understates it by
construction. Added `p5_max_dd` and report that instead. Still not wired, deliberately:
`probability_of_backtest_overfitting` needs N *configs* (walk_forward runs one config across
windows — wrong call site), and `run_stress_tests` needs price history spanning the crisis
windows (mostly `period_covered=False` on short backtests).

**`configure_logging()` — fixed.** It had no caller, so structlog ran on library defaults.
Verified via `structlog.get_config()` at runtime: renderer was `ConsoleRenderer` (not
`JSONRenderer`, so Render got unparseable text) and wrapper_class was
`BoundLoggerFilteringAtNotset`, which filters **nothing** — all **105** `logger.debug()` sites
in `app/` emitted every run. That is the noise floor under "a bunch of errors are reported
throughout day on discord". Now called at import in `main.py` (not in `lifespan`: module-level
code logs before startup, and `static_server.py` imports `app.main`, so that covers every
entrypoint).

**ML feature builders — resolved, not by wiring all three.** They were not equivalent:
`add_microstructure_features` is pure OHLCV arithmetic and is now **wired** (verified
no-lookahead: label is `close.pct_change(h).shift(-h)`, so bar t predicts t+h and bar t's own
OHLC is known); `add_alternative_features` calls the **Binance API** and is deliberately NOT
wired — backtests/training must be deterministic and offline, pinned by
`test_feature_engineering_makes_no_network_calls`; `add_sentiment_features` needs a
caller-supplied history and emits constants without one, so it needs a data-source decision.
Widening `FEATURE_COLS` was safe *only because* there are zero trained models (no artifacts on
disk, live `ml_models.count: 0`) — after the first model exists it would silently break
inference.

## 🚨 2026-07-27 — THE RISK ENGINE WAS NEVER SWITCHED ON (P0, fixed)
First pass of the principal-engineer review, following the money path (risk → execution →
broker). **`RiskManager.check_order()` had never executed in production.** Every documented
risk control — position cap, drawdown breaker, correlation cluster limit — was inert:
- `orders.py` skips the gate when `app.state.risk_manager` is None. **Nothing ever assigned it.**
- `main.py` built the strategy runner with `risk_manager=None` literally.
- The only `RiskManager()` in the codebase sat in `strategy_runner.start_strategy_runner()`,
  a function whose docstring claims main.py registers it. **main.py never called it** — one
  textual occurrence in `app/`, its own `def`. Deleted (95 dead lines).

`test_security_invariants.py` passed throughout: it asserts the *string* `"check_order"`
appears in orders.py twice. **A textual invariant cannot tell live wiring from dead code.**

Also fixed in the same gate: the size cap `return`ed before the correlation check (so the
largest orders were the only ones skipping the concentration limit); market orders were sized
against a hardcoded `$100` (1 BTC read as a $100 position, SHIB 10-million-fold too large);
and the new equity sync would have failed OPEN on negative equity — `update_equity()` raises
on negatives and the live Alpaca account is at **−$8,287.81**, which would have left the
manager on its seeded $100k approving orders forever. Now clamped to 0 so the halt fires.

**This answers the open question from the desk review** — "a paper account going $8k negative
means the risk layer did not stop it." It did not stop it because it was never running.

The correlation cluster limit needed a second fix one level down: `_clusters` is only
written by `update_returns()`, which **also had no caller**, so wiring the manager in left
that control inert. Now derived from the mark stream — downsampled (2-second ticks are
microstructure noise, not co-movement) and throttled (clustering is O(symbols²) behind a
live feed). All four controls are now live.

Pinned by `backend/tests/unit/test_risk_gate_wiring.py` (23 tests). Verified against the
pre-fix tree: 9 of 11 originals fail on old code, and disabling the cluster refresh fails
the two correlation tests. Two of the tests were themselves rewritten after failing that bar.

**VaR gate now wired — 4 of the 5 diagrammed gates run.** "Block if 1-day 99% VaR > 2% of
NAV". Two traps, both pinned by tests: (1) `historical_var()` returns a *default*
`var_99=0.03` below 10 observations, so a naive wiring would have blocked EVERY order at cold
start — it checks the count AND the `method` sentinel and returns "no opinion" instead;
(2) **units** — equity is polled every 60s, so raw returns give a 1-MINUTE VaR, ~30x too small
against a 1-day limit. Equity is downsampled (default hourly) and scaled by
square-root-of-time, which assumes i.i.d. returns and understates tail risk under positive
autocorrelation — a floor, not a ceiling.

## ✅ 2026-07-28 — "ZERO trades for 18 days": wrong premise, and the loss cap is behaving
Verified the fixes landed: **scanner fix confirmed LIVE** (`/api/v1/scanners/polymarket` → 200,
`score: 0.75`, `side: "buy"`; was 500). Then checked trade flow — **9 trades total, most recent
2026-07-10**. Those 9 are real (desk_trade_sync backfills Alpaca's 30-day history), not seed.
Cause, from the 01:05 UTC crypto desk run — NOT insufficient balance:
```
🛑 Loss cap ACTIVE — only risk-reducing orders allowed (0 open positions eligible to reduce)
🛑 .../UNI/USD BUY — blocked by loss cap (would increase exposure)   [x3]
Done. 0 orders placed across 9 desks.
```
Desks are HEALTHY — data, ensembles, `passed=3 filtered=0`. Under the cap only risk-REDUCING
orders pass, and with 0 open positions nothing can be reducing, so nothing passes. Correct as a
daily cooling-off rule; pathological at 18 days. I could not tell which and did not guess —
instead the Discord summary now carries `equity $X vs prior close $Y (-Z%, cap 2%) · N
position(s) eligible to reduce` + `⚠️ nothing can pass while this is 0`, so the next run answers
it in-channel. 8 tests.
**Follow-up 04:30:** the Discord-only numbers were unreadable from where I actually investigate
(Actions API, not Discord), so the CONSOLE banner now repeats them at the point of blocking —
a 20-line tail suffices. Also added a **contradiction detector**: with 0 positions and no fills
equity cannot move, so `equity == last_equity` while the cap is ACTIVE means the cap is firing
on **stale inputs**, not a real loss. The run now says so explicitly. Next desk run answers
"correct daily cap vs stale last_equity" without inference.
Also noted: nearly every crypto symbol logs `sell/buy conflict — stand aside` — only 3 signals
survived 9 desks. Separate ceiling on trade count, worth its own investigation.

### ✅ 05:45 — ANSWERED, and the premise was wrong on both counts
The instrumentation paid off. Pulled the full job logs for three consecutive desk runs:
```
17:49 Jul 27  equity $21,819.46  cash $-10,829.50  bp $25,001.91  → 13 ORDERS, +$9,634 notional
23:44 Jul 27  equity $21,745.65  cash $21,745.65   bp $86,982.60  → flat, cap ACTIVE, 0 orders
04:46 Jul 28  identical to the cent                               → still capped
```
**1. The desks ARE trading.** 13 orders on Jul 27. The "9 trades, nothing since 2026-07-10"
reading came from the **backend DB, which is on its sqlite fallback** and never sees orders the
Actions desks place directly at the broker. The DB is not a record of trading right now — it is
a record of what `desk_trade_sync` managed to backfill before Postgres went down. Do not read
trade flow from it until Postgres is back.

**2. The cap is CORRECT — the contradiction detector stayed silent, as it should.** `cash ==
equity` exactly ⇒ genuinely zero positions; $21,745.65 vs a $22,253.58 prior close is a real
**-2.28% day** against a 2% cap, booked while the account still held the positions it opened at
17:49. Not stale inputs. The freeze lifts when Alpaca rolls `last_equity` — which happens at
the **next session open, not at the closing bell**, which is why three runs spanning six hours
all read the same prior close and looked stale.

**3. A real bug, and it was mine.** The diagnostic shipped in #1121 computed the drawdown
*magnitude* (`1 - equity/last_equity`, positive on a loss) and formatted it `{:+.2%}`, so it
reported the -2.28% day as **"+2.28%"** — asserting the opposite of what happened, at both the
console and Discord sites. Read as a gain it makes a correctly-firing cap look broken, which is
precisely the wrong conclusion to hand the next investigation. Now a signed return
(`equity/last_equity - 1`) rendering `(-2.28%, cap -2%)`. 15 tests, incl. a generalised
"no loss is ever reported with a plus sign" property.

**Standing caveat:** the cap keys off the *equity* trading-day rollover, but crypto trades 24/7
— a bad equity day freezes the crypto desks for the ~17.5h until the next open. Same family as
the 2026-07-20 weekend-drift note in the source. Design question, not a bug; not addressed here.

### ❓ 05:56 — WHO closed the book? Nothing could answer, so now it can (#1123)
The 13 orders were flat within 6h, realising the loss that tripped the cap. **Nothing in the
repo could say what closed them**, and I checked every candidate rather than guessing:
- orders carry **no bracket/OCO legs** (plain limit-first, see the placement loop) → not Alpaca
- `recover_negative_cash` **never fired** — no 🚑 line in any run's log. Note its guard is
  `cash < 0 AND non_marginable_buying_power <= 0`; at 17:49 cash was −$10,829.50 (an ordinary
  **margin debit**, bp still $25,001.91), so it correctly declined
- no other Actions script closes positions — `live_trading_reporter` / `research_to_trade` only
  `GET /v2/positions`
- the backend DB returned **zero order rows** (sqlite fallback) → no record there either

The one witness that survives all of that is **`client_order_id`, which lives at the broker**.
Everything this repo places is tagged `qe-<strategy>-<sym>-<ts>`; anything else is external.
Prime suspect is the backend's **`PositionMonitor` exit loop** — wired up 2026-07-27, and if it
is the closer then it is doing exactly its job and this was a normal stop-out, not a fault.
**Unconfirmed — I could not verify it without Alpaca creds or Render logs, and did not assert
it.** A flat book under an active cap now prints the last 8 closed orders with each one's
origin, so the next occurrence names the closer. 11 tests, incl. that a lookalike prefix
(`qexit-`, `QE-`) is NOT claimed as ours — misattribution would send the next investigation
back to working code.

**Reusable lesson:** when a question survives a session, ship the instrument rather than the
inference. That is now 2-for-2 (the cap diagnostic answered its question in one run).

## 🔓 2026-07-28 21:15 — THE SECURITY GATE HAS NEVER INSPECTED A SINGLE PR (fixed)
Found while checking a PR was green before merging. Two independent defects, **either one
enough on its own** to make the gate decorative:

**1. It could not run.** `security-scan.yml` triggered only on `pull_request`. Agent branches
get their PR opened by `auto-pr.yml`, so the run's actor is `github-actions[bot]` — and a
bot-actored PR run lands in `action_required`, waiting on a human approval that never arrives.
Of the last 30 PR runs, **21 were `action_required` and every one of them was bot-actored**:
```
20:51 action_required  actor: github-actions[bot]  claude/stoic-johnson-7z4wtz
20:43 action_required  actor: github-actions[bot]  improver/run-30397450082
15:57 success          actor: bahllaavanye-afk     claude/stoic-johnson-7z4wtz
```
The correlation is exact. **Every `improver/*` PR has been in this state since 2026-07-27** —
that is the mechanism behind "the improver PRs bypass CI", now measured rather than assumed.

**2. When it did run, it scanned the wrong tree.** The checkout was pinned `ref: main`
*unconditionally*, so a PR run analysed main instead of the code being proposed. The five
`success` runs on my branch therefore proved nothing about my branch — bandit and the
red-team probe never saw the diff.

Fixed: a `push` trigger (actored by the pusher, no approval needed, fires before the PR
exists; main excluded — 27 of its 129 commits in the last 24h touch the scanned paths and the
weekly cron already covers merged code), and the checkout now resolves to the triggering
commit while scheduled runs still pin main. 10 tests, 7 fail pre-fix.

**Reusable lesson, and this is the third time this exact shape has bitten:** a gate that is
permanently *not-red* is indistinguishable from a gate that passed. `ModuleNotFoundError:
sqlalchemy` made it red-and-ignored; `action_required` made it grey-and-ignored; `ref: main`
made it green-and-meaningless. **Before trusting any check, confirm it EXECUTED and confirm
what it executed AGAINST.** A green tick is not evidence until both are known.

## 🔁 2026-07-28 21:00 — `MKR/USD is not active` CAME BACK, and the code could not say why
I recorded this class as closed at 17:30 ("0 occurrences, `_filter_tradable_crypto` handles
it"). It reappeared in run `30394875861` — **five times in one process**:
```
⚠ alpaca POST /v2/orders → 422: {"code":40010001,"message":"asset MKR/USD is not active"}
```
The run printed **no `ⓘ skipping` line**, so the filter dropped nothing, and there was no way
to distinguish two very different situations:

- **A.** the `/v2/assets` lookup failed, `_tradable_crypto_symbols()` returned `None`, and the
  filter fail-softly kept the whole universe — correct behaviour, completely invisible; or
- **B.** Alpaca's asset metadata says MKR/USD is active while its own order engine refuses it,
  in which case no amount of pre-filtering will ever help.

Same **silent-miss family** as `prices:{symbol}`: a fail-soft path that returns `None` quietly
is indistinguishable from a path that had nothing to do. `_tradable_crypto_symbols()` now
narrates its own failure (`ⓘ tradable-crypto lookup FAILED (<reason>)`, and a separate line for
a non-list response shape), so **the next live run names the branch**. The fail-soft contract
itself is unchanged — a blip still must never shrink the universe.

Independently of which branch it is: the first 422 is a definitive answer. Re-submitting the
same asset for four more desks is a guaranteed-wasted round trip, and it makes one problem look
like four in the log. Rejections carrying "not active"/"not tradable" are now remembered for the
rest of the process and skipped with a named line. Deliberately **not** persisted across runs —
a delisting can be reversed, and process-lifetime memory self-heals on the next scheduled run
with nobody clearing state. The blacklist is narrow on purpose: fractional-short, buying-power
and rate-limit 422s are retryable or side-specific and must never strand a healthy symbol
(tested). 24 tests, 13 fail on the pre-fix tree — the headline one **behaviourally**: pre-fix
`_place_order` returns a live order for an asset already refused this run.

**⚠️ My 17:30 "confirmed fixed" was premature.** I read one clean run as proof. One run without
a symptom is not the same as the symptom being gone — the standing rule ("verify on the NEXT
LIVE RUN") needs the corollary that a *recurring* fault needs *several* clean runs, and that
absence of evidence in a log that cannot report the relevant state is not evidence at all.

**Next tick:** grep a fresh desk run for `MKR/USD is not active`, for `ⓘ tradable-crypto lookup
FAILED`, and for `marked INACTIVE`. That triple settles A-vs-B and confirms the de-duplication.

### ✅ 21:36 — VERIFIED LIVE (run `30401244361`), and the answer is "not A"
```
► vol_of_vol_timing/MKR/USD signal=BUY conf=0.88 — placing $197 limit-first order
  ⚠ alpaca POST /v2/orders → 422: {"code":40010001,"message":"asset MKR/USD is not active"}
  ⓘ MKR/USD marked INACTIVE for the rest of this run — later desks will skip it instead of re-submitting
```
The memory fired on its first live encounter. And the run carried **no** `ⓘ tradable-crypto
lookup FAILED` line, so **branch A is eliminated — the lookup succeeded.**

**⚠️ But I could not yet claim branch B, and nearly did.** `_filter_tradable_crypto` has a
*second* silent bail-out: if the returned set lacks BTC/USD and ETH/USD it assumes a format
mismatch and declines to filter — returning with no log line at all. So "lookup succeeded +
nothing dropped" still had two readings:
- **B1** MKR/USD genuinely is in Alpaca's active-asset list and its order engine contradicts
  its own metadata — in which case *no* pre-filter can ever catch this and the run-scoped
  memory is the only defence;
- **B2** the set came back in another format (`MKRUSD`), the guard tripped, and the filter
  no-opped invisibly.

Both remaining bail-outs (format mismatch, and the never-return-an-empty-universe guard) now
print themselves, with the set size and a sample. **Next run settles B1 vs B2** — and the fix
differs completely between them, so it is worth the extra tick rather than guessing.

**The lesson keeps repeating in the same shape:** I closed this once on one clean run, then
closed "branch A vs B" on one instrumented run. Each time the silent path was one level deeper.
Instrument *every* early return in a fail-soft function at once, not the one you happen to
suspect.

**Cost while unresolved:** MKR/USD burns a top-K slot every run — this run it was 1 of only 3
signals that passed, so a third of the desk's capacity went to a guaranteed reject.

## 🫀 2026-08-03 05:40 — the merge gate now rides the pacemaker, not just cron

The `schedule` added at 02:40 has produced **zero runs in 2h47m**. That is the same starvation the pacemaker exists to route around — its own header says *"GitHub starves free-tier schedules under load"* — so relying on cron for the merge sweep was betting on the one mechanism this repo has already measured as unreliable.

- [x] **[P0] The pacemaker now dispatches `auto-merge.yml` alongside its CI heartbeat.** ~50-minute cadence, no cron dependency. Kept **both** mechanisms deliberately: they appear distinguishably in the run log (`event=schedule` vs `event=workflow_dispatch`), so whichever actually delivers can be identified instead of guessed at — which matters given I have now mis-attributed this mechanism once already.
- [x] **Failure is visible but non-fatal.** Unlike the CI dispatch above it, the merge dispatch does **not** `exit 1`: losing the sweep must not kill the heartbeat driving 36 downstream workflows. It still emits `::error::`, because a silent skip here would be the same class as the permanent 403 that hid in `continuous-improvement.yml` for its whole lifetime. Both properties are pinned by tests.
- [ ] **[P1] Still unverified.** No `event=schedule` run yet, and the pacemaker dispatch has not had a cycle. Do not record either as working until an auto-merge run with the corresponding event actually appears and the green backlog moves.

## ⚖️ 2026-08-03 04:40 — CORRECTION to the 02:40 entry: the mechanism was wrong, the fix is right
I wrote that **every** trigger on `auto-merge.yml` is suppressed for bot PRs. Not true as stated.

`auto-merge` **did** fire at 03:38 (`30782308923`, `event=pull_request_target`). Its entire output:
```
#232: base claude/advanced-trading-bot-d5Lmw != main
```
One PR evaluated, skipped, done. Because a `pull_request_target` event populates
`context.payload.pull_request`, `candidates` gets exactly one entry, and the
`candidates.size === 0` fallback — the branch that scans all open PRs — never runs.

**So the accurate statement is:** the gate does wake, but only ever for the single PR whose event
woke it. Nothing sweeps the backlog. The *effect* I described (90 PRs stranded) was right; the
mechanism I gave ("all triggers suppressed") was not, and I asserted it more confidently than the
evidence supported. Third time this session I have named a mechanism before confirming it.

**The shipped fix is still the correct one** — a `schedule` is the only trigger that reaches the
all-open-PRs scan, which is precisely the missing capability. Nothing to revert.

**Not yet confirmed:** as of 04:40, 1h47m after the fix merged, `auto-merge.yml` has **zero** runs
with `event=schedule`. That is inside the measured cron-starvation envelope for this repo
(1h22m–3h12m late), so it is not yet evidence of failure — but it is not evidence of success either.
Do not record the schedule as working until a run with `event=schedule` actually appears.

**Open observation, not diagnosed:** `#1341` was green WITH the `automerge` label last tick and now
has **no** label, still unmerged. Something removed it. Worth understanding before concluding
anything about why the backlog is not clearing.

## 💤 2026-08-03 02:40 — THE MERGE GATE HAS NOT FIRED IN THREE DAYS
Both earlier fixes are confirmed working in production. The improver's dispatch step now logs
`CI dispatched on improver/run-30773290001` instead of the 403, and improver PRs get full CI. But
the stage *after* that is dead.

**`auto-merge.yml`'s last run was 2026-07-29 23:45** — on a human-pushed branch. In the three days
since, the improver opened ~90 PRs (now at `#1341`) and the gate never woke once. `#1341` is green
(`test`, `test-agents`, `frontend-build` all success), carries the `automerge` label, is not a
draft, and is unmerged.

**Why:** every trigger it declared is suppressed for exactly the PRs it exists to land. GitHub does
not start workflow runs from events attributed to GITHUB_TOKEN, and every step of the improver's
loop uses it:
```
pull_request_target: labeled    the bot applies `automerge`  -> suppressed
check_suite: completed          checks from the bot's CI     -> suppressed
workflow_run: [CI] completed    CI dispatched by the bot     -> suppressed
```
Every historical auto-merge run is `pull_request_target` or `check_suite`, each traceable to a human
push. `workflow_dispatch` was already declared and nothing ever called it.

**This is the next link in the same chain as the missing `actions: write`.** That fix made CI run;
this one makes something read the result. Fixing one stage keeps exposing the next — worth expecting
rather than being surprised by.

**Fix:** a `schedule` (`17,47 * * * *`) — a heartbeat the gate owns, independent of any bot event.
It is a floor, not a guarantee: free-tier cron is starved here (measured 1h22m–3h12m late), but the
job is idempotent and cheap, so a late sweep still lands everything eligible.

**The schedule only works because of the zero-candidate fallback.** A scheduled run has no event
payload, so candidate collection yields nothing; without the existing `candidates.size === 0` branch
that scans all open PRs, it would succeed having examined zero PRs — a green no-op, this repo's
signature failure. `test_the_schedule_is_not_decorative` pins the two together. Also pinned: the
required-checks list and the `automerge` label gate, both of which matter *more* now that the gate
runs without any CI event to anchor it.

**Not claimed:** that this unblocks all ~90 PRs. `#1337` was checked and its `test` job genuinely
failed — that one is correctly blocked, and an unknown share of the backlog will be too.

## 🎯 2026-07-29 23:40 — the improver was spending half its budget on files it cannot improve
`improve_file()` rejects anything over `MAX_FILE_CHARS` (8000), and `main()` caps the loop at 10
attempts for 5 wanted improvements. `pick_target_file()` did not know about the limit, so it kept
returning files guaranteed to be rejected — 5 of 10 attempts in run `30476849972`.

I deferred this last tick as *"changes which files the improver ever touches"*. **That was wrong.**
Those files were already rejected 100% of the time by the guard, so filtering at selection changes
nothing about which files can be improved; it only stops spending a scarce attempt on a certain
rejection. Shipped.

`_too_large()` filters **both** candidate lists — the hour's pattern and the repo-wide fallback.
Filtering only the first leaks oversized files straight back in through the second, and a test pins
that (`_too_large` must appear ≥2 times in `pick_target_file`).

7 tests, 6 fail against the old code. One runs the real selector over the real tree for all 24
hour-slots to prove the filter has not *starved* selection — a filter that returns `None` would be
worse than the waste it removes. Another asserts the four measured offenders are now excluded.

**Explicitly NOT done, and it belongs to a human:** `backend/tests/unit/*.py` stays in
`CANDIDATE_PATTERNS`. The improver writing false invariants into a shared test (#1246,
`"Consecutive non-zero signals must alternate sign"` — false for any trend-follower) is a real
structural conflict, but `test_cases` is a *configured* improvement type ("Add 2-3 new unit test
cases for edge cases not currently tested"). Removing tests from selection would disable a designed
capability, so the trade is the operator's, not mine.

## ✅ 2026-07-29 22:45 — the dispatch fix works, and it fixed a STALL, not a safety hole
Verified on a live run rather than assumed. Improver run `30496380998` (22:30, head `fa4a99a2`,
first run carrying the `actions: write` fix) dispatched successfully, and **PR #1246 has CI check
runs — the first improver PR ever to get them**, starting 3 seconds after the dispatch.

CI immediately earned its keep: **`test` FAILED** on #1246, so the gate is holding it.

### CORRECTION to the 21:40 entry below
I wrote that the missing permission was "the mechanical cause of the standing *improver PRs bypass
CI so main can silently break* risk". **Half wrong, and the wrong half matters.**

`auto-merge.yml` lines 91–103 **already refuse** to merge a PR whose required checks never ran, with
a comment naming the exact incidents (`#876` strategy-router 400, `#929` PIPELINE_DEFS + boot
crashes). That hole was closed before this session. So the 403 did **not** cause unvalidated merges.

What it actually caused: **a completely stalled improvement pipeline.** 15 improver PRs are open,
back to `#1187` at 2026-07-28 22:02 — roughly 24 hours of autonomous work. Every one has **zero**
check runs, so the gate correctly refused all 15 and they simply rotted. Only `#1246`, the first
post-fix one, has CI.

So the fix unblocks a jam; it does not close a security gap. Worth being precise about — this is the
second correction in two ticks on this same subject, both from claiming a mechanism before reading
the thing that would confirm it.

### What #1246 shows about the improver's output quality
The gate is catching a genuinely bad change. In `backend/tests/unit/test_strategies.py` it rewrote a
**shared** test with assertions that are false as general invariants:
```python
assert signs[i] != signs[i - 1], "Consecutive non-zero signals must alternate sign"
```
A trend-follower holds `+1` across consecutive bars; this fails for essentially any momentum
strategy. Also `assert len(nonzero) >= 1` (a strategy may legitimately produce no signal on random
data) and an `else: pytest.fail(...)` converting a deliberate skip into a hard failure.

Two more from the same PR, not caught by pytest and worth knowing about:
- `ml/features/normalization.py` — `position[idx - 1]` where `idx` comes from a `DatetimeIndex`.
  `Timestamp - 1` is a `TypeError`; the function cannot run.
- `archive/trade_archiver.py` — a new `_validate_signal()` **silently drops** signals with
  confidence < 0.6 from the archive. That file's own docstring says it "writes every order, fill,
  and signal ... for long-term audit and replay". An audit log that discards records is worse than
  no filter.

**Left for a human decision:** 15 stale improver PRs, all based on old `main`. Not mass-merged or
mass-closed — that is 15 outward-facing actions on work this session did not author.

## 🔓 2026-07-29 21:40 — "improver PRs bypass CI" was ONE MISSING PERMISSION
The risk restated at the top of every monitor tick — *"the improver PRs bypass CI so main can
silently break (this has happened 3 sessions running)"* — was not GITHUB_TOKEN event suppression.
It was `actions: write`.

`continuous-improvement.yml` granted `contents: write` + `pull-requests: write`. `gh workflow run`
posts to `/actions/workflows/{id}/dispatches`, which requires **actions: write**. So every run since
the step was written ended:
```
could not create workflow dispatch event: HTTP 403: Resource not accessible by integration
##[error]Process completed with exit code 1
```
…while `continue-on-error: true` reported the step as **success**. Verified in runs `30476849972`
and `30483279439`: job conclusion `success`, step conclusion `success`, 403 in the log.

**The approach was never wrong.** `workflow_dispatch` is one of the two events *excepted* from the
GITHUB_TOKEN recursion guard; `test.yml` declares `workflow_dispatch:`; and `pacemaker.yml` proves
the mechanism works in this repo ("19 of the last 40 CI runs were dispatched by github-actions[bot]").
The improver simply never asked for the permission.

Kept `continue-on-error` deliberately — a lost dispatch must not discard a run's improvements — but
the failure now emits a `::warning::`, because a bare non-zero exit under `continue-on-error` renders
GREEN. That is exactly how a permanent 403 stayed invisible for the workflow's whole lifetime.

`test_workflow_dispatch_permissions.py` sweeps all 102 workflows for the class. 2 tests fail against
the old file.

**Note against myself:** the sweep's first version also flagged `pacemaker.yml`, and I nearly
reported "the pacemaker has never worked" — the workflow holding the fleet's heartbeat together. It
grants the permission on line 74 *with a trailing comment*, and my regex anchored on end-of-line.
Caught by opening the file instead of trusting my own red test.

## ⚖️ 2026-07-29 19:30 — CORRECTION to the 18:45 entry, plus the improver's real current failure
I claimed the fence-extraction bug caused "32 of 41 failures, 78% of everything that went wrong".
**Not supported.** Those 32 `syntax check failed` traces are historical. Reading the last full
pre-fix run end to end (`30476849972`, 17:46) shows **zero** syntax failures in it. The extraction
fix is still correct — `startswith("```")` cannot handle a preamble and this `llm()` helper
demonstrably returns preamble — but I have not shown it moved the syntax number, and I implied a
measured result from a log I had not read.

**The real current failure mode**, from that run: 10 attempts, 5 committed, and all 5 failures were
oversized files:
```
· backend/app/backtest/cpcv.py is 13806 chars (> 8000) — skipped, whole-file rewrite unsafe
  ✗ LLM returned nothing for backend/app/backtest/cpcv.py
```
`improve_file()` returns `None` for BOTH "too big to send" and "model gave me nothing", so the
caller charged a deliberate policy skip to the LLM. **Half the run's attempts**, attributed to a
model that never saw the input — corrupting the counter from 17:25 the moment it began working.

Fixed: the caller checks `MAX_FILE_CHARS` itself and skips without recording a failure; the guard
inside `improve_file()` stays as defence in depth (the PR #420 lesson); skips are counted and shown
separately in the run summary. 5 tests, all 5 fail against the old code.

**Two things left open, both deliberately not fixed blind:**
- `pick_target_file()` picks files it can never act on — 5 of 10 attempts wasted against a budget of
  10 for 5 wanted improvements. Size-filtering at selection would roughly double useful work, but it
  changes which files the improver ever touches.
- **The improver's CI dispatch has ALWAYS failed** — **FIXED 21:40, see below.**

## 📉 2026-07-29 17:25 — the improver's success rate was 100% because failures were never counted
Second half of the 15:40 roll-call fix. There I stopped the header printing a fabricated 100%; here
is why it was fabricated. `record_success()` initialises `improvement_stats[type]["failures"] = 0`
and the reporter sums that key — but `record_failure()` only ever appended to `failure_traces`. The
counter existed, was read, and was never written.

Measured in the live memory file: **41 failure traces, every counter 0, 61 successes.** The trace
list is capped at 50, so 41 is a floor. True rate ≈ **61/(61+41) ≈ 60%**, not 100%.

`record_failure()` now increments the counter too. **The 41 historical failures are deliberately not
backfilled** — traces are capped, so a backfill would be a known undercount presented as a total.
Consequence: the rate reads optimistically at first (accumulated successes vs fresh failures). That
is disclosed in the source and in IMPROVEMENTS.md, and a test pins the explanation.

**RESOLVED 18:45 — and it was not an output-quality problem.** 32 of the 41 failures were
`syntax check failed`, ~78% of everything that went wrong, and the cause was one line:
```python
if improved.startswith("```"):   # position 0, or nothing happens
```
`improve_file()` unwrapped the code fence only when the response *began* with one. The cascade's
free providers don't oblige: `llm_common._extract()` falls back to `reasoning_content` when
`content` is empty, so a reasoning model's chain-of-thought comes back AS the response. That is
observable right now in `agent_status.json`, from this same `llm()` helper — *"The user asks: …"*,
*"We need to respond as algo_agent, …"*. Any preamble makes `startswith` False, so the prose and
the fence went straight into `compile()`.

`_extract_code()` now finds a fenced block anywhere in the response and takes the longest one
(responses carry small illustrative snippets beside the real file). 11 tests, all 11 fail against
the old code. Deliberately does **not** salvage prose into code: a response with no fence and no
valid Python still fails the syntax check. The goal is to stop discarding good output, not to start
accepting bad output.

**The two fixes compound**: the counter (17:25) is what made this measurable, and this is what the
measurement pointed at. Expect the improver's real success rate to move sharply once the next runs
land — the previous ~60% was mostly this.

Also verified live this tick: the 15:40 roll-call fix works in production. The 17:14 run wrote
`total_runs: 61, success_rate_pct: 100.0, failures_recorded: 0` — real numbers where it had written
`0` and `0`, with the new `failures_recorded` field correctly exposing the gap this entry closes.

## ⚠️ 2026-07-29 15:55 — CORRECTION: the ML promotion fix I proposed at 15:10 cannot work
The 15:10 entry below lists three breaks between the trainer and inference and proposes emitting
`lstm_latest.pt` in `AbstractModel`'s schema. **There is a fourth break, and it invalidates that
fix.** There are two classes named `LSTMPredictor` and they are different networks:

```
.github/scripts/ci_lstm_trainer.py      backend/app/ml/models/lstm.py
──────────────────────────────────      ────────────────────────────────
self.lstm                               self.lstm
(no attention)                          self.attention   <- SelfAttention
self.norm                               self.norm
head: Linear(h*2,32)…Linear(32,1)       head: Linear(h*2,64)…Linear(64,1)
      + Sigmoid  → probabilities              no sigmoid → logits
__init__(n_features, hidden,            __init__(n_features, hidden_size,
         layers, dropout)                        num_layers, dropout, bidirectional)
```
`load_state_dict()` fails twice: missing `attention.*` keys, and `head.0.weight` shape 32×256 vs
64×256. Getting the wrapper schema right does not help when the tensors describe another network,
and a forced load would silently mis-scale every prediction (probabilities read as logits).

**Promotion requires unifying the architectures first** — the trainer should import and train
`app.ml.models.lstm.LSTMPredictor` instead of defining a second network with the same name. This is
the same duplicate-implementation failure as `run_desk()`: a copy that reads exactly like the real
thing, so a change to either looks correct and does nothing.

`.github/scripts/test_lstm_promotion_contract.py` encodes the invariant *conditionally* — green
today, fails the moment the trainer writes a path `InferenceService` reads while the architectures
still differ. The wrong recipe in `lstm-training.yml`'s header has been replaced with the real
constraints, so nobody follows it by hand either.

## ✅ 2026-07-29 15:19 — THE TRIM CHAIN IS LIVE. Verified in production, not inferred.
The whole path finally ran end to end, on a real desk run, 18 minutes after the persist fix merged.

`.github/state/strategy_trims.json` is now **tracked in the repo** — first time ever. The trimmer
committed it at 15:01 as `c10a64fa`:
```json
{"avellaneda": {"trimmed_at": "2026-07-29T15:01:25Z",
                "reason": "cumulative return -7.9% ≤ -5.0% over 10 trades",
                "stats_at_trim": {"trades": 10, "win_rate": 0.6, "total_return_pct": -7.9157}}}
```
Then the 15:19 equity desk run (`30465278667`, head `7f79e920`) printed:
```
✂ 2 strategy(ies) retired by the trimmer will not trade: avellaneda, avellaneda_stoikov_mm
```
Two names because `_expand_truncated` resolved the legacy 10-char attribution key to its single
registry match — exactly the mechanism that was only ever tested against a fake registry before.

**The proof that it took effect, not just printed:** `avellaneda` appears **exactly once** in the
805-line log — that line. No signal, no sizing, no order. The run was otherwise fully active:
`Done. 9 orders placed across 9 desks`, orders filling, no loss cap.

So the sequence attribution → trim → **persist** → desk read → key expansion → exclusion is now
closed and observed. Persist was the only broken link; every other stage had already been fixed
this session and was waiting on it.

## 🔬 2026-07-29 15:10 — two open questions from the sweep, answered by measurement
**`ml_models: ok=false, count=0` is NOT "training isn't running".** `lstm-training.yml` has run six
times, weekly, all six green, 1.05 MB of artifacts each. Three independent breaks stop any of it
reaching inference: the trainer writes `models_artifacts/lstm_spy_1d/model.pt` while
`InferenceService` reads flat `models_artifacts/lstm_latest.pt`; `AbstractModel.load()` expects
`{"state_dict", "metadata": {"init_kwargs"}}` while the trainer saves `{"model_state_dict",
"n_features", …}` (a correctly-named file would still `KeyError`); and nothing commits, so the
30-day artifact expires. **The manual promotion recipe in the workflow header is itself wrong** —
following it exactly puts the file where nothing looks, in a format that would not load.
Fix belongs in the trainer: `ml/inference.py` is on the AVOID list in `backend/app/ml/CLAUDE.md`.
**Not shipped blind** — torch is absent here, so the `save → load → predict` round trip cannot be
verified locally, and an unverified promotion path is the exact failure this file keeps recording.
The check belongs in the training workflow, where torch exists. Details in IMPROVEMENTS.md.

**`total_runs: 0, success_rate_pct: 0` across 18 agents was a metrics bug, now fixed.** Agents are
running fine. `improvement_stats` has two writers sharing one dict with incompatible key spaces AND
schemas: `continuous_improver.record_success()` keys by improvement type with
`{successes, failures, test_pass}`; `SharedContext.record_success()` keys by agent name with
`{runs, successes}`. The reporter read `runs` indexed by agent name — and
`SharedContext.record_success()` has **zero call sites** outside its own docstring, so that
dimension has never been written. Real figure: **61 recorded attempts**. Header now reads
`61 recorded runs`. Deliberately does not publish the 100% rate: `failures` is never incremented, so
that ratio is definitionally 100 and quoting it as measured would be fabricated — it says
`no failures recorded` and ships `failures_recorded` alongside.

## ✂ 2026-07-29 14:50 — the trimmer produced the right answer and the workflow deleted it
I went to *verify* the first live retirement rather than assume it, and found the run had already
happened and already lost. Run `30457733119`, 13:48 UTC, three consecutive log lines:
```
[TRIM] avellaneda: cumulative return -7.9% ≤ -5.0% over 10 trades
trimmed total: 1 | newly trimmed this run: 1
No trim changes.
```
The third line contradicts the first two. The persist step read
`if ! git diff --quiet -- .github/state/strategy_trims.json`, and **`git diff` cannot see an
untracked file** — it compares the worktree to the index for tracked paths only. The trims file had
never been committed (`git ls-files` confirms it has never existed here), so the gate said "no
change", the commit was skipped, and the file was reclaimed with the runner. The desk then read a
trims file that did not exist and `avellaneda_stoikov_mm` kept trading.

**This is the same blind spot I fixed in `fill-tracking.yml` this morning.** I fixed the instance
and not the class, and it cost a second full cycle. The class is now swept by a test.

`system-status.yml` had it too, and had **never once published**: its header promises "commits a
fresh SYSTEM_STATUS.md so the repo always shows live truth", and `SYSTEM_STATUS.md` has never
existed in this repository. Every run rendered the report and threw it away.

**Fix (both):** `git add -- "$f"` first, then `git diff --cached --quiet -- "$f"` — the index sees
an addition as a change.

**Verified end to end against the real registry**, not a stand-in: a trims file keyed `avellaneda`
expands via `_expand_truncated` to `{avellaneda, avellaneda_stoikov_mm}` (exactly one registry
prefix match), and `_desk_strategies(['avellaneda_stoikov_mm','momentum'], trims)` returns
`['momentum']`. The persist step was the only broken link in the chain.

**Guard:** `.github/scripts/test_state_persist_sees_new_files.py` walks every workflow, finds each
step that stages a named path and commits it, and fails when that path is untracked while the gate
omits `--cached`. It also builds a throwaway repo and *demonstrates* the untracked-invisibility
instead of asserting it. 4 tests; 3 fail against the pre-fix workflows (checked by reverting).

Not fixed, worth knowing: this workflow's `41 */6 * * *` cron fired at 03:53, 09:23, 15:02, 20:03 —
between 1h22m and 3h12m late, never near its slot. The 13:48 run that produced the trim was
`push`-triggered.

## 🧩 2026-07-29 13:55 — the LAST link in the trims chain was tested by presence only
With the pipeline correct end to end, the remaining untested step was the one that matters most:
**does the desk actually exclude a retired strategy?** Coverage was:
```
test_no_dead_desk_path   `_trimmed_strategies()` is CALLED, and appears before `_load_strategy`
```
Neither assertion notices if the `continue` is deleted. The exclusion lived inline inside
`main()`'s signal-generation stage, so nothing could reach it.

**That gap is exactly what bit twice already in this same pipeline:** `run_desk()` held this very
check and was never called at all; `strategy_trimmer.run()` iterated the envelope while
`evaluate_trim()` sat behind seven passing tests. Both times the *tested unit* was fine and the
*untested caller* did nothing.

Extracted to `_desk_strategies(names, trimmed)` and covered for **effect**: 8 tests, and
deleting the `continue` fails 5 of them. Includes `test_matching_is_exact_not_prefix` — trimming
`supertrend` must not take out `supertrend_rsi_tv`, the same prefix hazard the key expansion
refuses to guess at.

### ⚠️ The refactor broke an existing test, and the test was the thing that was wrong
`test_the_trimmer_runs_before_strategies_are_loaded` asserted **source byte offsets**: that
`_trimmed = …` appeared earlier *in the file* than `s = _load_strategy(sname)`. Moving the
selection into a helper defined above `main()` inverted those offsets while the behaviour got
strictly better — so it failed on a correct change (`assert 88480 < 78477`).

**Position in a file is not execution order.** Replaced with a real data-flow assertion: walk
`main()`'s AST, find the variable assigned from `_trimmed_strategies()`, and require that
variable to be *passed into* `_desk_strategies()`. Verified it fires by swapping the argument
for `set()` — which is precisely the silent regression a position check cannot see.

**Two kinds of weak test, both retired today:** presence ("the function is called") and position
("the lines are in this order"). Neither survives a refactor, and neither notices a deletion.
Assert data flow, or assert effect.

## 🎯 2026-07-29 13:40 — THE TRIMMER READ THE ENVELOPE, NOT THE PAYLOAD. Found before the live run.
The trims file still did not exist an hour after the 12:41 slot. First, the boring part: **no
trimmer run has yet seen the data.** The artifact was committed 10:19:16; the last trimmer run
was **09:23 — 56 minutes earlier**, and the 12:41 slot is late (this workflow's four observed
slots ran 1h22m–3h12m late, more cron-starvation evidence).

Rather than wait, I ran `strategy_trimmer.run()` against the **real committed artifact** in a
sandbox. It produced **nothing**:
```
trimmed total: 0 | newly trimmed this run: 0     EVENTS []
```
…even though `evaluate_trim()` on that exact row returns
`(True, "cumulative return -7.9% ≤ -5.0% over 10 trades")`.

**Cause: `load_perf()` returned fill_tracker's WHOLE document**, so `run()` iterated the
envelope —
```
{"generated_at": …, "period_days": 30, "strategies": {…}, "tracked_order_ids": […]}
```
`for name, stats in perf.items()` therefore evaluated `generated_at`, `period_days`… as if each
were a strategy's stats. Only `strategies` is dict-valued, and `evaluate_trim()` on that blob
sees `.get("trades") == 0` → *"insufficient sample"*. **No level of bad performance could ever
trigger a trim.** Now `saved.get("strategies", {})`, matching the producer.

Three things make this worth recording:
1. **Seven existing unit tests all passed throughout** — every one calls `evaluate_trim()`
   directly, and `evaluate_trim` was never broken. Nothing exercised `run()`, the function the
   workflow actually invokes.
2. **`strategy_auto_tuner.py` reads the same file and always unwrapped it correctly**
   (`saved.get("strategies", {})`). Two consumers, one schema, silent disagreement — and only
   the correct one was ever exercised. Verified the tuner works: it evaluated 1 strategy and
   correctly changed nothing (win_rate 60% inside its band).
3. **It was invisible until the artifact existed.** With no perf file, `load_perf()` returned
   `{}` and the loop did nothing either way. The bug needed data to become observable, and the
   data arrived three hours ago.

7 new tests, 3 fail pre-fix — including one that drives `run()` against the **real committed
artifact**, which is what found this. Post-fix, against real data:
`[TRIM] avellaneda: cumulative return -7.9% ≤ -5.0% over 10 trades`.

**Lesson: test the entry point the scheduler calls, not the pure helper underneath it.** The
helper had excellent coverage and was correct; the caller was broken and had none.

## 🔗 2026-07-29 12:45 — I broke a producer/consumer coupling and nothing noticed for 9 hours
Auditing the **third** consumer of the attribution artifact now that it finally exists.
`strategy_auto_tuner.py` is wired correctly end to end — it reads the perf file, writes
`tuned_thresholds.json`, its workflow **stages before comparing** (`git add` then
`git diff --cached`, the right order — unlike the bug I shipped in #1191), the desk loads it at
`desk_order_placer.py:282` and **uses** it at line 2047:
```python
threshold = max(_TUNED_THRESHOLDS.get(sname, desk.confidence_min), desk.confidence_min)
```
Nothing broken. But its schedule said:
```yaml
# Run at 22:30 UTC Mon–Fri (30 min after fill tracker completes)
- cron: "30 22 * * 1-5"
```
**That comment became false when I moved the tracker in #1202** from `0 22 * * 1-5` to
`11 */6 * * *`. At 22:30 the freshest attribution is from the 18:11 slot — **4h19m old, not 30
minutes**. I changed a producer's schedule and never checked who depended on it. Nothing failed,
because nothing tied the two schedules together; the comment just quietly stopped being true.

Weekday-only was wrong for the same reason it was wrong on the tracker: the crypto desks trade
24/7, so weekend fills never reached the tuner at all.

Now `30 0 * * *` — 19 min after the 00:11 producer slot, every day. **Deliberately daily rather
than matching the tracker's 6h cadence:** this moves per-strategy confidence thresholds, and
re-tuning those 4×/day invites churn in what the desk will trade. The tuner needs ≥5 trades,
which accrue over days — freshness matters less here than stability.

**The durable fix is the test, not the cron.** `test_perf_attribution_persists.py` now asserts
the *relationship*: each consumer must fire after a producer slot and within 90 minutes of it,
and no consumer may be weekday-only. Both fail against the old cron. The next time someone
moves the producer, the consumers fail loudly instead of drifting.

**Lesson: a schedule is an interface.** Three workflows are coupled by cron arithmetic that
exists nowhere in code, so changing one silently desynchronised another — and the only record of
the contract was a comment, which is not checkable. Encode cross-workflow timing relationships
as tests or they rot the moment anything moves.

## 🔬 2026-07-29 11:45 — the trim expansion was only tested against a FAKE 8-name registry
Closing a gap in my own work from an hour ago. `_expand_truncated()` expands a legacy truncated
attribution key only when **exactly one** registry entry shares the prefix — but *whether a
prefix is unambiguous is a property of the REAL registry*, and every test I wrote for it used a
hand-made 8-name set. A prefix that resolves uniquely there can be ambiguous among the real 116,
in which case the expansion silently stops, the trim reverts to a phantom, and **those unit
tests stay green throughout**.

Verified against the real registry, and it holds:
```
avellaneda -> avellaneda_stoikov_mm     supertrend -> supertrend      (correctly NOT expanded)
vol_of_vol -> vol_of_vol_timing         commodity_ -> commodity_      (correctly NOT expanded)
intraday_s -> intraday_seasonality      crypto_ada -> crypto_adaptive_trend
```
Now pinned by `backend/tests/unit/test_trim_expansion_real_registry.py` (7 tests), which lives
under `backend/` specifically so `STRATEGY_REGISTRY` is guaranteed importable — a `skip` on
import failure would make it vacuous, which is the exact thing it exists to prevent.

The strongest test is the general property: **truncate every one of the 116 names to 10 chars
and expand back — it must return the original or the untouched prefix, and NEVER a different
strategy.** Exactly four names are unresolvable (`commodity_×3`, `supertrend_rsi_tv`), pinned so
growth in that set is noticed rather than silently costing retirements.

Verified the guard fires by loosening `len(matches) == 1` to `>= 1` (the over-eager direction,
which would retire a strategy that was never judged): 3 tests fail, including the round-trip.

**Reusable point, and it is the same one three times now:** a guard is only as good as the
substrate it runs against. A fake registry, a locally-rebuilt coid, a substring instead of the
real import — each time the test passed while the thing it guarded was broken.

## ✅ 2026-07-29 10:37 — THE ATTRIBUTION ARTIFACT EXISTS. Five changes, verified end to end.
`336afa69 chore(fills): update strategy P&L attribution` is on main, and
`backend/performance_log/strategy_performance.json` is **in the repository** for the first time.
Contents, read rather than assumed:
```
generated_at: 2026-07-29T10:19:16Z   period_days: 30   tracked_order_ids: 10
avellaneda   trades=10  win_rate=0.60  total_return_pct=-7.9157
```
The chain that got here: commit step (#1191) → cadence (#1202) → chained trigger (#1209) →
stage-before-compare (#1215, the actual blocker) → this.

**`evaluate_trim` on that row returns `(True, "cumulative return -7.9% ≤ -5.0% over 10
trades")`** — the trimmer will retire it on its next run (`41 */6`, so 12:41 UTC).

### 🔑 …except the key is `avellaneda`, TRUNCATED — the trim would retire NOTHING
Those ten fills predate the full-name coid fix, so attribution is keyed by the old 10-char form
while the desk checks `sname in _trimmed` with the full registry name
`avellaneda_stoikov_mm`. They never match: the trim gets written and is a **phantom**.

This is the 7-day tail predicted in #1199, now concrete. It self-resolves as old fills age out —
but that wastes a week of data on a strategy that is bleeding **-7.9% over 10 trades right now**.
So `_trimmed_strategies()` now expands a truncated key against the registry, using the SAME rule
as `desk_trade_sync.parse_strategy_from_coid`: expand only when **exactly one** registry entry
shares the prefix, and never expand a key that is itself a registry name.

That last clause is not hypothetical — **`supertrend` is simultaneously a real strategy and the
10-char prefix of `supertrend_rsi_tv`**, so a naive expansion would retire the wrong one.
`commodity_` matches three and is likewise left alone. Both forms are kept, so a trims file in
either format matches. 10 tests, 9 fail pre-fix.

**⚠️ This is a real behaviour change, not just plumbing.** On the next desk run after the
trimmer fires, `avellaneda_stoikov_mm` will stop trading entirely. Today it trades at 0.60×
under the *other* (live, leaderboard-driven) pruning path. That is the trimmer working as
designed for the first time — but it is the first time, so the next tick should confirm a
`✂ … retired by the trimmer` line and no orders from that name.

## 🪤 2026-07-29 09:45 — `git diff` DOES NOT SEE UNTRACKED FILES. That was the real blocker.
The chain worked. **fill-tracking ran at 08:53:32, succeeded in 8 seconds — and the artifact is
still not in the repo.** The log says:

```
No attribution changes.
```

**My own #1191 fix was wrong.** The persist step tested, *before staging*:
```bash
if [ -f "$f" ] && ! git diff --quiet -- "$f"; then   git add "$f"; git commit; ...
```
`git diff` only compares **tracked** files. The artifact had never been committed, so git
reported no difference, `! git diff --quiet` was false, the `else` branch ran, and the commit
was skipped. **That condition could never make the FIRST commit of a new file** — which is the
only commit that ever mattered here.

So the four-tick chase resolves to: commit step (#1191) ✓ correct in intent but unable to create
the file; cadence (#1202) ✓; chained trigger (#1209) ✓ — *it delivered a run today*; and this,
the actual blocker, sitting inside the fix that was supposed to solve it. Now:
```bash
git add -- "$f"
if git diff --cached --quiet -- "$f"; then echo "No attribution changes."; else commit; push; fi
```
Staging first makes an **addition** and a **modification** look the same to the comparison.
A missing file now emits `::warning` instead of silently taking the no-op branch.

**Four behavioural tests run the SHIPPED shell against a throwaway git repo** — brand-new file,
modified file, unchanged file, missing file. `test_a_brand_new_artifact_is_committed` **fails on
the pre-fix step**, reproducing the live bug exactly. Static assertions would not have caught
this: the old step *contained* `git add`, `git commit` and `git push`, so every string-matching
check I wrote passed while the step did nothing.

**Lesson, and it is the sharpest one of the session:** *"the job succeeded"* and *"the job did
its job"* are different claims, and every layer here asserted the first. The workflow exited 0,
the step exited 0, the tests were green, and the output did not exist. **Test the effect on a
real substrate, not the presence of the commands that would produce it.**

## 🚨 2026-07-29 08:40 — THE 24/7 CRYPTO DESK IS RUNNING AT 15% OF ITS SCHEDULE
Chasing the attribution artifact turned up something far more important than the artifact.

`desk-trading-crypto-24x7.yml` has cron `7,27,47 * * * *` — **72 runs/day nominal**. Measured
over the last 24h: **11 runs.** Gaps between consecutive runs, in minutes:

```
173  208  88  58  66  73  89  103  88  140      (intended: 20)
```

All 11 succeeded. `concurrency: cancel-in-progress: false`, and **zero** runs were cancelled —
so this is not self-cancellation. GitHub simply **never created** the other 61 slots. The
overnight trend is worse: the two most recent gaps are 173 and 208 minutes.

**This is the live trading loop for a book that trades 24/7.** Signals are evaluated ~11×/day
instead of 72×; anything driven off a desk run — including exits — carries up to **3.5 hours**
of latency. The "24/7" in the filename is aspirational.

**⚠️ This falsifies a claim I made an hour ago.** In #1209 I chained fill-tracking off this
workflow and wrote that it *"fires ~72×/day and demonstrably lands"*, justified by having read
its logs all session. I had seen several runs and never **counted** them. It fires ~11×/day.
The chain is still a real improvement over a 6h cron that was dropped outright, and the two
triggers are redundant — but the premise as I stated it was wrong, and the workflow comment is
now corrected in place. *Seeing a thing happen a few times is not a rate.*

**Consequences for two standing IMPROVEMENTS items, both now evidenced rather than asserted:**
- **P0-GATE (always-on execution worker).** Last tick I attached one data point (62 min late).
  This is a distribution: 85% of scheduled runs dropped, worst gap 3h28m, on the trading loop
  itself. For paper this is degraded throughput; for live, an exit arriving 3.5h late is a loss.
- **P2 workflow consolidation (105 workflows).** Its stated rationale — *"fewer schedules = less
  cron starvation"* — is no longer speculative. The fleet is starving its own trading desk.

**Not shipping a workaround.** Every available trigger is GitHub-scheduled and subject to the
same dropping; adding more of them buys probability, not reliability, and the correct fix is
already written down. The measurement is the deliverable.

**Note on the artifact:** still absent, but the last crypto-desk run was **07:30 — before the
chain merged at 07:47**, so the chain has not yet been exercised. Its absence is not evidence
about the chain. Next desk run is the first real test.

## 🔁 2026-07-29 07:45 — the 06:11 cron was DROPPED, not delayed. Chained off the crypto desk.
Last tick I could not distinguish "late" from "dropped" and said so. **Now resolved: dropped.**
85 minutes past the slot, no run exists in the list at all — a delayed run would have appeared.
That was the stated trigger for acting, so I acted.

**Fix: chain off the crypto desk** (`Desk Trading — Crypto 24/7 (Paper Orders)`, cron
`7,27,47 * * * *`, 24/7). Two properties make it the right anchor:
- it is **cron-actored**, so it escapes the `GITHUB_TOKEN` suppression that makes the CI chain
  useless here (03:40 entry);
- it fires ~72×/day and **demonstrably lands** — I have been reading its run logs all session.

Redundancy, not replacement: the 6h cron stays. Delivery now needs *either* to work.

**A 20-minute anchor would run the tracker 72×/day**, so a freshness gate reads the artifact's
own `generated_at` and skips unless it is ≥5h old — effective rate stays ~4×/day. Deliberately
gated on the artifact's *contents*, not git history: `actions/checkout` is depth-1, so
`git log -- <file>` returns nothing and a git-based check would silently always say "stale".

**The gate fails OPEN** (missing file, unparseable timestamp → run). The entire defect being
fixed is *"never produced anything"*; a gate that failed closed would recreate it in a new form.
Verified all six cases against the shipped code by extracting the step's Python and running it:
missing→run, 1h→skip, 4h55m→skip, 5h05m→run, 6h→run, garbage→run.

5 new tests, all 5 failing on the pre-fix workflow.

**⚠️ My own test caught an arithmetic error in my own test.** `test_the_gate_window_is_under_
the_cron_period` computed the period of `*/6` as `24/6 = 4` hours — that is *runs per day*, not
hours per period, so a valid 5h window looked like it exceeded a 4h period. The workflow was
correct; the test was wrong. Worth noting because it failed *loudly on a correct change*, which
is the cheap direction for a guard to be wrong.

## ⏳ 2026-07-29 06:45 — cron starvation MEASURED: the attribution artifact still has not landed
The commit-step fix (#1191) and the cadence fix (#1202) are both correct and merged. The
artifact still does not exist — because **the cron did not fire**.

| | nominal | actual |
|---|---|---|
| fill-tracking, old cron `0 22 * * 1-5` | 22:00 | **23:02:41 → 62 min late** |
| fill-tracking, new cron `11 */6` | 06:11 | **not started by 06:41 → ≥30 min late, possibly dropped** |

The 62-minute figure is unambiguous: that workflow had exactly one cron at the time. Meanwhile
the rest of the fleet *is* firing (10+ scheduled runs between 06:19 and 06:37), so this is not a
global Actions outage — it is ordinary GitHub cron jitter, which **delays and sometimes silently
drops** scheduled runs under load. A dropped run never appears in the run list at all, so
"didn't fire" and "will fire soon" look identical until the next slot.

**This quantifies the P0-GATE claim.** IMPROVEMENTS has long asserted *"GitHub Actions cadence
(cron starvation, ~15-min floor, suppressed events) is acceptable for PAPER only"* as the reason
an always-on worker is required before `TRADING_MODE=live`. That was an assertion; it now has a
number attached — **62 minutes late on a measured run**, on the workflow that feeds strategy
retirement. For a paper book that is a nuisance. For a live book, a stop-loss sweep or an exit
arriving an hour late is a loss, and this is exactly the failure mode the gate exists for.

**Deliberately NOT shipping a workaround.** The tempting fix is to chain fill-tracking off a
`workflow_run` from a reliably-firing cron workflow (desk-trading fires every ~15 min and is
cron-actored, so it would not hit the GITHUB_TOKEN suppression from the 03:40 entry). But that
trades one best-effort GitHub trigger for another and adds a second trigger path to reason
about, when the real answer is already written down: the always-on worker. Cadence is now
4×/day nominal; the next slot is 12:11 UTC.

**Verify next tick:** if 12:11 also produces nothing, the cron is being dropped rather than
delayed and the trigger genuinely needs replacing — that would be new information, not a repeat.

## 📐 2026-07-29 05:55 — root principle #5 rests on ONE wire, and the record miscounted it
`CLAUDE.md` #5 is *"Walk-forward only: no in-sample-only backtests are accepted as valid."*
Re-audited it. There are **three** separately-written walk-forward implementations:

| implementation | state |
|---|---|
| `app/backtest/walk_forward.walk_forward()` | **LIVE — exactly one call site**, `api/v1/backtests.py:236` |
| `.github/scripts/ml_experiment.walk_forward()` | LIVE, but defined **locally at its own line 96** — unrelated code |
| `app/ml/training/walk_forward_validate()` | **DEAD** — 125 lines, torch; its only textual reference is a string inside its own `ImportError` |

**The IMPROVEMENTS entry was wrong:** it claimed the app function is called from *both*
`api/v1/backtests.py:236` **and** `ml_experiment.py:152`. The latter calls the same-named local
function, not the app one. So the principle rests on a **single wire**, not two — and nothing
tested that wire existed.

**What ML training does instead:** `ml_retrain.retrain_model` → `train_lstm.train` validates on
a **single chronological holdout** (`val_frac=0.15`, `shuffle=False`, contiguous slices). That
is an ordered split and *not* a leak — worth stating precisely, because "not walk-forward" is
easy to misread as "unvalidated". But it is not what principle #5 says, and the function written
to make it so is dead.

Pinned by `backend/tests/unit/test_walk_forward_coverage.py` (6 tests): the one live call site
must stay imported **and called**, DSR must stay applied, the ML gap stays visible, and the
three implementations must stay distinct.

**⚠️ I wrote a fake guard again, and only caught it by testing the negative case.** The first
version asserted `"from app.backtest.walk_forward import walk_forward" in src`. Aliasing the
import to `... as _wf_unwired` **still contains that substring**, so the guard passed against a
fully unwired principle. Rewrote it to resolve the import binding via AST and check that the
*bound name* is actually called. Now fails as intended.

That is twice in one session (the other was `make_coid`). **Substring assertions about code are
not guards.** Every guard needs its negative case exercised — write the bug, watch it fail, then
revert.

**Also a self-inflicted near-miss:** I first grepped `backend/app ... | grep -v test` and
concluded `deflated_sharpe_ratio` was unused. Every path under `backtest/` contains the
substring `test`, so the filter deleted the evidence. It **is** wired (`walk_forward.py:9,95`).
Caught before reporting it. Beware `grep -v` on path fragments.

## 🔎 2026-07-29 04:45 — arb-bucket audit: 4 of 32 strategies can never place an order
Answered half of a long-standing IMPROVEMENTS question (*"32 strategies in the arb bucket but
near-zero desk fills"*). Of the 32 registry strategies with `risk_bucket == "arbitrage"`:

**28 are desk-wired. 4 are not**, and each for a real reason:

| strategy | why it cannot trade |
|---|---|
| `covered_call` | needs share inventory — already excluded *inline* in the Options desk roster |
| `crypto_basis_roll` | short perp + long spot; Alpaca paper has **no perpetual futures** |
| `funding_rate_arb` | trades perpetual funding rates; same missing venue |
| `dex_cex_arb` | Uniswap v3 vs CEX; **no DEX connectivity** in this deployment |

**None is a defect** — but all four are counted in the bucket while being structurally unable to
produce an order, which inflates its apparent capacity by ~12% and is part of why fills
attributed to it look sparse. The 70/30 arb/directional capital split is stated against a
strategy count that overstates what can actually trade.

Now a maintained invariant rather than a one-off finding:
`backend/tests/unit/test_arb_bucket_reachability.py` fails if an arb strategy is neither
desk-wired nor listed in `DORMANT_BY_DESIGN` **with a stated reason**, and separately fails if a
dormant entry goes stale (gets wired, or leaves the bucket). Same idiom as
`test_factor_exposure_is_still_honestly_unwired`. 5 tests; verified it fires by removing one
dormant entry.

**⚠️ Deliberately only half-answered.** Whether the *28 wired* strategies actually fill, or die
at the confidence gate, needs per-strategy attribution — and `strategy_performance.json` has
never existed; its producer was only fixed hours ago and first runs 06:11 UTC. Guessing would
have been easy and worthless. Recorded as STILL OPEN with the date the data starts existing.

## ⏱️ 2026-07-29 03:40 — the attribution producer ran 1×/day for a consumer that reads 4×/day
Chasing why `strategy_performance.json` still had not appeared 3h after the commit-step fix.
It is not that the fix failed — **nothing has run it yet**, and the reason is structural.

**`workflow_run: [CI]` is dead for agent branches.** Agent-branch CI is dispatched by
`auto-pr.yml` using `GITHUB_TOKEN`, and GitHub does not create new workflow runs from
`GITHUB_TOKEN`-triggered events. So the chain is suppressed for *exactly the branches that
generate the fills*. Measured: **one firing in three hours** across many CI passes — and that
one came from the single CI run triggered by `pull_request` (PR I opened myself via the app, a
different actor). Every other CI run on the branch is `workflow_dispatch | actor:
github-actions[bot]`.

So the cron was the real schedule all along, and it was badly mismatched:

| | cadence |
|---|---|
| producer `fill-tracking` | `0 22 * * 1-5` — **1×/day, weekdays only** |
| consumer `strategy-trim` | `41 */6 * * *` — **4×/day, every day** |

Three of four trimmer runs read stale attribution, and every weekend run read data up to three
days old — while the crypto desks trade 24/7 and produce fills daily.

The old cron was aligned to "after US market close + settlement buffer", but the tracker only
scores fills **already ≥24h old**, so time-of-day cannot matter. The alignment bought nothing
and starved every consumer. Now `11 */6 * * *` — every 6h, all week, 30 min ahead of the
trimmer so it reads fresh data. Safe to run often: fills are de-duplicated via
`tracked_order_ids`, so a repeat run only adds newly-matured fills.

3 new tests compare the two workflows' cadences directly (producer ≥ consumer, no day-of-week
restriction, producer fires earlier in the shared slot); 2 fail against the old cron.

**Lesson: a trigger that LOOKS frequent is not a schedule.** I read `workflow_run: [CI]` as
"runs after every CI pass" and treated the missing artifact as merely pending, twice. The
trigger existed, was syntactically correct, and fired ~never. Same family as the rest of this
session — but the tell here was *rate*, not presence: **when an artifact is late, measure how
often its producer actually ran, don't infer it from the trigger list.**

## 🧹 2026-07-29 02:40 — THREE "open" IMPROVEMENTS items were already done. I nearly redid one.
Picking the next clean item, I chose the **stdlib-logger-kwargs guard** (P2, line ~383) — a
preventive AST scan, well specified, zero blast radius. I had the design worked out and was
about to write `backend/tests/unit/test_logger_kwargs.py` when I checked whether the path
existed.

**It already did — written 2026-07-27, 6 tests, passing.** Same design, down to the allowed
keyword set and the ambiguous-binding skip. It even has the meta-assertion I was going to add
(`assert stdlib_modules` — so a broken scanner cannot pass vacuously). The only thing missing
was the checkbox.

Swept the neighbours and found two more already delivered:

| item | evidence |
|---|---|
| P2 stdlib-logger kwargs guard | `test_logger_kwargs.py` exists, 6 tests pass |
| P1 test-isolation flake (measured) | `_isolate_each_file` per-file DB fixture landed 2026-07-27 |
| P2 original flake note | same fixture supersedes it |

Verified the flake items under the **exact failing configuration** rather than trusting the
fixture's existence — `pytest tests/ -n 4 --dist loadfile`: **1997 passed, 31 skipped, 5
xfailed, 0 failed**, including `test_seed_additive::test_seed_is_additive_and_idempotent`.
All three now closed with that evidence inline.

**Why this matters more than three checkboxes:** an item that says "open" but is done is a trap
that costs a whole tick, and the cost is silent — I would have shipped a duplicate file, it
would have passed CI, and nothing would have flagged it. This is the same shape as everything
else this session (dead code, uncommitted artifact, skipped workflow, self-referential guard):
**the record disagreed with reality, and only reality was checked by anything.**

**Rule going forward: before implementing any IMPROVEMENTS item, grep for its artifact first.**
Closing an item costs a minute; rediscovering it costs a session.

## 🔗 2026-07-29 01:20 — the trimmer chain had a FOURTH broken link: truncated attribution keys
Verifying #1191 (commit the P&L file) turned up the next link. The tracker **does** produce real
data — `✓ Saved performance data: 2 strategies, 18 new fills` — but the keys can never match.

The desk tagged every order `qe-{strategy.name[:10]}-{sym}-{ts}`. **Truncated to 10 chars.** So:
- `strategy_performance.json` is keyed `vol_of_vol`, `avellaneda`
- `strategy_trims.json` inherits those keys
- the desk checks `if sname in _trimmed` with the **full** registry name
  (`vol_of_vol_timing`, `avellaneda_stoikov_mm`) → **never matches**

And the truncation was **ambiguous**, not merely lossy. Two prefix groups collide across the
116-strategy registry:
```
commodity_ -> commodity_momentum, commodity_reversion, commodity_trend
supertrend -> supertrend, supertrend_rsi_tv
```
Three commodity strategies shared ONE P&L bucket, and every `supertrend_rsi_tv` fill was booked
against plain `supertrend`. That attribution feeds the trimmer and the auto-tuner — **a strategy
could be retired for another strategy's losses.**

Fixed by emitting the full name. Longest desk name is 29 chars → a 48-char coid, exactly
Alpaca's cap with zero margin, so the boundary is a hard test rather than a comment (`[:48]`
would clip the *timestamp*, not the name, corrupting the id silently). `fill_tracker` now splits
from the right, matching `desk_trade_sync.parse_strategy_from_coid`, which had the correct
logic all along. No migration needed: `strategy_performance.json` has never existed, so there is
no legacy data — only a 7-day tail of pre-change orders that ages out.

**⚠️ My first version of the guard was worthless and passed against the bug.** It built the
coid with its own helper instead of calling production, so 213/214 passed on the truncating
code. Extracting `make_coid()` and pointing the test at it turned that into **97 failures**.
*A guard that re-implements what it guards is not a guard* — same family as the dead `run_desk`
one entry down: verify against the real path, never a copy of it.

## 🪤 2026-07-29 00:50 — `[skip ci]` ANYWHERE IN A COMMIT MESSAGE SKIPS EVERY PUSH WORKFLOW
Cost ~15 minutes and it will recur, so it is written down. I pushed `09a01eee` and **no push
workflow ran at all** — not `auto-pr`, not `security-scan`. PR #1191 sat with only a Vercel
check and no CI.

The cause: my **commit message body** contained the literal token `[skip ci]` — in a sentence
*explaining why the workflow's own commit step needs it*:

> `[skip ci] because this workflow triggers on workflow_run:[CI] and would otherwise loop`

GitHub honours that token **anywhere in the commit message, not just the first line**, so it
silently skipped every push-triggered workflow for that SHA. Amending the message to
`a skip-ci marker because…` and force-pushing brought all 16 checks back immediately.

**Two traps in one:**
1. **Writing *about* `[skip ci]` disables your own CI.** Any commit that documents skip-ci
   behaviour must avoid the literal token — use `skip-ci` unbracketed.
2. **On this repo, no CI is invisible.** CI (`test.yml`) triggers on `pull_request` and
   `workflow_dispatch` only. PRs opened with `GITHUB_TOKEN` never fire `pull_request`
   (recursion guard), so `auto-pr.yml` **dispatches CI explicitly**. Skip auto-pr and you get
   *no CI at all* — and a PR with zero check runs looks calm, not broken. This also means:
   **if I open the PR myself before auto-pr does, CI is never dispatched.** Let auto-pr open it,
   or push again afterwards so auto-pr fires and dispatches CI onto the existing PR.

Same family as everything else this session: a green-looking absence. `git push` succeeded, the
PR existed, nothing was red — and nothing had run. **Check that the checks EXIST, not just that
none of them failed.**

## ✅ 2026-07-29 00:15 — MKR/USD RESOLVED, and two corrections to what I reported
**The wiring works, first live run.** Run `30409299307` (23:51, `head_sha 95710006`):
```
ⓘ Crypto: skipping 1 non-tradable pair(s): MKR/USD
[stage] ✓ Fetch account and market bars — bars_fetched=97   (was 98 — one fewer request)
```
MKR/USD no longer reaches signal generation, no longer consumes a top-K slot, and produced no
422. `bars_fetched` dropping 98 → 97 is the filter paying for itself before the batch request.

**Correction 1 — Alpaca was right all along.** The filter dropped MKR/USD as **non-tradable**,
which means `/v2/assets?asset_class=crypto&status=active` does **NOT** list it. My "B1
confirmed — the metadata contradicts the order engine" was wrong, and so was reopening the
IMPROVEMENTS premise as impossible. **The originally-proposed fix was correct**: filter the
universe against the active-asset list. It had simply never been wired to anything.

**Correction 2 — I overstated the trimmer damage.** I wrote that "the entire performance-pruning
loop was decorative, so a strategy retired for losing money kept trading". Not so. There are
**two** pruning mechanisms and only one was dead:
- **LIVE and working:** `_fetch_performance_weights()` reads `/api/v1/leaderboard/live`, sets
  weight `0.0` for sustained losers, and the order path skips them outright —
  `✂ {strategy}/{symbol} pruned by attribution (sustained negative live P&L) — no order`.
  Losing strategies *were* being stopped, every run.
- **Dead:** the file-based trimmer, a redundant second mechanism.

### 🔌 00:15 — and the trimmer is dead at the SOURCE, not the consumer (fixed)
Wiring `_trimmed_strategies()` into the pipeline last tick was necessary but not sufficient —
the file it reads is never produced. The chain breaks three links upstream:

`fill_tracker.py` attributes fills back to strategies and writes
`backend/performance_log/strategy_performance.json`. `fill-tracking.yml` runs it on schedule,
exits zero, and **never commits the output** — so on an ephemeral runner the file is computed
and thrown away. It has never existed in the repository. Three consumers read that exact path
and all three are inert:

| consumer | effect |
|---|---|
| `strategy_trimmer.py` | `load_perf() -> {}` → `strategy_trims.json` never written |
| `strategy_auto_tuner.py` | prints *"not found — no data to tune from"* and stops |
| `desk_order_placer.py` | reads a trims file the trimmer never produces |

Fixed: `permissions: contents: write` plus a conditional commit/push step mirroring
`strategy-trim.yml` (`[skip ci]` because this workflow triggers on `workflow_run: [CI]` and
would otherwise loop; backoff retry because state-bot pushes to main are constant). 9 tests,
5 fail pre-fix.

**Same shape as the dead `run_desk()`, one layer out:** a workflow that writes a file it does
not commit is indistinguishable from one that does — until something tries to read it. Green
job, zero exit, no output. **When a scheduled job's whole purpose is to produce an artifact,
check the artifact exists in the repo, not that the job succeeded.**

**Verify next:** after the 22:00 UTC run (or a manual dispatch), `strategy_performance.json`
should appear in the repo; then `strategy_trims.json` within 6h; then a `✂ N strategy(ies)
retired by the trimmer` line on the desk.

### ❌ 23:45 — THE "B1 CONFIRMED" BELOW IS WRONG. The filter was never running.
**Read this before the 22:00 entry.** I concluded from a missing `ⓘ skipping` line that Alpaca
must be listing MKR/USD as active while its order engine refused it. The reasoning was
*"`_filter_tradable_crypto` runs unconditionally for every desk and prints whatever it drops,
so silence means it found nothing to drop."* **The premise was false — the filter does not run
at all.**

`_filter_tradable_crypto` was only ever called from `run_desk()`, and **`run_desk()` has zero
call sites.** It was a complete SECOND implementation of the desk loop, superseded by the staged
pipeline in `main()` and then left in the file reading exactly like live code. So:

- the tradable filter has **never executed in production**;
- the denylist I shipped in #1188 was wired into it and was therefore **inert**;
- and worst, **`_trimmed_strategies()` was in there too** — so the performance-pruning loop has
  been decorative, and every strategy the trimmer retired for losing money kept trading.

All three are now wired into the real pipeline (universe trimmed before the bars batch, so a
dead symbol costs no request/analyze/top-K slot; trimmer applied before strategies load).
`run_desk()` and the now-orphaned `_get_bars()` are deleted rather than left as decoys, and
`test_no_dead_desk_path.py` fails if any guard loses its production call site — plus a sweep
that flags ANY unreachable top-level helper, which is how the next one gets caught early.

**Whether Alpaca's metadata contradicts its order engine is REOPENED and unknown.** The filter
now runs, so the next live run can actually answer it.

**The real lesson, and it is not the one I wrote three times:** I kept adding instrumentation to
a function without ever checking it was reached. *Confirm the code path executes before
interpreting its silence.* Absence of a log line is evidence only once you know the line could
have been printed — otherwise a dead code path and a satisfied condition look identical, and a
fix shipped into dead code passes every test while changing nothing.

### ⚠️ 22:00 — "B1 CONFIRMED" (SUPERSEDED — see 23:45 above, the premise was false)
Run `30402044105` (21:46, carrying the full instrumentation) is decisive. It logged the reject
and the memory line, and **none** of the three bail-out lines:
```
· ensemble[Crypto]: MKR/USD buy x2 (...) -> conf=0.89
  ⚠ alpaca POST /v2/orders → 422: {"code":40010001,"message":"asset MKR/USD is not active"}
  ⓘ MKR/USD marked INACTIVE for the rest of this run
                                                  <- no "lookup FAILED"    → lookup healthy
                                                  <- no "format mismatch"  → set is SYM/USD
                                                  <- no "skipping N"       → nothing dropped
```
`_filter_tradable_crypto` is called unconditionally for every desk and prints whenever it drops
anything, so only one reading survives: **`/v2/assets?asset_class=crypto&status=active` returns
MKR/USD as tradable while `POST /v2/orders` refuses it as not active.**

**This disproves the IMPROVEMENTS.md item** (*"the universe should be filtered against Alpaca's
active-asset list before signals are generated, not discovered at order time"*). Filtering
against that list **cannot** fix this class, because the list is the thing that's wrong. Marked
accordingly rather than left to be re-attempted by a future session.

**What actually works is memory.** The in-process set catches every repeat within a run but
never the FIRST attempt — and that attempt is expensive: MKR/USD took 1 of only 3 passing
signals in BOTH runs, roughly a third of the crypto desk's capacity. So the desk now seeds from
`.github/state/inactive_assets.json` and drops those symbols **before signal generation**,
freeing the slot for a pair that can actually trade. Follows the existing `strategy_trims.json`
pattern: the desk READS state, it never writes from the trading hot path.

Entries **expire after 7 days** (`DENYLIST_TTL_DAYS`). A delisting can be reversed, and a
denylist nobody re-confirms is exactly how a permanently-stale exclusion happens; decay forces
the evidence to stay fresh. If the reject recurs the log names it again and the entry is
refreshed. Fail-soft throughout: a missing/corrupt file, an undated entry, or a denylist that
would empty a desk's universe are all ignored rather than idling the desk. 19 tests.

**Three ticks, three levels of silence.** Closed on one clean run → reopened. Instrumented the
lookup → still ambiguous. Instrumented the remaining bail-outs → answered. Each step the silent
path was one level deeper, and each time the *stated* conclusion outran the evidence by exactly
one step. Instrument every early return in a fail-soft function at once.

## ✅ 2026-07-28 19:30 — fractional shorts CONFIRMED zero; two more order-time rejects fixed
**Verified on the next live run, as promised.** Fractional-short 422s across three runs:
`3 → 1 → 0`. Two market replacements occurred in the last run and neither failed, so the
market-path fix holds. Orders `11 → 15 → 16`.

Two other classes were still dying at the broker, both recurring:
```
403 {"code":40310000,"message":"insufficient qty available for order
     (requested: 1.77, available: 1)","symbol":"SPY"}          x2 in one run
422 {"code":42210000,"message":"asset \"EIDO\" cannot be sold short"}  x2 runs
```
1. **Alpaca will not let ONE order flip a short into a long.** Short 1 SPY, buy 1.77 → 403. The
   buy is now capped at the short size so it **closes** the short — the risk-reducing half of the
   intent, and what Alpaca requires be done first. The long can open next run.
2. **Some assets cannot be shorted at all.** EIDO was rejected *after* the qty was correctly
   rounded to 44 whole shares — whole-sharing does not help when the ASSET is the problem. Now
   checks `/v2/assets/{symbol}` `shortable`, cached, **fail-soft TRUE** so a lookup blip cannot
   silently stop the desks selling. Closing a long in a non-shortable asset is still allowed —
   blocking that would strand every such position.

31 tests, 2 fail against each new guard specifically.

**The pattern across all four of these:** a constraint discovered at ORDER time instead of before
it — the same shape as the delisted-asset filter that already existed. Worth checking the broker
error codes in a live run whenever order counts look lower than signal counts.

## 🔁 2026-07-28 18:00 — the fractional-short fix was HALF a fix (completed)
Verified the 17:30 fix live: **4 whole-share conversions** (HD 1.42→1, ORCL 2.72→2, UNG 37.27→37,
EIDO 44.29→44) and orders up **11 → 15**. But **one 422 survived**, and the log showed exactly why:
```
· UNG sell 37.27 -> 37 whole shares      <- limit path, correctly fixed
↻ limit unfilled after 20s — replaced with market
⚠ 422 {"code":42210000,"message":"fractional orders cannot be sold short"}
```
`_ensure_filled` cancel-replaces an unfilled limit by calling `_place_order` with **no limit
price**, taking the equity MARKET branch — which I had not covered. That branch sent a
**notional** order, so Alpaca derived the share count itself, fractionally, and rejected it.

Short-side equity market orders now carry an explicit whole `qty` (priced off
`/v2/stocks/trades/latest`). Buys keep `notional` — fractional longs are legal and precision
matters. Fail-soft: no price → notional, i.e. the old behaviour, never worse than not ordering.
22 tests, 2 fail against the market-path regression.

**Lesson: verify a fix on the NEXT live run, not just in tests.** The limit path was green and
the tests passed; only the production log showed the replacement route was still broken.

## 💸 2026-07-28 17:30 — ~21% OF EQUITY ORDERS DIED AT THE BROKER, EVERY RUN
Alpaca allows fractional shares on the LONG side but **rejects them short**. Measured on a live
run — **3 of 14 attempted orders**:
```
422 {"code":42210000,"message":"fractional orders cannot be sold short"}
place_order failed EIDO sell / ORCL sell / UNG sell
```
`qty = round(notional/limit, 2)` produces fractions, so every equity SELL that opened a short was
rejected. Those signals had already cleared data, ensembling, the confidence gate, Kelly sizing
and the risk manager — the most expensive place to discover an unplaceable order. Same shape hit
COST the day before, so it recurs.

**A sell is NOT always a short**, which is why this is not a blind floor: under the loss cap only
risk-REDUCING orders pass and those are closes, so flooring would strand a sub-1-share long
forever (`floor(0.4) == 0`). The held quantity decides — `held >= qty` keeps the fraction (a
close), otherwise floor and skip if < 1. Crypto exempt in both directions. Position map memoised,
since it is now consulted per sell order. 17 tests, 5 fail against the pre-fix path.

**~~Already fixed, confirmed:~~ WRONG — see the 21:00 entry.** I wrote that the `MKR/USD is not
active` 422s were gone (0 occurrences) on the strength of a single clean run. They came back
five times in one process a few hours later. `_filter_tradable_crypto` did not drop it.

## 🚨 2026-07-28 16:30 — THE STOP-LOSS ENGINE HAS NO CONFIG FOR ANY LIVE POSITION
The crypto pricing fix worked — and immediately exposed the larger gap behind it.

`PositionMonitor` skips any position with no `pos_exit:` config. Those keys are written **only by
`strategy_runner`, on its own fills**. Every live paper order is placed by the GitHub Actions
desks (`desk_order_placer.py`), whose workflow even sets `REDIS_URL=""` — so they write nothing.

**Result: all 16 open positions are skipped every sweep. No stop-loss, no take-profit.** Verified
live: `/api/v1/positions/UNIUSD/exit-config` → 404 "No active exit config found", and the same
for SHIBUSD. The skip was a `logger.debug`, invisible in production, so an unmonitored position
looked exactly like a monitored one.

Now a per-sweep **warning** with the count and symbols. **Deliberately did NOT apply a default
stop:** no global default exists in this codebase (bots carry per-template `stop_loss_pct` 2-3%,
`strategy_runner` uses `signal.stop_loss`), so picking one would start closing 16 real positions
on a number nobody chose. **That threshold is an owner decision.** Three routes:
  1. desks write `pos_exit:` on fill — needs `REDIS_URL` in the desk workflow
  2. PositionMonitor applies a documented default to configless positions
  3. leave as-is and rely on the desk-side daily loss cap alone

Also note `exit-config` cannot be queried for slashed symbols at all — `/positions/UNI%2FUSD/…`
404s on routing, because the slash breaks the path parameter.

**Deploy on main RECOVERED** (was failing): success on `6880391a`, the crypto-pricing commit.

## ✅ 2026-07-28 15:30 — VERIFIED LIVE: trades reconstructed, positions serving, Vercel BLOCKED
Confirmed against production, not asserted:
```
/api/v1/positions/  -> 15 rows      (was [] — env-credential fallback working)
/api/v1/trades/     -> 50 rows      (was [] — untagged-close fix working)
total realized P&L  -> -$103.76
by strategy: time_series_momentum 11 · stat_arb_etf 11 · avellaneda_stoikov_mm 9
             vol_of_vol_timing 7 · analyst_revision_momentum 2 · low_volatility 2
```
**The P&L feedback loop has data for the first time** — `compute_live_strategy_performance` and
the leaderboard were previously reading an empty set, so the self-scaling weighting had nothing
to learn from.

**✅ 16:00 — THE DASHBOARD IS LIVE.** Confirmed end-to-end through
`quantedge-eight.vercel.app`, which is what a browser actually hits:
```
/api/v1/positions/   -> 200, 16 rows
/api/v1/bots/        -> 200, 61 rows
/api/v1/strategies/  -> 200, 13 rows
```
**I used the wrong deploy indicator earlier.** I watched the JS bundle hash and concluded nothing
had shipped. Only `vercel.json` ROUTING changed — the frontend source did not — so the built
assets hash identically and the hash never moves. **Verify a routing change by hitting the route,
not by diffing the bundle.** (Vercel's commit status still reads `Deployment rate limited` on the
newest main commits, so the rate limit is real and `ignoreCommand` still matters — it just was
not blocking this.)

**`trades` is 0 again — and that is the ephemeral-sqlite problem, not a regression.** It was 50
rows an hour ago. A Render redeploy wiped the DB. Positions survive at 16 because they are read
**live from the broker**; trades are DB-backed. `desk_trade_sync` re-derives them from a 30-day
Alpaca lookback each run, so they return and then vanish on the next deploy. **This is the
clearest demonstration yet of what the Postgres password is costing.**

## 🌐 2026-07-28 14:30 — THE DASHBOARD 404'd AT THE CDN. THE BACKEND WAS FINE ALL ALONG
**The answer to three separate "still empty" reports.** Measured against the live deployment:
```
direct to backend   /api/v1/positions/           -> 200
through Vercel      /api/v1/positions/           -> 404 NOT_FOUND (iad1::…)
through Vercel      /api/v1/scanners/polymarket  -> 200
```
The rewrite source was `/api/:path*`. Vercel's named-segment matcher splits on `/`, so a
**trailing slash** leaves an empty final segment, the rule does not match, and the request falls
through to the SPA fallback. FastAPI declares every collection endpoint with a trailing slash and
axios calls `.get("/positions/")` — so **positions, trades, bots, strategies, orders and
analytics ALL 404'd in the browser** while the same paths returned data directly from Render.
Only `scanners/{desk}` worked, because it has no trailing slash. Fixed to `/api/(.*)` → `$1`.
19 tests, 7 fail on the old source.

**⚠️ I HAD BEEN CHECKING THE WRONG SITE ALL SESSION.** `quant-edge-nine.vercel.app` — which I
repeatedly reported as "frontend 200 ✓" — is an unrelated app titled *"My Google AI Studio App"*.
**The real dashboard is `quantedge-eight.vercel.app`** (title matches the repo build). Also live:
`quantedge.vercel.app` = "Create Next App". Verify by TITLE, not status code.

**Also found:** `Landing.tsx` defaults to `quantedge-api-agb8.onrender.com` — a *second, older*
backend that is alive but whose DB is fully broken. Only reached if `VITE_API_URL` is unset.

**Discord was reporting to nobody.** `channel-monitor.yml` passed **no Discord env at all**, so
every run printed `No token — printing report to stdout only` and exited 0 — success for 39
channels it never opened. `desk-trading-crypto-24x7.yml` had the same gap while its equity twin
posts fine, which is why the desk channels only ever showed equity. Both wired. Sweep: **27 of 43**
Discord-capable workflows have no token; the rest were left alone (many only import the helper).
The existing guard cannot catch this — it fails a workflow passing the webhook *without* the bot
token, so passing **neither** satisfies it vacuously.

**Security Scan has been red on every run — and it was my regression.** The "Secret-leak guard
(hard gate)" installs a light pytest set and runs `test_script_safety.py`, which is pure AST/regex
inspection using zero fixtures. But `_isolate_each_file` (the autouse per-file DB isolation added
at 12:00 today) depends on `_create_tables` → imports `app.models` + `app.database` → sqlalchemy.
So **every** backend test file now drags in the DB stack and the gate errored at setup rather than
gating. Fixed with `--noconftest`. **NOTE the scan checks out `ref: main`, so it only goes green
after landing — never on its own PR.** I had been watching only the `CI` workflow via Monitor and
merged all session without noticing a second red check.

## 🧾 2026-07-28 14:40 — "STILL EMPTY TRADES" WAS A REAL BUG, AND MY EXPLANATION WAS WRONG
Reported twice. I answered the first time with *"nothing has round-tripped yet — expected"*.
**That was wrong.** `recover_negative_cash` flattened **25 positions** on 2026-07-27 18:43 and
`/api/v1/trades/` still returned `[]` hours later. Those are 25 completed round trips.

**Cause.** `reconstruct_closed_trades` opened with:
```python
strat = parse_strategy_from_coid(o.get("client_order_id"), registry_names)
if strat is None:
    continue                      # <- dropped the fill entirely
```
`parse_strategy_from_coid` returns None for anything not `qe-` prefixed. But the flatten goes via
`DELETE /v2/positions`, so **Alpaca generates those closing orders itself** — no `qe-` tag. The
backend's `PositionMonitor` exits are the same shape. So every close this system did not
originate was discarded, and the opening `qe-` buy left a lot that could **never** close. No
Trade row, ever — which also starved the P&L feedback loop and the leaderboard.

**Fix.** An untagged fill now closes open lots for its **symbol**, oldest first, across whichever
strategies hold them — what actually happened at the broker. Attribution stays with the strategy
that **opened** the lot (the close introduced no strategy). Excess beyond open inventory is
**discarded**, not opened as a lot: inventing a strategy for it would corrupt the very
attribution the leaderboard reads. 12 tests, 8 fail pre-fix.

**⚠️ This will not fully show until Postgres is back.** Trades are written to the DB, and the
sqlite fallback is wiped on **every Render redeploy** — and each PR merge triggers one. The sync
re-derives from a 30-day Alpaca lookback so it self-heals, but expect the table to keep resetting
until the password is fixed.

**Lesson: "that's expected" is a claim that needs the same evidence as a bug report.** I asserted
it from reading the code path instead of checking whether closes had actually occurred.

## 💣 2026-07-28 14:00 — A NaN CONFIDENCE SIZED THE *LARGEST* POSITION, NOT NONE (fixed, desk path)
Chasing the constant-confidence note from 12:20. AST-swept all **106** files under
`app/strategies`: **88 compute confidence from data**, 13 have a hardcoded literal (7 mixed).
Hardcoded is mostly defensible — `supertrend` fires only on a discrete trend *flip*, so fixed
conviction is a design choice, not a defect. **The real bug was next to it.**

`Signal.confidence` is annotated "0.0 to 1.0" and **nothing enforces it**, while three consumers
trust that range: Kelly sizing, the desk confidence gate, and the conflict resolution shipped at
12:20. The desk read was `getattr(signal, "confidence", 1.0) or 1.0`, which failed OPEN three ways:
```
NaN   nan < 0.60  -> False  => the gate does NOT skip; approved and Kelly-sized.
      AND min(0.90, nan) -> 0.90, the clamp idiom yield_curve_momentum uses,
      so ONE BAD BAR becomes MAXIMUM conviction rather than none.
0.0   `or 1.0` treats zero conviction as falsy and promotes it to 1.0 — the
      largest size available.
>1 / junk  passed straight into sizing.
```
Fixed via `_sane_confidence()` at both desk read sites — malformed means "no conviction", never
"total conviction". Same direction as the scanner normaliser fix earlier today. 23 tests, 14 of
which fail against the pre-fix code.

**⚠️ SCOPE LIMIT — needs a decision.** The correct home is `Signal.__post_init__`, which would
cover the **backend bot path** too. I implemented it there first and the full suite passed
(1971), then reverted it: `backend/app/strategies/CLAUDE.md` says **"NEVER modify base.py"**.
So **the desk path is protected and the backend bot path is NOT.** Relaxing that rule is the
user's call — the guard's stated rationale is "interface change breaks everything", and adding
`__post_init__` validation changes no interface.

## 👁️ 2026-07-28 13:10 — "STILL NO TRADES": THE DASHBOARD COULD NOT SEE THE BOOK (fixed)
User reported no trades. **They were right about the product surface and wrong about the system** —
and the surface is what counts. Measured live:
```
GET /api/v1/positions/            -> []          Alpaca: equity $21,752.63
GET /api/v1/positions/?account_id -> []                  cash   $17,545.47
GET /api/v1/trades/               -> []          => ~$4.2k of OPEN positions
GET /api/v1/analytics/daily-pnl   -> all zeros
```
Meanwhile the 12:49 desk run placed 3 orders (+$978.87), two filled — including SHIB at conf
0.81, one of the symbols the new ensemble rule unblocked.

**Two separate causes, both now understood:**
1. **`trades: []` is CORRECT.** `desk_trade_sync` only writes **closed round trips** (FIFO). The
   desks have been buying; nothing has round-tripped yet. Not a bug.
2. **`positions: []` was a BUG.** The desks place orders straight at Alpaca and never write the
   `Position` table; nothing else populates it. The endpoint's live-Alpaca branch could not cover
   for that because it required BOTH an explicit `account_id` AND a per-account `encrypted_key` —
   and this deployment's Alpaca credentials live in the **environment**, not on an Account row.
   Every default call fell through to an empty table and reported an empty book.

Fixed: when the DB has no rows, serve live positions from the env-configured Alpaca account.
Follows the pattern `analytics.py` already uses in three places — those env credentials ARE this
deployment's trading account. Scoped to users owning an Alpaca account row; **DB rows keep
priority**, so bot-managed positions stay authoritative. Fail-soft to `[]` so a broker outage
degrades rather than 500s. 5 tests.

**Standing trap this is the third instance of:** the backend DB is on the sqlite fallback and is
NOT a record of live trading. `orders`/`trades`/`positions` being empty says nothing about
whether the desks are trading — check the Actions logs or Alpaca directly.

## ⚖️ 2026-07-28 12:20 — A LONE 0.16 DISSENT WAS VETOING A 0.97 CONSENSUS (behaviour CHANGED)
The instrumentation answered in one run. 76 conflicts across 7 desks, now attributed:
```
Crypto   crypto_adaptive_trend was the ONLY sell voice on all 16 conflicts,
         at 0.16-0.52, against buy consensus of 0.61-0.97
SHIB/USD avellaneda_stoikov_mm(0.90)               vetoed by a single 0.16
LINK/USD vol_of_vol(0.70)+avellaneda(0.90) = 0.97  vetoed by 0.26
```
Across the other six desks disagreement IS genuinely distributed — most strategies appear on
both sides across symbols — so the fix must not become "majority wins". Also visible: several
strategies emit a **single constant confidence** every time (`yield_curve_momentum` 0.89,
`supertrend` 0.72, `central_bank_window` 0.78, `breakeven_inflation` 0.82, `macro_risk_barometer`
0.75). Not addressed here; worth its own look.

**The defect is that the rule ignored confidence entirely** — any opposing signal stood the
symbol aside regardless of strength or count. Now each side is combined with the same
`1-prod(1-ci)` used for agreement, and the dominant side trades at the **NET** confidence
(dissent subtracted, not ignored) only if that net clears `ENSEMBLE_NET_MIN` (default **0.60**,
the desks' own `confidence_min`) — after which the desk threshold and per-strategy tuned
threshold still apply. Widens the funnel, does not bypass the gate.

**⚠️ THIS CHANGES WHAT TRADES** — the first such change this session; everything prior was
instrumentation. Measured delta by replaying all 76 conflicts: **5 unblock, 71 unchanged.**
```
Crypto  5 / 11    (AVAX, DOT, LINK, SHIB, SUSHI — all buys, nets 0.60-0.74)
Commodities 0/7 · Equities 0/26 · Macro/FX 0/14 · Options 0/5 · Intl 0/5 · TV 0/3
```
Every non-crypto conflict still stands aside because those dissents are credible (supertrend
0.72, yield_curve_momentum 0.89). **Revert lever, no code change: `ENSEMBLE_NET_MIN=1.01`.**
22 tests. My own first draft asserted DOGE(0.44) and BAT(0.46) would trade — they do NOT, and
that is now pinned: the dissent must be genuinely weak, not merely outvoted.

## 🔍 2026-07-28 11:40 — THE CONFLICT COUNT WAS DOUBLE, AND NAMED NOBODY
Chasing the open question from 10:40 (34 stand-asides regardless of data coverage). Signals group
by `(desk, symbol, side)`, so a symbol with both a buy and a sell forms **two** groups — and the
conflict line was printed from inside the per-side loop, producing a mirrored pair:
```
· ensemble[Crypto]: AVAX/USD buy/sell conflict — stand aside
· ensemble[Crypto]: AVAX/USD sell/buy conflict — stand aside
```
So **34 lines was 17 symbols**, not 34 events. The rate is still structural — identical on the
starved and full-universe runs — but half the headline number was double-counting, mine included.

The line also **named neither strategy**, which is the thing actually worth knowing: one pair
disagreeing on nearly everything (a systematic mismatch worth fixing) is indistinguishable from
disagreements spread across many strategies (genuinely no edge, stand-aside correct). Now one
line per conflicted (desk, symbol), naming every side with strategy names and confidences.

**Stand-aside behaviour deliberately UNCHANGED** — instrumentation to answer the question, not a
change to what trades. Changing the ensembling rule without knowing which case this is would be
guessing with real orders. 11 tests, incl. pinning that the surviving-group set is byte-identical
so the logging rewrite cannot have altered trading silently. Next desk run answers it.

## 🚨 2026-07-28 10:50 — THE BARS FIX WOKE A LATENT 500 IN THE CRYPTO SCANNER (fixed)
**My own guards caught it**, on a docs-only PR: `test_each_scanner_desk_answers_without_erroring
[crypto]` and `test_no_parameterised_get_endpoint_returns_5xx` both went red on main.
```
crypto scanner 500'd: 1 validation error for ScanResultOut
signals  Value error, signals list cannot be empty
```
`ScanResultOut.validate_signals` rejects an empty `signals`, but the equity AND crypto scanners
returned a `ScanResult` **unconditionally** — a symbol where no condition fired arrives with
`signals=[]`, `score=0`, `side="neutral"`. The endpoint serialises the batch, so **one** such row
500s the whole response.

**It was the pagination fix that surfaced it.** While crypto was starved of bars it returned
nothing and the endpoint served `results: []`; restoring the universe made it emit signal-less
rows. Exact same shape as the original polymarket 500 — latent until a desk produced a non-empty
result. **Live prod still answers 200 only because the crypto Redis cache is empty right now.**

Fixed at BOTH layers, deliberately: producers return `None` when nothing fired (a zero-score
no-signal row is the scanner saying "nothing here" — it only dilutes a ranked list), and the read
path drops such rows anyway via `_normalise_scan_items()` at all 3 sites, because **Redis rows
written before the fix outlive the deploy**. Dropping, not a placeholder signal name: fabricating
`["unspecified"]` would put a meaningless entry in a list the UI presents as opportunities.
13 tests. Note the fixture: a "flat" price series does NOT produce zero signals — constant gives
RSI 0 (`rsi_oversold`), a tight zig-zag gives `ema_stack_bearish`. Found a genuinely quiet series
(sine, period 30) **by testing rather than assuming**.

## 🚨 2026-07-28 09:40 — THE DESKS ONLY EVER SAW THE FIRST PAGE OF THE ALPHABET (P0, fixed)
**Two state changes first:** the **loss cap has LIFTED** (Alpaca rolled `last_equity` at the
session open, as predicted) and the desks are **trading again — 2 orders at 09:27**, with
`cash $21,262.65 < equity $21,744.56`, i.e. positions are being **held**, not flattened.

Then the real find. `_get_bars_batch` paginates on `next_page_token`, and on ANY exception it
`break`s — keeping the pages that landed and abandoning the rest silently. Alpaca paginates in
**SYMBOL ORDER**, so a 429 mid-pagination truncates *alphabetically and deterministically*.
Verified across three runs — the survivors are an exact alphabetical PREFIX of the universe:
```
09:27  bars_fetched=4   AAVE AVAX BAT BCH
06:46  bars_fetched=5   AAVE AVAX BAT BCH BTC
04:45  bars_fetched=11  AAVE AVAX BAT BCH BTC CRV DOGE DOT ETH GRT LINK
```
Now retries the SAME page on 429 (4 attempts, 1/2/4/8s) and, when it does give up, names the
symbols it is dropping instead of printing a generic failure. 13 tests.

### ✅ 10:40 — VERIFIED LIVE, and one of my claims was WRONG
Post-fix `desk-trading` run on `bbd2efac`, measured against the pre-fix run of the same workflow:
```
                       total  stocks  crypto   bars-path failures
pre-fix  06:46           70      50      20            1
post-fix 09:46           98      78      20            0
signals_generated       355 -> 506   (+43%)
passed the conf gate     17 ->  20
```
**The fix is real: +28 stock symbols per run, zero truncations.** The crypto-only workflow went
from 5/20 to full coverage, and UNI/USD and MKR/USD now resolve ensembles (conf 0.97 / 0.89).

**⚠️ I overstated the impact.** I wrote "SHIB, SOL, SUSHI, UNI, XRP, XTZ, YFI, MKR, LTC received
NO bars, ever — no ensemble has ever voted on them." **False.** `desk-trading.yml` always had
**20/20 crypto**; its crypto desks *did* vote on 7 of those 9 pre-fix. Only
`desk-trading-crypto-24x7.yml` was truncated to 5. I measured the starvation on the crypto-only
workflow and generalised it to the system without checking the other workflow's own symbol list —
which I already had in hand. **The truncation's real victim was the STOCKS batch** (50 of 78),
because stocks are the longer list and paginate further.

**⚠️ And my follow-on hypothesis is REFUTED.** I said the starvation likely explained the standing
`sell/buy conflict — stand aside`. It does not. Crypto ensemble resolution is essentially
unchanged with full data:
```
pre-fix   35 crypto ensemble lines -> 1 resolved, 34 stand-aside
post-fix  36 crypto ensemble lines -> 2 resolved, 34 stand-aside
```
**34 stand-asides either way.** The conflict rate is structural in the ensemble logic, not a data
artefact — two strategies systematically taking opposite sides on the same symbol. That is now an
open question with a clean measurement behind it, and it is the next thing worth investigating.

**⚠️ This corrects #1127/#1130.** I attributed the crypto starvation to the two desk workflows
colliding on shared triggers. **Wrong cause.** Decontending them was independently right (22 of
60 runs were duplicates), but the 09:27 run had NO competing run and *still* kept 4 of 20. The
collision was a real bug that was not this bug — and I let a plausible correlation stand in for a
mechanism. What settled it was the alphabetical-prefix check, which is falsifiable; "they compete
for a rate limit" was not.

**Method note:** stashing the source to test the pre-fix tree gave an ImportError (the new
constants were missing), which *looks* like a failing test but proves nothing about behaviour.
Re-verified by disabling ONLY the retry branch while keeping the constants — 4 tests then fail
for the right reason.

## 🚨 2026-07-28 — BUY ON MARGIN → GET LIQUIDATED → GET FROZEN (P0, fixed)
The origin instrument answered in ONE run. All 8 closes were `EXTERNAL`, and all filled
**within ~1.5 seconds of each other** — a mass flatten, not per-position stops. Traced it:
```
17:49  desk-trading         13 orders, +$9,634    cash -$10,829.50  bp $25,001.91  ← healthy
18:42  desk-trading-CRYPTO  🚑 RECOVERY: flattening 25 position(s)
                                                   cash -$14,972.80  bp $17,247.12  ← healthy
18:43  broker               25 sells all filled within ~1.5s
23:44  desk-trading         equity $21,745.65 == cash, -2.28%, 🛑 CAP, 0 positions, frozen
```
**`recover_negative_cash` was destroying a healthy book every cycle.** Its guard was
`cash < 0 AND non_marginable_buying_power <= 0` — but buying marginable equities drives cash
negative and nmbp to zero **by construction**, so that reduces to *"this account used margin"*
and matches every healthy long book the equity desks open. The pathology it is genuinely for
is "$0 available" (orphaned notional buys, Alpaca 403ing 'insufficient balance for USD'), which
has **no buying power left**. Healthy margin use does. Now also requires `buying_power <= 0`.

**Why it hid for so long:** the recovery fires from the **crypto** workflow while the positions
it destroys were opened by the **equity** desks. Reading either workflow's log alone shows only
half the loop — and I had checked only `desk-trading.yml`, so I recorded "recovery never fired"
in good faith and was wrong. **Always check `desk-trading-crypto-24x7.yml` too; it runs the
same script.**

**Follow-up 07:20 — the two desk workflows were starving each other.** They share BOTH the
`workflow_run: ["CI"] completed` trigger AND a `push` path filter (both list
`.github/scripts/desk_order_placer.py`). Neither has a job-level market gate, and they use
*different* concurrency groups (`desk-trading` vs `desk-crypto`) so nothing serialises them —
both launch in parallel against the same free-tier Alpaca data API:
```
over 60 runs of each, 22 fired at the identical timestamp:
  (workflow_run, workflow_run) -> 15
  (push,         push)         ->  7
one push pair, 06:46:05, both on sha 49e46ded:
  desk-trading.yml          bars_fetched=70   ALL 20 crypto symbols
  desk-trading-crypto-24x7  bars_fetched=5    ← lost the race, HTTP 429
```
**⚠️ #1127 was HALF a fix and its stated cause was wrong.** It guarded
`github.event_name != 'workflow_run'` and the PR claimed the 06:46 pair was "ignited by one CI
completion" — it was not, it was a **push** collision, so the headline evidence was not even an
instance of what the guard blocked. It did cover 15 of the 22. The `push` path stayed live until
the follow-up below. **Lesson: I read `workflow_run` off the YAML and never checked the runs'
actual `event` field.** One `curl .../runs | Counter(r['event'])` would have shown it.
`desk-trading.yml` already runs **every** desk, crypto included, 24/7 — so the crypto-only run
was doing duplicate work, doing it worse, and degrading the twin doing it properly. Thin data is
not cosmetic: it feeds the ensembles, and this is very likely a large part of the standing
`sell/buy conflict — stand aside` on nearly every crypto symbol. The crypto job now runs on a
**whitelist** — `if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'`
— rather than blacklisting one trigger at a time. That is the real lesson: this workflow exists
for its own cron, so anything it *shares* with the equity workflow it should cede, and a
whitelist stays correct when a new shared trigger is added later. 8 tests, verified failing
against the shipped half-fix, incl. one parameterised over both shared triggers so a partial fix
fails instead of looking complete, and one asserting the cron survives so "stop the duplicate"
can't be confused with "delete 24/7 crypto coverage".

**Test-methodology note, the important part.** My first draft asserted `recover_negative_cash(...)
is False` and **passed against the unfixed code** — because without credentials the flatten
*attempt* raises, the broad `except` swallows it, and the function returns False either way.
The return value cannot distinguish "declined" from "tried and failed". Rewrote to patch
`_alpaca_delete_sync` and assert on whether the destructive DELETE was *attempted*. Now 5 of 8
fail on pre-fix code while the 3 guarding the real pathology still pass — proving the fix stops
the bad liquidation without disabling the feature. **Third time a green test has been worthless
here; always run new guards against the pre-fix tree.**

## 🛡️ 2026-07-28 — the 5xx guard never covered parameterised routes
After fixing the scanner 500 the obvious question was: there IS a
`test_no_get_endpoint_returns_5xx` — why didn't it catch it? Because it walks **parameterless**
GETs only. **25 of 128 GET routes take a path param and were never smoke-tested.** Now covered:
placeholders are bogus ids and the assertion is "does not 5xx", so 404/422 passes and no
fixtures are needed. `test_every_path_parameter_has_a_placeholder` stops a new param silently
dropping a route out of the walk.
**One value per param is NOT enough** — my first draft used `desk="equity"` and PASSED against
the unfixed scanner (equity returns empty here; only polymarket produced a row). Placeholders
are now lists expanded over every value; re-verified failing on the pre-fix tree.
**500 vs 502/503:** three market-data routes 502 with `{"detail": "Alpaca bars error: 401"}`
and one 503s on Redis — deliberate HTTPExceptions for absent upstreams, not crashes. Those pass;
a 502/503 *without* a `detail` still fails. A guard that cries wolf gets deleted.

## 🔴 2026-07-28 — /api/v1/scanners 500'd on EVERY non-empty result (fixed)
Found by writing the first test for a live endpoint with 0% coverage. Producer and schema
disagreed on BOTH fields since they were written: scanners emit `min(score,100)` (0-100) against
a schema requiring `ge=0,le=1`, and emit `long`/`short`/`long_yes`/`long_no` against a validator
allowing only `{buy,sell,neutral,none}`. **All three desks** — equity/crypto only looked fine
because they returned empty in the test env, so serialisation never ran with rows. Invisible
outside: anonymous probes get 401 (verified against production).
Fixed at the boundary via `_normalise_scan_item()` at all 3 construction sites. **NaN fails
SAFE**: `min(1.0, nan)` returns 1.0 in Python, so naive clamping would make a malformed score
read as MAXIMUM confidence on a ranking signal; forced to 0.0.
Tests: 25 unit + 14 integration, verified failing on the unfixed tree.
**Mistake recorded:** first draft registered a user per test (~11 vs a 10/min limiter) and
starved neighbouring files — `test_auth_register_then_login` went red while passing alone. Now
one cached token per file. Second time this session a "fix" damaged a shared fixture.

## 🔴 2026-07-28 — the BACKEND's agent subsystem is unreachable (9 modules) — USER DECISION
Follow-on from the coverage baseline. 0% coverage ≠ dead, so the 37 zero-coverage modules were
cross-referenced against **transitive** reachability from the real entrypoints
(`static_server`, `main`, `api/v1/router`, `tasks/scheduler`). 318 modules, 206 reachable,
**112 unreachable**.
One-hop analysis nearly fooled me: `free_llm_router` *looks* imported — by
`ai_strategy_generator`, `research_pipeline`, `self_improving_loop` — but all three are
themselves unreachable. The cluster imports each other and nothing else reaches it:
`agent_bus`, `agent_memory`, `ai_strategy_generator`, `free_llm_router`, `knowledge_loop`,
`research_pipeline`, `self_improving_loop`, `strategy_auction`, `task_queue`. Six of the nine
have ZERO references outside their own file.
**Bears on "All employees working autonomously?" and "Free llm being used?"** — for the
BACKEND, no. **Qualifier: the GitHub Actions fleet in `.github/scripts/` is separate and DOES
run** with its own `llm_common` cascade, so the agents posting to Discord are real; the
backend's parallel implementation is not.
**NOT switched on — needs the user's call.** Unlike the risk gate/exit path/ML features (which
were *supposed* to run), starting these means the backend begins generating strategies and
modifying itself autonomously: a policy decision, not a wiring bug.
Guarded by `backend/tests/unit/test_module_reachability.py` (5 tests) — fails on a NEW orphan,
on a known orphan silently going live, or if unreachable count exceeds a ceiling. Verified to
fire. Includes a sanity floor so a broken graph walk fails loudly.

## 📊 2026-07-27 — coverage is measured for the first time: 51%, 37 modules at 0%
CI now runs `--cov=app` and publishes the total plus the **0%-module list** to the PR summary.
Baseline: **51% of 28,261 statements; 37 modules (3,594 statements) never executed by any
test.** Reported, NOT gated — no `--cov-fail-under`, because picking a threshold is a policy
call and a number chosen to pass today teaches nothing.
The 0% list is the point: it is the same *"implemented but never runs"* class as the risk gate,
the exit path, the ML feature builders and `configure_logging` — now found automatically
instead of by hand-rolled AST sweeps. Worth triaging: `ml/registry.py`, `ml/serving/serve.py`,
`ml/serving/ab_router.py`, `tasks/agent_bus.py`, `tasks/task_queue.py`,
`tasks/strategy_auction.py`, `tasks/stock_scanners.py`, `comparison/engine.py`,
`options/wheel.py`, `options/flow.py`. Cost: suite 72s → ~90s.

## ✅ 2026-07-27 — test suite is now deterministic (per-FILE DB isolation)
`_create_tables` is session-scoped, so every file landing on the same xdist worker shared not
just the schema but **every row**. Under `--dist loadfile` which files share a worker is a
scheduler detail, so the suite was deterministic only by luck of packing. Two real failures
came from this: `test_seed_additive`'s `assert 0 == 61`, and the `scalar_one_or_none`
production bug in `bots/seed.py` that it finally surfaced.
`conftest._isolate_each_file` (module-scoped, autouse) now wipes rows between files. Rows, not
tables — schema is still built once per session, so runtime is unchanged (72s before/after).
**Module** scope on purpose: within a file, tests building on each other is intended; only the
accidental cross-file coupling had to go.
Proven both ways on the exact CI failure: with the fixture it passes, without it fails
`assert 0 == 61`. The tolerant workaround assert was reverted back to the strict
`== len(BOT_TEMPLATES)`, which is stronger. 3 consecutive CI-invocation runs: 1897/1896/1896
passed, 0 failed.

## ⚡ 2026-07-27 — the contract test was hitting the LIVE Yahoo API (7m15s → 4.5s)
`test_strategy_contract.py`'s fixture is named `no_network` and patches Python's `socket` —
but **yfinance fetches via `curl_cffi` → libcurl, which never touches Python's socket module**
(`app/strategies/_failsoft.py` already documented this; the fixture didn't act on it). So all
115 strategies hit the real Yahoo API with real retry-backoff on every run.
**7m15s wall for 5.75s of CPU** — ~99% pure network wait. Now blocks `curl_cffi.requests` too:
**4.5s, 119 passed, zero network chatter.** In CI `--dist loadfile` puts all 115 cases on ONE
worker serialised while three idle, and a Yahoo outage could redden an unrelated PR.
Note the earlier plan in IMPROVEMENTS.md ("mark `@pytest.mark.network` and stub the fetch")
was **wrong** — it would have deleted the very behaviour under test. The contract is
"fail soft when the data source is unavailable"; that condition is now genuinely simulated
instead of merely intended. Guarded by `test_the_network_kill_actually_reaches_yfinance`,
verified to fail against the socket-only fixture.

**Still unwired: `factor_exposure.py`**, the last diagrammed gate. It needs an aligned SPY
benchmark series at the same cadence as the portfolio series; nothing produces one, and a
misaligned series yields a confident wrong beta — worse than no beta. Kept visible by
`test_factor_exposure_is_still_honestly_unwired` rather than quietly forgotten, which is
exactly how all five ended up unwired.

## ✅ 2026-07-27 — THREE DEAD AGENT PIPELINES REVIVED (all Slack-removal regressions)
Read the failures landing in **#ci-failures** rather than the backlog, and found three
pipelines that had been failing on a schedule for weeks. All three were my own regressions
from the Slack→Discord rename. Fixed in #1041 + #1044; **both confirmed green in production
at 08:06 UTC**, first success after 3+ consecutive failures each.
- **Research → Trade (24/7) had never executed once.** `chat_post` is annotated `-> dict`
  but returned `notify.post`'s **bool**; `research_to_trade.chat()` runs before any research,
  so it died on its first post with `AttributeError: 'bool' object has no attribute 'get'`.
  Now returns `{"ok": bool, "ts": str|None}`.
- **Company Brain Sync, dead every 15 min** — `chat_read_channel() got an unexpected keyword
  argument 'oldest'`. And beneath it, `company_brain` read `m["text"]` where Discord sends
  `content`, so it would have ingested **nothing while reporting success** even once the
  signature was fixed.
- **`employee_intros`** had both bugs in mirror form.
Guards added: `.github/scripts/test_call_signatures.py` (nothing watched the fleet's own code
— only `backend/app` was covered) and `backend/tests/unit/test_logger_kwargs.py`. Note the
signature guard **cannot** catch the `chat_post` bug — the call is well-formed, the *return
shape* lied — so that one is covered behaviourally.

### ⚠️ PROCESS: a GitHub squash merge silently dropped a commit
PR #1041 was merged at a **stale head** — the `chat_post` fix (the most important of the
batch) was not in the squash, despite CI having run on it. Caught only by verifying `main`
afterwards; recovered via cherry-pick and re-shipped as #1044. **Always diff `main` against
what you intended to land.** Related: the Actions API here serves heavily cached run/job
status — `test` shows `in_progress` for ~10 min after completing, across every endpoint.
Never merge on a status read without corroboration.

### 🔴 TWO LIVE RENDER SERVICES — the monitors watch the dead one (USER DECISION)
`quantedge-api-9jz0` is **current**: 61/61 templates, Redis connected, 9 trades, has the
`database_primary` health check. `quantedge-api-agb8` is a **stale build**: 29/61 templates,
no Redis, and it reports `status: ok` only because its old health payload predates the
`database_primary` check — **the healthy-looking one is the dead one.**
Every hardcoded fallback points at `agb8`: `render.yaml` (OAuth callback), `docs/DEPLOYMENT.md`,
`Landing.tsx`, `useWebSocket.ts`, `keep-alive.yml`, `smoke-test.yml`, and
**`desk_order_placer.py`** (the order placer). All read `vars.RENDER_API_URL || agb8`, and that
variable **is not set** — so everything falls back to the stale service. This is why smoke-test
has ~17 failures: it has been testing a ghost.
**Fix without touching code:** set repo variable `RENDER_API_URL` to the `9jz0` URL. Repointing
the frontend + OAuth callback needs the Google console changed in lockstep — deliberately left
to the user.

### Still open (not root-caused)
- `/analytics/tearsheet` **500s on the live service**. A real complex-number crash was found
  and fixed (fractional power of a negative base → `complex` → `round()` dies), but the live
  service has 9 trades, all winners, so it is NOT on that path. Needs the server traceback.
- 5 workflows (`agent-health-monitor`, `channel-monitor`, `model-audit`,
  `daily-employee-review`, `run-experiments-agent`) fail **100% of push runs with ZERO jobs
  created** = startup_failure. All 97 workflow YAMLs parse cleanly; the error text is not
  retrievable through the API from here. Their `schedule` trigger also never fires (0 scheduled
  runs in the last 20), consistent with the known dropped-free-tier-cron behaviour.

## 🩸 MONEY-PATH AUDIT — 4 passes done, one theme: code that looked like it worked
Running audit of every `except` handler and result construction in `execution/`, `risk/`,
`brokers/`, `strategies/`. Full detail in IMPROVEMENTS.md; the shape of what keeps turning up:
a guard that reports a *different* failure than the one it hit, or a failed operation that
returns something indistinguishable from success. Landed so far — `CompositeExit` returning
"nothing to do" when every exit rule threw (#—); Alpaca brackets degrading to naked fills
(#996); the bracket price guard that **had never once rejected an order** because it built
`OrderResult(reason=…)`, a kwarg that does not exist, and the TypeError was swallowed by the
enclosing handler (#1034); sliced executions (TWAP/VWAP/iceberg/Almgren-Chriss) returning
`status="partial"` + a fabricated `broker_order_id` when every slice failed, which
`strategy_runner` then wrote into Redis as a **phantom position** with a stop-loss on shares
nobody owned (#1035).
**Structural guard now in place:** `tests/unit/test_dataclass_kwargs.py` statically checks that
no dataclass is constructed with a keyword that is not a field — the defect behind three of the
above. Its first run found a fourth: `BacktestSignals(positions=…)` in `lorentzian_knn.py`,
which meant that strategy **could never be backtested**, hence never validated under the
walk-forward rule. Fixed.
**Open for the user (trading-policy calls, deliberately not made unilaterally):**
1. Should a rejected bracket fill naked at all, or abort the entry?
2. `_select_algorithm` looks unreachable past its first branch — `OrderRequest.execution_algo`
   defaults to `"limit_first"`, not `"auto"`, so size-based routing to TWAP/Almgren-Chriss/RL
   has likely never run for any caller that doesn't set it explicitly (including
   `strategy_runner`).

## 🗑️ SLACK REMOVED COMPLETELY — 2026-07-25 (user directive)
Discord is now the ONLY chat integration. Removed: `app/notifications/slack.py`,
`app/integrations/slack_bot.py`, `app/integrations/slack_workspace.py`, 11 slack-*.yml
workflows, 6 slack-only scripts, all `slack_*` config keys, the `/slack/*` API endpoints
(Events receiver, follow-ups, history backfill — all bound to Slack's Events/thread model),
and `_slack_startup_catchup` from `main.py`. `SLACK_BOT_TOKEN` is wired nowhere (68 env
lines stripped from 65 workflows). New `app/notifications/discord.py` (native embeds) is the
single backend notifier; `.github/scripts/notify.py` is the single agent-side one.
**The removal exposed weeks of SILENT message loss** — `llm_common.slack_post` returned `{}`
with no token, so 27 agent scripts had been posting into a void; the hourly employee report
had never worked (called a non-existent `post_message`); and ~10 scripts guarded delivery on
a token that is never set. All fixed — see the top section of IMPROVEMENTS.md.
Two structural guards now keep Slack out (`TestSlackStaysRemoved`,
`test_slack_integration_stays_removed`) because the autonomous improver edits these files
unattended. Renamed: `slack_agent_team.py`→`agent_team.py`, `slack_state.json`→
`agent_state.json`, `SLACK_TOKEN`→`CHAT_ENABLED`, brain category `slack_insights`→
`chat_insights`. Residual "Slack" prose in comments is tracked as a P3 in IMPROVEMENTS.md.

## ⚡ STATE AS OF 2026-07-24 (read this first)
**(Postgres status: see the ROOT CAUSE PROVEN block at the top of this file.)**
Superseded background (kept for the record) — three earlier theories, all correctly disproven:
  * **NOT wrong-region.** The direct DB host `db.vexzwnfbmznvxoxxktax.supabase.co` resolves to
    IPv6 `2600:1f1c:b5d:e600:…`, which AWS ip-ranges.json maps to `2600:1f1c::/36` = **us-west-1**.
    So the project genuinely lives in us-west-1 and the URL's us-west-1 pooler region is correct.
  * **NOT paused.** `https://vexzwnfbmznvxoxxktax.supabase.co/rest/v1/` returns **HTTP 401
    `{"message":"No API key found in request"}`** — a live PostgREST. A paused project would not.
  * **NOT (merely) a stale boot** — a redeploy alone won't help, because a fresh boot would hit
    the same rejection.
Yet the us-west-1 Supavisor answers `tenant/user postgres.<ref> not found`. Supavisor is
partitioned into CLUSTERS (`aws-0-<region>` vs the newer `aws-1-<region>`) and ports (txn `:6543`
vs session `:5432`); a tenant lives on exactly ONE combo, and the URL points at the wrong one.
NOTE the region-autofix "already pooler in region us-west-1 — OK" log does NOT prove the host is
`aws-0` — its regex only matches `aws-0-…`, so an `aws-1-…` host also prints "OK". We can't see
the masked URL value, and can't test Postgres from this sandbox (HTTPS-only proxy; port 6543 unreachable).
FIX SHIPPED — `.github/scripts/render_probe_pooler.py` + wired into `db-url-region-autofix.yml`
(runs every 3h, creds-guarded, fail-soft). GitHub runners CAN reach Postgres directly, so the
prober reads the current Render `DATABASE_URL`, and if it doesn't connect, TESTS candidate hosts
(`aws-1`/`aws-0` × `6543`/`5432`) with a real `SELECT 1`, then PATCHes Render to the first host
that verifiably works + redeploys. Safe by construction — it only ever patches to a proven host.
If NO host works, the sole remaining cause is a **rotated DB password** (needs a human: Supabase
dashboard → Database → reset password → update Render `DATABASE_URL`); the prober logs exactly that.
7 unit tests on the pure logic (parse/candidate/build). **Next `db-url-region-autofix` run (cron
`27 */3`) executes the prober** → expect auto-heal unless it's the password case.
The region-autofix stays as a drift guard. Downstream is READY: once Postgres is reachable, the
next boot runs `alembic upgrade head` → 22-table schema (incl. catch-up `k6f7a8b9c0d1`) → durable.
- Supabase "Trade" (`vexzwnfbmznvxoxxktax`, us-west-1, pg17) is ACTIVE_HEALTHY; `public`
  schema still empty until the first good boot lands.
- **How to verify it worked:** `/health/detailed` → `database.fallback` gone / `status` not
  degraded; Supabase `list_tables` shows ~22 tables. The hourly monitor watches for this.
- **If the region was somehow already correct** (autofix logs "already pooler in region
  us-west-1 — OK" and never redeploys) → then it IS a stale boot after all; the remaining
  lever is a manual Render "Deploy latest commit". But "tenant not found" on a healthy
  project ⇒ wrong region is by far the likeliest, which this fix targets.
**🔥 P0 FIXED THIS SESSION — main could not boot at all** (PR #934). Autonomous-improver PRs
put THREE import-time crashes / breakages on main: (1) `@api_router.middleware()` on an
APIRouter (no such method); (2) `pipeline.py` used FastAPI `Path(...)` where the name is
`pathlib.Path`; (3) PR #929 deleted `PIPELINE_DEFS` (3 endpoints 500) + rewrote a strategy
test to break on the correct `BacktestSignals` type (5 fails). All reverted/fixed; main is
green + bootable again. This is the 3rd straight session of improver-broke-main.
**Earlier P0 (still true):** improver PR #876 had 400'd every `/api/v1/strategies/*` — fixed
in #879, verified live 401.
**🔥 P0 LIVE REGRESSION found + fixed:** autonomous-improver PR #876 wrapped the ENTIRE
strategies router in a bogus `X-Strategy-Entry/Confirmation` HTTP-header gate → every
`/api/v1/strategies/*` GET returned 400 (dashboard strategy list + demo session dead;
verified live 400). Reverted in #879 (router.py); endpoint-smoke + demo-session tests
green; deploy shipping the fix.
**🕳️ SYSTEMIC (P1, documented in IMPROVEMENTS.md):** #876 reached main with ONLY a
Vercel check — GitHub suppresses `pull_request` CI runs for bot-`GITHUB_TOKEN` PRs, so
"reward gate = full CI green" NEVER runs on improver PRs. They merge unvalidated. Treat
main as untrusted after any bot merge. Durable fix = branch protection requiring the
`test`/`test-agents`/`frontend-build` checks (repo-admin setting).
**SHIPPED — desk posts → shared brain:** the missing consume side. `notify.
read_channel_recent` reads desk Discord summaries back → `company_brain.
fetch_desk_knowledge` → `desk_outcomes` brain category + CIO synthesis → `llm_common.
get_company_context` surfaces `Desk results:` into every employee prompt. Stateless
(no git-state churn). 13 tests.
**USER ACTIONS remaining (dashboard):** (1) Supabase is HEALTHY now — if the forced
redeploy does NOT reconnect (still "tenant not found"), fix the Render `DATABASE_URL`
to the us-west-1 pooler (see the Supabase block above); (2) suspend the stale `agb8`
Render service (double-executes vs the same Alpaca account); (3) optional
TRADIER_SANDBOX_TOKEN for real greeks; (4) consider branch protection on main (closes
the improver-PR CI-bypass hole).

## Earlier: STATE AS OF 2026-07-20 PM
**Shipped this session (all merged to main, free stack):** bot exit sweep was
silently dead on the SQLite-fallback deploy (naive-datetime TypeError killed the
5-min sweep → positions never closed) — FIXED + 6 lifecycle tests. Exploration
allocation (min-notional evidence clips so the ≥20-trade pruning loop can judge the
whole book). `regime` + `ml_signal` bot conditions (OA-style decision recipes; run
inference/regime once per tick, fail-soft). Desk-post dedupe (`notify.post_dedup`,
stateless via bot API — fixes the FX-desk "same line 8×/day" spam) + FX post enriched
with side/entry px. Funnel telemetry (`regime=` + generated→survived→explored→placed)
in the desk summary. Schema-drift CI gate + Supabase keep-alive. LLM backstop
(ANTHROPIC/OPENROUTER) wired into all 28 LLM workflows (was why Discord went silent).
Frontend→keeper backend + client base-URL guard. 226 stale improver PRs closed;
continuous_improver barred from money-path files + never commits state into PRs.
**USER ACTIONS (dashboard, ~5 min):** (1) suspend the stale `agb8` Render service
(double-executes vs the same Alpaca account); (2) run schema-drift-gate via
workflow_dispatch, then UNPAUSE Supabase (fallback SQLite is ephemeral until then);
(3) optional: add TRADIER_SANDBOX_TOKEN secret for real greeks.
**NEXT in queue (IMPROVEMENTS.md, specs ready):** Polymarket CLOB order path;
desk-posts→shared-brain; regime volatility axis in desk_order_placer; 0DTE-VRP mode
(after Expiration Protocol); TradingAgents-style debate gate; weekly OA-comparison
report + bot TP/SL tuner.

## Earlier 2026-07-20: TWO stacked infra faults, both root-caused + fixed:
1. **Supabase project PAUSED** (free tier, 7 idle days) → the keeper backend
   (quantedge-api-9jz0.onrender.com) 500'd on every DB endpoint while /health stayed
   green. FIX (PR #769): `ensure_database_alive()` probes the primary at boot and falls
   back to local SQLite (rebinds AsyncSessionLocal in place, creates schema; bots reseed;
   desk trades resync from Alpaca 30d). /health/detailed keeps a failing
   `database_primary` check → status stays degraded until the user unpauses Supabase.
2. **Frontend pointed at the WRONG backend**: vercel.json proxied /api to the STALE
   agb8 service (29-bot old build, 0 trades) — new merges deploy to 9jz0. FIX (PR #771):
   rewrite → 9jz0. Site login = "Explore as Guest (Demo)" button (POST /auth/demo, the
   demo user owns the account desk trades attribute to).
**Improver contained**: 226 stuck improver PRs closed (each carried a stale
.github/state snapshot that would REVERT live agent memory on merge; reward gate can't
validate whole-file LLM rewrites of trading logic — ml_breakout regression found in
review). continuous_improver.py now (a) never commits state files into PRs, (b) is
barred from strategies/execution/risk/ml.models/bots via _is_protected. Desk→Trades
attribution (PR #764) + Tradier delta-strikes (PR #762) landed earlier today.
USER ACTIONS pending: unpause Supabase (durable DB); add TRADIER_SANDBOX_TOKEN secret
(real greeks); optionally delete the stale agb8 Render service.

## Earlier: STATE AS OF 2026-07-18 PM
**THE QUEUE NOW DRAINS ITSELF**: improvements-worker.yml (event-chained, ~24h gate)
converts top unchecked IMPROVEMENTS.md items into deduped `agent-fix-needed` issues →
Free-Agent Engineer implements → reward-gated PR. Scouts append, worker drains — the
full loop runs with no session. Blocked items (user unlocks/missing feeds) are skipped,
never filed. OA backend analysis in docs/research/OA_BACKEND_STACK_2026.md: OA runs on
broker APIs (Tradier flagship; greeks are ORATS-via-Tradier) + a rules engine QuantEdge
already mirrors; queued the two real gaps — Tradier sandbox data adapter (free real
greeks/IV; needs TRADIER_SANDBOX_TOKEN) and SmartPricing laddered repricing.
Synthetic options backtester landed (app/backtest/options_synthetic.py, 9 tests).

## Earlier 2026-07-18 state
All queue P1s+P2s from the scale-up session are DONE except Polymarket CLOB (needs
POLYMARKET_PRIVATE_KEY wiring — queued). Landed since 07-15: auto-pruning (leaderboard
weight 0.0 at ≥20 trades/neg P&L/sharpe<-0.5, PLUS hit-rate rule ≥100 trades/losing/
win<45%); Symbol Scout (validates desk universes vs /v2/assets, proposes new pairs/ETFs,
runs in strategy-scout workflow); Options desk places REAL defined-risk mleg spreads for
wheel/condor/credit-spread/CSP/vol-carry (moneyness strikes ~35 DTE, never-partial-legs,
proxy fallback); research→registry pipeline (scout seeds strategy_generator's prompt via
state/research_seed.json).

## Previous state (2026-07-15 PM)
**SCALE-UP LANDED: 53→103 wired strategies, 9 desks.** New TV Indicators desk (12 TV
strategies, #desk-tv-indicators). Universes: Equities 30 syms, Crypto 20 pairs, Options 8,
Macro/FX 14 (full rates/credit complex), StatArb 12, Commodities 12, International 19.
Every wirable registry strategy now trades; the 10 leftovers are data-source-blocked
(documented in strategy_scout.KNOWN_EXCLUSIONS). **Strategy Scout** (workflow, event-
chained, 3-day gate) re-audits registry-vs-desks every run, posts to #desk-research,
appends to IMPROVEMENTS.md when new strategies appear, and rotates build-next ideas from
a 15-item research backlog — "always finding new strategies" is now automated.
**BOT_TEMPLATES 49→57**: added OA's flagship public bots (0DTE breakeven IC, short put
ladder, Black Swan hedge, 1-1-2, Rhino BWB, weekly diagonal, 16Δ strangle defined-risk,
Slow&Steady IC). Private leaderboard copies still blocked on OA_SESSION_COOKIE.

## Earlier 2026-07-15 state
**8 desks** (Commodities added: 8 ETF proxies, TSMOM/Donchian/MR) + OANDA FX desk with a
ForexFactory red-folder gate (stands aside ±30min around High-impact prints, fail-open).
**QUARANTINED is EMPTY** — all 115 registry strategies pass the contract suite via the
shared hard-budget fail-soft (`app/strategies/_failsoft.py`; yfinance uses curl_cffi, so
socket kills don't reach it — the daemon-thread budget is the only real guard).
**AlpacaBroker RESTORED**: improver PR #420 (Jul 9) truncated brokers/alpaca.py to a
"(truncated for brevity)" stub — 6 of 7 interface methods deleted, class un-instantiable,
backend silently on yfinance fallback. Also fixed the never-valid exception imports
(alpaca.common.exceptions). Improver now rejects shrunken/elided LLM outputs and skips
>8k-char files; `test_broker_interface.py` fails loudly if any broker goes abstract.
Cash-aware sizing live (#578): orders capped to 95% of relevant buying power, $25 floor.
Health hard gate: agent-health-check fails + pages #infra-alerts on critical findings.
Per-bot detail view (OA parity): P&L graph + open positions + trade history in BotBuilder.

## Previous state (2026-07-11)
**Everything runs event-driven 24x7** — 21 team/research workflows + keep-alive chained to CI completions (cron starved; that was THE autonomy bug). 7 desks (incl. new International, 13 country ETFs). Signal ensembling live (agree→boost, conflict→stand aside). 8 advanced strategies deployed. Claude backstop capped $1/day (state/claude_budget.json). Per-employee brains in agent_memory.json employee_context.

**🎉 FIRST TRADES PLACED 2026-07-12 07:59 UTC** — recovery flattened 22 orphaned positions (cash was -27,476) and the SAME run placed 2 real paper orders (UNI/USD + AAVE/USD, $755 notional, avellaneda_stoikov_mm conf 0.90/0.74). Desks are TRADING. Watch fill-tracking → Trades table → dashboard P&L now populate.

**Blocker history:** the 403 was 'insufficient balance for USD' — paper account overdrawn -$25,207, $0 available (orphaned notional buys nothing tracked). recover_negative_cash() in the desk now auto-flattens the orphaned PAPER book once (triple-guarded: paper URL assert, not-live, AUTO_FLATTEN_ON_NEGATIVE_CASH kill switch) — cash frees, next run trades. If Alpaca still refuses: reset the paper account in the dashboard (instant $100k).

**USER UNLOCKS (only manual steps left):** WORKFLOW_PAT secret (playbook exists) · OA_SESSION_COOKIE · Alpaca crypto check.

**DONE 2026-07-11 session 2 (7 queue items, PRs 488-494):** import_oa_bots.py · vol-targeted sizing (20% budget, 0.5-2x clamp) · money-path tests (9, mock Alpaca) · strategy contract test (115 audited, 18 violators found) · desk 10s per-strategy timeout (freezes impossible) · 5 violators fixed fail-soft + un-quarantined · OA public-scrape item was a wrong-domain false positive (optionalpha.com is auth-walled everywhere — cookie/Cowork only).
**DONE 2026-07-12 session 3 (PRs 496-517):** FX desk LIVE (OANDA practice, 7 majors 24/5, 8th desk) · fail-soft guards on 18 strategies, quarantine 18->7 · frontend-build CI gate (truncations now fail PRs, no Vercel dependency) · daily loss circuit breaker (2% vs prior close, stateless) · money-path silent-except sweep (trailing stops, HRP fallback, OCO cancels now logged) · basis_carry/funding_settlement_timer contract updates.
**TOP QUEUE NOW:** fix remaining 13 quarantined strategies (see backend/tests/unit/test_strategy_contract.py) · 887 broad-except sweep (brokers/execution/risk first) · OANDA FX desk · frontend component tests · workflow consolidation.
**Playbooks:** docs/playbooks/ (OA copy, bookmarklet, PAT, Alpaca). Verify a desk log + live-stats before claiming anything works.


## Mission
QuantEdge is an AI-first quant-trading company that must run **24/7**, cheaply, and
**self-improve**. `TRADING_MODE` stays `paper`. Never print secret values. Secrets live in
**Doppler** (single source of truth) and are injected as env vars.

## How tasks/memory persist (the source-of-truth map)
- **Canonical task queue:** GitHub Issues labeled `agent-fix-needed`.
- **Human board:** [Notion — QuantEdge Tasks](https://app.notion.com/p/bec54f8a79444c2399316365a07e0291)
  (mirrors GitHub Issues / `IMPROVEMENTS.md`).
- **Cross-session continuity:** **this file** + `IMPROVEMENTS.md` + `HANDOFF.md`.
- **Research (durable):** `docs/research/AI_COMPANY_SOTA.md`,
  `docs/research/LLM_COST_OPTIMIZATION.md`, `docs/MODEL_ROUTING.md`.
- **Slack:** notifications only — never the source of truth.
- **Agent runtime memory:** `.github/state/company_brain.json` (+ `llm_metrics.jsonl`).

## Resume procedure (what a fresh session should do)
1. The `SessionStart` hook prints this file + open `IMPROVEMENTS.md` items automatically.
2. `git log --oneline -10` to see what already landed.
3. Pick the top unchecked item in `IMPROVEMENTS.md` and continue.
4. Work on a branch off `main`; open a **draft PR**; never force-push.

## DONE (landed on `main`)
- ✅ LLM cascade revived — browser User-Agent fixes Cloudflare 1010 (#144).
- ✅ Key rotation across numbered free-provider keys (#145).
- ✅ Brain observability: `llm_metrics.jsonl`, `cascade_status()`, hourly `brain-health.yml`
  canary → Slack `#infra-alerts` (#146).
- ✅ Doppler single-source secrets (#139); Bot Archiver soft-delete/restore (#137).
- ✅ **2026-06-24 (11 PRs):** options productization (#188); brain cascade fixed on `main`+default
  branch incl. reasoning-model extraction + in-call key fallthrough (#189/#199); backend-health
  banner + Render runbook (#190); TradeStation spread routing (#198); Kalshi public reads (#203);
  guards for income/macro strategies (#202), momentum lookahead (#207), cross-tenant isolation (#208);
  pytest-asyncio deprecation removed (#206). Backend verified deploy-ready locally (158 routes, seeds
  29 bots). **Live blockers are human-only: Render build-minutes (#197) + default branch → `main` (#196).**

## 2026-07-07 — THE no-trades root cause found & fixed
- **Zero trades for days despite open markets** traced to ONE bug: 3 crypto
  strategies (basis_carry, funding_settlement_timer, mvrv_zscore_timing) did a
  module-level `import aiohttp`, which isn't in the desk workflow pip env. That
  crashed `from app.strategies import STRATEGY_REGISTRY` → desk_order_placer
  FATAL'd at line 690 on EVERY run (scheduled + event-driven), before placing a
  single order. One optional dep took down all 6 desks. Fixed: lazy aiohttp
  import at call sites + aiohttp added to desk deps + regression test
  (test_registry_import_safe). Merged #348.
- Gate race fixed (#324): auto-merge ignored ops-sync/Vercel non-gating checks;
  green PRs had stranded overnight. Improver now dispatches CI on its own PR
  branch (GITHUB_TOKEN fires no events) — closes the org perpetual-motion loop.
- 5 evidence-based hybrid strategies live (#303); 15 OA clones + factory + ML
  variant generation + synthetic-BS backtester scoring every options template.
- STILL MANUAL (1 paste): DISCORD_WEBHOOK_URL secret — the webhook works (204),
  it's just not saved as a GitHub secret, so automated Discord senders have no pipe.

## 2026-07-06 — org layer + honest data + the cron-starvation fix
- **Fake data purged:** live-stats hardcoded Sharpe 2.1/68%/14.7% → computed from real
  trades (null until data); tearsheet 500 (Postgres-only date_trunc) → DB-agnostic;
  Landing hero shows "—" until real metrics. Smoke gate (smoke-test.yml) verifies the
  DEPLOYED api twice hourly + post-merge and pages #ci-failures.
- **Google login 404 root cause:** OAuth callback redirected to cors_origins[0] = the
  dead quantedge.vercel.app stub. Now FRONTEND_URL-aware; render.yaml reordered.
- **Org layer:** team_lead.py (reviews automerge PRs — revokes label on protected-file
  edits (the #298 failure), assigns issues, reports to #leadership-summary) +
  data_team.py (bars coverage/freshness/ML-readiness for every desk symbol) +
  bot lifecycle manager (disable losers / promote winners / instantiate templates).
- **P&L loop closed:** desk_trade_sync (Alpaca fills → Trade rows, FIFO, idempotent);
  /leaderboard/live; perf-weighted desk sizing (winners 1.3x, losers 0.6x);
  /bots/{id}/performance (per-bot cum-P&L series).
- **Options desk:** wheel/iron_condor/credit_spread wired with iv_rank injection
  (verified firing); crypto desk 4 → 12 Alpaca pairs; India ETF sleeve (INDA/EPI/SMIN).
- **CRON STARVATION (critical infra fact):** GitHub schedules barely fire in this repo
  (keep-alive cron */5 = 162 runs EVER; discord-sync daily NEVER ran). Anything critical
  must be EVENT-driven → new ops-sync job in CI (test.yml) creates Discord channels,
  registers slash commands, and relays keys GitHub→Render on every CI run.
  VERIFIED 08:25Z: commands HTTP 200; DISCORD_BOT_TOKEN/GOOGLE_SERVICE_ACCOUNT_JSON/
  ALPACA keys → Render all HTTP 200. Bot was in NO server (root cause of empty
  Discord) — user re-invited it 08:2x.
- Hourly standup → Discord #engineering + self-provisioned "QuantEdge Standups" Google
  Doc (ensure_doc finds-or-creates + silent share; link posted in Discord, not email).

## 2026-07-05 — Discord two-way + pipeline self-heal
- Discord fully wired: webhook delivery (verified 204), per-employee bot profiles,
  per-channel routing, Discord-primary employee runs, shared `notify.py` chokepoint.
- **Slash commands** (`/status /pnl /health /run-bot`): Ed25519-verified endpoint at
  `/api/v1/discord/interactions` + `discord-commands-sync.yml` (needs DISCORD_BOT_TOKEN).
  Caught+fixed a real merged bug: equity lives on AccountSnapshot, not Account.
- **`auto-pr.yml`**: pushes to `claude/**` now auto-open an `automerge` PR + dispatch CI —
  closes the stranded-branch gap (needs repo setting "Allow Actions to create PRs" ON).
- Bars-fetch root cause fixed (#286): Alpaca defaulted `start` to today → every desk ran
  on an empty cache. Now 200d of real bars; 3 crypto strategies pruned (Binance 451 /
  CoinGecko 401 unreachable from US runners → issue #289 to reroute via Bybit/OKX).
- ML experiments VERIFIED working: Sunday run, real Alpaca data, honest OOS results (#288).

## 2026-07-04 — the big landing (12 PRs merged in one session)
- ✅ #270 commodity_reversion strategy + commodities research; #271 stale backend URLs → agb8;
  #272 keep-alive hardening; #273 website fixes (Google OAuth, WS revival, cold-start retry);
  #274 Slack fail-loud pipeline; #275 pydantic v2; #276 reward-gate all editors; #277 broker
  logging; #278 **ANTHROPIC_API_KEY wiring** (haiku default backstop, sonnet PR reviews,
  @claude mention agent, key-sync workflow); #279 **Discord failover** at all 3 posting
  chokepoints + models:read + weekly claude-queue-worker; #280 Option Alpha parity research.
- ✅ **Key sync VERIFIED live**: self-trigger hack fired on #278's merge — 1-token Haiku call OK,
  Doppler upserted, Render env var set (deploy call was 429 but the merge train's auto-deploy
  covers it). The sync re-runs whenever `sync-anthropic-key.yml` itself changes on main.
- ✅ #281 (`claude/stoic-johnson-7z4wtz`) **why-no-trades root fix**: BotRunner was NEVER
  instantiated (29 bots at runs=0) → ignition one-shot job in `tasks/scheduler.py`;
  `check_bot_exits` was never scheduled (positions would never close) → 5-min interval job;
  self-ping keeps the Render dyno awake; Option Alpha endpoints the frontend 404'd on
  (`/options/rules/validate`, `/flow`, `/put-call-ratio`, `/wheel`, `/macro-calendar`,
  `/next-fomc`); real order submit in Options.tsx; `automerge` drafts now auto-ready+merge.

## NEXT (see `IMPROVEMENTS.md` for the full list)
- Verify on prod after redeploy: bots show `last_run`/`run_count` > 0, `/api/v1/scheduler/jobs`
  exists, trades appear once markets open (Jul 6 Mon).
- ML experiments: torch can't run on the 512MB Render dyno — run training in GitHub Actions
  (CPU-friendly models) or a worker with the `[ml]` extra; nightly retrain job now actually
  fires (scheduler starts) but degrades without torch.
- Human-only: DISCORD_WEBHOOK_URL secret (GitHub + Render), UptimeRobot pinger, Alpaca keys
  rotation, delete the Vercel stub project (`quantedge.vercel.app`), Slack plan decision.
- Refresh dead free-provider keys in Doppler (only Groq works) → multi-provider resilience.
- Langfuse/OTel tracing on `llm_common`; pgvector memory (Mem0/Letta); verifiable-reward
  self-improvement gate; durable orchestration (Temporal); A2A protocol.

## Gotchas (don't relearn these the hard way)
- Cloudflare blocks default urllib UA with error 1010 → always send a browser `User-Agent`.
- pytest-xdist shares one DB → per-worker DB via `PYTEST_XDIST_WORKER` (already handled).
- bcrypt 5.x ≠ passlib 1.7.4 → use bcrypt directly (already handled).
- Scheduled workflows must `checkout` with `ref: main` (a CI test enforces this).
- Binance is geo-blocked (451) and Stooq unreachable in this env; yfinance is rate-limited.
