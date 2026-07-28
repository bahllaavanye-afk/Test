# CONTINUITY — read me first, every session

> **Purpose:** chat sessions are ephemeral and context resets when tokens run out. This
> file (committed to the repo) + the `SessionStart` hook in `.claude/settings.json` make
> every new/resumed session **auto-load the current state** so no memory or progress is
> lost. Keep it current: when you finish or start something material, update this file in
> the same commit.

_Last updated: 2026-07-27._

## 🟢 DURABLE POSTGRES — CLUSTER FIX CONFIRMED LIVE 2026-07-25 07:36 UTC (password is the ONLY blocker)
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

**Follow-up 07:20 — the two desk workflows were starving each other.** Both rode
`workflow_run: ["CI"] completed`, neither has a job-level market gate, and they use *different*
concurrency groups (`desk-trading` vs `desk-crypto`) so nothing serialises them. Every CI
completion launched both in parallel against the same free-tier Alpaca data API:
```
2026-07-28 06:46, one CI completion, two runs:
  desk-trading.yml          bars_fetched=70   ALL 20 crypto symbols
  desk-trading-crypto-24x7  bars_fetched=5    ← lost the race, HTTP 429
```
`desk-trading.yml` already runs **every** desk, crypto included, 24/7 — so the crypto-only run
was doing duplicate work, doing it worse, and degrading the twin doing it properly. Thin data is
not cosmetic: it feeds the ensembles, and this is very likely a large part of the standing
`sell/buy conflict — stand aside` on nearly every crypto symbol. The crypto job now skips
`workflow_run` (`if: github.event_name != 'workflow_run'`); its own cron + dispatch remain, which
is its real value for stretches when CI is not completing. 5 tests, verified failing pre-fix,
incl. one asserting the cron survives so "stop the duplicate" can't be confused with "delete
24/7 crypto coverage".

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
