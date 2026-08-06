# QuantEdge — Improvements & Task Tracker

## 🖥️ 2026-08-06 14:40 — THE EMPTY PANELS WEREN'T SILENT. THEY WERE BLAMING THE OPERATOR.

Shipped the `[P2] empty-state honesty` item. The item said pages "render an empty table with no explanation".
Checking first — the habit this session keeps earning — showed that was not the fault.

- [x] **Most pages already had empty states. Two were actively MISLEADING**, which is worse than blank:

        Experiments → "Click \"Train Model\" above to queue your first training run"
        Comparison  → "No comparison runs yet — run a strategy comparison in Backtest Lab"

  Both instruct an action that **cannot help**. Rows are wiped on every redeploy by the ephemeral SQLite
  fallback, and ML training needs PyTorch, deliberately excluded from the deployment image. Follow either
  instruction and you get the same empty table, having concluded you did something wrong. **A panel that
  attributes a platform cause to user inaction is the UI version of the defect this codebase keeps finding.**
- [x] **`EmptyState` names the cause instead** — four reasons (`ephemeral-db`, `no-rows-yet`,
  `subsystem-unreachable`, `ml-runtime`), house Tailwind style, optional page-specific note. The first draft
  used inline styles; rewritten to match the design system rather than import a second convention.
- [x] **The guard caught my own incomplete edit.** `Comparison.tsx` had the misleading string **three** times
  and my `replace(..., 1)` fixed one. `test_no_page_tells_the_operator_to_retry_something_that_cannot_persist`
  failed and named the survivors. **Same duplicate-anchor trap that produced three false mutations earlier
  today — this time a test caught it instead of me.**
- [x] **3 mutations, 3 caught**: restoring the misleading phrase, passing an undefined reason (which would
  render blank copy — the original bug), and renaming a reason out of the component. Plus a guard-on-the-guard
  pinning that the page scan finds ≥10 files, so a broken glob fails loudly rather than passing vacuously.
- [x] **Verified: `tsc --noEmit` clean, `vite build` succeeds, scripts suite 1440 passed.**
- [ ] **[P2] Remaining pages still render bare tables**, just without misleading advice — `Archive`,
  `Activity`, `Releases`, `RiskManager`, `AgentDashboard`, `MLInsights`. `EmptyState` exists for them now;
  applying it is mechanical. Left because two pages was enough to establish the pattern and prove the guard,
  and a large unverifiable UI diff is worth less than a small verified one.

## 🎯 2026-08-06 14:10 — MARKET-OPEN LIVE TEST: A DELIBERATE SKIP WAS LOGGED AS A BROKER FAILURE

First run with a working account and an open market in ~9 hours, so the first chance all session to exercise
the money path for real. It works — limits filling in 5-15s, market fallback after 20s, slippage measured in
single-digit bps. One defect fell out of the log.

- [x] **The desk reported its own correct decisions as failures.** Verbatim from run `31106632302`:

        · META sell 0.56 would be a fractional SHORT (held 0.0) — Alpaca rejects those,
          and flooring gives 0. Skipping instead of failing at the broker.
        ✗ order placement returned no ID

  The skip line explains a correct choice; the ✗ on the next line contradicts it. Twice in one run (META,
  TSLA). **`_place_order` returned `None` both when it CHOSE not to place and when placement genuinely
  failed**, and the caller could not tell them apart. Same defect class as the order-origin audit and the
  brain-health canary: the message did not depend on what actually happened.
- [x] **`SKIPPED_BY_DESIGN` now distinguishes them**, and the caller prints nothing extra for a skip because
  the reason was already printed. A genuine failure still says `✗ ... returned no ID` — pinned by a test, so
  the fix cannot swallow the case it was written for.
- [x] **A scan test found MORE skip sites than I did.** I fixed two; `test_every_deliberate_skip_inside_
  place_order_returns_the_sentinel` walks the function's AST and found **three**. The third was the market-order
  short-safety path. It also surfaced a **silent** `return None` on a missing crypto ask — no reason logged
  anywhere, so an unpriceable symbol read as a broker failure with no explanation in the entire run. Now
  announced.
- [x] **A guard I added was dead and mutation testing proved it.** An explicit `if _was_skipped(order)` early
  return in `_ensure_filled` could be deleted with the suite still green — the existing `not oid` check already
  passes the sentinel through. **Removed rather than kept**: a guard that cannot fail reads as load-bearing and
  is not. Replaced with a real mutation (break the passthrough) which the suite does catch.
- [x] **Two existing tests updated, and the change is a strengthening.** Both asserted `_place_order(...) is
  None`, which accepted a skip OR a failure indifferently — the very conflation being fixed. They now assert
  `_was_skipped(...)`, and both already carried the property assertion (`no POST was made`, `posted == []`)
  that proves intent is preserved.
- [x] **4 mutations, 4 caught**, with the mutation script asserting each anchor appears exactly once before
  applying — after three false mutations earlier today, uniqueness is now checked by the tool rather than by me.
- [x] **Suites: scripts 1435 passed, backend 2035 passed.**

## 🔑 2026-08-06 08:00 — THE IMPROVER REWROTE THE ALPACA CREDENTIAL PATH, AND CI HAD NOTHING TO SAY

Checked #1547 (`improve(optimization): autonomous run — 5 file(s)`) because it merged into main this cycle and
improver PRs bypass the gate. One of the five files was `backend/app/brokers/alpaca_headers.py` — where the
API keys are assembled.

- [x] **The change itself is fine, and I checked rather than assumed.** It wrapped the header builder in
  `@lru_cache(maxsize=1)` and returns a `.copy()`. I expected to find the `.copy()` made it *slower* and said
  so to myself first; measured over 2M calls it is **108ns vs 128ns — 16% faster**, so the optimization claim
  was true. Nothing mutates `settings` at runtime, so the cache cannot go stale in-process either.
- [x] **That it was benign is luck, not design, and the gap is the finding.** `backend/app/brokers/*.py` is
  declared Do-Not-Modify in **three** CLAUDE.md files (`tasks/`, `risk/`, and `strategies/options/` for
  `alpaca.py`) — but it was **not** in the improver's `PROTECTED_PREFIXES`, only in `CANDIDATE_PATTERNS`. The
  project's own rule and the improver's enforcement had drifted apart.
- [x] **`grep -rl alpaca_headers backend/tests/` returns NOTHING.** An autonomous whole-file LLM rewrite landed
  on the credential path with **zero test coverage**, and the green CI that merged it was not evidence about
  the change at all. That is the same shape as everything else today: a check whose output does not depend on
  the thing it appears to certify.
- [x] **`backend/app/brokers/` is now protected**, and pruned from the three `TYPE_TARGETS` entries that
  referenced it (`error_handling`, `constants`, `monitoring`) — because adding a prefix without pruning the
  targets would leave types pointing at patterns that can now only yield nothing, which is exactly the silent
  fallback fixed an hour ago. All 11 types verified to still resolve. 2 mutations, 2 caught.
- [x] **Trailing newline was stripped from the file** by the improver's rewrite. Cosmetic, not fixed here:
  `brokers/*.py` is Do-Not-Modify for me too, and the right move is to leave the money-path file alone rather
  than touch it for whitespace.
- [x] **~~[P2] `CANDIDATE_PATTERNS` still lists `backend/app/brokers/*.py`.~~ SHIPPED 2026-08-06 08:00 — and
  the reason I gave for deferring it was false.** I wrote "several existing tests index that list positionally".
  They do not: `grep` finds exactly one positional use in the whole repo, `continuous_improver.py:256`, in the
  legacy `improvement_type=None` branch. The entry came out in seconds. **I deferred my own item on a blocker I
  never checked** — the same failure as the queued items below whose text turned out wrong, except this one was
  mine and one day old.
- [ ] **[P1, OPERATOR] The real gap is unchanged: improver PRs still bypass CI.** Protecting one more
  directory narrows the blast radius; it does not fix the mechanism. A whole-file LLM rewrite still merges to
  main without the gate, and today it happened to be harmless.

## 🎯 2026-08-06 07:30 — HALF THE IMPROVER'S "TARGETING" WAS DECORATIVE, AND ONE OF ITS TYPES COULD NEVER RUN

Shipped the `[P2]` pairing item. Measuring it first changed what the fix had to be — the tracker's description
was directionally right and wrong in the specifics, twice over.

- [x] **The tracker said type and target were chosen "independently". Half true.** The type is
  `hour % 12`, fixed for a run; the pattern is `((hour + attempts) % 24) % 12`, rotating per attempt. So
  attempt 0 pairs type *i* with pattern *i* **deterministically**, and only the retries drift. Both lists were
  length 12, so the attempt-0 pairing was permanent and recurred every 12 hours:

        test_cases     -> backend/app/ml/models/*.py    unit tests written INTO model source
        strategy_logic -> backend/app/api/v1/*.py       route handlers have no entry/exit logic
        monitoring     -> backend/tests/unit/*.py       P&L logging added to unit tests

- [x] **The bigger finding was not in the item at all: 6 of the 12 patterns yielded ZERO usable files.** Once
  `PROTECTED_PREFIXES` and the 8000-char guard applied, `strategies/manual`, `strategies/ml_enhanced`,
  `ml/models`, `ml/features`, `execution` and `risk` were all empty. Those runs fell straight through to a glob
  over the whole backend — so on **half of all hours the per-hour targeting did nothing at all**, silently.
  That fallback is the mechanism behind `strategy_logic` reaching `models/account.py` (#1510).
- [x] **`strategy_logic` is now removed rather than re-pointed.** Its prompt only means something inside
  `backend/app/strategies/`, and that prefix is protected precisely because "behavior in the money-path is not
  the improver's to touch". **A type whose only legal target is off-limits is not a type** — it just spends one
  run in twelve aiming a strategy prompt at whatever it can reach.
- [x] **`TYPE_TARGETS` maps each type to locations where the change is meaningful**, every pattern verified to
  yield at least one usable file, and `test_every_improvement_type_has_a_live_target` **fails CI if that stops
  being true**. That is the part that matters: the old design could degrade silently, and now it cannot.
- [x] **The fallback announces itself.** A silent fallback is what let the targeting look like it was working.
- [x] **6 mutations, 6 caught — after two of them turned out not to be mutations at all.** My first M5/M6
  produced invalid Python, and pytest reported "2 errors", which is *not* a caught mutation: the suite never
  ran. Re-done properly, both were caught. **A mutation that breaks the file tests nothing** — the same trap as
  the skip-instead-of-fail one an hour ago, in a new costume.
- [x] **One existing test had to be rewritten, and that deserves scrutiny.**
  `test_both_candidate_lists_apply_the_filter` counted `_too_large` calls and demanded ≥2, one per candidate
  list. Both lists now route through one `_usable()` helper, so counting would demand duplication that no
  longer exists. Rewritten to assert the property directly — **no `glob()` result may reach the caller without
  passing through `_usable()`** — which is strictly stronger, because it also catches a *third* candidate list
  that the old count would have waved through. Rewriting a test to fit new code is how a guard gets quietly
  neutered; this one is stated so it fails for more cases than before, not fewer.

## 🔬 2026-08-06 07:00 — A GUARD BUILT SO A FLAKE COULDN'T FAIL CI WAS TURNING A FLAKE INTO A FAILURE

CI went red on a PR that touches only `.github/scripts` and a markdown file. The failing test was in the
backend, so it was not mine — but it was real, it was on main, and it blocked the merge.

- [x] **The failure named its own cause if you read the numbers.** `test_tearsheet_clean_404_when_no_trades`
  asked for a 404 on a user with no trades and got **200 with `n_trades: 5`**. The equity curve was verbatim
  the seed from the test two functions above it:

        +120, -40, +80, -20, +200   →   100120, 100080, 100160, 100140, 100340

- [x] **`auth_headers` falls back to `/api/v1/auth/demo` when the 10/min auth limiter trips**, and that
  endpoint is a **get-or-create on ONE shared user** (`demo@quantedge.app`, `auth.py:146`). Two tests that both
  trip the limiter become the same user, every account they create lands under it, and `_user_account_ids`
  then returns both — so the "no trades" test sees the other test's trades. The endpoint's own filtering was
  never at fault; it filters by account correctly.
- [x] **The fallback exists so a limiter artifact never turns the gate red, and for these tests it did the
  exact opposite.** It converted a limiter artifact into a *false failure* — the outcome it was written to
  prevent, arriving by a route nobody checked. **The theme again, third variant this session:** guards that
  cannot fail, a guard that cannot pass, and now a guard that causes the failure it exists to prevent.
- [x] **Fixed by making the substitution honest rather than removing it.** `auth_headers(isolated=True)` skips
  on a limiter trip instead of quietly borrowing a shared identity; the plain fallback is untouched for the
  many tests that only need *some* authenticated user, because deleting it would trade one flake class for
  another. Opted in for the three modules that assert on per-user data (`analytics_honest`, `bot_performance`,
  `bot_activity`), with a test that fails if a fourth is added without it.
- [x] **2 mutations, 2 caught — but only after the first attempt lied.** Flipping the default to `True` made
  the fallback test **skip**, and a skip is not a catch: it reads green while asserting nothing. The default is
  now pinned by `inspect.signature` so the mutation fails instead of vanishing. Same trap as the
  `pytest.raises(Skipped)` in the isolation test itself, which would otherwise have skipped *itself* and passed.
- [x] **~~[P2] The demo fallback is still reachable by any future test that forgets to say so.~~ SHIPPED
  2026-08-06 08:30 — and "a bigger change" was wrong, for the second time in a row.** I wrote that detecting
  per-user assertions automatically was too big for a tick. Measured: **exactly three integration modules seed
  per-user data, and all three already opt in.** A scan over `test_*.py` for `_seed_trades|Trade\(|Account\(`
  alongside `auth_headers` is precise with zero false positives — about ten lines. **That is now four stated
  blockers in two days that dissolved on contact**, three of them mine.

## 🧠 2026-08-06 09:30 — THREE CONFIGURED GROQ KEYS WERE EXCLUDED FROM EVERY LLM CALL

Chasing the LLM item from the last tick. **The guess was wrong and the worry was right**: the dead functions
are harmless superseded duplicates, but probing the live cascade found a capability bug behind a green canary.

- [x] **Brain-health run `31078219842` (06:42) reported `healthy: true` on ONE of eight providers.**

        gemini/sambanova/cerebras/groq/deepseek/together/hyperbolic → has_key: false
        nvidia_nim → ok, 18550 ms
        "working": ["nvidia_nim"],  "healthy": true

  `healthy = bool(working)` — at least one. The 03:32 run had already **failed outright**, so the cascade has
  been to zero once today and nothing about "BRAIN OK" conveys one-away-from-dark.
- [x] **Groq was not keyless. The same job's env block showed three keys.**

        GROQ_API_KEY:    (empty)
        GROQ_API_KEY_1:  ***      GROQ_API_KEY_2:  ***      GROQ_API_KEY_3:  ***

  `_provider_keys()` collects `_1.._3` and documents them as multiplied rate-limit headroom. **`_has_key()`
  checked only `key_env` and `key_env_alt`.** Two functions answering "does this provider have a key?" and
  disagreeing.
- [x] **NOT a reporting bug — a capability one.** `_call_parallel_race` builds the live cascade with
  `available = [p for p in _PROVIDERS if _has_key(p)]`, so Groq was excluded from **actual calls**, not merely
  from the report. The platform ran on one provider at 18.5s when it had a second with 3 rotating keys at
  30 rpm each. `_has_key` now delegates to `_provider_keys`, so the two answers cannot diverge again.
- [x] **3 mutations, 3 caught — after the third turned out not to be a mutation.** The anchor string
  `if v and v != "disabled" and v not in keys:` appears **twice** in the file, and `replace(..., 1)` hit the
  copy at line 435, in an unrelated function. The suite passed because **the code under test was never
  changed**. Re-applied inside `_provider_keys` only, it was caught. **Third false mutation this session**
  (after two that produced invalid Python) — the check is always the same: confirm the mutation landed where
  you aimed it, not merely that the suite reacted or didn't.
- [x] **LIVE VERIFICATION 2026-08-06 09:49, run `31090685392` on the fixed commit — and the timing is the
  point.**

        groq:       has_key: true,  keys: 3,  ok: true,   ms: 171
        nvidia_nim: has_key: true,  keys: 1,  ok: false,  error: "The read operation timed out"
        "working": ["groq"]

  Groq is detected with all three keys and answers in **171 ms — 108x faster** than the 18,550 ms nvidia_nim
  was taking. **And nvidia_nim timed out on this very run.** Without the fix this run had ZERO working
  providers: the cascade would have gone dark for the second time today. The fix landed roughly two minutes
  before the provider it replaced failed. That is luck, not design, and it is the argument for the two
  operator items below rather than a reason to feel good about the save.
- [x] **CORRECTION 2026-08-06 10:45 — the fix recovered TWO providers, not one.** I reported it as restoring
  Groq's three keys. Reading the full probe rather than the tail shows **Gemini was recovered too**:

        gemini: has_key: true,  keys: 3,  ok: false,  error: "HTTP Error 429: Too Many Requests"
        groq:   has_key: true,  keys: 3,  ok: true,   ms: 171

  Gemini read `has_key: false` before the fix and `keys: 3` after — the workflow passes `GEMINI_API_KEY_1/2/3`
  exactly as it does Groq's. So **six keys across two providers** were being ignored, and providers *with keys*
  went **1 → 3**. I under-reported my own fix by reading only the tail of the log; the correction is upward,
  but it is the same error as any other — describing evidence I had not fully read.
- [x] **Gemini's 429 is itself information.** The keys are valid and the free-tier quota is exhausted, which
  means **adding more Gemini keys buys little** — the useful move is a different provider, not another key on
  the same one. That sharpens the operator item below from "add a key" to "add a distinct provider".
- [x] **Still 1 of 8 ANSWERING, and `healthy: true` both times.** The canary reported the same verdict before the fix
  (one slow provider), after it (one fast provider), and would have reported `false` only when the count hit
  zero. **A health check that cannot distinguish "one provider" from "one provider, and the spare just died"
  is not measuring resilience** — it is measuring whether the platform is already broken.
- [x] **SHIPPED 2026-08-06 11:15 — the state is now REPORTED, while the alarm floor stays yours.**
  `cascade_status()` gained `single_point_of_failure` (true at exactly one working provider) and a `keyed`
  list, and the canary prints **`⚠ BRAIN AT RISK — only 1 of 3 keyed provider(s) answering`** instead of
  `BRAIN OK`. **The gate is deliberately untouched**: still `exit 0`, still no Discord page. Widening the alarm
  would make the operator's decision for them; saying "OK" over a one-deep cascade was simply false. Same
  split as the order-origin audit fix — correct the message, leave the gate. 5 mutations, 5 caught, including
  two that try to turn the at-risk path red or make it page.
- [ ] **[P1, OPERATOR] Whether at-risk should PAGE is still a policy question, deliberately not changed.** `healthy` means
  "≥1 provider answers". On a platform whose agents, improver and desk commentary all depend on the cascade,
  one provider is not health — it is a single point of failure that already failed once today. A floor of 2,
  or a warning below 3, would say so. Not shipped unattended: it changes when the infra alarm fires, and a
  noisier alarm is your call, not mine.
- [ ] **[P2, OPERATOR] Seven providers have no key at all.** All are free tiers. Adding even one more
  (`GEMINI_API_KEY` is the cheapest to obtain) removes the single-point-of-failure without any code change.

## 📉 2026-08-06 08:45 — THE "48% DEAD CODE" FIGURE WAS WRONG BY ~17×, AND SO WAS MY FIRST CORRECTION

`813 of 1,710 module-level functions never referenced (48%)` has been quoted since 2026-08-05 and was cited as
evidence the improver manufactures dead code *at scale*. The open item said **"sample before believing it."**
Doing that took two rounds, because the first correction was wrong too.

- [x] **The original figure counted TEST functions as dead.** Today `.github/scripts` holds **2,102**
  module-level functions, and **933 of them are `test_*`**. Pytest discovers those **by name**, so they have no
  call site anywhere and a naive reference scan marks every one dead. That alone is the bulk of the 813.
- [x] **My first re-derivation said 108, and sampling proved it inflated.** Scanning for the call pattern
  `name(` misses functions used as **first-class values**. Two of three sampled "dead" functions were live
  registry entries:

        claude_conversations.py:491   (call_cerebras,  "Cerebras Llama-3.1")
        agent_team.py:8348            ["desk-futures"], trading_desk_futures, ["futures", "trading"]

  **I sampled my own number and it failed the sample** — the same discipline the item demanded of the 813,
  applied to the correction, and it was needed.
- [x] **The defensible measurement: 23 of 815 unique production function names (2.8%)** have no reference of
  any kind. Method that survives scrutiny: parse every `.github/scripts/*.py`, collect module-level defs from
  NON-test files, then count `ast.Name` / `ast.Attribute` / `ast.alias` uses across **all** scripts including
  tests — identifier uses, not call patterns, so functions passed as values count. Then subtract anything named
  in `.github/workflows/*.yml` or appearing as a string literal (0 of the 23 were).
- [x] **Spot-checked 3 of the 23 and all three are genuinely dead**, including one where the only mention is a
  comment describing behaviour nothing invokes: `_PAID_CALLS_BLOCKED  # populated by call_claude() each time
  it's invoked` — `call_claude` has no call site.
- [x] **What this does and does not change.** The improver DOES manufacture dead code — #1510's two
  unreferenced methods on a SQLAlchemy model were verified directly, by reading the diff. But **813 never
  demonstrated scale**; it mostly counted tests. The improver-pairing fix (#1549) stands on the direct evidence,
  not on this number. **A headline figure that drove design decisions was never checked, for a day.**
- [x] **~~[P2] Five of the 23 are LLM plumbing.~~ INVESTIGATED 2026-08-06 09:30 — my stated mechanism was
  wrong and the underlying worry was right.** The dead functions are superseded per-provider duplicates in
  `agent_team.py`/`claude_conversations.py`; the live cascade is data-driven from `_PROVIDERS` and is complete
  at 8. **But probing it found a worse bug than the one I guessed** — see the entry at the top of this file.

## 🛡️ 2026-08-06 08:30 — A SCAN THAT FINDS NOTHING PASSES. SO THE SCAN NEEDED ITS OWN GUARD.

Replaced the isolation check's hardcoded three-module list with a scan over every integration module, so a
module added tomorrow is covered. The interesting part is what the scan itself required.

- [x] **The named list could not see a new module**, and a new module that seeds `Trade` rows and calls
  `auth_headers` without `isolated=True` is precisely the regression that reddened CI this morning.
- [x] **THE SCAN IS ITSELF A GUARD THAT CAN SILENTLY STOP WORKING.** If the regex or the glob breaks, it
  matches nothing, finds no offenders, and **passes while verifying nothing** — the exact failure this whole
  session has been about, reintroduced by the fix for it. So `test_the_scan_actually_finds_modules` pins a
  non-zero match count. Two of the three mutations below exist only because of it.
- [x] **3 mutations, 3 caught:** a module dropping `isolated=True` (the real regression), the regex matching
  nothing, and the glob finding no files. **Without the guard-on-the-guard, the last two would have passed
  silently** — a broken check reading as a green one.
- [x] **The scan is precise, not merely broad.** Measured: three modules match, all three already opt in, zero
  false positives. A check that flags everything is as useless as one that flags nothing; this one was worth
  measuring before shipping.

## 🧹 2026-08-06 06:30 — THE TRACKER WAS CARRYING TWO ITEMS THAT REALITY HAD ALREADY ANSWERED

Reconciled after the audit fix landed, because a tracker whose open items are stale is the thing that causes
rediscovery — the failure this file exists to prevent.

- [x] **`[P1] Bound the 5xx sweep's egress` was resting on a wrong diagnosis** and has been marked superseded
  rather than quietly deleted. The non-determinism was two twins disagreeing, not the network. Kept at `[P2]`
  on its remaining (weaker, honest) speed argument.
- [x] **`[P2] Extend the overnight-read pattern` shipped — and its own text contained a factual error.** It
  proposed Europe and wrote *"Europe, 15:30 UTC"* right next to the proposal, without noticing 15:30 is two
  hours AFTER the 13:30 US open. **A queued item is a hypothesis, not an instruction.** That error sat unread
  across several sessions because a filed item gets re-read as a decision rather than re-checked as a claim.
- [x] **The audit's alarm now names its candidates.** The old headline gave one — "a second writer is on this
  account" — and it read as an intruder because it offered nothing else to think. It now lists all three
  (`9jz0`'s PositionMonitor, the stale `agb8`, a hand-placed order), says two of them are ours, and states
  plainly that they cannot be told apart until backend orders carry a `client_order_id`. **A warning that
  names only its worst explanation will be read as its worst explanation.**
- [ ] **[P0] unchanged and now clearly load-bearing: tag backend orders.** Everything above is mitigation for
  the same root cause — `grep -c client_order_id backend/app/brokers/*.py` returns 0 for all eight files. Until
  that changes, "who placed this order" is answerable only by inference. Blocked: those files are Do-Not-Modify.

## 🚨 2026-08-06 06:00 — I RAISED AN INTRUDER ALARM. IT WAS OUR OWN RECOVERY FLATTEN.

I reported to the operator that "something placed orders on your account" and then, at the next tick, that it
had escalated to **50 of 50 orders**. Both readings came from this desk's own `audit_order_origins`, and the
audit was wrong — not in its counting, which was exact, but in the sentence it wrapped around the count.

- [x] **What the orders actually were.** All five sampled orders share a `submitted_at` of
  `2026-08-06T04:07:09.548–.550Z` — **five orders inside 2.1 milliseconds**. That is one server-side action,
  not five decisions. Run `31070310789` fired `recover_negative_cash` at 04:07, which calls
  `DELETE /v2/positions?cancel_orders=true`; Alpaca fulfils that by creating the closing orders **itself**,
  untagged by construction. The `TSLA buy qty=3` that looked out of place in a liquidation is a **short being
  covered** — `ichimoku_cloud_tv/TSLA` is short TSLA, and closing a short is a buy.
- [x] **The audit could never have been right.** `_order_origin` returns the honest label —
  `"EXTERNAL (backend exit loop / broker / manual)"`, and its docstring names all three — and then the headline
  collapsed it to *"a second writer is on this account."* Nothing we run tags anything except this placer:
  `grep -c client_order_id backend/app/brokers/*.py` returns **0 for all eight broker files**, so the backend's
  `PositionMonitor` exits are untagged too. **"Untagged" was being read as "foreign", and almost everything we
  own is untagged.**
- [x] **A guard that cannot tell our own recovery from an intruder will cry wolf on every ordinary flatten** —
  and it did, on every run, and I relayed it twice. This is the session's own failure mode arriving from the
  other side: the running theme has been *guards that cannot fail*, and this is a guard that **cannot pass**.
  Both are the same defect — a check whose output does not depend on the thing it claims to measure.
- [x] **The fix uses the one signal that does separate them: timing.** `bulk_burst_count` groups untagged
  orders submitted within 2.0s (≈1000× the observed 2.1ms spread). Orders inside a burst are reported as a
  close-all; only untagged orders *outside* one raise the alarm, with the wolf case still loud and still named.
  The real 5-order flatten is pinned verbatim as a fixture. **6 mutations, 5 caught** — the sixth is recorded
  below rather than hidden.
- [x] **A missing `import re` turned up because the suite RUNS the audit rather than reading it.** It would
  have crashed a live trading run: the fetch was fail-soft but the burst logic added after it was not. Now
  both are, with a test that injects a failure into the analysis itself.
- [x] **The recovery was also promising something it cannot deliver.** It printed *"cash frees as closes fill;
  next run trades normally"* unconditionally. Measured across runs `31070310789` (04:06) and `31072118909`
  (04:45): **the same 17 positions flattened twice, cash identical at -$48,471.29 to the cent, buying power
  $0.00 both times.** Closes submitted into a shut market sit `accepted` and free nothing, so the recovery
  re-fires every run until the open. The message is now conditional on the market state the caller already
  knows.
- [x] **Recorded honestly: mutation 3 survives on Python 3.11**, which every workflow pins, because 3.11's
  `fromisoformat` truncates nanoseconds natively and the shim is a no-op there. It is kept for older
  interpreters, where the raise would return `None`, make every order look isolated, and bring the false alarm
  straight back. Noted in the code rather than dressed up as a caught mutation.
- [x] **~~[P1] The account still cannot trade.~~ RESOLVED 2026-08-06 13:35 at the open, with no action taken.**
  Cash went **-$48,472 → +$21,916**, buying power **$0.00 → $87,662**, the book is flat and the desk placed 13
  orders. Total cost of the episode: **-$37.72 (-0.17%)**, and **the daily loss cap did not trip**. The recovery
  loop had already stopped itself at 13:08 once buying power turned positive — its `bp > 0` guard is precisely
  the check that distinguishes ordinary margin from the orphaned-notional pathology, and it worked.
  **I over-weighted the severity.** The hazard is real and documented (it fired on 2026-07-27), so flagging it
  was right; predicting the flatten "realises losses into the first prints" and could freeze the desk was the
  part reality did not support. Recorded because a warning that lands badly is worth as much scrutiny as a
  guard that reports wrongly — and this session has spent a day on the latter.
- [x] **The order-origin fix verified in production in the same run**: it printed
  `ⓘ ... all of them arrived in bulk bursts ... Not evidence of a third-party writer` rather than the intruder
  alarm. First live confirmation since #1539.
- [ ] **~~[P1] superseded~~ — original text kept below for the record.** equity $21,801.52, cash
  -$48,471.29, buying power **$0.00**, 17 positions queued to flatten at the open. The desk placed 0 orders
  with `reason=account_unavailable`. The flatten will realise losses into the open, which is exactly the
  documented "buy on margin, get liquidated, get frozen" path at `desk_order_placer.py:583`. **Operator
  decision**, because unwinding it deliberately is not something to do unattended.

## 🎯 2026-08-06 05:30 — THE CI NON-DETERMINISM WAS NOT THE EGRESS. TWO TWIN GUARDS DISAGREED.

Filed as `[P1] bound the 5xx sweep's egress` and deferred three times as too consequential for a tick. Reading
the code before building that turned up the actual cause, which needed no egress change at all.

- [x] **The two sweeps applied different criteria to the same question.**

      test_no_parameterised_get_endpoint_returns_5xx   →  if _is_real_server_error(r)
      test_no_get_endpoint_returns_5xx                 →  if r.status_code >= 500

- [x] **`_is_real_server_error` exists precisely for this**, and its docstring already said so: a **502/503
  carrying a structured `detail`** is a *handled* upstream outage — `{"detail": "Alpaca bars error: 401 ..."}`
  when the broker rejects our credentials — while a bare 500 is the unhandled exception the sweep was written
  for. Its twin has always used it. The parameterless one never did.
- [x] **That is the whole non-determinism.** With CI's env the identical command failed both sweeps in 5.9s on
  one run and passed 28–32s on three others: when Alpaca answered with a handled 502, the parameterless sweep
  called it our bug. **A guard whose verdict a third party decides is not a guard** — and the fix was to apply
  a helper that was already there, not to restructure the test.
- [x] **Both twins now share the criterion, and a test pins that they cannot drift again**
  (`test_both_5xx_sweeps_use_the_same_criterion`), plus one pinning the helper's contract directly. 2 mutations,
  2 caught.
- [x] **The deferral was right, and so was eventually reading the code.** Three ticks of "this deserves a
  deliberate change" kept a risky restructure out of unattended hours; the deliberate look then found a
  two-line fix. **Deferring is not the same as dropping.**
- [ ] **[P2] The egress item survives, reduced.** The sweep still makes real outbound calls, so its *runtime*
  stays load-dependent (1.4s without credentials, 14.9s with) even though its *verdict* no longer is. That is
  a speed and cost argument now, not a correctness one — much weaker, and correctly lower priority.

## 🌏 2026-08-06 05:10 — THE OVERNIGHT READ NOW COVERS SIX MORE MARKETS, AND EUROPE IS EXCLUDED ON PURPOSE

The `[P2]` item filed an hour ago, shipped. The NSE machinery is generic — a foreign close, a bounded
confidence nudge, a staleness check, a US-listed proxy that can actually be routed — so extending it is rows in
a map rather than new mechanism.

- [x] **The property that decides membership is the close time, and I checked it before extending.** Every
  source must close BEFORE the 13:30 UTC US open, or the read cannot inform an order at the open:

        Taiwan  ^TWII  05:30  -8h00      Korea      ^KS11  06:30  -7h00
        Japan   ^N225  06:00  -7h30      Hong Kong  ^HSI   08:00  -5h30
        Austral ^AXJO  06:00  -7h30      Singapore  ^STI   09:00  -4h30
        India   ^NSEI  10:00  -3h30

- [x] **EUROPE IS DELIBERATELY EXCLUDED, and this is the part I would have got wrong by extending blindly.**
  DAX and CAC close 15:30 UTC, the FTSE 16:30 — **two to three hours AFTER the US open**. A European close
  cannot inform an order at the open. The desks trade `EWG`/`EWQ`/`EWU`, so the temptation is real; the map
  carries the reason and `test_europe_is_not_in_the_map` enforces it.
- [x] **Weights are correspondence, not enthusiasm.** A country index and its MSCI tracker hold the same market
  with different weighting → 0.9. **`^HSI → FXI` is 0.7**: FXI holds China H-shares while the Hang Seng is a
  Hong Kong index — related, not the same market. Same honesty as `SMIN`'s 0.6 against a large-cap Nifty.
- [x] **A latent bug the extension forced out: the session close was one hardcoded constant** (10:00, NSE's).
  Applied to Taiwan's 05:30 close it overstates the read's age by 4.5 hours — forgiven by a 30h window,
  wrong on its own terms, and a trap the moment that window tightens. Now per-market, with half-hours preserved
  (Korea 06:30, Taiwan 05:30) and single names inheriting their home market via exchange suffix — without
  which every ADR silently fell back to 10:00, correct for India by luck.
- [x] **8 new tests, 5 mutations, 5 caught** — adding Europe, collapsing to one close constant, truncating
  half-hours, dropping the suffix lookup, and over-weighting the Hong Kong link.
- [x] **The live run caught a bug the extension itself introduced, which is the best argument for running it.**
  At 05:15 UTC it produced `EWT -0.0107 <- ^TWII -0.60%` — from a **Taiwan session with 15 minutes left to
  trade**. The future-date guard tolerated ±1 hour, which was harmless while NSE (10:00) was the only source
  and the workflow ran at 10:20; with markets closing 05:30-10:00 it silently admitted **partial sessions**.
  An intraday snapshot reported as a close is worse than no read, because it is indistinguishable from a real
  one. Tolerance is now clock-skew only (0.1h) and the message says `session … has not closed yet (0.2h to
  go)`. Two more tests, both mutation-verified.
- [x] **A tooling trap worth recording: `__pycache__` made a mutation test lie.** After restoring a mutated
  file, the module still imported the mutated constant — the source read `0.1` while `ins.CLOCK_SKEW_TOLERANCE_H`
  reported `1.0`, and a test passed alone but failed in the full suite. **Mutation testing writes bytecode;
  clear `__pycache__` or set `PYTHONDONTWRITEBYTECODE=1` between rounds.** Stale cache biases toward mutations
  appearing to *survive*, so it costs false alarms rather than false confidence — but it cost real time here.
- [ ] **[P3] The module is still called `india_nse_signal.py` and now covers seven markets.** Renaming it would
  also mean renaming the committed state file the desk reads by path, mid-flight, for a cosmetic gain. Left
  deliberately, recorded so the next reader knows the name is historical rather than a mistake.

## ✂️ 2026-08-06 04:40 — A PERMANENT, IRREVERSIBLE RETIREMENT MADE ON UNAUDITABLE EVIDENCE

The ML `beats` flag got a noise floor two hours ago. The same statistical error lives in the path that retires
**live** strategies, where it costs capital rather than a table cell. Every retirement in
`strategy_trims.json`, with each win count tested against a coin flip:

    strategy                 n  wins   P(<=wins | 50%)   strength
    avellaneda              10     6            0.828    indistinguishable from chance
    vol_of_vol              10     2            0.055    weak
    realized_vol_asymmetry  10     3            0.172    weak
    options_pcr_reversal    11     2            0.033    strong
    stat_arb_e              11     0            0.000    strong

- [x] **`avellaneda` won 6 of 10 trades and was retired permanently.** Its rule was a *magnitude* one
  (`cumulative return -7.9% <= -5.0%`), and win rate is admittedly the wrong test for that — a strategy can win
  often and still lose money on a few large losers, which is a legitimate reason to retire. **The problem is
  that nothing in the record could tell those apart.**
- [x] **`fill_tracker` recorded no dispersion whatsoever** — `trades, wins, win_rate, avg_return_pct,
  total_return_pct` and nothing else. A strategy bleeding -0.79% every trade and one that was flat nine times
  and lost 8.4% once produce **the identical `total_return_pct` at the identical n**. The trimmer could not
  distinguish them, and neither could a human reading the reason.
- [x] **And the decision is self-sealing.** Nothing ever removes an entry from `strategy_trims.json` — the
  trimmer only appends — while a retired strategy places no orders. So it can never accumulate the evidence
  that would overturn its own retirement. **Unauditable and irreversible is a bad pair.**
- [x] **SHIPPED — the record is now auditable.** `fill_tracker` carries a running `sum_sq_return_pct` (no
  per-trade history needed: `stdev = sqrt(E[x^2] - E[x]^2)`) plus `worst_trade_pct`, and the trim reason
  reports them. The two cases now read differently, which was the entire point:

      one bad trade:  -7.9% over 10 trades  [worst single trade -8.40% EXCEEDS the net loss
                                             — the other 9 were net positive, stdev 2.61%]
      steady bleed:   -7.9% over 10 trades  [worst single trade -1.20% = 15% of the loss, stdev 0.31%]

- [x] **Thresholds deliberately unchanged, and a test pins that.** What gets retired is capital-allocation
  policy; making a decision auditable must never silently alter it.
  `test_this_changes_REPORTING_not_WHICH_strategies_are_trimmed` fails if it does. 7 tests, 4 mutations, 4
  caught. Records written before today degrade to the old wording rather than inventing numbers.
- [ ] **[P1, OPERATOR] The policy question this exposes, now that it can be asked properly.** Two choices, both
  yours: (a) require a magnitude-rule retirement to clear a dispersion bar — a loss concentrated in one trade
  is a tail event, not a bleed; and (b) give retirement a **TTL or a re-audition path**, so a strategy retired
  on n=10 is not silenced forever by evidence that was never conclusive. Neither is shippable unattended: (a)
  keeps a losing strategy trading longer, (b) resumes a retired one. Both move real money.

## 🗺️ 2026-08-06 04:40 — FORWARD QUEUE, written from this session's evidence rather than from ideas

Everything below is grounded in something measured today. Ordered by value per unit of risk.

### Tractable in code, not yet done
- [x] **~~[P1] Bound the 5xx sweep's egress.~~ SUPERSEDED 2026-08-06 05:30 — the diagnosis was wrong.** The
  non-determinism was never the egress: the two sweeps were applying *different criteria* to the same response,
  one using `_is_real_server_error` and its twin a raw `>= 500`. Both now share the helper, so the verdict no
  longer depends on what Alpaca returns. **The egress item survives only as a speed/cost argument** (~25s of CI
  time), which is a much weaker case — re-filed at `[P2]` above rather than left at `[P1]` on a rationale that
  no longer holds. Deferring it three times is what created the room to read the code first.
- [x] **~~[P2] Pair the improver's improvement TYPE to plausible file paths.~~ SHIPPED 2026-08-06 07:30** —
  and the item under-described the fault. See the entry at the top of this file.
- [x] **~~[P2] Extend the overnight-read pattern to other markets.~~ SHIPPED 2026-08-06 05:10 (#1533)** — seven
  markets: India, Japan, Korea, Taiwan, Hong Kong, Australia, Singapore. **This item's own suggestion was wrong
  and shipping it honestly meant contradicting it.** It proposed `EWG`/`EWQ` (Europe) and even wrote the reason
  down — *"Europe, 15:30 UTC"* — without noticing that 15:30 is **two hours AFTER the 13:30 US open**, so a
  European close cannot inform an order at the open. Europe is now excluded by `test_europe_is_not_in_the_map`.
  The weights guidance was right and was followed (`^HSI → FXI` is 0.7, not 0.9: FXI holds China H-shares while
  the Hang Seng is a Hong Kong index). **A queued item is a hypothesis, not an instruction** — this one carried
  a factual error through several sessions because nobody checked its arithmetic before it was time to build.
- [ ] **[P3] Give `verdict` a fourth state for "the window is too short to judge".** Currently a sub-window
  under 30 days returns nothing at all, so a symbol with thin history silently has no breakdown rather than an
  explicit "insufficient". Same distinction the India feed already makes between absent, stale and flat.

### Measured and deliberately NOT built
- [x] **A repo-wide dead-code gate.** 813 of 1,710 functions flagged (48%); a gate that fires on half the
  codebase gates nothing. Per-feature call-site guards instead — used ~6 times this week.
- [x] **A vacuous-guard sweep beyond the one file.** All seven other scan-based tests derive their root from
  `__file__`; the defect was isolated. A recurrence guard shipped instead of ceremony in seven files.
- [x] **Tightening five of the six monitor entries that accept a 4xx.** 401/403/451 *prove* the host answered.
  Only Polymarket's 404 was wrong, and only because its root returns 200.

### The pattern worth carrying into the next session
- [x] **Today's highest-value finds were not broken code — they were guards that could not fail.** Four
  independent instances: two `test_regression.py` scans with hardcoded absolute paths (one hiding a live
  violation), a frontend uptime monitor pointing at a stub *and* accepting 404, a schema-drift gate that has
  reached a verdict once in 92 runs, and an order-origin report that only ran in a corner nobody reaches.
  **The question that finds them is "under what input does this fail?" — asked of the check, not the code.**
- [x] **The mirror-image rule, learned twice in one day.** A test that locates source by path must derive it
  from `__file__`: a *relative* path breaks outside the repo root, an *absolute* one silently disables the scan
  inside CI. Both failure modes are silent, and they point in opposite directions.
- [x] **A degenerate fixture cannot test a threshold.** Two mutations survived the noise-floor work because the
  fixture used `strat = bench.copy()`, making the margin *exactly* zero — so both `margin > 0` and
  `margin > floor` passed. Thresholds need small-but-nonzero cases on both sides.

## 🔎 2026-08-06 03:30 — THE VACUOUS-GUARD SWEEP CAME BACK NEGATIVE, AND THAT IS THE USEFUL RESULT

`test_regression.py`'s hardcoded absolute path made two guards pass on an empty scan. The obvious next question
is how many others do it. Answer: **none.**

- [x] **Every other scan-based test derives its root from `__file__`.** Checked all seven:
  `test_security_invariants` (parents[2]), `test_logger_kwargs`, `test_exit_path_wiring`,
  `test_risk_gate_wiring`, `test_walk_forward_coverage`, `test_alembic_single_head`,
  `test_agent_memory_is_bounded`, `test_fleet_liveness`. The defect was **isolated, not systemic**, so no
  ceremony was added to seven files for a risk that construction already prevents.
- [x] **My first heuristic was wrong and would have produced busywork.** It flagged 8 tests for "no scan guard",
  including `test_app_state_risk_manager_is_actually_assigned` — which asserts `assert found, …`, i.e. it
  *fails* on an empty scan. The dangerous shape is asserting **emptiness** (`assert violations == []`), not
  non-emptiness. Re-running with that distinction found 10 candidates and all 10 were already guarded.
- [x] **Shipped the recurrence guard instead:** `test_no_test_hardcodes_an_absolute_path_to_the_repo`. It is the
  durable half — the sweep proves today is clean, the guard keeps it that way. Mutation-verified by putting
  `Path("/home/user/Test")` back into `test_regression.py`, which turns it red.
- [x] **Its first draft had a false positive worth recording.** Scoped to every `.py` under `.github/scripts`,
  it flagged `agent_team.py`'s secret-redaction regexes — `r'/home/[^\s]+'`, patterns whose entire job is to
  FIND such paths and strip them from agent output. **A pattern that matches the defect is not the defect.**
  Now scoped to `test_*.py`, which is what the rule actually says.

## 📢 2026-08-06 02:45 — THE SUB-WINDOWS WERE COMPUTED, PERSISTED, AND INVISIBLE WHERE THE DECISION IS MADE

I built the per-period split at 20:20 and the noise floor at 02:00, then checked who actually reads them. The
`$GITHUB_STEP_SUMMARY` — the artifact a human opens after a run — printed **only the overall Sharpes**.

- [x] **Worse than invisible: it closed by stating a promotion rule on the misleading number.**

      "A strategy only earns promotion when its OOS Sharpe beats buy-and-hold"

  That is the exact criterion the sub-windows disproved twice in one evening, written as policy in the one
  place a reader would act on it.
- [x] **Fixed: a per-period table in the summary** with `margin`, `noise_floor` and `verdict` per window, so
  the aggregate can no longer be read alone. Rendered against the real 00:13 run to confirm it reads correctly
  — QQQ shows `beats / beats / loses` and NVDA `inconclusive / loses / beats`, which is the whole point.
- [x] **The promotion bar is rewritten to what the evidence supports:** an OOS Sharpe above buy-and-hold is
  *necessary and not sufficient* — it must clear the noise floor and hold across periods — with QQQ's
  1.36-vs-0.73 cited inline as the counterexample.
- [x] **A run too short to split says so** (`too few out-of-sample days to split`) rather than printing an empty
  section, since an empty table is indistinguishable from a broken one.
- [x] **This is the same failure family as the rest of today, one layer further out.** The guards that could
  not fail were *checks* nobody could trip; this was a *measurement* nobody could see. Computing the right
  number is only half of it — the other half is putting it where the decision happens.

## 📏 2026-08-06 02:00 — A NOISE FLOOR ON `beats`, AND IT CORRECTS MY OWN 01:20 CLAIM

The 01:20 entry logged `[P2] beats needs a magnitude floor` and then, three paragraphs earlier, **used the
unfloored flag to make a recommendation**. Shipping the floor re-judges the same run and moves 3 of 9 verdicts.

- [x] **`sharpe_noise_floor(n) = 2·sqrt(2/n)`** — a Sharpe from n daily returns has SE ≈ `sqrt(1/n)`; the
  difference of two has ≈ `sqrt(2/n)`; two of those is the usual "not noise" bar. **≈0.145 over a 382-day
  sub-window**, tightening as the window grows. Deliberately an approximation: it ignores strategy/benchmark
  correlation (making it conservative) and fat tails (making it optimistic). It exists to stop near-zero
  comparisons being reported as evidence, not to be a significance test.
- [x] **`verdict` is now three-state** — `beats` / `loses` / `inconclusive` — and `beats` requires the margin to
  clear the floor, so a caller tallying it is not counting noise. `margin` and `noise_floor` are both in the
  payload, because a verdict the reader cannot check is just an assertion.
- [x] **Re-judging run `2026-08-06T00:13`, 3 of 9 windows change:**

  | window | margin | floor | was | now |
  |---|---|---|---|---|
  | SPY w1 | +0.015 | 0.145 | beats | **inconclusive** |
  | SPY w3 | −0.100 | 0.145 | loses | **inconclusive** |
  | NVDA w1 | +0.136 | 0.145 | beats | **inconclusive** |

  The decisive ones survive untouched: QQQ w1 (+1.881), QQQ w3 (−0.699), SPY w2 (−0.613).
- [ ] **CORRECTION to 01:20.** I wrote that in the most recent 18 months "the model is behind on 2 of 3", and
  built the recommendation on it. Under the floor that window reads **QQQ decisively loses, NVDA decisively
  beats, SPY inconclusive** — 1 loss, 1 win, 1 unknown. The honest statement is narrower: **QQQ's edge has
  clearly decayed**; the three-symbol picture is mixed, not adverse. The direction of my advice (do not wire ML
  into live orders yet) still holds, but on thinner evidence than I gave it.
- [x] **Two mutations survived the first pass, and both were the bug itself** — `margin > 0` for `beats`, and
  `margin < 0` for `loses`. My fixture used `strat = bench.copy()`, so the margin was *exactly* zero and both
  mutants passed. **A degenerate fixture cannot test a threshold.** Fixed with small-but-nonzero margins
  (±3e-5 offset → ±0.05 margin against a 0.145 floor), which is precisely the real SPY case.

## 📉 2026-08-06 01:20 — THE SUB-WINDOWS LANDED AND THEY ARGUE **AGAINST** WIRING ML INTO ORDERS

Run `2026-08-06T00:13`, the first carrying `sub_windows`. This is why the split was worth building: the
headline number and the period breakdown point in opposite directions.

| symbol | overall | 2022-01→2023-07 | 2023-07→2025-01 | 2025-01→2026-08 |
|---|---|---|---|---|
| SPY | 0.606 vs 0.788 loses | 0.102 vs 0.087 "beats" | 1.028 vs 1.641 loses | 0.982 vs 1.082 loses |
| QQQ | **1.360 vs 0.729 beats** | 2.057 vs 0.176 beats | 1.697 vs 1.202 beats | **0.390 vs 1.089 LOSES** |
| NVDA | **1.196 vs 1.125 beats** | 1.000 vs 0.864 beats | 1.268 vs 1.453 loses | 1.508 vs 1.153 beats |

- [x] **QQQ's "1.36 vs 0.73" is carried by the 2022 bear market.** Its first sub-window is 2.057 vs 0.176 —
  an enormous beat while the benchmark was flat on its back. In the **most recent** window it does the
  opposite: **0.390 vs 1.089**, a clear loss. A single Sharpe over the full span hides a reversal.
- [x] **The pattern is consistent with the defensive-model hypothesis and now measured rather than asserted:**
  the edge concentrates where the benchmark suffers, and decays as the benchmark recovers. SPY loses in the
  two recent windows; QQQ loses badly in the latest; NVDA is the only one still ahead recently.
- [ ] **So the honest recommendation flips.** Last night's headline read as "beats buy-and-hold on 2 of 3, look
  further." With the breakdown, **the most recent 18 months show the model behind on 2 of 3**, which is an
  argument against routing live orders through it, not for. The overall figures were never wrong — they were
  just the wrong statistic to decide on.
- [x] **~~[P2] `beats` needs a magnitude floor.~~ ALREADY SHIPPED** — found stale 2026-08-06 07:30.
  `ml_experiment.py:234` computes `beats = margin > floor` against `sharpe_noise_floor(n)`, and `verdict` has
  the explicit third state this asked for: `inconclusive` when the margin sits inside the floor. The item was
  describing work that had already landed.
- [x] **This is what the 22:45 integration test was protecting.** Had the unmasked-index mutation shipped,
  every date above would be shifted by about a year and this table would have been quietly wrong while looking
  entirely reasonable.

## 🧱 2026-08-06 01:00 — A MIGRATION THAT CANNOT APPLY TO POSTGRES, AND IT BLOCKS OPERATOR ITEM #2

The Schema Drift Gate failed on my PR. It is not my change — my diff touches four lines of dead test code in
`models/backtest.py` and nothing in `models/` columns or `alembic/`. It is a **pre-existing migration bug**,
and it matters much more than the gate failure.

```
backend/alembic/versions/d4e5f6a7b8c9_create_bots_table.py:32
    sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("1"))

psycopg2.errors.DatatypeMismatch:
    column "is_enabled" is of type boolean but default expression is of type integer
```

- [x] **`sa.text("1")` is SQLite idiom.** SQLite stores booleans as integers and accepts it silently; Postgres
  is strict and rejects the DDL outright. The migration therefore **cannot apply to Postgres at all**.
- [ ] **[P0] This changes the plan for operator item #2.** "Unpause Supabase" has been the standing fix for the
  ephemeral-sqlite problem — but the app has been running on the SQLite fallback the whole time, which is
  exactly the engine that tolerates this default. **Unpausing Supabase is not sufficient**: the migration chain
  will fail on this column when it runs against Postgres. Worth knowing before spending the change window on it.
- [x] **Swept for siblings: exactly one occurrence.** No other `server_default=sa.text("0"|"1")` on a Boolean
  anywhere in `backend/alembic/versions/`. So this is a single-line fix, not a migration audit.
- [ ] **[P0] The fix, NOT shipped — `backend/alembic/versions/` is under "Do NOT Modify" in
  `scripts/CLAUDE.md`.** The correct change is `server_default=sa.true()`, which SQLAlchemy renders per
  dialect (`TRUE` on Postgres, accepted by SQLite ≥3.23), rather than hardcoding either engine's literal.
  Editing an applied migration is also a judgement call that belongs to a human — though note Postgres has
  never successfully applied this one, since it errors.
- [x] **And the gate that caught it has essentially never run: 92 runs, and this is the only one that ever
  reached a verdict.** The other 11 recent runs are all `action_required`, i.e. they never executed. So the
  drift has been sitting there undetected not because the check was missing, but because it was never allowed
  to run — **the third instance today of a guard that exists and does not execute**, after the two hardcoded
  absolute paths in `test_regression.py`.

## 🎲 2026-08-06 00:40 — THE 5xx ENDPOINT SWEEP LETS A THIRD PARTY DECIDE WHETHER CI IS GREEN

CI went red on `test_no_get_endpoint_returns_5xx` and its parameterised twin. Diagnosing it took three wrong
suspects, all measured and discarded, which is the useful part.

- [x] **Not my change.** It touched a regression test's path resolution and dead test code in a model.
- [x] **Not the improver, though it looked exactly like it.** This was the first CI run whose base included
  #1510 (main's last green CI at 23:34 predates it — checked with `merge-base --is-ancestor`), and #1510 added
  a **new GET route** (`/signal_quality`) that the sweep now walks. Measured by disabling that route and
  re-timing: **14.2s + 10.8s without it, 14.9s + 11.1s with it — a ~1.1s difference**, not the 45s needed.
- [x] **The mechanism: CI supplies real-looking Alpaca credentials** (`ALPACA_API_KEY="test"`), so every
  broker-backed route in the sweep makes a genuine outbound call to `paper-api.alpaca.markets` and waits for
  it to fail auth. With no keys locally the first test takes **1.4s**; with CI's env it takes **14.9s** — a
  10x swing driven entirely by egress. Runtime is network-bound, and the global `--timeout=60` left a margin
  CI exhausted.
- [x] **And the outcome is non-deterministic, not just the duration — which is the worse half.** One local run
  with CI's exact env failed both tests in **5.9s**; three consecutive runs of the identical command passed in
  28–32s. A 5xx from the upstream host propagates through the route, and the sweep cannot distinguish that
  from a 5xx we caused. **A third party currently decides whether this repo's CI is green.**
- [x] **Shipped: `@pytest.mark.timeout(240)` on both**, with the measurements in a comment so the next reader
  does not re-derive them. Deliberately not removing the egress — exercising these routes *with* credentials
  is the point, since it proves a broker auth failure surfaces as a handled error rather than a 5xx, the exact
  class the scanner-500 bug fell into.
- [x] **~~[P1] The real fix: bound the egress.~~ SUPERSEDED** — duplicate of the item reconciled at the top of
  this file, and resting on the same wrong diagnosis. The CI non-determinism was two twin sweeps applying
  different criteria, not the third party. Bounding the egress survives only as a ~25s speed argument at
  `[P2]`. **This is the second copy of one item found stale in a day**, which is the argument for reconciling
  the tracker as work lands rather than in a sweep later.
- [x] **Also worth recording: two of my background suite runs reported 510 and 1537 errors and both were my own
  artifacts** — I was editing app modules (`ml_model.py`, `improvements.py`) mid-run to mutation-test. A
  concurrent edit to a widely-imported module produces mass collection errors that look like catastrophic
  breakage. **Do not run the full suite in the background while editing.**

## 🚨 2026-08-06 00:05 — TWO REGRESSION GUARDS HAVE NEVER RUN IN CI, AND ONE HAD A LIVE VIOLATION

`test_no_datetime_utcnow_in_source` failed the local backend suite tonight while CI stayed green. I had noted
that as "a local/CI path discrepancy" earlier today and moved on. It is not a discrepancy — it is a dead guard.

```python
backend_dir = Path("/home/user/Test/backend/app")        # hardcoded ABSOLUTE path
for py_file in backend_dir.rglob("*.py"):                # no such dir on a runner
    ...
assert violations == []                                  # vacuously true
```

- [x] **On a CI runner that path does not exist, so `rglob` yields nothing and the assertion passes on an empty
  list.** Both `TestDeprecatedAPIRegression` guards do this — `get_event_loop` and `utcnow` — so **neither has
  ever checked anything in the only place they are enforced.**
- [x] **And one of them had a real violation the whole time.** `backend/app/models/backtest.py` used
  `datetime.utcnow()` three times. The guard existed, the violation existed, and they never met.
- [x] **The mirror image of this morning's improver-test bug, which is why it is worth stating as a rule.**
  There, *relative* paths broke the test outside the repo root. Here, an *absolute* path silently disabled it
  inside CI. **A test that locates source by path must derive that path from `__file__`** — anything else
  passes or fails for reasons unrelated to what it checks.
- [x] **Fixed both:** paths derived via `Path(__file__).resolve().parents[3]`, with
  `assert backend_dir.is_dir()` so a future move fails loudly instead of silently re-emptying the scan; and the
  three `datetime.utcnow()` calls replaced with `datetime.now(timezone.utc)`.
- [x] **Verified the guard can now fail:** injecting `datetime.utcnow()` into `models/ml_model.py` produces
  `utcnow() still used in: ['backend/app/models/ml_model.py']` and a red test. The `get_event_loop` scan comes
  back clean on 0 files, so making it live costs nothing.
- [x] **Worth noting where the violation lived:** `models/backtest.py` contains a `sessionmaker`-based test
  class inside an app module. pytest never collects it (the filename is not `test_*.py`), so it is dead test
  code in the model layer — the same manufacturing pattern as the 23:45 entry above.

## 🤖 2026-08-05 23:45 — A LIVE SAMPLE OF THE IMPROVER'S DEAD CODE, WHICH THE 813 FIGURE ASKED FOR

The 11:30 entry measured 813 unreferenced functions, rejected a gate built on it, and left one open item:
**"sample before believing it."** PR #1510 merged tonight and is that sample, caught fresh.

- [x] **No regression.** `static_server` imports, the three touched API/model modules import individually, and
  `test_bot_safeguards.py` — a test file the improver *rewrote* — passes 7/7. Improver PRs bypass CI, so this
  is checked by hand every tick; tonight it was clean.
- [x] **But two of the five files got trading logic they have no business holding.** The run's prompt was
  `strategy_logic` ("tighten entry conditions, add confirmation filters, improve exit logic") and it was
  applied to `api/v1/notifications.py`, `api/v1/improvements.py`, `models/ml_model.py`, `models/account.py`
  and a unit test. `models/account.py` — a **SQLAlchemy declarative model** — gained:

      def is_signal_allowed(self, signal, market_data) -> bool
      def should_exit_position(self, position, market_data) -> bool

  **Zero call sites** (`grep` across `backend/`). Two brand-new unreferenced functions in one commit, on the
  model layer, in +445/-54 lines.
- [x] **So the 813 figure is directionally right, and the mechanism is now named.** It is not stale legacy code
  — it is being *manufactured*, a few functions per improver run, because the improvement type is chosen by
  `hour % len(IMPROVEMENT_TYPES)` and the target file by `pick_target_file(hour, ...)`, **independently**.
  Nothing checks that a `strategy_logic` prompt landed on a file containing strategy logic.
- [x] **~~[P2] The cheap fix is pairing, not a dead-code gate.~~ SHIPPED 2026-08-06 07:30 (#1549) — and this
  item's own prescription was wrong.** It said to constrain `strategy_logic` to "`strategies/`, `execution/`,
  `risk/`". **All three are in `PROTECTED_PREFIXES`**, so following it literally would have aimed a type at
  directories the improver is forbidden to touch — the item proposed the one thing that cannot work. What
  shipped instead: `strategy_logic` was REMOVED, because those protected paths are the only places its prompt
  means anything. **Third queued item in one day whose text carried a factual error** (after Europe's close
  time and the "picked independently" mischaracterisation).

## 🧪 2026-08-05 22:45 — `walk_forward` HAD NEVER BEEN EXECUTED BY A TEST, ONLY STRING-MATCHED

Follow-up to the sub-window work, and the mutation pass is what exposed it.

- [x] **`test_walk_forward_actually_reports_sub_windows` greps the source.** That guard is real but partial:
  if `feat.index[mask]` raised at runtime, every test in the file would still pass and the weekly ML run would
  die. sklearn is not in the agent-test environment (CI installs pandas/numpy/pyyaml/requests), which is why
  the function had never been run — so the new test **stubs the classifier**. The subject is the plumbing:
  masking, indexing, and the sub-window call, not the model.
- [x] **The first mutation SURVIVED, and it was the one that mattered.** Replacing `feat.index[mask]` with
  `feat.index` passed everything. The slice indices stay in range against a *longer* index, so the windows
  still look well-formed while **every reported date is shifted by `MIN_TRAIN` — about a year**. Nothing else
  in the output changes. A reader would have compared sub-periods to the wrong calendar dates and never known.
- [x] **Fixed on both sides.** `sub_window_stats` now raises on a dates/returns length mismatch — it is a
  programming error and a silent one, so it must be loud — and the integration test asserts the first window
  starts at `feat.index[MIN_TRAIN]` and the last ends at the final bar. Both mutations now die.
- [x] **The general lesson, which this codebase keeps relearning: a call-site guard proves the call exists, not
  that it works.** Four features shipped dead this week and string-matching fixed that; this is the next layer
  — the call is there, the arguments are wrong, and only execution catches it.
- [ ] **Still pending: live sub-window data.** The latest persisted ML run is `20:17`, which predates the merge.
  The next run carries `sub_windows` and answers whether QQQ/NVDA beat buy-and-hold in all three sub-periods or
  just one.

## 📣 2026-08-05 21:40 — THE DISCORD CHANNEL INVENTORY, AND TWO PLACES POSTS DO NOT LAND WHERE THEY SAY

Triggered by the India MF run logging `no channel id for '#desk-india-mf' in this guild — webhook fallback`.
Swept every `notify.post` / `discord_post` / `chat_channel` reference: **17 distinct channels**.

    alpha-research      desk-commodities   desk-fx-rates    desk-polymarket    desk-tv-indicators
    ci-failures         desk-crypto        desk-india-mf    desk-research      engineering
    incidents           desk-equities      desk-options     desk-stat-arb      infra-alerts
    leadership-summary  squad-backend

- [x] **Ruled out first: the resolver is not the problem.** `#infra-alerts` and `infra-alerts` both appear in
  the codebase, which looks like a lookup mismatch — it is not. `discord_post` does
  `str(channel).lower().lstrip("#")` and `_load_channel_ids` strips the same way, so both spellings resolve.
- [ ] **[OPERATOR] `#desk-india-mf` does not exist in the guild.** Both India workflows post to it
  (`india-mf.yml` and `india-nse-signal.yml`), so both fall back to the default webhook and land in whichever
  channel that targets. The notifier reports this correctly on every run rather than dropping the message —
  the message is arriving, just not where it is addressed. **Creating the channel is the whole fix**; no code
  can create a Discord channel it has no ID for.
- [ ] **[OPERATOR] The International desk posts to `#desk-equities`, not a channel of its own.** Measured from
  `DESKS`: eight desks have a dedicated channel, `International` shares Equities'. Since the India expansion
  put `INDA`/`EPI`/`SMIN`/`INDY` on that desk, **India ETF activity is now being reported as Equities
  activity**. This is a defensible choice, not a defect — but it is worth knowing before reading either
  channel, and `#desk-international` would separate them. **Deliberately not repointed in code**: aiming a desk
  at a channel that does not exist would send its posts to the default webhook, which is strictly worse than
  the current honest mislabel.

## 🔬 2026-08-05 20:20 — "BEATS BUY-AND-HOLD" NOW HAS TO SURVIVE THREE SUB-PERIODS, NOT ONE

The 19:22 run beat buy-and-hold on QQQ (1.201 vs 0.734) and NVDA (1.184 vs 1.130). I flagged the caveat in the
same breath — **one window** — and this closes it.

- [x] **A single Sharpe cannot distinguish "the model has an edge" from "the model was long through one good
  stretch and flat through a crash",** and those two imply opposite decisions about wiring ML into live orders.
  The walk-forward now also reports Sharpe per equal-length sub-period (default 3) with a `beats` flag each.
- [x] **It costs nothing.** `walk_forward` already computed the strategy and benchmark return series; slicing
  them adds **zero model fits**. This is not a new experiment, it is the same one read properly.
- [x] **Under 30 OOS days a sub-window reports nothing rather than a number.** A Sharpe over a handful of days
  is noise, and a reader weighs a printed number the same whether or not it means anything.
- [x] **5 tests, 4 mutations, 4 caught.** The one that matters is
  `test_an_edge_concentrated_in_one_period_is_visible`: a synthetic series with the edge deliberately confined
  to the middle third must report `[False, True, False]`. Two separate mutations (a hardcoded `beats: True`,
  and slicing that ignores the window bounds) were caught by exactly that test — without it, a sub-window
  report that always said "consistent" would have looked like good news.
- [ ] **Next run tells us something new.** If QQQ and NVDA beat in all three sub-periods, that is a materially
  stronger claim than this morning's. If they beat in one, the honest read is that the headline number was a
  regime artefact — the same trap the window bug set, one level up.

## 🇮🇳 2026-08-05 20:06 — INDIA MUTUAL FUNDS: FIRST SCHEDULED RUN, FULL PIPELINE VERIFIED

`india-mf.yml` fired for the first time and every stage produced something. Not "exit 0" — output at each link:

```
14,241 schemes, 3,008 investable (Direct+Growth)
364 and 501 schemes with usable NAV history (per-AMC fetches)
category leaders (38 categories with enough peers)
[notify] delivered #desk-india-mf via WEBHOOK
[main 608eb26] india-mf: NAV ranking 2026-08-05 [skip ci]
```

- [x] **The category ranking is real: 38 categories cleared `min_peers`.** This is the half that was silently
  lost when the workflow scraped stdout by leading character and captured **0 of 114** category lines; the
  structured `/tmp/india_mf_discord.md` artifact replaced that, and this run is the proof it works.
- [x] **State committed to main** (`608eb260`, 4,767 bytes, keys `runs`/`as_of`/`investable`) — so the history
  accumulates rather than being recomputed and discarded, the failure `quick-backtest` lived with for months.
- [ ] **[P2] The Discord post did NOT land in `#desk-india-mf`.** The log is explicit:
  `no channel id for '#desk-india-mf' in this guild — webhook fallback`, then `delivered … (default → lands in
  whichever channel the default webhook targets)`. So the ranking posted *somewhere*, not where it belongs.
  **The notifier is behaving correctly** — it says exactly what it did instead of silently dropping the
  message, which is the whole point of that fallback — but the channel does not exist in the guild. Creating
  `#desk-india-mf` (and re-checking `desk-india-nse`) is an operator action; nothing in code can create a
  Discord channel it has no ID for.

## 🚨 2026-08-05 19:10 — THE AUDIT FIRED ON ITS FIRST LIVE RUN: SOMETHING ELSE IS BUYING ON THIS ACCOUNT

Run `31037485632`, the first desk run carrying `audit_order_origins()`:

```
⚠️ ORDER-ORIGIN AUDIT: 5 of 50 recent orders were NOT placed by this desk — a second writer is on this account
   2026-08-05T19:02:24Z USO  buy qty=None [filled] coid='7038f73a-d7a2-494e-b4d1-4a24b387705c'
   2026-08-05T19:01:51Z NVDA buy qty=None [filled] coid='0ac8049c-cc42-4fd8-a6d4-43031541046a'
   2026-08-05T19:01:42Z QQQ  buy qty=1    [filled] coid='04e8f2be-cbd7-46ae-b79c-121d36a865ab'
   2026-08-05T18:38:52Z NVDA buy qty=None [filled] coid='31dc9a10-17db-412d-8405-9ad9c14c7201'
   2026-08-05T17:58:46Z AAPL buy qty=1    [filled] coid='332050e9-fc0e-4535-b5fc-9d74aba3b2f8'
```

**What is proven:** 10% of the last 50 orders were not placed by the Actions desk placer; all five are **filled
buys**, all within 65 minutes, all carrying bare UUIDs — the id Alpaca generates when the caller supplies none.

**What is ruled out:** broker auto-liquidation. That sells; these are all buys.

- [x] **Root cause of the anonymity: NO backend order path sets a `client_order_id`. Zero occurrences in all
  four.** (A first pass here named only `submit_alpaca_order` — that was too narrow, and it also happens to be
  the *least* likely source, since it is reached only from the HTTP endpoint.)

  ```
  grep -c client_order_id
    backend/app/brokers/alpaca.py          0   ← AlpacaBroker.place_order, the background-task path
    backend/app/brokers/alpaca_orders.py   0   ← submit_alpaca_order, reached ONLY from the HTTP endpoint
    backend/app/bots/engine.py             0
    backend/app/api/v1/orders.py           0
  ```
- [x] **Narrowing that does hold, from the code:** `position_monitor` is the only background task that places
  orders and it places **exits** (`close_req`/`close_side`) — these are all buys, so it is not the source.
  `bot_runner._run_bot()` does `select(Bot)` before every run, and **agb8's database is `ok: false` with no
  sqlite fallback**, so its bot path cannot complete a run. That points at `9jz0`'s own bots (legitimate)
  rather than agb8 — but it is an inference from code, not proof, and it does not cover any agb8 path that
  skips the DB. **The tagging is what would settle it.**
- [ ] **[P0] Which backend is doing it CANNOT currently be determined, and that is the actual problem.** Two
  candidates both produce bare UUIDs:
  - `9jz0` (the live one) — health reports `scheduler: jobs_total 73, bot_jobs 64`, firing every 1-2 minutes.
    If this is the source, the orders are **legitimate** and the platform is working as designed.
  - `agb8` (operator item #1) — 11 background tasks, Alpaca connected, 113 strategies, dead DB. If this is the
    source, they are **rogue duplicates on an account it should not be touching**.

  Same evidence for both. That ambiguity is worth more than the count: it means nobody can currently answer
  "did our platform place this trade?" about 10% of its own book.
- [ ] **[P0] The fix is a `client_order_id` at each of the TWO submission points, deliberately NOT shipped
  unattended.** `AlpacaBroker.place_order` (`brokers/alpaca.py:157` — the path background tasks use) and
  `submit_alpaca_order` (`brokers/alpaca_orders.py:27` — the HTTP path), tagged `qb-{bot_or_strategy}-{sym}-{ts}`
  to mirror the desk's `qe-` scheme that already works and is already parsed by
  `desk_trade_sync.parse_strategy_from_coid`. `backend/app/brokers/*.py` is under **Do NOT Modify** in
  `backend/app/tasks/CLAUDE.md` and this is the live order-submission path — a change there moves real orders.
  It needs a human. Once tagged, `audit_order_origins()` names the writer instead of counting it, operator item
  #1 settles either way, **and desk_trade_sync gains attribution for the 10% of the book it currently skips.**
- [x] **This is what the audit was for.** Built 25 minutes earlier because a second writer had gone unreported
  across all 30 of today's desk runs; on its first live execution it found five filled orders nobody could
  account for.

## 🧰 2026-08-05 19:35 — A GREEN LOCAL RUN CARRIES NO INFORMATION ABOUT CI, AND THAT COST TWO RED PRs TODAY

    morning   test_backtest_covers_crypto.py   ModuleNotFoundError: requests
    evening   test_order_origin_audit.py       async def functions are not natively
                                               supported  (pytest-asyncio absent)

- [x] **Same cause both times, and it is not "I forgot".** The dev container has a fat Python environment and
  CI installs five packages on purpose. Nothing compared the two, so the only detector was a red PR — after the
  push, after the wait. `test_ci_installs_what_the_tests_import.py` is that comparison, run locally.
- [x] **The first draft of the guard missed the exact case it was written for, and mutation testing is what
  caught it.** The morning break came from `test_backtest_covers_crypto.py` importing `quick_backtest_runner`
  — a *local* module that imports `requests` at module scope. The test file itself names nothing third-party,
  so a guard reading only test files sees nothing while CI ERRORs at collection. It now follows local imports
  transitively and blames the test file, which is what a reader has to go and change.
- [x] **The two failures need two different detectors.** `pytest-asyncio` is demanded by a *marker in a file*;
  `pytest-timeout` is demanded by `--timeout=30` *on the command line* and by no marker anywhere. A
  marker-only check passes happily while the run command needs a plugin nobody installed — that mutation
  survived the first pass too.
- [x] **The guard found a third case on its first run:** `test_vol_target_sizing.py` imports `numpy` directly
  while CI only received it transitively via pandas. It never broke and it was never declared. Now explicit —
  *if a test imports it, the install line installs it* — rather than special-cased in the guard, because a
  guard with an exceptions list stops being one.
- [x] **Resolution chosen for the async tests: `asyncio.run()` in sync tests, not a new CI dependency.** Mine
  was the only file using the marker, and `pytest-asyncio` additionally needs `asyncio_mode` configured — a
  second failure surface for four tests. `test_the_agent_tests_run_without_pytest_asyncio` pins the decision
  and *skips itself* if someone later adds the plugin properly, so it constrains the mistake rather than the
  choice.
- [x] **6 mutations, 6 caught** — dropping each of `requests`, `pandas`, `numpy`, `pytest-timeout` from the
  install line, and re-adding `@pytest.mark.asyncio` to either async test.

## 👥 2026-08-05 19:15 — THE SECOND WRITER ON THE ALPACA ACCOUNT IS NOW REPORTED EVERY RUN

Operator item #1 has been "re-verified still live" by hand in three separate sessions. Re-verified again just
now, and this time the *rediscovery* is what got fixed.

`quantedge-api-agb8.onrender.com/health/detailed`, 2026-08-05 19:00:

    background_tasks: {"running": 11, "total": 11}
    alpaca:           {"ok": true, "note": "connected"}
    strategies:       {"count": 113}
    database:         {"ok": false, "error": "Name or service not known"}

- [x] **It cannot record what it does, but it can still place orders.** Its own DB is dead, so nothing on that
  side leaves a trace — and it moves the equity, buying power and positions that Kelly sizing, the daily loss
  cap and `is_risk_reducing` all read on this side.
- [x] **The existing origin report could almost never run.** `_report_recent_closes()` is called in exactly one
  place: inside the daily-loss-cap branch, and only when the book is additionally flat. On every ordinary run —
  including all 30 desk runs today — a second writer on the account was invisible.
- [x] **Shipped `audit_order_origins()`, called unconditionally in the account stage** before the sizing inputs
  it exists to warn about. `client_order_id` is the discriminator, and it is the right one because it lives at
  the *broker*: it survives the backend DB sitting on its ephemeral sqlite fallback, which is where it has been
  all week. Orders tagged `qe-…` are ours; anything else is named, with a bounded 5-line sample.
- [x] **It prints on the clean path too** (`✓ order-origin audit: all N recent order(s) placed by this desk`).
  A guard that only speaks when it fires cannot be told apart from one that stopped running — a mistake this
  file has recorded more than once.
- [x] **`status: "all"`, not `"closed"`.** `_report_recent_closes` asks about closed orders, which is right for
  *its* question ("what flattened the book?") and wrong for this one: a second writer's **open** orders are the
  ones about to move the book underneath the next sizing decision.
- [x] **11 tests, 5 mutations, 5 caught.** Including: an order with no `client_order_id` (Alpaca
  auto-liquidation, hand-placed) must count as **foreign**, since defaulting it to "ours" would hide precisely
  the writer this exists to find.
- [ ] **This does not fix the hazard, and is not meant to.** Suspending the service is still operator item #1;
  no code here can do it. What changes is that a duplicate writer now shows up in the log of every desk run
  instead of being re-derived by hand each session.

## 🖼️ 2026-08-05 19:00 — EVERY CONSUMER OF A FRONTEND URL WAS POINTED AT AN ABANDONED STUB

Three Vercel projects in this account answer **HTTP 200** and exactly one is this platform:

    quantedge-eight.vercel.app   "QuantEdge — Institutional Trading Platform"   ✅ the real app
    quantedge.vercel.app         "Create Next App"                              abandoned stub
    quant-edge-nine.vercel.app   "My Google AI Studio App"                      unrelated

- [x] **The rule was already written down and had not been applied to the code that does the verifying.**
  CONTINUITY records "verify by TITLE, not status code" from an earlier session that spent its whole length
  reporting "frontend 200 ✓" against the wrong app. Four consumers were still resolving the stub:

  | consumer | what it was actually doing |
  |---|---|
  | `backend/app/notifications/screenshot.py` | Discord dashboard screenshots captured a blank Next.js starter — successfully, no error, nothing to notice |
  | `.github/workflows/page-reporter.yml` | fallback when `PUBLIC_SITE_URL` is unset → screenshotted the stub |
  | `.github/scripts/third_party_monitor.py` | reported "Vercel (Frontend) healthy" about a site nobody uses |
  | `scripts/verify_live.py` | the script whose entire job is confirming the live system, confirming a different one |

- [x] **The uptime monitor could not fail under any condition.** Wrong target *and* `expected_status:
  [200, 301, 302, 404]` — with 404 counted healthy there is no response a dead deployment could return that it
  would reject. Now `[200]`, against the real host.
- [x] **Also stale in `scripts/CLAUDE.md`: the backend URL.** `quantedge-api.onrender.com/health` returns
  **404** (live-checked); the real host is `quantedge-api-9jz0.onrender.com`. Both wrong URLs sat in the same
  block a human is told to read first. Fixed, with the three-project table inline so the next reader does not
  have to rediscover it.
- [x] **The Notion iframe in `CTO_AGENT_WORKFLOW.md` pointed at the stub too** — the "live dashboard for
  everyone" showed a starter page to any squad member who opened it.
- [x] **11 tests, 4 mutations, 4 caught.** `FRONTEND_URL` is env-overridable now rather than a literal, since a
  hardcoded default is how this drifted and how it would drift again at the next domain change.

## ⏱️ 2026-08-05 18:30 — CRON DELIVERY RE-MEASURED AT ~11%, ON A DIFFERENT DAY AND TWO WORKFLOWS

Confirmation, not a new finding — the pacemaker comment estimated ~15% from one workflow on 2026-08-03. Fresh
numbers, and they hold:

| workflow | cron | nominal/day | actual `schedule` runs |
|---|---|---|---|
| `desk-trading.yml` | `*/15 9-22 * * 1-5` | ~38 by 18:26 | **4** (10.5%) |
| `desk-trading-crypto-24x7.yml` | `7,27,47 * * * *` | 72 | **8** (11%) |

- [x] **The pacemaker is load-bearing, and the numbers say by how much.** Of 30 desk-trading runs today:
  `workflow_dispatch` **16**, `workflow_run` 7, `schedule` 4, `push` 3. Cron contributes **13%** of the desks'
  actual cadence. Median gap 30.8 min against a nominal 15. All 30 succeeded.
- [x] **A near-miss worth recording: the crypto desk's 13 `skipped` runs are correct, not broken.** Its job
  carries `if: github.event_name == 'schedule' || 'workflow_dispatch'` so it *cedes* every trigger it shares
  with `desk-trading.yml` — the two use different concurrency groups, so on a shared trigger they run in
  parallel and compete for Alpaca's free-tier data limit (22 of 60 runs collided on 2026-07-28; one pair had
  desk-trading fetch 70 bars while the crypto-only run got 5 and 429'd).
- [x] **Which also means "the 24×7 desk only ran 8 times" is the wrong reading, and I nearly filed it.** Crypto
  is `always_open=True` and `desk-trading.yml` runs **all nine desks**, so every one of those 30 runs covered
  crypto. Real crypto coverage today is ~38 runs, not 8. The pacemaker dispatches only `desk-trading.yml`
  deliberately, for exactly this reason — dispatching both would recreate the collision by the one route the
  cede-rule cannot block, since `workflow_dispatch` is on its allowlist.
- [x] **I nearly generalised the 11% to the new daily India crons. That would have been wrong** — the failure
  mode splits by frequency, and the two regimes need opposite mitigations:

  | cron | measured |
  |---|---|
  | `*/15 * * * *` (quick-backtest, nominal 96/day) | **8-15 runs/day — 8-16%** |
  | `*/15 9-22 * * 1-5` (desk-trading) | **4 of ~38 — 10.5%** |
  | `7,27,47 * * * *` (crypto desk) | **8 of 72 — 11%** |
  | `17 4 * * 0` (ml-experiments, weekly) | **5 of last 5 Sundays — 100%** |

  **High-frequency crons get their slots dropped; low-frequency crons fire reliably but LATE.** Every one of
  those five ml-experiments runs landed at **06:4x UTC against a nominal 04:17** — a consistent ~2.5 hour
  delay, never a miss.
- [x] **CORRECTION 19:40, then AGAIN at 20:06 — the lag is *variable*, 1.5–4.7 hours, and quoting any range as
  if it were a schedule is itself the mistake.** `india-mf` fired at **20:06:35 against a nominal 18:30 —
  1h36m**, *shorter* than the "2.5–4.7h" I had published minutes earlier and outside the 21:00–23:15 arrival I
  predicted from it. Two revisions of one number inside an hour. The durable lesson is not the range: **the
  nominal minute predicts nothing, so a consumer must tolerate the whole spread** — which is exactly what the
  tilt file's 30h age window does, and why no code changed either time. The samples:

  | workflow | nominal | actual | lag |
  |---|---|---|---|
  | `daily-standup` | 13:30 | 18:13 | **4h43m** |
  | `strategy-auto-tune` | 00:30 | 03:41 | **3h11m** |
  | `ml-experiments` | 04:17 | 06:45 | 2h28m |

  Ruled out along the way: **the minute is not the cause.** 41 of this repo's crons sit on `:00` and it is
  tempting to blame collision, but other `:30` workflows fire fine — `daily-standup` has **235** scheduled
  runs, `strategy-auto-tune` 42. `india-mf` at `30 18` showed 0 runs simply because it had not fired
  yet — it arrived at **20:06:35**.
- [x] **So the risk to the India workflows is lateness, not absence — but the margin is thinner than I first
  wrote.** `india-nse-signal.yml` (`20 10 * * 1-5`) has ~3h before the 13:30 UTC open, and at the **corrected**
  2.5–4.7h lag it may land at 12:50–15:00 — i.e. **it can miss the open**, where the 2.5h figure said it never
  would. Moving the cron earlier is not an option: NSE closes at 10:00 UTC and the data does not exist before
  that.
- [x] **It degrades correctly anyway, which is why this needs no code change.** The desk's read-time age check
  accepts the *previous* session's file (30h window), so a late producer costs **yesterday's Indian read
  instead of no read**, and says so in the log. Desks running after the late arrival pick up today's. The
  design already absorbs the worst case — it just does not always deliver the best one. `india-mf.yml`
  (`30 18`) only posts a summary; lateness costs nothing there.
- [ ] **The real standing rule, correctly scoped:** anything on a **sub-hourly** cron that must actually run
  needs a dispatch path (this is what the pacemaker exists for). Anything daily or weekly can rely on cron, but
  **must not assume its stated minute** — budget hours of slack, and make the consumer tolerate a late producer.

## 🔬 2026-08-05 18:50 — THE ML EXPERIMENT WAS NEVER REPRODUCIBLE, AND "IT LOSES TO BUY-AND-HOLD" WAS WRONG

**This was found because persistence was fixed this morning.** Four runs in one day, identical params:

    09:28  SPY alpaca   940 rows   strat 0.932  buyhold 1.482   loses
    09:35  SPY yfinance 1399 rows  strat 0.645  buyhold 0.789   loses   ← 7 minutes later
    14:44  SPY alpaca   941 rows   strat 0.944  buyhold 1.490   loses
    16:54  SPY yfinance 1399 rows  strat 0.470  buyhold 0.791   loses

- [x] **`oos_days` alternated 688 / 1147 for the same symbol and the same config.** Alpaca's free **IEX** feed
  carries ~940 usable rows; yfinance carries **1399** — a 49% difference. `main()` called `fetch_alpaca` and
  fell back to yfinance only when it returned *nothing*, so a transient Alpaca failure silently changed the
  evaluation period. It did this **per symbol**: run 09:35 evaluated SPY on 1399 rows and QQQ on 940 and
  reported them side by side as one cross-sectional result.
- [x] **The benchmark moves with the window, and that is what made it dangerous.** SPY buy-and-hold Sharpe was
  **1.482 on 940 rows and 0.789 on 1399** — the longer series reaches back into the 2022 bear market. The
  *strategy* Sharpe barely moved. So the yardstick was changing, not the model.
- [x] **CORRECTION to the 2026-08-05 finding that the model simply loses to buy-and-hold.** That reading came
  from the short-window runs. On the **longer window**, which includes a real drawdown, it wins on two of three:

  | | short (≈940 rows, mostly bull) | long (1399 rows, incl. 2022) |
  |---|---|---|
  | QQQ | 1.118 vs 1.325 — loses | **1.128 vs 0.736 — beats** |
  | NVDA | 0.949 vs 1.487 — loses | **1.227 vs 1.119 — beats** |
  | SPY | 0.932 vs 1.482 — loses | 0.645 vs 0.789 — loses |

  That is the expected shape for a defensive model (`time_in_market` ≈ 0.5): it gives up upside in a bull run
  and earns its keep in a drawdown. **Neither the old "it loses" nor a new "it wins" is a safe claim** — the
  honest statement is that the answer depends on the window, and until today no run recorded which window it
  used in a way anyone compared.
- [x] **Fixed:** `fetch_bars()` takes whichever source has the **longer** history (deterministic for a given
  symbol and date, rather than dependent on which fetch succeeded); `common_window()` trims every symbol in a
  run to the window they all cover, **clamping both ends** — a lagging feed that has not published today's bar
  is the same defect at the other end and much harder to spot, since the row counts stay close. `first_date`
  and `last_date` are now in the payload, so a changed window is visible in one line instead of taking four
  runs and a diff.
- [x] **12 tests, 5 mutations, 5 caught** — one survived the first pass (an empty frame poisoning the shared
  window: an empty `DatetimeIndex` has `min() == NaT`, which makes every `>=` False and returns **zero rows for
  every symbol**, reported as "only 0 usable rows" for symbols whose data was fine). That is now pinned.
- [x] **✅ VERIFIED IN PRODUCTION, run `2026-08-05T19:22`** — the first ML run after the fix merged, and it is
  clean:

  | | source | rows | oos | window |
  |---|---|---|---|---|
  | SPY | yfinance | 1399 | 1147 | 2021-01-07 → 2026-08-04 |
  | QQQ | yfinance | 1399 | 1147 | 2021-01-07 → 2026-08-04 |
  | NVDA | yfinance | 1399 | 1147 | 2021-01-07 → 2026-08-04 |

  One source, one row count, **one window across all three** — against the 09:35 run that mixed yfinance/1399
  and alpaca/940 inside a single payload. `first_date`/`last_date` are present, so the next drift is a one-line
  diff rather than a four-run investigation.
- [x] **And the comparison finally means something. On a pinned 5.5-year window that includes the 2022 bear
  market, the model beats buy-and-hold on 2 of 3:**

      SPY   sharpe 0.634 vs buyhold 0.791   loses
      QQQ   sharpe 1.201 vs buyhold 0.734   BEATS  (+64%)
      NVDA  sharpe 1.184 vs buyhold 1.130   BEATS

  This is the same shape the mixed-window runs hinted at, now stated on evidence instead of coincidence: a
  model holding ~50% time-in-market gives up a bull run and earns its keep through a drawdown. **It is one
  window and three symbols — a reason to look further, not a green light.** Wiring ML into orders still needs
  the walk-forward repeated across windows and symbols, and it is still 0 references in
  `desk_order_placer.py`.

## ₿ 2026-08-05 18:30 — CRYPTO IS STRUCTURALLY STARVED, AND THE FIX IS AN ALLOCATION DECISION, NOT A BUG

Measured in run `31031318516`, the first healthy run after the margin freeze lifted:

    equity $21,903.38   buying_power $34,713.48   non_marginable_bp $0.00
    Crypto: 1 signal(s) fired, 0 placed — 1 insufficient cash

- [x] **The desk is not silently broken and does not need a reporting fix.** It says exactly what happened, in
  Discord, per run — that half is working (the per-desk zero-order post, shipped this morning, verified here).
- [x] **The mechanism.** Crypto sizes against `non_marginable_buying_power`, which is settled cash, because
  Alpaca crypto cannot use margin at all. Cash is currently **−$22,179.98** (a margin debit from the equity
  book), so non-marginable BP is **$0.00 by definition**. Crypto cannot place an order at any size. The
  `MARGIN_FLOOR_PCT` guard does not help: it reserves *buying power*, and crypto cannot spend buying power.
- [ ] **[OPERATOR] The only mechanism that would fix it is a cash reserve, and it is a real trade-off — not a
  free win.** To keep $X of cash for crypto, the equity desks must be capped at `cash − X`, which means
  **equity stops using margin** whenever cash approaches the reserve. The equity book only just started
  filling again (13 fills in this run) and throttling it to cash-only would be a significant, arguable
  behavioural change. **I am not shipping it unattended** — same call as `india_broker.py` not routing live
  orders. Same shape as the margin floor: it could not create cash today anyway, only prevent recurrence after
  a paper-account reset.
- [x] **Note what this means for the 24×7 claim.** Crypto is the only desk with `always_open=True` — the only
  one that can trade outside US hours. While cash is a margin debit, the platform's overnight coverage is
  **zero orders**, not "quiet". The India NSE tilt shipped today does not change this: it improves US-session
  India orders, it does not add an overnight *venue*.

## 🚫 2026-08-05 18:30 — TWO NON-FINDINGS, RECORDED SO THEY ARE NOT RE-INVESTIGATED

Both looked like live outages for a few minutes each. Neither was. Written down because the *next* sweep will
see the same surface and start the same dig.

- [x] **`/api/v1/positions/` and `/api/v1/trades/` are NOT returning an empty book while the broker holds one.**
  Both return **HTTP 401** to an unauthenticated probe. My probe parsed the JSON with
  `d.get('positions', d.get('data', []))`, which turns `{"detail":"Not authenticated"}` into `[]` and prints
  `0`. **An auth failure and an empty book are indistinguishable to a parser that defaults to empty** — the
  same shape as reading Alpaca positions with the wrong field names earlier today. Always print the HTTP status
  next to the count.
- [x] **`_live_platform_positions` is not self-defeatingly gated.** It exists so the dashboard survives an empty
  DB, and it is gated on the caller owning an `Account` row with `broker='alpaca'` — which reads like requiring
  the DB to be non-empty in order to survive the DB being empty. It is fine: `seed_all()` runs on boot
  (`main.py:161`, DEMO_MODE-gated, idempotent) and creates exactly that row at `bots/seed.py:92`, so the
  ephemeral sqlite fallback is re-seeded on every redeploy before any request arrives.

## 🇮🇳 2026-08-05 18:10 — THE INDIAN SESSION NOW FEEDS THE ORDERS IT CAN ACTUALLY REACH

- [x] **Shipped the `[P2]` intermediate play** from the 08-05 India entry, which is the only way to use NSE
  without an unexecutable desk. NSE trades **03:45–10:00 UTC** — a window US regular hours never touches — so
  by the time the desks run, Mumbai has already priced a full day of India-specific news that INDA/INFY/HDB
  will react to. `india_nse_signal.py` reads the close at 10:20 UTC; `desk_order_placer.py` applies it at the
  confidence gate, ~3 hours before pre-market.
- [x] **The tilt is deliberately weak.** A bounded nudge to an *existing* signal's confidence: `+` when the
  signal agrees with the Indian session, `−` when it disagrees, capped at **±0.06** (a 3% Nifty day saturates
  it). It cannot create a signal, flip a side, or push confidence outside [0,1]. Enough to move a marginal
  signal across a 0.60 bar; not enough to drag a weak one there.
- [x] **Single names carry weight 1.0, the index does not.** An ADR is a claim on the *same shares* that just
  traded in Mumbai. The Nifty 50 → ETF link is weaker and the weights say so: `INDA`/`INDY` 1.0, `EPI` 0.9
  (earnings re-weighting drifts from the index), `SMIN` **0.6** (small-cap has a different beta to a large-cap
  index, and pretending otherwise would overstate the edge). `MMYT` is deliberately unmapped — US-listed with
  no NSE line, so there is no Indian close to read, and mapping it to the index would invent one.
- [x] **State file, not a live fetch in the desk.** Six yfinance calls in the hot order path would buy a
  network dependency and a stall risk for data that changes once a day. The file also makes the input
  auditable after the fact.
- [x] **Staleness is re-checked by the CONSUMER, not trusted from the file.** This is the whole reason the
  design is safe: if the producing workflow stops, a file whose contents look perfectly valid keeps sitting in
  the repo, and every desk log line would stay green while the tilt rode a dead read. The desk re-derives the
  age at read time and refuses anything past `max_age_hours` (30). The producer *also* rewrites the file on
  every run — **even when nothing resolves** — so a total fetch failure overwrites yesterday's numbers instead
  of leaving them in place.
- [x] **Absent, stale, and genuinely flat are three different facts** and all three would serialise to
  `tilt: 0.0` if the code were careless. Each is pinned by its own test: no data → `skipped` with "fewer than
  2 usable daily closes"; too old → `skipped` with the age quoted; a real ±0.02% day → `skipped` with the move
  quoted, under the 0.15% noise floor.
- [x] **27 tests, every one mutation-verified** (10 mutations, 10 caught). The one that matters is
  `test_the_desk_actually_applies_the_tilt` — a call-site guard on the confidence gate, the fourth this week
  after three features shipped green and dead. It also asserts the write-back to `item["confidence"]`, without
  which top-K would rank on the pre-tilt number and the tilt would half-work.
- [x] **A second consumer-side cap (`INDIA_TILT_HARD_CAP`) is not redundant.** The producer caps at ±0.06; a
  hand-edited or corrupted file claiming 0.9 would otherwise force any signal past any threshold.
- [x] **VERIFIED LIVE on real NSE data, and the verification exposed a transport trap.** yfinance ships its own
  HTTP stack (`curl_cffi`) which dies behind an egress proxy — `curl: (35) Recv failure: Connection reset` on
  every one of the six symbols — while a plain `requests` call to *the identical Yahoo URL* returns the data.
  Without a fallback, that transport failure is **indistinguishable from a flat Indian day**: same empty frame,
  same zero tilts, same green log. Added `_via_chart_api` as the second source, and the log names which path
  each symbol took, because a silent fallback is how you stop knowing your primary source is dead.
- [x] **Real session 2026-08-05 read end to end**: Nifty 24,624.65 (+0.04%), HDFCBANK −0.94%, INFY +0.56%,
  ICICIBANK −0.31%, Wipro +0.37%, Dr Reddy's +0.29% — all in INR / `Asia/Kolkata`. Produced 5 real tilts
  (`HDB −0.0189`, `INFY +0.0111`, `WIT +0.0074`, `IBN −0.0062`, `RDY +0.0058`) and the desk consumer read them
  back correctly, sign-flipped per side.
- [x] **The index came back genuinely flat (+0.04%) on the first live run**, which is the best possible test of
  the noise floor: four honest `skipped` entries naming the move, not four fabricated `0.0` tilts. Absent,
  stale and flat stayed distinguishable on real data, not just in fixtures.
- [x] **Daily bars are stamped at the session OPEN (03:45 UTC), not the close.** Worth knowing before anyone
  "fixes" the timestamp handling — `build_payload` reconstructs the 10:00 UTC close from the bar's *date*, so
  the open-stamp is harmless, but only because nothing reads the bar's time.

## 🧪 2026-08-05 18:10 — THE IMPROVER'S SIZE GUARD HAD NO TEST, AND ITS OTHER TESTS ONLY PASSED FROM ONE DIRECTORY

- [x] **`_too_large` (the #1248 fix) was completely untested.** Added `test_oversized_files_are_never_picked`
  (real files on disk — the guard's job is to stat them) and `test_an_unreadable_path_is_treated_as_unusable`
  (OSError → True is deliberate: a path that cannot be stat'd cannot be read by `improve_file()` either).
  Both mutation-verified.
- [x] **Two existing improver tests were silently cwd-dependent.** `_too_large` returns True on OSError, and
  the tests' fake paths (`backend/app/utils/logging.py`) are *relative* — they resolve only when pytest runs
  from the repo root. CI does, so CI is green; running `pytest` from `.github/scripts` empties the candidate
  list and `pick_target_file` returns None, failing on an assertion about protected paths that has nothing to
  do with the real cause. Now `monkeypatch`ed explicitly.
- [x] **Worth recording that this was NOT a main regression, and I checked before acting.** It surfaced during
  the tick's regression sweep and looked exactly like one. The full agent suite is **1365 passed, 164 skipped**
  from the repo root. Same discipline as the 18 hardcoded dates in `test_desk_trade_sync.py` that turned out to
  be fine — the cheap check before the expensive "fix" was right both times.

## 🪙 2026-08-05 13:40 — THE BACKTEST'S CRYPTO HALF HAS NEVER PRODUCED A RESULT

- [x] **`quick-backtest.yml`'s job is named "Run backtests across all desks". Every persisted result reads
  `desks: ['equity']`.** Binance's public klines endpoint returns **HTTP 451** (geo-blocked from the runner
  region), `fetch_crypto_ohlcv` returned a bare `None` on any non-200, and the caller does
  `if not ohlcv: continue` — so each symbol was dropped **with no log line at all**. Only exceptions were
  logged, and a 451 is not an exception.
- [x] **Fixed:** the status is now printed (explicitly naming geo-blocking on 451) and there is a yfinance
  fallback mapping `BTCUSDT → BTC-USD`. yfinance already powers the equity backtests in the same process, so
  it is reachable where Binance is not.
- [ ] **Unverified from the dev container — both hosts are blocked here.** The next scheduled run will say
  which path it took; that log line is the point. If yfinance also fails in Actions, the fallback needs a
  third source (Coinbase/Kraken public OHLC), and the log will now show it instead of hiding it.
- [x] **This one wore a disguise the others did not:** `SYMBOLS["crypto"] = [BTCUSDT, ETHUSDT, SOLUSDT]` sat
  in the config looking wired, and the loop over it really does run. Config presence is not coverage — the
  same lesson as key presence not being capacity (07:43 entry).

## 🧪 2026-08-05 11:30 — A DEAD-CODE SCAN, MEASURED AND REJECTED

- [x] **Not shippable — 813 of 1,710 module-level functions in `.github/scripts` are never referenced
  anywhere** (48%), even after resolving `from X import orig as alias`. A gate that flags half the codebase
  cannot gate anything and will be ignored. **Do not build this**; the measurement is recorded so the idea is
  not re-attempted from scratch.
- [x] **What works instead, at the right granularity:** a per-feature call-site guard living next to the
  feature — `test_main_actually_uses_the_category_ranking`, `test_the_diagnostic_is_actually_printed`,
  `test_main_actually_calls_it`. Used three times on 2026-08-05 after making the identical mistake three
  times. Precise, cheap, and it fails for a reason the reader can act on.
- [x] **~~[P2] The 813 figure is worth a look on its own terms.~~ MEASURED 2026-08-06 08:30 — it was wrong by
  ~17x.** The real number is **23 of 815 (2.8%)**, not 813 of 1,710 (48%). Full derivation in the entry at the
  top of this file. The instruction "sample before believing it" was the right instruction and it took two
  rounds: **my own first re-derivation was also wrong**, and sampling caught that too.

## 🇮🇳 2026-08-05 10:10 — INDIAN MUTUAL FUNDS LIVE (data), and why Zerodha cannot be the bot

### ✅ Mutual funds — real data, no credentials needed
- [x] **AMFI feed built and verified**: `14,237` schemes across `52` AMCs, of which **3,156 are investable**
  (Direct + Growth). Free, unauthenticated, daily. `india_mf.py` — snapshot, history, universe filter,
  momentum ranking, bounded state.
- [x] **Two traps pinned by tests, both cost real debugging:**
  - `www.amfiindia.com` **302-redirects** to `portal.amfiindia.com`. Unfollowed, the payload is a 169-byte HTML
    stub that parses to zero schemes — an empty universe that reads as "no funds today", not a broken fetch.
  - **Ranking the raw history returns the same fund four times.** Direct/Regular × Growth/IDCW track one
    portfolio and post near-identical returns. Measured on Axis (`mf=53`), the unfiltered top 5 was: Axis IT ETF,
    then Nifty IT Index Fund as Direct-Growth, Direct-IDCW, Regular-Growth, Regular-IDCW — four slots for one
    fund, two of them the strictly-worse Regular plan. Filtered gives five distinct funds.
- [x] **Universe is Direct + Growth deliberately.** Direct plans carry no distributor commission (~0.5-1.0%
  lower expense ratio); IDCW options pay out of NAV, so their series has discontinuities that look exactly like
  negative returns.
- [x] **[P1] Daily workflow SHIPPED** — `india-mf.yml`, 18:30 UTC Mon-Fri (AMFI publishes 15:30-17:30 UTC), posts
  the top 10 to `#desk-india-mf`, commits bounded state, `contents: write` so the commit cannot 403 the way
  quick-backtest did. Live run ranks across 7 major AMCs (HDFC, SBI, Nippon, ABSL, Axis, UTI, Franklin).
- [x] **[P1] Category-relative ranking — SHIPPED.** `rank_within_categories(per_category=3)`, called from
  `main()` at `india_mf.py:344` and pinned by `test_main_actually_uses_the_category_ranking` (it shipped once
  as dead code and was caught the following tick). The current list is dominated by sector funds — the live top 10 is
  transportation/logistics, healthcare, technology and automotive. A sector fund that happened to run is a
  different claim from a top-quartile large-cap fund, and ranking them together is not a like-for-like
  comparison. Needs the AMFI category header (already parsed as a section marker, currently discarded).

### 🔴 [USER] Zerodha cannot be the automated bot — pick AngelOne or Dhan
- [ ] **Kite Connect issues `request_token` only via a browser redirect a human completes, and `access_token`
  expires daily (~06:00 IST).** A Kite-backed desk would place orders only on days somebody logged in by hand.
  It also costs ₹2,000/month. Automating it means scripting a login page — against their terms, and it breaks
  silently whenever the page changes.

  | broker | cost | unattended? |
  |---|---|---|
  | **AngelOne SmartAPI** | free | **yes** — TOTP from a stored seed |
  | **Dhan** | free | **yes** — long-lived token |
  | Upstox | free tier | no — daily OAuth redirect |
  | Zerodha Kite | ₹2,000/mo | no — daily interactive login |
- [ ] **To enable Indian execution, set four secrets:** `ANGELONE_API_KEY`, `ANGELONE_CLIENT_ID`,
  `ANGELONE_PASSWORD`, `ANGELONE_TOTP_SECRET`. `india_broker.status_line()` reports what is missing, and
  selection prefers unattended brokers over configured-but-manual ones.
- [x] **Deliberately no order routing yet.** `india_broker.py` is a capability/spec module — a test fails if
  `urlopen`/`place_order` appears in it. Routing live orders into a real Indian brokerage account moves real
  money and is not a change to make unsupervised.

## 🇮🇳 2026-08-05 09:40 — INDIA COVERAGE ADDED (tradeable today), and the limit on it

- [x] **The desk layer covered 19 countries and not India** — the world's 5th-largest equity market. Japan,
  China, Korea, Taiwan, Brazil, Mexico, Canada, UK, Germany, France, South Africa, Indonesia, Vietnam,
  Australia, Singapore, Thailand, Poland, Argentina, Malaysia — no India. (Macro/FX already held INDA/EPI/SMIN;
  nothing else did.)
- [x] **Added, all US-listed so they route through the existing Alpaca path — no new broker, no new venue code:**
  | desk | instruments |
  |---|---|
  | International | `INDA` (MSCI India), `EPI` (earnings-weighted), `SMIN` (small-cap), `INDY` (Nifty 50) |
  | Equities | `INFY`, `HDB`, `IBN`, `WIT`, `RDY`, `MMYT` — India ADRs, single-name exposure ETFs cannot express |
  | StatArb | `INDA`/`EPI` (same market, different weighting → mean-reverting spread), `INFY`/`WIT` (IT-services pair) |
  | TV Indicators | `INDA`, `INFY` |
  All eleven live-verified 2026-08-05: 5/5 daily bars and a live price each. `PIN` dropped — returned no close data.
  Symbol count 89 → 105 distinct, which is one extra batched bar request per run (5 → 6).

### [USER] Native NSE/BSE needs a broker — deliberately NOT faked
- [ ] **Data works; execution does not.** `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `^NSEI` all resolve
  through yfinance (verified — INR, `Asia/Kolkata`). **Alpaca has no route to NSE or BSE**, so a native India desk
  would generate confident signals that can never become orders — exactly the Polymarket situation already carried
  here. One such desk is a known limitation; two is a pattern. A test now fails if a `.NS`/`.BO`/`^` symbol is
  added to any desk, because the data resolving makes it *look* correct while every order fails.
- [ ] **To trade Indian equities natively, wire a real broker** — Zerodha Kite, Upstox, or ICICI Direct. That is
  an operator decision (account + API credentials); no code change can create the venue.
- [x] **[P2] The stronger intermediate play — SHIPPED 2026-08-05 18:10, see the top entry.** Use NSE data as a
  *signal input* for the India ETFs/ADRs that CAN execute. NSE trades 03:45–10:00 UTC — the window US RTH never
  reaches, and the platform is measured at ~27% of the clock. A Nifty/INFY.NS move informs an INDA or INFY order
  that is actually placeable. This gets India signal quality without an unexecutable desk.

### Deliberately excluded
- [x] **Options desk** — its eight underlyings are mega-liquid by design. INDA/INFY have listed options but thin
  chains, and the desk sizes real spreads; thin chains mean bad fills, not more coverage.
- [x] **Commodities** — India's gold demand is enormous but expressed through instruments the desk already trades
  (`GLD`). Nothing India-specific to add. **Crypto** is already global; **Polymarket** has no venue at all.

## 🩺 2026-08-05 08:15 — DEEP REVIEW: desks, trades, Discord, and every web page

Full sweep against the LIVE system, not the repo. Everything below is measured; run ids and figures included so
none of it has to be re-derived.

### 🔴 [P0][USER] THE ACCOUNT IS OUT OF MARGIN — nothing has traded for ~7 hours
- [ ] **This is now the single binding constraint on the whole platform, ahead of every other item in this file.**
  Buying power across the last ten desk runs:
  ```
  00:41 bp=$0.00     01:31 bp=$0.00     04:26 bp=$116.98
  00:58 bp=$0.00     02:22 bp=$46.35    05:15 bp=$101.23
  01:03 bp=$27.61    07:05 bp=$115.79   07:56 bp=$206.86
  ```
  **`cash` is pinned at exactly −$33,401.86 from 00:58 to 07:56** — an unchanging cash balance across eight runs
  means **zero fills in that window**. Equity drifts $21,892 → $22,013 on mark-to-market alone.
- [ ] **The desks are healthy; there is simply no capacity.** Latest run `30986611287` (07:56):
  `funnel: 51 generated → 17 survived gate+topK (3 exploration) → 0 placed`, dropped as
  `13 market closed · 3 no order path · 1 insufficient cash`. Signal generation, gating and top-K all work. With
  `MIN_ORDER_USD = 25` and available = `0.95 × buying_power`, a $0–$200 book can place essentially nothing.
- [ ] **The safety logic is correctly refusing to fix it.** `recover_negative_cash` declines to flatten while
  `bp > 0` ("MARGIN DEBIT, not orphaned notional") — deliberately, because the 2026-07-27 incident showed
  flattening a levered book realises losses, trips the daily loss cap, and freezes trading until session rollover.
  **Do not "fix" this by making it flatten.**
- [ ] **[USER] The decision is capital, not code.** Options: (a) reset the Alpaca paper account to restore buying
  power — fastest, loses the position history; (b) let the book run and only trade on days positions close;
  (c) add a margin-utilisation cap so desks stop sizing up before reaching 100%, which prevents recurrence but
  does not free the currently-committed capital. This supersedes the crypto-starvation entry (2026-08-04 17:45)
  — that described equity margin starving crypto; the equity margin is now exhausted too.

### 🟠 [P1] 18 OF 55 FRONTEND ENDPOINTS SERVE EMPTY PAYLOADS — ~10 pages render blank
- [ ] **Plumbing is sound: 54/55 genuine GET endpoints return HTTP 200** (the one 422 is `/market-data/bars`
  missing a required `symbol`, i.e. a probe artefact, not a defect). No 404s, no 5xx. **Pages are not broken —
  they are empty**, which to a user is indistinguishable.
- [ ] **Empty, grouped by root cause:**
  - **Ephemeral DB (unpause Supabase fixes all of these):** `/backtests/`, `/experiments/`, `/releases/`,
    `/releases/ab-tests/active`, `/risk/events`, `/notifications/activity`, `/notifications/stats`,
    `/archive/index`, `/comparison/results`, `/bots/?archived=true`, `/analytics/slippage`,
    `/analytics/arb-opportunities`. Pages: BacktestLab, Experiments, Releases, RiskManager, Activity, Archive,
    Comparison, Analytics, PnL, CryptoTrading.
  - **`/analytics/slippage` has a second cause worth noting:** it reads `Trade.execution_algo`, and `/trades/`
    returns **0 rows**. The desk measures slippage per fill (shipped 2026-08-04) but posts it to Discord and the
    Actions log only — it never reaches the backend. Even with a live DB this stays empty until desk fills land
    as `Trade` rows.
  - **Agent subsystem (already logged as unreachable, 9 modules):** `/agents/code-reviews`, `/agents/memory`,
    `/agents/skills`, `/agents/tasks` → AgentDashboard is entirely blank.
  - **ML (operator decision — torch excluded from Render):** `/ml/models` → MLInsights blank.
- [x] **~~[P2] Empty-state honesty.~~ SHIPPED 2026-08-06 14:40 — and the diagnosis in this item was wrong.**
  It said pages "render an empty table with no explanation". Most already had an empty state; the real fault
  was **misdirection**, which is worse than silence. See the entry at the top of this file.
- [ ] **~~[P2] superseded~~ — original text below.** [P2] Empty-state honesty.** Several of these pages render an empty table with no explanation. A one-line
  "no data yet — durable DB is paused" beats a blank grid that reads as a bug. Cheap, and it stops the next
  reviewer re-deriving this list.

### ✅ DESKS — all 9 verified correctly configured
- [x] Equities, Crypto, Options, Polymarket, Macro/FX, StatArb, Commodities, TV Indicators, International.
      Crypto is the only `always_open=True` (line 152); Polymarket the only `executable=False`, and it now reports
      `NO ORDER PATH` honestly rather than "closed" — 3 such signals at conf=0.99 in the latest run.
- [x] ~~**[P2] International shares `#desk-equities`**~~ **WITHDRAWN 2026-08-05** — the posts already open with
      `*International Desk*` (the format string interpolates `desk.name`), so they are labelled distinctly inside the
      shared channel. Splitting the channel would only add a `#general` fallback risk for a channel that may not exist.

### 📣 DISCORD — 21 real channels targeted
- [x] `cto-audit, desk-commodities, desk-crypto, desk-equities, desk-fx-rates, desk-lead-review, desk-options,
      desk-polymarket, desk-research, desk-stat-arb, desk-tv-indicators, engineering, infra-alerts,
      market-analysis, ml-experiments, pnl-daily, risk-alerts, signals, squad-backend, squad-data, strategy-lab`
- [ ] **[P2] `#desk-kalshi` is referenced by the unbuilt Kalshi desk item** but no producer posts to it — it will
      stay empty until that desk ships. Either build it or drop the channel from the plan.
- [ ] **[P2] Desk channels only receive output when a desk PLACES an order.** With buying power at zero the desk
      channels have been silent for hours while the desks ran fine — the same "silence means broken" ambiguity the
      run summary fixed for `#pnl-daily`. Post the per-desk funnel line even on a zero-order run.

## 🔇 2026-08-04 09:50 — the OA scout committed 56 times having never found anything
- [x] **Fixed: no-op runs no longer commit.** `oa_library.json` has read `{"known": []}` since it was created — every OA
  page redirects to a login and `OA_SESSION_COOKIE` has never been set, so the scout is a no-op by construction until a
  human supplies that secret. It still produced a commit on **every** run, because it rewrote `last_run`, a timestamp
  that moves each run and that **nothing reads** (written in `main()`, defaulted in `_load_state()`, and that is its
  entire lifetime). The file therefore always differed, so the workflow's `git diff --cached --quiet || commit` guard was
  never satisfied: **56 commits whose whole content was "the clock moved"**, stacked onto main, reading like progress.
- [x] **This is the inverse of the usual bug here.** The pattern this file keeps recording is work that succeeds and
  produces nothing. This is *nothing* dressed as work — arguably worse, because anyone scanning `git log` sees a scout
  that appears to be actively harvesting.
- [x] **Fix:** bump `last_run` only alongside a real change to `known` or `auth_wall`, so a no-op run leaves the file
  byte-identical and the existing guard suppresses the commit. Verified functionally: with the live state file and an
  auth-walled scrape, `changed == False` and the bytes are unmodified. Nothing is concealed — the workflow already pipes
  stdout into `$GITHUB_STEP_SUMMARY` (every run leaves a record in the Actions UI) and the auth-wall case posts to
  Discord asking for the cookie. 6 tests, mutation-checked; a first-run/missing-file case is covered so the 'skip when
  unchanged' logic can still create the file.
- [ ] **[USER] Still the actual blocker:** set `OA_SESSION_COOKIE` (Cookie header from a logged-in OA session) or the
  scout stays a no-op — now a quiet one instead of a noisy one.

## 🫀 2026-08-03 05:40 — the merge gate now rides the pacemaker, not just cron

The `schedule` added at 02:40 has produced **zero runs in 2h47m**. That is the same starvation the pacemaker exists to route around — its own header says *"GitHub starves free-tier schedules under load"* — so relying on cron for the merge sweep was betting on the one mechanism this repo has already measured as unreliable.

- [x] **[P0] The pacemaker now dispatches `auto-merge.yml` alongside its CI heartbeat.** ~50-minute cadence, no cron dependency. Kept **both** mechanisms deliberately: they appear distinguishably in the run log (`event=schedule` vs `event=workflow_dispatch`), so whichever actually delivers can be identified instead of guessed at — which matters given I have now mis-attributed this mechanism once already.
- [x] **Failure is visible but non-fatal.** Unlike the CI dispatch above it, the merge dispatch does **not** `exit 1`: losing the sweep must not kill the heartbeat driving 36 downstream workflows. It still emits `::error::`, because a silent skip here would be the same class as the permanent 403 that hid in `continuous-improvement.yml` for its whole lifetime. Both properties are pinned by tests.
- [x] **[P1] ~~Still unverified.~~ VERIFIED 2026-08-03 14:37 — triggers work, and the backlog is blocked by something else entirely.**
  Both events now appear on `auto-merge.yml`: **3 `schedule` + 6 `workflow_dispatch` runs today**, all `success`. The pacemaker
  dispatch also delivered on the desk side — 5 dispatches at exact 50-minute spacing (10:16, 11:07, 11:57, 12:47, 13:37), and
  the 13:37 one landed in RTH and logged **`Done. 7 orders placed across 9 desks.`** (run `30818846913`).
  
  **But the green backlog did NOT move — it grew to 100 open PRs.** Root cause found, and it is not the gate's triggers,
  its label rule, or CI:
  
  `#1358` carries `automerge`, is not a draft, bases on `main`, and has all three REQUIRED_CHECKS green
  (`test`, `test-agents`, `frontend-build`). `mergeable_state: "unstable"`. Its combined commit status:
  ```json
  {"state": "failure", "context": "Vercel",
   "description": "Deployment rate limited — retry in 24 hours.",
   "target_url": "https://vercel.com/...?upgradeToPro=build-rate-limit"}
  ```
  `auto-merge.yml` ends with `if (combined.state === 'failure' || combined.state === 'error') continue;`, so **every bot PR
  is refused because a preview deployment could not run.** Nothing is wrong with the code.
  
  This is a closed loop: each bot PR triggers a Vercel preview → the free-tier cap (100/day) is exhausted → Vercel posts
  `failure` → the gate refuses → PRs accumulate → more previews attempted.
  
  **Note the inconsistency in the gate itself:** it already treats Vercel as decorative for check-runs —
  `IGNORE = ['automerge', 'ops-sync', 'Vercel Preview Comments']`, on the stated grounds that "Vercel comments are
  decorative. Neither gates correctness — only real CI jobs do." — then hard-blocks on the Vercel *commit status*.
  Excluding the `Vercel` context from the combined-status check would be consistent with that stated intent and safe,
  because `frontend-build` is a REQUIRED check and is the actual frontend correctness gate.
- [ ] **[USER] That one-line gate fix is deliberately NOT shipped.** Its effect is to auto-merge ~50 labelled, green,
  stale PRs in one sweep — which is precisely the open stale-PR decision below, and an outward-facing action on work this
  session did not author. Several are known-defective: `#1246` adds a false invariant to a SHARED test
  (`assert signs[i] != signs[i-1]`, wrong for any trend-follower), a `position[idx - 1]` on a `DatetimeIndex` that cannot
  run, and an audit log that silently drops records. Decide the backlog first, then the gate fix is a one-liner.
- [ ] **[USER] The cheaper fix is on the Vercel side:** disable preview deployments for `improver/**` branches. That stops
  the cap burn, the bandwidth burn, and the `failure` status in one change — no repo edit needed.

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

## 💤 2026-08-03 02:40 — the merge gate has not fired in three days

Both earlier fixes verified live: the improver's dispatch logs `CI dispatched on improver/run-30773290001` instead of the 403, and improver PRs now get full CI. The stage *after* that is dead.

**`auto-merge.yml`'s last run was 2026-07-29 23:45**, on a human-pushed branch. In the three days since, the improver opened ~90 PRs (now `#1341`) and the gate never woke. `#1341` is green (`test`, `test-agents`, `frontend-build`), labelled `automerge`, not a draft, unmerged.

- [x] **[P0] Gave the gate a heartbeat it owns.** Every declared trigger is suppressed for bot PRs — GitHub does not start runs from GITHUB_TOKEN-attributed events, and the improver opens the PR, applies the label, and dispatches CI, all with that token. `pull_request_target: labeled`, `check_suite: completed` and `workflow_run: [CI] completed` are therefore all dead here; every historical run traces back to a human push. Added `schedule: "17,47 * * * *"`. A floor, not a guarantee — free-tier cron is starved (measured 1h22m–3h12m late) — but the job is idempotent so a late sweep still lands what is eligible.
- [x] **Pinned the coupling that makes it non-decorative.** A scheduled run has no event payload, so without the existing `candidates.size === 0` fallback that scans all open PRs it would exit green having inspected nothing. 4 tests; the schedule test fails against the old file, and the others pin the required-checks list and the `automerge` label gate — both of which matter *more* now that the gate can run with no CI event to anchor it.
- [ ] **[P1] The backlog is not all trigger-blocked.** `#1337`'s `test` genuinely failed; some unknown share of the ~90 open PRs will be legitimately red. The sweep will sort them, but do not read "gate fixed" as "90 PRs will land".
- [ ] **[USER] The stale-PR decision from 07-29 is now much larger.** ~90 open improver PRs, oldest based on `main` from days ago. Still not mass-merged or mass-closed here.

## ✅ 2026-07-29 22:45 — dispatch fix verified live; and it fixed a STALL, not a safety hole

Verified on a live run instead of assuming. Improver run `30496380998` (22:30, head `fa4a99a2` — the first carrying `actions: write`) dispatched cleanly, and **PR #1246 has CI check runs, the first improver PR ever to get them**, starting 3 seconds after the dispatch. `test` then **FAILED**, so the gate is holding it.

**Correction to the 21:40 entry.** I called the missing permission "the mechanical cause of the standing *improver PRs bypass CI so main can silently break* risk". Half wrong. `auto-merge.yml` lines 91–103 **already refuse** to merge a PR whose required checks never ran — with a comment naming `#876` (strategy-router 400) and `#929` (PIPELINE_DEFS + boot crashes). That hole was closed before this session, so the 403 never caused an unvalidated merge.

What it actually caused: **a fully stalled improvement pipeline.** 15 improver PRs open, back to `#1187` at 2026-07-28 22:02 — ~24h of autonomous work, every one with **zero** check runs, all correctly refused by the gate, all left to rot. Second correction in two ticks on this subject, both from asserting a mechanism before reading what would confirm it.

- [ ] **[USER] 15 stale improver PRs need a call.** All based on old `main`. Not mass-merged or mass-closed — that is 15 outward-facing actions on work this session did not author. Options: close them all and let the (now working) pipeline refill from current `main`; or rebase and let CI judge each. Closing is cleaner — they are stale by up to 24h and the improver re-picks targets every run.
- [ ] **[USER, P1] The improver rewrites SHARED tests with false invariants.** `#1246` edited `backend/tests/unit/test_strategies.py` to add `assert signs[i] != signs[i-1], "Consecutive non-zero signals must alternate sign"` — false for any trend-follower, which holds `+1` across consecutive bars. Plus `assert len(nonzero) >= 1` (a strategy may legitimately produce no signal on random data) and an `else: pytest.fail(...)` converting a deliberate skip into a failure. Letting it edit the suite that gates its own PRs is a structural conflict of interest.
  **Why this is yours and not mine:** the obvious fix — drop `backend/tests/unit/*.py` from `CANDIDATE_PATTERNS` — would disable a *configured* improvement type. `test_cases` is in `IMPROVEMENT_TYPES` as "Add 2-3 new unit test cases for edge cases not currently tested". So the choice is between letting it write tests (and sometimes weaken the ones that grade it) or turning that capability off. Distinguishing "add a new test" from "rewrite a shared assertion" is not something I can do reliably from the prompt side.
- [ ] **[P2] Two more defects in `#1246`, invisible to pytest.** `ml/features/normalization.py` does `position[idx - 1]` where `idx` comes from a `DatetimeIndex` (`Timestamp - 1` → `TypeError`, the function cannot run). `archive/trade_archiver.py` gained a `_validate_signal()` that **silently drops** signals with confidence < 0.6 — from a file whose docstring says it "writes every order, fill, and signal ... for long-term audit and replay". An audit log that discards records is worse than no filter.
- [ ] **[USER, P2] Vercel free-tier deploy cap is exhausted.** `#1246` carries a bot comment: `Resource is limited - try again in 24 hours (more than 100, code: "api-deployments-free-per-day")`. Every bot PR triggers a preview deployment, and the improver alone opens ~1/hour. Directly relevant to the bandwidth question — preview builds also consume transfer. Capping preview deploys for `improver/**` branches would fix both.

## ⚖️ 2026-07-29 19:30 — I over-attributed the improver's failures, and found the real current one

Went to verify the 18:47 extraction fix on a live run instead of assuming it. Read the last full **pre-fix** run end to end (`30476849972`, 17:46) and it does not support what I claimed.

**Correction.** I said the fence-extraction bug caused "32 of 41 failures, 78% of everything that went wrong". The 32 `syntax check failed` traces are real, but they are **historical**. That pre-fix run had **zero** syntax failures. All 5 of its failures were something else entirely, and the extraction fix does not touch them. The fix is still correct — `startswith("```")` demonstrably cannot handle a preamble, and this same `llm()` helper demonstrably returns preamble — but I have **not** shown it moved the syntax number, and I should not have implied a measured result from an unread log.

**What that run actually shows** — 10 attempts, 5 committed, 5 failed, every failure identical in kind:

```
· backend/app/backtest/cpcv.py is 13806 chars (> 8000) — skipped, whole-file rewrite unsafe
  ✗ LLM returned nothing for backend/app/backtest/cpcv.py
· backend/app/comparison/report_builder.py is 9242 chars (> 8000) — skipped…
· backend/tests/unit/test_all_employees.py is 21032 chars (> 8000) — skipped…
· backend/app/api/v1/discord_interactions.py is 8609 chars (> 8000) — skipped…
```

- [x] **[P1] An oversized file was charged to the LLM as a failure. The LLM was never called.** `improve_file()` returns `None` both for "too big to send" and "the model gave me nothing", so the caller logged the deliberate policy skip as `"LLM returned empty"`. **Half of that run's attempts**, all attributed to a model that never saw the input — corrupting the failure counter from #1235 at the moment it started working, and guaranteeing the first thing it measured would be a rate dominated by non-failures. The caller now checks `MAX_FILE_CHARS` itself and skips without recording; the guard inside `improve_file()` stays as defence in depth (it is the PR #420 lesson). Skips are counted and reported separately in the run summary. 5 tests, all 5 fail against the old code.
- [x] **[P1] `pick_target_file()` selected files it can never act on — FIXED.** 5 of 10 attempts in that run went to files above the 8000-char limit, against a budget of 10 for 5 wanted improvements. I deferred this last tick as "changes which files the improver ever touches"; on re-reading that was wrong — those files were **already rejected 100% of the time** by the guard, so filtering at selection changes nothing about which files can be improved and only stops spending a scarce attempt on a certain rejection. `_too_large()` now filters both candidate lists (the hour's pattern **and** the repo-wide fallback — filtering one leaks them back in through the other). 7 tests, 6 fail against the old code; one runs the real selector over the real tree for all 24 hour-slots to prove the filter has not starved selection, and one asserts the four measured offenders are now excluded.
- [x] **[P1 — was P2] "Improver PRs bypass CI" was one missing permission. FIXED.** The job granted `contents: write` + `pull-requests: write` and **not `actions: write`**, which `gh workflow run` requires (`POST /actions/workflows/{id}/dispatches`). So every run since the step was written ended `HTTP 403: Resource not accessible by integration` → `##[error]Process completed with exit code 1`, while `continue-on-error: true` reported the step as **success**. Verified in runs `30476849972` and `30483279439`: job conclusion `success`, step conclusion `success`, 403 in the log.
  The approach was never wrong — `workflow_dispatch` is one of two events *excepted* from GITHUB_TOKEN suppression, `test.yml` does declare `workflow_dispatch:`, and `pacemaker.yml` proves the mechanism works ("19 of the last 40 CI runs were dispatched by github-actions[bot]"). The improver just never asked for the permission. **This is the mechanical cause of the standing "improver PRs bypass CI so main can silently break" risk restated at the top of every monitor tick** — not event suppression, as assumed.
  `continue-on-error` is kept on purpose (a lost dispatch must not discard a run's improvements), but the failure now emits a `::warning::` annotation, because a bare non-zero exit under `continue-on-error` renders green — which is precisely how a permanent 403 hid for the workflow's entire lifetime. 6 tests sweep the class across all 102 workflows; 2 fail against the old file.
  **A note against myself:** the sweep's first version flagged `pacemaker.yml` too, and I nearly reported "the pacemaker has never worked" — the workflow holding the whole fleet's heartbeat together. It grants the permission on line 74 with a trailing comment (`actions: write   # required to dispatch CI`) and my regex anchored on end-of-line. Caught by reading the file instead of trusting my own red test.

## 📉 2026-07-29 17:25 — 41 failures were traced and none was counted, so the rate read 100%

Follow-on from the roll-call fix at 15:40. I made the header print `no failures recorded` instead of a fabricated 100% success rate, because `improvement_stats[*]["failures"]` was never incremented. This is the other half: **why** it was never incremented, and the fix.

`record_success()` initialises `failures: 0` and `agent_status_checker` sums that key. But `record_failure()` only appended to `failure_traces` — it never touched the counter. The counter existed, was read, and was never written, so the rate was pinned at 100% by construction.

Measured in the live memory file:

```
failure_traces stored : 41      (list is capped at 50, so this is a FLOOR)
improvement_stats     : every "failures" == 0
successes             : 61

by type          traces   successes          by reason
  constants          9       1   <- flawless   syntax check failed  32
  docstrings        10      10                 LLM returned empty    9
  cleanup           10      10
```

- [x] **[P1] `record_failure()` now increments the counter as well as the trace.** True rate is nearer **61/(61+41) ≈ 60%** than 100%. 8 tests, 6 fail against the old code — including one asserting the 50-item trace cap must NOT cap the counter, since that cap is precisely why traces can't serve as a total.
- [x] **Did not backfill the 41 historical failures.** The trace list is capped, so any backfill would be a known undercount presented as a total. The counter starts from the fix, which means the rate reads optimistically at first — accumulated successes against fresh failures. Disclosed here and in the source rather than papered over; a test pins the explanation so it can't quietly rot.
- [x] **[P1] 32 of 41 failures were "syntax check failed" — FOUND AND FIXED. It was a fence the code refused to look for.** Not an output-quality problem at all. `improve_file()` unwrapped the code fence only when the response *started* with one:
  ```python
  if improved.startswith("```"):   # position 0, or nothing happens
  ```
  The cascade's free providers don't oblige. `llm_common._extract()` deliberately falls back to `reasoning_content` when `content` is empty, so a reasoning model's chain-of-thought comes back **as** the response — and that is observable right now in `agent_status.json`, from this same `llm()` helper: *"The user asks: …"*, *"We need to respond as algo_agent, …"*. Any such preamble makes `startswith` False, so the prose **and** the fence went straight into `compile()`. SyntaxError, every time. `_extract_code()` now finds a fenced block anywhere and takes the longest one (responses carry illustrative snippets beside the real file). 11 tests, all 11 fail against the old code.
  Deliberately **not** done: no salvaging prose into code. A response with no fence and no valid Python still fails the syntax check — the goal is to stop discarding good output, not to start accepting bad output.
- [x] **Verified live: the 15:40 roll-call fix works in production.** The 17:14 run wrote `total_runs: 61, success_rate_pct: 100.0, failures_recorded: 0` — real numbers where it previously wrote `0` and `0`, and the new `failures_recorded` field is present and correctly reporting the gap this entry closes.

## 🧠 2026-08-03 21:45 — the ML work is aimed at the one model family that cannot run in production
Read before touching any LSTM promotion item below. The whole 07-29 section is real work that **cannot change production
behaviour on current hosting**, and the reason is deliberate.
- [x] **torch is excluded from Render on purpose.** `backend/pyproject.toml:54-56`: *"ML inference — in [ml] optional group
  so Render free tier skips the 800MB torch wheel … Render installs: pip install -e "." (no torch = ML strategies degrade
  gracefully)"*. `render.yaml:19` confirms: `pip install -e "."`, no `[ml]` extra. So `torch: available: false` is CORRECT,
  not a bug, and LSTM/SSM/Mamba/PatchTST inference is impossible there at any artifact quality.
- [x] **But xgboost, lightgbm and scikit-learn ARE installed** — base deps (`pyproject.toml:40,45,46`). `inference.py`
  loads four exact filenames and **three need no torch**: `xgboost_latest.ubj`, `lorentzian_latest.pkl`,
  `scaler_latest.pkl` (vs `lstm_latest.pt`).
- [x] **Nothing produces the torch-free artifacts.** The only CI trainer is `ci_lstm_trainer.py` (LSTM). A repo-wide grep
  for `xgboost_latest`/`lorentzian_latest` returns only `app/main.py` (health check) and `app/ml/inference.py` (loader).
  Backend trainers are `train_lstm`/`train_ssm` (torch) and `train_ppo_exec`/`train_rl` (stable-baselines3 → torch).
  `app/ml/models/xgboost_model.py` is a class with no pipeline behind it. Six successful weekly LSTM runs produced
  artifacts for the one runtime that is not installed; the two that ARE installed have no trainer.
- [ ] **[USER] Two real options, both operator calls.** (a) **Torch-free, no hosting change:** add an XGBoost/LightGBM
  trainer to CI and promote to `xgboost_latest.ubj` — runtime, model class and loader all already exist, only the trainer
  is missing, and it would work on Render today. (b) **Torch path:** host inference where torch fits (paid tier or a
  separate worker), which is what would make the `LSTMPredictor` unification below worth doing.
- [ ] **Until one is chosen, the LSTM items below are parked, not pending.** Leaving them on the active list is how six
  weeks of training ran into a runtime that was never going to load it.

## 🔬 2026-07-29 15:10 — six trained models, zero reaching inference; and the roll call was counting the wrong dimension

Two of the open questions from the 14:00 sweep, answered with measurements rather than inference.

### `ml_models: ok=false, count=0` — three independent breaks between the trainer and inference

The LSTM trainer is **not** broken. `lstm-training.yml` has run six times, weekly, **all six successful** (`30189512666`, `29674707432`, `29181196074`, `28731449600`, `28319490854`, `27903582392`), producing a 1.05 MB artifact each time. Nothing that it produces can ever be loaded:

1. **Filename.** `ci_lstm_trainer.py` writes `models_artifacts/lstm_spy_1d/model.pt`. `InferenceService.load_models()` reads the flat, fixed paths `models_artifacts/lstm_latest.pt`, `xgboost_latest.ubj`, `lorentzian_latest.pkl`, `scaler_latest.pkl`. It never recurses and never looks for `model.pt`.
2. **Checkpoint schema.** `AbstractModel.load()` does `checkpoint["state_dict"]` and `cls(**checkpoint["metadata"]["init_kwargs"])`. The CI trainer saves `{"model_state_dict", "n_features", "hidden", "n_layers", "dropout", "seq_len"}` — no `state_dict`, no `metadata`. A correctly-*named* file would still raise `KeyError`, swallowed by the `try/except` and logged as "Failed to load LSTM".
3. **Persistence.** The workflow only calls `upload-artifact` (30-day retention). Nothing commits. The 2026-06-21 artifact has already expired. Render's disk is ephemeral regardless.

The documented promotion path is a **comment** in the workflow header — "1. Download artifact from Actions tab 2. Commit to `backend/models_artifacts/<exp_name>/` 3. InferenceService auto-loads on next deploy". Step 3 is false: following steps 1 and 2 exactly would put the file where nothing reads it, in a format that would not load. The Discord post says "Models saved to `backend/models_artifacts/`", which is true only of the runner's disk. `/health/detailed` also globs `models_dir/*.pt` non-recursively, so it cannot see a subdirectory save either.

- [ ] ~~**[P1] Promote on the producer side, gated on the existing quality gate.** `ci_lstm_trainer.py` should additionally emit `lstm_latest.pt` in `AbstractModel`'s schema when the gate passes.~~ **CORRECTION 15:55 — this fix as written cannot work, and there is a FOURTH break I missed.** There are two classes named `LSTMPredictor` and they are **different networks**: the backend has a `SelfAttention` block the trainer lacks, a 64-wide head against the trainer's 32, and returns logits where the trainer returns sigmoid probabilities (init kwargs differ too — `hidden`/`layers` vs `hidden_size`/`num_layers`). `load_state_dict()` fails twice over: missing `attention.*` keys, and a shape mismatch on `head.0.weight` (32×256 vs 64×256). Writing the right *wrapper* schema does not help when the tensors inside describe a different architecture — and a forced load would silently mis-scale every prediction. **Promotion needs the architectures unified first**: the trainer should import and train `app.ml.models.lstm.LSTMPredictor` rather than maintain a second definition of the same name. `test_lstm_promotion_contract.py` now fails if promotion is wired without that. The misleading recipe in the workflow header has been replaced with the real constraints.
- [ ] **[P1] Verify the round trip where torch exists.** Deliberately not shipped blind: torch is not installed in this environment, so a `save → AbstractModel.load → predict` round trip cannot be checked locally, and an unverified promotion path is exactly the green-looking absence this file keeps recording. The training workflow has torch — the check belongs there, as a step that fails the run.
- [x] **[P2] `/health/detailed` `ml_models` count should `rglob`.** ~~Otherwise the metric stays 0 even once models land.~~ **Shipped 2026-08-03, but NOT as written — the literal fix would have been worse than the bug.** The premise was right: `ci_lstm_trainer.py` saves to `ARTIFACTS_DIR/lstm_<symbol>_1d/model.pt`, a subdirectory, so a top-level glob missed every model ever trained. But `ok` was `len(files) > 0`, so `rglob` alone flips `ok: false -> true` the moment the weekly trainer writes anything — `/health/detailed` would report "models loaded" while `inference.py` still loads nothing. That is the green-looking absence this file exists to record, shipped as a fix for itself. The check now reports **two** numbers: `artifacts_on_disk` (rglob — what training produced) and `count`/`ok` (the four exact top-level filenames `inference.py` actually opens). Extracted to `app.main.ml_models_check()`; 10 tests in `test_ml_models_health_check.py`, mutation-checked against both the naive-rglob and revert-to-glob regressions.

### `total_runs: 0, success_rate_pct: 0` across 18 agents — a metrics bug, not idle agents

- [x] **Fixed.** `improvement_stats` has **two writers with incompatible key spaces and incompatible schemas** sharing one dict: `continuous_improver.record_success()` keys by *improvement type* (`cleanup`, `docstrings`, …) with `{successes, failures, test_pass}`; `SharedContext.record_success()` keys by *agent name* with `{runs, successes, last_summary}`. The reporter read `runs` — the second schema — indexed by agent name — the second key space. `SharedContext.record_success()` has **zero call sites** outside its own docstring example, so that dimension has never been written by anything, and the live file holds only the first writer's entries, none of which carries `runs`. Every lookup returned 0; the `if total_runs else 0` guard then zeroed the percentage, so both numbers agreed and both were wrong. Real figure: **61 recorded attempts** across 8 improvement types. The roll call now reads `18 agents online · 61 recorded runs` instead of `0 total runs · 0% success rate`.
- [x] **Did not publish a fabricated 100%.** The only live writer never increments `failures`, so the ratio is definitionally 100 — the header says `no failures recorded` until a failure actually is, and the status file ships `failures_recorded` so a consumer can tell a real 100% from an untracked one. Per-agent `(0 runs)` suffixes are omitted rather than printed, since nothing writes that dimension. 10 tests, all 10 fail against the old code; one asserts against the live `agent_memory.json` rather than a hand-made dict.
- [ ] **[P2] `SharedContext.record_success()` is dead code.** Zero call sites. Either wire the agents to it — which is what
  would give genuine per-agent run counts — or delete it; leaving it is another live-looking decoy.
  **INVESTIGATED 2026-08-03 19:45 — "wire or delete" is the wrong question; the data already exists and is thrown away.**
  `agent_team.py:8692` builds `agent_tracking: {agent_name -> {posts, errors, channels, mode}}` and maintains it correctly
  throughout the wave loop (lines 8728-8751). At line 8803 it is formatted into an "Employee Run Report" and posted to
  `#engineering` — then **discarded**. Nothing writes it to `.github/state/agent_status.json`, which is why that file
  reports `total_agents: 18, total_runs: 61` while **all 18 per-agent entries show `runs: 0`** (verified). Same pattern as
  `review_employees_main`, which generates a 0-10 score per employee daily, posts it, and keeps no history.
  So the fix is ~10 lines of persistence, not a 48-agent wiring job — and `SharedContext` is redundant with a tracker that
  already works.
- [ ] **[USER] But persisting it now would be a pipe to a closed tap — sequence this deliberately.** The wave that
  populates `agent_tracking` runs ONLY from `auto-launch.yml`, whose triggers are `push` touching that workflow file plus
  `workflow_dispatch` — **no schedule**. So the full 48-agent company essentially never runs autonomously, and persisting
  per-agent counts would faithfully record a wave that almost never happens. Giving the wave a schedule is the real
  prerequisite, and that is an operator call because it puts 48 agents' worth of posts into Discord on a cadence —
  a noise decision, not a correctness one. Order: **decide the wave's cadence → persist `agent_tracking` → delete
  `SharedContext`** (redundant once the working tracker is the source of truth).

## ✂ 2026-07-29 14:50 — the trimmer worked, and the workflow threw the answer away

Went to verify the trimmer's first live retirement instead of assuming it. It had run — and lost. Run `30457733119`, 13:48 UTC, three consecutive lines:

```
[TRIM] avellaneda: cumulative return -7.9% ≤ -5.0% over 10 trades
trimmed total: 1 | newly trimmed this run: 1
No trim changes.
```

The third line contradicts the first two. The persist step gated on `! git diff --quiet -- .github/state/strategy_trims.json`, and **`git diff` is blind to untracked files** — the trims file had never been committed, so the gate reported "no change", the commit was skipped, and the file died with the runner. `git ls-files` confirms `strategy_trims.json` has never existed in this repo. This is the same blind spot I fixed in `fill-tracking.yml` a few hours earlier; I fixed the instance and not the class.

- [x] **[P0] `strategy-trim.yml` now stages before it diffs.** `git add -- "$f"` then `git diff --cached --quiet -- "$f"`, which sees an addition as a change. Verified the whole chain end-to-end against the real registry, not a stand-in: a trims file keyed `avellaneda` expands through `_expand_truncated` to `{avellaneda, avellaneda_stoikov_mm}` (exactly one registry prefix match), and `_desk_strategies(['avellaneda_stoikov_mm','momentum'], trims)` returns `['momentum']`. The persist step was the only broken link.
- [x] **[P1] `system-status.yml` had the identical bug, and had never once published.** Its own header promises "commits a fresh SYSTEM_STATUS.md so the repo always shows live truth" — `SYSTEM_STATUS.md` has never existed in the repository. Every run since it was written probed brain/Slack/Alpaca/backend, rendered the report, and discarded it. Same fix.
- [x] **[P1] Swept the class.** `test_state_persist_sees_new_files.py` walks every workflow, finds each step that stages an explicitly-named path and commits it, and fails if that path is untracked while the gate omits `--cached`. `channel-monitor.yml` is fine (`agent_state.json` is tracked); `agent-health-monitor.yml` is fine and deliberately so (bare `git diff --quiet` paired with `git add -u`, which stages tracked changes only — the check and the staging agree). The test also builds a throwaway git repo and *demonstrates* the untracked-invisibility rather than asserting it, so it does not rest on my description of git. 4 tests; 3 fail against the pre-fix workflows.
- [x] **VERIFIED LIVE at 15:19, 18 minutes after the fix merged.** `strategy_trims.json` is tracked in the repo for the first time (`c10a64fa`, 15:01). The 15:19 equity desk run (`30465278667`) printed `✂ 2 strategy(ies) retired by the trimmer will not trade: avellaneda, avellaneda_stoikov_mm` — two names because `_expand_truncated` resolved the legacy key against the real registry. Proof it took *effect* rather than just printing: `avellaneda` appears **exactly once** in the 805-line log, that line, with no signal/sizing/order following it, while the run was otherwise fully active (`Done. 9 orders placed across 9 desks`, orders filling, no loss cap). The full chain — attribution → trim → persist → desk read → key expansion → exclusion — is closed and observed.
- [ ] **[P2] Cron starvation on this workflow, measured.** `41 */6 * * *` fired at 03:53 (+3h12m), 09:23 (+2h42m), 15:02 (+2h21m), 20:03 (+1h22m) — never once near its slot. The 13:48 run that produced the trim was `push`-triggered, not scheduled. Consistent with the desk measurements; not fixed here.

## 🔴 2026-07-29 14:00 — LIVE SWEEP (market open): four findings

- [x] **[P0] The 61-bot OA fleet was evaluating on NO price data.** `_fetch_ohlcv` has two sources and BOTH failed, so it always returned an empty DataFrame. (a) The Redis read built `ohlcv:{symbol}:1d` by hand while the writer uses `ohlcv:{exchange}:{symbol}:{interval}` — no exchange segment, so it missed every symbol every tick (the documented `prices:{symbol}` topic-vs-key class, again). (b) `yf.download()` returns a **MultiIndex even for a single ticker** in yfinance ≥ 0.2.51 (`('Close','SPY')`), so `c.lower()` raised `'tuple' object has no attribute 'lower'` — visible in the live Render logs several times a minute for SPY and QQQ. Bots kept logging "Conditions not met", indistinguishable from a real no-signal verdict. Fixed; 6 tests, 3 fail pre-fix.
- [ ] **[USER, P0] Supabase project is PAUSED, not password-broken.** `list_projects` reports `status: "INACTIVE"` for `vexzwnfbmznvxoxxktax`. CONTINUITY has said all session that "the password is the ONLY blocker" — that is now wrong. A paused project has no tenant on the pooler, which is exactly the `(ENOTFOUND) tenant/user … not found` the backend reports. **Unpause it first**, then re-check the credential. The two faults compound: auth failure → no connections → inactivity → auto-pause. Everything downstream (ephemeral sqlite, trades wiped on every redeploy, empty leaderboard, inert attribution pruning) is a symptom of this one thing.
- [ ] **[P1] Every Render redeploy still wipes trade history.** Measured today: the 13:12 desk run logged `✓ Performance weights active for 11 strategies`; after the 13:48 deploy, `/api/v1/trades/` returns 0 and `/leaderboard/live` returns 0 strategies — so attribution-weight pruning (`✂ pruned by attribution`) is currently **inert**, leaving the file-based trimmer as the only working pruning path. Symptom of the Supabase pause above.
- [ ] **[P2] "Deploy on main" failed at 13:48** (run `30457733246`) but failed SAFE — the notification says "site still serves the previous build", and the streamed Render logs show the service healthy: bots evaluating each minute, order sync ticking, `/health 200`. No downtime. Worth root-causing separately.

## 🔴 2026-07-28 — ZERO trades for 18 days: every order blocked by the loss cap

Verified the money-path fixes actually landed, rather than assuming. **The scanner fix is live in production** — `/api/v1/scanners/polymarket` now returns `200` with `score: 0.75` (normalised from 75.0) and `side: "buy"` (mapped from `long_yes`), where it previously 500'd.

Then checked whether trades are actually flowing. They are not:

- `/api/v1/trades/` → **9 trades total**, most recent **2026-07-10** — eighteen days ago.
- Those 9 are genuine, not seed data: `desk_trade_sync` backfills from Alpaca's real 30-day order history on a schedule. (I initially assumed seed data and checked — the sqlite fallback resets on deploy, so anything surviving had to be re-derived. It is re-derived from Alpaca.)

**The cause, from the 01:05 UTC crypto desk run — and it is not "insufficient balance":**

```
🛑 Loss cap ACTIVE — only risk-reducing orders allowed (0 open positions eligible to reduce)
🛑 avellaneda_stoikov_mm/UNI/USD  BUY — blocked by loss cap (would increase exposure)
🛑 vol_of_vol_timing/MKR/USD      BUY — blocked by loss cap (would increase exposure)
🛑 avellaneda_stoikov_mm/AAVE/USD BUY — blocked by loss cap (would increase exposure)
Done. 0 orders placed across 9 desks.
```

**The desks are healthy.** They fetch data, run ensembles, and produce signals that clear the confidence gate (`passed=3 filtered=0`). Every one is then blocked. Under the cap only risk-**reducing** orders pass — and with **0 open positions, nothing can be risk-reducing**, so nothing can pass at all.

That shape is correct as a *daily* cooling-off rule and pathological if it persists for 18 days. I could not tell which from the available evidence, and did not guess.

- [x] **Made the CONSOLE say which, after the Discord-only version proved unreadable.** First pass put the numbers in the Discord summary — and the very next investigation still could not see them, because I read desk runs through the Actions API, not Discord. The `DAILY LOSS CAP: equity down X%` line that carries them is printed hundreds of lines earlier (account fetch, before the per-symbol ensemble output), so reaching it means paging the whole run. The banner at the point of blocking now repeats `equity $X vs prior close $Y (-Z%, cap 2%)` plus `⚠️ 0 reducible → NOTHING can pass`, so a 20-line tail is enough.
- [x] **Added a contradiction detector.** `daily_loss_cap_hit` is `equity < last_equity * (1 - cap)`. With **zero open positions and no fills, equity cannot move**, so it should EQUAL prior close and the cap cannot legitimately be active. If it is, the inputs are wrong — a stale `last_equity` being the known candidate (the source already carries a 2026-07-20 note about weekend crypto drift against Friday's close; eighteen days is not a weekend). The run now prints `⚠️ equity EQUALS prior close — cap should not be active; suspect a stale last_equity` when that holds, so the next run distinguishes "correct daily cap" from "firing on stale inputs" instead of leaving it to inference.
- [x] **Made the summary say which.** The drawdown figure was printed only to the CI log; the Discord line said just `🛑 loss cap ACTIVE — new exposure blocked`. It now carries `equity $X vs prior close $Y (-Z%, cap 2%) · N position(s) eligible to reduce`, plus an explicit `⚠️ nothing can pass while this is 0`. The next desk run answers the question in the channel instead of requiring an Actions-log dig. 8 tests (`test_loss_cap_visibility.py`), including the divide-by-zero when the broker fetch fails.
- [ ] **[USER] The Alpaca paper-account reset is now the confirmed blocker on trading**, not a background nag. Everything upstream of order placement works; the cap is the terminal gate and it cannot clear itself while the book is empty.
- [x] **[P2] ~~Ensemble signals cancel out at scale.~~ SUPERSEDED — fixed 2026-07-28, this entry predates the fix.**
  The always-stand-aside rule is gone. `desk_order_placer.py:696-713` now combines opposing sides with the same
  `1-prod(1-ci)` used for agreement and trades the dominant side at the NET confidence, gated by `_ENSEMBLE_NET_MIN`
  (default 0.60, matching `confidence_min`, and settable >1.0 to restore the old rule without a code change). The
  measurement that motivated it is in that comment: over 76 conflicts, `crypto_adaptive_trend` was the ONLY sell voice on
  all 16 crypto conflicts at 0.16-0.52, vetoing buy consensus of 0.61-0.97 — SHIB/USD had `avellaneda_stoikov_mm(0.90)`
  killed by a single 0.16. Confirmed healthy in production 2026-08-03: runs `30818846913` (13:37) and `30822832596`
  (14:28) each placed **7 orders across 9 desks**, `total_notional=+2587.46`.
## 📏 2026-08-03 16:45 — execution quality is now measured (it never was)
- [x] **Slippage vs arrival price is recorded per order.** The desk had both halves of an implementation-shortfall
  calculation all along and never compared them: `limit_price` is the decision-time bar close (arrival) and Alpaca returns
  `filled_avg_price` on the fill. `Done. 7 orders placed` told us trades happened and nothing about whether they happened
  at a good price. New pure helper `slippage_bps()` in `desk_order_placer.py`; every order record now carries
  `arrival_price`, `fill_price`, `slippage_bps`, and the run summary posts
  `execution: avg slippage +X bps · worst +Y bps · N/M measured (K unmeasured)` to `#pnl-daily` beside the funnel line.
- [x] **Two traps, both pinned by tests (15, mutation-checked).** (1) **Sign:** IS is a COST — a buy above arrival and a
  sell below arrival are both positive. A raw price delta makes them +100/-100 and a mixed book averages to ~0 no matter
  how bad execution gets; `test_a_mixed_book_of_equal_costs_does_not_average_to_zero` fails if that regresses.
  (2) **Unmeasured != free:** `_ensure_filled()` submits a market replacement without awaiting its fill, so
  `filled_avg_price` is legitimately absent there. Returning `0.0` would report *perfect execution for trades nobody
  measured*. `slippage_bps()` returns `None`, the aggregate filters with an explicit `is not None` (not truthiness, which
  would also drop genuine 0.0 bps fills), and the summary states the unmeasured count.
- [ ] **[P2] Next TCA step: persist it.** Slippage is currently per-run only — logged and posted, then gone with the
  runner. Per-strategy execution cost over time is what would actually inform sizing and retirement, and that needs the
  records committed or written to Postgres. Blocked behind the same Supabase pause as trade history.

- [x] **CORRECTION 2026-08-03 15:40 — the desks do NOT place naive market orders.** I previously reported they did, from
  `crypto_adaptive_trend.py:154`'s `metadata={"order_type": "market"}`. That metadata field is not what executes. The
  real path is `_ensure_filled()` (`desk_order_placer.py:1110`): **limit-first with cancel-replace**, falling back to a
  market order only after `FILL_WAIT_S` elapses without a fill, and double-fill safe (if the cancel races a fill, the fill
  wins and no replacement is sent). Live evidence, run `30822832596`:
  `► time_series_momentum/EWT signal=BUY conf=1.00 — placing $194 limit-first order` → `✓ limit filled after 5s`.
  This changes the standing research recommendation: "switch from market orders to IS-aware limit logic" is **already
  substantially done**. What is genuinely missing is the measurement half — no TCA, no slippage-vs-arrival-price logging.
  That, not the order type, is the real execution-quality gap.
- [ ] **[P2] ~~Original entry retained for provenance~~ Ensemble signals cancel out at scale.** Same run: nearly every crypto symbol logged `sell/buy conflict — stand aside` (SOL, AVAX, LTC, DOGE, LINK, BCH, DOT, XRP, SHIB, SUSHI, YFI, GRT, CRV, XTZ, BAT). Only 3 signals survived from 9 desks. Worth investigating separately from the cap — it caps the ceiling on trade count even once the account is healthy.

## 🛡️ 2026-07-28 — the 5xx guard never covered parameterised routes (the hole the scanner bug fell through)

After fixing the `/api/v1/scanners/{desk}` 500, the obvious question: there is already a `test_no_get_endpoint_returns_5xx` — why didn't it catch it?

Because it walks **parameterless** GETs only, by construction. **25 of the 128 GET routes take a path parameter and were never smoke-tested at all.** The parameterless twin `/scanners/` passed because it reads an empty cache in tests.

- [x] **Extended the walk to parameterised routes.** No fixtures needed: placeholders are deliberately non-existent ids, and the assertion is "does not 5xx" — a 404/422 on a bogus id is a pass. `test_every_path_parameter_has_a_placeholder` fails if a new param name appears with no placeholder, so a future `/{portfolio_id}` cannot silently drop out of the walk — which is exactly how this hole persisted.
- [x] **One value per parameter is not enough — my first draft proved it.** With `desk="equity"` the new walk **passed against the unfixed scanner**: equity returns an empty result set here and never reaches the serialisation path; only `polymarket` produced a row. I nearly shipped a guard advertised as catching the bug it did not catch. Placeholders are now **lists**, expanded over every value for enum-like params. Re-verified against the pre-fix tree, where it now reports `/api/v1/scanners/{desk} (as /api/v1/scanners/polymarket) → 500`.
- [x] **500 vs 502/503 distinguished.** Three market-data routes return `502 {"detail": "Alpaca bars error: 401"}` and one returns `503 {"detail": "Redis unavailable"}` in this environment — those are *deliberate* `HTTPException`s for absent upstreams, not crashes. Failing on them would make the guard environment-dependent, and a guard that cries wolf gets deleted. A 502/503 **without** a `detail` still fails: that shape means something escaped rather than being handled.

Full suite, two consecutive CI-invocation runs: **1941 passed**, 29 skipped, 5 xfailed.

## 🔴 2026-07-28 — /api/v1/scanners returned 500 for EVERY non-empty result

Found by writing the first-ever test for a live endpoint that had 0% coverage. The reachability triage split the 37 zero-coverage modules into dead ones (guarded) and **live but wholly untested** ones — `api/v1/scanners` → `tasks/stock_scanners` (275 statements), `api/v1/releases` → `ml/serving/serve` + `ab_router`. That second group is worse than dead code: dead code cannot break a user.

**Producer and schema disagreed on both fields, and had since they were written:**

| field | scanners emit | `ScanResultOut` requires |
|---|---|---|
| `score` | `min(score, 100)` — a 0–100 scale | `ge=0.0, le=1.0` |
| `side` | `long` / `short` / `long_yes` / `long_no` | `{buy, sell, neutral, none}` |

Every non-empty scan result raised `ValidationError` → 500. **All three desks**, not just polymarket — equity and crypto merely returned empty in the test environment, so the serialisation path never ran with rows. Invisible from outside because an anonymous probe gets 401 (verified against production).

- [x] **Fixed at the boundary** with `_normalise_scan_item()`, applied at all three construction sites (live scan, single-desk cache, all-desk cache). Normalising in the API rather than changing the scanners: their 0–100 score and long/short vocabulary are also written to Redis and consumed elsewhere. It also makes the documented contract ("score normalized between 0 and 1") true for the first time.
- [x] **NaN fails SAFE.** `min(1.0, nan)` returns `1.0` in Python — every NaN comparison is False — so naive clamping would have turned a malformed score into **maximum confidence** on a ranking signal. Explicitly forced to 0.0.
- [x] Tests: `tests/unit/test_scanner_normalisation.py` (25) + `tests/integration/test_untested_live_endpoints.py` (14). Verified to fail on the unfixed tree with the exact production error.

**A mistake worth recording:** the first draft called `auth_headers()` in every test — ~11 registrations against a **10/min** limiter — which starved other files on the same worker and turned `test_api_health::test_auth_register_then_login` red while it passed in isolation. Now one cached token per file. *A test that breaks its neighbours by consuming a shared budget is the same class of problem as one that writes to a shared database* — the second time this session that a "fix" damaged the shared fixture.

Full suite, two consecutive CI-invocation runs: **1935 passed**, 33 skipped, 5 xfailed.

## 🔴 2026-07-28 — the BACKEND's agent subsystem is unreachable (9 modules)

Direct follow-on from the coverage baseline. 0% coverage does not mean dead — a module can be exercised only in production — so the 37 zero-coverage modules were cross-referenced against **transitive** import reachability from the real entrypoints (`static_server`, `main`, `api/v1/router`, `tasks/scheduler`).

One-hop analysis is misleading here and nearly fooled me: `free_llm_router` *looks* imported — by `ai_strategy_generator`, `research_pipeline` and `self_improving_loop` — but **all three of those are themselves unreachable.** The whole cluster imports each other and nothing else reaches it:

```
agent_bus            agent_memory         ai_strategy_generator
free_llm_router      knowledge_loop       research_pipeline
self_improving_loop  strategy_auction     task_queue
```

**Six of the nine have ZERO references anywhere outside their own file.** The only dynamic-import machinery in `app/` is the strategy registry and a torch feature-probe; neither touches these. Totals: 318 modules, 206 reachable, **112 unreachable**.

**This bears on two standing questions.** *"All employees working autonomously?"* and *"Free llm being used?"* — for the **backend**, no: this subsystem never starts. **Important qualifier: the GitHub Actions agent fleet under `.github/scripts/` is a separate system and does run**, with its own `llm_common` cascade. So the agents you see posting to Discord are real; the backend's parallel implementation of the same idea is not.

- [ ] **[USER DECISION] Wire it up, or delete it.** **Deliberately not switched on.** Unlike the risk gate, the exit path and the ML features — all of which were *supposed* to be running — starting these means the backend begins **generating strategies and modifying itself autonomously**. That is a policy decision with real blast radius, not a wiring bug. Options: (a) wire selected modules into `lifespan` behind a flag, (b) delete as superseded by the Actions fleet, (c) leave dormant. Needs your call.
- [x] **Guarded meanwhile.** `backend/tests/unit/test_module_reachability.py` (5 tests) walks the import graph and fails if a **new** orphan appears in `tasks/`, if a known orphan silently becomes live without being removed from the list, or if total unreachable modules exceed a ceiling. Verified to fire: removing one entry from the baseline produces `new unreachable task modules: ['tasks/task_queue']`. It also asserts a sanity floor (`risk/manager`, `price_feed`, `brokers/alpaca`, `bots/engine` must be reachable) so a broken graph walk fails loudly instead of silently reporting everything as fine.

## 🚨 PRINCIPAL REVIEW 2026-07-27 (cont.) — nothing enforced a stop-loss

> Second pass. After the risk gate, I swept `app/` for the *shape* of that bug — public
> functions nothing references — because finding it twice by hand meant there would be more.
> 35 undecorated unreferenced functions; the exit path was the worst of them.

- [x] **[P0] `PositionMonitor` was never started — every strategy stop-loss was recorded and none was enforced.** `start_position_monitor()`'s docstring says *"Factory function called from scheduler.py"*. scheduler.py has no such job, and **nothing anywhere constructed a `PositionMonitor`**. The third instance of the identical lying-docstring pattern (after `start_strategy_runner` and `start_price_feed`).
  What makes it worse than dead code: the **producer was running the whole time**. `strategy_runner.py:308` writes `pos_exit:<symbol>` to Redis on every fill — `stop_loss`, `take_profit`, `peak_price` — under the comment *"Store exit config in Redis for position_monitor.py"*, with a 24h TTL. Configs were written, expired, and re-written, forever, read by nobody. The whole `execution/position_exit.py` `CompositeExit` engine (trailing stops, ATR stops, time stops, regime exits — 15KB) was reachable **only** from that never-started module.
  Now started from `lifespan` as a supervised 30s loop. Bot positions were separately covered by the `bot_exit_checker` scheduler job; this is the strategy-runner path, which had nothing.
- [x] **[P0] Every Redis price read in the codebase used a key nothing writes.** `set_price()` writes `price:<exchange>:<symbol>`. All three readers built `prices:<symbol>` — which is the **WebSocket topic** (`ws/prices.py`), not a Redis key. Two namespaces one character apart, and **a miss is indistinguishable from a cold cache**, so each reader silently took its fallback:
  - `bots/engine._fetch_current_price` → yfinance `period="2d", interval="1d"`. So the **live, currently-running** `bot_exit_checker` job has been evaluating intraday take-profit and stop-loss against a **daily close**. A daily bar cannot tell you whether an intraday stop was breached.
  - `tasks/position_monitor` → a broker quote per position per tick (dead anyway).
  - `api/v1/positions` → `pnl_pct` always `None`.

  Root cause was documentation: `backend/app/tasks/CLAUDE.md` listed `prices:<SYMBOL>` in its "Redis Key Schema" table. The code matched the docs rather than reality. Fixed the table too, with the warning attached.
  Added `redis_client.price_key()` / `exchange_for()` as the single key builder, routed the writer and all readers through it, and made the intraday bar the first yfinance fallback rather than the daily one.

**Together with the inert risk manager, this is a complete account of how a paper account reached −$8,287.81:** nothing capped position size, nothing halted on drawdown, nothing enforced a stop, and the one exit job that *did* run was pricing stops off yesterday's close.

**Tests:** `backend/tests/unit/test_exit_path_wiring.py`, 8 tests. Verified against the pre-fix tree with the new helpers kept in place so individual tests could be seen to fail: **5 of 8 fail**, the other 3 being properties of the new helper itself. One of these tests was rewritten after its first draft matched **its own explanatory comment** quoting the old code — the same comment-is-not-a-call-site trap as the reachability guard; it is now AST-based.

- [ ] **[P1] The unreferenced-function sweep found more.** Confirmed unreferenced repo-wide (not just in `app/`), grouped by what is actually at stake:

  **CORRECTION to a first draft of this entry.** I wrote that root `CLAUDE.md` principle #5 — *"Walk-forward only: no in-sample-only backtests are accepted as valid"* — is "enforced nowhere". **That was wrong, and I checked it only after writing it down.** There are *two* walk-forward implementations and I had found the dead one:
  - `app/backtest/walk_forward.walk_forward()` — **wired and live**, called from `api/v1/backtests.py:236` and `.github/scripts/ml_experiment.py:152`, with an overfit gate using `deflated_sharpe_ratio` (which *is* used in production, `walk_forward.py:71`). The principle IS enforced for strategy backtests.
  - `app/ml/training/walk_forward.walk_forward_validate()` (125 lines, torch-specific) — **dead**. So the principle is enforced for strategy backtests and *not* for ML model training, which is the narrower and accurate claim.

  Genuinely dead, each referenced only by its own unit test (i.e. tested, never run in production): `probability_of_backtest_overfitting()` and `probabilistic_sharpe_ratio()` (PBO/PSR overfit detection), `monte_carlo_simulation()`. `run_stress_tests()` has no reference at all, not even a test.

  **RE-AUDITED 2026-07-29, and this entry contained an error.** It said `app/backtest/walk_forward.walk_forward()` is called from *both* `api/v1/backtests.py:236` **and** `.github/scripts/ml_experiment.py:152`. It is not — `ml_experiment.py:152` calls a **same-named function defined locally at its own line 96** (a GBM-specific implementation). The app-level function has **one** production call site, so the root principle rests on a single wire. There are therefore **three** distinct walk-forward implementations: the app backtest one (live), `ml_experiment`'s own (live), and `app/ml/training/walk_forward_validate()` (dead — its only textual reference is a string inside its own `ImportError`). `deflated_sharpe_ratio` is at `walk_forward.py:95`, not `:71` — line drift only, still genuinely wired.

  What ML training does *instead* of walk-forward: `ml_retrain.retrain_model` → `train_lstm.train` uses a **single chronological holdout** (`val_frac=0.15`, `shuffle=False`, contiguous slices). That is a legitimate ordered split, not a leak — but it is not what principle #5 says, and the function written to make it so is dead. Pinned by `backend/tests/unit/test_walk_forward_coverage.py` (6 tests): the one live call site must stay wired *and called* (AST-checked, not substring-matched), DSR must stay applied, and the ML gap stays visible until someone closes it.
- [x] **[P1] Validation maths that is tested but never run is not validation.** `probabilistic_sharpe_ratio()` and `monte_carlo_simulation()` now run on every walk-forward, **reported but NOT gated** — changing which strategies clear the promotion bar is a risk decision, not a wiring fix, so `is_robust` is untouched and pinned by a test.
  PSR complements DSR rather than duplicating it: DSR corrects for **multiple testing** (best-of-n luck) from the dispersion of window Sharpes; PSR corrects for a **short, non-normal** track record (n_obs, skew, kurtosis) on the combined estimate. A strategy can pass one and fail the other.
  **Two bugs found while wiring, both in my own work or in the dead code:**
  - My first draft passed an **annualised** Sharpe to PSR. Its contract is that `observed_sr` must be on the same frequency as the moments, and skew/kurtosis are daily — annualising inflates it ~16× and wrecks both the denominator and the z-score. Fixed to the per-period Sharpe and pinned by a test that would fail on the annualised form.
  - **`MonteCarloResult.p95_max_dd` is the LUCKY tail, despite reading like a risk number.** `max_dd = dd.min()`, so drawdowns are negative, and the 95th percentile of a negative series is the *mildest* drawdown. Surfacing it as "the bad case" would have understated risk by construction. Added `p5_max_dd` (the severe tail, consistent with `p5_sharpe` already meaning the unlucky end) and report that. Nothing consumed `p95_max_dd`, so nothing broke.

  **Not wired, and why** — the same judgement as the ML features rather than wiring everything reachable:
  - `probability_of_backtest_overfitting()` needs performance across N **configurations**; `walk_forward` runs one config across windows. Wrong call site — it belongs to a parameter sweep or the strategy-selection loop.
  - `run_stress_tests()` needs the price history to actually span the named crisis windows; on a 2-year backtest nearly every scenario returns `period_covered=False`.

  Monte Carlo is skipped rather than faked below 60 OOS observations (`mc_simulations` stays 0), and the whole diagnostic block is exception-isolated — it hangs off a backtest that already succeeded and must not fail it.
  Tests: `backend/tests/unit/test_walk_forward_diagnostics.py`, 10 tests.
- [x] **[P1] `configure_logging()` was never called — production logging ran on structlog's library defaults.** One textual occurrence in the package: its own `def`. Verified by reading `structlog.get_config()` at runtime rather than inferring from source, before and after:

  | | before | after |
  |---|---|---|
  | renderer | `ConsoleRenderer` | `JSONRenderer` |
  | wrapper_class | `BoundLoggerFilteringAtNotset` | `BoundLoggerFilteringAtInfo` |

  Two consequences, both live: Render received **unstructured console text**, so nothing downstream could parse a log line; and `…AtNotset` filters **nothing**, so all **105** `logger.debug()` call sites in `app/` emitted on every production run. That is the noise floor underneath *"a bunch of errors are reported throughout day on discord"* — the signal was competing with every debug line in the codebase.
  Called at import in `main.py` rather than inside `lifespan`: module-level code logs before startup, and `static_server.py` imports `app.main`, so that one site covers every entrypoint.
  Tests: `backend/tests/unit/test_logging_configured.py`, 5 tests; **4 fail** on the pre-fix tree, one of them by showing the debug line actually reaching stdout.
  **The full suite caught a real consequence of this change**, which is the point of running it: `test_a_wholly_dead_cycle_is_escalated` asserted `consecutive_dead_cycles=2` — ConsoleRenderer's `key=value` form, which it only ever saw *because* logging was misconfigured. It also passed in isolation and failed in the suite, i.e. it was silently order-dependent on whether an earlier test had imported `app.main`. Rewritten to accept either renderer rather than weakened: the behaviour under test is the count being reported, not how structlog punctuates it.
- [x] **[P1] Three ML feature builders were implemented, unit-tested, and called from nowhere — every model trained without them.** Resolved with a judgement call rather than by wiring all three, because they are not equivalent:
  - **`add_microstructure_features` — WIRED.** Pure OHLCV arithmetic (order-book imbalance proxy, spread in bps), no network, deterministic. Verified not to look ahead: the label is `close.pct_change(h).shift(-h)`, so bar *t* predicts *t+h* and bar *t*'s own OHLC is known at prediction time — the same convention `add_technical_features` already relies on. Pinned by a test that perturbs future bars and asserts past features are unchanged.
  - **`add_alternative_features` — DELIBERATELY NOT WIRED.** It calls the Binance API. Feature engineering runs inside backtests and training, which must be deterministic and work offline; a network call there makes results unreproducible and breaks sandboxed runs. Pinned by `test_feature_engineering_makes_no_network_calls`, so a future change has to argue with a test rather than a comment.
  - **`add_sentiment_features` — not wired.** Takes a caller-supplied `fear_greed_history` (correctly lagged 1 bar). With no caller it emits constants, which widens the matrix for no signal. Needs a data source decision first.

  **Timing mattered here.** Widening `FEATURE_COLS` changes the model input width, which would silently break inference for any already-trained model. Checked first: **zero** `.pt`/`.pth`/`.pkl`/`.onnx` artifacts anywhere in the repo, and live `/health/detailed` reports `ml_models: {count: 0}`. Nothing to break — and doing it *before* the first model is trained is the only cheap moment.
  Tests: `backend/tests/unit/test_feature_set_wiring.py`, 8 tests; **6 fail** on the pre-wiring tree.

## 🚨 PRINCIPAL REVIEW 2026-07-27 — the risk engine was never switched on

> User: *"Improve the code base. Deep review from a principal engineer of everything."*
> First pass, following the money path (risk → execution → broker). One P0 and three real bugs, all in the gate every order is supposed to pass through.

- [x] **[P0] `RiskManager.check_order()` never ran in production. Not once.** Every documented risk control — position cap, drawdown circuit breaker, correlation cluster limit — was inert in the live system. Three independent facts, each verified by AST rather than by reading:
  1. `orders.py` gates on `getattr(request.app.state, "risk_manager", None)` and **skips the check when it is None**. Nothing in the entire codebase ever assigned `app.state.risk_manager` — zero assignments.
  2. `main.py` constructed the always-running strategy runner with `risk_manager=None` **literally**.
  3. The *only* `RiskManager()` construction in the package sat in `strategy_runner.start_strategy_runner()` — a function whose docstring states it is *"registered as a supervised background task in main.py"*. **main.py never called it.** It had exactly one textual occurrence in `app/`: its own `def`.

  So the sole code path that built a risk manager was dead, and both live paths were handed `None`. Wired it in `lifespan`, passed it to the strategy runner, and **deleted the dead 95-line function** rather than leaving the trap in place.

  **Why the existing test suite passed the whole time:** `test_security_invariants.py` asserts the *string* `"check_order"` appears in `orders.py` at least twice. That proves the call is typed, not that it executes. A textual invariant cannot distinguish live wiring from dead code — which is the entire lesson here.
- [x] **[P0] A wired gate with no data is still not a gate.** The manager seeds `initial_equity=100_000` and only leaves it via `update_equity()`. Nothing called it, so the drawdown breaker would have seen a single data point forever (one point has no drawdown) and the position cap would have measured against a fabricated NAV. Added `_risk_state_sync` — a supervised 60s loop feeding real broker equity and positions, registered in `bg_tasks` so shutdown cancels it.
- [x] **[P0] …and it would have failed OPEN on the one condition that must halt.** `update_equity()` **raises** on a negative value, and the live Alpaca paper account is at **−$8,287.81**. A naive sync loop lets that raise into its except-and-log, leaving the manager on its seeded $100k and **approving orders forever**. Negative equity is now clamped to `0.0` so `check_order`'s `equity <= 0` halt actually fires. Verified by reverting to the naive version and watching the test fail with `ValueError: Equity cannot be negative`.
  *This also answers the open question from the desk review — "a paper account going $8k negative means the risk layer did not stop it, which is its own investigation."* It did not stop it because it was never running.
- [x] **[P1] The position cap swallowed the correlation check.** `check_order()` returned `RiskDecision(True, "size capped")` immediately on breaching the size cap — **before** the correlation cluster limit. Only oversized orders reach that branch, so the largest orders, the ones most able to breach a concentration limit, were precisely the ones that skipped it. Capping now adjusts the quantity and falls through, and the cluster check runs against the *capped* notional.
  The existing `test_correlation_cluster_limit_blocks_overconcentration` passes `max_position_pct=0.50` — a value chosen so the cap does not trigger. The test worked around the bug instead of catching it.
- [x] **[P1] Every market order was sized against a hardcoded $100.** `price = request.limit_price if request.limit_price is not None else 100.0`. Market orders carry no limit price, so **1 BTC scored as a $100 position** — 0.1% of a $100k account instead of 60% — and SHIB at $0.00001 scored ten-million-fold too large. The cap was arithmetically incapable of firing on the crypto desks. Added `update_prices()` (a mark cache, non-raising, rejecting NaN/inf/≤0 so a bad tick cannot silently un-cap a symbol) and `_reference_price()`: limit price → last mark → **None**. When genuinely unpriced the order is allowed but the notional checks are **skipped visibly** and counted in `unpriced_orders`, rather than enforced against an invented number. Fabricating a price to keep a check "passing" is worse than admitting the check did not run.
- [x] **[P2] Cold start ran no crypto strategy at all.** `main.py` carried its own inline default watchlist (equities only: SPY/QQQ/AAPL/TSLA) that shadowed the richer shared `DEFAULT_ACTIVE_STRATEGIES` constant — which includes `btc_eth_stat_arb`. Two copies of the same defaults, and the live one was the worse one. main.py now uses the shared constant.

**Tests:** `tests/unit/test_risk_gate_wiring.py`, 15 tests. Verified against the pre-fix tree: **9 of 11** original tests fail on old code. The 2 that pass are deliberate regression guards on behaviour I preserved.
Two of these tests were themselves rewritten after failing this bar — one used `str.count()` to detect unreachable functions and was defeated by a comment in `main.py` naming the very function it was meant to catch (now AST reference resolution); the other claimed to protect equity updates from positions failures but passed against both versions, because equity is fetched first (now asserts distinct, triage-actionable log lines instead).

**What is now live, stated precisely — "the risk engine is on" would be an overclaim:**

| Control | Before | After |
|---|---|---|
| Circuit breakers (global + arb) | never ran | **live** — fed by `_risk_state_sync` |
| Zero/negative-equity halt | never ran | **live** |
| Position size cap | never ran | **live** — real NAV, real marks |
| Correlation cluster limit | never ran | **live** — see below |

- [x] **[P1] The correlation cluster limit could not fire either — `update_returns()` had no caller.** `check_order` guards it with `if self._clusters:`, and `_clusters` is only ever populated by `update_returns()`, which — exactly like `app.state.risk_manager` — was called from **nowhere in `app/`**. Wiring the manager in did not fix this; the limit stayed inert for precisely the same reason the whole gate was inert, one level down.
  Fixed by deriving returns from the mark stream the price feed now produces, with two deliberate choices:
  **(1) Downsampling.** Marks arrive every ~2s. Correlating 2-second ticks measures microstructure noise, not the co-movement `compute_correlation_clusters(threshold=0.70)` is about, so marks are sampled to one observation per `sample_interval_seconds` (default 300s). **(2) Throttling.** Clustering is O(symbols²) and sits directly behind the live price feed, so it recomputes at most every `cluster_refresh_seconds` (default 900s) rather than per tick. Sample history is a bounded `deque` — this runs for the process lifetime.
  Verified by disabling `_maybe_refresh_clusters` and confirming both the cluster-building test and the end-to-end blocking test fail. The end-to-end test sets no `_clusters` by hand: marks in, concentration limit out.
- [~] **[P1] `factor_exposure.py` and `var.py` are documented as risk gates but are not wired into `check_order()`.** `risk/CLAUDE.md` diagrams five checks. **VaR is now wired; factor exposure is not** — four of five now run.
  **VaR gate** — "block if 1-day 99% VaR > 2% of NAV", the documented rule. Two things made this easy to get wrong, and both are pinned by tests:
  - **Fail-open on thin data.** `historical_var()` returns a *default* `var_99` of **0.03** below 10 observations. Wired naively against a 2% limit, a cold start would have blocked **every order** until samples accumulated — a fleet-wide halt dressed as a risk control. The gate checks both the sample count and the returned `method` sentinel, and returns "no opinion" rather than a block.
  - **Units.** `_risk_state_sync` polls equity every 60s, so returns built from every update are **1-minute** returns and a 99% VaR off those is ~30× too small against a 1-day limit. Equity is downsampled to `var_sample_interval_seconds` (default hourly, matching the existing snapshot cadence) and scaled to one day by square-root-of-time. Stated honestly in the code: sqrt-time assumes i.i.d. returns and **understates** tail risk under positive autocorrelation — a floor, not a ceiling.

  A clamped-to-zero equity (the negative-NAV halt signal) is deliberately *not* recorded as an observation — a −100% return would poison the VaR window for as long as it stayed in it. The window is bounded, and a VaR failure means "no opinion", never a crash, since this sits in front of every order.
  Tests: `backend/tests/unit/test_var_gate.py`, 12 tests; **11 fail** on the pre-wiring tree.
- [ ] **[P1] `factor_exposure.py` is the one remaining diagrammed gate still unwired.** `compute_factor_exposure()` needs portfolio returns **and an aligned benchmark (SPY) series at the same cadence**. The portfolio series comes from hourly NAV samples; marks are sampled on a different schedule and SPY is not guaranteed to be in the universe at all. Feeding it a misaligned series would produce a confident, wrong beta — worse than no beta. Left out deliberately, with `test_factor_exposure_is_still_honestly_unwired` so the gap stays visible rather than being quietly forgotten, which is exactly how all five ended up unwired.
  **Trap to avoid when wiring VaR:** `historical_var()` returns a *default* `var_99=0.03` when it has fewer than 10 observations. Wired naively against the documented "block if 1-day 99% VaR > 2% of NAV", a cold start would block **every order** until 10 observations accumulate — a fail-closed halt of the entire fleet, dressed as a risk control.
- [x] **[P2] …and the mark cache needed a producer, or it was the same bug in a new place.** `update_prices()` alone would leave every market order on the "unpriced" path, i.e. still uncapped — just visibly so. Wired the price feed as the producer via an optional `on_mark(symbol, last)` sink on `run_price_feed`/`_fetch_and_publish` — a plain callable, not a `RiskManager` reference, so the feed stays decoupled from the risk layer. The sink is exception-isolated: a broken risk consumer must not cost the Redis write or the WebSocket broadcast. `unpriced_orders` remains as the measurement of any residual gap.

## 🧪 TESTS & CI — 2026-07-27
- [x] **[P0] ~15 ML model implementations had ZERO CI coverage.** CI runs `pytest tests/` with `--ignore=tests/unit/test_ml_models.py --ignore=tests/unit/test_a3c_lstm.py` because torch is not in the CI dependency list. So the entire `app/ml/models/` package was unexercised on every PR — and it showed: `lorentzian_knn.backtest_signals()` (four wrong keywords, both required fields missing) and `itransformer.evaluate()` (omitted the required `EvalMetrics.sharpe`) both shipped structurally broken and raised `TypeError` on **every** call. Neither was caught by a test; both were found by static analysis, because static analysis is the only thing that *can* run without torch.
  Added `tests/unit/test_ml_model_contract.py`, which checks the contract the same way it was found: parse each model module, verify every `AbstractModel` subclass declares the `@abstractmethod` set, verify every module parses, and verify no `evaluate()` returns a bare dict instead of `EvalMetrics`. No torch, no GPU, no fixtures — which is exactly why it runs on every PR. Verified against a planted regression: removing `evaluate` from iTransformer produces `itransformer.py::iTransformer missing evaluate`.
  It also cross-checks itself: `REQUIRED_METHODS` is asserted against the `@abstractmethod`s actually declared on `AbstractModel`, so the list cannot silently drift from the base class.
  **Scope, stated honestly:** this proves a method EXISTS. It cannot prove the body is right — `test_dataclass_kwargs.py` covers the return-value construction that broke in both real cases. Together they cover the failure mode; alone, neither does.
- [x] **[P1] `test_strategy_contract.py` was doing REAL network I/O while its fixture was named `no_network`.** Fixed — and the diagnosis in the earlier draft of this entry was wrong, so it is corrected here rather than left standing.
  I had written that it "should be marked `@pytest.mark.network` and the fetch stubbed". That would have *removed* the thing being tested — the contract is precisely "fail soft when the data source is unavailable". The real defect was that the unavailability was never actually simulated: the fixture patches Python's `socket`, but **yfinance fetches through `curl_cffi` → libcurl, which never touches Python's socket module**. `app/strategies/_failsoft.py` already documented this exact fact; the fixture just didn't act on it.
  So every fetching strategy hit the live Yahoo API, with real retry-backoff, on every run.

  | | before | after |
  |---|---|---|
  | wall | **7m15s** | **4.5s** |
  | CPU | 5.75s | 5.07s |
  | result | env-dependent | **119 passed**, hermetic |

  The near-identical CPU time is the proof: ~99% of the wall clock was waiting on Yahoo, not computing. In CI this is worse than it looks — `--dist loadfile` puts all 115 parametrised cases on **one** worker, serialised, while three sit idle, and a Yahoo outage could redden a PR for reasons unrelated to its diff.
  Fix: block `curl_cffi.requests` alongside `socket`. The contract is unweakened — strictly better tested, because the failure mode it asserts is now genuinely simulated.
  Guarded by `test_the_network_kill_actually_reaches_yfinance`, verified to fail (`DID NOT RAISE OSError`) against the socket-only fixture. A silent regression here would be invisible — the suite would still pass, just slowly and non-deterministically — which is exactly how this survived.
- [x] **[P2] No coverage measurement anywhere.** Now measured in CI and published to the PR summary. **Baseline at introduction: 51% of 28,261 statements, with 37 modules at 0%** (3,594 statements never executed by any test).
  **Reported, not gated** — deliberately no `--cov-fail-under`. Picking a threshold is a policy decision, and a number chosen to pass today teaches nothing. The actionable output is the 0%-module list, which is the **same "implemented but never runs" class** that the risk gate, the exit path, the ML feature builders and `configure_logging` all turned out to be — found automatically now, instead of by hand-rolled AST sweeps.
  Cost: suite goes 72s → ~90s. The summary-parsing commands were validated against real pytest output before shipping, not assumed.
  Sample of the 0% list, all of it worth a look: `ml/training/walk_forward.py` (the torch walk-forward already known dead), `ml/registry.py`, `ml/serving/serve.py`, `ml/serving/ab_router.py`, `tasks/agent_bus.py`, `tasks/task_queue.py`, `tasks/strategy_auction.py`, `tasks/stock_scanners.py`, `comparison/engine.py`, `options/wheel.py`, `options/flow.py`.

## 🔴 DEEP REVIEW 2026-07-27 — "why are there no trades" answered end to end
> User: *"Trade desks are very weak, there should be hundreds of trades. Fix Supabase yourself, it was working before."*
> Both answered with hard evidence. Every finding below came from live logs and the live API, not from reading code.

### The desks were never broken — they were bankrupt, and nothing said so
Live crypto-desk log, 08:40 UTC: nine desks generated signals, and:
```
403: insufficient balance for USD (requested: 134.58, available: 6.71, balance: -8287.81)
422: asset MKR/USD is not active
place_order failed SHIB/USD buy: float division by zero
Done. 1 orders placed across 9 desks.
```
- [x] **[P0] `cash_capped_notional()` was never applied on the desk path.** The function exists precisely to prevent "403 insufficient balance" (its docstring records the *previous* round of this bug), but `run_desk` passed `desk.notional_usd` **raw**. Every order asked for $135 against **$6.71** of buying power. Two order paths existed and only the Kelly one was capped. Now capped, and an unaffordable signal is skipped honestly instead of being sent to be rejected.
- [x] **[P0] Sub-cent crypto divided by zero.** `round(limit_price * 1.001, 2)` flattens SHIB (~$0.00001) to `0.0`, and the next line is `notional / lp`. The `ZeroDivisionError` surfaced as a generic "place_order failed", so it looked like a broker problem. Added `_price_precision()` — 2dp for equities, scaling to 8dp for sub-cent crypto — plus an explicit guard that refuses rather than divides. BTC and SHIB cannot share a rounding precision; nine orders of magnitude apart.
- [x] **[P1] A desk that funded nothing still reported success.** It ended with a tidy `✓ Place orders` and no further comment — which is how nine desks ran for weeks against a **negative account balance**. Now counts unfunded vs broker-rejected signals and emits a `🚨 DESK … PLACED NOTHING` line with the actual buying power.
- [ ] **[USER] The Alpaca paper account is at `-$8,287.81` with `$6.71` available.** No sizing fix can trade out of that — it needs a paper-account reset in Alpaca. **Also worth asking how it got there:** a paper account going $8k negative means the risk layer did not stop it, which is its own investigation.
- [x] **[P2] Inactive assets are still requested** (`MKR/USD is not active`, 422). **DONE, and the original premise here was right.** The proposed fix — filter the universe against Alpaca's active-asset list before signals are generated — was already implemented as `_filter_tradable_crypto`, but it was only reachable from `run_desk()`, which has **zero call sites**, so it had never run. Wired into the real pipeline 2026-07-28 23:45; confirmed on the first live run (`30409299307`): `ⓘ Crypto: skipping 1 non-tradable pair(s): MKR/USD`, and `bars_fetched` 98 → 97. Alpaca does **not** list MKR/USD as tradable — my earlier claim that its metadata contradicted its order engine was wrong, and was inferred from a missing log line that a dead code path could never have printed. See CONTINUITY.md 2026-07-29 00:15.
- [ ] **[P1] The file-based strategy trimmer is inert at the source.** `fill_tracker.py` writes `backend/performance_log/strategy_performance.json`, but `fill-tracking.yml` never committed it, so on an ephemeral runner it was computed and discarded — the file has never existed in the repo. Three consumers read that path and all three did nothing: `strategy_trimmer.py`, `strategy_auto_tuner.py`, and the desk's trims lookup. Commit step added 2026-07-29 00:15. **A fourth link was found 2026-07-29 01:20:** the desk tagged orders `qe-{strategy[:10]}-…`, so attribution keys were truncated (`vol_of_vol`) while the desk checks full registry names (`vol_of_vol_timing`) — they could never match. The truncation was also ambiguous: `commodity_` merged three strategies' P&L and `supertrend_rsi_tv` was booked against `supertrend`, meaning a strategy could be retired for another's losses. Full names now emitted. **Cadence fixed 2026-07-29 03:40:** the producer ran `0 22 * * 1-5` (1x/day, weekdays) while the trimmer reads `41 */6` (4x/day, every day), so most trimmer runs saw stale data and weekend runs saw data up to 3 days old. The `workflow_run: [CI]` trigger is also structurally dead here — agent-branch CI is dispatched by auto-pr.yml with GITHUB_TOKEN, and GitHub does not chain workflow runs from GITHUB_TOKEN events (measured: 1 firing in 3h). Now `11 */6 * * *`. **✅ LANDED 2026-07-29 10:19** — `strategy_performance.json` is in the repo (`336afa69`), first content `avellaneda trades=10 win_rate=0.60 total_return_pct=-7.92`. The blocker turned out to be inside the commit step itself: `git diff --quiet` was tested BEFORE staging and cannot see an untracked file, so it could never make the file's first commit (#1215). `evaluate_trim` returns True on that row, so the trimmer retires it at the next `41 */6` slot. The key is the legacy truncated `avellaneda`, which the desk would never match, so `_trimmed_strategies()` now expands truncated keys against the registry (unambiguous only). **✅ VERIFIED IN PRODUCTION 2026-08-05 19:00** (pending since 07-29). Runs `31031318516` and `31035271962` both log:
```
✂ 8 strategy(ies) retired by the trimmer will not trade: avellaneda, avellaneda_stoikov_mm,
  options_pcr_reversal, realized_vol_asymmetry, stat_arb_e, stat_arb_etf, vol_of_vol, vol_of_vol_timing
```
and `strategy_trims.json` carries the reasons — `avellaneda` "cumulative return -7.9% ≤ -5.0% over 10 trades"
(07-29), `vol_of_vol` "no edge: win_rate 20% < 35%" (07-31). **The truncation-expansion fix is what this
proves**: `avellaneda` is the legacy *truncated* key the desk could never have matched, and it is retired
alongside the full `avellaneda_stoikov_mm` — so both the expanded legacy key and the registry name resolve, and
neither strategy has placed an order since. The whole four-link chain (producer commits the file → cadence
aligned → full names emitted → truncated keys expanded) is live. ⚠️ Note this is the *redundant* pruning path: attribution-weight pruning (`✂ pruned by attribution`, weight 0.0 from `/api/v1/leaderboard/live`) is live and has been stopping losing strategies all along.

### The ML pipeline was never running — five workflows were silently disabled
- [x] **[P0] Duplicate YAML key disabled five workflows outright**, including **the entire ML pipeline**. `agent-health-monitor`, `channel-monitor`, `daily-employee-review`, `model-audit` and `run-experiments-agent` each carried `OPENROUTER_API_KEY` **twice in the same `env:` block**. PyYAML silently accepts duplicate mapping keys (last wins); GitHub Actions refuses to parse the file, creates a run with **zero jobs**, and marks it `failure`. No log, no error — no job ever starts. 100% of their runs failed for weeks, and `ml_models: {count: 0}` follows directly: nothing ever trained a model because `run-experiments-agent` could not start.
  The tell was in the API, not the logs: GitHub reported those five workflows' `name` as the **file path** (`.github/workflows/model-audit.yml`) rather than the declared name — its fallback when it cannot parse a workflow. Every other workflow shows its real name.
  Fixed all five. Added `.github/tests/test_workflow_yaml_valid.py`, which parses every workflow with a loader that rejects duplicates the way Actions does, and asserts each declares both jobs and triggers. **`yaml.safe_load` passing is not evidence a workflow is valid** — that is precisely why this went unnoticed. Verified to fail against the pre-fix tree.
- [x] **[P1] …and once parseable, `run-experiments-agent` still would not have run.** Its only triggers were `schedule` + `workflow_dispatch`, and GitHub drops/heavily delays free-tier cron — the reason this repo already routes `oa-scout` and `model-audit` off `workflow_run` from CI, which is *not* dropped. Added the same `workflow_run` trigger, plus a gate step that queries the last **successful** run of the workflow and skips unless it is over 20 hours old, so CI firing many times a day still yields one experiment run a day. A manual dispatch always runs, and a failed API lookup fails *open* (runs) rather than silently skipping forever.
- [~] **[P1] Free-LLM cascade is one provider deep — now MEASURED and alarmed, still one deep.** Only `GEMINI_API_KEY` is populated; `GROQ`, `DEEPSEEK`, `SAMBANOVA`, `CEREBRAS`, `TOGETHER`, `HYPERBOLIC`, `NVIDIA_NIM` are all empty in the live workflow env, so a Gemini rate-limit means no LLM at all.
  **Correction to an earlier note in this file:** I wrote that `_record_metric` "has never written a line". That was wrong. It is called from **7 sites** and writes on every run — but `.github/state/llm_metrics.jsonl` is **explicitly gitignored** (`.gitignore:83`) and no workflow reads it, so the metrics are written inside the ephemeral runner and discarded when the job ends. Full observability, collected every run, visible to nobody. That is a worse failure than never collecting it, because it looks like instrumentation exists.
  Fixed the visibility rather than the file: `llm_cascade_report.py` probes each provider live, writes a table to the GitHub step summary, and posts a **deduped** alert to `#infra-alerts` when the cascade is dead or fewer than 2 providers answer — because one working provider is a single point of failure, not a cascade. Wired into the hourly `error-triage` workflow rather than adding another. 8 tests. **Adding a second free key is still a user action.**

### The monitoring — and the desks' own strategy weighting — read a dead service
- [x] **[P0] Every agent-side URL defaulted to the STALE Render deployment.** `smoke-test.yml`, `keep-alive.yml`, `smoke_test_live.py`, `third_party_monitor.py` and — worst — **`desk_order_placer.py`** all fell back to `quantedge-api-agb8`, because they read `vars.RENDER_API_URL || <hardcoded agb8>` and that repo variable **is not set**. The desk placer's `_API_BASE` feeds `_fetch_performance_weights()`, so **the desks were weighting and pruning strategies from a stale deployment's leaderboard**. Repointed all five to `quantedge-api-9jz0` (61/61 template parity, Redis connected, current health schema). The `vars.RENDER_API_URL` override is retained so it can still be redirected centrally. Frontend + `render.yaml` OAuth callback deliberately NOT touched — those need the Google console changed in lockstep.
  Immediate effect: the live smoke test went from **2 failures to 1**, and `deploy parity` now passes (it had been reporting `live=29 repo=61 STALE DEPLOY` for weeks — against a service nobody deploys to).
- [x] **[P1] `os.environ.get(k, default)` returns `""` for a set-but-empty var**, so an empty `SMOKE_BASE_URL`/`QUANTEDGE_API_URL` produced a *relative* URL and `ValueError: unknown url type: '/api/v1/health'` rather than falling back. Switched both to `os.environ.get(k) or default`. Found by running the smoke test with the override blanked — exactly what `vars.RENDER_API_URL` being defined-but-empty would do.

### Supabase — "it was working before" has a mechanism, and it was my own script
- [x] **[P0] `render_probe_pooler` could destroy the password it rewrites.** It reads the password with `urlparse`. If the stored `DATABASE_URL` holds an **unencoded** password containing `#` or `?` — exactly what a human pasting into the Render dashboard produces — urlparse treats them as fragment/query markers and truncates:
  ```
  Abc!@#$%^&*()  ->  Abc!          a:b@c#d?e  ->  a:b
  ```
  The script then percent-encodes that stub and **PATCHes it into Render**, replacing a working credential with a 4-character prefix. The symptom is `password authentication failed` — indistinguishable from a rotated password, and self-inflicted. Added `password_roundtrips()`, which compares against the **raw** credential text rather than a re-encoded copy of what urlparse already returned (re-encoding always round-trips, which would make the check vacuous). The patch path now refuses and prints the encode command. **This cannot recover a password already destroyed** — that still needs one reset in the Supabase dashboard — but it ends the loop where every probe run corrupts it again.

## 🔴 DEEP REVIEW 2026-07-25 — Slack removal uncovered a class of SILENT MESSAGE LOSS
> User directive: "Remove slack completely. Make discord better. Improve tests."
> Ripping Slack out turned up something worse than dead code: **large parts of the
> agent fleet had been posting into a void for weeks.** Every one of these was a
> silent failure — no exception, no red CI, just no message.

- [x] **[P0 FOUND+FIXED] `llm_common.slack_post` silently discarded EVERY message from 27 scripts.** It returned `{}` when `SLACK_BOT_TOKEN` was unset, and that token has been unset since the free-plan quota died 2026-06-29. So 27 agent scripts — collective learner, continuous improver, deep code review, frontend design agent, employee intros, … — believed they were reporting and were not. Replaced with `chat_post()`, which delegates to `notify.post()` (Discord). Same for `slack_read_thread`/`slack_read_channel` → `chat_read_channel()` via `notify.read_channel_recent`.
- [x] **[P0 FOUND+FIXED] The hourly employee report has NEVER worked.** `scheduler._slack_employee_report` called `discord.post_message(...)` — a method that does not exist on the notifier and never did (not on the old `SlackClient` either). Every hourly run raised `AttributeError` straight into a `except Exception: logger.error(...)`, so the failure was logged as noise and the report was never delivered. Rewritten to use the real `send()` interface with structured fields; job renamed `slack_report` → `employee_report`.
- [x] **[P1 FOUND+FIXED] `free_agent_engineer` and `gemini_task_runner` dropped their own reports** — both had a local `_slack_post` that returned early with no token. Now `_chat_post` → `notify.post`.
- [x] **[P1 FOUND+FIXED] Token guards were about to block Discord too.** ~10 scripts guarded delivery with `if not SLACK_TOKEN: return`. Left alone after the migration those would have kept short-circuiting forever (the var is never set). Replaced the dead constant with `CHAT_ENABLED = bool(DISCORD_BOT_TOKEN or DISCORD_WEBHOOK_URL)`, so the guard now means "is chat configured" and passes when Discord is.
- [x] **[P1 FOUND+FIXED] `agent_health_monitor` pointed at a script path that no longer exists** after the rename (`slack_agent_team.py`), which would have broken the health monitor at runtime. Repointed to `agent_team.py`.

### Discord quality upgrades shipped in the same pass
- [x] **Rich embeds instead of flattened text.** The old client built a coloured, field-structured Slack attachment and then, on Discord failover, threw all of it away and posted `**title**\n• k: v`. Since Slack was dead, *every* alert rendered as that degraded plain text. `app/notifications/discord.py` posts native embeds — colour per event type, proper fields, footer, timestamp.
- [x] **Slack mrkdwn no longer leaks into Discord.** `*bold*` and `<url|label>` and `:emoji:` render as literal junk in Discord. Fixed in the employee-conversations report, p0 watchdog, strategy trimmer, heartbeat monitor, and in the **LLM prompts** that were instructing agents to "use *bold* — Slack format".
- [x] **`_enabled` is now dynamic.** The old client froze it at construction, so credentials configured after import could never enable notifications without a process restart.
- [x] **Per-channel routing preserved and tested** — bot-token channel resolution first (each alert in its own channel), per-channel webhook override next, catch-all webhook last with a `[#channel]` label so a shared webhook is still traceable.

### Tests added/strengthened
- [x] `backend/tests/unit/test_discord_notifier.py` — **14 tests, first ever coverage of the notification client** (the Slack one had none, which is how the two defects above survived): embed structure/colour/limits, dynamic `_enabled`, channel routing, webhook precedence, bot→webhook failover, typed-helper fields.
- [x] `test_security_invariants.py` — replaced the now-vacuous `/slack/events` signature tests with a **stronger** invariant (every notifications route requires auth) plus a guard that Slack cannot creep back into the backend.
- [x] `test_script_safety.py` — `TestSlackBootstrap` → `TestDiscordChannelCoverage` (same invariant, now against `discord_setup_channels.py`), and `TestSlackTokenGuard` → `TestSlackStaysRemoved` (no script may call the Slack API or read a Slack token). These make the removal **permanent** against the autonomous improver.

### Second pass 2026-07-25 — the first sweep's guard was scoped too narrowly
- [x] **[P1 FOUND+FIXED] Two more dead alert paths.** `deploy-verify.yml` posted deploy-drift alerts to Slack and `exit 0`'d when the token was missing — so **every deploy-drift alert since the quota died was lost**. `secrets-check.yml` did the same with its missing-secrets report. Both now go through `notify.post()`.
- [x] **[P1 FOUND+FIXED] `auto-launch.yml` had a step that would `KeyError` at runtime** — it read `os.environ["SLACK_BOT_TOKEN"]` while its `env:` block only defined the Discord vars. Deleted (it invited a Slack workspace admin to Slack channels).
- [x] **[P1 FOUND+FIXED] A whole directory was missed** — repo-root `scripts/` (distinct from `.github/scripts/`) still held `slack_message_monitor.py` and `slack_invite_all.py`. Deleted; no live callers.
- [x] **Guards widened to close the hole that caused this.** `TestSlackStaysRemoved` now scans `scripts/`, `.github/scripts/`, `.github/tests/` **and** `.github/workflows/` — the first version only checked `.github/scripts/`, which is exactly why the above survived. Repo-wide audit is now 0 Slack API calls, 0 token reads, 0 slack workflows, 0 slack modules.
- [x] **`test_slack_config_present` → `test_discord_config_present`** — the health test asserted a Slack token existed.

### Remaining Slack-related cleanup (cosmetic only — no functional Slack left)
- [x] **[P3] Prose sweep — DONE 2026-07-25.** 499 → 53 mentions, and all 53 remaining are deliberate history ("Slack removed 2026-07-25", "Previously POSTed to slack.com", the guard tests' forbidden strings). **Zero non-historical mentions left.** This was not cosmetic: the leftovers sat inside LLM prompt text, so agents were still being told they were Slack bots and to write Slack-flavoured output. Also renamed the remaining identifiers the first pass missed (`post_to_slack`→`post_to_chat`, `_post_slack`/`_slack_post`→`_post_chat`/`_chat_post`, `fetch_slack_knowledge`→`fetch_chat_knowledge`, `_slack_channel_has_recent_bot_post`→`_channel_has_recent_bot_post`, `slack_name`→`display_name`, `posted_to_slack`→`posted_to_chat`) and fixed more Slack single-star bold that renders literally in Discord. 321 agent tests pass.
- [~] **[P2] `agent_team.py` Slack-era dead code** — first pass DONE 2026-07-25: deleted `_build_summon_blocks` and replaced `_post_with_blocks` with `_post_reply`. Those built Slack Block Kit blocks and passed a `thread_ts` on EVERY agent summon, both of which `chat_call` discards under Discord — so the work was done and thrown away on every reply. Verified no user-visible change: the plain `text` was always the full reply and is what actually landed, and agent attribution comes through `username`. Also corrected a log line that printed `blocks=True` for blocks that were never sent. 321 agent tests pass.
- [x] **[P2] Discord reactions — FIXED 2026-07-25.** Agent acknowledgements (👀 on receipt, ✅ answered, ❌ failed, ⏳ queued) were silent no-ops: `notify._post_to_channel_id` threw away the created message's id and `chat_call` returned `ts: ""`, so there was never anything to react to. Added `notify.post_returning_id()` (returns Discord's real message id) and `notify.add_reaction()` (`PUT /channels/{id}/messages/{mid}/reactions/{emoji}/@me`, with Slack-style names like `white_check_mark` mapped to real Unicode and URL-encoded). `chat_call` now returns the real id as `ts` and routes `reactions.add` through it. Fails soft everywhere — a webhook-only deployment gets no reaction but the message still lands, verified by test. 16 new tests (11 notify + 5 chat_call); 337 agent tests pass.
- [ ] **[P2] `agent_team.py` is still 10.6k lines** — split it; the dead-code pass above only removed the block builder.

## 🚨 MONDAY 2026-07-20 POST-MORTEM — "all desks bad, 0 trades, OA doesn't work" (diagnosed + fixed 2026-07-21)
Evidence-based, three stacked causes:
1. **[FIXED] Loss cap froze the entire book all session** — every market-hours desk run logged `🛑 DAILY LOSS CAP: equity down 2.72% vs prior close (cap 2%)`. The cap compares to Alpaca `last_equity` = FRIDAY's close, so weekend crypto drift on existing positions tripped it before Monday even opened — and it blocked ALL orders including exits (couldn't add, couldn't de-risk). Desks were otherwise perfect: 410 signals, DIA/JNJ/GLD/EWT at conf 1.00. FIX: under the cap, risk-REDUCING orders stay allowed (`is_risk_reducing` vs live Alpaca positions, fail-strict on fetch error); only new exposure is blocked; cap state surfaced in the run log + Discord funnel line. 6 tests.
2. **[FIXED] Bots never evaluated (all 61 last_run_at=None)** — APScheduler interval jobs wait one FULL interval before the first run, and every merge→deploy wipes the ephemeral SQLite + restarts the app, resetting that clock. 1h/1d bots never got to run. FIX: `next_run_time` = boot + 30–150s stagger, so every bot evaluates within ~2 min of every deploy.
3. **[USER ACTION — the remaining one] Bot trades are DB-only and die with every deploy** on the SQLite fallback (desk trades survive because they re-sync from Alpaca 30d). Durable bot P&L requires unpausing Supabase (schema-drift-gate is DONE — the catch-up migration `k6f7a8b9c0d1` for slippage_records IS fields shipped 2026-07-22 and applies automatically on the first boot that reaches Postgres). STATUS 2026-07-22: catch-up migration landed on main (PR #878, af42dc8 live — verified via the new `/health/detailed` scheduler job-table). BUT the restored-Supabase reconnection did NOT hold — the fresh af42dc8 boot STILL fell back to SQLite: `database_primary: (ENOTFOUND) tenant/user postgres.vexzwnfbmznvxoxxktax not found` (Supavisor "tenant not found" = the project is paused again / restore reverted). The Supabase MCP tools were NOT available this session to re-restore. **USER ACTION: supabase.com/dashboard → the `vexzwnfbmznvxoxxktax` project → Restore/Unpause.** The moment it's reachable, the next boot binds Postgres + applies `k6f7a8b9c0d1` cleanly (schema is now in sync). Until then bot activity is visible between deploys but resets on each merge.
- [ ] **[P2] Loss-cap window redesign** — `last_equity` spans the whole weekend for a 24/7 crypto book; measure vs a
  session anchor (portfolio-history API) instead. Needs validation — threshold semantics change.
  **MEASURED 2026-08-03 22:40, and deliberately NOT changed.** Today was exactly the scenario this describes: the Monday
  after a full weekend of 24/7 crypto drift, the same setup that froze the whole book on 2026-07-20. **The cap did not
  trip.** Evidence: run `30836526905` (17:23) placed `time_series_momentum/EWT signal=BUY conf=1.00` and it filled — and
  `desk_order_placer.py:2254` blocks any order where `_cap_active and not is_risk_reducing(...)`. A BUY with no offsetting
  short is exposure-increasing, so an active cap would have blocked it. Eleven in-window runs today placed 7-9 orders each;
  none was cap-blocked.
  This is a **safety control**, and the entry itself says the threshold semantics change. Rewriting the baseline of a risk
  limit speculatively — with no observed misfire, and while the 2026-07-21 mitigation (risk-reducing orders stay allowed
  under the cap) is already in place and working — would be changing a brake because it *might* be too sensitive. Park it
  until a run actually logs `🛑 DAILY LOSS CAP` spuriously; that log line is the trigger to revisit, and it is already
  surfaced in both the run output and the Discord funnel.


## 🌡️ REGIME-AWARE STRATEGY SELECTION (user directive 2026-07-20: "different market conditions → different strategies")
EXISTS today: `_detect_regime_from_bars` (SPY trend → bear/sideways/bull) +
`_STRATEGY_REGIME_MAP` gating in desk_order_placer. Gaps: trend-only (no volatility
axis — the dimension that decides options premium selling vs buying), unmapped
strategies default to ALL regimes, and the active regime is invisible.
- [~] **[P1] Add the volatility axis** — SHIPPED (partial) 2026-07-20: `_detect_vol_regime_from_bars` (calm/stressed from SPY realized-vol ratio, free); premium sellers (all _MLEG_STRUCTURES + vol_carry) face a +0.08 confidence bar in CALM vol (thin premium — not a hard block, income desk still trades); vol shown in run log + funnel summary (`regime=trend/vol`). 6 tests. Remaining: full 6-cell trend×vol map for long-premium/trend/mean-reversion (needs backtest validation before hard gating). — realized-vol percentile on SPY bars (free, no new feed) → calm/stressed; 3×2 = 6 regimes. Map options income (condors/credit spreads/0DTE-VRP) to STRESSED-elevated-IV regimes, long-premium + trend-followers to CALM-trending, mean-reversion to SIDEWAYS-calm. Audit every default-mapped strategy into an explicit cell.
- [~] **[P1] Make the regime visible + felt** — bot `regime` condition SHIPPED 2026-07-20 (detect_regime helper: trend×vol; ConditionConfig.regimes; 7 tests). Remaining: Discord regime-transition post + funnel regime-gated line. — post regime transitions to Discord (with the SPY/vol chart via discord_post_chart); per-run funnel post shows which strategies were regime-gated; bot engine gets a `regime` condition type so OA-style bots can say "only in bull-calm" (same pattern as ml_signal).
- [ ] **[P2] Regime-aware exploration** — the 5% exploration budget (below) rotates strategies within their FITTING regime, so accrued evidence is regime-conditional (a mean-reverter judged in a trend is bad science).

## 🎯 STRATEGY DIVERSITY (user directive 2026-07-20: "trades across strategies irrespective of market confusion")
Evidence: dry run funnel 49 signals → conf≥0.6 gate → 34 conflict stand-asides → top-K
→ 5 survivors; live fills concentrate in 1–2 strategies. Consequence: the pruning/
promotion loop (needs ≥20 live trades per strategy) is EVIDENCE-STARVED for ~95% of the
book — most strategies can never be judged. Fix = exploration, not looser risk:
- [x] **[P1] Exploration allocation in desk_order_placer** — SHIPPED 2026-07-20 (regime-gated, 0.45 noise floor, min-notional clips, daily rotation, pruned strategies excluded; funnel now reports `explored=`; coid tagging deferred to keep attribution parsing intact) — reserve ~5% of each desk's budget for MIN-NOTIONAL clips ($10–25) allocated to wirable strategies with the FEWEST live fills in 30d (rotate round-robin; skip pruned-to-zero strategies; conflict stand-asides still apply within a symbol). Every strategy accrues real stats → pruning/promotion becomes decisive instead of starved. Tag fills `qe-explore-*` so attribution can report exploration P&L separately from conviction P&L.
- [x] **[P2] Funnel telemetry post** — SHIPPED 2026-07-20: desk run summary now carries `regime=` + `funnel: N generated → M survived gate+topK (K exploration) → P placed`, so "why so few trades" is visible in the #pnl-daily post (no extra message — enriched the existing one). — after each desk run, post the funnel (generated → gated → conflicted → topK → placed, per desk) as a Discord chart so "why no trades" is visible at a glance, not a mystery.

## 🔬 SOTA RESEARCH SWEEP 2026-07-20 (web-grounded; apply-not-cite items)
Multi-agent trading (arXiv 2412.20138 TradingAgents; FinCon; HedgeAgents; ContestTrade 2508.00554; eval taxonomy 2603.27539):
- [ ] **[P1] TradingAgents-style role separation for the debate gate** — our queued bull/bear/judge debate should mirror the proven analyst→researcher→trader→risk-manager pipeline: analysts summarize (cacheable), researchers debate, trader proposes, risk manager holds VETO. Fits llm_common cascade; run pre-order for large notionals only (cost-aware — the 2603.27539 finding: coordination structure matters more than model size).
- [ ] **[P2] ContestTrade-style internal contest** — score each desk-strategy's LIVE hit-rate weekly and allocate the top-K budget by contest rank (we already have perf weighting/pruning; add the contest layer to the top-K selection itself).
- [x] **[P2] Look-ahead hygiene** — AUDITED + GUARDED 2026-07-20. Finding: the vectorized engine is look-ahead-FREE (`position=signal.shift(1)`; P&L = position × forward bar-return, so a position is only earned the bar AFTER its signal); options_synthetic is clean too (entry priced off entry-bar data available at entry, exit re-priced at the actual future spot = the realized outcome). Locked in by test_backtest_no_lookahead.py (4 tests: same-bar spike NOT captured; pre-spike signal IS; flat→0; earns only bar-after). SEPARATE conservative finding (NOT shipped — needs A/B validation): strategies `.shift(1)` features AND the engine shifts → a 2-bar execution lag that UNDERSTATES returns; "fixing" it raises reported returns and could promote overfit strategies, so it stays as a validated-change item.
Options income (0DTE VRP evidence 2016→2026 positive+significant; deep-learning options 2407.21791):
- [ ] **[P1] 0DTE variance-risk-premium desk mode** — evidence: implied > realized variance holds for 0DTE SPX through 2026. Our iron-condor/credit-spread templates already exist; add a 0DTE defined-risk variant sized off expected-move, with the documented practice of LETTING expire (avoids 2–5% spread-crossing close cost) under the Expiration Protocol item (do that first).
- [ ] **[P2] End-to-end learned options signals** — 2407.21791 shows learning position sizing directly from option surfaces beats hand-crafted rules; feasible later via the CI-trained-GBM pipeline on Tradier chain snapshots (start LOGGING daily chain snapshots now — free, and the dataset compounds).

## 📣 WEAK DESKS + DEAD MESSAGES (user evidence 2026-07-20: #desk-fx-rates screenshot)
Screenshot shows the FX desk posting the IDENTICAL line ("10 signals ≥ 0.6, 3 orders
EUR_USD, GBP_USD, USD_JPY") ~8×/day: no direction, no prices, no fills, no P&L — and
nothing consumes these posts (peer_learnings only captures agent discussions, not desk
output). Write-only noise. Same weakness class hits Polymarket + arbitrage desks.
- [x] **[P1] Desk posts: dedupe + enrich + chart** — SHIPPED 2026-07-20: dedupe+enrich for FX desk (notify.post_dedup, stateless via bot API) AND an orders-by-desk bar chart on the desk_order_placer #pnl-daily summary (discord_post_chart, only on non-empty runs). 11 tests. Remaining P&L-bars-per-strategy chart folded into the daily P&L report (already charts net notional). (notify.post_dedup: stateless, reads channel history via bot API — no git state; FX post now shows side+entry px per order and suppresses identical repeats; 6 tests). Remaining: apply to desk_order_placer P&L posts + attach discord_post_chart P&L bars. — suppress consecutive identical posts (hash last message per channel in state); every desk post must carry direction, entry px, open-position count, running desk P&L; attach `discord_post_chart` P&L bars (helper shipped). Applies to fx_desk.py AND desk_order_placer P&L posts.
- [x] **[P1] Desk posts → shared brain** — SHIPPED 2026-07-22: the missing CONSUME side is wired. `notify.read_channel_recent` reads desk run summaries back from Discord (stateless, via the bot API — no git-state churn, same pattern as `post_dedup`); `company_brain.fetch_desk_knowledge()` pulls the newest substantive post per desk channel (pnl-daily, desk-fx-rates/crypto/equities/options/commodities/polymarket/kalshi, bot-fleet, trading-floor) into the brain as a `desk_outcomes` category AND feeds them to the CIO synthesis; `llm_common.get_company_context()` surfaces `Desk results:` into EVERY employee prompt. Desk outcomes are written directly (not only via LLM) so the brain stays grounded even when the free-LLM keys are unset. 13 tests (test_notify_channel_read.py + test_desk_brain_bridge.py). — pipe each desk's run summary into peer_learnings/company_brain so employees DISCUSS actual desk results in the morning loop (the missing consume-side; extends the queued outcome-linked-learning item).
- [~] **[P1] FX desk audit: same-3-orders monotony** — NO-REPEAT GUARD SHIPPED 2026-07-20: fx_desk fetches OANDA openPositions and skips signals already positioned in the SAME direction (opposite direction stays allowed — reduces/flips exposure; fetch failure fails OPEN so monitoring never blocks the desk); skips surface in the run log + Discord post ("N skipped (already positioned)"). 5 tests. Remaining: signal-variance audit (why the same 3 pairs always rank top — needs several live runs of funnel data to diagnose). — always EUR_USD/GBP_USD/USD_JPY means top-K by confidence is static → likely stale/constant signal inputs or too-narrow universe ranking. Audit signal variance; add no-repeat-position guard (skip if an equivalent open position exists) and log WHY each pair won.
- [~] **[P1] Polymarket desk is signal-only — EVIDENCE ADDED + honest reporting shipped 2026-08-04 05:50.**
  Concrete confirmation of the gap: `desk_order_placer.py` references `brokers` **zero times** — every order is a
  POST to Alpaca `/v2/orders` (line ~1101) — while `backend/app/brokers/polymarket.py` exists and is never imported
  there. So the desk cannot reach a venue at any hour.
  **What was actively harmful:** those `conf=1.00` signals were dropped as `(Polymarket closed)`, wrong twice over —
  prediction markets never close, and even open there is nowhere to send the order. On 2026-08-03 that message
  nearly produced the wrong fix: setting `always_open=True` would have shipped
  `PM:Will Tucker Carlson win the 2028 Republi…` to Alpaca. Shipped `DeskConfig.executable` (default True), set
  False for Polymarket, checked BEFORE the market clock so the clock's message cannot win, with its own
  `no order path` drop reason in the Discord funnel. 6 tests, mutation-checked. **No trading behaviour change** —
  the same signals are dropped, they are just described truthfully.
  **Still open (the actual fix):** wire py-clob-client signing for real CLOB orders, or retire the desk.
  Original entry: — — signals flow (dry run: live markets, conf=1.00 ensembles) but NO order path: py-clob-client signing still unwired (POLYMARKET_PRIVATE_KEY is in the relay). Implement CLOB order placement with $1–5 clips + the same never-partial guard, or the desk stays a commentary bot.
- [~] **[P2] Arbitrage-bucket audit** — ~~32 strategies in the arb bucket but near-zero desk fills attributed to them; verify their signals reach a desk with an order path~~ **HALF ANSWERED 2026-07-29.** Of the 32 registry strategies with `risk_bucket == "arbitrage"`, **28 are desk-wired and 4 are not**: `covered_call` (needs share inventory — already excluded inline in the Options roster), `crypto_basis_roll` and `funding_rate_arb` (both need perpetual futures; Alpaca paper has none), `dex_cex_arb` (needs Uniswap/DEX connectivity). None is a defect, but all four are counted in the bucket while being structurally unable to produce an order, inflating its apparent capacity by ~12%. Now a maintained invariant: `backend/tests/unit/test_arb_bucket_reachability.py` fails if an arb strategy is neither desk-wired nor listed as dormant WITH a reason. **STILL OPEN:** whether the 28 wired strategies actually fill or are filtered at the confidence gate — that needs per-strategy attribution, and `strategy_performance.json` only starts being produced from 2026-07-29 06:11 UTC (see CONTINUITY 03:40). Revisit once it has a few cycles of data.

## 🤖 BOT FLEET = FULLY AUTOMATED MANAGEMENT (user directive 2026-07-20)
Owner of record for the 61 bots is the AUTOMATION, not the user. Already live:
`bot_lifecycle` scheduler job (disable proven losers, promote winners, grow fleet from
templates — deterministic policy over real closed trades), 5-min exit sweep (fixed
today), safeguards, additive seeding, reward-gated code changes. Gaps to close:
- [x] **[P1] Lifecycle decisions → Discord** — SHIPPED 2026-07-20: bot_lifecycle posts each enable/disable to Discord (slack.send fails over to Discord) WITH the stats that drove it — `name (N trades, win X%, P&L $Y)` — to #bot-fleet, no black-box decisions. Remaining: append to peer_learnings for employee discussion (folded into the desk-posts→shared-brain item). — every enable/disable/promote decision posts WHY (stats in hand) to #bot-fleet; decisions also append to peer_learnings so employees can veto/discuss.
- [ ] **[P2] Bot parameter tuner** — weekly: for bots with ≥30 closed trades, grid-walk TP/SL% against their own trade history (pure pandas, free) and open a reward-gated PR adjusting template params; never touches live config directly.
- [ ] **[P2] Weekly OA-comparison report** — auto-generate a positions/P&L table per bot (entry/exit/hold/P&L) as a Discord chart + markdown artifact so the user's "match vs Option Alpha" check is a 2-minute read, not manual data pulling.

## 🤖 OA-BOT THOROUGH TEST 2026-07-20 PM — found + fixed a live P0
- [x] **[P0 FOUND+FIXED] Bot positions NEVER closed on the live (SQLite-fallback) deploy** — the new `test_bot_lifecycle.py` (first-ever coverage of `check_bot_exits`, the OA profit-taking half) reproduced it: SQLite returns NAIVE datetimes even for `DateTime(timezone=True)` columns, so `now − order.created_at` raised `TypeError`, and the scheduler's catch-all silently killed the ENTIRE exit sweep every 5 min ("Bot exit checker failed"). Positions opened with TP/SL brackets and then sat there forever. FIX: normalize `created_at` to aware-UTC at both subtraction sites in `engine.py`. Lifecycle now pinned by 6 tests: bracket math on open (±TP%/−SL% both sides), profit-target close AT target with +P&L, stop-loss close AT stop with −P&L, short-side TP, inside-bracket stays open, 7-day safety expiry.
- [x] **[P1, was P2] Test-isolation flake pattern — worse than "two known flakes", and now MEASURED.** Making `test_strategy_contract.py` 100× faster (7m15s → 4.5s) changed how `--dist loadfile` packs files onto workers, and CI went red on `test_seed_additive::test_seed_is_additive_and_idempotent` with `assert 0 == 61`. That test asserts `n_first == len(BOT_TEMPLATES)`, which **silently requires it to be the first thing that seeds on its worker** — anything that boots the app (lifespan → `seed_all`) leaves the templates already seeded. Previously the contract file occupied one worker for seven minutes doing nothing else, which incidentally spread the remaining files thinner; now all four workers churn through many more files each, so cross-file contamination is far likelier. **The speedup did not create this bug; it removed the accident that was hiding it.**  **✅ DONE — verified 2026-07-29 02:40.** `backend/tests/conftest.py::_isolate_each_file` (per-file DB isolation) landed 2026-07-27. Re-ran the full suite under the exact failing configuration, `pytest tests/ -n 4 --dist loadfile`: **1997 passed, 31 skipped, 5 xfailed, 0 failed.** `test_seed_additive::test_seed_is_additive_and_idempotent` passes.
  **A failed first attempt worth recording.** I rewrote the test to seed unconditionally and then assert completeness. Full-suite runs under CI's exact flags: **21 failed, then 30 failed** — versus **1889 passed** three times in a row with the original test. Unconditionally seeding poisons the shared per-worker DB for every other file on that worker (`test_cross_user_isolation`, `test_analytics_honest`, `test_system_whole_app` all key off row counts). Reverted; the contract fix ships alone. *A fix for an isolation flake that itself writes to the shared fixture is not a fix.*
  **RESOLVED.** `conftest._isolate_each_file` — a module-scoped autouse fixture — now wipes all rows between test FILES. Rows, not tables: the schema is still built once per session, so the cost is a handful of DELETEs per file and the suite runtime is unchanged (72s before and after).
  Deliberately **module** scope, not function: within one file, tests building on each other is normal and intended (a fixture creating a user, then tests acting as it). The coupling that had to go is the *accidental* kind, between files that never meant to share anything.
  Proven both directions on the exact CI failure — running the seeding file before `test_seed_additive` in one session: **with** the fixture it passes, **without** it fails `assert 0 == 61`. That let the tolerant `n_first in (0, len(BOT_TEMPLATES))` workaround be reverted to the original strict `== len(BOT_TEMPLATES)`, which is the stronger assertion: it verifies seeding actually does the work rather than tolerating someone else having done it.
  Validated with three consecutive CI-invocation runs: 1897 / 1896 / 1896 passed, 0 failed.
- [x] **[P2] Original note: test-isolation flake pattern** — two known cross-file flakes under `-n 4 --dist loadfile` (breakeven_inflation contract, tearsheet-on-sqlite): shared per-worker DB lets one file's rows leak into another's expectations. Fix: per-file DB fixtures or scoped assertions (lifecycle tests already scope per-bot).  **✅ DONE — superseded by the per-file DB isolation fixture** (`_isolate_each_file`, 2026-07-27) and verified 2026-07-29 under `-n 4 --dist loadfile`: 1997 passed, 0 failed.

## 🔍 DEEP REVIEW 2026-07-20 PM — findings + tech/scalability roadmap
> Full-repo sweep: security, schema, workflows, code health, infra. Secrets scan CLEAN
> (no real keys committed — only doc placeholders). Suite: 1,699 passed / 0 failed.

- [x] **[P0] Schema-drift landmine when Supabase unpauses** — GATE SHIPPED: schema-drift-gate.yml (ephemeral Postgres → alembic head → autogenerate diff, fails with the missing ops). Run it via workflow_dispatch BEFORE unpausing; write the catch-up revision it prints — the live schema has evolved via `create_all` on the SQLite fallback, but `create_all` NEVER adds columns to existing tables and only 6 alembic migrations exist for a fast-moving model layer (e.g. `strategy_name` referenced by 6 model files, covered by 1 migration). The moment the user unpauses Supabase, the backend binds to an OLD Postgres schema → column-not-found 500s on the endpoints we just fixed. FIX: CI job that spins ephemeral Postgres, runs `alembic upgrade head`, diffs against `Base.metadata` (alembic autogenerate), FAILS on drift and opens a catch-up-revision PR. Must land before the unpause.
- [ ] **[P1] agb8 double-execution hazard — RE-VERIFIED STILL LIVE 2026-08-04 02:45.** `GET agb8/health/detailed` returns
  HTTP 200: `mode: paper`, `alpaca: {ok: true, "connected"}`, `background_tasks: {running: 11, total: 11}`,
  `strategies: {count: 113}`, `scheduler: {ok: true}`, `database: {ok: false, "[Errno -2] Name or service not known"}`.
  An old build with a dead DB is running 11 background tasks against the SAME paper account as `9jz0`. This also
  contaminates three live-Alpaca reads the desk depends on — `_kelly_notional` (equity), `daily_loss_cap_hit`
  (equity vs last_equity) and `is_risk_reducing` (position map) — plus the 2026-08-03 slippage attribution.
  Proven: service up, paper, Alpaca-connected, tasks live. NOT proven: that it placed an order today.
  Original entry: — TWO backends run 24×7 against the SAME Alpaca paper account: the stale `quantedge-api-agb8` (old build, own working DB, own scheduler → places bot orders + runs desk sync) plus the keeper `9jz0`. Duplicate order placement and split-brain state. User action (30s): suspend/delete the agb8 service in the Render dashboard.
- [x] **[P1] Slack long-tail removal — DONE 2026-07-25.** Superseded by the full removal at the top of this file: 0 slack.com callers, 0 SLACK_BOT_TOKEN reads, 0 slack-*.yml workflows, 0 slack modules, repo-wide, with structural guards. (This entry still claimed "65 of 105 workflows wire SLACK_BOT_TOKEN", which would have sent the improver chasing work that no longer exists.)
- [ ] **[P2] Workflow consolidation** — **evidenced 2026-07-29: the fleet is starving its own trading desk.** The 24/7 crypto desk ran 11 times against a nominal 72 in 24h (see the P0-GATE item), so "fewer schedules = less cron starvation" is a measurement now, not a hypothesis. 105 workflows with overlapping families (agent-health-check / agent-health-monitor / agent-heartbeat / agent-status-check; gemini-task-runner vs generic runners). Merge each family into one parameterized workflow; fewer schedules = less cron starvation.
- [~] **[P2] Money-path exception audit** — first pass DONE 2026-07-25. Audited all 40 `except` handlers in `execution/`, `risk/`, `brokers/`. Result is mostly reassuring: the 3 handlers with no logging at all are `except ImportError` optional-dependency guards (correct), and the money paths generally log. **One real defect found and fixed:** `CompositeExit.should_exit` (position_exit.py) caught a failing exit rule, logged a *warning*, and returned `(False, None)` — indistinguishable from "checked, nothing to do". If the stop-loss rule raised, the position kept running with **no stop and no alarm**. Now tracks how many rules actually evaluated and escalates to `logger.error` ("position is UNPROTECTED", with symbol + failed rules) when NONE did, while keeping the fail-soft behaviour that one broken rule must not disable the others. Trading behaviour unchanged — it still returns False rather than forcing a spurious exit. 6 tests. **Second finding fixed 2026-07-25:** `AlpacaBroker.place_order` caught a failed BRACKET order, logged a *warning*, and fell through to a plain market order — so the take-profit and stop-loss legs the caller asked for were dropped while the entry still filled, and the returned `OrderResult` looked like an ordinary success. Downstream had no way to know the position was unprotected. Now logs at error level naming the consequence and tags the result `bracket_degraded: True` with the `unapplied_stop_loss`/`unapplied_take_profit`, so the exit sweep and risk layer can see it. Fill behaviour deliberately unchanged. 3 tests — which stub the alpaca-py symbols rather than importing them, because **alpaca-py is not in the CI dep list**, so an `ALPACA_AVAILABLE`-gated test would skip in CI and never run. **OPEN QUESTION FOR THE USER:** should a rejected bracket fill naked at all, or should it abort the entry? Filling unprotected changes the risk profile the strategy sized for; aborting loses the trade. Left as-is (fill + alarm) because changing it is a trading-policy decision, not a bug fix. **Third pass 2026-07-27 — `execution/advanced_orders.py`, and this one was worse than the first two.** Reviewed the remaining handlers; `risk/manager.py` came out clean (every handler re-raises a typed error — fail-closed, correct). The bracket/OCO layer did not:
> - **[P0] The entry-price confirmation filter has NEVER rejected an order.** On a tolerance breach it built `OrderResult(broker_order_id=..., reason="price_tolerance_exceeded")` — but `reason` is not a field on that `slots=True` dataclass. So every rejection raised `TypeError` from *inside* the surrounding `try`, landed in `except Exception: logger.warning("Failed to fetch market price for entry confirmation")`, and execution fell straight through to submitting the entry. A guard that reports a quote-fetch problem while doing the exact thing it exists to prevent. The decision now sits OUTSIDE the try — only the quote *fetch* may fail soft (no quote = nothing to check against) — and the rejection carries its detail in `raw_payload`. **Behaviour change:** limit entries deviating more than `price_tolerance` from market are now actually rejected. That is the originally-intended behaviour, but it has never once run in production, so it is worth watching.
> - **[P1] `config.price_tolerance` was silently ignored.** `_price_within_tolerance` read `entry.price_tolerance` behind a `hasattr` guard; `OrderRequest` has no such field, so it always fell through to a hardcoded `0.02` and any configured value was dead. Tolerance is now passed in explicitly.
> - **[P1] Three more ways a bracket returned an ordinary-looking success for an unprotected position** — same defect class as the `CompositeExit` and Alpaca findings above: (a) `tp_price <= sl_price` logged an error and returned the bare filled entry; (b) a raising OCO submission propagated *after* the entry had filled, so the caller saw a failure for a bracket that did open a position — and a retry would have doubled the entry; (c) an 8h OCO timeout cancelled both legs and handed back what looked like a live protective order. All three now log at error level naming the consequence and tag the result `bracket_unprotected: True` with `unprotected_reason` + the intended TP/SL, so the exit sweep and risk layer can see it.
> - **[P1] A failed OCO leg B left leg A resting, live, and unmanaged** — that is not a one-cancels-other, it is a lone order nobody will ever cancel. Leg A is now cancelled before the error propagates; if that cleanup cancel also fails, the live order id is named at error level.
> 12 tests. Fill/abort semantics deliberately unchanged everywhere except the confirmation filter noted above.
>
**Fourth pass 2026-07-27 — the sliced execution algorithms, and this one reached all the way into Redis.** TWAP, VWAP, iceberg and Almgren-Chriss all ended with the identical construction `broker_order_id=last_result.broker_order_id if last_result else "vwap"` / `status="filled" if total_filled >= qty*0.95 else "partial"`.
> - **[P0] A run where every slice failed returned `status="partial"`, `filled_qty=0`, and a fabricated `broker_order_id` ("vwap"/"twap"/"iceberg"/"ac_exec")** — an id no broker ever issued, attached to an execution that filled nothing, reported as a partial fill. Anything that tried to poll or cancel that id would fail. Centralised into `execution/slice_result.py::build_slice_result`; a zero-fill is now `status="rejected"` with an empty order id, an error log, and `slices_attempted`/`slices_failed`/`last_error` in `raw_payload`.
> - **[P0] …and `strategy_runner` turned that into a phantom position.** It gated on `if result:` — `OrderResult` is a plain dataclass, so that is *always* true; only `None` (a risk block) was ever filtered. A failed execution therefore logged "Order submitted" and wrote a `pos_exit:<symbol>` config into Redis, leaving `position_monitor` tracking a stop-loss and take-profit on shares nobody owned. Now gated on `result.status not in _DEAD_ORDER_STATUSES` with an error log naming the skip. (The same log line called `getattr(result, "order_id", "?")` — not a field either, so it printed "?" every time; fixed to `broker_order_id`.)
> - **[P1] The RL branch of `SmartOrderRouter` raised `TypeError` on every successful execution** — it aggregated fills into `OrderResult(order_id=..., symbol=...)` and neither is a field on that dataclass. The fills had already happened at the broker by then, so the caller saw a crash for an order that did execute. Latent in prod today only because torch is absent so `_RL_EXEC_AVAILABLE` is False. Same defect signature as the bracket `reason=` bug above — worth a lint rule. **[SHIPPED 2026-07-27 — see below.]**

- [x] **[P0] The lint rule — and it immediately found a third instance that had disabled a whole strategy.** After the same defect shipped twice (`OrderResult(reason=…)`, `OrderResult(order_id=…, symbol=…)`), added `backend/tests/unit/test_dataclass_kwargs.py`: a static AST check that parses every module under `app/`, resolves each dataclass's declared fields (following dataclass base classes), and asserts no construction passes a keyword that is not a field. Python only raises at call time, so a bad construction on a rare error path — which is exactly where these live — stays invisible until production hits it.
  On its first run it found **`BacktestSignals(positions=, returns=, probabilities=, execution_time_ms=)` in `strategies/ml_enhanced/lorentzian_knn.py`** — all four keywords wrong, and both *required* fields (`entries`, `exits`) missing. `LorentzianStrategy.backtest_signals()` therefore raised `TypeError` on every call: **the strategy could never be backtested at all**, which under the repo's own "walk-forward only / 2 weeks paper before live" rule means it could never have been validated. The tell was a comment reading "the concrete fields depend on the dataclass definition" — an agent writing against a definition it never opened. Fixed to return the four boolean series the function already computes (the discarded P&L approximation was dead weight; VectorBT computes P&L from the signals). Also caught in the same edit: the replacement `logger.debug(...)` I first wrote used structlog-style kwargs, but this module's `logger` is stdlib — that would have raised too.
  The check is deliberately conservative: it skips `**kwargs` unpacking, classes with a hand-written `__init__`, and unresolvable bases, so a pass is not proof of correctness — but a failure is always real. 4 tests, one of which plants a violation to prove the check can actually fail.

- [x] **[P1] yfinance price feed: non-deterministic loop cleanup + a wholly dead feed logged at DEBUG** — SHIPPED 2026-07-27. `_yf_publish_sync` runs in an executor thread, once per symbol, every 60s, forever, and hand-managed its own event loop: `new_event_loop()` / `run_until_complete(...)` / `loop.close()`, with the close as an ordinary statement rather than a `finally`. Any raise skipped it. **Correction to my first read of this:** it is NOT an unbounded fd leak — CPython's refcounting reclaims the loop via `BaseEventLoop.__del__`. Measured empirically: 57 descriptors outstanding across 300 failed iterations, back to baseline after `gc.collect()`. What it *did* do was leave cleanup to the GC and emit `ResourceWarning: unclosed event loop` plus two unclosed sockets on every failure. Replaced with `asyncio.run`, which closes deterministically on the error path and additionally cancels pending tasks and shuts down async generators — neither of which the old form ever did. Second and more consequential half: `_yfinance_price_feed` swallowed every per-symbol failure at `logger.debug`, so a feed publishing **nothing at all** was indistinguishable from a healthy one while every strategy downstream read stale Redis prices. Per-symbol drops stay at debug (yfinance drops ticks routinely); a cycle that publishes zero now logs at error with a consecutive-dead-cycle count, and recovery is logged too. 6 tests — the loop-cleanup one was verified to actually FAIL against the old code (an earlier fd-counting version passed against both and was thrown away).

- [x] **[P0] The mirror-image check found a second dead ML model.** The guard above catches *unknown* keywords; it said nothing about *missing required* ones — yet the Lorentzian bug had both (it also omitted the required `entries`/`exits`). Extended `test_dataclass_kwargs.py` with `_resolve_required` (fields annotated with no default, base-class fields first, ClassVar/InitVar excluded) and a second check. First run: **`EvalMetrics(loss=, accuracy=, auc=)` in `ml/models/itransformer.py` omits the required `sharpe`** — so `iTransformerPredictor.evaluate()` raised `TypeError` on every call and that model could never be evaluated. Every other model in the package passes `sharpe=0.0`; iTransformer alone omitted it. Fixed to match. That is now **two** ML components (Lorentzian KNN, iTransformer) that were structurally incapable of running the one function used to validate them, both found by static checks rather than tests — because neither can execute in CI at all: torch is not in the CI dependency list.
  The required-field check is stricter than the unknown-keyword one: it skips calls with positional args (resolving those means honouring `kw_only` and inherited field ordering) and skips any class name defined more than once, since unioning "required" across two classes would invent requirements neither has. Verified to fail against the unfixed call site. 6 tests total.

- [x] **[P2] Guard: stdlib logger called with structlog-style keyword fields** — `logger.info("msg", symbol=x)` raises `TypeError` on a stdlib logger but is correct on a structlog one, and this repo mixes both conventions: 32 modules bind `logging.getLogger(...)`, 79 bind the structlog `logger`. I introduced exactly this bug while fixing `lorentzian_knn.py` and caught it only by eye. A scan of `app/` currently finds **zero** violations, so this is purely preventive — but the improver agents edit these files unattended, which is precisely the case `TestSlackStaysRemoved` exists for. Same AST-scan shape as `test_dataclass_kwargs.py`: resolve how `logger` is bound per module, then flag keyword args other than `exc_info`/`stack_info`/`stacklevel`/`extra` on stdlib-bound loggers.  **✅ DONE — it already existed and nobody ticked the box.** `backend/tests/unit/test_logger_kwargs.py` (2026-07-27) implements exactly the described design: resolve the per-module binding, flag any keyword outside `exc_info`/`stack_info`/`stacklevel`/`extra` on a stdlib-bound logger, skip modules where the binding is ambiguous. 6 tests, all passing, and it asserts it actually resolved some stdlib loggers so a vacuous pass cannot hide a broken scanner. **I nearly rewrote it from scratch this tick** — see CONTINUITY 2026-07-29 02:40.
  Rejected while investigating this: a scan for discarded coroutine calls (`foo()` as a bare statement where `foo` is `async def`). Too noisy to ship — generic method names (`create_task`, `execute`, `write`, `run`, `set`, `close`) collide across classes, producing 21 hits that were **all** false positives on inspection. Would need real type inference to be useful.
> - **[P1] VWAP had no consecutive-failure abort** (TWAP and Almgren-Chriss both do) — it worked the entire schedule against a dead broker. Now aborts after 3, matching the others.
> - **[P1] Iceberg could spin forever**: an accepted-but-unfilled slice returns `filled_qty=0`, which leaves `remaining` untouched, so `while remaining > 0.01` never terminates. Now breaks with a warning.
> 12 tests.
>
> Remaining: `_select_algorithm` is likely unreachable past its first branch — `OrderRequest.execution_algo` defaults to `"limit_first"`, not `"auto"`, so the explicit-override branch fires for every caller that doesn't set it (including `strategy_runner`), and the size-based routing to twap/almgren_chriss/rl_exec never runs. Deliberately NOT fixed here: changing which algo executes every order is a trading-behaviour decision, not a bug fix. **Worth a user call.** Also: none of these error paths page Discord yet — worth a judgement call per case. Note structlog does NOT propagate to stdlib logging, so assertions about these logs must use capsys, not caplog (caplog passes for the wrong reason).
- [~] **[P3] Test hygiene batch** — pytest config-source dedup SHIPPED 2026-07-24: removed the redundant/ignored `[tool.pytest.ini_options]` block from `backend/pyproject.toml`; `pytest.ini` is now the single source of truth (`configfile: pytest.ini`, "ignoring pytest config in pyproject.toml" warning gone), with a guard comment in both files so a future improver doesn't re-add the duplicate. Zero behavior change (pytest.ini already won). Pydantic-V2 model-example + Starlette-422 fixes SHIPPED 2026-07-24: the 4 `class Config: schema_extra` blocks (scanners.py ×2, macro_signals.py ×2) were being SILENTLY DROPPED by Pydantic V2 — the model-level OpenAPI examples never rendered — so converted them to `model_config = ConfigDict(json_schema_extra=...)` (examples restored + both the "schema_extra renamed" and "class-based config" warnings gone; verified example present in `model_json_schema()`); renamed the deprecated `status.HTTP_422_UNPROCESSABLE_ENTITY` → `HTTP_422_UNPROCESSABLE_CONTENT` in webhooks.py (same value 422, starlette 1.3.1). 32 targeted tests green. Pydantic-V2 field/validator migration SHIPPED 2026-07-25 — the batch is now essentially done: all 34 per-field `example=` Field kwargs → `json_schema_extra={"example": …}` (field examples verified present in `model_json_schema()`); all 8 V1 `@validator` → `@field_validator` + `@classmethod`; `min_items` → `min_length`; and `each_item=True` (REMOVED in V2, so it would have silently stopped validating per-item on the V3 upgrade) rewritten as an explicit loop. Behavior pinned by direct assertions before/after: side lower-cased + restricted, empty-signals rejected, ticker upper-cased, macro_bias/vix_regime restricted, empty feature ITEM rejected, empty feature LIST rejected, bad symbol rejected. Zero pydantic deprecation warnings left in these modules; full backend suite green. Remaining (both need a dependency change, not a code fix — deliberately out of scope): Starlette TestClient deprecation (wants `httpx2` installed); wavelet_features fragmentation warnings.

### Tech roadmap (scalability assessment 2026-07-20 — architecture is sound, free-tier infra is the risk)
- [ ] **[P1] Database durability** — NOW: unpause Supabase + keep-alive ping (queued below). NEXT: migrate to Neon serverless Postgres (auto-wakes on connection — eliminates the pause failure class at $0) or Supabase Pro. The SQLite fallback stays as the last-resort guard.
- [ ] **[P0-GATE for live trading] Always-on execution worker** — GitHub Actions cadence (cron starvation, ~15-min floor, suppressed events) is acceptable for PAPER only. **Now MEASURED, 2026-07-29 — and the worst case is the TRADING LOOP itself.** `desk-trading-crypto-24x7.yml` has cron `7,27,47 * * * *` = **72 runs/day nominal**; it managed **11 in 24h (15%)**, with gaps of 173/208/88/58/66/73/89/103/88/140 minutes against an intended 20. All 11 succeeded and none were cancelled (`cancel-in-progress: false`) — GitHub never created the other 61. Anything driven off a desk run, exits included, carries up to **3h28m** of latency. Also `fill-tracking` with cron `0 22 * * 1-5` actually started at **23:02:41 — 62 minutes late**; its replacement `11 */6` slot at 06:11 had not started by 06:41 (≥30 min late, possibly dropped — GitHub silently drops scheduled runs under load, and a dropped run never appears in the run list). The rest of the fleet fired normally in that window, so this is ordinary cron jitter, not an outage. An hour of jitter is a nuisance for paper and a loss for live. Before `TRADING_MODE=live` ever flips: move desk execution into an always-on worker (Fly.io/Railway/Render starter ~$7/mo) driving the existing APScheduler loop. Missed exits on live capital is not an acceptable failure mode.
- [ ] **[P2] Agent state out of git** — `.github/state/*.json` committed hundreds of times/day caused the improver-clobbering incident and pollutes history. Move agent memory/company brain to Postgres tables + the queued BM25 retrieval; keep git snapshots as daily backup only.
- [ ] **[P2] Single-origin serving** — drop Vercel: `static_server` already serves the built frontend from the backend. One origin kills the CORS, rewrite-target, VITE_API_URL-mismatch AND deploy-rate-limit failure classes (all four bit us this week).
- **Keep as-is (scales fine):** FastAPI/SQLAlchemy async, React+Vite, broker plugin layer, reward-gated PR loop, LLM cascade with paid backstop. No rewrites, no Kubernetes.

## OA UI/UX parity roadmap (from 10 live OA screenshots, 2026-07-19)
> Full setting inventory: `.github/state/oa_settings_catalog.json`. Study doc: `docs/research/OA_DOCS_STUDY.md`.
> Goal (user): make the QuantEdge dashboard look/work like Option Alpha — "lots of vizz,
> dashboard + graph in bot rows" — and use the Screener/Trade Ideas to find symbols and scale bots.
- [x] **Bots list = OA layout** — SHIPPED this session: `30D` sparkline column (real /performance
      series, dashed zero baseline), Total P/L, Return %, Win Rate, Allocation columns, AUTOS toggle,
      + aggregate top cards (Total P/L / Return % / Change / Change % / Allocation). `BotBuilder.tsx`.
- [x] **Per-bot dashboard = OA layout** — SHIPPED: filled Closed-P/L equity curve + Position Stats grid
      (Closed Positions, Closed P/L, Profit Factor, Max Drawdown, Win Rate, Wins, Losses, Avg P/L,
      Avg Win, Avg Loss, Streak, Sharpe) + Capital sidebar (Allocation/Net Liquid/At Risk/Available/
      Maintenance). Backend `/bots/{id}/performance` now returns all OA metrics + weekday/hour/symbol breakdowns.
- [x] **[P1] Analyze page** (screenshot 3) — SHIPPED (first pass): the expanded bot panel now has a
      **Dashboard / Analyze** tab toggle; Analyze renders Metrics (Positions/Wins/Losses/Sharpe/Sortino/
      Profit Factor) + signed bar charts for **P/L by Day-of-Week**, **by Hour-of-Day**, and horizontal
      **P/L by Symbol** — all from the real `/performance` breakdown. `BotAnalyzePanel` in BotBuilder.tsx.
      Follow-ups: donut cards, big area chart with Total/Daily/Calendar tabs, Averages (Return-on-Risk/
      Entry-POP/DTE/Days-in-Trade), Hindsight Report, Export Data.
- [ ] **[P1] Positions tables** (screenshots 5,6) — Open Positions (Bot icon, Description, Legs w/
      call/put chips, Last, DTE, Qty, P/L, ROR, Net Liq, Premium, Risk, DIT + aggregate cards) and
      Closed Positions (adds Exp, Status Expired/Closed, Trade Price, Close Price, Price at OI/CI).
      Needs per-leg option instrument fields on Order/Position (Desk-consolidation Stage 3).
- [ ] **[P2] Trade Log** (screenshot 4) — chronological grouped-by-day feed: Bot icon, Time, Action
      (open/close), Description (full legs), Pricing (entry→exit), Status (filled at $x / canceled).
- [x] **Backtest→bot generator** (screenshot 8, OA "Automate your strategy") — SHIPPED:
      `POST /bots/from-backtest/{run_id}` maps the backtest's strategy family to real engine
      conditions (mapped vs approx confidence), bakes provenance (source run + Sharpe/return/win)
      into the description, creates the bot **disabled** (paper-first). Frontend: "→ Create Bot"
      button on each done run in BacktestLab. `app/bots/backtest_to_bot.py` + 3 integration tests.
      Follow-ups: the full "Create Bot" side-panel (name/allocation/entry-time/trade-pricing inputs)
      and Update-existing mode.
- [ ] **[P1] Screener page → symbol-scaling feed** (screenshot 9) — Symbol, 1M sparkline, Last, Today%,
      Liquidity bars, Beta, IV Rank, RSI, Techs, 3M/6M/12M returns, Earnings-in-Nd. Wire the ranked
      output into Symbol Scout so bots auto-scale onto the best-ranked new symbols (user: "use this to
      find symbols and scaling up").
- [ ] **[P1] Trade Ideas / EV finder** (screenshot 10) — math/probability opportunity ranker: Position,
      Leg/Delta, DTE, Reward/Risk, POP, PMP, PML, Alpha, EV, Max Profit, Max Loss, Earnings, Traders;
      Credit/Debit + Bull/Bear/Neutral + DTE/RR/POP/OTM% filters. Compute from `options_synthetic` +
      probability (shares the EV/probability P2 item below).
- [ ] **[P1] Calendar + earnings strategies** (screenshot, user: "make calendars like this, add earnings
      strategies") — month grid of Ex-Dividend, Earnings AM/PM, Market Holiday, Nonfarm Payrolls / FOMC,
      End of Quarter/Month events (FF calendar feed + an earnings-date source). THEN earnings strategies:
      pre-earnings IV-crush short premium (iron condor/strangle sized off Earnings Edge expected move),
      post-earnings drift — gated by the calendar so they only fire around confirmed earnings dates.

## OA parity gaps (from docs.optionalpha.com audit, 2026-07-19 — see docs/research/OA_FEATURE_PARITY_2026.md)
- [ ] **[P1] Options Expiration Protocol** — auto-manage 0DTE/expiring ITM positions before the bell (close or flag); critical now that the desk places real mleg spreads and 3 user bots trade 0DTE into the close.
- [x] **[P1] Bot safeguard depth** — SHIPPED: `ActionConfig.max_open_positions` / `max_daily_positions` (OA "N at once / N per day"); the engine counts this bot's open paper positions + today's opens and refuses to open once a limit is hit ("Position limit reached"). Also wired `no_position`/`position_exists` conditions to real open-position state (they were stubs returning True). `app/bots/engine.py` + `test_bot_safeguards.py` (5 tests).
- [ ] **[P1] Decision-recipe catalog expansion** — condition types evidenced by the user's bots: FOMC-day gate (FF calendar feed exists), price-change-vs-N-min-ago, option-leg OI threshold, position-return%, touch-$ exit; then loops over watchlists and position tags.
- [ ] **[P2] Failsafe family** — per-bot excessive-errors auto-disable (10/day → off + alert), overlapping-strikes check, pricing-anomaly gate.
- [ ] **[P2] EV/probability metrics** — HV-based probability + EV per defined-risk trade (OA's Trade Ideas 2.0 'Alpha'), computable from options_synthetic.
- [x] **[P2] Backtest→bot generator** — SHIPPED (see the OA-UI roadmap above): `POST /bots/from-backtest/{run_id}` + BacktestLab "→ Create Bot" button.

- [ ] **[P1] Kalshi desk — mirror the Polymarket pattern** (2026-07-19: public API live-verified, KALSHI_KEY_ID/KEY_SECRET provided in relay): kalshi_data.py desk feed from the public /trade-api/v2/markets + candlesticks (top markets by volume, KX: symbol prefix, guarded from the Alpaca order path like PM:), poly_* strategies run on it signal-only into #desk-kalshi; then RSA-signed order placement with the provided key pair. Backend /market-data/kalshi browse endpoint already exists.

- [x] **[P0] Render deploys fail at STARTUP — ROOT-CAUSED + FIXED 2026-07-19.** The Render application-log dump (added to deploy-on-main.yml) gave the real traceback: `app/api/v1/pipeline.py:15` did `Path(__file__).resolve().parents[5] / "pipeline_runs.json"` -> **IndexError: 5** at IMPORT on Render, where the file is `/app/app/api/v1/pipeline.py` (only 5 parents, 0-4) -> crashed the whole app before uvicorn could bind (nonZeroExit:1). Classic works-locally-dies-in-prod (local repo path is deeper). FIX: `_resolve_state_file()` searches ancestors for the file, honours a `PIPELINE_STATE_FILE` override, and falls back to `parents[min(4,len-1)]` — never IndexErrors; the reader already guards `if not _STATE_FILE.exists()`. Verified: full `app.static_server` import clean + pipeline tests green. NOTE: the sibling `parents[4]` files (monitoring/agents/experiments) resolve to `/` on Render — import-safe but wrong path; queued a shared repo-root helper as cleanup. SECRET_KEY was NOT the cause (already valid, len 44); the idempotent ensure-step stays as a safety net.

## 🚨 DEEP REVIEW 2026-07-18 PM — why "nothing is working" was TRUE on the live site
Root cause of almost everything: **render.yaml `autoDeploy: false`** — merges never
deployed. The live backend was a weeks-old build (29/57 templates, new endpoints 404,
bot scheduler dead since Jul 5, only 2 bots ever ran). FIXED this session: CI now has a
`deploy` job (green main → Render deploy → poll to live → page on failure) and bot
seeding is ADDITIVE (new templates become site bots on next boot). Remaining from the
evidence:
- [~] **[P0] Verify post-deploy revival chain — VERIFIED 2026-08-04 04:40, links 1-3 alive; link 4 is Supabase-blocked.**
  Checked each link that can be checked without auth:
  - **new build deploys** ✅ — proven this session: the `artifacts_on_disk` field added to `ml_models` appeared in
    production `/health/detailed` right after PR #1360 merged. `autoDeploy` works.
  - **bots on site** ✅ — `scheduler: {jobs_total: 73, bot_jobs: 64}`. The entry expected 57; there are 64, because
    `bots/factory.py` generates `[gen]` variants once the static templates are exhausted.
  - **scheduler alive / last_run_at advancing** ✅ — the strongest evidence. Sampled at 04:37 UTC, three bot jobs
    carried `next_run` of `04:38:52`, `04:38:59`, `04:39:02` — i.e. firing within the next minute, staggered.
    Also `background_tasks: 13/13 running`, `algo_agent: ok`, `strategies: 116`.
  - **paper orders → check_bot_exits → Trade rows → leaderboard → perf weighting/pruning** ❌ **BLOCKED** — this tail
    cannot complete while `database_primary` is down. Trades land in ephemeral sqlite and are wiped on every
    redeploy, so the leaderboard cannot stay non-empty and attribution-weight pruning stays inert. `/trades` and
    `/leaderboard` also require auth, so an anonymous probe cannot confirm even a transient non-empty state.
  **Net:** the revival this P0 was written to check DID happen — the deploy chain, the bot fleet and the scheduler
  are all live. What remains is not a revival problem; it is the single Supabase dependency already tracked above.
  Original entry: — — after the first auto-deploy: /health shows the new build; 57 bots on site; bot last_run_at advancing (scheduler alive); paper orders → check_bot_exits → Trades rows → leaderboard non-empty → perf weighting/pruning engage. Each link was dead on the old build.
- [x] **[P0] Discord empty — agent chat posted only to Slack (dead token)** — DIAGNOSED 2026-07-19 from live run logs: `multi_agent_discussion.py` and the desk's `_post_slack` delivered ONLY to Slack (invalid_auth), never Discord — so every conversation and desk P&L vanished. FIXED: both now deliver via `notify.discord_post` (bot-token→channel routing, same helper other flows use). Discussions post the opening + real LLM replies to the matching #channel; the desk posts P&L/fills to #desk-*. Remaining: the ~1s discussion runtime shows the **free-LLM keys are unset** → no real replies; the script now posts ONE actionable Discord notice ("add GROQ_API_KEY_1/GEMINI_API_KEY_1/DEEPSEEK_API_KEY") instead of silence. **User action: add any one free-LLM key** to turn on real conversations. Same Slack-only audit still TODO for team-lead-issues + daily-employee-review + employee-conversations.

- [x] **[P1] Move completely off Slack → Discord** (user directive 2026-07-19) — **DONE 2026-07-25.** The whole long-tail is converted: no script calls slack.com/api, none reads a Slack token, the 11 slack-*.yml workflows and 6 slack-only scripts are deleted, SLACK_BOT_TOKEN is stripped from all 65 workflows, and the backend Slack notifier is replaced by `app/notifications/discord.py`. See the 2026-07-25 deep-review section at the top for the silent-message-loss defects this uncovered.
- [x] **[P0] Desk fills → backend Trades attribution** (this is why the website showed "no trades") — FIXED 2026-07-20. ROOT CAUSE found: `sync_desk_trades` (already wired into the scheduler every 15 min) filtered accounts on `encrypted_key IS NOT NULL`, but the ONLY Alpaca paper account is the seeded demo account (`demo@quantedge.app`) which stores NO key — the desks trade on the **env** `ALPACA_API_KEY` (GH-Actions secret relayed to Render), which had no DB Account. So the account filter matched nothing → **zero** `qe-*` desk fills ever became `Trade` rows (global leaderboard + every per-user view empty). FIX (`app/tasks/desk_trade_sync.py`): added an **env-keyed fallback** — when `settings.alpaca_api_key/secret` are set, fetch the desk account's closed `qe-*` orders directly and attribute the reconstructed round trips to the system/demo (keyless) paper account; idempotent by `close_order_id`; skips if a keyed account already covers that key (no double-count). 2 integration tests + refactor keeps the 13 unit tests green. REMAINING UX follow-up (P1, below): these are shared *house* desk trades attributed to the demo account, so they show under the demo login + the global `compute_live_strategy_performance` weighting; surfacing them to an arbitrary user's own login (or a guest view) is a separate product decision. Note on cadence: crypto trades 24×7, equities only weekdays 9:30–16:00 ET — low weekend volume is expected, not a bug.
- [ ] **[P1] Surface house/desk activity to the logged-in user / guest view** (follow-up to the P0 above; user: "My login shows 0 data … Guest doesn't work") — desk Trades are attributed to the demo account; `/trades` scopes to `Account.user_id == current_user.id`, so a fresh user login sees nothing. Decide + implement: either a read-only "Platform / House desk" activity surface visible to every login, or attach the demo account's activity to the guest/demo session so the site is never empty.
- [x] **[P1] Employee individual memory depth** — SHIPPED 2026-07-24: the multi-agent discussion now wires each speaker's private `EmployeeBrain` in. `EmployeeBrain` gained an in-memory `mem=` mode (operates on the caller's already-loaded agent_memory.json dict, `save()` no-ops so it can't clobber the discussion's other in-flight writes — the bug that would've corrupted conversations). Each speaker now: (1) recalls its own `context_block()` (private history + durable facts + what teammates just shared) prepended to its prompt, and (2) records its contribution back to `employee_context[emp].history` so next run it remembers what it said. Prompt also nudges "build on what teammates said" for real back-and-forth, not parallel monologue. 3 new tests (in-place, shared-bus, disk-untouched). Runs on Actions → live immediately, no backend deploy.
- [x] **[P1] Fix all Discord channels** — SHIPPED 2026-07-24: `discord_setup_channels.py` created only 14 channels but scripts post to ~25 (audited via grep of discord_post/post_dedup + company_brain KNOWLEDGE_CHANNELS + desk placers + bot lifecycle + multi-agent). The missing ones (desk-commodities/kalshi, desk-research, desk-lead-review, market-analysis, strategy-lab, strategy-performance, signals, incidents, bot-fleet, trading-floor, squad-backend, ml-research, desk-tv-indicators, bot-research) were falling back to a #general dump. STRUCTURE now covers all 29 across 4 categories → every post lands in its real channel.

## 🚨 LIVE OUTAGE 2026-07-20 — "everything seems broken" ROOT-CAUSED
- [x] **[P0] Live site dead: Supabase project PAUSED** — evidence from `/health/detailed` on the keeper backend (quantedge-api-9jz0.onrender.com): `database: (ENOTFOUND) tenant/user postgres.vexzwnfbmznvxoxxktax not found — SUPABASE PROJECT MAY BE PAUSED`. Every DB-touching endpoint 500'd (login, /auth/demo, register, trades, bots) while `/health` stayed green — the site looked completely broken. FIXED in code: `ensure_database_alive()` (app/database.py) probes the primary at boot and falls back to local SQLite (rebinds `AsyncSessionLocal` in place; creates schema; bots reseed; desk trades resync from Alpaca 30-day history). Surfaced as a failing `database_primary` check so status stays `degraded` and watchdogs keep paging. 4 tests (`test_db_fallback.py`). **USER ACTION for durable state: supabase.com/dashboard → find the project → Unpause** (free tier pauses after 7 idle days; ~90 days until data loss). Until then the fallback DB is ephemeral (resets each deploy).
- [x] **[P1] Supabase keep-alive** — SHIPPED: db-keepalive.yml, 3× daily — once unpaused, add a tiny scheduled workflow ping (one cheap SELECT via the pooler a few times/day) so the free-tier project never idles into a pause again.
- [x] **[P0] Frontend pointed at the WRONG backend** — root-caused 2026-07-20: `frontend/vercel.json` proxied `/api/*` to `quantedge-api-agb8.onrender.com`, a STALE deploy (29 bots = pre-07-18 build, 0 trades, no relayed env keys) — while every new merge deploys to the keeper `quantedge-api-9jz0.onrender.com`. So users saw an old, empty app no matter what shipped. FIXED: rewrite now targets 9jz0. Follow-up decision for user: delete/suspend the agb8 service (it costs a free-tier slot and confuses debugging).

## LLM/Discord silence — ROOT-CAUSED + FIXED 2026-07-20 (second pass)
- [x] **[P0] 29 workflows had NO paid-LLM backstop** — `ai-pr-review.yml` (the one flow that visibly produces real LLM output) passes `OPENROUTER_API_KEY` + `ANTHROPIC_API_KEY`; the other 29 LLM workflows (multi-agent-discussion, daily-standup, collective-learning, peer-review, team-lead-issues, employee reviews, watchdogs…) passed ONLY free-tier keys — when those rate-limit/fail the conversation dies silently. That asymmetry is why the AI review works while Discord stays quiet. FIXED: backstop keys added to all 28 consumer workflows (key-relay excluded — not an LLM consumer). The Anthropic Messages API path already existed in `llm_common` (`ANTHROPIC_API_KEY`/`_2`) — it was simply never fed in these jobs.
- [x] **[P1] Everyday-improvement visibility loop** — SHIPPED 2026-08-05. `backend/performance_log/strategy_performance.json` is written by `fill_tracker.py` from real fills and IS committed (22 strategies, 247 tracked order ids) — **nothing consumed it into agent context**, so the daily discussion ran entirely on self-reported status while the results sat in a file. `shared_context.outcome_learnings()` now injects ranked attribution into `peer_learnings` via `multi_agent_discussion`, and **always includes the worst performer** (`stat_arb_e`: 21 trades, 0% win, −37.17% — never once discussed) because surfacing only winners is how status theater starts. Second half, same loop: **56 of 200 peer_learnings entries (28%) were LLM prompt echoes** — the model restating its instruction after a failed generation, stored as a learning by `agent_status_checker` and `multi_agent_discussion`, and (since the retrieval fix landed the same night) *retrieved into other agents' prompts*. `is_low_quality_learning()` filters at the write boundary; patterns are deliberately narrow because a false positive silently discards real signal. 24 tests, 7 mutations caught — the last only after pinning list COMPOSITION rather than membership: reversing the sort kept the worst performer present while turning "top N winners + the worst" into "worst N + the best". — now that desk fills become real `Trade` rows (PR #764) and conversations have a working LLM backstop, wire the daily P&L attribution INTO `peer_learnings` so agents discuss actual results each morning in #trading-floor (outcome-linked learning, not status theater).

## Session 2026-07-20 PM — shipped
- [x] **ml_signal bot condition (OA format)** — bots can now use the ML model as a decision recipe: `{"type":"ml_signal","direction":"up","min_confidence":0.65}` (engine runs inference once per tick, only when used; no trained model → condition is False, never an error). 8 tests. Combine with indicator conditions under ALL/ANY exactly like OA stacks decisions. TV webhooks already existed (`POST /api/v1/webhooks/tradingview` → Redis `tradingview:alerts`).
- [x] **Discord charts** — `notify.discord_post_chart()` renders QuickChart image embeds (green/red signed bars, no deps/keys); daily P&L report now posts a net-notional-by-symbol chart alongside the text. Roll out to desk P&L, leaderboard digest, bot lifecycle next.
- [x] **client.ts base-URL normalization** — a `VITE_API_URL` ending in `/api/v1` (the OLD documented value, likely still in the Vercel env) double-pathed every request (`…/api/v1/api/v1/auth/demo` → 404): reproduced in a real browser, this alone breaks guest login + all data even with a healthy backend. Client now strips the suffix. NOTE: Vercel hit its 100-deploys/day free limit (~05:11 UTC) — no frontend change goes live until it resets.

## ML experiments — why "not working at all" (diagnosed 2026-07-20)
Live evidence: `/health/detailed` → `ml_models: count 0`. Three stacked causes: (1) **torch is not installed on Render** (free tier: LSTM/PatchTST/SSM/Mamba log "unavailable" at every boot — by design, graceful degrade); (2) `models_dir` is **ephemeral** on Render — even a trained artifact vanishes on redeploy; (3) experiments that DO run in CI never ship artifacts anywhere the backend can load.
- [ ] **[P1] Scalable ML pipeline (the fix)**: train sklearn/GBM models (torch-free — this is also what most desks actually run in production) in a scheduled GitHub Action, commit versioned artifacts (small .pkl/.ubj) to the repo or a GitHub Release, and have the backend download/load at boot. Then `ml_signal` bots + `/ml` endpoints go live with real predictions. Walk-forward gate before any artifact is promoted.

## What top firms do that we don't (gap audit 2026-07-20, queued)
- [x] **[P1] Overfit killers in strategy promotion** — SHIPPED 2026-07-22: three Bailey & López de Prado stats now live in `backtest/cpcv.py` as pure, reusable functions — `deflated_sharpe_ratio` (rewritten to the correct PROBABILITY form: multiple-testing haircut over the trial Sharpes; the old dead z-score with a `sqrt(var+1)` bug is gone), `probabilistic_sharpe_ratio` (short-track-record + skew/kurtosis significance), and `probability_of_backtest_overfitting` (CSCV over an N-config returns matrix — noise→~0.5, genuine edge→~0). WIRED into the walk-forward path: `walk_forward()` now returns a `robustness_verdict` (`is_robust`/`deflated_sharpe`/`consistency`/`n_windows`/`verdict`) enforcing the module's own documented protocol (≥12 OOS windows, avg & majority of windows ≥0.7 Sharpe, DSR≥0.90) — previously that protocol was a comment, unenforced. The `/backtests/walk-forward` task persists the verdict into `BacktestRun.params.overfit` (no migration) so the leaderboard/promotion can refuse overfit runs. ALSO fixed a latent bug: `walk_forward()` was missing the `initial_equity` param the API always passed → the endpoint `TypeError`'d and silently failed every run. 20 tests. NEXT (optional): wire PBO into the multi-config leaderboard/auction selection (needs a per-config returns matrix); gate `backtest_to_bot` on PSR. — deflated Sharpe ratio + probability-of-backtest-overfitting (PBO) gates before a strategy is promoted/scaled (Bailey & López de Prado line of work). We only do walk-forward today.
- [ ] **[P1] GBM-on-features as the ML workhorse** — production quant desks overwhelmingly run gradient-boosted trees on engineered features with strict leakage control, not deep nets; aligns with the CI-trained-artifact pipeline above.
- [ ] **[P2] Portfolio-level risk (factor exposures + stress)** — we cap drawdown/Kelly per strategy; firms add net factor exposure limits (beta/sector/duration) + scenario stress across ALL desks and countries at once. Build on the desk-consolidated Trades now that attribution works.
- [ ] **[P2] Execution: queue-aware child orders** — SmartPricing ladder (queued) + participation-rate caps; measure realized slippage vs arrival in the slippage dashboard.
- [ ] **[P2] Cross-desk/country correlation watch** — international desk (19 ADRs) + FX + crypto share risk factors; add a rolling correlation matrix alert when diversification collapses (crisis mode → cut gross).
- [ ] **[P2] Agentic-trading literature sweep** — multi-agent debate (Du et al. '23), Reflexion (Shinn '23 — already used by the improver), skill libraries (Voyager '23), memory-augmented trading agents (FinMem '23), RL execution (FinRL). Apply-not-cite: debate gate + skill library are queued above.

## SOTA multi-agent 24×7 queue (user ask 2026-07-20)
- [ ] **[P1] Debate gate for large desk orders** — bull/bear/judge 3-role LLM debate (one `llm()` call each, shared context) before any paper order above a notional threshold; verdict + reasoning posted to the desk channel. (Du et al. multi-agent debate; cheap with the cascade.)
- [ ] **[P2] Market-based task allocation on AgentBus** — auction `agent-fix-needed` issues to employee agents by bid (past success rate per improvement_type from improvement_stats) instead of round-robin; the auction record IS the Discord conversation.
- [ ] **[P2] Verifier-role expansion** — extend ai-pr-review into a two-pass generate→verify pattern for Free-Agent Engineer PRs (verifier must reproduce the failure the PR claims to fix before automerge label is granted).
- [x] **[P2] Shared-brain retrieval upgrade** — **ALREADY BUILT, and it had silently regressed.** `memory_manager.SemanticRetriever` (pure-python TF-IDF, no ML libs) has been live for a while: `llm_common._MEMORY_MANAGER_OK` is True, so `inject_company_context` already routes through `_build_context(prompt)`, not the newest-N snapshot. **2026-08-05:** its default category list searched three names that have never existed in the brain (`skills`, `chat_insights`, `trade_outcomes`) and omitted `desk_outcomes`, which held **100 of the brain's 403 entries** — every desk run with side, notional and confidence per order. The recency path it replaced *did* include them, so the upgrade regressed agent context for trading outcomes while looking like an improvement. Fixed + `_unsearched_categories()` warns at runtime (not in CI — the brain is bot-written; a test asserting on it can turn the suite red at 3am).

## SOTA research + pipeline upgrades queue (user ask 2026-07-20)
- [ ] **[P2] SOTA sweep — execution**: SmartPricing-style laddered repricing (already queued below), plus survey adaptive limit-order placement literature (queue-position aware repricing) for _ensure_filled.
- [ ] **[P2] SOTA sweep — ML**: evaluate PatchTST/iTransformer-family for the existing feature pipeline vs the current LSTM/ensemble; only via walk-forward gate, paper-first.
- [ ] **[P2] SOTA sweep — portfolio**: compare current HRP against NCO (nested clustered optimization) and turnover-penalized variants on the desk universes.
- [x] **[P2] Pipeline hardening**: ~~CI: cache uv wheels for faster runs~~ **NOT WORTH DOING — measured 2026-08-05** (run 30966385122). The `test` job already caches `~/.cache/uv`, and its install step no longer registers (<2s); the cache restore itself is 4s. `test-agents` installs in **5s of a 36s job**. The actual cost is `Run tests in parallel` at **145s of 178s** — caching wheels cannot touch it. Any future CI-time work belongs in the test run, not the install. ~~deploy: post-deploy smoke now hits /health — extend to /health/detailed and fail deploy on `database.ok=false`~~ **DONE 2026-08-05** — with a correction: `database.ok` is **true** on the SQLite fallback (`main.py:487` — the fallback answers `SELECT 1`), so the guard as specified could never fire. It keys on `database_primary.ok` (`main.py:502`), and is strict only on `push` (`SMOKE_FAIL_ON_DEGRADED_DB`) so the 30-min schedule doesn't page 48×/day for an operator-blocked pause. Also asserts `mode == "paper"`.

- [x] **[P1] `agent_memory.json` was unbounded and is 47% of the git repo** — SHIPPED 2026-08-05. Its `conversations` dict had three writers appending timestamp-keyed entries and no trim: **915 entries / 576 KB inside a 933 KB file**, growing ~7 KB per commit, with **200 of the last 200 commits rewriting the whole blob** — so git holds 2340 copies, **59.1 MB of a 125 MB `.git`**. The only reader is `context_sync.py`, which takes `sorted(convs.items())[-20:]` and displays 10; ~900 entries were retained to serve a consumer that wants 20. The same file already capped `peer_learnings[-100:]`, `failure_traces[-200:]` and `employee_context` (`_HISTORY_CAP = 60`) — the discipline just never reached the largest structure. Capped at 300 via `shared_context.trim_conversations`; applied to the live file it is **915 → 300 entries, 44% smaller**, newest retained. Only the three *producers* trim: the other ~11 writers of this file add no conversation entries, and since each is a load→mutate→rewrite process, trimmed entries are never resurrected. 14 tests; 5 mutations caught after fixing a fixture that inserted keys pre-sorted and so could not distinguish sorted order from insertion order. Does **not** shrink the existing 59 MB of history — that needs a history rewrite (operator decision).

## Autonomy hardening (2026-07-20)
- [x] **[P1] ~~Improver PRs BYPASS CI entirely — the reward gate never runs on them~~ RESOLVED — verified 2026-08-04 03:45.**
  Improver PRs now receive the full required suite. Evidence, two independent PRs: `#1372` and `#1358` each show
  `test`, `test-agents` and `frontend-build` all `success` (plus `ops-sync`). `#1377` merged through `auto-merge.yml`,
  which refuses any PR whose REQUIRED_CHECKS never ran (lines 119-125), so its passing is itself proof CI executed.
  **Cause of the fix:** `continuous-improvement.yml` lacked `actions: write`, so its `POST /actions/workflows/.../dispatches`
  returned 403 on every run — and the failure was invisible because the step carried `continue-on-error: true`, which
  renders a non-zero exit as a GREEN step. Permission added earlier this session; improver PRs have had CI since.
  **This retires a standing operating assumption**, so it is worth stating plainly: the premise "improver PRs bypass CI
  so main can silently break" is no longer true. The per-tick backend-import check remains worthwhile as cheap
  defence-in-depth (it costs seconds and covers non-PR paths), but it is no longer guarding an open hole.
  Original entry: (found 2026-07-22). Evidence: PR #876 ("improve(strategy_logic)", bot-opened) reached `main` with ONLY a `Vercel` check — `test`/`test-agents`/`frontend-build` NEVER ran. Root cause: CI triggers on `pull_request`, but GitHub suppresses workflow runs for PRs opened by the bot's `GITHUB_TOKEN` (the same recursion guard that stops auto-merge.yml firing on bot PRs). So the "reward gate = full CI green" is a NO-OP for every improver PR — it merges unguarded. #876 shipped a nonsensical LLM hallucination (HTTP `X-Strategy-Entry/Confirmation` header validation wrapped around the WHOLE strategies router) that 400'd every `/api/v1/strategies/*` GET — dashboard strategy list + demo session dead. FIXED the live regression this session (revert in `router.py`, endpoint-smoke + demo-session tests green). **[MITIGATED 2026-07-24 — option (c) shipped]** `auto-merge.yml` had the exact hole: it filtered check-runs then checked `unfinished`/`failed` — but a bot PR with ZERO check-runs (CI never ran) gave empty arrays → "no checks" read as "all green" → merged unvalidated (this is literally how #876/#929 reached main and broke boot 3 sessions running). FIX: the gate now REQUIRES `test`/`test-agents`/`frontend-build` to be PRESENT and successful on the head sha; missing = refuse to merge. Improver PRs now pile up open (harmless) instead of landing unchecked. Takes effect immediately (workflow layer, no deploy). REMAINING (better, needs owner/PAT): (a) branch protection on `main` requiring those checks — belt-and-suspenders; (b) make the improver dispatch CI via `WORKFLOW_PAT` so its PRs actually get validated and can legitimately merge again.
- [x] **Continuous-improver safety rails** — FIXED (this session). Root-caused why the ~dozens of stuck `improver/run-*` PRs (backlog to 2026-07-05) could never be safely merged: (1) each PR bundled a stale snapshot of `.github/state/agent_memory.json` + `skill_library.json`, so merging ANY of them REVERTED newer live agent memory that other workflows write to main continuously; (2) the reward gate is full-CI-green, which does NOT cover the behavior of whole-file LLM rewrites — so green PRs still regressed the money path (e.g. `ml_breakout` stopped suppressing ML-unconfirmed signals; `ensemble`/`rsi2_pullback`/`hrp`/`iceberg` silently reworked). FIX (`continuous_improver.py`): (a) `_is_protected` bars the improver from `strategies/`, `execution/`, `risk/`, `ml/models/`, `bots/` — it may only touch non-core files (api, schemas, utils, tasks, integrations); (b) the run branch is reset so `.github/state/**` never enters the PR diff. 4 guard tests. Stale backlog to be closed (unmergeable, superseded).

## New queue (added 2026-07-18, OA-backend session)
- [x] **[P1] Tradier sandbox options-data adapter** — SHIPPED 2026-07-20 (PR #762): `.github/scripts/tradier_data.py` (fail-soft: quote/expirations/nearest_expiration/chain-with-greeks/pick_by_delta/atm_iv). Desk mleg spreads now pick short/long legs by target delta (~0.30Δ short / ~0.15Δ long) via one live chain fetch per underlying, falling back to spot*moneyness when the feed is down. `TRADIER_SANDBOX_TOKEN` wired through desk-trading.yml + test.yml relay. Live-verified: SPY 0.30Δ put→728 (δ −0.2947), ATM IV 0.1512; iron condor puts 728/699 calls 763/776. 14 tests. **User action to activate the real-greeks path: add the `TRADIER_SANDBOX_TOKEN` secret** (absent it, moneyness fallback runs — no regression).
- [ ] **[P1] SmartPricing-style laddered repricing** — extend _ensure_filled's one-shot cancel-replace into an OA-style ladder: post at mid, step limit toward market every ~7s (3 steps), then market out; measure in the slippage dashboard.

## New queue (added 2026-07-15, scale-up session)
- [x] **[P1] Per-desk performance attribution + auto-pruning** — DONE 2026-07-16: the live-leaderboard weighting now prunes proven losers to 0.0 (≥20 trades, negative P&L, sharpe<-0.5 → desk skips the strategy entirely; auto-revives when stats recover). 5 tests incl. missing-sharpe-never-prunes.
- [x] **[P1] Route Options-desk income structures through REAL multi-leg orders** — DONE 2026-07-18: wheel/condor/credit-spread/CSP/vol-carry signals now place actual defined-risk mleg spreads (moneyness-picked strikes ~35 DTE via /v2/options/contracts, 1 contract, day). Unresolvable legs place NOTHING and fall back to the underlying proxy. 9 tests incl. never-partial-spread.
- [x] **[P1] Symbol Scout** — DONE 2026-07-16: symbol_scout.py validates every desk symbol against /v2/assets (dead symbols get loud + queued), proposes unwired tradable crypto pairs + curated liquid ETFs; runs in the strategy-scout workflow; 6 tests.
- [x] **[P2] TV-desk hit-rate tracking** — DONE 2026-07-18 as a general pruning rule (applies to every strategy incl. TV): ≥100 trades + losing + win_rate<45% → weight 0.0, desk skips it. Profitable low-hit-rate trend riders are exempt by design (tested).
- [~] **[P1] Polymarket desk** — REAL DATA DONE 2026-07-18: desk now feeds the top-6 markets' hourly price bars from the public Gamma+CLOB APIs (live-verified) and logs real signals; PM: symbols are guarded from the Alpaca order path. REMAINING: order placement via py-clob-client signing — POLYMARKET_PRIVATE_KEY/KEY_ID/KEY_SECRET are provided in the secret relay, so this is implementable now.
- [x] **[P2] Research backlog → registry pipeline** — DONE 2026-07-18: Strategy Scout writes its top rotating idea to state/research_seed.json; strategy_generator injects it into the LLM prompt as PRIORITY RESEARCH DIRECTION (fail-open when absent). Verified end-to-end.

- [ ] **[P2] Unblock the 10 data-source-blocked strategies** (strategy-scout 2026-07-15; every WIRABLE strategy now trades — coverage 53→103): each needs a feed, not a desk: covered_call (share inventory), funding_rate_arb + crypto_basis_roll + dex_cex_arb (geo-permitted derivatives/DEX data), token_unlock_fade (unlock calendar), news_momentum (headline feed), earnings_accruals (fundamentals), micro_cap_momentum (small-cap universe), moc_auction_imbalance + order_flow_imbalance (intraday/L2 data).

> **Session 2026-07-15:** QUARANTINED emptied (hard-budget fail-soft); Commodities desk;
> FF red-folder gate for FX; **AlpacaBroker restored** (improver PR #420 had truncated
> 6 of 7 interface methods — improver now rejects shrunken/elided outputs, and
> test_broker_interface.py guards all brokers); employee-health hard gate pages Discord;
> tasks/ silent-except sweep; per-bot activity view (OA parity). Remaining open items
> below are the live queue.

- [x] **[P0] Fix 18 quarantined strategies** — DONE 2026-07-15: QUARANTINED is EMPTY. Final 6 (+5 latent) fixed via shared `app/strategies/_failsoft.apply_hard_budget` (detached-daemon-thread hard timeout; yfinance's curl_cffi bypasses socket kills, so this is the only real guard). Contract suite 115/115, 28s. ~~ (desk now guarded by a 10s per-strategy timeout — freezes impossible; remaining work is per-strategy fail-soft hygiene, remove from QUARANTINED as fixed) (contract-test audit 2026-07-11; the cause of desk freezes). Wrap fetches fail-soft w/ hard timeout, then remove from QUARANTINED in backend/tests/unit/test_strategy_contract.py. DESK-WIRED: ~~gamma_exposure, kalman_pairs, skew_arb, vrp_systematic, pead_sue~~ FIXED fail-soft 2026-07-11 (un-quarantined, contract-proven); credit_spread_income guarded but still slow offline (yfinance retries — stays quarantined); multi_factor_equity remains. Rest: lorentzian_knn, breakeven_inflation, dollar_carry, macro_risk_barometer, mvrv_zscore_timing, duration_momentum, yield_curve_momentum, pmi_sector_rotation, tlt_spy_rotation, yield_spread_reversion, basis_carry.


- [x] **[P0] scripts/import_oa_bots.py** + ~~scrape OA PUBLIC leaderboard~~ INVALID: 'optionsalpha.com' 200s were a wrong-domain false positive; real site (optionalpha.com) redirects all bot pages to /login (verified 2026-07-11). Cookie or Cowork are the only paths — both built
- [x] **[P0] OANDA FX desk** — fx_desk.py + fx-desk.yml, 7 majors 24/5, 7 tests; **[P1]** WORKFLOW_PAT fallback wiring; ADRs; EM cap 30%


> **How tasks are tracked (the answer to "where do tasks live"):**
> - **Canonical queue:** GitHub Issues labeled `agent-fix-needed` (the agents already
>   create/work these via `team_lead_issues.py` → `free_agent_engineer.py`).
> - **Human board:** [Notion — QuantEdge Tasks](https://app.notion.com/p/bec54f8a79444c2399316365a07e0291)
>   (seeded from this file; mirror via the *Notion ↔ GitHub Issues Sync* workflow).
> - **Cross-session continuity:** this file + `HANDOFF.md`, committed to the repo
>   (chat sessions are ephemeral — only what's committed survives).
> - **Slack:** notifications/visibility only — never the source of truth.

_Last updated: 2026-07-15_

---

## Session 2026-06-29 — review backlog (see `docs/REVIEW_2026-06-29.md`)
Queued for the autonomous loop / employees. Priority order top-to-bottom.
- [~] **[P0] OA Scout — auto-copy new Options Alpha bots daily** — BLOCKED on user unlock `OA_SESSION_COOKIE` (public pages auth-walled, verified 2026-07-11); importer + playbooks + issue template all built. — workflow fetches the
  public optionalpha.com template/library pages, diffs against .github/state/oa_library.json,
  LLM-parses any NEW bot into a BOT_TEMPLATES entry (delta/DTE/TP/entry window), opens a
  reward-gated PR, posts the find to #alpha-research. Runs daily + on CI events. Private
  account bots can't be scraped (auth) — screenshots remain the path for those.
- [x] **[P1] ForexFactory calendar feed** — DONE 2026-07-15: red-folder gate in fx_desk.py (±30min blackout per pair currency, fail-open, live-verified 99 events). — ingest the public ff_calendar_thisweek.json into
  /market-data/forex-calendar and gate Macro/FX desk entries around red-folder events.
- [x] **[P2] TradingView/FxReplay/Tradezilla** — DONE 2026-07-15: POST /webhooks/tradingview receiver (secret-gated, disabled-without-secret, normalizes alerts, /recent ring buffer, Redis fan-out; receives only — never trades). FxReplay/Tradezilla have no APIs — documented, closed. — no public trade APIs (manual UIs);
  TradingView useful as charts + webhook-IN alerts (receiver endpoint), not for dummy
  trading automation. Document + build the webhook receiver only.
- [x] **[P0] Options Alpha dashboard parity in the frontend** — DONE 2026-07-15: per-bot P&L graph (shipped earlier), + GET /bots/{id}/activity with open-positions & trade-history tables in BotBuilder expanded row; settings editor existed. — per-bot detail view in
  BotBuilder: cumulative P&L graph (endpoint /bots/{id}/performance is LIVE), open
  positions table (orders with bot_id in raw_payload), trade history (Trades by
  strategy_name == bot.name), settings editor (PATCH /bots/{id}). Use LWEquityCurve.
- [x] **[P0] Alpaca multi-leg options orders** — implementation existed (submit_alpaca_multileg_order, OCC symbols, delta picking, engine-wired) but had ZERO tests; 2026-07-15 added 8 tests which caught a real crash (structlog-style kwargs on a stdlib logger — a broker REJECTION crashed the caller instead of returning None). Fixed. — Alpaca paper supports options + multi-leg;
  extend brokers/alpaca_orders.py with the legs order shape so every oa_* bot fills REAL
  option legs with existing keys. Kills the TradeStation dependency.
- [x] **[P1] Synthetic options backtester** — DONE 2026-07-18: app/backtest/options_synthetic.py (BS pricer, realized-vol IV proxy, spread backtests mirroring the desk's mleg structures; limits stated honestly — no skew, conservative for short premium). 9 tests: put-call parity, defined-risk cap, theta harvest in flat tape, crash bleed. Original ask: Black-Scholes pricer over underlying OHLCV +
  realized vol to approximate premium-structure backtests (no chain history yet); gate
  bot enablement on a passing synthetic backtest (paper-first stays).
- [x] **[P0] Discord per-channel routing via bot token** — DONE 2026-07-12 (notify.py bot-token routing + DiscordBot UA + webhook fallback). — notify.py posts everything through ONE
  webhook into #general with a [#channel] prefix; channels exist now, so resolve channel name → id
  via the bot token (GET /guilds/{id}/channels) and POST /channels/{id}/messages, webhook fallback.
  Embed author = employee name for per-employee identity. Kills the "all channels empty" state.
- [ ] **[P1] LLM-brain employee personas on Discord** — desk/agent posts composed by llm_routed()
  with per-employee persona prompts (slack_agent_team.py personas exist), not fixed templates;
  numbers stay deterministic, only the commentary is generated. Two-way: reply when @mentioned
  via the interactions endpoint.
- [x] **[P0] Forex desk** — DONE 2026-07-13 as the OANDA FX desk (fx_desk.py, 7 majors 24/5, practice orders) rather than backend strategies; superset of the ask. — add `market_type="forex"` strategies (carry, trend/momentum), register,
      add "Forex" to `_MARKET_TYPE_DESK`, route data_loader to `EURUSD=X` etc., + scheduled desk.
- [x] **[P0] Commodities desk** — DONE 2026-07-15 via ETF proxies on the GitHub-Actions desk layer (GLD/SLV/USO/UNG/DBA/PDBC/GDX/CPER, TSMOM+Donchian+MR, regime-mapped, config guard tests). — add `market_type="commodity"` strategies (term-structure roll,
      momentum, gold/oil mean-reversion), register, add "Commodities" desk, route `GC=F`/`CL=F`.
- [x] **[P0] Render sleep** — keep-alive workflow chained to CI events pings /health 24x7 (event-driven, no cron starvation). — external uptime pinger (UptimeRobot) or paid tier so in-app employees
      don't halt (`/health` returned 000 — backend asleep).
- [~] **[P1] Audit & consolidate 86 workflows** — employee manifest shipped (`docs/WORKFLOWS.md`,
      via `scripts/gen_workflow_manifest.py`): 87 workflows / 70 scheduled, dup-families flagged
      (`slack-*`×10, `agent-*`×6, `render-*`×5, `strategy-*`×5). Next: actually dedupe the families.
- [x] **[P1] Durable auto-merge** — `auto-merge.yml` lands `automerge`-labeled PRs once all checks
      pass (no human merge). Removes the last manual step for the autonomous loops/employees.
- [x] **[P1] Employee-health hard gate** — DONE 2026-07-15: agent-health-check.yml no longer continue-on-error; critical findings fail the job AND page Discord #infra-alerts. — make the agent smoke test page on failure; verify
      `agent-health-*`/`system-status` actually alert when an employee is stale.
- [x] **[P1] Reward-gate self-improvement** — `continuous_improver.py` now pushes a throwaway
      `improver/run-*` branch and opens an `automerge` PR instead of pushing to `main`. The full CI
      suite must pass before changes land (auto-merge.yml). Stops the unvalidated direct-to-main
      commits that broke the app 3× (slots=True, @root_validator, dead scheduler) in one session.
- [x] **[P1] Wire Alpaca crypto into `price_feed`** — DONE 2026-07-15: root cause was improver PR #420 truncating brokers/alpaca.py (6 of 7 interface methods deleted → AlpacaBroker un-instantiable → silent yfinance fallback) + stale exception imports. Restored, guarded by test_broker_interface.py. for live quotes (Binance still geo-blocked for live).
- [~] **[P1] Narrow 435 broad `except Exception`** — tasks/ sweep DONE 2026-07-15 (every silent pass now logs); brokers/execution/risk money paths done earlier; remainder (llm, api) queued. — start with `tasks/`, `brokers/`, `llm`; add logging.
- [x] **[P1] Audit stale provider model IDs** — done: Cerebras gpt-oss-120b + NVIDIA deepseek slug live-verified, both env-overridable (CEREBRAS_MODEL/NVIDIA_MODEL).
- [x] **[P2] ML employees inert on prod** — already handled honestly: /health reports torch availability with non-critical 'degrades gracefully' status; ML strategies fail-soft. No further action. — run with `[ml]` extra on a worker, or mark degraded.

---

## Session 2026-06-24 — shipped (11 PRs merged to `main`)
- [x] **Options productization end-to-end** (#188): `OptionLeg` + `open_option_spread` schema,
      engine branch, TradeStation options API (chain + multi-leg order builders), 4 templates.
- [x] **Brain cascade fixed** — reasoning-model content extraction (Cerebras gpt-oss / R1) on
      `main` (#188) **and** on the default branch (#189: User-Agent, live model IDs) + in-call
      key fallthrough (#199). Verified live: groq/cerebras/nvidia answer.
- [x] **Backend-health banner + fresh-Render runbook** (#190) — `docs/RENDER_NEW_ACCOUNT.md`.
- [x] **TradeStation spread routing** (live-only, paper-first proven) + broker tests (#198).
- [x] **Kalshi public market reads** wired (#203) — matches the existing Polymarket endpoint.
- [x] **Tests added/guards:** income/macro strategy contracts (#202), TS options parsing (#198),
      pytest-asyncio deprecation removed (#206), **momentum lookahead causality guard** (#207),
      **cross-tenant isolation guard** (#208).
- [x] **Backlog hygiene:** closed 8 stale tsconfig issues + #193 (brain canary already exists).
- **Verified deploy-readiness:** booted backend locally → 158 routes, demo auth, seeds 29 bots
      /13 strategies/3 risk rules, Kalshi live. **Only blocker to going live = Render build-minute
      quota (#197) + default-branch flip to `main` (#196).**

---

## P0 — Reliability (the brain must never silently die)
- [x] **LLM cascade dead (Cloudflare 1010 / no User-Agent)** → fixed (#144).
- [x] **Cascade used only the primary key** → rotate across numbered variants (#145).
- [x] **Brain observability + always-on canary** → `cascade_status()`, `llm_metrics.jsonl`,
      `brain_health.py`, hourly `brain-health.yml` that alerts Slack #infra-alerts (this PR).
- [~] **Provider keys** — as of 2026-06-24, **3 work live** (Groq, Cerebras via gpt-oss-120b,
      NVIDIA via current model). Gemini=quota(429, recovers), DeepSeek=balance(402). Optional:
      add free SambaNova/Together/Hyperbolic keys to Doppler for more headroom. *(Drop key in Doppler; I wire the rest.)*
- [x] **Make the agent "smoke test" a hard gate** — smoke-test.yml pages Discord #ci-failures on failure; agent-health-check.yml hard-gated 2026-07-15 (fails + pages #infra-alerts on critical findings).

## P1 — Real bugs found this session
- [x] **`/ws/prices` all-symbols bug:** subscribed to literal topic `prices:*` but the feed
      broadcasts `prices:{symbol}`. Fixed: `ConnectionManager.broadcast` now fans concrete
      `prices:{symbol}` updates out to `prices:*` wildcard subscribers (+ regression tests).
- [x] **`test_realtime_endpoints.py` auth helper** — superseded by `test_realtime_live.py`,
      which authenticates with `email` + an `@example.com` address (no false-green skip).
- [x] **Redis default `localhost:6379`** spammed connection-refused. Fixed: prod default is
      now *unset* (`REDIS_URL` empty ⇒ clean no-op cache) **and** a connection-failure circuit
      breaker trips once, logs once, then no-ops for the rest of the process.
- [x] **3 broken workflows failing at YAML parse** (run name shown as the file path, 0 jobs):
      `slack-on-deploy.yml`, `agent-health-check.yml`, `gemini-ml-training.yml` — multi-line
      `run:` block scalars whose continuation lines lost their indentation. **Direct cause of
      "Slack dead except scheduled messages"** (deploy/health Slack posts never fired). Fixed;
      repo-wide workflow YAML lint now shows 0 broken.
- [x] **"TV Indicator SOTA" scheduled workflow** — investigated 2026-07-15: it was still cron-only (cron is starved on free tier → effectively never ran). Now event-chained to CI completions like the other 21 team workflows, with a ~6h cadence gate (git-log stamp) so the 20-min LLM job doesn't fire on every CI run.

## P1 — Issues the agents themselves flagged in Slack (live triage, 69/97 channels active)
- [x] `#deploys` — **cross-user data leak**: verified closed — all core routers scope to
      `current_user` (bots by user_id; orders/positions/trades by Account.user_id). Guard test (#208).
- [ ] `#leadership-summary` / risk — **VaR threshold exceeded**.
- [x] `#alpha-research` — **lookahead bias** in momentum strategies: verified — all 13 already
      `shift(1)`; causality regression guard added (#207).
- [ ] `#squad-qa` / `#ci-failures` — **test failures / bug** backlog.
- [ ] `#okrs` — **Sharpe-ratio shortfall** vs target.
- [ ] `#squad-backend` — **latency issues**; `#squad-frontend` — **screenshot upload failed**.
- [ ] `#finance-ops` — **upcoming paid triggers** (add spend caps before they fire).

## P2 — SOTA upgrades to make this a top-tier AI-first company
> Full durable research: `docs/research/AI_COMPANY_SOTA.md`,
> `docs/research/LLM_COST_OPTIMIZATION.md`, `docs/MODEL_ROUTING.md`.
1. **Observability + model routing** — Langfuse/OpenTelemetry traces on `llm_common`; route by
   task tier. *(Phase-1 metrics shipped; **cost-tiered `llm_routed()` ladder shipped** —
   free → OpenRouter open-mid → Claude backstop, env-configurable; Langfuse tracing next.)*
6. **Open-weight mid-tier so Claude is the rare backstop** — ✅ shipped in `llm_routed()` /
   `docs/MODEL_ROUTING.md`. DeepSeek/Qwen/Kimi/GLM/MiniMax via OpenRouter handle "hard" work at
   10–50× lower cost; Claude only on `tier="hard"` or last resort. Refresh `OPENROUTER_MODELS`
   to the exact current SOTA slugs as they rotate.
2. **Real memory layer** — replace flat `.github/state/*.json` with Mem0 or Letta backed by your
   existing **Supabase pgvector** (episodic + semantic recall).
3. **Outcome-driven self-improvement** — give the self-improver a *verifiable reward*
   (CI-green + coverage Δ + paper backtest Sharpe Δ); gate agent PRs behind an eval + LLM-judge.
   (DeepSWE / Darwin-Gödel-Machine pattern.)
4. **Durable, event-driven orchestration** — move the core loop (lead→engineer→reviewer) onto
   Temporal/Inngest/LangGraph durable execution instead of fire-and-forget cron.
5. **A2A agent protocol** — typed agent-to-agent coordination; demote Slack to a human digest
   (kills the repeated-message noise).

## Desk consolidation (staged — combine best of all desks/orders/tracking/risk)
> Execution (`execution/`) and risk (`risk/`) are already shared, desk-agnostic layers.
> `Bot` is already one unified JSON format across equity/crypto/polymarket. The work is
> consolidation, not a rewrite.
- [x] **Stage 1 — desk taxonomy (no migration):** `desk_of()` / `strategies_by_desk()` /
      `list_desks()` derive desks from existing attributes; `GET /strategies/desks` exposes the
      unified view (Equities 62 · Crypto 16 · TV 12 · Prediction Markets 8 · Options 7) + tests.
- [ ] **Stage 2 — extend the unified `Bot` format to all desks:** add `options`/`macro`/`rates`
      to `Bot.market_type`; tag finer desks via an explicit `desk` class attr (override hook
      already supported by `desk_of`).
- [ ] **Stage 3 — unified cross-desk tracking:** add `asset_class` + options instrument fields
      (strike/expiry/right/multiplier) to `Position`/`Order` (Alembic migration).
- [ ] **Options productization:** options desk is research-complete but not in the Bot builder
      (blocked on Stage 2/3); move scattered options strategies into `strategies/options/`.

## Housekeeping
- [ ] Deprecations: pytest-asyncio `event_loop_policy` fixture, Starlette `TestClient`+httpx,
      now-unused `passlib`.
- [x] Audit stale provider model IDs/endpoints in `llm_common` — duplicate of the 2026-06-29 item, done: live-verified + env-overridable (CEREBRAS_MODEL/NVIDIA_MODEL).
