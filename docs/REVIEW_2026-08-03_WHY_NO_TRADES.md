# Deep review — why there were no trades (2026-08-03)

Question asked: *"There should have been thousands of trades. Make sure each desk
works."* This is what the run logs, workflow history and code actually say.

Headline: **the desks are not broken and the strategies are not silent.** Signals
are generated on nearly every run. They are discarded, for two different reasons
on two different halves of the fleet, and both reasons are mechanical.

---

## 1. The equity-side desks: right signals, wrong time

Eight of the nine desks are gated by `desk_open = is_open or desk.always_open`,
where `is_open` comes from Alpaca's `/v2/clock`. They generate signals whenever
they run, and throw them all away when they run outside market hours.

Desk-trading run `30673525449` (2026-07-31 23:40 UTC), verbatim:

```
· stat_arb_etf/IWM            signal=BUY  conf=0.80 — logged (StatArb closed)
· time_series_momentum/SLV    signal=BUY  conf=1.00 — logged (Commodities closed)
· time_series_momentum/CPER   signal=BUY  conf=0.98 — logged (Commodities closed)
· time_series_momentum/UNG    signal=SELL conf=0.91 — logged (Commodities closed)
· low_volatility/EPOL         signal=BUY  conf=1.00 — logged (International closed)
· low_volatility/THD          signal=BUY  conf=1.00 — logged (International closed)
· sector_rotation/EIDO        signal=SELL conf=0.98 — logged (International closed)
[stage] ✓ Place orders — 0.0s  orders_placed=0  total_notional=+0.000
Done. 0 orders placed across 9 desks.
```

Confidence is not the problem here — these are 0.80 to 1.00. The problem is the
clock. `desk-trading.yml` is scheduled `*/15 9-22 * * 1-5`: nominally 52 runs per
weekday, 26 of them inside US regular trading hours (13:30–20:00 UTC). Measured
across its last 30 runs:

| day | runs inside RTH | runs total |
|---|---|---|
| Wed 2026-07-29 | 5 | 12 |
| Thu 2026-07-30 | 4 | 9 |
| Fri 2026-07-31 | 3 | 9 |

12 of 30 runs (40%) landed in the window, against a nominal 26 *per day*. The
desk is getting roughly **15% of its intended in-window cadence**, and the other
60% of runs are pure signal-generation with a guaranteed zero at the end.

Two causes, both confirmed:

**a. The anti-starvation trigger was dormant.** Both desk workflows declare
`workflow_run: workflows: ["CI"]` specifically so they can ride CI completions
instead of depending on cron. Across the last 30 runs of each: desk-trading was
28× `schedule` + 2× `push`, crypto 30× `schedule`. Zero `workflow_run` events on
either.

> **CORRECTION, 2026-08-03 08:50.** The first version of this section read
> "*has never fired … the mechanism exists in the YAML and has never delivered a
> run*". The 30-run sample was accurate; the conclusion drawn from it was not.
> `workflow_run` fires only for upstream runs on the **default branch**, and CI
> runs almost entirely on `pull_request`, whose `head_branch` is the PR branch —
> so no `workflow_run` event is emitted. The trigger was **dormant, not dead**.
> As soon as CI actually ran on main (this review's own PR), it fired on both
> desks at 06:43 and 07:45. On desk-trading both runs concluded `success`.
>
> On the crypto desk both concluded `skipped`, because its job carries
> `if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'`.
> That gate is **correct and deliberate** — see §4 — not a bug.

The practical consequence for cadence is unchanged: because CI so rarely runs on
main, the desks have in practice depended on the starved cron alone.

**b. The pacemaker was cancelling itself.** `pacemaker.yml` is shaped "sleep
3000s, then dispatch CI", and carried `concurrency: cancel-in-progress: true`.
Its ignition sources are independently phased and arrive far more often than
every 50 minutes, so each arrival killed the sleeper before it reached the
dispatch. Last 30 runs:

```
cancelled  25   durations 0.1 – 47.9 min   (never 50)
success     4   durations 50.2 – 50.4 min  — the only ones that dispatched
running     1
```

The losses were recorded as `cancelled`, which no alert path watches and which
reads as green at a glance.

### Not a bug: the weekend gap
desk-trading's last run before this review was Fri 2026-07-31 23:40, which looks
like a three-day outage. It is not — `* * 1-5` correctly excludes Sat/Sun, and
2026-08-03 is a Monday whose window had not opened yet.

---

## 2. The crypto desk: right time, unreachable gate

The crypto desk is `always_open=True` and runs 24/7, so the clock is never its
problem. Its problem is arithmetic.

Crypto desk run `30782697088` (2026-08-03 03:47 UTC), verbatim:

```
[stage] ✓ Generate trading signals — 3.84s  signals_generated=16
· crypto_adaptive_trend/BTC/USD  conf=0.37 < 0.60 — skipped
· crypto_adaptive_trend/ETH/USD  conf=0.26 < 0.60 — skipped
  ... 16 of 16 skipped, confidences 0.23 – 0.44 ...
[stage] ✓ Apply confidence threshold + top-K filter — passed=0  filtered=16  explored=0
[stage] Place orders — ⚠ Equity market closed — only always-open desks may trade
[stage] ✓ Place orders — orders_placed=0
Done. 0 orders placed across 9 desks.
```

Two things are wrong.

**a. Only one strategy is producing signals.** The desk lists 14 strategies over
20 symbols. 16 signals arrived and every one was `crypto_adaptive_trend` —
16 symbols × 1 strategy. The others fail on data acquisition from US runners:
`MVRVZScoreTimingStrategy: failed to fetch CoinGecko data — 429`, and
`OI fetch failed for XRPUSDT: HTTP Error 451`. They are counted as desk coverage
and deliver nothing.

**b. The confidence gate is above the strategy's ceiling.** `analyze()` computes

```python
vol_scalar   = min(target_vol / max(rv_21, 0.05), 3.0)   # target_vol = 0.40
sized_signal = raw_signal * vol_scalar                    # |raw_signal| <= 1
confidence   = min(abs(sized_signal) / 2.0, 0.95)
```

so `confidence` is a **position size**, not a conviction, and it *falls as
volatility rises*. The ceiling is `target_vol / (2 · rv_21)`:

| realized vol (21d, ann) | max attainable confidence | clears 0.60 gate | clears 0.45 explore floor |
|---|---|---|---|
| 20% | 0.95 | yes | yes |
| 33% | 0.60 | just | yes |
| 40% | 0.50 | no | yes |
| 50% | 0.40 | no | **no** |
| 65% | 0.26 | no | no |
| 80% | — | signal suppressed by `min_signal` | |

Crypto's realized vol sits in the 45–80% band. Both the order gate (0.60) and
the exploration floor (0.45) are above what the strategy can emit. Reproduced
locally against synthetic bars with a +0.4%/day trend — as unambiguous an uptrend
as crypto produces — confidence was 0.419 at 50% vol and 0.255 at 65%.

The desk's best signal in production was **0.44 against a 0.45 exploration
floor**. It missed even the consolation path by one hundredth.

### The obvious fix is wrong, and was rejected on evidence
Dropping the vol scalar so `confidence = |raw_signal|` makes the gate reachable —
and makes the strategy trade noise. `analyze()` derives conviction from
`tanh(composite_raw * 5)`, which saturates almost immediately: on a **zero-drift**
random walk at 55% vol it returns 0.834, 0.922 and 0.936 for three seeds. That is
worse than not trading, so it was reverted rather than shipped.

A correct repair recalibrates conviction itself (e.g. a risk-adjusted
momentum/vol score with a gain that discriminates), which under this repo's own
"walk-forward only" standard is a backtested strategy change, not a patch. It is
left for the operator.

### Related: live and backtest are different functions
`backtest_signals()` builds its composite from `.rank(pct=True)` percentiles;
`analyze()` uses `tanh(composite_raw * 5)` on raw returns. The docstring claims
"uses same logic as backtest_signals on recent bars". It does not. Any backtest
of this strategy describes behaviour the live desk never exhibits — the same
class of defect as the two divergent `LSTMPredictor` architectures.

---

## 3. Discord

Delivery works: `[notify] delivered #pnl-daily via BOT → channel_id=…` on every
run. Two observations.

- **The per-desk channels are silent by construction.** `_post_chat(desk.chat_channel, …)`
  is inside `if desk_order_list:`. No orders means no post, so eight desk
  channels have had nothing to say for as long as orders have been zero.
- **The failure was being reported all along.** The `#pnl-daily` summary carries
  per-desk funnel telemetry — `⚠️ *Commodities*: 3 signal(s) fired, **0 placed** —
  3 market closed`. That message has been posted every run, for days. The
  instrumentation was correct and unread.
- Minor: `Equities` and `International` both use `chat_channel="#desk-equities"`,
  so International's summaries land in the equities channel.

---

## 4. What was changed in this pass

Only the mechanical, no-judgement-required half. The strategy calibration is
deliberately untouched.

- `pacemaker.yml`: `cancel-in-progress: true` → `false`. Restores the ~50-minute
  heartbeat that was landing 4 times in 30.
- `pacemaker.yml`: new step dispatching **`desk-trading.yml` only** directly.
  `workflow_dispatch` is the documented exception to the GITHUB_TOKEN recursion
  guard and is already proven in this repo, so it depends on neither starved cron
  nor CI happening to run on main. Safe at any hour: `desk_order_placer` checks
  Alpaca's clock itself, and desk-trading uses `cancel-in-progress: false`.

  **The crypto workflow is deliberately excluded**, and the first version of this
  step got that wrong by dispatching both. `desk-trading.yml` runs all nine desks
  and crypto is `always_open=True`, so one dispatch already covers it 24/7.
  `desk-trading-crypto-24x7.yml` gates its job on `schedule || workflow_dispatch`
  expressly to cede any trigger it shares with the equity workflow: the two use
  different concurrency groups, so on a shared trigger they run in **parallel**
  and compete for Alpaca's free-tier data limit. Measured over 60 runs
  (2026-07-28), 22 collided — one pair on sha `49e46ded` had desk-trading fetch
  70 bars while the crypto-only run got 5 and 429s, i.e. the crypto run was
  strictly worse *and* degraded its own twin. `workflow_dispatch` is on that
  allowlist, so a pacemaker dispatch is the one route the cede-rule cannot block.
  Pinned by `test_the_crypto_desk_is_not_dispatched_too`.
- `.github/scripts/test_pacemaker_actually_delivers.py` — 12 tests pinning the
  concurrency setting, the sleep-vs-timeout relationship, all four dispatch
  targets, that each target declares `workflow_dispatch` (or the POST 404s), and
  that a desk-dispatch failure cannot abort the CI heartbeat. Mutation-checked:
  reintroducing each of the three regressions fails the suite.
- `backend/tests/unit/test_desk_confidence_gate_is_reachable.py` — records the
  crypto gate arithmetic, reading the live threshold out of `DESKS` rather than
  copying it. The two unreachability assertions are `xfail(strict=True)`, so CI
  stays green today and reports XPASS the moment someone recalibrates.

## 5. Left for the operator

1. **Recalibrate crypto conviction** (needs a walk-forward backtest) and
   reconcile `analyze()` with `backtest_signals()`.
2. **Reroute or retire the geo-blocked crypto strategies** — CoinGecko 429 and
   Binance/OI 451 make several of them permanently dead weight on US runners.
3. **Supabase project `vexzwnfbmznvxoxxktax` is paused.** Until it is resumed the
   backend runs on ephemeral sqlite, so whatever trades *do* get placed are wiped
   from history on every Render redeploy — which is why the leaderboard looks
   empty independently of everything above.
4. Give `International` its own Discord channel.
