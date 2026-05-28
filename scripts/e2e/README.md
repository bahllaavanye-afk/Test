# QuantEdge E2E Demo Workflow

This directory contains the end-to-end demo recorder (`demo-recorder.js`) and seed helper (`seed_demo.py`) that drive the GitHub Actions `e2e-demo` workflow. The workflow boots the full stack, walks a Playwright browser through every page of the dashboard, records videos + screenshots, and uploads the result as a GitHub Actions artifact.

This README is for engineers and AI agents who need to run, debug, or extend the demo.

---

## 1. Required GitHub Secrets

The workflow needs four repository-level secrets to run. Set them in **Repo → Settings → Secrets and variables → Actions → New repository secret**:

| Secret name           | Where to get it                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| `ALPACA_API_KEY`      | https://app.alpaca.markets — Paper account → API Keys → generate                |
| `ALPACA_SECRET_KEY`   | Paired secret shown once at the same time as the key. Save it immediately.      |
| `DATABASE_URL`        | Supabase project → Settings → Database → Connection string (pooler, port 6543). |
| `REDIS_URL`           | Upstash Redis console → REST URL (`rediss://default:<token>@<host>:6379`).      |

Use the **paper** Alpaca account, never the live one. The workflow places real-looking but non-funded orders.

Optional secrets (the workflow degrades gracefully without them):

- `TRADESTATION_CLIENT_ID`, `TRADESTATION_SECRET` — enables the TradeStation broker page test.
- `BINANCE_API_KEY`, `BINANCE_SECRET` — enables the Binance broker page test (use testnet keys).
- `POLYMARKET_PRIVATE_KEY` — enables the Polymarket page test.

If you only set the four required secrets, the broker pages for TradeStation / Binance / Polymarket will render an empty state instead of live data, which is expected and not a failure.

---

## 2. Triggering the Workflow Manually

The workflow file lives at `.github/workflows/e2e-demo.yml`. To run it on demand:

1. Open the repo on GitHub.
2. Click the **Actions** tab.
3. In the left sidebar, click **e2e-demo**.
4. Click **Run workflow** (top right).
5. Pick a branch (defaults to `main`). For PR validation, pick the PR branch.
6. Optionally set inputs:
   - `headed` (bool, default `false`) — set to `true` for slow-motion mode (useful when debugging selectors).
   - `pages` (string, default `all`) — comma-separated list of pages to record (e.g. `dashboard,backtest`).
7. Click the green **Run workflow** button.

The job typically takes 8-12 minutes: 2 min for dependency install, 1 min for stack boot, 4-7 min for the Playwright walk, 1 min for artifact upload.

---

## 3. Downloading the Artifacts

Once the workflow finishes (green checkmark or red X — artifacts upload either way for diagnostics):

1. Click the workflow run in the Actions tab.
2. Scroll to the **Artifacts** section at the bottom.
3. Two artifacts are produced per run:
   - `e2e-videos-<run-id>.zip` — one `.webm` per page visited (Playwright trace + video).
   - `e2e-screenshots-<run-id>.zip` — one `.png` per page at the final settled state.
4. Click an artifact name to download. Extract locally; the directory structure is `output/<page-name>/{video.webm, screenshot.png, console.log, network.har}`.

Artifacts are retained for 30 days by default. If you need longer retention, copy them out manually or change `retention-days` in the workflow.

---

## 4. Interpreting Results

The workflow exits 0 if every page rendered without a JavaScript exception and every API call returned a 2xx or expected 4xx. It exits nonzero if any page threw an uncaught error or a critical endpoint (e.g. `/api/account`) returned 5xx.

**Per-page diagnostics** are in the artifact directories. Look in this order:

1. **screenshot.png** — eyeball the page. If it shows the QuantEdge layout with data, the page works. If it shows a generic error boundary ("Something went wrong") or a blank white screen, the page failed.
2. **console.log** — captured browser console output. Filter for `[error]` lines. A handful of `[warn]` lines are normal (third-party tracker noise, React strict-mode double-render warnings).
3. **network.har** — every HTTP request the page made. Open it in https://toolbox.googleapps.com/apps/har_analyzer/ or Chrome DevTools → Network → Import HAR. Look for red rows (4xx/5xx). The most common failures:
   - `5xx on /api/quotes/*` — Alpaca rate limit hit; rerun with stagger.
   - `connection refused on :6379` — `REDIS_URL` secret missing or malformed.
   - `5xx on /api/orders` — `DATABASE_URL` pooler exhausted; reboot the Supabase project.
4. **video.webm** — useful when the page renders briefly then crashes. Skim at 2x speed.

**A common pattern**: dashboard, backtest, and live-strategies pages succeed (they only need Alpaca + DB), while the broker-specific pages (TradeStation, Binance, Polymarket) show empty states. That is the expected outcome when only the four required secrets are configured.

---

## 5. Automatic Trigger on Branch Push

The workflow also runs automatically on every push to `claude/advanced-trading-bot-d5Lmw`. That branch is the integration target for the autonomous coding agent — every push gets a fresh e2e recording so reviewers can see the change in action without booting locally.

To disable the auto-trigger temporarily, comment out the `push:` trigger block in `.github/workflows/e2e-demo.yml`. To redirect it to a different branch, edit the `branches:` list under that block. Pushes to `main` are intentionally NOT auto-triggered; release validation goes through a separate, slower workflow with live-trading guardrails.

---

## 6. Local Reproduction

If you need to debug a failure locally without spending GitHub Actions minutes:

```bash
# Boot the stack
./scripts/launch.sh dev

# Seed demo accounts/strategies (idempotent)
python scripts/e2e/seed_demo.py

# Run the recorder against the local stack
node scripts/e2e/demo-recorder.js --base-url http://localhost:5173 --output ./scripts/e2e/output
```

The output directory will mirror the structure of the CI artifact, so you can use the same interpretation steps from section 4.

---

## 7. Extending the Demo

When you add a new dashboard page, add it to the `PAGES` array near the top of `demo-recorder.js`. Each entry needs:

- `name` — slug used for the artifact subdirectory.
- `path` — URL path relative to base URL.
- `waitFor` — selector to wait for before screenshotting (proves the page hydrated).
- `optionalSecrets` — array of env var names whose absence should downgrade failure to "empty state" instead of "error".

Open a PR and let the workflow validate the new page. If it goes green on `claude/advanced-trading-bot-d5Lmw`, it is ready to merge.
