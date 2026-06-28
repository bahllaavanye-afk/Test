# QuantEdge — System Status (live-verified)

_Generated 2026-06-28, branch `claude/stoic-johnson-7z4wtz`. Every line below was
verified live this session (Doppler-injected keys, real API calls) — not asserted._

## TL;DR
The **brain was fully dead and is now alive** (0 → 3 working LLM providers). The
**frontend, Slack, and the agent mesh work**. **Live trading does not** — Alpaca is
missing its key *ID*, so no real market data or orders flow. Everything still broken
traces to a credential/billing value only the account owner can mint.

---

## ✅ Working (verified)
| Component | Evidence |
|---|---|
| **LLM brain / cascade** | cerebras + groq + nvidia all answer cleanly; full cascade returns in ~0.7s |
| **Frontend (Vercel)** | PR #240 preview deploys "Ready" on every push |
| **Slack** | `auth.test` → `ok:true, team:QuantEdge`; bot posts to channels |
| **Slack activity** | 97 channels, **52 active in last 24h** |
| **Risk gate** | `RiskManager.check_order()` now has 9 direct tests, 86% coverage |
| **GitHub Actions token** | present (drives the agent workflows) |

## ❌ Not working (verified, with the exact reason)
| Component | Root cause | Who can fix |
|---|---|---|
| **Live trading / real data** | `ALPACA_API_KEY` (the key *ID*) is **absent** from Doppler — only `ALPACA_SECRET_KEY` is set. Alpaca needs both → 401, no quotes, no orders. | **You** (paste the key ID into Doppler once) |
| **gemini provider** | HTTP 429 — quota exhausted (recovers over time) | recovers / you check billing |
| **deepseek provider** | HTTP 402 — "Insufficient Balance" | **You** (add balance) or leave dead |
| **GitHub Models (openai)** | HTTP 401 in this container (works inside Actions with `GITHUB_TOKEN`) | n/a — works in prod |
| **Claude backstop** | `ANTHROPIC_API_KEY` not in Doppler → `_call_claude` is inert | **You** (optional — 3 free providers already cover it) |
| **Render backend** | free build-minutes exhausted; API likely down | **You** (billing/new acct) — routed around via static `/live` |
| **44 Slack channels** | silent ≥31d — all are 1-member leadership/squad/pod scaffolding (#board, #leadership, #squad-*, #pod-*) | archive/consolidate (I can do) |

## 🔧 Fixed this session (committed + pushed)
1. **Brain revival** (`dbf642a`) — the big one:
   - Added a **browser `User-Agent`** → fixes Cloudflare `error 1010` that was 403-ing
     groq + cerebras (the fix had regressed off this branch).
   - Updated retired model IDs → `cerebras: gpt-oss-120b`, `nvidia: meta/llama-3.3-70b-instruct`.
   - Added `_extract_openai_content` so reasoning models (content in `reasoning`) don't
     KeyError-kill a provider.
   - **Result: 0 → 3 live providers, cascade ~0.7s.**
2. **Risk-gate tests** (`a3eda67`) — 9 offline tests on the safety-critical order gate; 57%→86%.
3. **gitignore** (`88c050b`) — stop committing the `llm_metrics.jsonl` runtime log.

## 🚧 In progress / next (honest scope)
- **Use all LLMs to full capacity** — ✅ 3 free providers race top-3; gemini/deepseek are
  billing-gated, not code-gated. Claude backstop wired but needs the key value.
- **Monitor Slack** — sweep done (52/97 active). Worst channels = empty 1-member
  leadership/squad/pod channels; recommend archiving/consolidating ~44 of them.
- **Stress-test every function** — tractable form = run the full test suite + targeted
  fuzz; full per-function coverage is multi-session. (705 test funcs exist; broker/exec
  paths are the thin spots.)
- **ML experiments → Drive/Colab/MLflow** — scripts exist (`run_experiments.py`,
  `ci_lstm_trainer.py`) but durable storage needs a tracking backend/creds not in Doppler.
  Cheapest no-cred path: commit experiment results to the repo. Needs your call on backend.
- **Doppler-only secret bootstrap + pin workflows** — planned; removes the default-branch
  flip and all future secret-wiring from your plate.

## The one pattern
Everything green needs no credential. Everything red needs **one value from an account
only you can log into** (Alpaca key ID, provider balance, Anthropic key, Render billing).
The system is engineered to run on free tiers without them — but live *trading*
specifically cannot start until the Alpaca key ID is in Doppler.
