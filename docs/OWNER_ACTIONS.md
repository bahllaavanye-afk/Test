# Owner Actions — the only things QuantEdge needs from a human

_Everything else (brain, tests, desks, strategy gate, Slack, deploys) runs without
you. These items require logging into an account only you control — an agent
cannot mint a credential or pay a bill. Ranked by impact. Last verified 2026-06-28._

> Secrets live in **Doppler** (single source) and, for GitHub Actions workflows,
> in **GitHub → Settings → Secrets and variables → Actions**. Some keys are needed
> in both. Never paste secret values into chat or commits.

---

## 🔴 1. Add `ALPACA_API_KEY` (the key *ID*) — biggest unlock
**Symptom it fixes:** no live market data, no paper trades, and the empty
dashboard panels (positions / orders / P&L).

**Why:** Doppler currently has `ALPACA_SECRET_KEY` but **not** the key ID. Alpaca
auth needs *both* → every call returns 401 today.

**Steps:**
1. Go to https://alpaca.markets → log in → **Paper Trading** account.
2. **Generate API Keys** → copy the **API Key ID** (looks like `PK…`) and the secret.
3. Add the **Key ID** as `ALPACA_API_KEY` in:
   - **Doppler** (`doppler secrets set ALPACA_API_KEY` or the dashboard), and
   - **GitHub → Settings → Secrets → Actions** (the desk workflows read
     `${{ secrets.ALPACA_API_KEY }}`).
4. Confirm `ALPACA_SECRET_KEY` matches the same key pair.

**Unlocks:** real quotes, 24/7 crypto paper trading, and populated dashboard data.

---

## 🟠 2. Bring the backend online (Render)
**Symptom it fixes:** the other half of the "empty website" — any feature that
calls the backend API.

**Why:** the free Render build-minutes are exhausted, so no new image deploys and
the API is down. The frontend (Vercel) is fine; it just has nothing to call.

**Options (pick one):**
- Wait for the monthly build-minute reset, **or**
- Upgrade the Render plan, **or**
- Create a fresh free Render account and point the service at it.

Until then, the static **`/live`** page is the no-backend fallback.

---

## 🟡 3. Optional — more LLM headroom (NOT blocking; 3 providers already live)
The brain runs on cerebras + groq + nvidia today. These only add redundancy:
- **DeepSeek** → 402 "insufficient balance": add credit, or leave it.
- **Gemini** → 429 quota: recovers on its own.
- **`ANTHROPIC_API_KEY`** → add to Doppler to **activate the Claude backstop**
  (the "never goes dark" last resort). Inert until the key exists.

---

## 🔵 4. Security hygiene
Rotate the **Render API key** and **deploy-hook** that were pasted into chat in an
earlier session — treat them as compromised.

---

## What's already handled (no action needed)
- LLM cascade revived (Cloudflare-1010 UA fix + current model IDs) — 0→3 live providers.
- Claude backstop wired (activates when key #3 is added).
- Risk-gate direct tests (86%); full suite 930+ green; flaky live test bounded.
- 24/7 crypto desk schedule; multi-criteria promotion gate (Sharpe + Sortino +
  Calmar + max-DD + win-rate + profit-factor + min-trades + Deflated Sharpe).
- Slack token verified working; channel-activity swept.

**TL;DR:** do **#1** for data + trading + a populated dashboard; do **#2** for the
full backend site. #3/#4 are optional/hygiene.
