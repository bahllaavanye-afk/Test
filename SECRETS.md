# Secrets — Single Source of Truth (Doppler)

All API keys/secrets live in **one place: Doppler**. Every platform (GitHub Actions,
Render, Claude Code, local dev) reads from Doppler instead of keeping its own copy. Set a
key once in Doppler and it propagates everywhere — no more "7 copies to maintain".

```
                ┌─────────────┐
                │   DOPPLER   │  ← the only place you edit secrets
                │ project:    │
                │  quantedge  │
                └──────┬──────┘
        ┌─────────────┼──────────────┬─────────────────┐
        ▼             ▼              ▼                 ▼
  GitHub Actions    Render      Claude Code         Local dev
  (sync → repo     (native     (doppler run in     (doppler run --
   secrets)         sync)       .mcp.json)          ./scripts/…)
```

---

## 1. One-time Doppler setup

1. **Create the project + configs** (Doppler dashboard → Projects → Create):
   - Project: `quantedge`
   - Configs: `prd` (production / Render), `dev` (local + Claude Code). You can start with
     just `prd` and add `dev` later.
2. **Add every secret once** (Doppler → quantedge → prd → Secrets). The full list the repo
   expects (see `backend/app/config.py` + `.github/workflows/*`):
   ```
   # Brokers / market data
   ALPACA_API_KEY  ALPACA_SECRET_KEY  ALPACA_BASE_URL
   BINANCE_API_KEY  BINANCE_SECRET
   # Infra
   DATABASE_URL  REDIS_URL  SECRET_KEY  SUPABASE_ACCESS_TOKEN
   # LLM / research (free tiers)
   GROQ_API_KEY  GEMINI_API_KEY  PERPLEXITY_API_KEY  DEEPSEEK_API_KEY
   # Slack
   SLACK_BOT_TOKEN  SLACK_TEAM_ID  SLACK_ADMIN_EMAIL  SLACK_ADMIN_USER_ID
   # Render ops
   RENDER_API_KEY  RENDER_SERVICE_ID  RENDER_WORKER_SERVICE_ID
   # Non-secret config (fine to keep here too)
   TRADING_MODE=paper  DEMO_MODE=true
   ```
   > Keep `TRADING_MODE=paper`. Never put live-trading keys in `prd` until paper-tested.

---

## 2. GitHub Actions

**Recommended — Doppler → GitHub sync (zero workflow changes).** All existing
`${{ secrets.X }}` references keep working, sourced from Doppler:

1. Doppler dashboard → quantedge → `prd` → **Integrations → GitHub Actions**.
2. Authorize the GitHub app, pick repo `bahllaavanye-afk/test`, sync.
   Doppler now pushes its secrets into the repo's Actions secrets and keeps them updated.

That's it — no YAML edits. (Alternative: fetch at runtime with
`dopplerHQ/secrets-fetch-action`, but the sync integration is simpler and needs no changes.)

---

## 3. Render

Doppler has a native Render integration that fills the service env vars `render.yaml`
declares as `sync: false`:

1. Doppler → quantedge → `prd` → **Integrations → Render**.
2. Select the QuantEdge web (and worker) service, map config `prd`, enable sync.

This **replaces** the manual `render-sync-secrets.yml` GitHub→Render hop (you can leave that
workflow in place or delete it later — Doppler is now the source).

---

## 4. Claude Code (this repo's MCP servers)

The MCP servers in `.mcp.json` are wrapped with `doppler run --`, so the session needs
**only one** secret — a Doppler **service token** — and Doppler injects the rest
(`SLACK_BOT_TOKEN`, `ALPACA_*`, etc.) into each server.

1. Doppler → quantedge → `dev` → **Access → Service Tokens → Generate** (read-only).
2. In your Claude Code environment config (code.claude.com → your environment):
   - **Environment variable:** `DOPPLER_TOKEN = dp.st.dev.xxxxx`
   - **Setup script:** install the Doppler CLI so `doppler run` exists in the container:
     ```bash
     curl -Ls https://cli.doppler.com/install.sh | sh
     ```
3. Start a fresh session. `.mcp.json` runs e.g.
   `doppler run -- npx -yq @modelcontextprotocol/server-slack`, which auths with
   `DOPPLER_TOKEN` and provides `SLACK_BOT_TOKEN` — the Slack/Alpaca/etc. MCP tools now work.

> If `DOPPLER_TOKEN` is absent or the CLI isn't installed, those MCP servers won't start
> (they also don't work today without keys) — so this is strictly an improvement once set up.
> The GitHub MCP server is configured outside this file and is unaffected.

---

## 5. Local development

```bash
curl -Ls https://cli.doppler.com/install.sh | sh   # once
doppler login                                       # once
doppler setup -p quantedge -c dev                   # in repo root, once
# then run anything with secrets injected:
doppler run -- ./scripts/launch.sh paper
doppler run -- python scripts/slack_message_monitor.py
# convenience wrapper:
./scripts/with-doppler.sh ./scripts/launch.sh dev
```

---

## Summary — where each platform reads from

| Platform       | Mechanism                                   | You configure                         |
|----------------|---------------------------------------------|---------------------------------------|
| GitHub Actions | Doppler → GitHub **sync** integration       | authorize once (dashboard)            |
| Render         | Doppler → Render **native** integration     | map service once (dashboard)          |
| Claude Code    | `doppler run` in `.mcp.json`                | `DOPPLER_TOKEN` + CLI in env setup    |
| Local dev      | `doppler run -- …`                          | `doppler login` + `doppler setup`     |

Edit a key in Doppler → it updates in all four. One copy.
