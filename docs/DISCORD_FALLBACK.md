# Discord Failover for Agent/Alert Messaging

Slack's free-plan message quota exhausted on 2026-06-29 (`message_limit_exceeded`)
and stayed dead for days — every employee post and every outage alert was lost.
The posting layer now fails over to a **Discord webhook** (free, generous limits)
whenever Slack can't deliver.

## Behavior
- **Slack healthy** → everything goes to Slack, exactly as before. Discord silent.
- **Slack fatally rejects** (quota exhausted, revoked token) → the message is
  delivered to Discord as `[#channel] username: text`, capped at
  `DISCORD_MAX_POSTS_PER_RUN` (default 20) per CI run.
- **No Slack configured at all** → Discord becomes the primary channel (backend).
- The agent-team run only pages "POSTING BROKEN" when **neither** Slack nor
  Discord delivered anything.

Wired at all three chokepoints: `.github/scripts/slack_agent_team.py` (employee
chatter), `.github/scripts/third_party_monitor.py` (outage alerts), and
`backend/app/notifications/slack.py` (in-app order/risk/system notifications).

## Setup (~3 minutes, free)
1. Create a Discord server (or reuse one) → create a channel, e.g. `#quantedge`.
2. Channel settings → **Integrations → Webhooks → New Webhook** → copy the URL.
3. Add it as `DISCORD_WEBHOOK_URL`:
   - GitHub → Settings → Secrets and variables → Actions (for the CI agents), and
   - Render → the backend service's environment (for in-app notifications).

No token scopes, no app review, no quota to manage. One webhook = one channel;
messages carry their intended Slack channel as a `[#channel]` prefix so a single
Discord channel works as a unified feed.
