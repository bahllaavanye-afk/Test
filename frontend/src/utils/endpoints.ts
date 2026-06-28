// Single source of truth for backend endpoints, so production builds never silently
// fall back to localhost or the orphan `quantedge-api.onrender.com` stub when a
// VITE_* env var is missing on the deploy.
//
//  - HTTP API: same-origin `/api/v1` flows through the Vite dev proxy and the Vercel
//    prod rewrite (see frontend/vercel.json) — no CORS, and it works on Vercel
//    preview domains too. VITE_API_URL (which must already include `/api/v1`) overrides.
//  - WebSocket: Vercel cannot proxy WebSockets, so in production we connect straight
//    to the backend service; local dev hits the uvicorn server. VITE_WS_URL overrides.

const isLocalHost =
  typeof location !== 'undefined' &&
  /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)

// The real Render service (new account, branch `main`). The old `-9jz0` service
// is dead (build-minutes exhausted). Override with VITE_WS_URL on the deploy.
const PROD_WS_BASE = 'wss://quantedge-api-agb8.onrender.com'

/** Base URL for HTTP API calls (already includes the `/api/v1` prefix). */
export function apiBase(): string {
  return import.meta.env.VITE_API_URL || '/api/v1'
}

/** Origin for WebSocket connections (no trailing path). */
export function wsBase(): string {
  const override = import.meta.env.VITE_WS_URL
  if (override) return override as string
  return isLocalHost ? 'ws://localhost:8000' : PROD_WS_BASE
}
