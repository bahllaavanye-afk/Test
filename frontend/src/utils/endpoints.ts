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

// The real Render service. NOTE: the bare `quantedge-api.onrender.com` host is an
// orphan stub — always use the `-6orc` service (or set VITE_WS_URL on the deploy).
const PROD_WS_BASE = 'wss://quantedge-api-6orc.onrender.com'

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
