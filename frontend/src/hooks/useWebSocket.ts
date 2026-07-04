import { useEffect, useRef, useState, useCallback } from 'react'

export type WSMessage = { type: string; [key: string]: unknown }

const BASE_DELAY_MS = 1000
const MAX_DELAY_MS = 30000

export function useWebSocket(path: string, enabled = true, shouldReconnect = true) {
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null)
  const [connected, setConnected] = useState(false)
  const ws = useRef<WebSocket | null>(null)
  const reconnectDelay = useRef(BASE_DELAY_MS)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Track whether the hook is still mounted to avoid state updates after unmount
  const mounted = useRef(true)

  const connect = useCallback(() => {
    if (!enabled || !mounted.current) return
    // Don't stack sockets: a live or in-flight connection means nothing to do.
    const state = ws.current?.readyState
    if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return
    // Prod WS base falls back to the live Render service because Vercel rewrites
    // cannot proxy WebSocket upgrades (unlike the /api HTTP rewrite). VITE_WS_URL
    // overrides; localhost only in dev.
    const isLocal = typeof location !== 'undefined' && /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)
    const wsBase = import.meta.env.VITE_WS_URL || (isLocal ? 'ws://localhost:8000' : 'wss://quantedge-api-agb8.onrender.com')
    const url = `${wsBase}${path}`
    ws.current = new WebSocket(url)

    ws.current.onopen = () => {
      if (!mounted.current) return
      setConnected(true)
      // Reset backoff on successful connection
      reconnectDelay.current = BASE_DELAY_MS
    }

    ws.current.onmessage = (e) => {
      if (!mounted.current) return
      try {
        setLastMessage(JSON.parse(e.data as string))
      } catch {
        // Ignore non-JSON frames
      }
    }

    ws.current.onclose = () => {
      if (!mounted.current) return
      setConnected(false)
      if (!shouldReconnect) return
      // The backend sleeps on Render's free tier (~15 min idle), so the socket
      // drops on every sleep cycle. Never give up permanently — keep retrying on
      // capped backoff, but only while the tab is visible; hidden tabs resume via
      // the visibilitychange/backendUp listeners below instead of burning retries.
      if (typeof document !== 'undefined' && document.hidden) return
      const delay = Math.min(reconnectDelay.current, MAX_DELAY_MS)
      reconnectTimer.current = setTimeout(connect, delay)
      reconnectDelay.current = Math.min(reconnectDelay.current * 2, MAX_DELAY_MS)
    }

    ws.current.onerror = () => {
      // Let onclose handle reconnect
      ws.current?.close()
    }
  }, [path, enabled, shouldReconnect])

  useEffect(() => {
    mounted.current = true
    connect()
    // Revive the socket the moment the user comes back to the tab, or the axios
    // layer sees the backend answer again (it dispatches 'backendUp' on any
    // successful response) — instant recovery after a Render sleep cycle.
    const revive = () => {
      if (typeof document !== 'undefined' && document.hidden) return
      reconnectDelay.current = BASE_DELAY_MS
      connect()
    }
    document.addEventListener('visibilitychange', revive)
    window.addEventListener('backendUp', revive)
    return () => {
      mounted.current = false
      document.removeEventListener('visibilitychange', revive)
      window.removeEventListener('backendUp', revive)
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      ws.current?.close()
    }
  }, [connect])

  const send = useCallback((data: unknown) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data))
    }
  }, [])

  return { lastMessage, connected, send }
}
