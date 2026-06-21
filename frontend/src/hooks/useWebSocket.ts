import { useEffect, useRef, useState, useCallback } from 'react'
import { wsBase } from '../utils/endpoints'

export type WSMessage = { type: string; [key: string]: unknown }

const MAX_RECONNECT_ATTEMPTS = 10
const BASE_DELAY_MS = 1000
const MAX_DELAY_MS = 30000

export function useWebSocket(path: string, enabled = true, shouldReconnect = true) {
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null)
  const [connected, setConnected] = useState(false)
  const ws = useRef<WebSocket | null>(null)
  const reconnectDelay = useRef(BASE_DELAY_MS)
  const reconnectAttempts = useRef(0)
  // Track whether the hook is still mounted to avoid state updates after unmount
  const mounted = useRef(true)

  const connect = useCallback(() => {
    if (!enabled || !mounted.current) return
    const url = `${wsBase()}${path}`
    ws.current = new WebSocket(url)

    ws.current.onopen = () => {
      if (!mounted.current) return
      setConnected(true)
      // Reset backoff on successful connection
      reconnectDelay.current = BASE_DELAY_MS
      reconnectAttempts.current = 0
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
      if (
        shouldReconnect &&
        reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS
      ) {
        const delay = Math.min(reconnectDelay.current, MAX_DELAY_MS)
        setTimeout(connect, delay)
        reconnectDelay.current = Math.min(reconnectDelay.current * 2, MAX_DELAY_MS)
        reconnectAttempts.current += 1
      }
    }

    ws.current.onerror = () => {
      // Let onclose handle reconnect
      ws.current?.close()
    }
  }, [path, enabled, shouldReconnect])

  useEffect(() => {
    mounted.current = true
    connect()
    return () => {
      mounted.current = false
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
