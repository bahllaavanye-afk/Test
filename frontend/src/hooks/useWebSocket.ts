import { useEffect, useRef, useState, useCallback } from 'react'

export type WSMessage = { type: string; [key: string]: unknown }

export function useWebSocket(path: string, enabled = true) {
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null)
  const [connected, setConnected] = useState(false)
  const ws = useRef<WebSocket | null>(null)
  const reconnectDelay = useRef(1000)
  // Track whether the hook is still mounted to avoid state updates after unmount
  const mounted = useRef(true)

  const connect = useCallback(() => {
    if (!enabled || !mounted.current) return
    const url = `${import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}${path}`
    ws.current = new WebSocket(url)

    ws.current.onopen = () => {
      if (!mounted.current) return
      setConnected(true)
      reconnectDelay.current = 1000
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
      const delay = Math.min(reconnectDelay.current, 30000)
      setTimeout(connect, delay)
      reconnectDelay.current *= 2
    }

    ws.current.onerror = () => {
      // Let onclose handle reconnect
      ws.current?.close()
    }
  }, [path, enabled])

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
