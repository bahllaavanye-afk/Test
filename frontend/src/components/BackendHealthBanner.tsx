import { useEffect, useState } from 'react'

/**
 * Global, honest status bar. When the API is unreachable (network error on any
 * request — e.g. the backend is asleep, redeploying, or its host is down) the
 * pages would otherwise render as mysteriously empty boxes. This shows a single
 * clear message instead, and disappears automatically the moment a request
 * succeeds again. Driven by `backendDown`/`backendUp` events dispatched from the
 * Axios response interceptor (src/api/client.ts), so it reflects real traffic.
 */
export function BackendHealthBanner() {
  const [down, setDown] = useState(false)

  useEffect(() => {
    const onDown = () => setDown(true)
    const onUp = () => setDown(false)
    window.addEventListener('backendDown', onDown)
    window.addEventListener('backendUp', onUp)
    return () => {
      window.removeEventListener('backendDown', onDown)
      window.removeEventListener('backendUp', onUp)
    }
  }, [])

  if (!down) return null

  return (
    <div
      role="alert"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
        background: '#ff1744',
        color: '#fff',
        textAlign: 'center',
        padding: '6px 12px',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: 12,
        letterSpacing: 0.2,
      }}
    >
      ⚠ Backend unreachable — the API is starting up or offline. Data loads automatically once it's back.
    </div>
  )
}

export default BackendHealthBanner
