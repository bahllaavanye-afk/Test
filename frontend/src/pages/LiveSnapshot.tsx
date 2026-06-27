import { useEffect, useState } from 'react'

/**
 * Backend-independent live view. Reads the static snapshot published every 30 min
 * by the snapshot-publish workflow (frontend/public/data/snapshot.json), so it
 * shows REAL data — Alpaca P&L/positions, strategy library, desk roster, brain
 * health — even when the Render backend is down. Public route: /live.
 */
type Snap = {
  generated_at: string
  repo: { strategy_count: number; model_count: number; strategies: string[] }
  desks: { name: string; symbols: string[]; count: number }[]
  brain: { healthy: boolean | null; working: string[] }
  trading: {
    available: boolean
    equity?: number
    day_pnl?: number
    day_pnl_pct?: number
    market_open?: boolean
    open_positions?: { symbol: string; qty: string; market_value: number; unrealized_pl: number }[]
    recent_orders?: { symbol: string; side: string; qty: string; status: string }[]
  }
}

const card: React.CSSProperties = {
  background: '#111111', border: '1px solid #1e1e1e', borderRadius: 8,
  padding: 16, margin: 8, minWidth: 200, flex: '1 1 240px',
}
const label: React.CSSProperties = { color: '#888', fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5 }
const big: React.CSSProperties = { color: '#e8e8e8', fontSize: 26, fontFamily: 'JetBrains Mono, monospace', marginTop: 4 }

export default function LiveSnapshot() {
  const [snap, setSnap] = useState<Snap | null>(null)
  const [err, setErr] = useState<string>('')

  useEffect(() => {
    const load = () =>
      fetch('/data/snapshot.json', { cache: 'no-store' })
        .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
        .then(setSnap)
        .catch((e) => setErr(String(e)))
    load()
    const id = setInterval(load, 60000)
    return () => clearInterval(id)
  }, [])

  if (err) return <div style={{ color: '#ff1744', padding: 24, fontFamily: 'monospace' }}>Snapshot not available yet ({err}). The publisher runs every 30 min.</div>
  if (!snap) return <div style={{ color: '#888', padding: 24, fontFamily: 'monospace' }}>Loading live snapshot…</div>

  const t = snap.trading
  const pnlColor = (t.day_pnl ?? 0) >= 0 ? '#00c853' : '#ff1744'

  return (
    <div style={{ background: '#0a0a0a', minHeight: '100vh', padding: 24, fontFamily: 'Inter, sans-serif' }}>
      <h1 style={{ color: '#f5a623', fontFamily: 'JetBrains Mono, monospace' }}>QuantEdge — Live Snapshot</h1>
      <div style={{ color: '#666', fontSize: 12, marginBottom: 16 }}>
        Updated {new Date(snap.generated_at).toLocaleString()} · static feed (works without the backend)
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap' }}>
        <div style={card}><div style={label}>Paper Equity</div>
          <div style={big}>{t.available ? `$${(t.equity ?? 0).toLocaleString()}` : 'n/a'}</div></div>
        <div style={card}><div style={label}>Day P&L</div>
          <div style={{ ...big, color: pnlColor }}>{t.available ? `${(t.day_pnl ?? 0) >= 0 ? '+' : ''}$${t.day_pnl} (${t.day_pnl_pct}%)` : 'n/a'}</div></div>
        <div style={card}><div style={label}>Market</div>
          <div style={big}>{t.market_open ? '🟢 OPEN' : '🔴 CLOSED'}</div></div>
        <div style={card}><div style={label}>Strategies</div><div style={big}>{snap.repo.strategy_count}</div></div>
        <div style={card}><div style={label}>ML Models</div><div style={big}>{snap.repo.model_count}</div></div>
        <div style={card}><div style={label}>Brain (free LLMs)</div>
          <div style={{ ...big, fontSize: 16 }}>{snap.brain.working.length ? snap.brain.working.join(', ') : '—'}</div></div>
      </div>

      <h2 style={{ color: '#e8e8e8', marginTop: 24 }}>Desks</h2>
      <div style={{ display: 'flex', flexWrap: 'wrap' }}>
        {snap.desks.map((d) => (
          <div key={d.name} style={card}>
            <div style={{ color: '#f5a623', fontWeight: 600 }}>{d.name}</div>
            <div style={{ color: '#888', fontSize: 12, marginTop: 6 }}>{d.symbols.join(', ')}</div>
          </div>
        ))}
      </div>

      {t.available && (t.open_positions?.length ?? 0) > 0 && (
        <>
          <h2 style={{ color: '#e8e8e8', marginTop: 24 }}>Open Positions ({t.open_positions!.length})</h2>
          <div style={{ display: 'flex', flexWrap: 'wrap' }}>
            {t.open_positions!.map((p) => (
              <div key={p.symbol} style={card}>
                <div style={{ color: '#e8e8e8', fontWeight: 600 }}>{p.symbol} ×{p.qty}</div>
                <div style={{ color: p.unrealized_pl >= 0 ? '#00c853' : '#ff1744', fontFamily: 'monospace' }}>
                  ${p.market_value.toLocaleString()} · {p.unrealized_pl >= 0 ? '+' : ''}${p.unrealized_pl}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
