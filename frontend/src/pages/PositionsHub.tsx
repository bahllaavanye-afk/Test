/**
 * PositionsHub — Option Alpha-style positions management.
 * Tabs: Open Positions | Closed Positions | Trade Log | Analyze
 */
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Plus, Filter, TrendingUp, TrendingDown } from 'lucide-react'
import api from '../api/client'
import PositionDetailPanel from '../components/trading/PositionDetailPanel'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Position {
  id: string | null
  symbol: string
  quantity: number
  avg_cost: number
  current_price: number | null
  unrealized_pnl: number | null
  side: string
}

interface Trade {
  id: string
  symbol: string
  side: string
  realized_pnl: number | null
  entry_price: number | null
  exit_price: number | null
  quantity: number
  opened_at: string | null
  closed_at: string | null
  strategy_name: string | null
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const fmt$ = (v: number, always_sign = false) => {
  const sign = v >= 0 ? (always_sign ? '+' : '') : '-'
  return `${sign}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`
}
const fmtPrice = (v: number) => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`
const pnlColor = (v: number | null) => (v == null ? '#555' : v >= 0 ? '#00c853' : '#ff1744')
const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`

function daysSince(dateStr: string | null): number | null {
  if (!dateStr) return null
  const ms = Date.now() - new Date(dateStr).getTime()
  return Math.floor(ms / 86400000)
}

// ── Summary Bar ───────────────────────────────────────────────────────────────

function SummaryBar({ items }: { items: { label: string; value: string; color?: string }[] }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 1,
        background: '#1e1e1e',
        borderRadius: 10,
        overflow: 'hidden',
        marginBottom: 16,
        border: '1px solid #1e1e1e',
      }}
    >
      {items.map((item) => (
        <div key={item.label} style={{ flex: 1, background: '#111', padding: '14px 18px' }}>
          <div style={{ fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', marginBottom: 4 }}>
            {item.label}
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, color: item.color || '#e8e8e8', fontFamily: 'JetBrains Mono, monospace' }}>
            {item.value}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Filter Chip ───────────────────────────────────────────────────────────────

function Chip({
  label,
  active,
  onClick,
}: {
  label: string
  active?: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '5px 14px',
        borderRadius: 6,
        border: active ? '1.5px solid #f5a623' : '1px solid #2a2a2a',
        background: active ? '#f5a62322' : '#111',
        color: active ? '#f5a623' : '#888',
        fontSize: 12,
        fontWeight: active ? 600 : 400,
        cursor: 'pointer',
        fontFamily: 'JetBrains Mono, monospace',
        transition: 'all 0.15s',
      }}
    >
      {label}
    </button>
  )
}

// ── Open Positions Tab ────────────────────────────────────────────────────────

function OpenPositionsTab() {
  const [sideFilter, setSideFilter] = useState<'all' | 'long' | 'short'>('all')
  const [search, setSearch] = useState('')
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null)

  const { data: positions = [], isLoading, error } = useQuery<Position[]>({
    queryKey: ['positions'],
    queryFn: () => api.get('/positions/').then((r) => r.data),
    refetchInterval: 15_000,
  })

  const filtered = useMemo(() => {
    let p = positions
    if (sideFilter !== 'all') p = p.filter((x) => x.side === sideFilter)
    if (search) p = p.filter((x) => x.symbol.toLowerCase().includes(search.toLowerCase()))
    return p
  }, [positions, sideFilter, search])

  const totalUnrealised = positions.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0)
  const totalValue = positions.reduce((s, p) => s + Math.abs(p.quantity) * (p.current_price ?? p.avg_cost), 0)
  const ror = totalValue > 0 ? (totalUnrealised / totalValue) * 100 : 0

  const summaryItems = [
    { label: 'TOTAL P/L', value: totalUnrealised !== 0 ? fmt$(totalUnrealised, true) : '—', color: pnlColor(totalUnrealised) },
    { label: 'RETURN ON RISK', value: ror !== 0 ? fmtPct(ror) : '—', color: pnlColor(ror) },
    { label: 'MARKET VALUE', value: totalValue > 0 ? fmt$(totalValue) : '—' },
    { label: 'POSITIONS', value: positions.length.toString() },
    { label: 'LONG', value: positions.filter((p) => p.side === 'long').length.toString(), color: '#00c853' },
    { label: 'SHORT', value: positions.filter((p) => p.side === 'short').length.toString(), color: '#ff1744' },
  ]

  if (error) return <div style={{ color: '#ff1744', padding: 24 }}>Error loading positions: {(error as Error).message}</div>

  return (
    <div>
      {/* Summary */}
      <SummaryBar items={summaryItems} />

      {/* Filter row */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 14, flexWrap: 'wrap' }}>
        <Chip label="All" active={sideFilter === 'all'} onClick={() => setSideFilter('all')} />
        <Chip label="Long" active={sideFilter === 'long'} onClick={() => setSideFilter('long')} />
        <Chip label="Short" active={sideFilter === 'short'} onClick={() => setSideFilter('short')} />
        <div style={{ flex: 1 }} />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Symbol…"
          style={{
            background: '#111', border: '1px solid #2a2a2a', borderRadius: 7,
            padding: '6px 12px', color: '#e8e8e8', fontSize: 12,
            fontFamily: 'JetBrains Mono, monospace', width: 140, outline: 'none',
          }}
        />
      </div>

      {/* Position Detail Panel */}
      {selectedPosition && (
        <PositionDetailPanel
          position={selectedPosition}
          onClose={() => setSelectedPosition(null)}
        />
      )}

      {/* Table */}
      <div style={{ border: '1px solid #1e1e1e', borderRadius: 10, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#0d0d0d', borderBottom: '1px solid #1e1e1e' }}>
              {['SYMBOL', 'SIDE', 'QTY', 'AVG COST', 'LAST', 'NET LIQ', 'P/L', 'ROR', 'DIT'].map((h) => (
                <th key={h} style={{ padding: '10px 14px', textAlign: 'left', fontSize: 10, color: '#555', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.08em' }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #1a1a1a' }}>
                    {Array.from({ length: 9 }).map((_, j) => (
                      <td key={j} style={{ padding: 14 }}>
                        <div style={{ height: 13, background: '#1a1a1a', borderRadius: 4, width: j === 0 ? 80 : 50, animation: 'pulse 1.5s infinite' }} />
                      </td>
                    ))}
                  </tr>
                ))
              : filtered.length === 0
              ? (
                <tr>
                  <td colSpan={9} style={{ padding: 48, textAlign: 'center', color: '#555', fontSize: 13 }}>
                    No open positions
                  </td>
                </tr>
              )
              : filtered.map((pos) => {
                  const marketValue = Math.abs(pos.quantity) * (pos.current_price ?? pos.avg_cost)
                  const rorPct = marketValue > 0 ? ((pos.unrealized_pnl ?? 0) / marketValue) * 100 : 0
                  return (
                    <tr
                      key={pos.id ?? pos.symbol}
                      style={{ borderBottom: '1px solid #1a1a1a', transition: 'background 0.15s', cursor: 'pointer' }}
                      onClick={() => setSelectedPosition(pos)}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#131313' }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent' }}
                    >
                      <td style={{ padding: '11px 14px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          {pos.side === 'long'
                            ? <TrendingUp size={14} color="#00c853" />
                            : <TrendingDown size={14} color="#ff1744" />}
                          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13, fontWeight: 600, color: '#e8e8e8' }}>{pos.symbol}</span>
                        </div>
                      </td>
                      <td style={{ padding: '11px 14px' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: pos.side === 'long' ? '#00c85322' : '#ff174422',
                          color: pos.side === 'long' ? '#00c853' : '#ff1744',
                          fontFamily: 'JetBrains Mono, monospace',
                        }}>{pos.side.toUpperCase()}</span>
                      </td>
                      <td style={{ padding: '11px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#e8e8e8' }}>{pos.quantity}</td>
                      <td style={{ padding: '11px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#888' }}>{fmtPrice(pos.avg_cost)}</td>
                      <td style={{ padding: '11px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#e8e8e8' }}>
                        {pos.current_price != null ? fmtPrice(pos.current_price) : '—'}
                      </td>
                      <td style={{ padding: '11px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#e8e8e8' }}>
                        {fmtPrice(marketValue)}
                      </td>
                      <td style={{ padding: '11px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, fontWeight: 600, color: pnlColor(pos.unrealized_pnl) }}>
                        {pos.unrealized_pnl != null ? fmt$(pos.unrealized_pnl, true) : '—'}
                      </td>
                      <td style={{ padding: '11px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: pnlColor(rorPct) }}>
                        {fmtPct(rorPct)}
                      </td>
                      <td style={{ padding: '11px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#888' }}>
                        —
                      </td>
                    </tr>
                  )
                })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Closed Positions / Trade Log ──────────────────────────────────────────────

function TradesTab({ mode }: { mode: 'closed' | 'log' }) {
  const [search, setSearch] = useState('')
  const [stratFilter, setStratFilter] = useState('all')

  const { data: trades = [], isLoading, error } = useQuery<Trade[]>({
    queryKey: ['trades', mode],
    queryFn: () => api.get('/trades/', { params: { limit: 200 } }).then((r) => r.data),
    refetchInterval: 30_000,
  })

  const strategies = useMemo(() => ['all', ...Array.from(new Set(trades.map((t) => t.strategy_name).filter(Boolean) as string[]))], [trades])

  const filtered = useMemo(() => {
    let t = trades
    if (stratFilter !== 'all') t = t.filter((x) => x.strategy_name === stratFilter)
    if (search) t = t.filter((x) => x.symbol.toLowerCase().includes(search.toLowerCase()))
    return t
  }, [trades, stratFilter, search])

  const totalPnl = filtered.reduce((s, t) => s + (t.realized_pnl ?? 0), 0)
  const wins = filtered.filter((t) => (t.realized_pnl ?? 0) > 0).length
  const winRate = filtered.length > 0 ? (wins / filtered.length * 100) : 0

  const summaryItems = [
    { label: 'TOTAL P/L', value: totalPnl !== 0 ? fmt$(totalPnl, true) : '—', color: pnlColor(totalPnl) },
    { label: 'TRADES', value: filtered.length.toString() },
    { label: 'WIN RATE', value: filtered.length > 0 ? `${winRate.toFixed(1)}%` : '—', color: winRate >= 50 ? '#00c853' : '#ff1744' },
    { label: 'WINS', value: wins.toString(), color: '#00c853' },
    { label: 'LOSSES', value: (filtered.length - wins).toString(), color: '#ff1744' },
    { label: 'AVG P/L', value: filtered.length > 0 ? fmt$(totalPnl / filtered.length, true) : '—', color: pnlColor(totalPnl) },
  ]

  if (error) return <div style={{ color: '#ff1744', padding: 24 }}>Error: {(error as Error).message}</div>

  return (
    <div>
      <SummaryBar items={summaryItems} />

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 14, flexWrap: 'wrap' }}>
        {strategies.slice(0, 6).map((s) => (
          <Chip key={s} label={s === 'all' ? 'All Bots' : s.replace(/_/g, ' ')} active={stratFilter === s} onClick={() => setStratFilter(s)} />
        ))}
        <div style={{ flex: 1 }} />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Symbol…"
          style={{
            background: '#111', border: '1px solid #2a2a2a', borderRadius: 7,
            padding: '6px 12px', color: '#e8e8e8', fontSize: 12,
            fontFamily: 'JetBrains Mono, monospace', width: 140, outline: 'none',
          }}
        />
      </div>

      <div style={{ border: '1px solid #1e1e1e', borderRadius: 10, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#0d0d0d', borderBottom: '1px solid #1e1e1e' }}>
              {['BOT', 'SYMBOL', 'SIDE', 'QTY', 'ENTRY', 'EXIT', 'P/L', 'ROR', 'DIT', 'CLOSED'].map((h) => (
                <th key={h} style={{ padding: '10px 14px', textAlign: 'left', fontSize: 10, color: '#555', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.08em' }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 8 }).map((_, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #1a1a1a' }}>
                    {Array.from({ length: 10 }).map((_, j) => (
                      <td key={j} style={{ padding: 12 }}>
                        <div style={{ height: 12, background: '#1a1a1a', borderRadius: 4, width: j === 0 ? 120 : 60, animation: 'pulse 1.5s infinite' }} />
                      </td>
                    ))}
                  </tr>
                ))
              : filtered.length === 0
              ? (
                <tr>
                  <td colSpan={10} style={{ padding: 48, textAlign: 'center', color: '#555', fontSize: 13 }}>
                    No closed trades yet
                  </td>
                </tr>
              )
              : filtered.map((trade) => {
                  const dit = daysSince(trade.opened_at)
                  const entry = trade.entry_price
                  const exit = trade.exit_price
                  const rorPct = entry && exit && entry > 0
                    ? ((exit - entry) / entry * 100 * (trade.side === 'short' ? -1 : 1))
                    : null
                  return (
                    <tr
                      key={trade.id}
                      style={{ borderBottom: '1px solid #1a1a1a', transition: 'background 0.15s' }}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#131313' }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent' }}
                    >
                      <td style={{ padding: '10px 14px', maxWidth: 140 }}>
                        <div style={{
                          fontSize: 11, color: '#888', fontFamily: 'JetBrains Mono, monospace',
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        }}>
                          {trade.strategy_name?.replace(/_/g, ' ') ?? '—'}
                        </div>
                      </td>
                      <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, fontWeight: 600, color: '#e8e8e8' }}>{trade.symbol}</td>
                      <td style={{ padding: '10px 14px' }}>
                        <span style={{
                          padding: '2px 7px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: trade.side === 'buy' ? '#00c85322' : '#ff174422',
                          color: trade.side === 'buy' ? '#00c853' : '#ff1744',
                          fontFamily: 'JetBrains Mono, monospace',
                        }}>{trade.side.toUpperCase()}</span>
                      </td>
                      <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#888' }}>{trade.quantity}</td>
                      <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#888' }}>
                        {entry != null ? fmtPrice(entry) : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#888' }}>
                        {exit != null ? fmtPrice(exit) : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, fontWeight: 600, color: pnlColor(trade.realized_pnl) }}>
                        {trade.realized_pnl != null ? fmt$(trade.realized_pnl, true) : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: pnlColor(rorPct) }}>
                        {rorPct != null ? fmtPct(rorPct) : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#888' }}>
                        {dit != null ? `${dit}d` : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#555' }}>
                        {trade.closed_at ? new Date(trade.closed_at).toLocaleDateString() : '—'}
                      </td>
                    </tr>
                  )
                })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Analyze Tab ───────────────────────────────────────────────────────────────

function AnalyzeTab() {
  const { data: trades = [], isLoading } = useQuery<Trade[]>({
    queryKey: ['trades', 'analyze'],
    queryFn: () => api.get('/trades/', { params: { limit: 500 } }).then((r) => r.data),
  })

  if (isLoading) return <div style={{ padding: 48, textAlign: 'center', color: '#555' }}>Loading analysis…</div>

  const totalPnl = trades.reduce((s, t) => s + (t.realized_pnl ?? 0), 0)
  const wins = trades.filter((t) => (t.realized_pnl ?? 0) > 0)
  const losses = trades.filter((t) => (t.realized_pnl ?? 0) < 0)
  const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + (t.realized_pnl ?? 0), 0) / wins.length : 0
  const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((s, t) => s + (t.realized_pnl ?? 0), 0) / losses.length) : 0
  const profitFactor = avgLoss > 0 ? (avgWin * wins.length) / (avgLoss * losses.length) : 0
  const winRate = trades.length > 0 ? wins.length / trades.length * 100 : 0

  // By strategy
  const byStrategy: Record<string, { pnl: number; count: number; wins: number }> = {}
  trades.forEach((t) => {
    const k = t.strategy_name ?? 'unknown'
    if (!byStrategy[k]) byStrategy[k] = { pnl: 0, count: 0, wins: 0 }
    byStrategy[k].pnl += t.realized_pnl ?? 0
    byStrategy[k].count++
    if ((t.realized_pnl ?? 0) > 0) byStrategy[k].wins++
  })
  const stratRows = Object.entries(byStrategy).sort((a, b) => b[1].pnl - a[1].pnl)

  // By symbol
  const bySymbol: Record<string, { pnl: number; count: number }> = {}
  trades.forEach((t) => {
    if (!bySymbol[t.symbol]) bySymbol[t.symbol] = { pnl: 0, count: 0 }
    bySymbol[t.symbol].pnl += t.realized_pnl ?? 0
    bySymbol[t.symbol].count++
  })
  const symbolRows = Object.entries(bySymbol).sort((a, b) => b[1].pnl - a[1].pnl).slice(0, 10)

  const metrics = [
    ['Total P/L', fmt$(totalPnl, true), pnlColor(totalPnl)],
    ['Total Trades', trades.length.toString(), '#e8e8e8'],
    ['Win Rate', `${winRate.toFixed(1)}%`, winRate >= 50 ? '#00c853' : '#ff1744'],
    ['Profit Factor', profitFactor.toFixed(2), profitFactor >= 1 ? '#00c853' : '#ff1744'],
    ['Avg Win', fmt$(avgWin, true), '#00c853'],
    ['Avg Loss', `-$${avgLoss.toFixed(0)}`, '#ff1744'],
    ['Expectancy', fmt$((avgWin * wins.length - avgLoss * losses.length) / Math.max(trades.length, 1), true), pnlColor(totalPnl)],
    ['Winners', wins.length.toString(), '#00c853'],
    ['Losers', losses.length.toString(), '#ff1744'],
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
      {/* Key metrics */}
      <div style={{ border: '1px solid #1e1e1e', borderRadius: 10, overflow: 'hidden' }}>
        <div style={{ padding: '14px 18px', background: '#0d0d0d', borderBottom: '1px solid #1e1e1e', fontSize: 12, color: '#888', fontWeight: 600 }}>KEY METRICS</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 0 }}>
          {metrics.map(([label, value, color]) => (
            <div key={label} style={{ padding: '16px 18px', borderBottom: '1px solid #1a1a1a', borderRight: '1px solid #1a1a1a' }}>
              <div style={{ fontSize: 10, color: '#555', letterSpacing: '0.06em', marginBottom: 4, fontFamily: 'JetBrains Mono, monospace' }}>{label}</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: color as string, fontFamily: 'JetBrains Mono, monospace' }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* By strategy */}
      <div style={{ border: '1px solid #1e1e1e', borderRadius: 10, overflow: 'hidden' }}>
        <div style={{ padding: '14px 18px', background: '#0d0d0d', borderBottom: '1px solid #1e1e1e', fontSize: 12, color: '#888', fontWeight: 600 }}>P/L BY STRATEGY</div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#0a0a0a' }}>
              {['STRATEGY', 'TRADES', 'WIN%', 'TOTAL P/L'].map((h) => (
                <th key={h} style={{ padding: '8px 14px', textAlign: 'left', fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stratRows.length === 0
              ? <tr><td colSpan={4} style={{ padding: 24, textAlign: 'center', color: '#555', fontSize: 12 }}>No trade data yet</td></tr>
              : stratRows.map(([name, s]) => (
                  <tr key={name} style={{ borderTop: '1px solid #1a1a1a' }}>
                    <td style={{ padding: '9px 14px', fontSize: 11, color: '#aaa', fontFamily: 'JetBrains Mono, monospace' }}>{name.replace(/_/g, ' ')}</td>
                    <td style={{ padding: '9px 14px', fontSize: 12, color: '#888', fontFamily: 'JetBrains Mono, monospace' }}>{s.count}</td>
                    <td style={{ padding: '9px 14px', fontSize: 12, fontFamily: 'JetBrains Mono, monospace', color: s.wins / s.count >= 0.5 ? '#00c853' : '#ff1744' }}>
                      {(s.wins / s.count * 100).toFixed(0)}%
                    </td>
                    <td style={{ padding: '9px 14px', fontSize: 12, fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', color: pnlColor(s.pnl) }}>
                      {fmt$(s.pnl, true)}
                    </td>
                  </tr>
                ))
            }
          </tbody>
        </table>
      </div>

      {/* Top symbols */}
      <div style={{ border: '1px solid #1e1e1e', borderRadius: 10, overflow: 'hidden', gridColumn: '1 / -1' }}>
        <div style={{ padding: '14px 18px', background: '#0d0d0d', borderBottom: '1px solid #1e1e1e', fontSize: 12, color: '#888', fontWeight: 600 }}>P/L BY SYMBOL (TOP 10)</div>
        <div style={{ display: 'flex', gap: 0, flexWrap: 'wrap' }}>
          {symbolRows.length === 0
            ? <div style={{ padding: 24, color: '#555', fontSize: 12 }}>No trade data yet</div>
            : symbolRows.map(([symbol, s]) => (
                <div key={symbol} style={{ flex: '1 1 120px', padding: '14px 18px', borderRight: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a' }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#e8e8e8', fontFamily: 'JetBrains Mono, monospace', marginBottom: 4 }}>{symbol}</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: pnlColor(s.pnl), fontFamily: 'JetBrains Mono, monospace' }}>{fmt$(s.pnl, true)}</div>
                  <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>{s.count} trades</div>
                </div>
              ))
          }
        </div>
      </div>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

type Tab = 'open' | 'closed' | 'log' | 'analyze'

export default function PositionsHub() {
  const [tab, setTab] = useState<Tab>('open')

  const tabs: { key: Tab; label: string }[] = [
    { key: 'open', label: 'Open Positions' },
    { key: 'closed', label: 'Closed Positions' },
    { key: 'log', label: 'Trade Log' },
    { key: 'analyze', label: 'Analyze' },
  ]

  return (
    <div style={{ padding: '24px 32px', minHeight: '100vh', background: '#0a0a0a' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 0, borderBottom: '2px solid #1e1e1e' }}>
          {tabs.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              style={{
                padding: '10px 20px',
                background: 'transparent',
                border: 'none',
                borderBottom: tab === key ? '2px solid #f5a623' : '2px solid transparent',
                marginBottom: -2,
                color: tab === key ? '#f5a623' : '#888',
                fontSize: 13,
                fontWeight: tab === key ? 600 : 400,
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        <button
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '8px 18px', borderRadius: 8,
            background: '#f5a623', border: 'none', color: '#000',
            fontSize: 13, fontWeight: 700, cursor: 'pointer',
          }}
        >
          <Plus size={14} />
          New Position
        </button>
      </div>

      {/* Tab Content */}
      <div>
        {tab === 'open' && <OpenPositionsTab />}
        {tab === 'closed' && <TradesTab mode="closed" />}
        {tab === 'log' && <TradesTab mode="log" />}
        {tab === 'analyze' && <AnalyzeTab />}
      </div>

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 0.4 } 50% { opacity: 0.7 } }
      `}</style>
    </div>
  )
}
