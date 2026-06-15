import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ────────────────────────────────────────────────────────────────────

interface Bot {
  id: string
  name: string
  display_name: string
  market_type: string
  strategy_type: string
  risk_bucket: string
  is_enabled: boolean
  symbols: string[]
  total_pnl: number
  return_pct: number
  today_pnl: number
  today_change_pct: number
  total_trades: number
  win_rate: number | null
  allocation: number
  risk_usd: number
  vol_scalar: number | null
  confidence_threshold: number
  tick_interval_seconds: number
}

interface Summary {
  total_pnl: number
  return_pct: number
  today_pnl: number
  today_change_pct: number
  total_risk: number
  total_allocation: number
  total_bots: number
  active_bots: number
  overall_win_rate: number | null
}

interface DashboardData {
  summary: Summary
  bots: Bot[]
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmt$ = (v: number) =>
  v >= 0
    ? `+$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
    : `-$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`

const fmtK = (v: number) => {
  const abs = Math.abs(v)
  if (abs >= 1000) return `$${(abs / 1000).toFixed(1)}K`
  return `$${abs.toFixed(0)}`
}

const pnlColor = (v: number) => (v >= 0 ? '#00c853' : '#ff1744')
const pctColor = (v: number) => (v >= 0 ? '#00c853' : '#ff1744')

const MARKET_ICONS: Record<string, string> = {
  equity: '📈',
  crypto: '₿',
  polymarket: '🎯',
}

const BUCKET_COLOR: Record<string, string> = {
  arbitrage: '#f5a623',
  directional: '#2196F3',
}

// ── Bot Avatar ────────────────────────────────────────────────────────────────

function BotAvatar({ bot }: { bot: Bot }) {
  const icon = MARKET_ICONS[bot.market_type] || '🤖'
  const color = BUCKET_COLOR[bot.risk_bucket] || '#888'
  return (
    <div
      style={{
        width: 36,
        height: 36,
        borderRadius: 8,
        background: `${color}22`,
        border: `1.5px solid ${color}55`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 16,
        flexShrink: 0,
      }}
    >
      {icon}
    </div>
  )
}

// ── Toggle ────────────────────────────────────────────────────────────────────

function Toggle({
  botId,
  enabled,
  onToggle,
}: {
  botId: string
  enabled: boolean
  onToggle: (id: string, val: boolean) => void
}) {
  return (
    <button
      onClick={() => onToggle(botId, !enabled)}
      style={{
        width: 44,
        height: 24,
        borderRadius: 12,
        border: 'none',
        background: enabled ? '#00c853' : '#333',
        cursor: 'pointer',
        position: 'relative',
        transition: 'background 0.2s',
        flexShrink: 0,
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 3,
          left: enabled ? 23 : 3,
          width: 18,
          height: 18,
          borderRadius: 9,
          background: '#fff',
          transition: 'left 0.2s',
        }}
      />
    </button>
  )
}

// ── Filter Bar ────────────────────────────────────────────────────────────────

type Filter = 'all' | 'on' | 'off' | 'equity' | 'crypto' | 'polymarket' | 'arbitrage' | 'directional'

function FilterTab({
  label,
  active,
  count,
  onClick,
}: {
  label: string
  active: boolean
  count?: number
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
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        transition: 'all 0.15s',
      }}
    >
      {label}
      {count !== undefined && (
        <span
          style={{
            background: active ? '#f5a62344' : '#222',
            borderRadius: 4,
            padding: '0 5px',
            fontSize: 10,
            color: active ? '#f5a623' : '#555',
          }}
        >
          {count}
        </span>
      )}
    </button>
  )
}

// ── Summary Bar ───────────────────────────────────────────────────────────────

function SummaryBar({ summary }: { summary: Summary }) {
  const items = [
    { label: 'TOTAL P/L', value: fmt$(summary.total_pnl), color: pnlColor(summary.total_pnl) },
    { label: 'RETURN %', value: `${summary.return_pct >= 0 ? '+' : ''}${summary.return_pct.toFixed(1)}%`, color: pctColor(summary.return_pct) },
    { label: 'TODAY', value: fmt$(summary.today_pnl), color: pnlColor(summary.today_pnl) },
    { label: 'CHANGE %', value: `${summary.today_change_pct >= 0 ? '+' : ''}${summary.today_change_pct.toFixed(2)}%`, color: pctColor(summary.today_change_pct) },
    { label: 'RISK', value: `$${summary.total_risk.toLocaleString('en-US', { maximumFractionDigits: 0 })}`, color: '#e8e8e8' },
    { label: 'ALLOCATION', value: `$${summary.total_allocation.toLocaleString('en-US', { maximumFractionDigits: 0 })}`, color: '#e8e8e8' },
    { label: 'WIN RATE', value: summary.overall_win_rate != null ? `${summary.overall_win_rate}%` : '—', color: '#e8e8e8' },
  ]
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${items.length}, 1fr)`,
        gap: 1,
        background: '#1e1e1e',
        border: '1px solid #1e1e1e',
        borderRadius: 10,
        overflow: 'hidden',
        marginBottom: 16,
      }}
    >
      {items.map((item) => (
        <div
          key={item.label}
          style={{
            background: '#111',
            padding: '14px 18px',
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          <div style={{ fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.05em' }}>
            {item.label}
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, color: item.color, fontFamily: 'JetBrains Mono, monospace' }}>
            {item.value}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Bot Row ───────────────────────────────────────────────────────────────────

function BotRow({
  bot,
  onToggle,
}: {
  bot: Bot
  onToggle: (id: string, val: boolean) => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <>
      <tr
        onClick={() => setExpanded((e) => !e)}
        style={{
          cursor: 'pointer',
          background: expanded ? '#141414' : 'transparent',
          transition: 'background 0.15s',
        }}
        onMouseEnter={(e) => { if (!expanded) (e.currentTarget as HTMLElement).style.background = '#131313' }}
        onMouseLeave={(e) => { if (!expanded) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
      >
        <td style={{ padding: '10px 14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <BotAvatar bot={bot} />
            <div>
              <div style={{ fontSize: 12, color: '#e8e8e8', fontWeight: 600 }}>{bot.display_name}</div>
              <div style={{ fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace' }}>
                {bot.symbols.slice(0, 3).join(', ')}{bot.symbols.length > 3 ? ` +${bot.symbols.length - 3}` : ''}
              </div>
            </div>
          </div>
        </td>
        <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: pnlColor(bot.total_pnl), fontWeight: 600 }}>
          {bot.total_pnl !== 0 ? fmt$(bot.total_pnl) : '—'}
        </td>
        <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: pctColor(bot.return_pct) }}>
          {bot.total_pnl !== 0 ? `${bot.return_pct >= 0 ? '+' : ''}${bot.return_pct.toFixed(1)}%` : '—'}
        </td>
        <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: pnlColor(bot.today_pnl) }}>
          {bot.today_pnl !== 0 ? fmt$(bot.today_pnl) : '—'}
        </td>
        <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#888' }}>
          {bot.risk_usd > 0 ? fmtK(bot.risk_usd) : '—'}
        </td>
        <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#888' }}>
          {fmtK(bot.allocation)}
        </td>
        <td style={{ padding: '10px 14px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#e8e8e8', fontWeight: 600 }}>
          {bot.win_rate != null ? `${bot.win_rate}%` : '—'}
        </td>
        <td style={{ padding: '10px 14px' }} onClick={(e) => e.stopPropagation()}>
          <Toggle botId={bot.id} enabled={bot.is_enabled} onToggle={onToggle} />
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={8} style={{ padding: 0 }}>
            <div
              style={{
                padding: '12px 64px 16px',
                background: '#0d0d0d',
                borderTop: '1px solid #1e1e1e',
                display: 'grid',
                gridTemplateColumns: 'repeat(5, 1fr)',
                gap: 16,
              }}
            >
              {[
                ['Trades', bot.total_trades.toString()],
                ['Market', bot.market_type],
                ['Type', bot.strategy_type],
                ['Bucket', bot.risk_bucket],
                ['Confidence', `${(bot.confidence_threshold * 100).toFixed(0)}%`],
                ['Tick Interval', `${bot.tick_interval_seconds}s`],
                ['Vol Scalar', bot.vol_scalar != null ? bot.vol_scalar.toFixed(3) : '—'],
                ['Today Change', bot.today_pnl !== 0 ? `${bot.today_change_pct >= 0 ? '+' : ''}${bot.today_change_pct.toFixed(3)}%` : '—'],
                ['Symbols', bot.symbols.join(', ') || '—'],
                ['ID', bot.id.slice(0, 8) + '…'],
              ].map(([label, value]) => (
                <div key={label}>
                  <div style={{ fontSize: 10, color: '#555', letterSpacing: '0.05em', marginBottom: 2 }}>{label}</div>
                  <div style={{ fontSize: 12, color: '#aaa', fontFamily: 'JetBrains Mono, monospace', wordBreak: 'break-all' }}>{value}</div>
                </div>
              ))}
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function BotDashboard() {
  const queryClient = useQueryClient()
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')

  const { data, isLoading, error } = useQuery<DashboardData>({
    queryKey: ['bot-dashboard'],
    queryFn: () => api.get('/strategies/dashboard').then((r) => r.data),
    refetchInterval: 15_000,
  })

  const toggleMut = useMutation({
    mutationFn: ({ id, val }: { id: string; val: boolean }) =>
      api.patch(`/strategies/${id}/toggle`, { is_enabled: val }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bot-dashboard'] }),
  })

  const handleToggle = (id: string, val: boolean) => toggleMut.mutate({ id, val })

  const bots = data?.bots ?? []

  const filtered = useMemo(() => {
    let b = bots
    if (filter === 'on') b = b.filter((x) => x.is_enabled)
    else if (filter === 'off') b = b.filter((x) => !x.is_enabled)
    else if (filter === 'equity') b = b.filter((x) => x.market_type === 'equity')
    else if (filter === 'crypto') b = b.filter((x) => x.market_type === 'crypto')
    else if (filter === 'polymarket') b = b.filter((x) => x.market_type === 'polymarket')
    else if (filter === 'arbitrage') b = b.filter((x) => x.risk_bucket === 'arbitrage')
    else if (filter === 'directional') b = b.filter((x) => x.risk_bucket === 'directional')
    if (search) {
      const q = search.toLowerCase()
      b = b.filter((x) => x.display_name.toLowerCase().includes(q) || x.name.toLowerCase().includes(q) || x.symbols.some((s) => s.toLowerCase().includes(q)))
    }
    return b
  }, [bots, filter, search])

  const counts = useMemo(
    () => ({
      all: bots.length,
      on: bots.filter((b) => b.is_enabled).length,
      off: bots.filter((b) => !b.is_enabled).length,
      equity: bots.filter((b) => b.market_type === 'equity').length,
      crypto: bots.filter((b) => b.market_type === 'crypto').length,
      polymarket: bots.filter((b) => b.market_type === 'polymarket').length,
      arbitrage: bots.filter((b) => b.risk_bucket === 'arbitrage').length,
      directional: bots.filter((b) => b.risk_bucket === 'directional').length,
    }),
    [bots]
  )

  if (error) {
    return (
      <div style={{ padding: 32, color: '#ff1744' }}>
        Failed to load bot dashboard: {(error as Error).message}
      </div>
    )
  }

  return (
    <div style={{ padding: '24px 32px', minHeight: '100vh', background: '#0a0a0a' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#e8e8e8', margin: 0 }}>Bots</h1>
          <p style={{ fontSize: 12, color: '#555', margin: '4px 0 0', fontFamily: 'JetBrains Mono, monospace' }}>
            {data?.summary.active_bots ?? 0} active · {data?.summary.total_bots ?? 0} total
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search bots…"
            style={{
              background: '#111',
              border: '1px solid #2a2a2a',
              borderRadius: 8,
              padding: '7px 12px',
              color: '#e8e8e8',
              fontSize: 12,
              fontFamily: 'JetBrains Mono, monospace',
              width: 200,
              outline: 'none',
            }}
          />
        </div>
      </div>

      {/* Summary Bar */}
      {data && <SummaryBar summary={data.summary} />}

      {/* Filters */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
        {(
          [
            ['all', 'All'],
            ['on', 'On'],
            ['off', 'Off'],
            ['equity', 'Equity'],
            ['crypto', 'Crypto'],
            ['polymarket', 'Polymarket'],
            ['arbitrage', 'Arbitrage'],
            ['directional', 'Directional'],
          ] as [Filter, string][]
        ).map(([key, label]) => (
          <FilterTab
            key={key}
            label={label}
            active={filter === key}
            count={counts[key]}
            onClick={() => setFilter(key)}
          />
        ))}
      </div>

      {/* Bot Table */}
      <div style={{ border: '1px solid #1e1e1e', borderRadius: 10, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#0d0d0d', borderBottom: '1px solid #1e1e1e' }}>
              {['BOT', 'TOTAL P/L', 'RETURN %', 'TODAY', 'RISK', 'ALLOCATION', 'WIN RATE', ''].map((h) => (
                <th
                  key={h}
                  style={{
                    padding: '10px 14px',
                    textAlign: 'left',
                    fontSize: 10,
                    color: '#555',
                    fontWeight: 600,
                    fontFamily: 'JetBrains Mono, monospace',
                    letterSpacing: '0.08em',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #1a1a1a' }}>
                  {Array.from({ length: 8 }).map((_, j) => (
                    <td key={j} style={{ padding: '14px' }}>
                      <div
                        style={{
                          height: 14,
                          background: '#1a1a1a',
                          borderRadius: 4,
                          width: j === 0 ? 160 : 60,
                          animation: 'pulse 1.5s infinite',
                        }}
                      />
                    </td>
                  ))}
                </tr>
              ))
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ padding: '48px', textAlign: 'center', color: '#555', fontSize: 13 }}>
                  {search ? `No bots matching "${search}"` : 'No bots found. Enable strategies to see them here.'}
                </td>
              </tr>
            ) : (
              filtered.map((bot) => (
                <tr key={bot.id} style={{ borderBottom: '1px solid #1a1a1a' }}>
                  <td colSpan={8} style={{ padding: 0 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <tbody>
                        <BotRow bot={bot} onToggle={handleToggle} />
                      </tbody>
                    </table>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 0.4 } 50% { opacity: 0.7 } }
      `}</style>
    </div>
  )
}
