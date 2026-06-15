/**
 * BotDetail — detailed Option Alpha-style bot view with 5 tabs.
 * Route: /bot-dashboard/:botId
 */
import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ToggleLeft, ToggleRight } from 'lucide-react'
import api from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

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
  total_trades: number
  win_rate: number | null
  allocation: number
  risk_usd: number
  vol_scalar: number | null
  confidence_threshold: number
  tick_interval_seconds: number
  params?: Record<string, unknown>
}

interface Tearsheet {
  total_pnl?: number
  win_rate?: number
  profit_factor?: number
  max_drawdown_pct?: number
  total_trades?: number
  avg_pnl?: number
  avg_win?: number
  avg_loss?: number
  sharpe?: number
  equity_curve?: Array<{ date: string; value: number }>
}

interface AgentLog {
  id: string
  action: string
  status: string
  input_summary: string | null
  output_summary: string | null
  created_at: string
  strategy_name: string | null
  symbol: string | null
  duration_ms: number | null
}

interface Position {
  id: string | null
  symbol: string
  quantity: number
  avg_cost: number
  current_price: number | null
  unrealized_pnl: number | null
  side: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const fmt$ = (v: number | null | undefined) => {
  if (v == null) return '—'
  const sign = v >= 0 ? '+' : '-'
  return `${sign}$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`
}

const fmtPct = (v: number | null | undefined) => {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`
}

const fmtNum = (v: number | null | undefined, dec = 2) =>
  v == null ? '—' : v.toFixed(dec)

const pnlColor = (v: number | null | undefined) =>
  v == null ? '#888' : v >= 0 ? '#00c853' : '#ff1744'

function fmtTs(ts: string) {
  const d = new Date(ts)
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
}

// ── Mini Sparkline (SVG) ──────────────────────────────────────────────────────

function Sparkline({ data }: { data: Array<{ date: string; value: number }> }) {
  if (!data || data.length < 2) {
    return <div className="h-32 flex items-center justify-center text-xs text-[#555]">Insufficient data</div>
  }
  const vals = data.map(d => d.value)
  const min = Math.min(...vals)
  const max = Math.max(...vals)
  const range = max - min || 1
  const W = 400
  const H = 120
  const pts = data.map((d, i) => {
    const x = (i / (data.length - 1)) * W
    const y = H - ((d.value - min) / range) * (H - 10) - 5
    return `${x},${y}`
  }).join(' ')
  const lastVal = vals[vals.length - 1]
  const firstVal = vals[0]
  const color = lastVal >= firstVal ? '#00c853' : '#ff1744'
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-32" preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
      <circle cx={(data.length - 1) / (data.length - 1) * W} cy={H - ((lastVal - min) / range) * (H - 10) - 5} r="3" fill={color} />
    </svg>
  )
}

// ── Stat Card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-3">
      <p className="text-[10px] text-[#555] uppercase tracking-wider mb-1">{label}</p>
      <p className="text-base font-bold font-mono" style={{ color: color ?? '#e8e8e8' }}>{value}</p>
    </div>
  )
}

// ── Tab Components ────────────────────────────────────────────────────────────

function DashboardTab({ bot, tearsheet }: { bot: Bot; tearsheet: Tearsheet | null }) {
  const equity = tearsheet?.equity_curve ?? []
  const wins = tearsheet?.total_trades != null && tearsheet?.win_rate != null
    ? Math.round(tearsheet.total_trades * (tearsheet.win_rate / 100))
    : null
  const losses = wins != null && tearsheet?.total_trades != null ? tearsheet.total_trades - wins : null

  return (
    <div className="space-y-6">
      {/* Equity Curve */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <p className="text-xs font-semibold text-[#888] uppercase tracking-wider mb-3">Equity Curve</p>
        {equity.length >= 2 ? (
          <Sparkline data={equity} />
        ) : (
          <div className="h-32 flex items-center justify-center text-xs text-[#555]">
            No equity curve data yet
          </div>
        )}
      </div>

      {/* Position Stats Grid */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <p className="text-xs font-semibold text-[#888] uppercase tracking-wider mb-3">Position Stats</p>
        <div className="grid grid-cols-3 md:grid-cols-4 gap-3">
          <StatCard label="Closed Positions" value={fmtNum(tearsheet?.total_trades, 0)} />
          <StatCard label="Closed P/L" value={fmt$(tearsheet?.total_pnl)} color={pnlColor(tearsheet?.total_pnl)} />
          <StatCard label="Profit Factor" value={fmtNum(tearsheet?.profit_factor)} color="#f5a623" />
          <StatCard label="Max Drawdown" value={fmtPct(tearsheet?.max_drawdown_pct)} color="#ff1744" />
          <StatCard label="Win Rate" value={fmtPct(tearsheet?.win_rate)} color="#00c853" />
          <StatCard label="Wins" value={fmtNum(wins, 0)} color="#00c853" />
          <StatCard label="Losses" value={fmtNum(losses, 0)} color="#ff1744" />
          <StatCard label="Avg P/L" value={fmt$(tearsheet?.avg_pnl)} color={pnlColor(tearsheet?.avg_pnl)} />
          <StatCard label="Avg Win" value={fmt$(tearsheet?.avg_win)} color="#00c853" />
          <StatCard label="Avg Loss" value={fmt$(tearsheet?.avg_loss)} color="#ff1744" />
          <StatCard label="Sharpe Ratio" value={fmtNum(tearsheet?.sharpe)} color="#2196F3" />
          <StatCard label="Capital at Risk" value={fmt$(bot.risk_usd)} color="#f5a623" />
        </div>
      </div>
    </div>
  )
}

function PositionsTab({ positions }: { positions: Position[] | null }) {
  if (!positions) return <div className="text-xs text-[#555] py-8 text-center">Loading positions...</div>
  if (positions.length === 0) return <div className="text-xs text-[#555] py-8 text-center">No open positions</div>

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl overflow-hidden">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[#1e1e1e]">
            {['Symbol', 'Side', 'Qty', 'Avg Cost', 'Current', 'Unrealized P/L'].map(h => (
              <th key={h} className="text-left px-4 py-2.5 text-[#555] font-medium uppercase tracking-wider text-[10px]">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {positions.map((pos, i) => {
            const pnl = pos.unrealized_pnl
            return (
              <tr key={pos.id ?? i} className="border-b border-[#1e1e1e] hover:bg-[#0d0d0d] transition-colors">
                <td className="px-4 py-2.5 font-bold text-[#f5a623] font-mono">{pos.symbol}</td>
                <td className="px-4 py-2.5">
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                    style={{ color: pos.side === 'long' ? '#00c853' : '#ff1744', background: pos.side === 'long' ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)' }}>
                    {pos.side.toUpperCase()}
                  </span>
                </td>
                <td className="px-4 py-2.5 font-mono">{pos.quantity}</td>
                <td className="px-4 py-2.5 font-mono">${pos.avg_cost.toFixed(2)}</td>
                <td className="px-4 py-2.5 font-mono">{pos.current_price != null ? `$${pos.current_price.toFixed(2)}` : '—'}</td>
                <td className="px-4 py-2.5 font-bold font-mono" style={{ color: pnlColor(pnl) }}>
                  {fmt$(pnl)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function LogTab({ logs, botName }: { logs: AgentLog[] | null; botName: string }) {
  const [typeFilter, setTypeFilter] = useState<string>('all')

  const filtered = (logs ?? []).filter(l =>
    l.strategy_name === botName || l.strategy_name == null
  ).filter(l => typeFilter === 'all' || l.action === typeFilter)

  const types = Array.from(new Set((logs ?? []).map(l => l.action))).slice(0, 8)

  return (
    <div className="space-y-3">
      {/* Filters */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setTypeFilter('all')}
          className="text-xs px-2.5 py-1 rounded transition-colors"
          style={{ background: typeFilter === 'all' ? 'rgba(245,166,35,0.15)' : '#1e1e1e', color: typeFilter === 'all' ? '#f5a623' : '#888', border: `1px solid ${typeFilter === 'all' ? 'rgba(245,166,35,0.4)' : '#1e1e1e'}` }}>
          All
        </button>
        {types.map(t => (
          <button key={t} onClick={() => setTypeFilter(t)}
            className="text-xs px-2.5 py-1 rounded transition-colors"
            style={{ background: typeFilter === t ? 'rgba(245,166,35,0.15)' : '#1e1e1e', color: typeFilter === t ? '#f5a623' : '#888', border: `1px solid ${typeFilter === t ? 'rgba(245,166,35,0.4)' : '#1e1e1e'}` }}>
            {t}
          </button>
        ))}
      </div>

      {/* Log entries */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl overflow-hidden max-h-[500px] overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="text-xs text-[#555] py-8 text-center">No log entries</div>
        ) : (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-[#111111]">
              <tr className="border-b border-[#1e1e1e]">
                {['Date/Time', 'Action', 'Summary', 'Status', 'Duration'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-[#555] font-medium uppercase tracking-wider text-[10px]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 100).map(log => (
                <tr key={log.id} className="border-b border-[#1e1e1e] hover:bg-[#0d0d0d] transition-colors">
                  <td className="px-4 py-2 font-mono text-[#555] whitespace-nowrap">{fmtTs(log.created_at)}</td>
                  <td className="px-4 py-2 font-bold text-[#f5a623]">{log.action}</td>
                  <td className="px-4 py-2 text-[#888] max-w-xs truncate">{log.input_summary ?? log.output_summary ?? '—'}</td>
                  <td className="px-4 py-2">
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                      style={{ color: log.status === 'ok' ? '#00c853' : '#ff1744', background: log.status === 'ok' ? 'rgba(0,200,83,0.1)' : 'rgba(255,23,68,0.1)' }}>
                      {log.status}
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-[#555]">{log.duration_ms != null ? `${log.duration_ms}ms` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function SettingsTab({ bot }: { bot: Bot }) {
  const tickLabel = bot.tick_interval_seconds >= 3600
    ? `${bot.tick_interval_seconds / 3600}h`
    : `${bot.tick_interval_seconds / 60}m`

  return (
    <div className="space-y-4">
      {/* Safeguards */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <p className="text-xs font-semibold text-[#888] uppercase tracking-wider mb-3">Safeguards</p>
        <div className="grid grid-cols-2 gap-3">
          <div className="flex justify-between items-center py-2 border-b border-[#1e1e1e]">
            <span className="text-xs text-[#888]">Allocation USD</span>
            <span className="text-xs font-mono font-bold text-[#f5a623]">${bot.allocation.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center py-2 border-b border-[#1e1e1e]">
            <span className="text-xs text-[#888]">Risk USD</span>
            <span className="text-xs font-mono font-bold text-[#ff1744]">${bot.risk_usd.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center py-2 border-b border-[#1e1e1e]">
            <span className="text-xs text-[#888]">Confidence Threshold</span>
            <span className="text-xs font-mono font-bold text-[#e8e8e8]">{(bot.confidence_threshold * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center py-2 border-b border-[#1e1e1e]">
            <span className="text-xs text-[#888]">Risk Bucket</span>
            <span className="text-xs font-mono font-bold text-[#2196F3]">{bot.risk_bucket}</span>
          </div>
        </div>
      </div>

      {/* Scan Speed */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <p className="text-xs font-semibold text-[#888] uppercase tracking-wider mb-3">Scan Speed</p>
        <div className="flex items-center justify-between py-2">
          <span className="text-xs text-[#888]">Tick Interval</span>
          <span className="text-xs font-mono font-bold text-[#f5a623]">{tickLabel}</span>
        </div>
        <div className="flex items-center justify-between py-2">
          <span className="text-xs text-[#888]">Market Type</span>
          <span className="text-xs font-mono font-bold text-[#e8e8e8]">{bot.market_type}</span>
        </div>
      </div>

      {/* Symbols */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold text-[#888] uppercase tracking-wider">Symbols Watchlist</p>
          <button className="text-xs px-2.5 py-1 rounded border border-[#2a2a2a] text-[#888] hover:text-[#f5a623] hover:border-[#f5a623]/40 transition-colors">
            Edit Watchlist
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {(bot.symbols ?? []).map(sym => (
            <span key={sym} className="text-xs font-mono font-bold px-2 py-1 rounded bg-[#0d0d0d] border border-[#1e1e1e] text-[#f5a623]">
              {sym}
            </span>
          ))}
        </div>
      </div>

      {/* Activity Alerts */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <p className="text-xs font-semibold text-[#888] uppercase tracking-wider mb-3">Activity Alerts</p>
        {['Open position', 'Close position', 'Warning'].map(label => (
          <label key={label} className="flex items-center gap-3 py-2 cursor-pointer">
            <input type="checkbox" defaultChecked className="accent-[#f5a623]" />
            <span className="text-xs text-[#e8e8e8]">{label}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

function AutomationsTab({ bot }: { bot: Bot }) {
  const params = (bot.params ?? {}) as Record<string, unknown>

  return (
    <div className="space-y-4">
      {/* Scanners */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <p className="text-xs font-semibold text-[#888] uppercase tracking-wider mb-3">Scanners</p>
        <div className="flex items-center gap-2 py-2 border border-[#1e1e1e] rounded-lg px-3">
          <span className="w-2 h-2 rounded-full bg-[#00c853]" />
          <span className="text-xs font-semibold text-[#e8e8e8]">{bot.strategy_type} scanner</span>
          <span className="ml-auto text-[10px] text-[#555]">active</span>
        </div>
      </div>

      {/* Bot Inputs */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <p className="text-xs font-semibold text-[#888] uppercase tracking-wider mb-3">Bot Inputs</p>
        {Object.keys(params).length === 0 ? (
          <p className="text-xs text-[#555]">No custom parameters configured</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(params).map(([k, v]) => (
              <div key={k} className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-3">
                <p className="text-[10px] text-[#555] uppercase tracking-wider mb-1">{k}</p>
                <p className="text-xs font-mono font-bold text-[#f5a623]">{String(v)}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Buttons section */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4">
        <p className="text-xs font-semibold text-[#888] uppercase tracking-wider mb-3">Buttons</p>
        <div className="flex gap-2">
          <button className="text-xs px-3 py-1.5 rounded bg-[#1e1e1e] text-[#888] hover:bg-[#2a2a2a] transition-colors">
            Force Signal
          </button>
          <button className="text-xs px-3 py-1.5 rounded bg-[#1e1e1e] text-[#888] hover:bg-[#2a2a2a] transition-colors">
            Reset State
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

const TABS = ['Dashboard', 'Positions', 'Log', 'Settings', 'Automations'] as const
type Tab = typeof TABS[number]

export default function BotDetail() {
  const { botId } = useParams<{ botId: string }>()
  const [activeTab, setActiveTab] = useState<Tab>('Dashboard')
  const [autoEnabled, setAutoEnabled] = useState(true)

  const { data: dashData, isLoading: botLoading } = useQuery({
    queryKey: ['bot-dashboard'],
    queryFn: () => api.get('/strategies/dashboard').then(r => r.data),
    staleTime: 30_000,
  })

  const { data: tearsheet } = useQuery({
    queryKey: ['tearsheet', botId],
    queryFn: () => api.get('/analytics/tearsheet?days=90').then(r => r.data).catch(() => null),
    staleTime: 60_000,
  })

  const { data: positions } = useQuery({
    queryKey: ['positions'],
    queryFn: () => api.get('/positions/').then(r => r.data),
    refetchInterval: 10_000,
  })

  const { data: logs } = useQuery({
    queryKey: ['agent-logs-bot', botId],
    queryFn: () => api.get('/agent-logs/?limit=100').then(r => r.data),
    staleTime: 30_000,
  })

  const bot: Bot | null = dashData?.bots?.find((b: Bot) => b.id === botId) ?? null

  if (botLoading) {
    return (
      <div className="space-y-4">
        <div className="h-8 bg-[#1e1e1e] rounded animate-pulse w-48" />
        <div className="h-40 bg-[#1e1e1e] rounded animate-pulse" />
      </div>
    )
  }

  if (!bot) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-4">
        <p className="text-[#555] text-sm">Bot not found: {botId}</p>
        <Link to="/bot-dashboard" className="text-xs text-[#f5a623] hover:underline flex items-center gap-1">
          <ArrowLeft size={12} /> Back to Bot Dashboard
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Link to="/bot-dashboard" className="text-[#555] hover:text-[#f5a623] transition-colors">
          <ArrowLeft size={16} />
        </Link>
        <div>
          <h1 className="text-lg font-bold text-[#e8e8e8]">{bot.display_name || bot.name}</h1>
          <p className="text-xs text-[#555]">{bot.strategy_type} · {bot.market_type}</p>
        </div>
      </div>

      {/* Status bar */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl px-4 py-3 flex items-center gap-6 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[#555] uppercase tracking-wider">Account</span>
          <span className="text-xs font-bold text-[#e8e8e8]">Paper Trading</span>
        </div>
        <div className="h-3 w-px bg-[#1e1e1e]" />
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[#555] uppercase tracking-wider">Bot Group</span>
          <span className="text-xs font-bold text-[#888]">None</span>
        </div>
        <div className="h-3 w-px bg-[#1e1e1e]" />
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[#555] uppercase tracking-wider">Automatic</span>
          <button onClick={() => setAutoEnabled(e => !e)} className="flex items-center gap-1 transition-colors">
            {autoEnabled
              ? <ToggleRight size={18} className="text-[#00c853]" />
              : <ToggleLeft size={18} className="text-[#555]" />}
            <span className="text-xs font-bold" style={{ color: autoEnabled ? '#00c853' : '#555' }}>
              {autoEnabled ? 'ON' : 'OFF'}
            </span>
          </button>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs font-mono" style={{ color: pnlColor(bot.total_pnl) }}>
            Total P/L: {fmt$(bot.total_pnl)}
          </span>
          <span className="text-xs font-mono text-[#888]">
            Win Rate: {bot.win_rate != null ? `${bot.win_rate.toFixed(1)}%` : '—'}
          </span>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-[#1e1e1e]">
        {TABS.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className="px-4 py-2.5 text-xs font-semibold transition-colors relative"
            style={{ color: activeTab === tab ? '#f5a623' : '#555' }}
          >
            {tab}
            {activeTab === tab && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#f5a623]" />
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {activeTab === 'Dashboard' && (
          <DashboardTab bot={bot} tearsheet={tearsheet ?? null} />
        )}
        {activeTab === 'Positions' && (
          <PositionsTab positions={Array.isArray(positions) ? positions : null} />
        )}
        {activeTab === 'Log' && (
          <LogTab logs={Array.isArray(logs) ? logs : null} botName={bot.name} />
        )}
        {activeTab === 'Settings' && (
          <SettingsTab bot={bot} />
        )}
        {activeTab === 'Automations' && (
          <AutomationsTab bot={bot} />
        )}
      </div>
    </div>
  )
}
