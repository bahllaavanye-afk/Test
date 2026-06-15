import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSelector, useDispatch } from 'react-redux'
import { Link } from 'react-router-dom'
import api from '../api/client'
import { RegimeIndicator } from '../components/risk/RegimeIndicator'
import { selectTradingMode, setMode } from '../store/slices/tradingModeSlice'
import { TVAdvancedChart } from '../components/charts/TVAdvancedChart'
import NewsSentimentPanel from '../components/trading/NewsSentimentPanel'
import TradeMarkerChart from '../components/charts/TradeMarkerChart'
import TraderLevel from '../components/gamification/TraderLevel'
import { WatchlistPanel } from '../components/charts/WatchlistPanel'
import { MarketHeatmap } from '../components/charts/MarketHeatmap'
import { AlertCenter } from '../components/alerts/AlertCenter'
import { EconomicCalendar } from '../components/charts/EconomicCalendar'

function vixColor(vix: number | null | undefined): string {
  if (vix == null) return '#888888'
  if (vix > 30) return '#ff1744'
  if (vix > 20) return '#f5a623'
  return '#00c853'
}

function biasColor(bias: string | undefined): string {
  if (bias === 'risk_on') return '#00c853'
  if (bias === 'risk_off') return '#ff1744'
  return '#f5a623'
}

function MetricCard({ label, value, sub, color = 'var(--accent)', glowClass = '', href }: { label: string; value: string; sub?: string; color?: string; glowClass?: string; href?: string }) {
  const inner = (
    <div className={`kpi-card ${glowClass} transition-all duration-200 ${href ? 'hover:border-[#f5a623]/40 cursor-pointer' : ''}`}>
      <p className="section-header" style={{marginBottom:8}}>{label}</p>
      <p className="mono-num" style={{fontSize:22,fontWeight:700,color,lineHeight:1}}>{value}</p>
      {sub && <p style={{fontSize:10,color:'var(--muted)',marginTop:4}}>{sub}</p>}
    </div>
  )
  if (href) return <Link to={href}>{inner}</Link>
  return inner
}

function SkeletonRow({ count = 1 }: { count?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="h-5 bg-[#1e1e1e] rounded animate-pulse" />
      ))}
    </div>
  )
}

// ── System Status Row ───────────────────────────────────────────────────────
function SystemStatusRow() {
  const { data: status, isLoading } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => api.get('/analytics/system-status').then(r => r.data),
    refetchInterval: 15_000,
  })

  const regimeLabel = status?.regime === 1 ? 'RISK ON' : status?.regime === -1 ? 'RISK OFF' : 'NEUTRAL'
  const regimeColor = status?.regime === 1 ? '#00c853' : status?.regime === -1 ? '#ff1744' : '#f5a623'

  const lastSignal = status?.last_signal_at
    ? (() => {
        const diff = Math.round((Date.now() - new Date(status.last_signal_at).getTime()) / 60000)
        return diff < 60 ? `${diff}m ago` : `${Math.round(diff / 60)}h ago`
      })()
    : '—'

  const deskItems = status?.strategies_by_desk
    ? Object.entries(status.strategies_by_desk as Record<string, number>)
    : []

  return (
    <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg px-4 py-3" aria-busy={isLoading} aria-live="polite">
      <div className="flex items-center gap-6 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00c853] animate-pulse" />
          <span className="text-[10px] text-[#555] uppercase tracking-wider">System Status</span>
        </div>

        {isLoading ? (
          <div className="flex gap-4">
            {[1,2,3,4].map(i => <div key={i} className="w-20 h-4 bg-[#1e1e1e] rounded animate-pulse" />)}
          </div>
        ) : (
          <>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-[#555]">Active Strategies</span>
              <span className="text-xs font-bold font-mono text-[#f5a623]">
                {status?.active_strategies ?? '—'} / {status?.total_strategies ?? '—'}
              </span>
            </div>

            <div className="h-3 w-px bg-[#1e1e1e]" />

            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-[#555]">Last Signal</span>
              <span className="text-xs font-mono text-[#e8e8e8]">{lastSignal}</span>
            </div>

            <div className="h-3 w-px bg-[#1e1e1e]" />

            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-[#555]">Regime</span>
              <span className="text-xs font-bold px-1.5 py-0.5 rounded" style={{ color: regimeColor, background: `${regimeColor}18` }}>
                {regimeLabel}
              </span>
            </div>

            {status?.vix != null && (
              <>
                <div className="h-3 w-px bg-[#1e1e1e]" />
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-[#555]">VIX</span>
                  <span className="text-xs font-bold font-mono" style={{ color: vixColor(status.vix) }}>
                    {status.vix.toFixed(1)}
                  </span>
                </div>
              </>
            )}

            <div className="h-3 w-px bg-[#1e1e1e]" />

            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-[#555]">Open Positions</span>
              <span className="text-xs font-bold font-mono text-[#2196f3]">{status?.open_positions ?? '—'}</span>
            </div>

            {deskItems.length > 0 && (
              <>
                <div className="h-3 w-px bg-[#1e1e1e]" />
                <div className="flex items-center gap-3">
                  {deskItems.map(([desk, count]) => (
                    <div key={desk} className="flex items-center gap-1">
                      <span className="text-[10px] text-[#444] capitalize">{desk}</span>
                      <span className="text-xs font-mono text-[#888]">{count}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── Top Movers (by % P&L) ───────────────────────────────────────────────────
function TopMovers({ positions }: { positions: any[] | undefined }) {
  if (!positions || positions.length === 0) return null

  const sorted = [...positions]
    .filter(p => p.avg_cost > 0 && p.current_price != null)
    .map(p => ({
      symbol: p.symbol,
      pct: ((p.current_price - p.avg_cost) / p.avg_cost * 100) * (p.side === 'short' ? -1 : 1),
      pnl: p.unrealized_pnl ?? 0,
    }))
    .sort((a, b) => Math.abs(b.pct) - Math.abs(a.pct))
    .slice(0, 3)

  if (sorted.length === 0) return null

  return (
    <div className="kpi-card">
      <p className="section-header" style={{ marginBottom: 10 }}>Top Movers</p>
      <div className="space-y-2">
        {sorted.map(p => (
          <div key={p.symbol} className="flex items-center justify-between">
            <span className="text-xs font-bold font-mono text-[#f5a623]">{p.symbol}</span>
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono" style={{ color: p.pnl >= 0 ? '#00c853' : '#ff1744' }}>
                {p.pnl >= 0 ? '+' : '-'}${Math.abs(p.pnl).toFixed(2)}
              </span>
              <span className="text-xs font-bold font-mono px-1.5 py-0.5 rounded"
                style={{ color: p.pct >= 0 ? '#00c853' : '#ff1744', background: p.pct >= 0 ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)' }}>
                {p.pct >= 0 ? '+' : ''}{p.pct.toFixed(2)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Risk Gauge (portfolio drawdown) ────────────────────────────────────────
function RiskGauge() {
  const { data: tearsheet } = useQuery({
    queryKey: ['tearsheet-drawdown'],
    queryFn: () => api.get('/analytics/tearsheet?days=90').then(r => r.data).catch(() => null),
    refetchInterval: 300_000,
    retry: false,
  })

  const dd = tearsheet?.max_drawdown_pct ?? 0
  const ddAbs = Math.abs(dd)
  // Scale: 0% = safe (green), 10% = caution (yellow), 20%+ = danger (red)
  const pct = Math.min(ddAbs / 20, 1) * 100
  const gaugeColor = ddAbs < 5 ? '#00c853' : ddAbs < 10 ? '#f5a623' : '#ff1744'

  return (
    <div className="kpi-card">
      <div className="flex items-center justify-between mb-2">
        <p className="section-header" style={{ marginBottom: 0 }}>Risk Gauge</p>
        <span className="text-[10px] text-[#555]">Max Drawdown (90d)</span>
      </div>
      {!tearsheet ? (
        <div className="h-8 bg-[#1e1e1e] rounded animate-pulse" />
      ) : (
        <>
          <div className="flex items-end gap-2 mb-2">
            <span className="text-xl font-bold font-mono" style={{ color: gaugeColor }}>
              {dd.toFixed(2)}%
            </span>
            <span className="text-xs text-[#555] mb-0.5">drawdown</span>
          </div>
          <div className="h-2 bg-[#1e1e1e] rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-all duration-700"
              style={{ width: `${pct}%`, background: gaugeColor, boxShadow: `0 0 6px ${gaugeColor}60` }} />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[9px] text-[#333]">0%</span>
            <span className="text-[9px] text-[#333]">10%</span>
            <span className="text-[9px] text-[#333]">20%+</span>
          </div>
        </>
      )}
    </div>
  )
}

function ConfirmLiveModal({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) {
  const [input, setInput] = useState('')
  const valid = input.trim() === 'CONFIRM LIVE'
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-[#111111] border border-[#ff1744]/40 rounded-xl p-6 w-full max-w-md shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <span className="w-3 h-3 rounded-full bg-[#ff1744] animate-pulse inline-block" />
          <h2 className="text-[#ff1744] font-bold text-base">Switch to Live Trading</h2>
        </div>
        <p className="text-[#888888] text-sm mb-2">
          You are about to switch to <span className="text-[#ff1744] font-bold">LIVE trading mode</span>.
          Real money will be used. Strategies will execute against live markets.
        </p>
        <ul className="text-xs text-[#888888] mb-4 space-y-1 list-disc list-inside">
          <li>All active strategies will trade with real capital</li>
          <li>Orders will be sent to live broker connections</li>
          <li>Risk limits and position sizing apply immediately</li>
        </ul>
        <p className="text-xs text-[#888888] mb-2">Type <span className="text-white font-mono font-bold">CONFIRM LIVE</span> to proceed:</p>
        <input autoFocus value={input} onChange={e => setInput(e.target.value)}
          className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-3 py-2 text-sm font-mono text-white mb-4 focus:outline-none focus:border-[#ff1744]/60"
          placeholder="CONFIRM LIVE" />
        <div className="flex gap-3">
          <button onClick={onCancel} className="flex-1 px-4 py-2 rounded bg-[#1e1e1e] text-[#888888] text-sm hover:bg-[#2e2e2e] transition-colors">Cancel</button>
          <button onClick={() => valid && onConfirm()} disabled={!valid} className="flex-1 px-4 py-2 rounded text-sm font-bold transition-all duration-200"
            style={{ background: valid ? '#ff1744' : '#3a1a1e', color: valid ? '#fff' : '#666', cursor: valid ? 'pointer' : 'not-allowed' }}>
            Switch to Live
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Bot Activity Feed ──────────────────────────────────────────────────────
function BotActivityFeed() {
  const { data: bots, isLoading } = useQuery({
    queryKey: ['bots-summary'],
    queryFn: () => api.get('/bots/summary/all').then(r => r.data).catch(() => [] as any[]),
    refetchInterval: 15_000,
    retry: false,
  })
  const { data: signals } = useQuery({
    queryKey: ['recent-signals'],
    queryFn: () => api.get('/strategies/signals/recent').then(r => r.data).catch(() => [] as any[]),
    refetchInterval: 10_000,
    retry: false,
  })

  const botList: any[] = Array.isArray(bots) ? bots : []
  const signalList: any[] = Array.isArray(signals) ? signals : []

  // Combine bots + signals into a unified feed
  const feedItems = [
    ...botList.slice(0, 5).map((b: any) => ({
      id: `bot-${b.id}`,
      type: 'bot' as const,
      label: b.name || 'Bot',
      description: `${b.status || 'running'} · ${b.open_positions ?? 0} open positions`,
      pnl: b.total_pnl_30d ?? 0,
      ts: b.last_signal_at || b.updated_at || null,
    })),
    ...signalList.slice(0, 8).map((s: any) => ({
      id: `sig-${s.id || Math.random()}`,
      type: 'signal' as const,
      label: s.strategy_name || s.strategy || 'Strategy',
      description: `${s.direction || s.side || 'signal'} ${s.symbol || ''}`.trim(),
      pnl: s.realized_pnl ?? 0,
      ts: s.created_at || s.ts || null,
    })),
  ].slice(0, 8)

  const fmtTime = (ts: string | null) => {
    if (!ts) return ''
    try {
      const diff = Math.round((Date.now() - new Date(ts).getTime()) / 60_000)
      if (diff < 60) return `${diff}m ago`
      if (diff < 1440) return `${Math.floor(diff / 60)}h ago`
      return new Date(ts).toLocaleDateString()
    } catch { return '' }
  }

  const dotColor = (type: 'bot' | 'signal') =>
    type === 'signal' ? '#f5a623' : '#2196f3'

  return (
    <div className="kpi-card">
      <div className="flex items-center justify-between mb-3">
        <p className="section-header" style={{ marginBottom: 0 }}>Bot Activity Feed</p>
        <a href="/bot-dashboard" className="text-[10px] text-[#f5a623] hover:underline">View all →</a>
      </div>
      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3].map(i => <div key={i} className="h-8 bg-[#1e1e1e] rounded animate-pulse" />)}
        </div>
      ) : feedItems.length === 0 ? (
        <p className="text-xs text-[#555] py-3 text-center">No bot activity yet. <a href="/bots" className="text-[#f5a623] hover:underline">Create a bot →</a></p>
      ) : (
        <div className="space-y-1.5">
          {feedItems.map(item => (
            <div key={item.id} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-[#0d0d0d] border border-[#1e1e1e] hover:border-[#2a2a2a] transition-colors">
              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: dotColor(item.type), boxShadow: `0 0 4px ${dotColor(item.type)}80` }} />
              {item.ts && <span className="text-[10px] text-[#444] font-mono shrink-0 w-14">{fmtTime(item.ts)}</span>}
              <span className="text-xs font-semibold text-[#f5a623] shrink-0">{item.label}</span>
              <span className="text-xs text-[#777] truncate flex-1">{item.description}</span>
              {item.pnl !== 0 && (
                <span className="text-xs font-mono font-bold shrink-0" style={{ color: item.pnl >= 0 ? '#00c853' : '#ff1744' }}>
                  {item.pnl >= 0 ? '+' : ''}${Math.abs(item.pnl).toFixed(2)}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Top 5 symbols for the "Recent Trades" section
const TRADE_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']

// ── Bot Activity Feed ───────────────────────────────────────────────────────
interface BotActivity {
  id: string
  type: 'signal' | 'fill' | 'info'
  ts: string
  bot_name: string
  description: string
  pnl: number
}

const PLACEHOLDER_ACTIVITIES: BotActivity[] = [
  { id: '1', type: 'fill',   ts: new Date(Date.now() - 1  * 60000).toISOString(), bot_name: 'MomentumBot-SPY',    description: 'BUY 50 SPY @ $521.40 — filled',           pnl:  87.50 },
  { id: '2', type: 'signal', ts: new Date(Date.now() - 3  * 60000).toISOString(), bot_name: 'ArbitrageBot-BTC',   description: 'Long signal triggered — RSI 28 divergence', pnl:   0    },
  { id: '3', type: 'fill',   ts: new Date(Date.now() - 7  * 60000).toISOString(), bot_name: 'TrendFollower-QQQ',  description: 'SELL 25 QQQ @ $448.10 — filled',           pnl: -32.25 },
  { id: '4', type: 'signal', ts: new Date(Date.now() - 14 * 60000).toISOString(), bot_name: 'MLPredictor-NVDA',   description: 'Short signal: model confidence 0.81',       pnl:   0    },
  { id: '5', type: 'info',   ts: new Date(Date.now() - 22 * 60000).toISOString(), bot_name: 'RiskGuard',          description: 'Portfolio drawdown limit 4.2% — within threshold', pnl: 0 },
]

function formatTime(ts: string): string {
  const diff = Math.round((Date.now() - new Date(ts).getTime()) / 60000)
  if (diff < 1) return 'just now'
  if (diff < 60) return `${diff}m ago`
  return `${Math.round(diff / 60)}h ago`
}

function BotActivityFeed() {
  const { data: botSummary } = useQuery({
    queryKey: ['bot-summary-all'],
    queryFn: () => api.get('/bots/summary/all').then(r => r.data).catch(() => null),
    refetchInterval: 15_000,
    retry: false,
  })
  const { data: recentSignals } = useQuery({
    queryKey: ['strategies-signals-recent'],
    queryFn: () => api.get('/strategies/signals/recent').then(r => r.data).catch(() => null),
    refetchInterval: 15_000,
    retry: false,
  })

  // Build activity list from live data if available, else show placeholder
  let activities: BotActivity[] = []

  if (Array.isArray(botSummary?.events) || Array.isArray(recentSignals)) {
    const botEvents: BotActivity[] = Array.isArray(botSummary?.events)
      ? botSummary.events.map((e: any) => ({
          id: e.id ?? String(Math.random()),
          type: (e.type as BotActivity['type']) ?? 'info',
          ts: e.ts ?? e.timestamp ?? new Date().toISOString(),
          bot_name: e.bot_name ?? e.name ?? 'Bot',
          description: e.description ?? e.message ?? '',
          pnl: e.pnl ?? 0,
        }))
      : []
    const signalEvents: BotActivity[] = Array.isArray(recentSignals)
      ? recentSignals.map((s: any) => ({
          id: s.id ?? String(Math.random()),
          type: 'signal' as const,
          ts: s.ts ?? s.timestamp ?? new Date().toISOString(),
          bot_name: s.strategy_name ?? s.bot_name ?? 'Strategy',
          description: `${s.direction ?? s.side ?? 'Signal'} on ${s.symbol ?? ''} — ${s.reason ?? 'triggered'}`,
          pnl: s.pnl ?? 0,
        }))
      : []
    activities = [...botEvents, ...signalEvents]
      .sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime())
      .slice(0, 20)
  }

  const isPlaceholder = activities.length === 0
  const displayActivities = isPlaceholder ? PLACEHOLDER_ACTIVITIES : activities

  return (
    <div className="kpi-card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <p className="section-header" style={{ marginBottom: 0 }}>Bot Activity Feed</p>
          {isPlaceholder && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1e1e1e] text-[#555] font-mono">placeholder</span>
          )}
        </div>
        <span className="text-[10px] text-[#555] font-mono">last 20 events</span>
      </div>
      <div className="space-y-2">
        {displayActivities.map(event => (
          <div
            key={event.id}
            className="flex items-center gap-3 p-3 bg-[#111] border border-[#1e1e1e] rounded-lg"
          >
            <span
              className={`w-2 h-2 rounded-full flex-shrink-0 ${
                event.type === 'signal' ? 'bg-[#f5a623]'
                : event.type === 'fill'   ? 'bg-[#00c853]'
                : 'bg-[#888]'
              }`}
            />
            <span className="text-xs text-[#888] flex-shrink-0 font-mono w-14">{formatTime(event.ts)}</span>
            <span className="text-sm font-medium text-[#e8e8e8] flex-shrink-0 truncate max-w-[120px]">{event.bot_name}</span>
            <span className="text-sm text-[#aaa] flex-1 truncate">{event.description}</span>
            {event.pnl !== 0 && (
              <span className={`ml-auto text-xs flex-shrink-0 font-mono font-bold ${event.pnl >= 0 ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
                {event.pnl >= 0 ? '+' : ''}${event.pnl.toFixed(2)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const dispatch = useDispatch()
  const mode = useSelector(selectTradingMode)
  const [showLiveModal, setShowLiveModal] = useState(false)
  const [chartSymbol, setChartSymbol] = useState('NYSE:SPY')
  const [tradeSymbol, setTradeSymbol] = useState('SPY')

  const { data: perf } = useQuery({ queryKey: ['performance'], queryFn: () => api.get('/analytics/performance').then(r => r.data), refetchInterval: 30_000 })
  const { data: positions } = useQuery({ queryKey: ['positions'], queryFn: () => api.get('/positions/').then(r => r.data), refetchInterval: 10_000 })
  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: () => api.get('/strategies/').then(r => r.data) })
  const { data: macro } = useQuery({ queryKey: ['macro'], queryFn: () => api.get('/analytics/macro').then(r => r.data), refetchInterval: 300_000 })
  const { data: sentiment } = useQuery({ queryKey: ['sentiment'], queryFn: () => api.get('/analytics/sentiment').then(r => r.data), refetchInterval: 600_000 })
  const { data: agentStatus } = useQuery({ queryKey: ['agents-status'], queryFn: () => api.get('/agents/status').then(r => r.data), refetchInterval: 15_000 })
  const { data: accounts } = useQuery({ queryKey: ['accounts'], queryFn: () => api.get('/accounts/').then(r => r.data), refetchInterval: 30_000 })
  const { data: recentOrders } = useQuery({ queryKey: ['recent-orders'], queryFn: () => api.get('/orders/?limit=5').then(r => r.data), refetchInterval: 15_000, retry: false })

  const activeCount = Array.isArray(strategies) ? strategies.filter((s: any) => s.is_active || s.is_enabled).length : 0
  const totalPnl = perf?.total_pnl ?? 0
  const noAccountConnected = !accounts || (Array.isArray(accounts) && accounts.length === 0)
  const agentList: any[] = Array.isArray(agentStatus?.agents) ? agentStatus.agents : []

  const isLive = mode === 'live'
  const isPaper = mode === 'paper'
  const CHART_SYMBOLS = ['NYSE:SPY', 'NASDAQ:AAPL', 'NASDAQ:MSFT', 'NASDAQ:QQQ']

  return (
    <div className="space-y-5">
      {showLiveModal && <ConfirmLiveModal onConfirm={() => { dispatch(setMode('live')); setShowLiveModal(false) }} onCancel={() => setShowLiveModal(false)} />}

      {/* ── Bot Activity Feed (top of dashboard) ── */}
      <BotActivityFeed />

      <TraderLevel />

      <div className={`rounded-lg px-4 py-3 flex items-center justify-between transition-all duration-500 ${isLive ? 'bg-[#ff1744]/10 border border-[#ff1744]/40' : 'bg-[#f5a623]/10 border border-[#f5a623]/30'}`}>
        <div className="flex items-center gap-3">
          <span className="w-3 h-3 rounded-full inline-block"
            style={{ background: isLive ? '#ff1744' : '#f5a623', boxShadow: isLive ? '0 0 8px #ff1744' : '0 0 8px #f5a623', animation: isLive ? 'pulse 1s infinite' : 'none' }} />
          <div>
            <p className="text-sm font-bold" style={{ color: isLive ? '#ff1744' : '#f5a623' }}>
              {isLive ? 'LIVE TRADING — REAL MONEY AT RISK' : 'PAPER TRADING MODE'}
            </p>
            <p className="text-xs text-[#888888] mt-0.5">
              {isLive ? 'Strategies are executing against live markets. Monitor positions closely.' : 'All orders are simulated. No real capital at risk. Run paper for 2 weeks before going live.'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isPaper && (
            <button onClick={() => setShowLiveModal(true)} className="px-3 py-1.5 rounded text-xs font-bold text-white transition-all duration-200 hover:opacity-90 active:scale-95"
              style={{ background: 'linear-gradient(135deg, #ff1744, #c62828)' }}>Switch to Live Trading</button>
          )}
          {isLive && (
            <button onClick={() => dispatch(setMode('paper'))} className="px-3 py-1.5 rounded text-xs font-bold text-black transition-all duration-200 hover:opacity-90 active:scale-95"
              style={{ background: '#f5a623' }}>Switch to Paper</button>
          )}
        </div>
      </div>

      {noAccountConnected && (
        <div className="bg-[#111111] border border-[#f5a623]/30 rounded-lg p-4 flex items-center gap-3">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f5a623" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
          </svg>
          <div>
            <p className="text-sm text-[#f5a623] font-semibold">Connect your Alpaca account to see live P&amp;L</p>
            <p className="text-xs text-[#888888] mt-0.5">No broker account detected. <a href="/settings" className="text-[#f5a623] underline">Add API keys in Settings</a> to start paper trading.</p>
          </div>
        </div>
      )}

      <SystemStatusRow />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Total P&L" value={perf ? `$${totalPnl.toFixed(2)}` : '—'} sub={perf ? `${perf.total_trades ?? 0} trades` : 'Connect Alpaca to see P&L'} color={perf ? (totalPnl >= 0 ? 'var(--green)' : 'var(--red)') : 'var(--muted)'} glowClass={perf ? (totalPnl >= 0 ? 'glow-green' : 'glow-red') : ''} href="/pnl" />
        <MetricCard label="Open Positions" value={Array.isArray(positions) ? String(positions.length) : '—'} sub="live positions" color="var(--blue)" glowClass="glow-blue" href="/equity" />
        <MetricCard label="Active Strategies" value={String(activeCount)} sub="running 24/7" color="var(--accent)" glowClass="glow-accent" href="/comparison" />
        <MetricCard label="Target Sharpe" value=">2.0" sub="vs SPY 0.47" color="var(--purple)" href="/analytics" />
      </div>

      <RegimeIndicator />

      <BotActivityFeed />

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 flex flex-col gap-2">
          <div className="flex gap-2">
            {CHART_SYMBOLS.map(s => (
              <button key={s} onClick={() => setChartSymbol(s)}
                className="text-xs px-2 py-1 rounded transition-colors"
                style={{
                  background: chartSymbol === s ? 'var(--accent-dim)' : 'var(--surface2)',
                  border: `1px solid ${chartSymbol === s ? 'rgba(245,166,35,0.4)' : 'var(--border)'}`,
                  color: chartSymbol === s ? 'var(--accent)' : 'var(--muted)',
                }}>
                {s.split(':')[1]}
              </button>
            ))}
          </div>
          <div role="img" aria-label={`TradingView advanced chart for ${chartSymbol}`}>
            <TVAdvancedChart symbol={chartSymbol} />
          </div>
        </div>

        <div className="space-y-3">
          <div className="kpi-card">
            <p className="section-header" style={{marginBottom:12}}>Account Summary</p>
            {noAccountConnected ? (
              <div className="text-center py-4 space-y-2">
                <p className="text-xs text-[#888888]">No account connected</p>
                <a href="/settings" className="text-xs text-[#f5a623] underline">Add API keys in Settings</a>
              </div>
            ) : (
              <div className="space-y-2">
                {(Array.isArray(accounts) ? accounts : [accounts]).filter(Boolean).map((acc: any, i: number) => (
                  <div key={acc?.id ?? i} className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-[#888888]">{acc?.broker ?? 'Account'}</span>
                      <span className="text-xs font-mono font-bold" style={{ color: (acc?.total_pnl ?? 0) >= 0 ? '#00c853' : '#ff1744' }}>
                        {acc?.total_pnl != null ? `${acc.total_pnl >= 0 ? '+' : ''}$${acc.total_pnl.toFixed(2)}` : '---'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-[#555]">Equity</span>
                      <span className="text-xs font-mono text-[#e8e8e8]">
                        {acc?.equity != null ? `$${Number(acc.equity).toLocaleString()}` : '---'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="kpi-card">
            <div className="flex items-center justify-between mb-3">
              <p className="section-header" style={{marginBottom:0}}>
                Agent Team
              </p>
              {agentList.length > 0 && (
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] font-mono text-[#888888]">{agentList.filter((a: any) => a.running).length}/{agentList.length} active</span>
                  <span className="w-1.5 h-1.5 rounded-full bg-[#00c853] pulse-green" />
                </div>
              )}
            </div>
            {agentList.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-4 text-center space-y-1">
                <p className="text-xs text-[#555]">No agent data</p>
                <p className="text-[10px] text-[#444]">Start the backend to see agent health.</p>
              </div>
            ) : (
              <div className="space-y-1.5">
                {agentList.map((agent: any) => {
                  const isRunning = agent.running ?? false
                  const lastRun = agent.last_run ? new Date(agent.last_run) : null
                  const minutesAgo = lastRun ? Math.round((Date.now() - lastRun.getTime()) / 60000) : null
                  return (
                    <div key={agent.name} className="flex items-center gap-2 py-1 border-b border-[#1a1a1a] last:border-0">
                      <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ background: isRunning ? '#00c853' : '#555555', boxShadow: isRunning ? '0 0 4px #00c853' : 'none' }} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-[#e8e8e8] truncate">{agent.name}</span>
                          <span className="text-[10px] text-[#555] ml-1 flex-shrink-0 font-mono">{agent.total_runs ?? 0}r</span>
                        </div>
                        <div className="flex items-center justify-between mt-0.5">
                          <span className="text-[10px] text-[#555] truncate">{agent.role ?? ''}</span>
                          <span className="text-[10px] text-[#444] flex-shrink-0">
                            {minutesAgo !== null ? (minutesAgo < 60 ? `${minutesAgo}m ago` : `${Math.round(minutesAgo/60)}h ago`) : '---'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Top Movers + Risk Gauge ── */}
      {Array.isArray(positions) && positions.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <TopMovers positions={positions} />
          <RiskGauge />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="kpi-card">
          <p className="section-header" style={{marginBottom:12}}>Macro Signals</p>
          {!macro ? (
            <SkeletonRow count={4} />
          ) : (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">VIX Level</span>
                <span className="text-sm font-bold font-mono" style={{ color: vixColor(macro.vix) }}>
                  {macro.vix != null ? macro.vix.toFixed(2) : '---'}
                  {macro.signals?.vix_regime && <span className="ml-1 text-xs font-normal">({macro.signals.vix_regime})</span>}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Yield Curve (10Y-2Y)</span>
                <span className="text-sm font-mono" style={{ color: macro.signals?.yield_curve_inverted ? '#ff1744' : '#00c853' }}>
                  {macro.yield_spread_10y2y != null ? `${(macro.yield_spread_10y2y * 100).toFixed(0)} bps` : '---'}
                  {macro.signals?.yield_curve_inverted != null && <span className="ml-1 text-xs">({macro.signals.yield_curve_inverted ? 'INVERTED' : 'normal'})</span>}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Macro Bias</span>
                <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ color: biasColor(macro.macro_bias), background: `${biasColor(macro.macro_bias)}20` }}>
                  {macro.macro_bias?.replace('_', ' ').toUpperCase() ?? '---'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Macro Score</span>
                <span className="text-sm font-bold font-mono" style={{ color: biasColor(macro.macro_bias) }}>
                  {macro.macro_score != null ? (macro.macro_score > 0 ? `+${macro.macro_score}` : macro.macro_score) : '---'} / 3
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="kpi-card">
          <p className="section-header" style={{marginBottom:12}}>Reddit Buzz (WSB)</p>
          {!sentiment ? (
            <SkeletonRow count={5} />
          ) : sentiment.error ? (
            <p className="text-xs text-[#888888]">Sentiment unavailable</p>
          ) : (Array.isArray(sentiment.results) && sentiment.results.length === 0) ? (
            <p className="text-xs text-[#555]">No sentiment data available</p>
          ) : (
            <div className="space-y-2">
              {(Array.isArray(sentiment.results) ? sentiment.results : []).slice(0, 5).map((item: any, i: number) => {
                const maxMentions = sentiment.results?.[0]?.mentions_24h ?? 1
                const pct = Math.round(((item.mentions_24h ?? 0) / Math.max(maxMentions, 1)) * 100)
                return (
                  <div key={item.ticker ?? i} className="space-y-0.5">
                    <div className="flex justify-between text-xs">
                      <span className="font-bold text-[#f5a623]">{item.ticker}</span>
                      <span className="text-[#888888]">{item.mentions_24h?.toLocaleString()} mentions</span>
                    </div>
                    <div className="h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
                      <div className="h-full bg-[#f5a623] rounded-full transition-all duration-500" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Recent Signals Feed ── */}
      <div className="kpi-card">
        <div className="flex items-center justify-between mb-3">
          <p className="section-header" style={{marginBottom:0}}>Recent Signals</p>
          <span className="text-[10px] text-[#555] font-mono">last 5 orders</span>
        </div>
        {!recentOrders ? (
          <div className="space-y-2">{[1,2,3].map(i => <div key={i} className="h-8 bg-[#1e1e1e] rounded animate-pulse" />)}</div>
        ) : !Array.isArray(recentOrders) || recentOrders.length === 0 ? (
          <p className="text-xs text-[#555555] text-center py-4">No recent orders. Strategies begin executing once accounts are connected.</p>
        ) : (
          <div className="space-y-1.5">
            {(recentOrders as any[]).slice(0, 5).map((order: any, i: number) => {
              const isBuy = order.side === 'buy'
              const filled = order.status === 'filled' || order.status === 'completed'
              return (
                <div key={order.id ?? i} className="flex items-center gap-3 px-2.5 py-2 bg-[#0a0a0a] rounded border border-[#1e1e1e] hover:border-[#2a2a2a] transition-colors animate-slide-up">
                  <span
                    className="text-[10px] font-black px-1.5 py-0.5 rounded shrink-0"
                    style={{
                      color: isBuy ? '#00c853' : '#ff1744',
                      background: isBuy ? 'rgba(0,200,83,0.15)' : 'rgba(255,23,68,0.15)',
                    }}
                  >
                    {isBuy ? 'BUY' : 'SELL'}
                  </span>
                  <span className="text-xs font-mono font-bold text-[#f5a623]">{order.symbol ?? '---'}</span>
                  <span className="text-xs text-[#888888]">{order.qty ?? order.quantity ?? '---'} shares</span>
                  {order.price != null && <span className="text-xs font-mono text-[#e8e8e8]">${Number(order.price).toFixed(2)}</span>}
                  <div className="ml-auto flex items-center gap-2">
                    <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${filled ? 'text-[#00c853] bg-[#00c853]/10' : 'text-[#888888] bg-[#1e1e1e]'}`}>
                      {order.status ?? 'pending'}
                    </span>
                    {order.created_at && (
                      <span className="text-[10px] text-[#444444] font-mono">
                        {new Date(order.created_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* ── TradingView-style panels ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <WatchlistPanel
          symbols={['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'BTC/USD']}
          onSelectSymbol={(sym) => setChartSymbol(`NASDAQ:${sym}`)}
          className="h-72"
        />
        <AlertCenter className="h-72" />
        <EconomicCalendar className="h-72" />
      </div>

      {/* ── Sector Heatmap ── */}
      <MarketHeatmap className="h-40" />

      {/* ── Market News ── */}
      <div className="kpi-card overflow-hidden max-h-64 overflow-y-auto">
        <NewsSentimentPanel symbols={['SPY', 'QQQ', 'AAPL', 'NVDA', 'META']} />
      </div>

      {/* ── Recent Trades (candlestick + buy/sell markers) ── */}
      <div className="kpi-card">
        <div className="flex items-center justify-between mb-3">
          <p className="section-header" style={{marginBottom:0}}>Recent Trades</p>
          <div className="flex gap-1.5">
            {TRADE_SYMBOLS.map(sym => (
              <button
                key={sym}
                onClick={() => setTradeSymbol(sym)}
                className="text-xs px-2.5 py-1 rounded transition-colors"
                style={{
                  background: tradeSymbol === sym ? 'var(--accent-dim)' : 'var(--surface2)',
                  border: `1px solid ${tradeSymbol === sym ? 'rgba(245,166,35,0.4)' : 'var(--border)'}`,
                  color: tradeSymbol === sym ? 'var(--accent)' : 'var(--muted)',
                }}
              >
                {sym}
              </button>
            ))}
          </div>
        </div>
        <div role="img" aria-label={`Trade marker chart for ${tradeSymbol}`}>
          <TradeMarkerChart symbol={tradeSymbol} height={360} />
        </div>
      </div>

      <style>{`@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
    </div>
  )
}
