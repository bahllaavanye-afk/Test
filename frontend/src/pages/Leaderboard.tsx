import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

function FlatSparkline({ width = 80, height = 32 }: { width?: number; height?: number }) {
  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="#333" strokeWidth="1.5" strokeDasharray="3,3" />
    </svg>
  )
}

function RealSparkline({ values, color, height = 32, width = 80 }: { values: number[]; color: string; height?: number; width?: number }) {
  if (!values || values.length < 2) return <FlatSparkline width={width} height={height} />
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const step = width / (values.length - 1)
  const points = values.map((v, i) => {
    const x = i * step
    const y = height - ((v - min) / range) * (height - 4) - 2
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  const pathD = `M ${points.join(' L ')}`
  const fillD = `M 0,${height} L ${pathD.slice(2)} L ${((values.length - 1) * step).toFixed(1)},${height} Z`
  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id={`grad-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={fillD} fill={`url(#grad-${color.replace('#', '')})`} />
      <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function TypeBadge({ type }: { type: string }) {
  const isML = type === 'ml' || type === 'ML'
  return (
    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wide"
      style={{ background: isML ? '#9C27B022' : '#2196F322', color: isML ? '#CE93D8' : '#64B5F6', border: `1px solid ${isML ? '#9C27B0' : '#2196F3'}44` }}>
      {isML ? 'ML' : 'RULE'}
    </span>
  )
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button onClick={() => onChange(!checked)}
      className="relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200 focus:outline-none"
      style={{ background: checked ? '#00c853' : '#1e1e1e', border: `1px solid ${checked ? '#00c853' : '#333'}` }}>
      <span className="inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform duration-200"
        style={{ transform: checked ? 'translateX(18px)' : 'translateX(2px)', boxShadow: '0 1px 3px rgba(0,0,0,0.5)' }} />
    </button>
  )
}

function BotCard({ strategy, index, toggled, onToggle }: { strategy: any; index: number; toggled: boolean; onToggle: (id: string, val: boolean) => void }) {
  const isRunning = toggled
  const sharpe: number | null = strategy.sharpe_ratio ?? null
  const winRate: number | null = strategy.win_rate != null ? (strategy.win_rate > 1 ? strategy.win_rate : strategy.win_rate * 100) : null
  const totalPnl: number | null = strategy.total_pnl ?? null
  const pnlColor = totalPnl != null ? (totalPnl >= 0 ? '#00c853' : '#ff1744') : '#555555'
  const equityCurve: number[] = Array.isArray(strategy.equity_curve) ? strategy.equity_curve : []
  const stratType: string = strategy.type ?? 'rule'
  const lastSignal: string | null = strategy.last_signal ?? null

  return (
    <div className="group bg-[#111111] border border-[#1e1e1e] rounded-xl p-4 flex flex-col gap-3 cursor-pointer transition-all duration-200 hover:border-[#2e2e2e] hover:bg-[#141414] hover:shadow-lg hover:shadow-black/40 hover:-translate-y-0.5">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: isRunning ? '#00c853' : '#444', boxShadow: isRunning ? '0 0 5px #00c85380' : 'none' }} />
            <h3 className="text-sm font-bold text-[#e8e8e8] truncate">{strategy.name ?? `Strategy #${index + 1}`}</h3>
          </div>
          <div className="flex items-center gap-1.5 ml-4">
            <TypeBadge type={stratType} />
            <span className="text-[10px] px-1.5 py-0.5 rounded font-medium"
              style={{ background: isRunning ? '#00c85318' : '#1e1e1e', color: isRunning ? '#00c853' : '#555' }}>
              {isRunning ? 'RUNNING' : 'STOPPED'}
            </span>
          </div>
        </div>
        <Toggle checked={isRunning} onChange={v => onToggle(String(strategy.id ?? index), v)} />
      </div>
      <div className="flex items-end justify-between gap-2">
        <div className="flex-1">
          <RealSparkline values={equityCurve} color={pnlColor} height={36} width={120} />
        </div>
        <div className="text-right">
          <p className="text-xs text-[#888888]">Total P&L</p>
          {totalPnl != null ? (
            <p className="text-base font-bold font-mono" style={{ color: pnlColor }}>
              {totalPnl >= 0 ? '+' : ''}${Math.abs(totalPnl).toFixed(0)}
            </p>
          ) : <p className="text-xs text-[#555]">No data yet</p>}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 border-t border-[#1e1e1e] pt-3">
        <div className="text-center">
          <p className="text-[10px] text-[#555] uppercase mb-0.5">Sharpe</p>
          {sharpe != null ? (
            <p className="text-sm font-bold font-mono" style={{ color: sharpe >= 1.5 ? '#00c853' : sharpe >= 1.0 ? '#f5a623' : '#ff1744' }}>{sharpe.toFixed(2)}</p>
          ) : <p className="text-xs text-[#555]">No backtest data yet</p>}
        </div>
        <div className="text-center border-x border-[#1e1e1e]">
          <p className="text-[10px] text-[#555] uppercase mb-0.5">Win Rate</p>
          {winRate != null ? (
            <p className="text-sm font-bold font-mono" style={{ color: winRate >= 55 ? '#00c853' : winRate >= 45 ? '#f5a623' : '#ff1744' }}>{winRate.toFixed(1)}%</p>
          ) : <p className="text-xs text-[#555]">—</p>}
        </div>
        <div className="text-center">
          <p className="text-[10px] text-[#555] uppercase mb-0.5">Symbol</p>
          <p className="text-sm font-bold font-mono text-[#f5a623]">{strategy.symbol ?? '—'}</p>
        </div>
      </div>
      {lastSignal && (
        <div className="flex items-center gap-2 bg-[#0a0a0a] rounded-lg px-3 py-1.5 border border-[#1e1e1e]">
          <span className="text-[10px] text-[#555] uppercase flex-shrink-0">Last Signal</span>
          <span className="text-xs font-mono text-[#888888] truncate">{lastSignal}</span>
        </div>
      )}
    </div>
  )
}

export default function Leaderboard() {
  const [toggleMap, setToggleMap] = useState<Record<string, boolean>>({})
  const [filterType, setFilterType] = useState<'all' | 'ml' | 'rule'>('all')
  const [filterStatus, setFilterStatus] = useState<'all' | 'running' | 'stopped'>('all')
  const [sortBy, setSortBy] = useState<'sharpe' | 'pnl' | 'winrate'>('sharpe')

  const { data: strategies, isLoading, isError } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => api.get('/strategies/').then(r => r.data),
    refetchInterval: 15000,
  })

  const { data: agentStatus } = useQuery({
    queryKey: ['agents-status'],
    queryFn: () => api.get('/agents/status').then(r => r.data),
    refetchInterval: 10000,
  })

  const { data: history } = useQuery({
    queryKey: ['improvements-history'],
    queryFn: () => api.get('/improvements/history').then(r => r.data),
    refetchInterval: 30000,
  })

  function handleToggle(id: string, val: boolean) {
    setToggleMap(prev => ({ ...prev, [id]: val }))
  }

  const rawList: any[] = Array.isArray(strategies) ? strategies : []

  const filtered = rawList.filter(s => {
    const id = String(s.id ?? rawList.indexOf(s))
    const isRunning = id in toggleMap ? toggleMap[id] : (s.is_active || s.is_enabled || false)
    if (filterStatus === 'running' && !isRunning) return false
    if (filterStatus === 'stopped' && isRunning) return false
    const t: string = s.type ?? 'rule'
    if (filterType === 'ml' && t !== 'ml' && t !== 'ML') return false
    if (filterType === 'rule' && (t === 'ml' || t === 'ML')) return false
    return true
  })

  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === 'sharpe') return (b.sharpe_ratio ?? -Infinity) - (a.sharpe_ratio ?? -Infinity)
    if (sortBy === 'pnl') return Math.abs(b.total_pnl ?? 0) - Math.abs(a.total_pnl ?? 0)
    return (b.win_rate ?? 0) - (a.win_rate ?? 0)
  })

  const runningCount = rawList.filter((s, i) => {
    const id = String(s.id ?? i)
    return id in toggleMap ? toggleMap[id] : (s.is_active || s.is_enabled || false)
  }).length

  const recent: any[] = Array.isArray(history) ? history.slice(0, 8) : []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-[#e8e8e8]">Bot Cards</h1>
          <p className="text-xs text-[#555] mt-0.5">{rawList.length} strategies — {runningCount} running</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 bg-[#111111] border border-[#1e1e1e] rounded-lg p-1">
            {(['sharpe', 'pnl', 'winrate'] as const).map(s => (
              <button key={s} onClick={() => setSortBy(s)} className="text-xs px-2 py-1 rounded transition-colors"
                style={{ background: sortBy === s ? '#f5a623' : 'transparent', color: sortBy === s ? '#000' : '#888' }}>
                {s === 'sharpe' ? 'Sharpe' : s === 'pnl' ? 'P&L' : 'Win%'}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 bg-[#111111] border border-[#1e1e1e] rounded-lg p-1">
            {(['all', 'ml', 'rule'] as const).map(f => (
              <button key={f} onClick={() => setFilterType(f)} className="text-xs px-2 py-1 rounded transition-colors capitalize"
                style={{ background: filterType === f ? '#2196F3' : 'transparent', color: filterType === f ? '#fff' : '#888' }}>
                {f === 'all' ? 'All' : f === 'ml' ? 'ML' : 'Rule'}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 bg-[#111111] border border-[#1e1e1e] rounded-lg p-1">
            {(['all', 'running', 'stopped'] as const).map(f => (
              <button key={f} onClick={() => setFilterStatus(f)} className="text-xs px-2 py-1 rounded transition-colors capitalize"
                style={{ background: filterStatus === f ? '#00c853' : 'transparent', color: filterStatus === f ? '#000' : '#888' }}>
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg px-4 py-3 flex items-center gap-6">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full"
            style={{ background: agentStatus?.running ? '#00c853' : '#ff1744', boxShadow: agentStatus?.running ? '0 0 6px #00c853' : 'none' }} />
          <span className="text-xs font-bold" style={{ color: agentStatus?.running ? '#00c853' : '#ff1744' }}>
            AlgoAgent {agentStatus?.running ? 'ACTIVE' : 'IDLE'}
          </span>
        </div>
        <div className="h-4 w-px bg-[#1e1e1e]" />
        <div><span className="text-xs text-[#555]">Total Runs </span><span className="text-sm font-bold text-[#f5a623]">{agentStatus?.total_runs ?? 0}</span></div>
        <div><span className="text-xs text-[#555]">Candidates </span><span className="text-sm font-bold text-[#2196f3]">{agentStatus?.candidates ?? 0}</span></div>
        <div className="ml-auto text-xs text-[#444]">Refreshes every 15s</div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4 space-y-3 animate-pulse">
              <div className="h-4 bg-[#1e1e1e] rounded w-3/4" />
              <div className="h-9 bg-[#1e1e1e] rounded" />
              <div className="grid grid-cols-3 gap-2">
                <div className="h-8 bg-[#1e1e1e] rounded" /><div className="h-8 bg-[#1e1e1e] rounded" /><div className="h-8 bg-[#1e1e1e] rounded" />
              </div>
            </div>
          ))}
        </div>
      ) : isError ? (
        <div className="flex flex-col items-center justify-center py-16 text-center space-y-3">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ff1744" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
          </svg>
          <p className="text-sm text-[#ff1744]">Failed to load strategies</p>
          <p className="text-xs text-[#555]">Check that the backend API is running at /api/v1/strategies/</p>
        </div>
      ) : rawList.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center space-y-3 bg-[#111111] border border-[#1e1e1e] rounded-xl">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
            <rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/>
          </svg>
          <p className="text-sm text-[#888888] font-medium">No strategies yet</p>
          <p className="text-xs text-[#555] max-w-xs">Strategies will appear here once registered with the backend. Add API keys in Settings and run a backtest to get started.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {sorted.map((s, i) => {
            const id = String(s.id ?? i)
            const defaultOn = s.is_active || s.is_enabled || false
            const toggled = id in toggleMap ? toggleMap[id] : defaultOn
            return <BotCard key={id} strategy={s} index={rawList.indexOf(s)} toggled={toggled} onToggle={handleToggle} />
          })}
          {sorted.length === 0 && rawList.length > 0 && (
            <div className="col-span-full text-center py-16 text-[#444] text-sm">No strategies match the current filter.</div>
          )}
        </div>
      )}

      {recent.length > 0 && (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
          <div className="p-3 border-b border-[#1e1e1e] flex items-center justify-between">
            <h2 className="text-sm font-semibold">Recent Improvements</h2>
            <span className="text-xs text-[#555]">Last {recent.length} upgrades</span>
          </div>
          <table className="w-full">
            <thead className="bg-[#0a0a0a]">
              <tr className="text-xs text-[#555]">
                {['Time', 'Strategy', 'Sharpe Change', 'Params'].map(h => <th key={h} className="text-left px-4 py-2.5">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {recent.map((h: any, i: number) => (
                <tr key={i} className="border-t border-[#1e1e1e] hover:bg-[#0d0d0d] transition-colors">
                  <td className="px-4 py-2.5 text-xs text-[#555] whitespace-nowrap">{h.timestamp ? new Date(h.timestamp).toLocaleString() : '—'}</td>
                  <td className="px-4 py-2.5 text-xs font-mono text-[#e8e8e8]">{h.strategy ?? '—'}</td>
                  <td className="px-4 py-2.5 text-xs">
                    <span className="text-[#555]">{h.old_sharpe?.toFixed(3) ?? '—'}</span>
                    <span className="text-[#333] mx-1">to</span>
                    <span className="text-[#00c853] font-bold">{h.new_sharpe?.toFixed(3) ?? '—'}</span>
                    {h.old_sharpe != null && h.new_sharpe != null && (
                      <span className="ml-1.5 text-[#00c853] text-[10px]">(+{(h.new_sharpe - h.old_sharpe).toFixed(3)})</span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-[#555]">
                    {Array.isArray(h.params_changed) ? h.params_changed.join(', ')
                      : typeof h.params_changed === 'object' && h.params_changed ? Object.keys(h.params_changed).join(', ')
                      : h.params_changed ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
