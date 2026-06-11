import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span
      className="inline-block w-2.5 h-2.5 rounded-full mr-2"
      style={{ backgroundColor: ok ? '#00c853' : '#ff1744' }}
    />
  )
}

function HealthRow({ label, ok, detail }: { label: string; ok: boolean; detail?: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-[#1e1e1e] last:border-0">
      <div className="flex items-center">
        <StatusDot ok={ok} />
        <span className="text-xs text-[#e8e8e8]">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        {detail && <span className="text-xs text-[#888888]">{detail}</span>}
        <span
          className="text-xs font-medium px-2 py-0.5 rounded"
          style={{
            color: ok ? '#00c853' : '#ff1744',
            backgroundColor: ok ? '#00c85322' : '#ff174422',
          }}
        >
          {ok ? 'OK' : 'DOWN'}
        </span>
      </div>
    </div>
  )
}

// ── Strategy Health Table ────────────────────────────────────────────────────
type DeskFilter = 'all' | 'equity' | 'crypto' | 'options' | 'arbitrage'
type StatusFilter = 'all' | 'active' | 'paused'

function StrategyHealthTable() {
  const [deskFilter, setDeskFilter] = useState<DeskFilter>('all')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [sortCol, setSortCol] = useState<'name' | 'market_type' | 'risk_bucket'>('name')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')

  const { data: strategies, isLoading } = useQuery({
    queryKey: ['strategies-health'],
    queryFn: () => api.get('/strategies/').then(r => r.data),
    refetchInterval: 30_000,
  })
  const { data: activeStrategies } = useQuery({
    queryKey: ['active-strategies'],
    queryFn: () => api.get('/strategies/active').then(r => r.data).catch(() => []),
    refetchInterval: 15_000,
  })
  const { data: recentOrders } = useQuery({
    queryKey: ['recent-orders-health'],
    queryFn: () => api.get('/orders/?limit=100').then(r => r.data).catch(() => []),
    refetchInterval: 30_000,
  })

  const strats: any[] = Array.isArray(strategies) ? strategies : []
  const activeNames: string[] = Array.isArray(activeStrategies)
    ? activeStrategies.map((s: any) => s.name)
    : []
  const orders: any[] = Array.isArray(recentOrders) ? recentOrders : []

  // Build last signal map: strategy_name → last order created_at
  const lastSignalMap: Record<string, string> = {}
  for (const o of orders) {
    if (o.strategy_id && o.created_at) {
      // We only have strategy_id in order but try matching by name if available
      if (!lastSignalMap[o.strategy_id] || o.created_at > lastSignalMap[o.strategy_id]) {
        lastSignalMap[o.strategy_id] = o.created_at
      }
    }
  }

  function deskForStrategy(s: any): string {
    const rb = (s.risk_bucket || '').toLowerCase()
    const mt = (s.market_type || '').toLowerCase()
    if (rb.includes('arb')) return 'arbitrage'
    if (mt === 'crypto') return 'crypto'
    if (mt === 'options' || mt.includes('option')) return 'options'
    return 'equity'
  }

  function handleSort(col: typeof sortCol) {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('asc') }
  }

  const filtered = strats
    .filter(s => {
      if (deskFilter !== 'all' && deskForStrategy(s) !== deskFilter) return false
      if (statusFilter === 'active' && !s.is_enabled) return false
      if (statusFilter === 'paused' && s.is_enabled) return false
      return true
    })
    .sort((a, b) => {
      const va = a[sortCol] ?? ''
      const vb = b[sortCol] ?? ''
      const cmp = String(va).localeCompare(String(vb))
      return sortDir === 'asc' ? cmp : -cmp
    })

  const DESK_COLORS: Record<string, string> = {
    equity: '#00c853', crypto: '#f7931a', options: '#9c27b0', arbitrage: '#2196f3',
  }

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e1e1e]">
        <div>
          <h2 className="text-sm font-semibold text-[#e8e8e8]">Strategy Health</h2>
          <p className="text-xs text-[#555] mt-0.5">{filtered.length} of {strats.length} strategies</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Desk filter */}
          {(['all', 'equity', 'crypto', 'options', 'arbitrage'] as DeskFilter[]).map(d => (
            <button key={d} onClick={() => setDeskFilter(d)}
              className="px-2.5 py-1 rounded text-[10px] capitalize transition-colors"
              style={{
                background: deskFilter === d ? 'rgba(245,166,35,0.15)' : '#0a0a0a',
                color: deskFilter === d ? '#f5a623' : '#555',
                border: `1px solid ${deskFilter === d ? 'rgba(245,166,35,0.3)' : '#1e1e1e'}`,
              }}>
              {d}
            </button>
          ))}
          <div className="w-px h-4 bg-[#1e1e1e]" />
          {/* Status filter */}
          {(['all', 'active', 'paused'] as StatusFilter[]).map(s => (
            <button key={s} onClick={() => setStatusFilter(s)}
              className="px-2.5 py-1 rounded text-[10px] capitalize transition-colors"
              style={{
                background: statusFilter === s ? 'rgba(33,150,243,0.15)' : '#0a0a0a',
                color: statusFilter === s ? '#2196f3' : '#555',
                border: `1px solid ${statusFilter === s ? 'rgba(33,150,243,0.3)' : '#1e1e1e'}`,
              }}>
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-[#0a0a0a]">
            <tr className="text-xs text-[#888888]">
              <th className="text-left px-4 py-3 cursor-pointer hover:text-[#e8e8e8] transition-colors"
                onClick={() => handleSort('name')}>
                Strategy {sortCol === 'name' ? (sortDir === 'asc' ? '▲' : '▼') : ''}
              </th>
              <th className="text-left px-4 py-3 cursor-pointer hover:text-[#e8e8e8] transition-colors"
                onClick={() => handleSort('market_type')}>
                Desk {sortCol === 'market_type' ? (sortDir === 'asc' ? '▲' : '▼') : ''}
              </th>
              <th className="text-left px-4 py-3 cursor-pointer hover:text-[#e8e8e8] transition-colors"
                onClick={() => handleSort('risk_bucket')}>
                Bucket {sortCol === 'risk_bucket' ? (sortDir === 'asc' ? '▲' : '▼') : ''}
              </th>
              <th className="text-left px-4 py-3">Status</th>
              <th className="text-left px-4 py-3">Health</th>
              <th className="text-left px-4 py-3">Symbols</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && Array.from({ length: 4 }).map((_, i) => (
              <tr key={i} className="border-t border-[#1e1e1e]">
                {Array.from({ length: 6 }).map((_, j) => (
                  <td key={j} className="px-4 py-3">
                    <div className="h-4 bg-[#1e1e1e] rounded animate-pulse" />
                  </td>
                ))}
              </tr>
            ))}
            {!isLoading && filtered.map((s: any) => {
              const desk = deskForStrategy(s)
              const deskColor = DESK_COLORS[desk] ?? '#888'
              const isActive = s.is_enabled
              const isRunning = activeNames.includes(s.name)
              const lastSignal = lastSignalMap[s.id]
              const lastSignalLabel = lastSignal
                ? (() => {
                    const diff = Math.round((Date.now() - new Date(lastSignal).getTime()) / 60000)
                    return diff < 60 ? `${diff}m ago` : `${Math.round(diff / 60)}h ago`
                  })()
                : 'no signals'
              const healthOk = isActive && isRunning
              return (
                <tr key={s.id} className="border-t border-[#1e1e1e] hover:bg-[#0d0d0d] transition-colors">
                  <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8] font-medium">{s.display_name ?? s.name}</td>
                  <td className="px-4 py-3">
                    <span className="text-xs font-medium capitalize px-2 py-0.5 rounded"
                      style={{ color: deskColor, background: `${deskColor}18` }}>
                      {desk}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-[#555] capitalize">{s.risk_bucket}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                      isActive ? 'bg-[#00c853]/15 text-[#00c853]' : 'bg-[#1e1e1e] text-[#555]'}`}>
                      {isActive ? 'ACTIVE' : 'PAUSED'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ background: healthOk ? '#00c853' : isActive ? '#f5a623' : '#444',
                          boxShadow: healthOk ? '0 0 4px #00c853' : 'none' }} />
                      <span className="text-[10px] text-[#555]">{lastSignalLabel}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-[10px] text-[#555] font-mono">
                    {Array.isArray(s.symbols) && s.symbols.length > 0
                      ? s.symbols.slice(0, 3).join(', ') + (s.symbols.length > 3 ? ` +${s.symbols.length - 3}` : '')
                      : '—'}
                  </td>
                </tr>
              )
            })}
            {!isLoading && filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center text-xs text-[#555]">
                  No strategies match the selected filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function SystemMonitor() {
  const { data: agentStatus } = useQuery({
    queryKey: ['agents-status-monitor'],
    queryFn: () => api.get('/agents/status').then(r => r.data),
    refetchInterval: 5000,
  })

  const { data: quality } = useQuery({
    queryKey: ['improvements-quality'],
    queryFn: () => api.get('/improvements/quality').then(r => r.data),
    refetchInterval: 30000,
  })

  const { data: bestParams } = useQuery({
    queryKey: ['improvements-best-params'],
    queryFn: () => api.get('/improvements/best_params').then(r => r.data),
    refetchInterval: 60000,
  })

  const isRunning: boolean = agentStatus?.algo_agent?.running ?? false
  const top3 = agentStatus?.algo_agent?.top_3 ?? []

  // Flatten bestParams: { strategy: { param: value, ... }, ... } → rows
  const paramRows: { strategy: string; param: string; value: string }[] = []
  if (bestParams && typeof bestParams === 'object') {
    for (const [strategy, params] of Object.entries(bestParams)) {
      if (params && typeof params === 'object') {
        for (const [param, value] of Object.entries(params as Record<string, any>)) {
          paramRows.push({ strategy, param, value: String(value) })
        }
      }
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">System Monitor</h1>
        <span className="text-xs text-[#888888]">Agent refreshes every 5s</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Agent Status Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-4">
          <h2 className="text-xs text-[#888888] uppercase tracking-wider">AlgoAgent Status</h2>
          <div className="flex items-center gap-2">
            <StatusDot ok={isRunning} />
            <span
              className="text-sm font-bold"
              style={{ color: isRunning ? '#00c853' : '#ff1744' }}
            >
              {isRunning ? 'RUNNING' : 'STOPPED'}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-[#888888]">Total Runs</p>
              <p className="text-2xl font-bold text-[#f5a623]">{agentStatus?.algo_agent?.total_runs ?? 0}</p>
            </div>
            <div>
              <p className="text-xs text-[#888888]">Candidates</p>
              <p className="text-2xl font-bold text-[#2196f3]">{agentStatus?.algo_agent?.candidates ?? 0}</p>
            </div>
          </div>
          {top3.length > 0 && (
            <div>
              <p className="text-xs text-[#888888] mb-2">Top 3 Strategies (by UCB1 score)</p>
              <div className="space-y-1">
                {top3.map((row: any, i: number) => {
                  const label = typeof row === 'string'
                    ? row
                    : `${row.strategy}:${row.symbol}`
                  const sharpe = typeof row === 'object' && row.best_sharpe != null
                    ? row.best_sharpe.toFixed(2)
                    : null
                  return (
                    <div key={i} className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs" style={{ color: ['#FFD700', '#C0C0C0', '#CD7F32'][i] }}>
                          {['#1', '#2', '#3'][i]}
                        </span>
                        <span className="text-xs font-mono text-[#e8e8e8]">{label}</span>
                      </div>
                      {sharpe != null && (
                        <span className="text-[10px] font-mono text-[#888888]">
                          Sharpe {sharpe}
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* System Health Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <h2 className="text-xs text-[#888888] uppercase tracking-wider mb-3">System Health</h2>
          <HealthRow label="Database" ok={true} detail="Supabase" />
          <HealthRow label="Redis" ok={true} detail="Upstash" />
          <HealthRow label="AlgoAgent" ok={isRunning} detail={isRunning ? `${agentStatus?.algo_agent?.total_runs ?? 0} runs` : 'Not running'} />
          <HealthRow label="ResearchScientist" ok={agentStatus?.research_scientist?.running ?? false} detail={`${agentStatus?.research_scientist?.cycles_completed ?? 0} cycles`} />
          <HealthRow label="ModelingEngineer" ok={agentStatus?.modeling_engineer?.running ?? false} detail={`${agentStatus?.modeling_engineer?.cycles_completed ?? 0} cycles`} />
          <HealthRow label="QAMonitor" ok={agentStatus?.qa_monitor?.running ?? false} detail="Auto-fixing bugs" />
        </div>
      </div>

      {/* Code Quality Metrics */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-xs text-[#888888] uppercase tracking-wider mb-4">Code Quality Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Total Files', key: 'total_files', color: '#f5a623' },
            { label: 'Total Lines', key: 'total_lines', color: '#2196f3' },
            { label: 'Strategies', key: 'total_strategies', color: '#00c853' },
            { label: 'Tests', key: 'total_tests', color: '#9c27b0' },
          ].map(({ label, key, color }) => (
            <div key={key}>
              <p className="text-xs text-[#888888]">{label}</p>
              <p className="text-2xl font-bold mt-1" style={{ color }}>
                {quality?.[key] != null ? quality[key].toLocaleString() : '—'}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Strategy Health Table */}
      <StrategyHealthTable />

      {/* Best Params Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold">Best Parameters</h2>
          <p className="text-xs text-[#888888] mt-0.5">Optimal params per strategy from AlgoAgent</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-[#0a0a0a]">
              <tr className="text-xs text-[#888888]">
                <th className="text-left px-4 py-3">Strategy</th>
                <th className="text-left px-4 py-3">Parameter</th>
                <th className="text-left px-4 py-3">Value</th>
              </tr>
            </thead>
            <tbody>
              {paramRows.map((row, i) => (
                <tr key={i} className="border-t border-[#1e1e1e] hover:bg-[#0a0a0a] transition-colors">
                  <td className="px-4 py-3 text-xs font-mono text-[#f5a623]">{row.strategy}</td>
                  <td className="px-4 py-3 text-xs font-mono text-[#888888]">{row.param}</td>
                  <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{row.value}</td>
                </tr>
              ))}
              {paramRows.length === 0 && (
                <tr>
                  <td colSpan={3} className="px-4 py-8 text-center text-xs text-[#888888]">
                    No best params yet — AlgoAgent needs to complete optimization runs.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
