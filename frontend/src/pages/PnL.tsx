import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

function KpiCard({ label, value, color = '#f5a623', sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 transition-all duration-200 hover:border-[#2e2e2e] hover:bg-[#141414]">
      <p className="text-xs text-[#888888] uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold mt-1 transition-colors" style={{ color }}>{value}</p>
      {sub && <p className="text-xs text-[#888888] mt-1">{sub}</p>}
    </div>
  )
}

function SideChip({ side }: { side: string }) {
  const isBuy = side?.toLowerCase() === 'buy'
  return (
    <span
      className="px-2 py-0.5 rounded text-xs font-bold uppercase"
      style={{
        color: isBuy ? '#00c853' : '#ff1744',
        background: isBuy ? '#00c85318' : '#ff174418',
        border: `1px solid ${isBuy ? '#00c85340' : '#ff174440'}`,
      }}
    >
      {side?.toUpperCase() ?? '—'}
    </span>
  )
}

function PnlCell({ value }: { value: number | null | undefined }) {
  if (value == null) return <span className="text-[#555]">—</span>
  const color = value >= 0 ? '#00c853' : '#ff1744'
  return <span style={{ color }} className="font-bold font-mono">{value >= 0 ? '+' : ''}${value.toFixed(2)}</span>
}

function PctCell({ value }: { value: number | null | undefined }) {
  if (value == null) return <span className="text-[#555]">—</span>
  const color = value >= 0 ? '#00c853' : '#ff1744'
  return <span style={{ color }} className="font-mono">{value >= 0 ? '+' : ''}{value.toFixed(2)}%</span>
}

function exportToCSV(trades: any[]) {
  const headers = ['Date', 'Symbol', 'Strategy', 'Side', 'Entry', 'Exit', 'P&L ($)', 'P&L (%)', 'Hold Time', 'Algo']
  const rows = trades.map(t => [
    t.filled_at ? new Date(t.filled_at).toISOString() : '',
    t.symbol ?? '',
    t.strategy ?? '',
    t.side ?? '',
    t.entry_price ?? t.avg_fill_price ?? '',
    t.exit_price ?? '',
    t.realized_pnl ?? '',
    t.pnl_pct ?? '',
    t.hold_time ?? '',
    t.execution_algo ?? t.algo ?? '',
  ])
  const csv = [headers, ...rows].map(r => r.map(String).map(v => `"${v.replace(/"/g, '""')}"`).join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `quantedge-trades-${new Date().toISOString().split('T')[0]}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

type SortKey = 'date' | 'symbol' | 'pnl' | 'pnlpct' | 'hold'
type SortDir = 'asc' | 'desc'

export default function PnL() {
  const [sortKey, setSortKey] = useState<SortKey>('date')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [filterSide, setFilterSide] = useState<'all' | 'buy' | 'sell'>('all')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const PAGE_SIZE = 25

  const { data: perf, isLoading: perfLoading } = useQuery({
    queryKey: ['performance'],
    queryFn: () => api.get('/analytics/performance').then(r => r.data),
    refetchInterval: 10000,
  })

  const { data: slippage } = useQuery({
    queryKey: ['slippage'],
    queryFn: () => api.get('/analytics/slippage').then(r => r.data),
  })

  const { data: trades, isLoading: tradesLoading } = useQuery({
    queryKey: ['trades'],
    queryFn: () => api.get('/trades/').then(r => r.data),
    refetchInterval: 10000,
  })

  const tradeList: any[] = trades ?? []
  const winCount = tradeList.filter((t: any) => (t.realized_pnl ?? 0) > 0).length
  const lossCount = tradeList.filter((t: any) => (t.realized_pnl ?? 0) < 0).length
  const winRate = tradeList.length > 0 ? ((winCount / tradeList.length) * 100).toFixed(1) : '0.0'
  const totalPnl = perf?.total_pnl ?? tradeList.reduce((acc, t) => acc + (t.realized_pnl ?? 0), 0)
  const avgPnl = tradeList.length > 0 ? totalPnl / tradeList.length : 0
  const bestTrade = tradeList.reduce((best, t) => Math.max(best, t.realized_pnl ?? 0), -Infinity)
  const worstTrade = tradeList.reduce((worst, t) => Math.min(worst, t.realized_pnl ?? 0), Infinity)

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
    setPage(0)
  }

  const filteredTrades = useMemo(() => {
    let list = [...tradeList]
    if (filterSide !== 'all') list = list.filter(t => t.side?.toLowerCase() === filterSide)
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      list = list.filter(t => (t.symbol ?? '').toLowerCase().includes(q) || (t.strategy ?? '').toLowerCase().includes(q))
    }
    list.sort((a, b) => {
      let av: any, bv: any
      if (sortKey === 'date') { av = new Date(a.filled_at ?? 0).getTime(); bv = new Date(b.filled_at ?? 0).getTime() }
      else if (sortKey === 'symbol') { av = a.symbol ?? ''; bv = b.symbol ?? '' }
      else if (sortKey === 'pnl') { av = a.realized_pnl ?? 0; bv = b.realized_pnl ?? 0 }
      else if (sortKey === 'pnlpct') { av = a.pnl_pct ?? 0; bv = b.pnl_pct ?? 0 }
      else if (sortKey === 'hold') { av = a.hold_seconds ?? 0; bv = b.hold_seconds ?? 0 }
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })
    return list
  }, [tradeList, filterSide, search, sortKey, sortDir])

  const pagedTrades = filteredTrades.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const totalPages = Math.ceil(filteredTrades.length / PAGE_SIZE)

  function SortHeader({ label, col }: { label: string; col: SortKey }) {
    const active = sortKey === col
    return (
      <th
        className="text-left px-4 py-3 cursor-pointer select-none group"
        onClick={() => handleSort(col)}
      >
        <span className="flex items-center gap-1">
          <span className={active ? 'text-[#f5a623]' : 'text-[#555] group-hover:text-[#888]'} style={{ transition: 'color 0.15s' }}>{label}</span>
          <span className={active ? 'text-[#f5a623]' : 'text-[#333]'}>
            {active ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}
          </span>
        </span>
      </th>
    )
  }

  function formatHoldTime(secs: number | undefined | null) {
    if (secs == null) return '—'
    if (secs < 60) return `${secs}s`
    if (secs < 3600) return `${Math.floor(secs / 60)}m`
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`
    return `${Math.floor(secs / 86400)}d`
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-[#e8e8e8]">P&amp;L Dashboard</h1>
          <p className="text-xs text-[#555] mt-0.5">{tradeList.length} total trades • Refreshes every 10s</p>
        </div>
        <button
          onClick={() => exportToCSV(filteredTrades)}
          disabled={filteredTrades.length === 0}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200
            disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 active:scale-95"
          style={{ background: '#f5a623', color: '#000' }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
          </svg>
          Export CSV
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {perfLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 animate-pulse">
              <div className="h-3 bg-[#1e1e1e] rounded w-2/3 mb-3" />
              <div className="h-8 bg-[#1e1e1e] rounded w-1/2" />
            </div>
          ))
        ) : (
          <>
            <KpiCard
              label="Total P&L"
              value={`${totalPnl >= 0 ? '+' : ''}$${Math.abs(totalPnl).toFixed(2)}`}
              color={totalPnl >= 0 ? '#00c853' : '#ff1744'}
              sub={`${tradeList.length} total trades`}
            />
            <KpiCard
              label="Win Rate"
              value={`${winRate}%`}
              color={parseFloat(winRate) >= 50 ? '#00c853' : '#ff1744'}
              sub={`${winCount}W / ${lossCount}L`}
            />
            <KpiCard
              label="Avg P&L / Trade"
              value={`${avgPnl >= 0 ? '+' : ''}$${Math.abs(avgPnl).toFixed(2)}`}
              color={avgPnl >= 0 ? '#00c853' : '#ff1744'}
            />
            <KpiCard
              label="Best Trade"
              value={bestTrade === -Infinity ? '—' : `+$${bestTrade.toFixed(2)}`}
              color="#00c853"
              sub={worstTrade === Infinity ? '' : `Worst: -$${Math.abs(worstTrade).toFixed(2)}`}
            />
          </>
        )}
      </div>

      {/* Slippage by Algo */}
      {slippage && slippage.length > 0 && (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
          <div className="p-3 border-b border-[#1e1e1e]">
            <h2 className="text-sm font-semibold">Slippage by Execution Algorithm</h2>
          </div>
          <table className="w-full">
            <thead className="bg-[#0a0a0a]">
              <tr className="text-xs text-[#555]">
                <th className="text-left px-4 py-3">Algorithm</th>
                <th className="text-right px-4 py-3">Avg Slippage (bps)</th>
                <th className="text-right px-4 py-3">Orders</th>
              </tr>
            </thead>
            <tbody>
              {slippage.map((s: any) => (
                <tr key={s.execution_algo ?? s.algo} className="border-t border-[#1e1e1e] hover:bg-[#0a0a0a] transition-colors">
                  <td className="px-4 py-2.5 text-xs font-mono text-[#e8e8e8]">{s.execution_algo ?? s.algo ?? '—'}</td>
                  <td className="px-4 py-2.5 text-xs text-right text-[#f5a623] font-mono">{s.avg_bps ?? '—'}</td>
                  <td className="px-4 py-2.5 text-xs text-right text-[#555]">{s.count ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Trade History Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        {/* Table header with filters */}
        <div className="p-3 border-b border-[#1e1e1e] flex items-center gap-3 flex-wrap">
          <h2 className="text-sm font-semibold">Trade History</h2>
          <div className="flex-1" />
          {/* Search */}
          <div className="relative">
            <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#555]" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
            <input
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(0) }}
              placeholder="Search symbol/strategy…"
              className="bg-[#0a0a0a] border border-[#1e1e1e] rounded px-3 py-1.5 pl-7 text-xs text-[#e8e8e8] w-44 focus:outline-none focus:border-[#f5a623]/40 transition-colors"
            />
          </div>
          {/* Side filter */}
          <div className="flex items-center gap-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-1">
            {(['all', 'buy', 'sell'] as const).map(f => (
              <button
                key={f}
                onClick={() => { setFilterSide(f); setPage(0) }}
                className="text-xs px-2 py-0.5 rounded capitalize transition-colors"
                style={{
                  background: filterSide === f ? (f === 'buy' ? '#00c853' : f === 'sell' ? '#ff1744' : '#f5a623') : 'transparent',
                  color: filterSide === f ? (f === 'all' ? '#000' : '#fff') : '#555',
                }}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {tradesLoading ? (
          <div className="p-8 space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-8 bg-[#1e1e1e] rounded animate-pulse" />
            ))}
          </div>
        ) : filteredTrades.length === 0 ? (
          <div className="py-16 text-center">
            <div className="w-12 h-12 rounded-full bg-[#1e1e1e] flex items-center justify-center mx-auto mb-3">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/>
                <path d="M12 8v4M12 16h.01"/>
              </svg>
            </div>
            <p className="text-[#888] text-sm font-medium">
              {tradeList.length === 0 ? 'Paper trading active' : 'No trades match your filter'}
            </p>
            <p className="text-[#555] text-xs mt-1">
              {tradeList.length === 0
                ? 'First trade will appear here once strategies generate a signal'
                : 'Try adjusting your search or filter settings'}
            </p>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[800px]">
                <thead className="bg-[#0a0a0a]">
                  <tr className="text-xs">
                    <SortHeader label="Date" col="date" />
                    <SortHeader label="Symbol" col="symbol" />
                    <th className="text-left px-4 py-3 text-[#555]">Strategy</th>
                    <th className="text-left px-4 py-3 text-[#555]">Side</th>
                    <th className="text-right px-4 py-3 text-[#555]">Entry</th>
                    <th className="text-right px-4 py-3 text-[#555]">Exit</th>
                    <SortHeader label="P&L ($)" col="pnl" />
                    <SortHeader label="P&L (%)" col="pnlpct" />
                    <SortHeader label="Hold Time" col="hold" />
                    <th className="text-left px-4 py-3 text-[#555]">Algo</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedTrades.map((t: any, i: number) => (
                    <tr
                      key={t.id ?? i}
                      className="border-t border-[#1e1e1e] hover:bg-[#0d0d0d] transition-colors"
                    >
                      <td className="px-4 py-2.5 text-xs text-[#555] whitespace-nowrap">
                        {t.filled_at ? new Date(t.filled_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-xs font-mono font-bold text-[#e8e8e8]">{t.symbol ?? '—'}</td>
                      <td className="px-4 py-2.5 text-xs text-[#888] truncate max-w-[120px]">{t.strategy ?? '—'}</td>
                      <td className="px-4 py-2.5">
                        <SideChip side={t.side ?? ''} />
                      </td>
                      <td className="px-4 py-2.5 text-xs text-right font-mono text-[#888]">
                        {t.entry_price ?? t.avg_fill_price ? `$${(t.entry_price ?? t.avg_fill_price).toFixed(2)}` : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-right font-mono text-[#888]">
                        {t.exit_price ? `$${t.exit_price.toFixed(2)}` : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-right">
                        <PnlCell value={t.realized_pnl} />
                      </td>
                      <td className="px-4 py-2.5 text-xs text-right">
                        <PctCell value={t.pnl_pct} />
                      </td>
                      <td className="px-4 py-2.5 text-xs text-right font-mono text-[#888]">
                        {formatHoldTime(t.hold_seconds ?? t.hold_time)}
                      </td>
                      <td className="px-4 py-2.5 text-xs font-mono text-[#555] truncate max-w-[100px]">
                        {t.execution_algo ?? t.algo ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="border-t border-[#1e1e1e] px-4 py-2.5 flex items-center justify-between">
                <span className="text-xs text-[#555]">
                  Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filteredTrades.length)} of {filteredTrades.length}
                </span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setPage(p => Math.max(0, p - 1))}
                    disabled={page === 0}
                    className="px-2.5 py-1 text-xs rounded bg-[#1e1e1e] text-[#888] disabled:opacity-40 hover:bg-[#2e2e2e] transition-colors"
                  >
                    ←
                  </button>
                  {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                    const p = totalPages <= 7 ? i : (page <= 3 ? i : page - 3 + i)
                    if (p >= totalPages) return null
                    return (
                      <button
                        key={p}
                        onClick={() => setPage(p)}
                        className="w-7 h-7 text-xs rounded transition-colors"
                        style={{
                          background: page === p ? '#f5a623' : '#1e1e1e',
                          color: page === p ? '#000' : '#888',
                        }}
                      >
                        {p + 1}
                      </button>
                    )
                  })}
                  <button
                    onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                    disabled={page >= totalPages - 1}
                    className="px-2.5 py-1 text-xs rounded bg-[#1e1e1e] text-[#888] disabled:opacity-40 hover:bg-[#2e2e2e] transition-colors"
                  >
                    →
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
