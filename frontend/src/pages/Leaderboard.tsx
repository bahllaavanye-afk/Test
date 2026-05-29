import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronUp, ChevronDown, ChevronsUpDown, TrendingUp, Activity, DollarSign, Award, SlidersHorizontal, X } from 'lucide-react'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface MetricsBlock {
  total_return: number | null
  annualized_return: number | null
  sharpe_ratio: number | null
  sortino_ratio: number | null
  calmar_ratio: number | null
  max_drawdown: number | null
  win_rate: number | null
  profit_factor: number | null
  total_trades: number | null
  avg_trade_pnl: number | null
  last_updated: string | null
}

interface LeaderboardEntry {
  id: string
  name: string
  display_name: string | null
  market_type: string
  strategy_type: string
  risk_bucket: string
  is_enabled: boolean
  symbols: string[]
  backtest: MetricsBlock | null
  paper: MetricsBlock | null
  live: MetricsBlock | null
  forward_test: MetricsBlock | null
  vs_spy_sharpe: number | null
  ml_improvement_pct: number | null
  rank: number
}

interface LeaderboardSummary {
  total_strategies: number
  running_count: number
  avg_sharpe: number | null
  best_strategy: string | null
  total_paper_pnl: number
  total_live_pnl: number
}

// ─── Formatting helpers ───────────────────────────────────────────────────────

const pct = (v: number | null | undefined, decimals = 1): string =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(decimals)}%`

const num = (v: number | null | undefined, decimals = 2): string =>
  v == null ? '—' : v.toFixed(decimals)

const dollar = (v: number | null | undefined): string => {
  if (v == null) return '—'
  const abs = Math.abs(v)
  const sign = v >= 0 ? '+' : '-'
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}K`
  return `${sign}$${abs.toFixed(0)}`
}

const int = (v: number | null | undefined): string =>
  v == null ? '—' : v.toLocaleString()

// ─── Color helpers ────────────────────────────────────────────────────────────

const sharpeColor = (v: number | null): string => {
  if (v == null) return '#555555'
  if (v >= 2.0) return '#00ff88'
  if (v >= 1.5) return '#00c853'
  if (v >= 1.0) return '#f5a623'
  return '#ff1744'
}

const winRateColor = (v: number | null): string => {
  if (v == null) return '#555555'
  if (v >= 0.6) return '#00c853'
  if (v >= 0.5) return '#f5a623'
  return '#ff1744'
}

const drawdownColor = (v: number | null): string => {
  if (v == null) return '#555555'
  const abs = Math.abs(v)
  if (abs <= 0.05) return '#00c853'
  if (abs <= 0.15) return '#f5a623'
  return '#ff1744'
}

const returnColor = (v: number | null): string => {
  if (v == null) return '#555555'
  return v >= 0 ? '#00c853' : '#ff1744'
}

// ─── Small components ─────────────────────────────────────────────────────────

function TypeBadge({ type }: { type: string }) {
  const isML = type === 'ml_enhanced'
  return (
    <span className="text-[9px] font-bold px-1 py-0.5 rounded uppercase tracking-wider flex-shrink-0"
      style={{
        background: isML ? '#7c3aed22' : '#1d4ed822',
        color: isML ? '#a78bfa' : '#60a5fa',
        border: `1px solid ${isML ? '#7c3aed44' : '#1d4ed844'}`,
      }}>
      {isML ? 'ML' : 'RULE'}
    </span>
  )
}

function MarketBadge({ market }: { market: string }) {
  const config: Record<string, { color: string; label: string }> = {
    equity:     { color: '#2196F3', label: 'EQ' },
    crypto:     { color: '#f59e0b', label: 'CRYPTO' },
    polymarket: { color: '#8b5cf6', label: 'POLY' },
  }
  const c = config[market] ?? { color: '#555', label: market.toUpperCase().slice(0, 5) }
  return (
    <span className="text-[9px] font-bold px-1 py-0.5 rounded uppercase tracking-wider flex-shrink-0"
      style={{ background: `${c.color}18`, color: c.color, border: `1px solid ${c.color}33` }}>
      {c.label}
    </span>
  )
}

function StatusDot({ enabled }: { enabled: boolean }) {
  return (
    <span className="flex items-center gap-1">
      <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
        style={{
          background: enabled ? '#00c853' : '#333',
          boxShadow: enabled ? '0 0 5px #00c85380' : 'none',
        }} />
      <span className="text-[10px] font-medium" style={{ color: enabled ? '#00c853' : '#444' }}>
        {enabled ? 'ON' : 'OFF'}
      </span>
    </span>
  )
}

function Cell({ value, color, bold = false }: { value: string; color?: string; bold?: boolean }) {
  return (
    <td className="px-3 py-2.5 text-xs font-mono whitespace-nowrap"
      style={{ color: color ?? '#888888', fontWeight: bold ? 700 : 400 }}>
      {value}
    </td>
  )
}

function SortHeader({
  label, col, sortBy, sortDir, onClick
}: {
  label: string
  col: string
  sortBy: string
  sortDir: 'asc' | 'desc'
  onClick: (col: string) => void
}) {
  const active = sortBy === col
  return (
    <th
      className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider cursor-pointer select-none whitespace-nowrap"
      style={{ color: active ? '#f5a623' : '#555555' }}
      onClick={() => onClick(col)}>
      <span className="flex items-center gap-1">
        {label}
        {active
          ? (sortDir === 'desc' ? <ChevronDown size={10} /> : <ChevronUp size={10} />)
          : <ChevronsUpDown size={10} className="opacity-30" />}
      </span>
    </th>
  )
}

function SummaryKPI({ icon: Icon, label, value, color = '#e8e8e8' }: {
  icon: React.ElementType, label: string, value: string, color?: string
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-1.5">
        <Icon size={12} className="text-[#555]" />
        <span className="text-[10px] text-[#555] uppercase tracking-wider">{label}</span>
      </div>
      <span className="text-base font-bold font-mono" style={{ color }}>{value}</span>
    </div>
  )
}

// ─── Column group header ──────────────────────────────────────────────────────

function ColGroup({ label, span, color }: { label: string; span: number; color: string }) {
  return (
    <th colSpan={span}
      className="px-3 py-1.5 text-[9px] font-bold uppercase tracking-widest text-center border-b"
      style={{ color, borderColor: `${color}33`, background: `${color}08` }}>
      {label}
    </th>
  )
}

// ─── Sort logic helpers ───────────────────────────────────────────────────────

type SortKey = string

function getVal(entry: LeaderboardEntry, col: SortKey): number {
  const [section, field] = col.split('.')
  const block: MetricsBlock | null =
    section === 'bt' ? entry.backtest :
    section === 'paper' ? entry.paper :
    section === 'live' ? entry.live :
    section === 'ft' ? entry.forward_test : null

  if (!block) return -Infinity

  switch (field) {
    case 'return':  return block.total_return ?? -Infinity
    case 'ann_ret': return block.annualized_return ?? -Infinity
    case 'sharpe':  return block.sharpe_ratio ?? -Infinity
    case 'sortino': return block.sortino_ratio ?? -Infinity
    case 'calmar':  return block.calmar_ratio ?? -Infinity
    case 'maxdd':   return block.max_drawdown ?? Infinity   // lower is worse
    case 'winrate': return block.win_rate ?? -Infinity
    case 'pf':      return block.profit_factor ?? -Infinity
    case 'trades':  return block.total_trades ?? -Infinity
    case 'avgpnl':  return block.avg_trade_pnl ?? -Infinity
    default: return -Infinity
  }
}

// ─── Column filter definitions ────────────────────────────────────────────────

interface ColFilter {
  key: SortKey
  label: string
  op: '>=' | '<='
  value: number
}

const PRESET_FILTERS: ColFilter[] = [
  { key: 'bt.sharpe',  label: 'Sharpe ≥ 1.5',   op: '>=', value: 1.5  },
  { key: 'bt.sharpe',  label: 'Sharpe ≥ 2.0',   op: '>=', value: 2.0  },
  { key: 'bt.winrate', label: 'Win% ≥ 55%',      op: '>=', value: 0.55 },
  { key: 'bt.winrate', label: 'Win% ≥ 60%',      op: '>=', value: 0.60 },
  { key: 'bt.maxdd',   label: 'MaxDD ≤ −10%',    op: '<=', value: -0.10},
  { key: 'bt.maxdd',   label: 'MaxDD ≤ −15%',    op: '<=', value: -0.15},
  { key: 'bt.calmar',  label: 'Calmar ≥ 1.0',    op: '>=', value: 1.0  },
  { key: 'bt.pf',      label: 'Profit Factor ≥ 1.5', op: '>=', value: 1.5 },
]

function applyColFilter(entry: LeaderboardEntry, f: ColFilter): boolean {
  const v = getVal(entry, f.key)
  if (v === -Infinity || v === Infinity) return false
  return f.op === '>=' ? v >= f.value : v <= f.value
}

// ─── Main Page ────────────────────────────────────────────────────────────────

type FilterType = 'all' | 'ml' | 'rule' | 'equity' | 'crypto' | 'enabled'

export default function Leaderboard() {
  const [sortBy, setSortBy] = useState<SortKey>('bt.sharpe')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [filter, setFilter] = useState<FilterType>('all')
  const [colFilters, setColFilters] = useState<ColFilter[]>([])
  const [showFilters, setShowFilters] = useState(false)

  const { data: entries = [], isLoading, isError } = useQuery<LeaderboardEntry[]>({
    queryKey: ['leaderboard'],
    queryFn: () => api.get('/leaderboard/').then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: summary } = useQuery<LeaderboardSummary>({
    queryKey: ['leaderboard-summary'],
    queryFn: () => api.get('/leaderboard/summary').then(r => r.data),
    refetchInterval: 30_000,
  })

  function handleSort(col: SortKey) {
    if (sortBy === col) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortBy(col)
      setSortDir('desc')
    }
  }

  function toggleColFilter(f: ColFilter) {
    setColFilters(prev => {
      const exists = prev.some(p => p.label === f.label)
      return exists ? prev.filter(p => p.label !== f.label) : [...prev, f]
    })
  }

  const filtered = useMemo(() => {
    return entries.filter(e => {
      // Type / market filters
      if (filter === 'ml') return e.strategy_type === 'ml_enhanced'
      if (filter === 'rule') return e.strategy_type !== 'ml_enhanced'
      if (filter === 'equity') return e.market_type === 'equity'
      if (filter === 'crypto') return e.market_type === 'crypto'
      if (filter === 'enabled') return e.is_enabled
      return true
    }).filter(e => colFilters.every(f => applyColFilter(e, f)))
  }, [entries, filter, colFilters])

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      if (sortBy === 'rank') {
        return sortDir === 'asc' ? a.rank - b.rank : b.rank - a.rank
      }
      const va = getVal(a, sortBy)
      const vb = getVal(b, sortBy)
      // For maxdd: asc means less drawdown first (better)
      if (sortBy.endsWith('.maxdd')) {
        return sortDir === 'asc' ? (va - vb) : (vb - va)
      }
      return sortDir === 'desc' ? (vb - va) : (va - vb)
    })
  }, [filtered, sortBy, sortDir])

  const FILTERS: { key: FilterType; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'ml', label: 'ML Enhanced' },
    { key: 'rule', label: 'Rule-Based' },
    { key: 'equity', label: 'Equity' },
    { key: 'crypto', label: 'Crypto' },
    { key: 'enabled', label: 'Active Only' },
  ]

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-lg font-bold text-[#e8e8e8]">Strategy Leaderboard</h1>
          <p className="text-xs text-[#555] mt-0.5">Backtest · Paper · Live · Forward Test — all modes compared</p>
        </div>
        <div className="flex items-center gap-1 bg-[#111111] border border-[#1e1e1e] rounded-lg p-1 flex-wrap">
          {FILTERS.map(f => (
            <button key={f.key} onClick={() => setFilter(f.key)}
              className="text-xs px-2.5 py-1 rounded transition-colors whitespace-nowrap"
              style={{
                background: filter === f.key ? '#f5a623' : 'transparent',
                color: filter === f.key ? '#000' : '#888',
              }}>
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Column value filters */}
      <div className="flex items-start gap-2 flex-wrap">
        <button
          onClick={() => setShowFilters(v => !v)}
          className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded border transition-colors"
          style={{
            background: showFilters ? '#f5a62320' : 'transparent',
            color: showFilters ? '#f5a623' : '#888',
            borderColor: showFilters ? '#f5a62344' : '#1e1e1e',
          }}>
          <SlidersHorizontal size={12} />
          Column Filters
          {colFilters.length > 0 && (
            <span className="ml-1 px-1 rounded text-[9px] font-bold bg-[#f5a623] text-black">{colFilters.length}</span>
          )}
        </button>

        {showFilters && (
          <div className="flex flex-wrap gap-1.5">
            {PRESET_FILTERS.map(f => {
              const active = colFilters.some(c => c.label === f.label)
              return (
                <button key={f.label} onClick={() => toggleColFilter(f)}
                  className="text-[10px] px-2 py-1 rounded transition-colors"
                  style={{
                    background: active ? '#f5a62320' : '#111111',
                    color: active ? '#f5a623' : '#888',
                    border: `1px solid ${active ? '#f5a62344' : '#1e1e1e'}`,
                  }}>
                  {f.label}
                </button>
              )
            })}
          </div>
        )}

        {colFilters.length > 0 && (
          <button onClick={() => setColFilters([])}
            className="flex items-center gap-1 text-[10px] px-2 py-1 rounded text-[#ff1744] border border-[#ff174433] hover:bg-[#ff174410] transition-colors">
            <X size={10} /> Clear filters
          </button>
        )}
      </div>

      {/* Summary KPIs */}
      {summary && (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg px-5 py-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <SummaryKPI icon={Activity} label="Strategies" value={String(summary.total_strategies)} />
          <SummaryKPI icon={Activity} label="Active" value={String(summary.running_count)} color="#00c853" />
          <SummaryKPI
            icon={TrendingUp}
            label="Avg Sharpe"
            value={summary.avg_sharpe == null ? '—' : summary.avg_sharpe.toFixed(2)}
            color={sharpeColor(summary.avg_sharpe)}
          />
          <SummaryKPI
            icon={Award}
            label="Best Strategy"
            value={summary.best_strategy ?? '—'}
            color="#f5a623"
          />
          <SummaryKPI
            icon={DollarSign}
            label="Paper P&L"
            value={dollar(summary.total_paper_pnl)}
            color={returnColor(summary.total_paper_pnl)}
          />
          <SummaryKPI
            icon={DollarSign}
            label="Live P&L"
            value={dollar(summary.total_live_pnl)}
            color={returnColor(summary.total_live_pnl)}
          />
        </div>
      )}

      {/* Table */}
      {isLoading ? (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="flex gap-4 px-4 py-3 border-b border-[#1e1e1e] animate-pulse">
              <div className="w-6 h-4 bg-[#1e1e1e] rounded flex-shrink-0" />
              <div className="w-40 h-4 bg-[#1e1e1e] rounded" />
              <div className="w-12 h-4 bg-[#1e1e1e] rounded" />
              <div className="w-12 h-4 bg-[#1e1e1e] rounded" />
              <div className="w-16 h-4 bg-[#1e1e1e] rounded" />
              <div className="w-12 h-4 bg-[#1e1e1e] rounded" />
              <div className="w-12 h-4 bg-[#1e1e1e] rounded" />
            </div>
          ))}
        </div>
      ) : isError ? (
        <div className="flex flex-col items-center justify-center py-20 space-y-3 bg-[#111111] border border-[#1e1e1e] rounded-lg">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ff1744" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
          </svg>
          <p className="text-sm text-[#ff1744] font-medium">Failed to load leaderboard</p>
          <p className="text-xs text-[#555]">Check that the backend is running at <code className="text-[#888]">/api/v1/leaderboard/</code></p>
        </div>
      ) : entries.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 space-y-3 bg-[#111111] border border-[#1e1e1e] rounded-lg">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#333" strokeWidth="1.5">
            <rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/>
          </svg>
          <p className="text-sm text-[#555]">No strategies yet</p>
          <p className="text-xs text-[#444] max-w-xs text-center">
            Run a backtest or enable a strategy to see it appear here.
          </p>
        </div>
      ) : (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left" style={{ minWidth: 1100 }}>
              <thead>
                {/* Column group row */}
                <tr className="border-b border-[#1e1e1e]">
                  <th colSpan={4} className="px-3 py-1.5 text-[9px] text-[#333] uppercase tracking-widest border-b border-[#1e1e1e]">
                    Strategy
                  </th>
                  <ColGroup label="Backtest" span={9} color="#2196F3" />
                  <ColGroup label="Paper Trading" span={4} color="#00c853" />
                  <ColGroup label="Live Trading" span={4} color="#f5a623" />
                  <ColGroup label="Forward Test" span={3} color="#9c27b0" />
                </tr>
                {/* Column header row */}
                <tr className="bg-[#0a0a0a] border-b border-[#1e1e1e]">
                  {/* Strategy identity cols */}
                  <th className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-[#555] w-10">#</th>
                  <th className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-[#555]">Strategy</th>
                  <th className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-[#555]">Market</th>
                  <th className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-[#555]">Status</th>
                  {/* Backtest cols */}
                  <SortHeader label="Return" col="bt.return" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Ann Ret" col="bt.ann_ret" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Sharpe" col="bt.sharpe" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Sortino" col="bt.sortino" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Calmar" col="bt.calmar" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Max DD" col="bt.maxdd" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Win%" col="bt.winrate" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="P.Factor" col="bt.pf" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Trades" col="bt.trades" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  {/* Paper cols */}
                  <SortHeader label="P&L" col="paper.return" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Win%" col="paper.winrate" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="P.Factor" col="paper.pf" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Trades" col="paper.trades" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  {/* Live cols */}
                  <SortHeader label="P&L" col="live.return" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Win%" col="live.winrate" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="P.Factor" col="live.pf" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Trades" col="live.trades" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  {/* Forward test cols */}
                  <SortHeader label="Sharpe" col="ft.sharpe" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Return" col="ft.return" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                  <SortHeader label="Max DD" col="ft.maxdd" sortBy={sortBy} sortDir={sortDir} onClick={handleSort} />
                </tr>
              </thead>
              <tbody>
                {sorted.length === 0 ? (
                  <tr>
                    <td colSpan={24} className="px-4 py-10 text-center text-xs text-[#444]">
                      No strategies match the current filter.
                    </td>
                  </tr>
                ) : sorted.map((entry) => {
                  const bt = entry.backtest
                  const paper = entry.paper
                  const live = entry.live
                  const ft = entry.forward_test

                  return (
                    <tr key={entry.id}
                      className="border-b border-[#1a1a1a] hover:bg-[#131313] transition-colors">
                      {/* Rank */}
                      <td className="px-3 py-2.5 text-xs font-mono text-[#555] w-10">
                        {entry.rank}
                      </td>
                      {/* Strategy name */}
                      <td className="px-3 py-2.5 max-w-[180px]">
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-1.5">
                            <TypeBadge type={entry.strategy_type} />
                            <span className="text-xs font-medium text-[#e8e8e8] truncate">
                              {entry.display_name ?? entry.name}
                            </span>
                          </div>
                          {entry.symbols.length > 0 && (
                            <span className="text-[10px] text-[#444] font-mono truncate pl-0.5">
                              {entry.symbols.slice(0, 3).join(' · ')}
                              {entry.symbols.length > 3 && ` +${entry.symbols.length - 3}`}
                            </span>
                          )}
                        </div>
                      </td>
                      {/* Market */}
                      <td className="px-3 py-2.5">
                        <MarketBadge market={entry.market_type} />
                      </td>
                      {/* Status */}
                      <td className="px-3 py-2.5">
                        <StatusDot enabled={entry.is_enabled} />
                      </td>

                      {/* ── Backtest ───────────────────────────────── */}
                      <Cell value={pct(bt?.total_return)} color={returnColor(bt?.total_return ?? null)} />
                      <Cell value={pct(bt?.annualized_return)} color={returnColor(bt?.annualized_return ?? null)} />
                      <Cell value={num(bt?.sharpe_ratio)} color={sharpeColor(bt?.sharpe_ratio ?? null)} bold />
                      <Cell value={num(bt?.sortino_ratio)} color={sharpeColor(bt?.sortino_ratio ?? null)} />
                      <Cell value={num(bt?.calmar_ratio)} color={sharpeColor(bt?.calmar_ratio ?? null)} />
                      <Cell value={pct(bt?.max_drawdown)} color={drawdownColor(bt?.max_drawdown ?? null)} />
                      <Cell value={bt?.win_rate == null ? '—' : `${(bt.win_rate * 100).toFixed(1)}%`} color={winRateColor(bt?.win_rate ?? null)} />
                      <Cell value={num(bt?.profit_factor)} color={bt?.profit_factor != null && bt.profit_factor > 1 ? '#00c853' : '#ff1744'} />
                      <Cell value={int(bt?.total_trades)} />

                      {/* ── Paper ─────────────────────────────────── */}
                      <Cell value={dollar(paper?.total_return)} color={returnColor(paper?.total_return ?? null)} />
                      <Cell value={paper?.win_rate == null ? '—' : `${(paper.win_rate * 100).toFixed(1)}%`} color={winRateColor(paper?.win_rate ?? null)} />
                      <Cell value={num(paper?.profit_factor)} color={paper?.profit_factor != null && paper.profit_factor > 1 ? '#00c853' : '#555'} />
                      <Cell value={int(paper?.total_trades)} />

                      {/* ── Live ──────────────────────────────────── */}
                      <Cell value={dollar(live?.total_return)} color={returnColor(live?.total_return ?? null)} />
                      <Cell value={live?.win_rate == null ? '—' : `${(live.win_rate * 100).toFixed(1)}%`} color={winRateColor(live?.win_rate ?? null)} />
                      <Cell value={num(live?.profit_factor)} color={live?.profit_factor != null && live.profit_factor > 1 ? '#00c853' : '#555'} />
                      <Cell value={int(live?.total_trades)} />

                      {/* ── Forward Test ───────────────────────────── */}
                      <Cell value={num(ft?.sharpe_ratio)} color={sharpeColor(ft?.sharpe_ratio ?? null)} />
                      <Cell value={pct(ft?.total_return)} color={returnColor(ft?.total_return ?? null)} />
                      <Cell value={pct(ft?.max_drawdown)} color={drawdownColor(ft?.max_drawdown ?? null)} />
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Footer */}
          <div className="px-4 py-2.5 border-t border-[#1e1e1e] flex items-center justify-between">
            <span className="text-[10px] text-[#444]">
              {sorted.length} of {entries.length} strategies · refreshes every 30s
            </span>
            <div className="flex items-center gap-3 text-[10px] text-[#444]">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-[#2196F3]/30 border border-[#2196F3]/40" /> Backtest
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-[#00c853]/20 border border-[#00c853]/40" /> Paper
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-[#f5a623]/20 border border-[#f5a623]/40" /> Live
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-[#9c27b0]/20 border border-[#9c27b0]/40" /> Forward Test
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
