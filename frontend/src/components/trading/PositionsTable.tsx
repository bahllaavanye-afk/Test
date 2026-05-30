/**
 * PositionsTable — sortable, filterable table of open positions.
 *
 * Features:
 *   • Click column headers to sort ascending / descending
 *   • Search/filter input filters across symbol and side
 *   • No external dependencies beyond React useState
 */
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface Position {
  id?: string
  symbol: string
  side: string
  quantity: number
  avg_cost: number
  current_price: number | null
  unrealized_pnl: number | null
}

type SortKey = keyof Pick<Position, 'symbol' | 'side' | 'quantity' | 'avg_cost' | 'current_price' | 'unrealized_pnl'>
type SortDir = 'asc' | 'desc'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pnlColor(v: number | null | undefined): string {
  if (v == null) return '#555555'
  return v >= 0 ? '#00c853' : '#ff1744'
}

function fmtPnl(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : '-'}$${Math.abs(v).toFixed(2)}`
}

function fmtPct(pos: Position): string {
  if (pos.current_price == null || pos.avg_cost === 0) return '—'
  const dir = pos.side === 'short' ? -1 : 1
  const pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost) * 100 * dir
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`
}

function SortIcon({ col, sortKey, sortDir }: { col: SortKey; sortKey: SortKey; sortDir: SortDir }) {
  if (col !== sortKey) {
    return <span className="ml-1 text-[#333333] select-none">⇅</span>
  }
  return (
    <span className="ml-1 text-[#f5a623] select-none">
      {sortDir === 'asc' ? '↑' : '↓'}
    </span>
  )
}

// ─── Component ────────────────────────────────────────────────────────────────

interface PositionsTableProps {
  /** Optional account_id filter passed to the API. */
  accountId?: string
}

export default function PositionsTable({ accountId }: PositionsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('unrealized_pnl')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [filter, setFilter] = useState('')

  const queryParams = new URLSearchParams()
  if (accountId) queryParams.set('account_id', accountId)

  const { data, isLoading, isError, refetch } = useQuery<Position[]>({
    queryKey: ['positions-table', accountId],
    queryFn: () =>
      api.get(`/positions/${queryParams.toString() ? `?${queryParams}` : ''}`).then(r => r.data),
    refetchInterval: 5_000,
  })

  const positions: Position[] = Array.isArray(data) ? data : []

  // ── Sort handler ──
  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  // ── Filter + sort ──
  const processed = useMemo(() => {
    const q = filter.trim().toLowerCase()
    const filtered = q
      ? positions.filter(
          p =>
            p.symbol?.toLowerCase().includes(q) ||
            p.side?.toLowerCase().includes(q),
        )
      : positions

    return [...filtered].sort((a, b) => {
      const av: number | string = a[sortKey] ?? (typeof a[sortKey] === 'number' ? 0 : '')
      const bv: number | string = b[sortKey] ?? (typeof b[sortKey] === 'number' ? 0 : '')
      const an = Number(av)
      const bn = Number(bv)
      if (!isNaN(an) && !isNaN(bn)) {
        return sortDir === 'asc' ? an - bn : bn - an
      }
      const as = String(av).toLowerCase()
      const bs = String(bv).toLowerCase()
      if (as < bs) return sortDir === 'asc' ? -1 : 1
      if (as > bs) return sortDir === 'asc' ? 1 : -1
      return 0
    })
  }, [positions, filter, sortKey, sortDir])

  const totalUnrealizedPnl = positions.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0)

  const cols: { key: SortKey; label: string; align?: 'right' }[] = [
    { key: 'symbol',        label: 'Symbol' },
    { key: 'side',          label: 'Side' },
    { key: 'quantity',      label: 'Qty',          align: 'right' },
    { key: 'avg_cost',      label: 'Avg Cost',      align: 'right' },
    { key: 'current_price', label: 'Current',       align: 'right' },
    { key: 'unrealized_pnl', label: 'Unrealized P&L', align: 'right' },
  ]

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-[#1e1e1e]">
        <div>
          <h2 className="text-sm font-semibold text-white">Open Positions</h2>
          {positions.length > 0 && (
            <p className="text-[10px] mt-0.5" style={{ color: pnlColor(totalUnrealizedPnl) }}>
              Total unrealized: {fmtPnl(totalUnrealizedPnl)}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 flex-1 max-w-xs">
          <input
            type="text"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter symbol, side…"
            className="flex-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1 text-xs text-[#e8e8e8] placeholder-[#555] focus:outline-none focus:border-[#333]"
          />
          {filter && (
            <button onClick={() => setFilter('')} className="text-xs text-[#555] hover:text-[#888] transition-colors">✕</button>
          )}
        </div>
        <button
          onClick={() => refetch()}
          className="text-[10px] text-[#444] hover:text-[#888] transition-colors"
          title="Refresh"
        >
          ↻
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        {isLoading ? (
          <div className="p-4 space-y-2">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-8 bg-[#1a1a1a] rounded animate-pulse" />
            ))}
          </div>
        ) : isError ? (
          <div className="px-4 py-6 text-center">
            <p className="text-sm text-[#ff1744]">Failed to load positions.</p>
          </div>
        ) : processed.length === 0 ? (
          <div className="px-4 py-8 text-center space-y-1">
            <p className="text-sm text-[#888888]">
              {filter ? 'No positions match your filter.' : 'No open positions.'}
            </p>
            {filter && (
              <button onClick={() => setFilter('')} className="text-xs text-[#f5a623] underline">
                Clear filter
              </button>
            )}
          </div>
        ) : (
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-[#555555] uppercase tracking-wider border-b border-[#1e1e1e]">
                {cols.map(c => (
                  <th
                    key={c.key}
                    className={`py-2 px-3 cursor-pointer select-none hover:text-[#888] transition-colors ${c.align === 'right' ? 'text-right' : 'text-left'}`}
                    onClick={() => handleSort(c.key)}
                  >
                    {c.label}
                    <SortIcon col={c.key} sortKey={sortKey} sortDir={sortDir} />
                  </th>
                ))}
                <th className="py-2 px-3 text-right text-[#555555] uppercase tracking-wider">Change</th>
              </tr>
            </thead>
            <tbody>
              {processed.map((pos, idx) => {
                const isLong = pos.side === 'long'
                const pctStr = fmtPct(pos)
                const pctColor = pctStr.startsWith('+') ? '#00c853' : pctStr === '—' ? '#555555' : '#ff1744'
                return (
                  <tr
                    key={pos.id ?? pos.symbol ?? idx}
                    className="border-b border-[#1a1a1a] last:border-0 hover:bg-[#1a1a1a] transition-colors"
                  >
                    <td className="py-2.5 px-3 font-bold text-[#e8e8e8]">{pos.symbol}</td>
                    <td className="py-2.5 px-3">
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-black"
                        style={{
                          color: isLong ? '#00c853' : '#ff1744',
                          backgroundColor: isLong ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)',
                        }}
                      >
                        {(pos.side ?? 'LONG').toUpperCase()}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right text-[#e8e8e8]">{pos.quantity}</td>
                    <td className="py-2.5 px-3 text-right text-[#888888]">
                      ${pos.avg_cost.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-[#e8e8e8]">
                      {pos.current_price != null ? `$${pos.current_price.toFixed(2)}` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold" style={{ color: pnlColor(pos.unrealized_pnl) }}>
                      {fmtPnl(pos.unrealized_pnl)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold" style={{ color: pctColor }}>
                      {pctStr}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Footer */}
      {processed.length > 0 && (
        <div className="px-4 py-2 border-t border-[#1e1e1e] flex items-center justify-between text-[10px] text-[#444444]">
          <span>
            {processed.length} position{processed.length !== 1 ? 's' : ''}
            {filter && ` (filtered from ${positions.length})`}
          </span>
          <span>Click headers to sort</span>
        </div>
      )}
    </div>
  )
}
