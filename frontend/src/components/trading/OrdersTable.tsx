/**
 * OrdersTable — sortable, filterable table of order history.
 *
 * Features:
 *   • Click column headers to sort ascending / descending
 *   • Search/filter input filters across symbol, side, status, order_type
 *   • No external dependencies beyond React useState
 */
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface Order {
  id: string
  symbol: string
  side: string
  order_type: string
  quantity: number | null
  limit_price?: number | null
  filled_qty: number
  status: string
  execution_algo?: string | null
  created_at: string
}

type SortKey = keyof Pick<Order, 'symbol' | 'side' | 'order_type' | 'quantity' | 'filled_qty' | 'status' | 'created_at'>
type SortDir = 'asc' | 'desc'

// ─── Helpers ──────────────────────────────────────────────────────────────────

const STATUS_STYLES: Record<string, { color: string; bg: string }> = {
  filled:    { color: '#00c853', bg: 'rgba(0,200,83,0.12)' },
  cancelled: { color: '#ff1744', bg: 'rgba(255,23,68,0.12)' },
  rejected:  { color: '#ff1744', bg: 'rgba(255,23,68,0.12)' },
  pending:   { color: '#f5a623', bg: 'rgba(245,166,35,0.12)' },
  submitted: { color: '#2979ff', bg: 'rgba(41,121,255,0.12)' },
  partial:   { color: '#f5a623', bg: 'rgba(245,166,35,0.12)' },
}

function statusStyle(status: string) {
  return STATUS_STYLES[status?.toLowerCase()] ?? { color: '#888888', bg: 'rgba(136,136,136,0.12)' }
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

interface OrdersTableProps {
  /** How many orders to request from the API. Default 100. */
  limit?: number
  /** Optional account_id filter passed to the API. */
  accountId?: string
}

export default function OrdersTable({ limit = 100, accountId }: OrdersTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('created_at')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [filter, setFilter] = useState('')

  const queryParams = new URLSearchParams({ limit: String(limit) })
  if (accountId) queryParams.set('account_id', accountId)

  const { data, isLoading, isError, refetch } = useQuery<Order[]>({
    queryKey: ['orders-table', limit, accountId],
    queryFn: () => api.get(`/orders/?${queryParams}`).then(r => r.data),
    refetchInterval: 10_000,
  })

  const orders: Order[] = Array.isArray(data) ? data : []

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
      ? orders.filter(
          o =>
            o.symbol?.toLowerCase().includes(q) ||
            o.side?.toLowerCase().includes(q) ||
            o.status?.toLowerCase().includes(q) ||
            o.order_type?.toLowerCase().includes(q) ||
            (o.execution_algo ?? '').toLowerCase().includes(q),
        )
      : orders

    return [...filtered].sort((a, b) => {
      let av: string | number = a[sortKey] ?? ''
      let bv: string | number = b[sortKey] ?? ''
      if (sortKey === 'quantity' || sortKey === 'filled_qty') {
        av = Number(av)
        bv = Number(bv)
      } else {
        av = String(av).toLowerCase()
        bv = String(bv).toLowerCase()
      }
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })
  }, [orders, filter, sortKey, sortDir])

  const cols: { key: SortKey; label: string; align?: 'right' }[] = [
    { key: 'symbol',     label: 'Symbol' },
    { key: 'side',       label: 'Side' },
    { key: 'order_type', label: 'Type' },
    { key: 'quantity',   label: 'Qty',    align: 'right' },
    { key: 'filled_qty', label: 'Filled', align: 'right' },
    { key: 'status',     label: 'Status', align: 'right' },
    { key: 'created_at', label: 'Time',   align: 'right' },
  ]

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-[#1e1e1e]">
        <h2 className="text-sm font-semibold text-white shrink-0">Order History</h2>
        <div className="flex items-center gap-2 flex-1 max-w-xs">
          <input
            type="text"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter symbol, side, status…"
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
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-8 bg-[#1a1a1a] rounded animate-pulse" />
            ))}
          </div>
        ) : isError ? (
          <div className="px-4 py-6 text-center">
            <p className="text-sm text-[#ff1744]">Failed to load orders.</p>
          </div>
        ) : processed.length === 0 ? (
          <div className="px-4 py-8 text-center space-y-1">
            <p className="text-sm text-[#888888]">
              {filter ? 'No orders match your filter.' : 'No orders yet.'}
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
              </tr>
            </thead>
            <tbody>
              {processed.map((o, idx) => {
                const isBuy = o.side === 'buy'
                const ss = statusStyle(o.status)
                const ts = o.created_at
                  ? new Date(o.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                  : '—'
                return (
                  <tr
                    key={o.id ?? idx}
                    className="border-b border-[#1a1a1a] last:border-0 hover:bg-[#1a1a1a] transition-colors"
                  >
                    <td className="py-2 px-3 font-bold text-[#e8e8e8]">{o.symbol}</td>
                    <td className="py-2 px-3">
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-black"
                        style={{
                          color: isBuy ? '#00c853' : '#ff1744',
                          backgroundColor: isBuy ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)',
                        }}
                      >
                        {(o.side ?? '').toUpperCase()}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-[#888888] capitalize">
                      {(o.order_type ?? '').replace('_', ' ')}
                    </td>
                    <td className="py-2 px-3 text-right text-[#e8e8e8]">
                      {o.quantity != null ? o.quantity : '—'}
                    </td>
                    <td className="py-2 px-3 text-right text-[#888888]">
                      {o.filled_qty ?? 0}
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-black tracking-wider"
                        style={{ color: ss.color, backgroundColor: ss.bg }}
                      >
                        {(o.status ?? '').toUpperCase()}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right text-[#555555]">{ts}</td>
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
            {processed.length} order{processed.length !== 1 ? 's' : ''}
            {filter && ` (filtered from ${orders.length})`}
          </span>
          <span>Click headers to sort</span>
        </div>
      )}
    </div>
  )
}
