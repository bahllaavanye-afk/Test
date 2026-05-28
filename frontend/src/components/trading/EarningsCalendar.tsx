import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

interface EarningsItem {
  symbol: string
  report_date: string | null
  fiscal_year: number | null
  fiscal_quarter: number | null
  estimate_eps: number | null
  reported_eps: number | null
  surprise_pct: number | null
}

interface EarningsResponse {
  earnings: EarningsItem[]
  data_source?: string
  reason?: string
}

interface Props {
  symbols?: string[]
}

const DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']

function fmt(n: number | null | undefined, decimals = 2): string {
  if (n == null) return '—'
  return n.toFixed(decimals)
}

function fmtDate(dateStr: string | null): string {
  if (!dateStr) return '—'
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: '2-digit',
    })
  } catch {
    return dateStr
  }
}

function isUpcoming(dateStr: string | null): boolean {
  if (!dateStr) return false
  return new Date(dateStr) >= new Date()
}

export default function EarningsCalendar({ symbols = DEFAULT_SYMBOLS }: Props) {
  const symbolsStr = symbols.join(',')

  const { data, isLoading, isError } = useQuery<EarningsResponse>({
    queryKey: ['earnings', symbolsStr],
    queryFn: () =>
      api
        .get(`/market-data/earnings?symbols=${encodeURIComponent(symbolsStr)}`)
        .then((r) => r.data)
        .catch(() => ({ earnings: [], data_source: 'unavailable' })),
    staleTime: 3_600_000, // 1 hour
    refetchInterval: 3_600_000,
  })

  const earnings: EarningsItem[] = (data?.earnings ?? []).slice().sort((a, b) => {
    const da = a.report_date ? new Date(a.report_date).getTime() : 0
    const db = b.report_date ? new Date(b.report_date).getTime() : 0
    return da - db
  })

  const upcoming = earnings.filter((e) => isUpcoming(e.report_date))
  const past = earnings.filter((e) => !isUpcoming(e.report_date))

  const unavailable =
    isError ||
    data?.data_source === 'unavailable' ||
    (data?.earnings?.length === 0 && data?.data_source !== 'alpaca')

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-[#1e1e1e]">
        <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">
          Earnings Calendar
        </span>
        {!isLoading && !unavailable && earnings.length > 0 && (
          <span className="ml-auto text-[9px] text-[#333]">{earnings.length} events</span>
        )}
      </div>

      {isLoading ? (
        <div className="p-3 space-y-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-8 bg-[#1a1a1a] rounded animate-pulse" />
          ))}
        </div>
      ) : unavailable || earnings.length === 0 ? (
        <div className="flex items-center justify-center py-8 px-4">
          <p className="text-xs text-[#444] text-center">
            Earnings data unavailable — premium data required
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[#1e1e1e]">
                {['Symbol', 'Report Date', 'Quarter', 'EPS Est', 'EPS Actual', 'Surprise %'].map(
                  (col) => (
                    <th
                      key={col}
                      className="px-2.5 py-1.5 text-left text-[9px] text-[#444] uppercase tracking-wider font-medium"
                    >
                      {col}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-[#0f0f0f]">
              {/* Upcoming */}
              {upcoming.length > 0 && (
                <>
                  <tr>
                    <td
                      colSpan={6}
                      className="px-2.5 py-1 bg-[#0d0d0d] text-[9px] text-[#f5a623] uppercase tracking-wider font-bold"
                    >
                      📅 Upcoming
                    </td>
                  </tr>
                  {upcoming.map((e, i) => (
                    <EarningsRow key={`up-${e.symbol}-${i}`} item={e} upcoming />
                  ))}
                </>
              )}

              {/* Past */}
              {past.length > 0 && (
                <>
                  <tr>
                    <td
                      colSpan={6}
                      className="px-2.5 py-1 bg-[#0d0d0d] text-[9px] text-[#555] uppercase tracking-wider font-bold"
                    >
                      Past Earnings
                    </td>
                  </tr>
                  {past.map((e, i) => (
                    <EarningsRow key={`past-${e.symbol}-${i}`} item={e} upcoming={false} />
                  ))}
                </>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function EarningsRow({ item, upcoming }: { item: EarningsItem; upcoming: boolean }) {
  const beat =
    item.reported_eps != null &&
    item.estimate_eps != null &&
    item.reported_eps >= item.estimate_eps

  const surpriseColor =
    item.surprise_pct == null
      ? '#555'
      : item.surprise_pct > 0
      ? '#00c853'
      : '#ff1744'

  return (
    <tr className="hover:bg-[#0f0f0f] transition-colors">
      {/* Symbol */}
      <td className="px-2.5 py-2">
        <div className="flex items-center gap-1">
          {!upcoming && item.reported_eps != null && (
            <span style={{ color: beat ? '#00c853' : '#ff1744' }}>
              {beat ? '✓' : '✗'}
            </span>
          )}
          <span className="font-mono font-bold text-[#e8e8e8]">{item.symbol}</span>
        </div>
      </td>

      {/* Report date */}
      <td className="px-2.5 py-2">
        <div className="flex items-center gap-1">
          {upcoming && <span className="text-[#f5a623]">📅</span>}
          <span className="text-[#888] font-mono">{fmtDate(item.report_date)}</span>
        </div>
      </td>

      {/* Quarter */}
      <td className="px-2.5 py-2 text-[#555] font-mono">
        {item.fiscal_quarter != null ? `Q${item.fiscal_quarter}` : '—'}
        {item.fiscal_year != null && (
          <span className="text-[#333] ml-0.5">'{String(item.fiscal_year).slice(-2)}</span>
        )}
      </td>

      {/* EPS Est */}
      <td className="px-2.5 py-2 text-[#888] font-mono">{fmt(item.estimate_eps)}</td>

      {/* EPS Actual */}
      <td className="px-2.5 py-2 font-mono font-bold text-[#e8e8e8]">
        {item.reported_eps != null ? fmt(item.reported_eps) : '—'}
      </td>

      {/* Surprise % */}
      <td className="px-2.5 py-2 font-mono font-bold" style={{ color: surpriseColor }}>
        {item.surprise_pct != null
          ? `${item.surprise_pct > 0 ? '+' : ''}${item.surprise_pct.toFixed(1)}%`
          : '—'}
      </td>
    </tr>
  )
}
