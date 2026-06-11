/**
 * SlippageDisplay — shows average slippage and recent fills.
 * Calls GET /analytics/slippage
 * No props — fetches its own data.
 */
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

interface SlippageRecord {
  symbol: string
  algo: string
  expected_price: number
  fill_price: number
  slippage_bps: number
  timestamp?: string
}

interface SlippageData {
  avg_slippage_bps: number | null
  total_saves_usd: number | null
  records: SlippageRecord[]
}

function slippageColor(bps: number): string {
  if (bps < 5) return '#00c853'
  if (bps < 20) return '#f5a623'
  return '#ff1744'
}

function fmtPrice(v: number): string {
  return v.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 })
}

export default function SlippageDisplay() {
  const { data, isLoading, isError } = useQuery<SlippageData>({
    queryKey: ['analytics-slippage'],
    queryFn: async () => {
      const res = await api.get('/analytics/slippage')
      return res.data
    },
    refetchInterval: 30_000,
    retry: false,
  })

  const containerStyle = {
    background: '#131722',
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
  }

  if (isLoading) {
    return (
      <div style={containerStyle} className="p-4 rounded-lg">
        <div className="flex items-center gap-2 text-[#888]">
          <div className="w-4 h-4 border-2 border-[#f5a623] border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Loading slippage data…</span>
        </div>
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div style={containerStyle} className="p-4 rounded-lg">
        <div className="rounded-lg p-6 flex flex-col items-center gap-2" style={{ background: '#1e2433' }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <p className="text-sm text-[#555]">No fills recorded</p>
        </div>
      </div>
    )
  }

  const recentFills = (data.records ?? []).slice(-5).reverse()

  const noFills = recentFills.length === 0 && data.avg_slippage_bps == null

  return (
    <div style={containerStyle} className="p-4 rounded-lg space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg p-3" style={{ background: '#1e2433' }}>
          <p className="text-[10px] text-[#555] uppercase tracking-widest mb-1">Avg Slippage</p>
          <p
            className="text-xl font-bold font-mono"
            style={{
              color: data.avg_slippage_bps != null
                ? slippageColor(data.avg_slippage_bps)
                : '#888',
            }}
          >
            {data.avg_slippage_bps != null ? `${data.avg_slippage_bps.toFixed(1)} bps` : '—'}
          </p>
        </div>

        <div className="rounded-lg p-3" style={{ background: '#1e2433' }}>
          <p className="text-[10px] text-[#555] uppercase tracking-widest mb-1">Saved vs Market</p>
          <p
            className="text-xl font-bold font-mono"
            style={{ color: (data.total_saves_usd ?? 0) >= 0 ? '#00c853' : '#ff1744' }}
          >
            {data.total_saves_usd != null
              ? fmtPrice(data.total_saves_usd)
              : '—'}
          </p>
        </div>
      </div>

      {/* Recent fills table */}
      <div className="rounded-lg overflow-hidden" style={{ background: '#1e2433' }}>
        <div
          className="px-3 py-2 border-b border-[#ffffff0d]"
          style={{ background: '#131722' }}
        >
          <span className="text-[10px] text-[#555] uppercase tracking-widest">Last 5 Fills</span>
        </div>

        {noFills ? (
          <div className="p-6 text-center text-sm text-[#555]">No fills recorded</div>
        ) : (
          <>
            {/* Header */}
            <div className="grid grid-cols-5 px-3 py-2 border-b border-[#ffffff0d]">
              {['Symbol', 'Algo', 'Expected', 'Fill', 'Slippage'].map(col => (
                <span key={col} className="text-[10px] text-[#444] uppercase tracking-widest">{col}</span>
              ))}
            </div>

            {/* Rows */}
            {recentFills.map((rec, idx) => (
              <div
                key={idx}
                className="grid grid-cols-5 px-3 py-2 border-b border-[#ffffff08] hover:bg-[#ffffff04] transition-colors"
              >
                <span className="text-xs text-white font-medium">{rec.symbol}</span>
                <span className="text-xs text-[#888]">{rec.algo}</span>
                <span className="text-xs text-[#aaa] font-mono">{fmtPrice(rec.expected_price)}</span>
                <span className="text-xs text-[#aaa] font-mono">{fmtPrice(rec.fill_price)}</span>
                <span
                  className="text-xs font-semibold font-mono"
                  style={{ color: slippageColor(rec.slippage_bps) }}
                >
                  {rec.slippage_bps.toFixed(1)} bps
                </span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  )
}
