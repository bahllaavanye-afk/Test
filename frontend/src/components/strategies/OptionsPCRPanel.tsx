/**
 * OptionsPCRPanel — displays live Put/Call Ratio for a symbol.
 *
 * Color coding:
 *   PCR > 1.2  → bullish  (green)  — excessive bearishness = contrarian buy
 *   PCR 0.8-1.2 → neutral (yellow)
 *   PCR < 0.8  → bearish  (red)   — excessive bullishness = contrarian sell
 *
 * Data source: GET /api/v1/market-data/pcr?symbol=<symbol>
 */
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

interface PCRData {
  symbol: string
  pcr: number | null
  put_volume: number | null
  call_volume: number | null
  signal: string
  confidence: number | null
  regime: string
  pcr_high_threshold?: number
  pcr_low_threshold?: number
  source: string
  computed_at: string
}

interface OptionsPCRPanelProps {
  symbol?: string
  /** Re-fetch interval in ms. Default 60 000 (1 min). */
  refetchInterval?: number
}

function pcrColor(pcr: number | null, high = 1.2, low = 0.8): string {
  if (pcr === null) return '#888888'
  if (pcr > high) return '#00c853'
  if (pcr < low) return '#ff1744'
  return '#f5a623'
}

function regimeLabel(regime: string): string {
  switch (regime) {
    case 'bullish': return 'BULLISH'
    case 'bearish': return 'BEARISH'
    case 'neutral': return 'NEUTRAL'
    default: return regime.toUpperCase()
  }
}

function signalArrow(signal: string): string {
  if (signal === 'buy') return '▲'
  if (signal === 'sell') return '▼'
  return '—'
}

export default function OptionsPCRPanel({ symbol = 'SPY', refetchInterval = 60_000 }: OptionsPCRPanelProps) {
  const { data, isLoading, isError, dataUpdatedAt } = useQuery<PCRData>({
    queryKey: ['pcr', symbol],
    queryFn: () =>
      api.get(`/market-data/pcr?symbol=${encodeURIComponent(symbol)}`).then(r => r.data),
    refetchInterval,
    retry: false,
  })

  const pcrHigh = data?.pcr_high_threshold ?? 1.2
  const pcrLow = data?.pcr_low_threshold ?? 0.8
  const color = pcrColor(data?.pcr ?? null, pcrHigh, pcrLow)
  const updatedTime = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : '—'

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
            Put/Call Ratio
          </h3>
          <p className="text-[10px] text-[#555555] mt-0.5">
            {symbol.toUpperCase()} · contrarian reversal signal
          </p>
        </div>
        <span className="text-[10px] text-[#444444] font-mono">{updatedTime}</span>
      </div>

      {isLoading && (
        <div className="space-y-2">
          <div className="h-10 bg-[#1a1a1a] rounded animate-pulse" />
          <div className="h-4 bg-[#1a1a1a] rounded animate-pulse w-3/4" />
        </div>
      )}

      {isError && (
        <p className="text-xs text-[#ff1744]">Failed to load PCR data.</p>
      )}

      {!isLoading && !isError && data && (
        <>
          {/* PCR value + regime badge */}
          <div className="flex items-end gap-3 mb-3">
            <div>
              <p className="text-[10px] text-[#555555] uppercase tracking-wider mb-0.5">PCR</p>
              {data.pcr != null ? (
                <p className="text-3xl font-black font-mono" style={{ color }}>
                  {data.pcr.toFixed(3)}
                </p>
              ) : (
                <p className="text-xl font-black text-[#555555]">—</p>
              )}
            </div>

            {data.regime !== 'unavailable' && (
              <div className="mb-1">
                <span
                  className="text-[10px] font-black px-2 py-1 rounded tracking-wider"
                  style={{
                    color,
                    backgroundColor: `${color}22`,
                    border: `1px solid ${color}44`,
                  }}
                >
                  {regimeLabel(data.regime)}
                </span>
              </div>
            )}
          </div>

          {/* Signal row */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span
                className="text-base font-black"
                style={{ color: data.signal === 'buy' ? '#00c853' : data.signal === 'sell' ? '#ff1744' : '#888888' }}
              >
                {signalArrow(data.signal)}
              </span>
              <span className="text-xs text-[#e8e8e8] uppercase tracking-wider font-semibold">
                {data.signal === 'buy' ? 'Buy signal' : data.signal === 'sell' ? 'Sell signal' : 'No signal'}
              </span>
            </div>
            {data.confidence != null && (
              <span className="text-xs font-mono text-[#888888]">
                {(data.confidence * 100).toFixed(0)}% conf
              </span>
            )}
          </div>

          {/* Put / Call volumes */}
          {data.put_volume != null && data.call_volume != null && (
            <div className="grid grid-cols-2 gap-2 mb-3">
              <div className="bg-[#0a0a0a] border border-[#1e1e1e] rounded p-2 text-center">
                <p className="text-[10px] text-[#555555] uppercase tracking-wider">Put Vol</p>
                <p className="text-sm font-bold text-[#ff1744] font-mono">
                  {data.put_volume > 1_000_000
                    ? `${(data.put_volume / 1_000_000).toFixed(1)}M`
                    : data.put_volume > 1_000
                    ? `${(data.put_volume / 1_000).toFixed(0)}K`
                    : data.put_volume.toFixed(0)}
                </p>
              </div>
              <div className="bg-[#0a0a0a] border border-[#1e1e1e] rounded p-2 text-center">
                <p className="text-[10px] text-[#555555] uppercase tracking-wider">Call Vol</p>
                <p className="text-sm font-bold text-[#00c853] font-mono">
                  {data.call_volume > 1_000_000
                    ? `${(data.call_volume / 1_000_000).toFixed(1)}M`
                    : data.call_volume > 1_000
                    ? `${(data.call_volume / 1_000).toFixed(0)}K`
                    : data.call_volume.toFixed(0)}
                </p>
              </div>
            </div>
          )}

          {/* Threshold legend */}
          <div className="border-t border-[#1e1e1e] pt-2 flex items-center justify-between text-[10px] text-[#444444]">
            <span>
              <span className="text-[#00c853]">Bull</span> &gt;{pcrHigh} ·{' '}
              <span className="text-[#f5a623]">Neutral</span> {pcrLow}–{pcrHigh} ·{' '}
              <span className="text-[#ff1744]">Bear</span> &lt;{pcrLow}
            </span>
            {data.source !== 'unavailable' && data.source !== 'no_credentials' && (
              <span className="text-[#333333]">{data.source}</span>
            )}
          </div>

          {/* Unavailable notice */}
          {(data.pcr === null || data.source === 'no_credentials') && (
            <p className="text-[10px] text-[#555555] mt-2">
              {data.source === 'no_credentials'
                ? 'Add Alpaca API keys to enable live PCR.'
                : 'PCR data unavailable — market may be closed or no options data.'}
            </p>
          )}
        </>
      )}
    </div>
  )
}
