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
    ref