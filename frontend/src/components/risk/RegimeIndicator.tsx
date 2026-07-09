import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

type RegimeType = 'bull' | 'bear' | 'sideways' | 'unknown'

interface RegimeData {
  regime: RegimeType
  confidence: number          // 0–1
  updated_at: string | null
}

const REGIME_CONFIG: Record<
  RegimeType,
  { label: string; dot: string; accent: string; bg: string; border: string }
> = {
  bull: {
    label: 'BULL',
    dot: '🟢',
    accent: '#00c853',
    bg: 'rgba(0,200,83,0.08)',
    border: 'rgba(0,200,83,0.25)',
  },
  bear: {
    label: 'BEAR',
    dot: '🔴',
    accent: '#ff1744',
    bg: 'rgba(255,23,68,0.08)',
    border: 'rgba(255,23,68,0.25)',
  },
  sideways: {
    label: 'SIDEWAYS',
    dot: '🟡',
    accent: '#f5a623',
    bg: 'rgba(245,166,35,0.08)',
    border: 'rgba(245,166,35,0.25)',
  },
  unknown: {
    label: 'UNKNOWN',
    dot: '⚪',
    accent: '#888888',
    bg: 'rgba(136,136,136,0.08)',
    border: 'rgba(136,136,136,0.25)',
  },
}

export function RegimeIndicator() {
  const { data, isLoading, isError } = useQuery<RegimeData>({
    queryKey: ['regime', 'current'],
    queryFn: () => api.get('/regime/current').then((r) => r.data),
    refetchInterval: 60_000,
    retry: 1,
  })

  if (isLoading) {
    return (
      <div className="rounded-lg p-3 bg-[#111111] border border-[#1e1e1e] animate-pulse" style={{ minHeight: 80 }}>
        <div className="h-3 bg-[#1e1e1e] rounded w-1/4 mb-2" />
        <div className="h-2 bg-[#1e1e1e] rounded w-full" />
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="rounded-lg p-3 bg-[#111111] border border-[#1e1e1e] flex items-center gap-2">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
          <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
        </svg>
        <p className="text-xs text-[#555]">Market regime data unavailable — backend not connected</p>
      </div>
    )
  }

  const regime: RegimeData = data
  const cfg = REGIME_CONFIG[regime.regime] ?? REGIME_CONFIG.unknown
  const confidencePct = Math.round(regime.confidence * 100)

  return (
    <div
      className="rounded-lg p-3 flex flex-col gap-2"
      style={{ backgroundColor: cfg.bg, border: `1px solid ${cfg.border}`, minHeight: 120 }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base leading-none">{cfg.dot}</span>
          <span className="text-xs font-black tracking-widest font-mono" style={{ color: cfg.accent }}>
            {cfg.label}
          </span>
        </div>
        <span className="text-[10px] text-[#555555] font-mono">
          {regime.updated_at ? new Date(regime.updated_at).toLocaleTimeString() : 'live'}
        </span>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[9px] text-[#555555] uppercase tracking-wider">Confidence</span>
          <span className="text-[10px] font-mono" style={{ color: cfg.accent }}>{confidencePct}%</span>
        </div>
        <div className="h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-500"
            style={{ width: `${confidencePct}%`, backgroundColor: cfg.accent }} />
        </div>
      </div>
    </div>
  )
}

export default RegimeIndicator
