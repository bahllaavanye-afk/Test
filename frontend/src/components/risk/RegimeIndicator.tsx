import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

type RegimeType = 'trending' | 'mean_reverting' | 'high_volatility' | 'crisis'

interface RegimeData {
  regime: RegimeType
  confidence: number          // 0–1
  description: string
  recommended_strategies: string[]
}

const REGIME_CONFIG: Record<
  RegimeType,
  { label: string; dot: string; accent: string; bg: string; border: string }
> = {
  trending: {
    label: 'TRENDING',
    dot: '🟢',
    accent: '#00c853',
    bg: 'rgba(0,200,83,0.08)',
    border: 'rgba(0,200,83,0.25)',
  },
  mean_reverting: {
    label: 'MEAN-REVERTING',
    dot: '🟡',
    accent: '#f5a623',
    bg: 'rgba(245,166,35,0.08)',
    border: 'rgba(245,166,35,0.25)',
  },
  high_volatility: {
    label: 'HIGH-VOL',
    dot: '🟠',
    accent: '#ff9800',
    bg: 'rgba(255,152,0,0.08)',
    border: 'rgba(255,152,0,0.25)',
  },
  crisis: {
    label: 'CRISIS',
    dot: '🔴',
    accent: '#ff1744',
    bg: 'rgba(255,23,68,0.08)',
    border: 'rgba(255,23,68,0.25)',
  },
}

// Fallback mock when the API isn't ready
const MOCK_REGIME: RegimeData = {
  regime: 'trending',
  confidence: 0.78,
  description: 'Markets displaying persistent directional momentum across sectors.',
  recommended_strategies: ['momentum', 'trend_following', 'breakout'],
}

export function RegimeIndicator() {
  const { data, isLoading, isError } = useQuery<RegimeData>({
    queryKey: ['regime', 'current'],
    queryFn: () => api.get('/regime/current').then((r) => r.data),
    refetchInterval: 60_000,
    retry: 1,
  })

  const regime: RegimeData = data ?? MOCK_REGIME
  const cfg = REGIME_CONFIG[regime.regime] ?? REGIME_CONFIG.trending
  const confidencePct = Math.round(regime.confidence * 100)

  return (
    <div
      className="rounded-lg p-3 flex flex-col gap-2"
      style={{
        backgroundColor: cfg.bg,
        border: `1px solid ${cfg.border}`,
        minHeight: 120,
      }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base leading-none">{cfg.dot}</span>
          <span
            className="text-xs font-black tracking-widest font-mono"
            style={{ color: cfg.accent }}
          >
            {cfg.label}
          </span>
        </div>
        <span className="text-[10px] text-[#555555] font-mono">
          {isLoading ? 'loading…' : isError ? 'mock' : 'live'}
        </span>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[9px] text-[#555555] uppercase tracking-wider">Confidence</span>
          <span className="text-[10px] font-mono" style={{ color: cfg.accent }}>
            {confidencePct}%
          </span>
        </div>
        <div className="h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${confidencePct}%`, backgroundColor: cfg.accent }}
          />
        </div>
      </div>

      {/* Description */}
      <p className="text-[10px] text-[#888888] leading-snug line-clamp-2">
        {regime.description}
      </p>

      {/* Recommended strategy chips */}
      {regime.recommended_strategies.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-auto">
          {regime.recommended_strategies.slice(0, 3).map((s) => (
            <span
              key={s}
              className="text-[9px] font-mono px-1.5 py-0.5 rounded"
              style={{
                color: cfg.accent,
                backgroundColor: `${cfg.accent}18`,
                border: `1px solid ${cfg.accent}30`,
              }}
            >
              {s.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export default RegimeIndicator
