import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'
import { GlassCard } from '../ui/GlassCard'
import { AnimatedCounter } from '../ui/AnimatedCounter'

interface SnapshotData {
  total_pnl: number
  today_pnl: number
  sharpe: number
  win_rate: number
  max_drawdown: number
  open_positions: number
  total_pnl_trend?: number
  today_pnl_trend?: number
  sharpe_trend?: number
  win_rate_trend?: number
  max_drawdown_trend?: number
  open_positions_trend?: number
}

interface MetricCardProps {
  label: string
  value: string
  rawValue?: number
  trend?: number
  trendLabel?: string
  color?: string
  invertTrend?: boolean
  glowVariant?: 'green' | 'blue' | 'amber' | 'none'
}

function MetricCard({ label, value, rawValue, trend, trendLabel, color = '#f5a623', invertTrend = false, glowVariant = 'none' }: MetricCardProps) {
  const trendPositive = invertTrend ? (trend ?? 0) < 0 : (trend ?? 0) >= 0
  const trendColor = trend == null ? '#555555' : trendPositive ? '#00c853' : '#ff1744'
  const trendSign = (trend ?? 0) >= 0 ? '+' : ''
  const trendArrow = trend == null ? '' : trendPositive ? '↑' : '↓'

  return (
    <GlassCard
      glow={glowVariant}
      className="p-4 flex flex-col gap-1 min-w-0 hover:scale-[1.03] transition-transform duration-200 cursor-default"
    >
      <p className="text-[10px] text-[#555555] uppercase tracking-wider truncate">{label}</p>
      <p className="text-2xl font-black leading-none truncate" style={{ color }}>
        {rawValue !== undefined ? (
          <AnimatedCounter
            value={rawValue}
            decimals={rawValue % 1 !== 0 ? 2 : 0}
            duration={900}
          />
        ) : (
          value
        )}
      </p>
      {trend != null && (
        <p className="text-[10px] font-mono flex items-center gap-0.5" style={{ color: trendColor }}>
          <span>{trendArrow}</span>
          <span>
            {trendSign}
            {trendLabel ?? trend.toFixed(2)} vs yesterday
          </span>
        </p>
      )}
    </GlassCard>
  )
}

function SkeletonCard() {
  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 flex flex-col gap-2 min-w-0 animate-pulse">
      <div className="h-2 w-16 bg-[#1e1e1e] rounded" />
      <div className="h-7 w-24 bg-[#1e1e1e] rounded" />
      <div className="h-2 w-20 bg-[#1e1e1e] rounded" />
    </div>
  )
}

export function PortfolioSnapshot() {
  const { data, isLoading, isError, error } = useQuery<SnapshotData>({
    queryKey: ['portfolio', 'snapshot'],
    queryFn: () => api.get('/analytics/portfolio/snapshot').then((r) => r.data),
    refetchInterval: 30_000,
    retry: 2,
  })

  if (isLoading) {
    return (
      <GlassCard className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-white">Portfolio Snapshot</h2>
          <span className="text-[10px] text-[#555555] font-mono">loading…</span>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      </GlassCard>
    )
  }

  if (isError || !data) {
    return (
      <GlassCard className="p-4 border-[#ff1744]/20">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-white">Portfolio Snapshot</h2>
          <span className="text-[10px] text-[#ff1744] font-mono">error</span>
        </div>
        <p className="text-xs text-[#ff1744]/80 font-mono">
          {(error as Error)?.message ?? 'Failed to load portfolio data'}
        </p>
      </GlassCard>
    )
  }

  // Normalize numeric fields so a missing/null value from the API can never
  // crash the whole dashboard with "Cannot read properties of undefined".
  const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0)
  const d = {
    ...data,
    total_pnl: num(data.total_pnl),
    today_pnl: num(data.today_pnl),
    sharpe: num(data.sharpe),
    win_rate: num(data.win_rate),
    max_drawdown: num(data.max_drawdown),
    open_positions: num(data.open_positions),
  }
  const winRatePct = (d.win_rate > 1 ? d.win_rate : d.win_rate * 100).toFixed(1)
  const winRateTrendPct =
    d.win_rate_trend != null
      ? (d.win_rate_trend > 0.5 ? d.win_rate_trend : d.win_rate_trend * 100).toFixed(1) + '%'
      : undefined

  const sharpeGlow: 'green' | 'amber' | 'none' = d.sharpe >= 2.0 ? 'green' : d.sharpe >= 1.0 ? 'amber' : 'none'

  const cards: MetricCardProps[] = [
    {
      label: 'Total P&L',
      value: `$${d.total_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      rawValue: d.total_pnl,
      color: d.total_pnl >= 0 ? '#00ff88' : '#ff1744',
      glowVariant: d.total_pnl >= 0 ? 'green' : 'none',
      trend: d.total_pnl_trend,
      trendLabel:
        d.total_pnl_trend != null
          ? `$${Math.abs(d.total_pnl_trend).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
          : undefined,
    },
    {
      label: "Today's P&L",
      value: `${d.today_pnl >= 0 ? '+' : ''}$${d.today_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      rawValue: d.today_pnl,
      color: d.today_pnl >= 0 ? '#00ff88' : '#ff1744',
      glowVariant: d.today_pnl >= 0 ? 'green' : 'none',
      trend: d.today_pnl_trend,
      trendLabel:
        d.today_pnl_trend != null
          ? `$${Math.abs(d.today_pnl_trend).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
          : undefined,
    },
    {
      label: 'Sharpe Ratio',
      value: d.sharpe.toFixed(2),
      rawValue: d.sharpe,
      color: d.sharpe >= 2.0 ? '#00ff88' : d.sharpe >= 1.0 ? '#ffb347' : '#ff1744',
      glowVariant: sharpeGlow,
      trend: d.sharpe_trend,
      trendLabel: d.sharpe_trend != null ? d.sharpe_trend.toFixed(2) : undefined,
    },
    {
      label: 'Win Rate',
      value: `${winRatePct}%`,
      rawValue: parseFloat(winRatePct),
      color: '#00ff88',
      glowVariant: 'green',
      trend: d.win_rate_trend,
      trendLabel: winRateTrendPct,
    },
    {
      label: 'Max Drawdown',
      value: `${d.max_drawdown.toFixed(1)}%`,
      rawValue: d.max_drawdown,
      color: '#ff1744',
      glowVariant: 'amber',
      trend: d.max_drawdown_trend,
      trendLabel: d.max_drawdown_trend != null ? `${Math.abs(d.max_drawdown_trend).toFixed(1)}%` : undefined,
      invertTrend: true,
    },
    {
      label: 'Open Positions',
      value: String(d.open_positions),
      rawValue: d.open_positions,
      color: '#00d4ff',
      glowVariant: 'blue',
      trend: d.open_positions_trend,
      trendLabel: d.open_positions_trend != null ? String(Math.abs(d.open_positions_trend)) : undefined,
    },
  ]

  return (
    <GlassCard className="p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-white">Portfolio Snapshot</h2>
        <span className="text-[10px] text-[#555555] font-mono">live · refreshes 30s</span>
      </div>
      <div className="grid grid-cols-3 gap-3">
        {cards.map((c) => (
          <MetricCard key={c.label} {...c} />
        ))}
      </div>
    </GlassCard>
  )
}

export default PortfolioSnapshot
