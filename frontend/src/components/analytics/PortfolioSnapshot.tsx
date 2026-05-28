import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

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
  trend?: number
  trendLabel?: string
  color?: string
  invertTrend?: boolean
}

function MetricCard({ label, value, trend, trendLabel, color = '#f5a623', invertTrend = false }: MetricCardProps) {
  const trendPositive = invertTrend ? (trend ?? 0) < 0 : (trend ?? 0) >= 0
  const trendColor = trend == null ? '#555555' : trendPositive ? '#00c853' : '#ff1744'
  const trendSign = (trend ?? 0) >= 0 ? '+' : ''
  const trendArrow = trend == null ? '' : trendPositive ? '↑' : '↓'

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 flex flex-col gap-1 min-w-0">
      <p className="text-[10px] text-[#555555] uppercase tracking-wider truncate">{label}</p>
      <p className="text-2xl font-black leading-none truncate" style={{ color }}>
        {value}
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
    </div>
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
      <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-white">Portfolio Snapshot</h2>
          <span className="text-[10px] text-[#555555] font-mono">loading…</span>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="bg-[#0d0d0d] border border-[#ff1744]/30 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-white">Portfolio Snapshot</h2>
          <span className="text-[10px] text-[#ff1744] font-mono">error</span>
        </div>
        <p className="text-xs text-[#ff1744]/80 font-mono">
          {(error as Error)?.message ?? 'Failed to load portfolio data'}
        </p>
      </div>
    )
  }

  const d = data
  const winRatePct = (d.win_rate > 1 ? d.win_rate : d.win_rate * 100).toFixed(1)
  const winRateTrendPct =
    d.win_rate_trend != null
      ? (d.win_rate_trend > 0.5 ? d.win_rate_trend : d.win_rate_trend * 100).toFixed(1) + '%'
      : undefined

  const cards: MetricCardProps[] = [
    {
      label: 'Total P&L',
      value: `$${d.total_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      color: d.total_pnl >= 0 ? '#00c853' : '#ff1744',
      trend: d.total_pnl_trend,
      trendLabel:
        d.total_pnl_trend != null
          ? `$${Math.abs(d.total_pnl_trend).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
          : undefined,
    },
    {
      label: "Today's P&L",
      value: `${d.today_pnl >= 0 ? '+' : ''}$${d.today_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      color: d.today_pnl >= 0 ? '#00c853' : '#ff1744',
      trend: d.today_pnl_trend,
      trendLabel:
        d.today_pnl_trend != null
          ? `$${Math.abs(d.today_pnl_trend).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
          : undefined,
    },
    {
      label: 'Sharpe Ratio',
      value: d.sharpe.toFixed(2),
      color: d.sharpe >= 2.0 ? '#00c853' : d.sharpe >= 1.0 ? '#f5a623' : '#ff1744',
      trend: d.sharpe_trend,
      trendLabel: d.sharpe_trend != null ? d.sharpe_trend.toFixed(2) : undefined,
    },
    {
      label: 'Win Rate',
      value: `${winRatePct}%`,
      color: '#00c853',
      trend: d.win_rate_trend,
      trendLabel: winRateTrendPct,
    },
    {
      label: 'Max Drawdown',
      value: `${d.max_drawdown.toFixed(1)}%`,
      color: '#ff1744',
      trend: d.max_drawdown_trend,
      trendLabel: d.max_drawdown_trend != null ? `${Math.abs(d.max_drawdown_trend).toFixed(1)}%` : undefined,
      invertTrend: true,
    },
    {
      label: 'Open Positions',
      value: String(d.open_positions),
      color: '#2979ff',
      trend: d.open_positions_trend,
      trendLabel: d.open_positions_trend != null ? String(Math.abs(d.open_positions_trend)) : undefined,
    },
  ]

  return (
    <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-white">Portfolio Snapshot</h2>
        <span className="text-[10px] text-[#555555] font-mono">live · refreshes 30s</span>
      </div>
      <div className="grid grid-cols-3 gap-3">
        {cards.map((c) => (
          <MetricCard key={c.label} {...c} />
        ))}
      </div>
    </div>
  )
}

export default PortfolioSnapshot
