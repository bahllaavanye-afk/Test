import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

interface SnapshotData {
  total_pnl: number
  today_pnl: number
  sharpe: number
  win_rate: number
  max_drawdown: number
  open_positions: number
  // trend vs prior period
  total_pnl_trend?: number
  today_pnl_trend?: number
  sharpe_trend?: number
  win_rate_trend?: number
  max_drawdown_trend?: number
  open_positions_trend?: number
}

// ─── Mock / fallback ─────────────────────────────────────────────────────────
const MOCK: SnapshotData = {
  total_pnl: 52_610,
  today_pnl: 1_430,
  sharpe: 2.14,
  win_rate: 0.69,
  max_drawdown: -14.2,
  open_positions: 18,
  total_pnl_trend: 3200,
  today_pnl_trend: 280,
  sharpe_trend: 0.07,
  win_rate_trend: 0.02,
  max_drawdown_trend: 1.1,
  open_positions_trend: -2,
}

// ─── Sub-components ───────────────────────────────────────────────────────────

interface MetricCardProps {
  label: string
  value: string
  trend?: number
  trendLabel?: string
  color?: string
  invertTrend?: boolean // for metrics where lower is better (drawdown)
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

// ─── Main component ────────────────────────────────────────────────────────────

export function PortfolioSnapshot() {
  const { data, isError } = useQuery<SnapshotData>({
    queryKey: ['portfolio', 'snapshot'],
    queryFn: () => api.get('/portfolio/snapshot').then((r) => r.data),
    refetchInterval: 30_000,
    retry: 1,
  })

  const d: SnapshotData = data ?? MOCK

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
        <span className="text-[10px] text-[#555555] font-mono">
          {isError ? 'mock data' : 'live · refreshes 30s'}
        </span>
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
