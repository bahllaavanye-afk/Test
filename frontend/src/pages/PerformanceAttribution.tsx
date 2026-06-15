/**
 * Performance Attribution — which strategies/bots generate what % of total P&L.
 * Connects to GET /analytics/attribution and GET /analytics/daily-pnl.
 * Institutional feature: shows contribution by strategy, day-of-week heat, hour-of-day heat.
 */
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  PieChart, TrendingUp, Clock, Calendar, Award, BarChart3,
  ArrowUpRight, ArrowDownRight, Minus,
} from 'lucide-react'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface StrategyAttribution {
  name: string
  total_pnl: number
  contribution_pct: number
  num_trades: number
  win_rate: number
  avg_pnl_per_trade: number
  sharpe_proxy: number | null
}

interface AttributionData {
  period_days: number
  total_pnl: number
  total_trades: number
  expectancy_usd: number
  profit_factor: number | null
  by_strategy: StrategyAttribution[]
  by_day_of_week: Record<string, number>
  by_hour_of_day: Record<string, number>
  best_day: string
  worst_day: string
  best_hour_utc: number
  worst_hour_utc: number
}

interface DailyPnLData {
  series: Array<{ date: string; pnl: number; trades: number }>
  total_pnl: number
  today_pnl: number
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt$(v: number, decimals = 0): string {
  const abs = Math.abs(v)
  const sign = v >= 0 ? '+' : '-'
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}K`
  return `${sign}$${abs.toFixed(decimals)}`
}

function pnlColor(v: number): string {
  return v >= 0 ? '#00c853' : '#ff1744'
}

function heatColor(v: number, min: number, max: number): string {
  if (max === min) return '#1e1e1e'
  const t = (v - min) / (max - min)
  if (t >= 0.7) return '#00c853'
  if (t >= 0.4) return '#f5a623'
  return '#ff1744'
}

const DAYS_OPTIONS = [
  { label: '7d', value: 7 },
  { label: '30d', value: 30 },
  { label: '90d', value: 90 },
  { label: '365d', value: 365 },
]

// ─── KPI Card ────────────────────────────────────────────────────────────────

function KPICard({
  label, value, sub, color,
}: {
  label: string
  value: string
  sub?: string
  color?: string
}) {
  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      <div className="text-[10px] text-[#555] uppercase tracking-widest font-mono mb-1">{label}</div>
      <div className="text-xl font-bold font-mono" style={{ color: color ?? '#e8e8e8' }}>{value}</div>
      {sub && <div className="text-[10px] text-[#555] font-mono mt-1">{sub}</div>}
    </div>
  )
}

// ─── Strategy Row ─────────────────────────────────────────────────────────────

function StrategyRow({
  strat,
  totalAbsPnl,
  rank,
}: {
  strat: StrategyAttribution
  totalAbsPnl: number
  rank: number
}) {
  const barWidth = totalAbsPnl > 0 ? Math.abs(strat.total_pnl) / totalAbsPnl * 100 : 0
  const isPositive = strat.total_pnl >= 0
  const Arrow = isPositive ? ArrowUpRight : strat.total_pnl === 0 ? Minus : ArrowDownRight

  return (
    <tr className="border-b border-[#1a1a1a] hover:bg-[#131313] transition-colors">
      {/* Rank */}
      <td className="px-4 py-3 text-xs font-mono text-[#555]">
        {rank <= 3
          ? <span style={{ color: ['#f5a623', '#aaa', '#cd7f32'][rank - 1] }}>
              {'#'.repeat(rank)}
            </span>
          : <span className="text-[#555]">#{rank}</span>}
      </td>
      {/* Strategy name */}
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <Arrow size={12} style={{ color: pnlColor(strat.total_pnl) }} />
          <span className="text-sm text-[#e8e8e8] font-medium">{strat.name}</span>
        </div>
      </td>
      {/* Contribution bar */}
      <td className="px-4 py-3 w-40">
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${barWidth}%`,
                background: isPositive ? '#00c853' : '#ff1744',
              }}
            />
          </div>
          <span
            className="text-xs font-mono w-12 text-right"
            style={{ color: pnlColor(strat.contribution_pct) }}
          >
            {strat.contribution_pct >= 0 ? '+' : ''}{strat.contribution_pct.toFixed(1)}%
          </span>
        </div>
      </td>
      {/* P&L */}
      <td className="px-4 py-3 text-xs font-mono font-bold" style={{ color: pnlColor(strat.total_pnl) }}>
        {fmt$(strat.total_pnl, 2)}
      </td>
      {/* Trades */}
      <td className="px-4 py-3 text-xs font-mono text-[#888]">{strat.num_trades}</td>
      {/* Win rate */}
      <td className="px-4 py-3 text-xs font-mono" style={{ color: strat.win_rate >= 0.55 ? '#00c853' : '#888' }}>
        {(strat.win_rate * 100).toFixed(1)}%
      </td>
      {/* Avg per trade */}
      <td className="px-4 py-3 text-xs font-mono" style={{ color: pnlColor(strat.avg_pnl_per_trade) }}>
        {fmt$(strat.avg_pnl_per_trade, 2)}
      </td>
      {/* Sharpe proxy */}
      <td className="px-4 py-3 text-xs font-mono" style={{
        color: strat.sharpe_proxy == null ? '#555'
          : strat.sharpe_proxy >= 2 ? '#00c853'
          : strat.sharpe_proxy >= 1 ? '#f5a623'
          : '#ff1744',
      }}>
        {strat.sharpe_proxy != null ? strat.sharpe_proxy.toFixed(2) : '—'}
      </td>
    </tr>
  )
}

// ─── Day-of-Week Heatmap ──────────────────────────────────────────────────────

const DOW_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

function DayOfWeekHeat({ data, bestDay, worstDay }: {
  data: Record<string, number>
  bestDay: string
  worstDay: string
}) {
  const vals = DOW_ORDER.map(d => data[d] ?? 0)
  const min = Math.min(...vals)
  const max = Math.max(...vals)

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <Calendar size={14} className="text-[#f5a623]" />
        <span className="text-xs font-semibold text-[#e8e8e8] uppercase tracking-wide">
          Day of Week P&amp;L
        </span>
        <span className="ml-auto text-[10px] text-[#555] font-mono">avg per trade</span>
      </div>
      <div className="flex gap-1.5">
        {DOW_ORDER.map((day) => {
          const v = data[day] ?? 0
          const color = heatColor(v, min, max)
          const isBest = day === bestDay
          const isWorst = day === worstDay
          return (
            <div key={day} className="flex-1 flex flex-col items-center gap-1.5">
              <div
                className="w-full h-14 rounded-lg flex items-center justify-center transition-colors"
                style={{ background: `${color}22`, border: `1.5px solid ${color}55` }}
              >
                <span className="text-[10px] font-mono font-bold" style={{ color }}>
                  {v >= 0 ? '+' : ''}{v.toFixed(0)}
                </span>
              </div>
              <span className="text-[10px] font-mono text-[#555]">{day}</span>
              {isBest && <span className="text-[8px] text-[#00c853] font-bold">BEST</span>}
              {isWorst && <span className="text-[8px] text-[#ff1744] font-bold">WORST</span>}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Hour-of-Day Heatmap ──────────────────────────────────────────────────────

function HourOfDayHeat({ data, bestHour, worstHour }: {
  data: Record<string, number>
  bestHour: number
  worstHour: number
}) {
  const vals = Array.from({ length: 24 }, (_, h) => data[String(h)] ?? 0)
  const min = Math.min(...vals)
  const max = Math.max(...vals)

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <Clock size={14} className="text-[#2196F3]" />
        <span className="text-xs font-semibold text-[#e8e8e8] uppercase tracking-wide">
          Hour of Day P&amp;L (UTC)
        </span>
        <span className="ml-auto text-[10px] text-[#555] font-mono">avg per trade</span>
      </div>
      <div className="flex gap-px">
        {Array.from({ length: 24 }, (_, h) => {
          const v = data[String(h)] ?? 0
          const color = heatColor(v, min, max)
          const isBest = h === bestHour
          const isWorst = h === worstHour
          return (
            <div key={h} className="flex-1 flex flex-col items-end gap-1">
              <div
                className="w-full rounded-sm transition-colors"
                style={{
                  height: 40,
                  background: `${color}33`,
                  border: `1px solid ${color}66`,
                  borderColor: (isBest || isWorst) ? color : `${color}44`,
                }}
                title={`${h}:00 UTC — avg $${v.toFixed(2)}`}
              />
              {h % 6 === 0 && (
                <span className="text-[8px] font-mono text-[#444]">{String(h).padStart(2, '0')}h</span>
              )}
            </div>
          )
        })}
      </div>
      <div className="flex items-center gap-4 mt-3 text-[10px] text-[#555] font-mono">
        <span>Best: <span className="text-[#00c853]">{String(bestHour).padStart(2, '0')}:00 UTC</span></span>
        <span>Worst: <span className="text-[#ff1744]">{String(worstHour).padStart(2, '0')}:00 UTC</span></span>
      </div>
    </div>
  )
}

// ─── Equity Curve Mini ────────────────────────────────────────────────────────

function EquityCurveMini({ series }: { series: Array<{ date: string; pnl: number }> }) {
  if (!series.length) return null

  // Build cumulative P&L
  let cum = 0
  const cum_series = series.map(s => { cum += s.pnl; return { date: s.date, cum } })
  const max_cum = Math.max(...cum_series.map(s => s.cum), 0)
  const min_cum = Math.min(...cum_series.map(s => s.cum), 0)
  const range = max_cum - min_cum || 1

  const W = 600
  const H = 80
  const pts = cum_series.map((s, i) => {
    const x = (i / (cum_series.length - 1 || 1)) * W
    const y = H - ((s.cum - min_cum) / range) * H
    return `${x},${y}`
  })

  const finalPnl = cum_series[cum_series.length - 1]?.cum ?? 0
  const color = finalPnl >= 0 ? '#00c853' : '#ff1744'

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 size={14} className="text-[#9C27B0]" />
        <span className="text-xs font-semibold text-[#e8e8e8] uppercase tracking-wide">
          Cumulative P&amp;L
        </span>
        <span className="ml-auto text-xs font-mono font-bold" style={{ color }}>
          {fmt$(finalPnl, 2)}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
        {/* Zero line */}
        {min_cum < 0 && (
          <line
            x1={0}
            y1={H - ((0 - min_cum) / range) * H}
            x2={W}
            y2={H - ((0 - min_cum) / range) * H}
            stroke="#1e1e1e"
            strokeWidth={1}
            strokeDasharray="4,4"
          />
        )}
        {/* Gradient fill */}
        <defs>
          <linearGradient id="curve_fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <polyline
          points={pts.join(' ')}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
        />
      </svg>
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function PerformanceAttribution() {
  const [days, setDays] = useState(30)

  const { data, isLoading, error } = useQuery<AttributionData>({
    queryKey: ['attribution', days],
    queryFn: () => api.get(`/analytics/attribution?days=${days}`).then(r => r.data),
    refetchInterval: 60_000,
  })

  const { data: dailyPnL } = useQuery<DailyPnLData>({
    queryKey: ['daily-pnl', days],
    queryFn: () => api.get(`/analytics/daily-pnl?days=${days}`).then(r => r.data),
    refetchInterval: 60_000,
  })

  const totalAbsPnl = useMemo(
    () => (data?.by_strategy ?? []).reduce((s, x) => s + Math.abs(x.total_pnl), 0),
    [data]
  )

  return (
    <div className="p-6 space-y-5 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2">
            <PieChart size={18} className="text-[#f5a623]" />
            <h1 className="text-xl font-bold text-[#e8e8e8]">Performance Attribution</h1>
          </div>
          <p className="text-xs text-[#555] mt-0.5 font-mono">
            Which strategies generate your P&amp;L — and when
          </p>
        </div>

        {/* Period filter */}
        <div className="flex gap-1.5 bg-[#111111] border border-[#1e1e1e] rounded-lg p-1">
          {DAYS_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setDays(opt.value)}
              className="px-3 py-1.5 rounded text-xs font-mono transition-colors"
              style={{
                background: days === opt.value ? '#f5a623' : 'transparent',
                color: days === opt.value ? '#000' : '#888',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 bg-[#ff1744]/10 border border-[#ff1744]/30 rounded-lg text-[#ff1744] text-sm font-mono">
          Failed to load attribution data. Is the backend running?
        </div>
      )}

      {/* KPI Bar */}
      {data && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <KPICard
            label="Total P&L"
            value={fmt$(data.total_pnl, 2)}
            sub={`${days}d period`}
            color={pnlColor(data.total_pnl)}
          />
          <KPICard
            label="Total Trades"
            value={String(data.total_trades)}
            sub="closed positions"
          />
          <KPICard
            label="Expectancy"
            value={fmt$(data.expectancy_usd, 2)}
            sub="avg per trade"
            color={pnlColor(data.expectancy_usd)}
          />
          <KPICard
            label="Profit Factor"
            value={data.profit_factor != null ? data.profit_factor.toFixed(2) : '—'}
            sub="gross profit / gross loss"
            color={
              data.profit_factor == null ? '#555'
              : data.profit_factor >= 2 ? '#00c853'
              : data.profit_factor >= 1.2 ? '#f5a623'
              : '#ff1744'
            }
          />
        </div>
      )}

      {/* Loading skeleton */}
      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-12 bg-[#111111] border border-[#1e1e1e] rounded-lg animate-pulse" />
          ))}
        </div>
      )}

      {/* Equity curve */}
      {dailyPnL && dailyPnL.series.length > 0 && (
        <EquityCurveMini series={dailyPnL.series} />
      )}

      {/* Strategy attribution table */}
      {data && data.by_strategy.length > 0 && (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-[#1e1e1e]">
            <TrendingUp size={14} className="text-[#00c853]" />
            <span className="text-xs font-semibold text-[#e8e8e8] uppercase tracking-wide">
              Strategy Breakdown
            </span>
            <span className="ml-auto text-[10px] text-[#555] font-mono">
              {data.by_strategy.length} strategies
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-[#1e1e1e]">
                  {['#', 'Strategy', 'Contribution', 'Total P&L', 'Trades', 'Win %', 'Avg/Trade', 'Sharpe*'].map(h => (
                    <th
                      key={h}
                      className="px-4 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-[#555]"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.by_strategy.map((strat, i) => (
                  <StrategyRow
                    key={strat.name}
                    strat={strat}
                    totalAbsPnl={totalAbsPnl}
                    rank={i + 1}
                  />
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-2 border-t border-[#1e1e1e] text-[10px] text-[#444] font-mono">
            * Sharpe proxy = (avg_pnl / std_pnl) × √252 — annualized using closed-trade distribution
          </div>
        </div>
      )}

      {/* Empty state — no trades yet */}
      {data && data.by_strategy.length === 0 && !isLoading && (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-12 text-center">
          <Award size={32} className="mx-auto text-[#333] mb-3" />
          <div className="text-sm text-[#555] font-semibold mb-1">No closed trades yet</div>
          <div className="text-xs text-[#444] font-mono">
            Attribution populates as strategies close trades. Enable strategies in Bot Desk.
          </div>
        </div>
      )}

      {/* Heat maps row */}
      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <DayOfWeekHeat
            data={data.by_day_of_week}
            bestDay={data.best_day}
            worstDay={data.worst_day}
          />
          <HourOfDayHeat
            data={data.by_hour_of_day}
            bestHour={data.best_hour_utc}
            worstHour={data.worst_hour_utc}
          />
        </div>
      )}
    </div>
  )
}
