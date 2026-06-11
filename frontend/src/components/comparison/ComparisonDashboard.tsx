/**
 * ComparisonDashboard — side-by-side manual vs ML strategy comparison.
 *
 * Props: { strategyName: string, symbol: string }
 * Fetches GET /comparison/results?strategy={strategyName}&symbol={symbol}
 */
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'
import LWEquityCurve from '../charts/LWEquityCurve'

// ─── Types ────────────────────────────────────────────────────────────────────

interface StrategyMetrics {
  sharpe: number | null
  annual_return_pct: number | null
  max_drawdown_pct: number | null
  win_rate_pct: number | null
  num_trades: number | null
  equity_curve?: { time: number; value: number }[]
}

interface ComparisonResult {
  manual: StrategyMetrics
  ml_enhanced: StrategyMetrics
  winner: 'manual' | 'ml_enhanced' | 'tie'
}

interface ComparisonDashboardProps {
  strategyName: string
  symbol: string
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  suffix = '',
  color,
}: {
  label: string
  value: number | null
  suffix?: string
  color?: string
}) {
  return (
    <div className="bg-[#131722] border border-[#1e2433] rounded p-3">
      <p className="text-[10px] uppercase tracking-wider text-[#555555] mb-1">{label}</p>
      <p
        className="text-lg font-mono font-bold"
        style={{ color: color ?? '#e8e8e8' }}
      >
        {value != null ? `${value.toFixed(2)}${suffix}` : '—'}
      </p>
    </div>
  )
}

function ColumnPanel({
  title,
  metrics,
  color,
  isWinner,
}: {
  title: string
  metrics: StrategyMetrics
  color: string
  isWinner: boolean
}) {
  const curve = metrics.equity_curve ?? []

  return (
    <div className="flex-1 bg-[#1e2433] border border-[#2a3347] rounded-lg overflow-hidden">
      {/* Header */}
      <div
        className="px-4 py-3 flex items-center justify-between border-b"
        style={{ borderColor: '#2a3347' }}
      >
        <h3 className="text-sm font-bold" style={{ color }}>
          {title}
        </h3>
        {isWinner && (
          <span
            className="text-[10px] font-black px-2 py-0.5 rounded"
            style={{
              backgroundColor:
                color === '#00c853' ? 'rgba(0,200,83,0.15)' : 'rgba(245,166,35,0.15)',
              color,
            }}
          >
            WINNER
          </span>
        )}
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-2 p-3">
        <MetricCard
          label="Sharpe Ratio"
          value={metrics.sharpe}
          color={
            metrics.sharpe != null && metrics.sharpe >= 1.5
              ? '#00c853'
              : metrics.sharpe != null && metrics.sharpe >= 0.5
              ? '#f5a623'
              : '#ff1744'
          }
        />
        <MetricCard
          label="Annual Return"
          value={metrics.annual_return_pct}
          suffix="%"
          color={
            metrics.annual_return_pct != null && metrics.annual_return_pct >= 0
              ? '#00c853'
              : '#ff1744'
          }
        />
        <MetricCard label="Max Drawdown" value={metrics.max_drawdown_pct} suffix="%" color="#ff1744" />
        <MetricCard
          label="Win Rate"
          value={metrics.win_rate_pct}
          suffix="%"
          color={
            metrics.win_rate_pct != null && metrics.win_rate_pct >= 55 ? '#00c853' : '#f5a623'
          }
        />
      </div>

      {/* Num trades */}
      <div className="px-4 pb-2 text-[11px] text-[#888888]">
        Trades:{' '}
        <span className="text-[#e8e8e8] font-mono font-semibold">
          {metrics.num_trades ?? '—'}
        </span>
      </div>

      {/* Equity curve */}
      <div className="border-t border-[#2a3347] px-2 py-2">
        <p className="text-[10px] text-[#555555] uppercase tracking-wider mb-1 px-2">
          Equity Curve
        </p>
        {curve.length > 0 ? (
          <LWEquityCurve data={curve} height={180} />
        ) : (
          <div
            className="flex items-center justify-center text-[#555555] text-xs"
            style={{ height: 180 }}
          >
            No curve data
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function ComparisonDashboard({ strategyName, symbol }: ComparisonDashboardProps) {
  const { data, isLoading, isError, error } = useQuery<ComparisonResult>({
    queryKey: ['comparison-results', strategyName, symbol],
    queryFn: () =>
      api
        .get('/comparison/results', { params: { strategy: strategyName, symbol } })
        .then(r => r.data),
    staleTime: 30_000,
    retry: false,
  })

  return (
    <div
      className="rounded-xl border p-4 space-y-4"
      style={{ backgroundColor: '#131722', borderColor: '#1e2433' }}
    >
      {/* Title row */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-bold text-white">Strategy Comparison</h2>
          <p className="text-[11px] text-[#555555] mt-0.5">
            {strategyName} · {symbol}
          </p>
        </div>
        {data?.winner && data.winner !== 'tie' && (
          <div
            className="text-[11px] font-bold px-3 py-1 rounded-full"
            style={{
              backgroundColor:
                data.winner === 'ml_enhanced'
                  ? 'rgba(0,200,83,0.15)'
                  : 'rgba(245,166,35,0.15)',
              color: data.winner === 'ml_enhanced' ? '#00c853' : '#f5a623',
            }}
          >
            {data.winner === 'ml_enhanced' ? 'ML Enhanced wins' : 'Manual wins'}
          </div>
        )}
        {data?.winner === 'tie' && (
          <div className="text-[11px] font-bold px-3 py-1 rounded-full bg-[#1e2433] text-[#888888]">
            Tie
          </div>
        )}
      </div>

      {/* Loading skeleton */}
      {isLoading && (
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-10 bg-[#1e2433] rounded animate-pulse" />
          ))}
        </div>
      )}

      {/* Error */}
      {isError && (
        <div className="py-6 text-center space-y-1">
          <p className="text-sm text-[#ff1744]">
            {(error as Error)?.message ?? 'Failed to load comparison data.'}
          </p>
        </div>
      )}

      {/* No data yet */}
      {!isLoading && !isError && !data && (
        <div className="py-12 text-center space-y-2">
          <svg
            className="mx-auto"
            width="40"
            height="40"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#555"
            strokeWidth="1.5"
          >
            <rect x="3" y="3" width="8" height="18" rx="1" />
            <rect x="13" y="3" width="8" height="18" rx="1" />
          </svg>
          <p className="text-sm text-[#888888]">Run a comparison first</p>
          <p className="text-xs text-[#555555]">
            Use the backtest lab to generate comparison results.
          </p>
        </div>
      )}

      {/* Two-column layout */}
      {data && (
        <div className="flex gap-4">
          <ColumnPanel
            title="Manual Strategy"
            metrics={data.manual}
            color="#f5a623"
            isWinner={data.winner === 'manual'}
          />
          <ColumnPanel
            title="ML Enhanced"
            metrics={data.ml_enhanced}
            color="#00c853"
            isWinner={data.winner === 'ml_enhanced'}
          />
        </div>
      )}

      {/* Side-by-side metrics table */}
      {data && (
        <div className="bg-[#1e2433] border border-[#2a3347] rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#2a3347]">
                <th className="py-2 px-3 text-left text-[10px] uppercase tracking-wider text-[#555555]">
                  Metric
                </th>
                <th className="py-2 px-3 text-right text-[10px] uppercase tracking-wider text-[#f5a623]">
                  Manual
                </th>
                <th className="py-2 px-3 text-right text-[10px] uppercase tracking-wider text-[#00c853]">
                  ML Enhanced
                </th>
              </tr>
            </thead>
            <tbody>
              {[
                { label: 'Sharpe Ratio', manualV: data.manual.sharpe, mlV: data.ml_enhanced.sharpe, higher: true },
                { label: 'Annual Return %', manualV: data.manual.annual_return_pct, mlV: data.ml_enhanced.annual_return_pct, higher: true },
                { label: 'Max Drawdown %', manualV: data.manual.max_drawdown_pct, mlV: data.ml_enhanced.max_drawdown_pct, higher: false },
                { label: 'Win Rate %', manualV: data.manual.win_rate_pct, mlV: data.ml_enhanced.win_rate_pct, higher: true },
                { label: 'Num Trades', manualV: data.manual.num_trades, mlV: data.ml_enhanced.num_trades, higher: false },
              ].map(({ label, manualV, mlV, higher }) => {
                const manualBetter =
                  manualV != null && mlV != null && (higher ? manualV > mlV : manualV < mlV)
                const mlBetter =
                  manualV != null && mlV != null && (higher ? mlV > manualV : mlV < manualV)
                return (
                  <tr key={label} className="border-b border-[#1e2433] last:border-0">
                    <td className="py-2 px-3 text-xs text-[#888888]">{label}</td>
                    <td
                      className="py-2 px-3 text-xs font-mono text-right font-semibold"
                      style={{ color: manualBetter ? '#f5a623' : '#e8e8e8' }}
                    >
                      {manualV != null ? manualV.toFixed(2) : '—'}
                    </td>
                    <td
                      className="py-2 px-3 text-xs font-mono text-right font-semibold"
                      style={{ color: mlBetter ? '#00c853' : '#e8e8e8' }}
                    >
                      {mlV != null ? mlV.toFixed(2) : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
