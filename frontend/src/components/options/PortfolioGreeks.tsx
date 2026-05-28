/**
 * PortfolioGreeks — Net portfolio-level options Greeks dashboard.
 *
 * Fetches /analytics/portfolio-greeks every 30 seconds and displays:
 *  - Four Greek meters (Delta, Theta, Gamma, Vega)
 *  - Daily income progress bar (theta / theta_target)
 *  - Warnings list
 *  - Compact per-position table
 */
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

// ── Types ──────────────────────────────────────────────────────────────────

interface PositionGreeks {
  symbol: string
  quantity: number
  delta: number
  gamma: number
  theta: number
  vega: number
  iv: number | null
  position_delta: number
  position_gamma: number
  position_theta: number
  position_vega: number
}

interface PortfolioGreeksData {
  net_delta: number
  net_gamma: number
  net_theta: number
  net_vega: number
  theta_target: number
  theta_pct_of_target: number
  delta_limit: number
  is_delta_neutral: boolean
  warnings: string[]
  position_count: number
  options_positions: PositionGreeks[]
  account_equity: number
  computed_at: string
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function fmtSigned(v: number, decimals = 2): string {
  const sign = v >= 0 ? '+' : ''
  return `${sign}${v.toFixed(decimals)}`
}

function fmtDollar(v: number): string {
  const sign = v >= 0 ? '+$' : '-$'
  return `${sign}${Math.abs(v).toFixed(2)}`
}

// ── Sub-components ─────────────────────────────────────────────────────────

interface GreekMeterProps {
  label: string
  symbol: string
  value: number
  subLabel?: string
  colorMode: 'delta' | 'theta' | 'neutral'
  deltaLimit?: number
  thetaTarget?: number
  decimals?: number
  isDollar?: boolean
}

function GreekMeter({
  label,
  symbol,
  value,
  subLabel,
  colorMode,
  deltaLimit,
  thetaTarget,
  decimals = 2,
  isDollar = false,
}: GreekMeterProps) {
  let valueColor = '#888888'

  if (colorMode === 'delta' && deltaLimit != null) {
    valueColor = Math.abs(value) < deltaLimit ? '#00c853' : '#ff1744'
  } else if (colorMode === 'theta' && thetaTarget != null) {
    const pct = thetaTarget > 0 ? value / thetaTarget : 0
    if (pct >= 0.8 && pct <= 1.2) valueColor = '#00c853'
    else if (pct >= 0.5) valueColor = '#f5a623'
    else valueColor = '#ff1744'
  } else if (colorMode === 'neutral') {
    valueColor = value >= 0 ? '#00c853' : '#ff1744'
  }

  const displayValue = isDollar
    ? fmtDollar(value)
    : fmtSigned(value, decimals)

  return (
    <div className="bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-3 flex flex-col gap-1">
      <div className="text-[10px] text-[#555] uppercase tracking-wider">{label}</div>
      <div className="text-[18px] font-mono font-bold" style={{ color: valueColor }}>
        {symbol} {displayValue}
      </div>
      {subLabel && (
        <div className="text-[10px] text-[#555]">{subLabel}</div>
      )}
    </div>
  )
}

interface ThetaProgressProps {
  theta: number
  thetaTarget: number
  thetaPct: number
}

function ThetaProgress({ theta, thetaTarget, thetaPct }: ThetaProgressProps) {
  const clampedPct = Math.min(Math.max(thetaPct, 0), 150)
  const barPct = Math.min(clampedPct, 100)

  let barColor = '#ff1744'
  if (thetaPct >= 80 && thetaPct <= 120) barColor = '#00c853'
  else if (thetaPct >= 50) barColor = '#f5a623'

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3 space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[#888888] uppercase tracking-wider">Daily Income Target</span>
        <span className="text-[10px] font-mono text-[#555]">
          {thetaPct.toFixed(0)}% of target
        </span>
      </div>
      <div className="flex items-center justify-between text-xs">
        <span className="font-mono font-semibold text-white">
          {fmtDollar(theta)} / day
        </span>
        <span className="text-[#555] font-mono">
          target: {fmtDollar(thetaTarget)}
        </span>
      </div>
      <div className="h-2 bg-[#1e1e1e] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${barPct}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  )
}

interface PositionsTableProps {
  positions: PositionGreeks[]
}

function PositionsTable({ positions }: PositionsTableProps) {
  if (positions.length === 0) return null

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
      <div className="px-3 py-2 border-b border-[#1e1e1e]">
        <span className="text-[10px] text-[#555] uppercase tracking-wider">Options Positions</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-[#555] text-[10px] border-b border-[#1e1e1e]">
              <th className="px-3 py-1.5 text-left">Symbol</th>
              <th className="px-2 py-1.5 text-right">Qty</th>
              <th className="px-2 py-1.5 text-right">IV</th>
              <th className="px-2 py-1.5 text-right">Δ Pos Delta</th>
              <th className="px-2 py-1.5 text-right">Θ Pos Theta</th>
              <th className="px-2 py-1.5 text-right">Γ Pos Gamma</th>
              <th className="px-2 py-1.5 text-right">ν Pos Vega</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((pos) => (
              <tr key={pos.symbol} className="border-b border-[#1e1e1e]/50 hover:bg-[#1a1a1a]">
                <td className="px-3 py-1.5 text-[#f5a623]">{pos.symbol}</td>
                <td className="px-2 py-1.5 text-right text-[#888]">{pos.quantity}</td>
                <td className="px-2 py-1.5 text-right text-[#888]">
                  {pos.iv != null ? `${(pos.iv * 100).toFixed(1)}%` : '—'}
                </td>
                <td className={`px-2 py-1.5 text-right ${pos.position_delta >= 0 ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
                  {fmtSigned(pos.position_delta, 2)}
                </td>
                <td className={`px-2 py-1.5 text-right ${pos.position_theta >= 0 ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
                  {fmtSigned(pos.position_theta, 2)}
                </td>
                <td className="px-2 py-1.5 text-right text-[#888]">
                  {fmtSigned(pos.position_gamma, 3)}
                </td>
                <td className={`px-2 py-1.5 text-right ${pos.position_vega >= 0 ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
                  {fmtSigned(pos.position_vega, 2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Main Component ──────────────────────────────────────────────────────────

interface PortfolioGreeksProps {
  accountId?: string
}

export default function PortfolioGreeks({ accountId }: PortfolioGreeksProps) {
  const { data, isLoading, error } = useQuery<PortfolioGreeksData>({
    queryKey: ['portfolio-greeks', accountId],
    queryFn: () =>
      api
        .get('/analytics/portfolio-greeks', {
          params: accountId ? { account_id: accountId } : {},
        })
        .then((r) => r.data),
    staleTime: 25_000,
    refetchInterval: 30_000,
  })

  if (isLoading) {
    return (
      <div className="space-y-3 animate-pulse">
        <div className="grid grid-cols-4 gap-2">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="h-20 bg-[#111111] border border-[#1e1e1e] rounded-lg" />
          ))}
        </div>
        <div className="h-12 bg-[#111111] border border-[#1e1e1e] rounded-lg" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 text-center">
        <div className="text-[#888888] text-xs">Portfolio Greeks unavailable</div>
        <div className="text-[10px] text-[#555] mt-1">
          Requires open options positions with Alpaca data.
        </div>
      </div>
    )
  }

  const computedAt = new Date(data.computed_at)
  const timeStr = computedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[#555] uppercase tracking-wider">
          Portfolio Greeks
          {data.position_count > 0 && (
            <span className="ml-2 text-[#f5a623]">{data.position_count} position{data.position_count !== 1 ? 's' : ''}</span>
          )}
        </span>
        <span className="text-[10px] text-[#444]">updated {timeStr}</span>
      </div>

      {/* Four Greek meters */}
      <div className="grid grid-cols-2 gap-2">
        <GreekMeter
          label="Net Delta"
          symbol="Δ"
          value={data.net_delta}
          subLabel={`limit: ±${data.delta_limit.toFixed(1)}`}
          colorMode="delta"
          deltaLimit={data.delta_limit}
          decimals={2}
        />
        <GreekMeter
          label="Net Theta"
          symbol="Θ"
          value={data.net_theta}
          subLabel={`target: +$${data.theta_target.toFixed(2)}/day`}
          colorMode="theta"
          thetaTarget={data.theta_target}
          decimals={2}
          isDollar={false}
        />
        <GreekMeter
          label="Net Gamma"
          symbol="Γ"
          value={data.net_gamma}
          colorMode="neutral"
          decimals={3}
        />
        <GreekMeter
          label="Net Vega"
          symbol="ν"
          value={data.net_vega}
          subLabel="vega > -1000 = caution"
          colorMode="neutral"
          decimals={2}
        />
      </div>

      {/* Theta income progress bar */}
      <ThetaProgress
        theta={data.net_theta}
        thetaTarget={data.theta_target}
        thetaPct={data.theta_pct_of_target}
      />

      {/* Delta neutral badge */}
      <div className="flex items-center gap-2">
        <div
          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
            data.is_delta_neutral
              ? 'bg-[#00c853]/15 text-[#00c853] border border-[#00c853]/30'
              : 'bg-[#ff1744]/15 text-[#ff1744] border border-[#ff1744]/30'
          }`}
        >
          {data.is_delta_neutral ? 'DELTA NEUTRAL' : 'DELTA EXPOSED'}
        </div>
        {data.account_equity > 0 && (
          <span className="text-[10px] text-[#444]">
            equity: ${data.account_equity.toLocaleString()}
          </span>
        )}
      </div>

      {/* Warnings */}
      {data.warnings.length > 0 && (
        <div className="space-y-1">
          {data.warnings.map((w, i) => (
            <div
              key={i}
              className="flex items-start gap-2 bg-[#f5a623]/8 border border-[#f5a623]/20 rounded px-2.5 py-1.5"
            >
              <span className="text-[#f5a623] text-[10px] mt-0.5 shrink-0">⚠</span>
              <span className="text-[10px] text-[#f5a623]">{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Per-position table */}
      <PositionsTable positions={data.options_positions} />

      {data.position_count === 0 && (
        <div className="text-center text-[#444] text-xs py-2">
          No open options positions found.
        </div>
      )}
    </div>
  )
}
