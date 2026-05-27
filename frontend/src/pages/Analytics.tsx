import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

// ─── Components ─────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  sub,
  color = '#f5a623',
  arrow,
}: {
  label: string
  value: string
  sub?: string
  color?: string
  arrow?: 'up' | 'down'
}) {
  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      <p className="text-xs text-[#888888] mb-1">{label}</p>
      <p className="text-2xl font-black mt-1" style={{ color }}>
        {value}
        {arrow && (
          <span className={`ml-1.5 text-sm ${arrow === 'up' ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
            {arrow === 'up' ? '↑' : '↓'}
          </span>
        )}
      </p>
      {sub && <p className="text-[10px] text-[#555555] mt-0.5">{sub}</p>}
    </div>
  )
}

function HeatmapCell({ month, ret }: { month: string; ret: number }) {
  const [hovered, setHovered] = useState(false)
  const intensity = Math.min(Math.abs(ret) / 12, 1)
  const bg = ret >= 0
    ? `rgba(0, 200, 83, ${0.12 + intensity * 0.5})`
    : `rgba(255, 23, 68, ${0.12 + intensity * 0.5})`
  const border = ret >= 0 ? 'rgba(0, 200, 83, 0.3)' : 'rgba(255, 23, 68, 0.3)'

  return (
    <div
      className="relative rounded-lg p-2 text-center cursor-default transition-all"
      style={{ backgroundColor: bg, border: `1px solid ${border}` }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <p className="text-[9px] text-[#888888] font-medium">{month}</p>
      <p className={`text-xs font-black mt-0.5 ${ret >= 0 ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
        {ret >= 0 ? '+' : ''}{ret.toFixed(1)}%
      </p>
      {hovered && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-[#1e1e1e] border border-[#333333] rounded px-2 py-1 text-xs whitespace-nowrap z-10 pointer-events-none">
          {month}: {ret >= 0 ? '+' : ''}{ret.toFixed(2)}%
        </div>
      )}
    </div>
  )
}

// ─── Equity Curve from real data points ──────────────────────────────────────

function EquityCurveFromPoints({ points }: { points: { date: string; equity: number }[] }) {
  if (!points || points.length < 2) {
    return (
      <div className="flex flex-col items-center justify-center h-40 text-center space-y-2">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        <p className="text-sm text-[#888888]">No trades yet — connect Alpaca to start paper trading</p>
        <p className="text-xs text-[#555]">The equity curve will appear here once strategies begin executing.</p>
      </div>
    )
  }

  const W = 600
  const H = 160
  const PAD = { top: 12, right: 8, bottom: 20, left: 48 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const values = points.map(p => p.equity)
  const minV = Math.min(...values) * 0.995
  const maxV = Math.max(...values) * 1.005
  const N = values.length

  const scaleX = (i: number) => PAD.left + (i / (N - 1)) * innerW
  const scaleY = (v: number) => PAD.top + innerH - ((v - minV) / (maxV - minV)) * innerH

  const lineD = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${scaleX(i).toFixed(1)},${scaleY(v).toFixed(1)}`).join(' ')
  const fillD = `${lineD} L${scaleX(N - 1).toFixed(1)},${(PAD.top + innerH).toFixed(1)} L${PAD.left},${(PAD.top + innerH).toFixed(1)} Z`

  const yLabels = [minV, (minV + maxV) / 2, maxV].map(v => ({
    y: scaleY(v),
    label: `$${(v / 1000).toFixed(0)}K`,
  }))

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 160 }} preserveAspectRatio="none">
      <defs>
        <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f5a623" stopOpacity="0.18" />
          <stop offset="100%" stopColor="#f5a623" stopOpacity="0.01" />
        </linearGradient>
      </defs>
      {yLabels.map((yl, i) => (
        <text key={i} x={PAD.left - 4} y={yl.y + 4} textAnchor="end" fontSize="9" fill="#555555">{yl.label}</text>
      ))}
      {yLabels.map((yl, i) => (
        <line key={i} x1={PAD.left} y1={yl.y} x2={PAD.left + innerW} y2={yl.y} stroke="#1e1e1e" strokeWidth="1" strokeDasharray="3,3" />
      ))}
      <path d={fillD} fill="url(#equityFill)" />
      <path d={lineD} fill="none" stroke="#f5a623" strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx={scaleX(N - 1).toFixed(1)} cy={scaleY(values[N - 1]).toFixed(1)} r="3" fill="#f5a623" />
    </svg>
  )
}

// ─── Main page ───────────────────────────────────────────────────────────────

export default function Analytics() {
  const { data: perf, isError: perfError } = useQuery({
    queryKey: ['performance'],
    queryFn: () => api.get('/analytics/performance').then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: slippageRaw, isError: slippageError } = useQuery({
    queryKey: ['slippage'],
    queryFn: () => api.get('/analytics/slippage').then(r => r.data),
  })

  const { data: attributionRaw, isError: attrError } = useQuery({
    queryKey: ['attribution'],
    queryFn: () => api.get('/analytics/attribution').then(r => r.data),
  })

  const { data: monthlyRaw, isError: monthlyError } = useQuery({
    queryKey: ['monthly-returns'],
    queryFn: () => api.get('/analytics/monthly-returns').then(r => r.data),
  })

  const { data: equityCurveRaw, isError: curveError } = useQuery({
    queryKey: ['equity-curve'],
    queryFn: () => api.get('/analytics/equity-curve').then(r => r.data),
  })

  const { data: strategiesRaw, isError: strategiesError } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => api.get('/strategies/').then(r => r.data),
  })

  const totalPnl: number = perf?.total_pnl ?? 0
  const totalTrades: number = perf?.total_trades ?? 0
  const avgPnl: number = perf?.avg_pnl ?? 0
  const winRate: number = perf?.win_rate ?? 0
  const totalReturn = perf ? ((totalPnl / 100_000) * 100).toFixed(1) : null
  const sharpe: string | null = perf?.sharpe_ratio != null ? perf.sharpe_ratio.toFixed(2) : null
  const maxDD: string | null = perf?.max_drawdown != null ? perf.max_drawdown.toFixed(1) : null

  const slippage: any[] = Array.isArray(slippageRaw) ? slippageRaw : []
  const attribution: any[] = Array.isArray(attributionRaw) ? attributionRaw : []
  const monthlyReturns: { month: string; ret: number }[] = Array.isArray(monthlyRaw) ? monthlyRaw : []
  const equityCurvePoints: { date: string; equity: number }[] = Array.isArray(equityCurveRaw) ? equityCurveRaw : []
  const strategies: any[] = Array.isArray(strategiesRaw) ? strategiesRaw : []

  const maxPnl = attribution.length > 0 ? Math.max(...attribution.map(s => Math.abs(s.total_pnl))) : 0
  const marketBps = slippage.find(s => s.algo === 'market' || s.execution_algo === 'market')?.avg_bps ?? null

  const noData = !perf && !perfError
  const hasNoAccount = perfError || (!noData && totalTrades === 0 && totalPnl === 0)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Analytics</h1>
          <p className="text-xs text-[#888888] mt-0.5">Institutional performance metrics · walk-forward validated</p>
        </div>
        <div className="text-xs text-[#555555] font-mono">
          Updated {new Date().toLocaleTimeString()}
        </div>
      </div>

      {/* No account connected banner */}
      {hasNoAccount && (
        <div className="bg-[#111111] border border-[#f5a623]/30 rounded-lg p-4 flex items-center gap-3">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f5a623" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
          </svg>
          <div>
            <p className="text-sm text-[#f5a623] font-semibold">No account connected</p>
            <p className="text-xs text-[#888888] mt-0.5">
              Connect your Alpaca account to see live P&amp;L and analytics.{' '}
              <a href="/settings" className="text-[#f5a623] underline">Add API keys in Settings.</a>
            </p>
          </div>
        </div>
      )}

      {/* Performance metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          label="Total Return"
          value={totalReturn != null ? `${Number(totalReturn) >= 0 ? '+' : ''}${totalReturn}%` : '—'}
          sub="from $100K baseline"
          color={totalReturn != null ? (Number(totalReturn) >= 0 ? '#00c853' : '#ff1744') : '#555555'}
          arrow={totalReturn != null ? (Number(totalReturn) >= 0 ? 'up' : 'down') : undefined}
        />
        <MetricCard
          label="Sharpe Ratio"
          value={sharpe ?? '—'}
          sub="annualized"
          color={sharpe ? '#f5a623' : '#555555'}
          arrow={sharpe ? 'up' : undefined}
        />
        <MetricCard
          label="Max Drawdown"
          value={maxDD != null ? `${maxDD}%` : '—'}
          sub="from peak"
          color={maxDD != null ? '#ff1744' : '#555555'}
          arrow={maxDD != null ? 'down' : undefined}
        />
        <MetricCard
          label="Win Rate"
          value={winRate > 0 ? `${(winRate > 1 ? winRate : winRate * 100).toFixed(0)}%` : '—'}
          sub="all strategies"
          color={winRate > 0 ? '#00c853' : '#555555'}
          arrow={winRate > 0 ? 'up' : undefined}
        />
        <MetricCard
          label="Total P&L"
          value={perf ? `$${totalPnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '—'}
          sub="realized + unrealized"
          color={perf ? (totalPnl >= 0 ? '#00c853' : '#ff1744') : '#555555'}
          arrow={perf ? (totalPnl >= 0 ? 'up' : 'down') : undefined}
        />
        <MetricCard
          label="Total Trades"
          value={perf ? String(totalTrades) : '—'}
          sub={avgPnl ? `avg $${Math.abs(avgPnl).toFixed(0)}/trade` : undefined}
          color={perf ? '#2979ff' : '#555555'}
        />
      </div>

      {/* Equity curve */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-white">Equity Curve</h2>
            <p className="text-xs text-[#888888] mt-0.5">Starting capital · amber = equity · red shading = drawdown periods</p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-6 h-0.5 bg-[#f5a623]" />
              <span className="text-[#888888]">Equity</span>
            </div>
          </div>
        </div>
        {curveError ? (
          <div className="flex items-center justify-center h-40 text-center">
            <p className="text-sm text-[#ff1744]">Failed to load equity curve data.</p>
          </div>
        ) : (
          <EquityCurveFromPoints points={equityCurvePoints} />
        )}
      </div>

      {/* Strategy attribution + slippage side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Strategy Attribution */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-white mb-4">Strategy Attribution</h2>
          {attrError ? (
            <p className="text-xs text-[#ff1744]">Failed to load attribution data.</p>
          ) : attribution.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center space-y-2">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
                <rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/>
              </svg>
              <p className="text-sm text-[#888888]">No attribution data yet</p>
              <p className="text-xs text-[#555]">Strategy P&amp;L breakdown will appear once trades are executed.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {attribution.map(s => {
                const pct = maxPnl > 0 ? (Math.abs(s.total_pnl) / maxPnl) * 100 : 0
                const isPos = s.total_pnl >= 0
                const wr = s.win_rate > 1 ? s.win_rate : s.win_rate * 100
                const winDots = Math.round(wr / 10)
                return (
                  <div key={s.strategy}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-[#e8e8e8]">{s.strategy}</span>
                        <div className="flex gap-0.5">
                          {[...Array(10)].map((_, i) => (
                            <div
                              key={i}
                              className="w-1.5 h-1.5 rounded-full"
                              style={{ backgroundColor: i < winDots ? '#00c853' : '#1e1e1e' }}
                            />
                          ))}
                        </div>
                        <span className="text-[10px] text-[#555555]">{wr.toFixed(0)}% WR</span>
                      </div>
                      <div className="text-right">
                        <span className={`text-xs font-bold font-mono ${isPos ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
                          {isPos ? '+' : ''}${s.total_pnl.toLocaleString()}
                        </span>
                        <span className="text-[10px] text-[#555555] ml-2">{s.trade_count}t</span>
                      </div>
                    </div>
                    <div className="h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{ width: `${pct}%`, backgroundColor: isPos ? '#00c853' : '#ff1744', opacity: 0.8 }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Slippage analysis */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-white mb-1">Slippage Analysis</h2>
          <p className="text-xs text-[#888888] mb-4">LimitFirst policy saves 5–15 bps vs direct market orders on average.</p>
          {slippageError ? (
            <p className="text-xs text-[#ff1744]">Failed to load slippage data.</p>
          ) : slippage.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center space-y-2">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              <p className="text-sm text-[#888888]">No slippage data yet</p>
              <p className="text-xs text-[#555]">Execution analytics will appear after orders are filled.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {slippage.map(s => {
                const algo = s.execution_algo ?? s.algo ?? '—'
                const savings = algo !== 'market' && marketBps != null ? (marketBps - s.avg_bps).toFixed(1) : null
                const maxBps = Math.max(...slippage.map(x => x.avg_bps))
                const barPct = maxBps > 0 ? (s.avg_bps / maxBps) * 100 : 0
                return (
                  <div key={algo}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-mono text-[#e8e8e8] capitalize">{algo.replace('_', ' ')}</span>
                      <div className="flex items-center gap-3">
                        {savings && <span className="text-xs text-[#00c853] font-mono">-{savings} bps</span>}
                        <span className="text-xs font-bold font-mono text-[#f5a623]">{s.avg_bps} bps</span>
                        <span className="text-[10px] text-[#555555]">{s.count} orders</span>
                      </div>
                    </div>
                    <div className="h-2 bg-[#1e1e1e] rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${barPct}%`,
                          backgroundColor: algo === 'market' ? '#ff1744' : '#f5a623',
                          opacity: algo === 'market' ? 0.6 : 0.8,
                        }}
                      />
                    </div>
                  </div>
                )
              })}
              {marketBps != null && (
                <div className="mt-4 pt-3 border-t border-[#1e1e1e]">
                  <p className="text-xs text-[#888888]">
                    Worst: <span className="text-[#ff1744] font-mono">Market {marketBps} bps</span>
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* P&L Heatmap calendar */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-white">Monthly Returns Heatmap</h2>
            <p className="text-xs text-[#888888] mt-0.5">Last 12 months · hover for details</p>
          </div>
          <div className="flex items-center gap-3 text-xs text-[#555555]">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-[#ff1744]/40" />
              <span>Negative</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-[#00c853]/40" />
              <span>Positive</span>
            </div>
          </div>
        </div>
        {monthlyError ? (
          <p className="text-xs text-[#ff1744]">Failed to load monthly returns data.</p>
        ) : monthlyReturns.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center space-y-2">
            <p className="text-sm text-[#888888]">No monthly return data yet</p>
            <p className="text-xs text-[#555]">Monthly P&amp;L will appear after the first full trading month.</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-12 gap-2">
              {monthlyReturns.map(m => (
                <HeatmapCell key={m.month} month={m.month} ret={m.ret} />
              ))}
            </div>
            <div className="mt-3 pt-3 border-t border-[#1e1e1e] flex items-center gap-6 text-xs text-[#555555]">
              <span>
                Best month: <span className="text-[#00c853]">+{Math.max(...monthlyReturns.map(m => m.ret)).toFixed(1)}%</span>
              </span>
              <span>
                Worst month: <span className="text-[#ff1744]">{Math.min(...monthlyReturns.map(m => m.ret)).toFixed(1)}%</span>
              </span>
              <span>
                Positive months: <span className="text-white">{monthlyReturns.filter(m => m.ret > 0).length}/{monthlyReturns.length}</span>
              </span>
            </div>
          </>
        )}
      </div>

      {/* Strategy Leaderboard from API */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-sm font-semibold text-white mb-4">Strategy Leaderboard</h2>
        {strategiesError ? (
          <p className="text-xs text-[#ff1744]">Failed to load strategies from API.</p>
        ) : strategies.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center space-y-2">
            <p className="text-sm text-[#888888]">No strategies registered yet</p>
            <p className="text-xs text-[#555]">
              Add strategies via the backend API or run a backtest to see performance here.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-[#555555] uppercase tracking-wider border-b border-[#1e1e1e]">
                  <th className="text-left pb-2 pr-3 w-8">#</th>
                  <th className="text-left pb-2 pr-4">Strategy</th>
                  <th className="text-right pb-2 pr-4">Sharpe</th>
                  <th className="text-right pb-2 pr-4">Ann. Return</th>
                  <th className="text-right pb-2 pr-4">Win Rate</th>
                  <th className="text-right pb-2 pr-4">Max DD</th>
                  <th className="text-right pb-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {strategies.map((s: any, idx: number) => {
                  const sharpeVal: number | null = s.sharpe_ratio ?? null
                  const annReturn: number | null = s.annual_return ?? null
                  const winRateVal: number | null = s.win_rate ?? null
                  const maxDDVal: number | null = s.max_drawdown ?? null
                  const status = s.status ?? (s.is_active ? 'live' : 'paused')
                  const statusStyles: Record<string, { label: string; color: string; bg: string }> = {
                    live:     { label: 'LIVE',     color: '#00c853', bg: 'rgba(0,200,83,0.15)' },
                    paper:    { label: 'PAPER',    color: '#2979ff', bg: 'rgba(41,121,255,0.15)' },
                    paused:   { label: 'PAUSED',   color: '#f5a623', bg: 'rgba(245,166,35,0.15)' },
                    backtest: { label: 'BACKTEST', color: '#888888', bg: 'rgba(136,136,136,0.15)' },
                  }
                  const st = statusStyles[status] ?? statusStyles['paused']
                  const sharpeColor = sharpeVal != null ? (sharpeVal >= 2.5 ? '#00c853' : sharpeVal >= 1.5 ? '#f5a623' : '#ff1744') : '#555555'
                  return (
                    <tr key={s.id ?? s.name ?? idx} className="border-b border-[#1a1a1a] last:border-0 hover:bg-[#1a1a1a] transition-colors">
                      <td className="py-2.5 pr-3 text-[#555555]">{idx + 1}</td>
                      <td className="py-2.5 pr-4 text-[#e8e8e8] font-semibold">{s.name ?? '—'}</td>
                      <td className="py-2.5 pr-4 text-right font-black" style={{ color: sharpeColor }}>
                        {sharpeVal != null ? sharpeVal.toFixed(2) : <span className="text-[#555]">No backtest data yet</span>}
                      </td>
                      <td className={`py-2.5 pr-4 text-right font-bold ${annReturn != null ? (annReturn >= 0 ? 'text-[#00c853]' : 'text-[#ff1744]') : 'text-[#555]'}`}>
                        {annReturn != null ? `${annReturn >= 0 ? '+' : ''}${annReturn.toFixed(1)}%` : '—'}
                      </td>
                      <td className="py-2.5 pr-4 text-right text-[#e8e8e8]">
                        {winRateVal != null ? `${((winRateVal > 1 ? winRateVal : winRateVal * 100)).toFixed(0)}%` : '—'}
                      </td>
                      <td className="py-2.5 pr-4 text-right text-[#ff1744]">
                        {maxDDVal != null ? `${maxDDVal.toFixed(1)}%` : '—'}
                      </td>
                      <td className="py-2.5 text-right">
                        <span className="px-2 py-0.5 rounded text-[9px] font-black tracking-wider" style={{ color: st.color, backgroundColor: st.bg }}>
                          {st.label}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Risk & architecture notes */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-sm font-semibold text-white mb-3">Risk Architecture</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-[#888888]">
          <div className="space-y-1.5">
            <p className="flex items-start gap-1.5"><span className="text-[#f5a623] shrink-0">•</span> Walk-forward validated — no in-sample overfitting</p>
            <p className="flex items-start gap-1.5"><span className="text-[#f5a623] shrink-0">•</span> All strategies paper-tested 2 weeks before live activation</p>
            <p className="flex items-start gap-1.5"><span className="text-[#f5a623] shrink-0">•</span> Slippage minimized via TWAP/LimitFirst routing</p>
            <p className="flex items-start gap-1.5"><span className="text-[#f5a623] shrink-0">•</span> Kelly criterion position sizing (25% fractional)</p>
          </div>
          <div className="space-y-1.5">
            <p className="flex items-start gap-1.5"><span className="text-[#f5a623] shrink-0">•</span> Correlation-based cluster limits (max 30%/cluster)</p>
            <p className="flex items-start gap-1.5"><span className="text-[#f5a623] shrink-0">•</span> Global circuit breaker at 10% drawdown</p>
            <p className="flex items-start gap-1.5"><span className="text-[#f5a623] shrink-0">•</span> Arb circuit breaker at 5% drawdown</p>
            <p className="flex items-start gap-1.5"><span className="text-[#f5a623] shrink-0">•</span> AES-256 encrypted broker credentials</p>
          </div>
        </div>
      </div>
    </div>
  )
}
