import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

function vixColor(vix: number | null | undefined): string {
  if (vix == null) return '#888888'
  if (vix > 30) return '#ff1744'
  if (vix > 20) return '#f5a623'
  return '#00c853'
}

function biasColor(bias: string | undefined): string {
  if (bias === 'risk_on') return '#00c853'
  if (bias === 'risk_off') return '#ff1744'
  return '#f5a623'
}

function MetricCard({ label, value, sub, color = '#f5a623' }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      <p className="text-[#888888] text-xs uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold mt-1" style={{ color }}>{value}</p>
      {sub && <p className="text-[#888888] text-xs mt-1">{sub}</p>}
    </div>
  )
}

export default function Dashboard() {
  const { data: perf } = useQuery({ queryKey: ['performance'], queryFn: () => api.get('/analytics/performance').then(r => r.data), refetchInterval: 30_000 })
  const { data: positions } = useQuery({ queryKey: ['positions'], queryFn: () => api.get('/positions/').then(r => r.data), refetchInterval: 10_000 })
  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: () => api.get('/strategies/').then(r => r.data) })
  const { data: macro } = useQuery({ queryKey: ['macro'], queryFn: () => api.get('/analytics/macro').then(r => r.data), refetchInterval: 300_000 })
  const { data: sentiment } = useQuery({ queryKey: ['sentiment'], queryFn: () => api.get('/analytics/sentiment').then(r => r.data), refetchInterval: 600_000 })

  const activeCount = strategies?.filter((s: any) => s.is_active || s.is_enabled)?.length ?? 0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">Dashboard</h1>
        <span className="text-xs text-[#00c853] bg-[#00c853]/10 px-3 py-1 rounded-full border border-[#00c853]/20">● PAPER TRADING</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Total P&L" value={`$${(perf?.total_pnl ?? 0).toFixed(2)}`} sub={`${perf?.total_trades ?? 0} trades`} color="#00c853" />
        <MetricCard label="Open Positions" value={String(positions?.length ?? 0)} sub="live positions" color="#2979ff" />
        <MetricCard label="Active Strategies" value={String(activeCount)} sub="running 24/7" color="#f5a623" />
        <MetricCard label="Target Sharpe" value=">2.0" sub="vs SPY 0.47" color="#9C27B0" />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
          <div className="p-3 border-b border-[#1e1e1e]">
            <h2 className="text-sm font-semibold">Market Overview</h2>
          </div>
          <div style={{ height: 420 }}>
            <iframe
              src="https://www.tradingview.com/widgetembed/?frameElementId=tv_widget&symbol=NASDAQ%3AAAPL&interval=D&theme=dark&style=1&locale=en&toolbar_bg=111111&hide_side_toolbar=0&allow_symbol_change=1&save_image=0"
              style={{ width: '100%', height: '100%', border: 'none' }}
              title="TradingView Chart"
            />
          </div>
        </div>

        <div className="space-y-3">
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
            <h3 className="text-xs text-[#888888] uppercase tracking-wider mb-3">Benchmark Targets</h3>
            {[
              { label: 'S&P 500 (SPY)', sharpe: '0.47', color: '#2196F3' },
              { label: 'NASDAQ (QQQ)', sharpe: '0.61', color: '#9C27B0' },
              { label: 'Buffett (BRK.B)', sharpe: '0.79', color: '#FF9800' },
              { label: 'All Weather', sharpe: '0.67', color: '#4CAF50' },
              { label: 'QuantEdge Target', sharpe: '>2.0', color: '#f5a623' },
            ].map(b => (
              <div key={b.label} className="flex justify-between items-center py-1.5 border-b border-[#1e1e1e] last:border-0">
                <span className="text-xs" style={{ color: b.color }}>{b.label}</span>
                <span className="text-xs font-mono">{b.sharpe}</span>
              </div>
            ))}
          </div>

          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
            <h3 className="text-xs text-[#888888] uppercase tracking-wider mb-3">Strategies</h3>
            <div className="space-y-1.5 max-h-40 overflow-y-auto">
              {strategies?.slice(0, 8).map((s: any) => (
                <div key={s.id} className="flex justify-between text-xs">
                  <span className="truncate">{s.name}</span>
                  <span className={s.is_enabled || s.is_active ? 'text-[#00c853]' : 'text-[#888888]'}>
                    {s.is_enabled || s.is_active ? '● ON' : '○ OFF'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Macro Signals Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <h3 className="text-xs text-[#888888] uppercase tracking-wider mb-3">Macro Signals</h3>
          {!macro ? (
            <p className="text-xs text-[#888888]">Loading macro data...</p>
          ) : (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">VIX Level</span>
                <span className="text-sm font-bold font-mono" style={{ color: vixColor(macro.vix) }}>
                  {macro.vix != null ? macro.vix.toFixed(2) : '—'}
                  {macro.signals?.vix_regime && (
                    <span className="ml-1 text-xs font-normal">({macro.signals.vix_regime})</span>
                  )}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Yield Curve (10Y-2Y)</span>
                <span className="text-sm font-mono" style={{ color: macro.signals?.yield_curve_inverted ? '#ff1744' : '#00c853' }}>
                  {macro.yield_spread_10y2y != null ? `${(macro.yield_spread_10y2y * 100).toFixed(0)} bps` : '—'}
                  {macro.signals?.yield_curve_inverted != null && (
                    <span className="ml-1 text-xs">({macro.signals.yield_curve_inverted ? 'INVERTED' : 'normal'})</span>
                  )}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Macro Bias</span>
                <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ color: biasColor(macro.macro_bias), background: `${biasColor(macro.macro_bias)}20` }}>
                  {macro.macro_bias?.replace('_', ' ').toUpperCase() ?? '—'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Macro Score</span>
                <span className="text-sm font-bold font-mono" style={{ color: biasColor(macro.macro_bias) }}>
                  {macro.macro_score != null ? (macro.macro_score > 0 ? `+${macro.macro_score}` : macro.macro_score) : '—'} / 3
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Reddit Buzz Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <h3 className="text-xs text-[#888888] uppercase tracking-wider mb-3">Reddit Buzz (WSB)</h3>
          {!sentiment ? (
            <p className="text-xs text-[#888888]">Loading sentiment data...</p>
          ) : sentiment.error ? (
            <p className="text-xs text-[#888888]">Sentiment unavailable</p>
          ) : (
            <div className="space-y-2">
              {(sentiment.results ?? []).slice(0, 5).map((item: any, i: number) => {
                const maxMentions = sentiment.results?.[0]?.mentions_24h ?? 1
                const pct = Math.round(((item.mentions_24h ?? 0) / Math.max(maxMentions, 1)) * 100)
                return (
                  <div key={item.ticker ?? i} className="space-y-0.5">
                    <div className="flex justify-between text-xs">
                      <span className="font-bold text-[#f5a623]">{item.ticker}</span>
                      <span className="text-[#888888]">{item.mentions_24h?.toLocaleString()} mentions</span>
                    </div>
                    <div className="h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
                      <div className="h-full bg-[#f5a623] rounded-full transition-all" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
