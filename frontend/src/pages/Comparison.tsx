import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

function ImprovementBar({ pct }: { pct: number }) {
  const w = Math.min(Math.abs(pct) * 2, 100)
  const color = pct > 0 ? '#00c853' : '#ff1744'
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${w}%`, background: color }} />
      </div>
      <span className="text-xs font-mono font-bold" style={{ color }}>
        {pct > 0 ? '+' : ''}{pct.toFixed(1)}%
      </span>
    </div>
  )
}

export default function Comparison() {
  const { data: benchmarks, isLoading: benchLoading, isError: benchError } = useQuery({
    queryKey: ['benchmarks'],
    queryFn: () => api.get('/comparison/benchmarks').then(r => r.data),
  })

  const { data: comparisons, isLoading: compLoading, isError: compError } = useQuery({
    queryKey: ['comparisons'],
    queryFn: () => api.get('/comparison/').then(r => r.data),
    refetchInterval: 60_000,
  })

  const compList: any[] = Array.isArray(comparisons) ? comparisons : []
  const benchList: any[] = benchmarks && typeof benchmarks === 'object' && !Array.isArray(benchmarks)
    ? Object.entries(benchmarks).map(([key, v]: any) => ({ key, ...v }))
    : Array.isArray(benchmarks) ? benchmarks : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-bold text-[#e8e8e8]">Strategy Comparison — Manual vs ML</h1>
        <p className="text-xs text-[#888888] mt-0.5">
          Every strategy runs in both manual and ML-enhanced mode. Results are statistically tested (p&lt;0.05).
        </p>
      </div>

      {/* Benchmark Performance Reference */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold text-[#f5a623]">Benchmark Performance Reference</h2>
        </div>
        {benchLoading ? (
          <div className="p-4 space-y-2">
            {[1, 2, 3].map(i => <div key={i} className="h-8 bg-[#1e1e1e] rounded animate-pulse" />)}
          </div>
        ) : benchError ? (
          <div className="p-4">
            <p className="text-xs text-[#ff1744]">Failed to load benchmark data from /api/v1/comparison/benchmarks</p>
          </div>
        ) : benchList.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-center space-y-2">
            <p className="text-sm text-[#888888]">No benchmark data available</p>
            <p className="text-xs text-[#555]">Run backtests to populate benchmark comparisons.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#1e1e1e] text-[#888888]">
                  <th className="text-left px-4 py-2">Benchmark</th>
                  <th className="text-right px-4 py-2">Annual Return</th>
                  <th className="text-right px-4 py-2">Sharpe</th>
                  <th className="text-right px-4 py-2">Max Drawdown</th>
                </tr>
              </thead>
              <tbody>
                {benchList.map((b: any, i: number) => (
                  <tr key={b.key ?? i} className="border-b border-[#1e1e1e]">
                    <td className="px-4 py-2">{b.name ?? b.key ?? '—'}</td>
                    <td className="px-4 py-2 text-right font-mono">{b.annual_return != null ? `${(b.annual_return * 100).toFixed(1)}%` : '—'}</td>
                    <td className="px-4 py-2 text-right font-mono">{b.sharpe ?? '—'}</td>
                    <td className="px-4 py-2 text-right font-mono text-[#ff1744]">{b.max_dd != null ? `${(b.max_dd * 100).toFixed(1)}%` : '—'}</td>
                  </tr>
                ))}
                <tr className="border-b border-[#f5a623]/30">
                  <td className="px-4 py-2 text-[#f5a623] font-bold">QuantEdge Target</td>
                  <td className="px-4 py-2 text-right text-[#f5a623] font-bold">20–35%</td>
                  <td className="px-4 py-2 text-right text-[#f5a623] font-bold">&gt;2.0</td>
                  <td className="px-4 py-2 text-right text-[#f5a623] font-bold">&lt;15%</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Manual vs ML Results */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold text-[#e8e8e8]">Manual vs ML Results</h2>
          <p className="text-xs text-[#555] mt-0.5">Head-to-head comparison from walk-forward backtests. P-value from two-sided t-test.</p>
        </div>
        {compLoading ? (
          <div className="p-4 space-y-2">
            {[1, 2, 3, 4].map(i => <div key={i} className="h-8 bg-[#1e1e1e] rounded animate-pulse" />)}
          </div>
        ) : compError ? (
          <div className="p-4">
            <p className="text-xs text-[#ff1744]">Failed to load comparison data from /api/v1/comparison/</p>
          </div>
        ) : compList.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center space-y-3">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            <p className="text-sm text-[#888888] font-medium">Run backtests in the Backtest Lab to see strategy comparisons</p>
            <p className="text-xs text-[#555] max-w-sm">
              Once you run a backtest for any strategy, the Manual vs ML comparison will appear here automatically.
            </p>
            <a href="/backtest" className="text-xs text-[#f5a623] underline">Go to Backtest Lab</a>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#1e1e1e] text-[#888888]">
                  <th className="text-left px-4 py-2">Strategy</th>
                  <th className="text-right px-4 py-2">Symbol</th>
                  <th className="text-right px-4 py-2">Manual Sharpe</th>
                  <th className="text-right px-4 py-2">ML Sharpe</th>
                  <th className="text-right px-4 py-2">Improvement</th>
                  <th className="text-right px-4 py-2">Significant</th>
                  <th className="text-center px-4 py-2">Winner</th>
                </tr>
              </thead>
              <tbody>
                {compList.map((c: any) => {
                  const improvement = c.manual_sharpe != null && c.ml_sharpe != null
                    ? ((c.ml_sharpe - c.manual_sharpe) / c.manual_sharpe) * 100
                    : c.ml_improvement_pct != null ? c.ml_improvement_pct * 100 : null
                  return (
                    <tr key={c.id ?? c.strategy_name} className="border-b border-[#1e1e1e] hover:bg-[#0d0d0d] transition-colors">
                      <td className="px-4 py-2.5 font-mono text-[#e8e8e8]">{c.strategy_name ?? c.strategy ?? '—'}</td>
                      <td className="px-4 py-2.5 text-right text-[#888]">{c.symbol ?? '—'}</td>
                      <td className="px-4 py-2.5 text-right font-mono">{c.manual_sharpe?.toFixed(3) ?? '—'}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-[#00c853]">{c.ml_sharpe?.toFixed(3) ?? '—'}</td>
                      <td className="px-4 py-2.5 text-right">
                        {improvement != null ? <ImprovementBar pct={improvement} /> : <span className="text-[#555]">—</span>}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        {c.is_significant ? (
                          <span className="text-[#00c853]">✓ p&lt;0.05</span>
                        ) : (
                          <span className="text-[#888888]">—</span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-center font-bold">
                        {c.winner === 'ml' ? '🤖 ML' : c.winner === 'manual' ? '📊 Manual' : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
