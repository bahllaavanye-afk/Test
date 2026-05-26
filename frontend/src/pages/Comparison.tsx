import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

export default function Comparison() {
  const { data: benchmarks } = useQuery({ queryKey: ['benchmarks'], queryFn: () => api.get('/comparison/benchmarks').then(r => r.data) })
  const { data: comparisons } = useQuery({ queryKey: ['comparisons'], queryFn: () => api.get('/comparison/').then(r => r.data), refetchInterval: 60_000 })

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-bold">Strategy Comparison — Manual vs ML</h1>
      <p className="text-xs text-[#888888]">Every strategy runs in both manual and ML-enhanced mode. Results are statistically tested (p&lt;0.05).</p>

      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-4 text-[#f5a623]">Benchmark Performance Reference</h2>
        <table className="w-full">
          <thead>
            <tr className="border-b border-[#1e1e1e] text-xs text-[#888888]">
              <th className="text-left py-2">Benchmark</th>
              <th className="text-right py-2">Annual Return</th>
              <th className="text-right py-2">Sharpe</th>
              <th className="text-right py-2">Max Drawdown</th>
            </tr>
          </thead>
          <tbody>
            {benchmarks && Object.entries(benchmarks).map(([key, v]: any) => (
              <tr key={key} className="border-b border-[#1e1e1e] text-xs">
                <td className="py-2">{v.name}</td>
                <td className="py-2 text-right">{(v.annual_return * 100).toFixed(1)}%</td>
                <td className="py-2 text-right">{v.sharpe}</td>
                <td className="py-2 text-right text-[#ff1744]">{(v.max_dd * 100).toFixed(1)}%</td>
              </tr>
            ))}
            <tr className="border-b border-[#f5a623]/30 text-xs">
              <td className="py-2 text-[#f5a623] font-bold">QuantEdge Target</td>
              <td className="py-2 text-right text-[#f5a623] font-bold">20–35%</td>
              <td className="py-2 text-right text-[#f5a623] font-bold">&gt;2.0</td>
              <td className="py-2 text-right text-[#f5a623] font-bold">&lt;15%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-4">Manual vs ML Results</h2>
        {comparisons?.length === 0 && <p className="text-xs text-[#888888]">Run backtests to see comparison results here.</p>}
        <table className="w-full">
          <thead>
            <tr className="border-b border-[#1e1e1e] text-xs text-[#888888]">
              <th className="text-left py-2">Strategy</th>
              <th className="text-right py-2">Symbol</th>
              <th className="text-right py-2">Manual Sharpe</th>
              <th className="text-right py-2">ML Sharpe</th>
              <th className="text-right py-2">Improvement</th>
              <th className="text-right py-2">Significant</th>
              <th className="text-right py-2">Winner</th>
            </tr>
          </thead>
          <tbody>
            {comparisons?.map((c: any) => (
              <tr key={c.id} className="border-b border-[#1e1e1e] text-xs">
                <td className="py-2 font-mono">{c.strategy_name}</td>
                <td className="py-2 text-right">{c.symbol}</td>
                <td className="py-2 text-right">{c.manual_sharpe?.toFixed(3) ?? '—'}</td>
                <td className="py-2 text-right text-[#00c853]">{c.ml_sharpe?.toFixed(3) ?? '—'}</td>
                <td className="py-2 text-right">
                  {c.ml_improvement_pct != null
                    ? <span className={c.ml_improvement_pct > 0 ? 'text-[#00c853]' : 'text-[#ff1744]'}>{c.ml_improvement_pct > 0 ? '+' : ''}{(c.ml_improvement_pct * 100).toFixed(1)}%</span>
                    : '—'}
                </td>
                <td className="py-2 text-right">{c.is_significant ? <span className="text-[#00c853]">✓ p&lt;0.05</span> : <span className="text-[#888888]">—</span>}</td>
                <td className="py-2 text-right font-bold">{c.winner === 'ml' ? '🤖 ML' : c.winner === 'manual' ? '📊 Manual' : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
