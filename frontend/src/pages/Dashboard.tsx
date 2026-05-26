import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

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
    </div>
  )
}
