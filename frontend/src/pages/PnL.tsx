import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

function KpiCard({ label, value, color = '#f5a623', sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      <p className="text-xs text-[#888888] uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold mt-1" style={{ color }}>{value}</p>
      {sub && <p className="text-xs text-[#888888] mt-1">{sub}</p>}
    </div>
  )
}

export default function PnL() {
  const { data: perf } = useQuery({
    queryKey: ['performance'],
    queryFn: () => api.get('/analytics/performance').then(r => r.data),
    refetchInterval: 10000,
  })

  const { data: slippage } = useQuery({
    queryKey: ['slippage'],
    queryFn: () => api.get('/analytics/slippage').then(r => r.data),
  })

  const { data: trades } = useQuery({
    queryKey: ['trades'],
    queryFn: () => api.get('/trades/').then(r => r.data),
    refetchInterval: 10000,
  })

  const tradeList: any[] = trades ?? []
  const winCount = tradeList.filter((t: any) => (t.realized_pnl ?? 0) > 0).length
  const winRate = tradeList.length > 0 ? ((winCount / tradeList.length) * 100).toFixed(1) : '0.0'

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">P&amp;L Dashboard</h1>
        <span className="text-xs text-[#888888]">Refreshes every 10s</span>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          label="Total P&L"
          value={`$${(perf?.total_pnl ?? 0).toFixed(2)}`}
          color={(perf?.total_pnl ?? 0) >= 0 ? '#00c853' : '#ff1744'}
        />
        <KpiCard
          label="Total Trades"
          value={String(perf?.total_trades ?? tradeList.length)}
          color="#f5a623"
          sub={`${winCount} winners`}
        />
        <KpiCard
          label="Avg P&L / Trade"
          value={`$${(perf?.avg_pnl ?? 0).toFixed(2)}`}
          color={(perf?.avg_pnl ?? 0) >= 0 ? '#00c853' : '#ff1744'}
        />
        <KpiCard
          label="Win Rate"
          value={`${winRate}%`}
          color={parseFloat(winRate) >= 50 ? '#00c853' : '#ff1744'}
          sub={`${winCount} / ${tradeList.length}`}
        />
      </div>

      {/* Slippage Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold">Slippage by Execution Algorithm</h2>
        </div>
        <table className="w-full">
          <thead className="bg-[#0a0a0a]">
            <tr className="text-xs text-[#888888]">
              <th className="text-left px-4 py-3">Algorithm</th>
              <th className="text-right px-4 py-3">Avg Slippage (bps)</th>
              <th className="text-right px-4 py-3">Orders</th>
            </tr>
          </thead>
          <tbody>
            {(slippage ?? []).map((s: any) => (
              <tr key={s.execution_algo ?? s.algo} className="border-t border-[#1e1e1e] hover:bg-[#0a0a0a] transition-colors">
                <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{s.execution_algo ?? s.algo ?? '—'}</td>
                <td className="px-4 py-3 text-xs text-right text-[#f5a623]">{s.avg_bps ?? '—'}</td>
                <td className="px-4 py-3 text-xs text-right text-[#888888]">{s.count ?? '—'}</td>
              </tr>
            ))}
            {!(slippage ?? []).length && (
              <tr>
                <td colSpan={3} className="px-4 py-6 text-center text-xs text-[#888888]">
                  No slippage data yet — place orders to populate this table.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Recent Trades Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold">Recent Trades</h2>
        </div>
        <table className="w-full">
          <thead className="bg-[#0a0a0a]">
            <tr className="text-xs text-[#888888]">
              <th className="text-left px-4 py-3">Symbol</th>
              <th className="text-left px-4 py-3">Side</th>
              <th className="text-right px-4 py-3">Realized P&amp;L</th>
              <th className="text-right px-4 py-3">Filled At</th>
            </tr>
          </thead>
          <tbody>
            {tradeList.slice(0, 50).map((t: any, i: number) => (
              <tr key={t.id ?? i} className="border-t border-[#1e1e1e] hover:bg-[#0a0a0a] transition-colors">
                <td className="px-4 py-3 text-xs font-mono font-bold text-[#e8e8e8]">{t.symbol ?? '—'}</td>
                <td className="px-4 py-3 text-xs">
                  <span
                    className="px-2 py-0.5 rounded font-medium"
                    style={{
                      color: t.side === 'buy' ? '#00c853' : '#ff1744',
                      backgroundColor: t.side === 'buy' ? '#00c85322' : '#ff174422',
                    }}
                  >
                    {(t.side ?? '—').toUpperCase()}
                  </span>
                </td>
                <td
                  className="px-4 py-3 text-xs text-right font-bold"
                  style={{ color: (t.realized_pnl ?? 0) >= 0 ? '#00c853' : '#ff1744' }}
                >
                  {t.realized_pnl != null ? `$${t.realized_pnl.toFixed(2)}` : '—'}
                </td>
                <td className="px-4 py-3 text-xs text-right text-[#888888]">
                  {t.filled_at ? new Date(t.filled_at).toLocaleString() : '—'}
                </td>
              </tr>
            ))}
            {tradeList.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-xs text-[#888888]">
                  No trades yet — execute orders to see P&amp;L here.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
