import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

const RANK_COLORS: Record<number, string> = { 1: '#FFD700', 2: '#C0C0C0', 3: '#CD7F32' }

export default function Leaderboard() {
  const { data: board } = useQuery({
    queryKey: ['leaderboard'],
    queryFn: () => api.get('/agents/leaderboard').then(r => r.data),
    refetchInterval: 10000,
  })

  const { data: agentStatus } = useQuery({
    queryKey: ['agents-status'],
    queryFn: () => api.get('/agents/status').then(r => r.data),
  })

  const { data: history } = useQuery({
    queryKey: ['improvements-history'],
    queryFn: () => api.get('/improvements/history').then(r => r.data),
    refetchInterval: 30000,
  })

  const sorted: any[] = [...(board ?? [])].sort((a, b) => (b.best_sharpe ?? 0) - (a.best_sharpe ?? 0))
  const recent: any[] = (history ?? []).slice(0, 10)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">Strategy Leaderboard</h1>
        <span className="text-xs text-[#888888]">Refreshes every 10s</span>
      </div>

      {/* Agent Status */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-xs text-[#888888] uppercase tracking-wider mb-3">AlgoAgent Status</h2>
        <div className="flex flex-wrap gap-6 items-center">
          <div className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full inline-block"
              style={{ backgroundColor: agentStatus?.running ? '#00c853' : '#ff1744' }}
            />
            <span className="text-xs font-medium" style={{ color: agentStatus?.running ? '#00c853' : '#ff1744' }}>
              {agentStatus?.running ? 'RUNNING' : 'STOPPED'}
            </span>
          </div>
          <div>
            <p className="text-xs text-[#888888]">Total Runs</p>
            <p className="text-xl font-bold text-[#f5a623]">{agentStatus?.total_runs ?? 0}</p>
          </div>
          <div>
            <p className="text-xs text-[#888888]">Candidates</p>
            <p className="text-xl font-bold text-[#2196f3]">{agentStatus?.candidates ?? 0}</p>
          </div>
        </div>
      </div>

      {/* Leaderboard Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold">Rankings</h2>
        </div>
        <table className="w-full">
          <thead className="bg-[#0a0a0a]">
            <tr className="text-xs text-[#888888]">
              {['#', 'Strategy', 'Symbol', 'Runs', 'Avg Sharpe', 'Best Sharpe', 'UCB Score'].map(h => (
                <th key={h} className="text-left px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row: any, i: number) => {
              const rank = i + 1
              const rankColor = RANK_COLORS[rank]
              return (
                <tr key={row.name ?? i} className="border-t border-[#1e1e1e] hover:bg-[#0a0a0a] transition-colors">
                  <td className="px-4 py-3 text-xs font-bold" style={{ color: rankColor ?? '#888888' }}>
                    {rank <= 3 ? ['🥇', '🥈', '🥉'][rank - 1] : `#${rank}`}
                  </td>
                  <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{row.name ?? '—'}</td>
                  <td className="px-4 py-3 text-xs font-mono text-[#f5a623]">{row.symbol ?? '—'}</td>
                  <td className="px-4 py-3 text-xs text-[#888888]">{row.n_runs ?? 0}</td>
                  <td className="px-4 py-3 text-xs">{row.avg_sharpe?.toFixed(3) ?? '—'}</td>
                  <td className="px-4 py-3 text-xs font-bold" style={{ color: rankColor ?? '#00c853' }}>
                    {row.best_sharpe?.toFixed(3) ?? '—'}
                  </td>
                  <td className="px-4 py-3 text-xs text-[#9c27b0]">{row.ucb_score?.toFixed(4) ?? '—'}</td>
                </tr>
              )
            })}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-xs text-[#888888]">
                  No strategies ranked yet — AlgoAgent needs at least one run.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Improvement History */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold">Improvement History</h2>
          <p className="text-xs text-[#888888] mt-0.5">Last 10 upgrades</p>
        </div>
        <table className="w-full">
          <thead className="bg-[#0a0a0a]">
            <tr className="text-xs text-[#888888]">
              {['Time', 'Strategy', 'Sharpe Change', 'Params Changed'].map(h => (
                <th key={h} className="text-left px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {recent.map((h: any, i: number) => (
              <tr key={i} className="border-t border-[#1e1e1e] hover:bg-[#0a0a0a] transition-colors">
                <td className="px-4 py-3 text-xs text-[#888888] whitespace-nowrap">
                  {h.timestamp ? new Date(h.timestamp).toLocaleString() : '—'}
                </td>
                <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{h.strategy ?? '—'}</td>
                <td className="px-4 py-3 text-xs">
                  <span className="text-[#888888]">{h.old_sharpe?.toFixed(3) ?? '—'}</span>
                  <span className="text-[#888888] mx-1">→</span>
                  <span className="text-[#00c853] font-bold">{h.new_sharpe?.toFixed(3) ?? '—'}</span>
                  {h.old_sharpe != null && h.new_sharpe != null && (
                    <span className="ml-2 text-[#00c853]">
                      (+{(h.new_sharpe - h.old_sharpe).toFixed(3)})
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-xs text-[#888888]">
                  {Array.isArray(h.params_changed)
                    ? h.params_changed.join(', ')
                    : typeof h.params_changed === 'object' && h.params_changed
                    ? Object.keys(h.params_changed).join(', ')
                    : h.params_changed ?? '—'}
                </td>
              </tr>
            ))}
            {recent.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-xs text-[#888888]">
                  No improvement history yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
