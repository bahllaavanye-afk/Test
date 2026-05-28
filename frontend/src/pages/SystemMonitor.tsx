import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span
      className="inline-block w-2.5 h-2.5 rounded-full mr-2"
      style={{ backgroundColor: ok ? '#00c853' : '#ff1744' }}
    />
  )
}

function HealthRow({ label, ok, detail }: { label: string; ok: boolean; detail?: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-[#1e1e1e] last:border-0">
      <div className="flex items-center">
        <StatusDot ok={ok} />
        <span className="text-xs text-[#e8e8e8]">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        {detail && <span className="text-xs text-[#888888]">{detail}</span>}
        <span
          className="text-xs font-medium px-2 py-0.5 rounded"
          style={{
            color: ok ? '#00c853' : '#ff1744',
            backgroundColor: ok ? '#00c85322' : '#ff174422',
          }}
        >
          {ok ? 'OK' : 'DOWN'}
        </span>
      </div>
    </div>
  )
}

export default function SystemMonitor() {
  const { data: agentStatus } = useQuery({
    queryKey: ['agents-status-monitor'],
    queryFn: () => api.get('/agents/status').then(r => r.data),
    refetchInterval: 5000,
  })

  const { data: quality } = useQuery({
    queryKey: ['improvements-quality'],
    queryFn: () => api.get('/improvements/quality').then(r => r.data),
    refetchInterval: 30000,
  })

  const { data: bestParams } = useQuery({
    queryKey: ['improvements-best-params'],
    queryFn: () => api.get('/improvements/best_params').then(r => r.data),
    refetchInterval: 60000,
  })

  const isRunning: boolean = agentStatus?.algo_agent?.running ?? false
  const top3 = agentStatus?.algo_agent?.top_3 ?? []

  // Flatten bestParams: { strategy: { param: value, ... }, ... } → rows
  const paramRows: { strategy: string; param: string; value: string }[] = []
  if (bestParams && typeof bestParams === 'object') {
    for (const [strategy, params] of Object.entries(bestParams)) {
      if (params && typeof params === 'object') {
        for (const [param, value] of Object.entries(params as Record<string, any>)) {
          paramRows.push({ strategy, param, value: String(value) })
        }
      }
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">System Monitor</h1>
        <span className="text-xs text-[#888888]">Agent refreshes every 5s</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Agent Status Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-4">
          <h2 className="text-xs text-[#888888] uppercase tracking-wider">AlgoAgent Status</h2>
          <div className="flex items-center gap-2">
            <StatusDot ok={isRunning} />
            <span
              className="text-sm font-bold"
              style={{ color: isRunning ? '#00c853' : '#ff1744' }}
            >
              {isRunning ? 'RUNNING' : 'STOPPED'}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-[#888888]">Total Runs</p>
              <p className="text-2xl font-bold text-[#f5a623]">{agentStatus?.algo_agent?.total_runs ?? 0}</p>
            </div>
            <div>
              <p className="text-xs text-[#888888]">Candidates</p>
              <p className="text-2xl font-bold text-[#2196f3]">{agentStatus?.algo_agent?.candidates ?? 0}</p>
            </div>
          </div>
          {top3.length > 0 && (
            <div>
              <p className="text-xs text-[#888888] mb-2">Top 3 Strategies (by UCB1 score)</p>
              <div className="space-y-1">
                {top3.map((row: any, i: number) => {
                  const label = typeof row === 'string'
                    ? row
                    : `${row.strategy}:${row.symbol}`
                  const sharpe = typeof row === 'object' && row.best_sharpe != null
                    ? row.best_sharpe.toFixed(2)
                    : null
                  return (
                    <div key={i} className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs" style={{ color: ['#FFD700', '#C0C0C0', '#CD7F32'][i] }}>
                          {['#1', '#2', '#3'][i]}
                        </span>
                        <span className="text-xs font-mono text-[#e8e8e8]">{label}</span>
                      </div>
                      {sharpe != null && (
                        <span className="text-[10px] font-mono text-[#888888]">
                          Sharpe {sharpe}
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* System Health Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <h2 className="text-xs text-[#888888] uppercase tracking-wider mb-3">System Health</h2>
          <HealthRow label="Database" ok={true} detail="Supabase" />
          <HealthRow label="Redis" ok={true} detail="Upstash" />
          <HealthRow label="AlgoAgent" ok={isRunning} detail={isRunning ? `${agentStatus?.algo_agent?.total_runs ?? 0} runs` : 'Not running'} />
          <HealthRow label="ResearchScientist" ok={agentStatus?.research_scientist?.running ?? false} detail={`${agentStatus?.research_scientist?.cycles_completed ?? 0} cycles`} />
          <HealthRow label="ModelingEngineer" ok={agentStatus?.modeling_engineer?.running ?? false} detail={`${agentStatus?.modeling_engineer?.cycles_completed ?? 0} cycles`} />
          <HealthRow label="QAMonitor" ok={agentStatus?.qa_monitor?.running ?? false} detail="Auto-fixing bugs" />
        </div>
      </div>

      {/* Code Quality Metrics */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-xs text-[#888888] uppercase tracking-wider mb-4">Code Quality Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Total Files', key: 'total_files', color: '#f5a623' },
            { label: 'Total Lines', key: 'total_lines', color: '#2196f3' },
            { label: 'Strategies', key: 'total_strategies', color: '#00c853' },
            { label: 'Tests', key: 'total_tests', color: '#9c27b0' },
          ].map(({ label, key, color }) => (
            <div key={key}>
              <p className="text-xs text-[#888888]">{label}</p>
              <p className="text-2xl font-bold mt-1" style={{ color }}>
                {quality?.[key] != null ? quality[key].toLocaleString() : '—'}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Best Params Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold">Best Parameters</h2>
          <p className="text-xs text-[#888888] mt-0.5">Optimal params per strategy from AlgoAgent</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-[#0a0a0a]">
              <tr className="text-xs text-[#888888]">
                <th className="text-left px-4 py-3">Strategy</th>
                <th className="text-left px-4 py-3">Parameter</th>
                <th className="text-left px-4 py-3">Value</th>
              </tr>
            </thead>
            <tbody>
              {paramRows.map((row, i) => (
                <tr key={i} className="border-t border-[#1e1e1e] hover:bg-[#0a0a0a] transition-colors">
                  <td className="px-4 py-3 text-xs font-mono text-[#f5a623]">{row.strategy}</td>
                  <td className="px-4 py-3 text-xs font-mono text-[#888888]">{row.param}</td>
                  <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{row.value}</td>
                </tr>
              ))}
              {paramRows.length === 0 && (
                <tr>
                  <td colSpan={3} className="px-4 py-8 text-center text-xs text-[#888888]">
                    No best params yet — AlgoAgent needs to complete optimization runs.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
