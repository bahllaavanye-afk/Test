import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

interface Experiment {
  id: string
  name: string
  status: 'done' | 'running' | 'failed' | 'pending'
  val_accuracy: number | null
  val_sharpe: number | null
  test_sharpe: number | null
  started_at: string | null
  completed_at: string | null
}

export default function Experiments() {
  const { data: exps } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: () => api.get('/experiments/').then(r => r.data),
    refetchInterval: 5_000,
  })

  const experiments: Experiment[] = exps ?? []

  // Only consider done experiments with a non-null test_sharpe for best Sharpe
  const completedWithSharpe = experiments.filter(
    e => e.status === 'done' && e.test_sharpe !== null
  )
  const bestSharpe = completedWithSharpe.length > 0
    ? Math.max(...completedWithSharpe.map(e => e.test_sharpe as number)).toFixed(2)
    : '—'

  return (
    <div className="space-y-6">
      {compareIds && (
        <ComparePanel ids={compareIds} experiments={experiments} onClose={() => setCompareIds(null)} />
      )}

      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold">ML Experiments</h1>
        <div className="flex items-center gap-3">
          {selected.length === 2 && (
            <button
              onClick={() => setCompareIds([selected[0], selected[1]])}
              className="px-3 py-1.5 rounded text-xs font-bold bg-[#f5a623] text-black hover:opacity-90 transition-opacity"
            >
              Compare Selected
            </button>
          )}
          {selected.length > 0 && (
            <button
              onClick={() => setSelected([])}
              className="px-3 py-1.5 rounded text-xs text-[#555] bg-[#1e1e1e] hover:bg-[#2a2a2a] transition-colors"
            >
              Clear ({selected.length})
            </button>
          )}
          <span className="text-xs text-[#888888]">Auto-refreshes every 5s · MLflow + PyTorch Lightning</span>
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Total Runs', value: experiments.length, color: '#f5a623' },
          { label: 'Completed', value: experiments.filter(e => e.status === 'done').length, color: '#00c853' },
          { label: 'Running', value: experiments.filter(e => e.status === 'running').length, color: '#2979ff' },
          { label: 'Best Sharpe', value: bestSharpe, color: '#9C27B0' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
            <p className="text-xs text-[#888888]">{label}</p>
            <p className="text-xl font-bold mt-1" style={{ color }}>{value}</p>
          </div>
        ))}
      </div>

      {/* Filter bar */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-[#555]">Filter:</span>
        {STATUS_OPTS.map(s => (
          <button
            key={s}
            onClick={() => setStatusFilter(s)}
            className="px-3 py-1 rounded text-xs capitalize transition-colors"
            style={{
              background: statusFilter === s ? 'rgba(245,166,35,0.15)' : '#111',
              color: statusFilter === s ? '#f5a623' : '#555',
              border: `1px solid ${statusFilter === s ? 'rgba(245,166,35,0.35)' : '#1e1e1e'}`,
            }}
          >
            {s}
          </button>
        ))}
        {selected.length > 0 && (
          <span className="ml-auto text-xs text-[#555]">
            {selected.length}/2 selected for comparison
          </span>
        )}
      </div>

      {/* Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-[#0a0a0a]">
            <tr className="text-xs text-[#888888]">
              <th className="text-left px-4 py-3 w-8">
                <span className="text-[#444]">Sel</span>
              </th>
              <SortableHeader label="Name" sortKey="name" current={sortKey} dir={sortDir} onClick={handleSort} />
              <SortableHeader label="Status" sortKey="status" current={sortKey} dir={sortDir} onClick={handleSort} />
              <SortableHeader label="Val Acc" sortKey="val_accuracy" current={sortKey} dir={sortDir} onClick={handleSort} />
              <SortableHeader label="Val Sharpe" sortKey="val_sharpe" current={sortKey} dir={sortDir} onClick={handleSort} />
              <SortableHeader label="Test Sharpe" sortKey="test_sharpe" current={sortKey} dir={sortDir} onClick={handleSort} />
              <SortableHeader label="Started" sortKey="started_at" current={sortKey} dir={sortDir} onClick={handleSort} />
              <th className="text-left px-4 py-3">Completed</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map(e => (
              <tr key={e.id} className="border-t border-[#1e1e1e] hover:bg-[#111111]/50 transition-colors">
                <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{e.name}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    e.status === 'done' ? 'bg-[#00c853]/20 text-[#00c853]' :
                    e.status === 'running' ? 'bg-[#2979ff]/20 text-[#2979ff]' :
                    e.status === 'failed' ? 'bg-[#ff1744]/20 text-[#ff1744]' :
                    'bg-[#1e1e1e] text-[#888888]'}`}>
                    {e.status === 'running' ? '● ' : ''}{e.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs">{e.val_accuracy != null ? `${(e.val_accuracy * 100).toFixed(1)}%` : '—'}</td>
                <td className="px-4 py-3 text-xs">{e.val_sharpe != null ? e.val_sharpe.toFixed(3) : '—'}</td>
                <td className="px-4 py-3 text-xs text-[#00c853] font-bold">{e.test_sharpe != null ? e.test_sharpe.toFixed(3) : '—'}</td>
                <td className="px-4 py-3 text-xs text-[#888888]">{e.started_at ? new Date(e.started_at).toLocaleString() : '—'}</td>
                <td className="px-4 py-3 text-xs text-[#888888]">{e.completed_at ? new Date(e.completed_at).toLocaleString() : '—'}</td>
              </tr>
            ))}
            {experiments.length === 0 && (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-xs text-[#888888]">No experiments yet. Run: python experiments/run_experiment.py --config lstm_btc_1h.yaml</td></tr>
            )}
            {experiments.length > 0 && completedWithSharpe.length === 0 && (
              <tr><td colSpan={7} className="px-4 py-4 text-center text-xs text-[#555]">No completed experiments — Best Sharpe will appear once a run finishes</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
