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

const STATUS_OPTS = ['all', 'done', 'running', 'failed', 'pending'] as const
type StatusOpt = typeof STATUS_OPTS[number]
type SortKey = keyof Experiment
type SortDir = 'asc' | 'desc'

function SortableHeader({
  label,
  sortKey: sk,
  current,
  dir,
  onClick,
}: {
  label: string
  sortKey: SortKey
  current: SortKey
  dir: SortDir
  onClick: (key: SortKey) => void
}) {
  const active = current === sk
  return (
    <th
      className="text-left px-4 py-3 cursor-pointer select-none hover:text-[#f5a623] transition-colors"
      onClick={() => onClick(sk)}
    >
      {label}
      {active && (
        <span className="ml-1 text-[#f5a623]">{dir === 'asc' ? '↑' : '↓'}</span>
      )}
    </th>
  )
}

function ComparePanel({
  ids,
  experiments,
  onClose,
}: {
  ids: [string, string]
  experiments: Experiment[]
  onClose: () => void
}) {
  const a = experiments.find(e => e.id === ids[0])
  const b = experiments.find(e => e.id === ids[1])
  if (!a || !b) return null

  const rows: { label: string; va: string; vb: string }[] = [
    { label: 'Status', va: a.status, vb: b.status },
    { label: 'Val Accuracy', va: a.val_accuracy != null ? `${(a.val_accuracy * 100).toFixed(1)}%` : '—', vb: b.val_accuracy != null ? `${(b.val_accuracy * 100).toFixed(1)}%` : '—' },
    { label: 'Val Sharpe', va: a.val_sharpe?.toFixed(3) ?? '—', vb: b.val_sharpe?.toFixed(3) ?? '—' },
    { label: 'Test Sharpe', va: a.test_sharpe?.toFixed(3) ?? '—', vb: b.test_sharpe?.toFixed(3) ?? '—' },
  ]

  return (
    <div className="bg-[#111] border border-[#f5a623]/30 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-bold text-[#f5a623]">Comparison</h2>
        <button onClick={onClose} className="text-xs text-[#555] hover:text-[#888]">✕ Close</button>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-[#888]">
            <th className="text-left py-1 w-32">Metric</th>
            <th className="text-left py-1 font-mono text-[#e8e8e8]">{a.name}</th>
            <th className="text-left py-1 font-mono text-[#e8e8e8]">{b.name}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ label, va, vb }) => (
            <tr key={label} className="border-t border-[#1e1e1e]">
              <td className="py-1.5 text-[#555]">{label}</td>
              <td className="py-1.5">{va}</td>
              <td className="py-1.5">{vb}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Experiments() {
  const { data: exps } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: () => api.get('/experiments/').then(r => r.data),
    refetchInterval: 5_000,
  })

  const experiments: Experiment[] = exps ?? []

  const [statusFilter, setStatusFilter] = useState<StatusOpt>('all')
  const [selected, setSelected] = useState<string[]>([])
  const [compareIds, setCompareIds] = useState<[string, string] | null>(null)
  const [sortKey, setSortKey] = useState<SortKey>('started_at')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  function toggleSelect(id: string) {
    setSelected(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : prev.length < 2 ? [...prev, id] : prev
    )
  }

  const filtered = useMemo(() => {
    const base = statusFilter === 'all' ? experiments : experiments.filter(e => e.status === statusFilter)
    return [...base].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (av == null && bv == null) return 0
      if (av == null) return sortDir === 'asc' ? -1 : 1
      if (bv == null) return sortDir === 'asc' ? 1 : -1
      if (typeof av === 'string' && typeof bv === 'string') {
        return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      }
      return sortDir === 'asc' ? (av as number) - (bv as number) : (bv as number) - (av as number)
    })
  }, [experiments, statusFilter, sortKey, sortDir])

  const completedWithSharpe = experiments.filter(e => e.status === 'done' && e.test_sharpe !== null)
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
            {filtered.map(e => (
              <tr key={e.id} className={`border-t border-[#1e1e1e] hover:bg-[#1a1a1a] transition-colors ${selected.includes(e.id) ? 'bg-[#f5a623]/5' : ''}`}>
                <td className="px-4 py-3">
                  <input
                    type="checkbox"
                    checked={selected.includes(e.id)}
                    onChange={() => toggleSelect(e.id)}
                    disabled={selected.length >= 2 && !selected.includes(e.id)}
                    className="accent-[#f5a623] cursor-pointer disabled:cursor-not-allowed"
                  />
                </td>
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
            {filtered.length === 0 && experiments.length === 0 && (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-xs text-[#888888]">No experiments yet. Run: python experiments/run_experiment.py --config lstm_btc_1h.yaml</td></tr>
            )}
            {filtered.length === 0 && experiments.length > 0 && (
              <tr><td colSpan={8} className="px-4 py-4 text-center text-xs text-[#555]">No experiments match the current filter.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
