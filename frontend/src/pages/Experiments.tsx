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
  progress?: number | null          // 0–100 when status === 'running'
  epochs_done?: number | null
  epochs_total?: number | null
  started_at: string | null
  completed_at: string | null
  // extra metrics shown in comparison
  train_loss?: number | null
  val_loss?: number | null
  model_type?: string | null
  config?: Record<string, unknown> | null
}

// ── Desk filter options ─────────────────────────────────────────────────────
const STATUS_OPTS = ['all', 'running', 'done', 'failed', 'pending'] as const
type StatusFilter = (typeof STATUS_OPTS)[number]

// ── Training Progress Bar ────────────────────────────────────────────────────
function TrainingProgressBar({ exp }: { exp: Experiment }) {
  const progress = exp.progress ?? 0
  const epochText = (exp.epochs_done != null && exp.epochs_total != null)
    ? `${exp.epochs_done}/${exp.epochs_total} epochs`
    : `${progress.toFixed(0)}%`
  return (
    <div className="mt-1">
      <div className="flex justify-between items-center mb-0.5">
        <span className="text-[9px] text-[#555]">Training…</span>
        <span className="text-[9px] text-[#2979ff] font-mono">{epochText}</span>
      </div>
      <div className="h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${Math.min(progress, 100)}%`,
            background: 'linear-gradient(90deg, #2979ff, #00e5ff)',
            boxShadow: '0 0 6px rgba(41,121,255,0.5)',
          }}
        />
      </div>
    </div>
  )
}

// ── Side-by-side Comparison Panel ───────────────────────────────────────────
function ComparePanel({ ids, experiments, onClose }: {
  ids: [string, string]
  experiments: Experiment[]
  onClose: () => void
}) {
  const [a, b] = ids.map(id => experiments.find(e => e.id === id))

  const METRICS: { label: string; key: keyof Experiment; fmt?: (v: unknown) => string }[] = [
    { label: 'Status', key: 'status' },
    { label: 'Val Accuracy', key: 'val_accuracy', fmt: v => v != null ? `${((v as number) * 100).toFixed(1)}%` : '—' },
    { label: 'Val Sharpe', key: 'val_sharpe', fmt: v => v != null ? (v as number).toFixed(3) : '—' },
    { label: 'Test Sharpe', key: 'test_sharpe', fmt: v => v != null ? (v as number).toFixed(3) : '—' },
    { label: 'Train Loss', key: 'train_loss', fmt: v => v != null ? (v as number).toFixed(4) : '—' },
    { label: 'Val Loss', key: 'val_loss', fmt: v => v != null ? (v as number).toFixed(4) : '—' },
    { label: 'Model Type', key: 'model_type', fmt: v => (v as string | null) ?? '—' },
    { label: 'Started', key: 'started_at', fmt: v => v ? new Date(v as string).toLocaleString() : '—' },
    { label: 'Completed', key: 'completed_at', fmt: v => v ? new Date(v as string).toLocaleString() : '—' },
  ]

  function fmtVal(exp: Experiment | undefined, metric: typeof METRICS[0]): string {
    if (!exp) return '—'
    const v = exp[metric.key]
    return metric.fmt ? metric.fmt(v) : String(v ?? '—')
  }

  function winnerClass(va: string, vb: string, metric: typeof METRICS[0]): { aClass: string; bClass: string } {
    // Highlight numeric wins for sharpe / accuracy
    const numericKeys: (keyof Experiment)[] = ['val_accuracy', 'val_sharpe', 'test_sharpe']
    if (!numericKeys.includes(metric.key)) return { aClass: '', bClass: '' }
    const na = parseFloat(va.replace('%', ''))
    const nb = parseFloat(vb.replace('%', ''))
    if (isNaN(na) || isNaN(nb) || na === nb) return { aClass: '', bClass: '' }
    return na > nb
      ? { aClass: 'text-[#00c853] font-bold', bClass: 'text-[#888]' }
      : { aClass: 'text-[#888]', bClass: 'text-[#00c853] font-bold' }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-[#111111] border border-[#2a2a2a] rounded-xl p-6 w-full max-w-2xl shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-bold text-[#e8e8e8]">Experiment Comparison</h2>
          <button onClick={onClose} className="text-[#555] hover:text-[#888] text-lg leading-none">✕</button>
        </div>

        <div className="overflow-hidden rounded-lg border border-[#1e1e1e]">
          <table className="w-full text-xs">
            <thead className="bg-[#0a0a0a]">
              <tr>
                <th className="text-left px-4 py-2.5 text-[#555] font-medium">Metric</th>
                <th className="text-left px-4 py-2.5 text-[#f5a623] font-mono truncate max-w-[160px]">
                  {a?.name ?? ids[0]}
                </th>
                <th className="text-left px-4 py-2.5 text-[#2979ff] font-mono truncate max-w-[160px]">
                  {b?.name ?? ids[1]}
                </th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map(m => {
                const va = fmtVal(a, m)
                const vb = fmtVal(b, m)
                const { aClass, bClass } = winnerClass(va, vb, m)
                return (
                  <tr key={m.key} className="border-t border-[#1e1e1e] hover:bg-[#0d0d0d] transition-colors">
                    <td className="px-4 py-2 text-[#555]">{m.label}</td>
                    <td className={`px-4 py-2 font-mono ${aClass || 'text-[#e8e8e8]'}`}>{va}</td>
                    <td className={`px-4 py-2 font-mono ${bClass || 'text-[#e8e8e8]'}`}>{vb}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded bg-[#1e1e1e] text-[#888] text-xs hover:bg-[#2a2a2a] transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Sortable column header ───────────────────────────────────────────────────
type SortKey = 'name' | 'status' | 'val_accuracy' | 'val_sharpe' | 'test_sharpe' | 'started_at'

function SortableHeader({ label, sortKey, current, dir, onClick }: {
  label: string
  sortKey: SortKey
  current: SortKey
  dir: 'asc' | 'desc'
  onClick: (k: SortKey) => void
}) {
  const active = current === sortKey
  return (
    <th
      className="text-left px-4 py-3 cursor-pointer select-none hover:text-[#e8e8e8] transition-colors"
      onClick={() => onClick(sortKey)}
    >
      <span className="flex items-center gap-1">
        {label}
        <span className="text-[10px]" style={{ color: active ? '#f5a623' : '#333' }}>
          {active ? (dir === 'asc' ? '▲' : '▼') : '⇅'}
        </span>
      </span>
    </th>
  )
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function Experiments() {
  const { data: exps, isLoading } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: () => api.get('/experiments/').then(r => r.data),
    refetchInterval: 5_000,
  })

  const experiments: Experiment[] = exps ?? []

  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [sortKey, setSortKey] = useState<SortKey>('started_at')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [selected, setSelected] = useState<string[]>([])
  const [compareIds, setCompareIds] = useState<[string, string] | null>(null)

  // Toggle selection for comparison (max 2)
  function toggleSelect(id: string) {
    setSelected(prev => {
      if (prev.includes(id)) return prev.filter(x => x !== id)
      if (prev.length >= 2) return [prev[1], id]
      return [...prev, id]
    })
  }

  function handleSort(key: SortKey) {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const filtered = useMemo(() => {
    let list = statusFilter === 'all' ? experiments : experiments.filter(e => e.status === statusFilter)
    list = [...list].sort((a, b) => {
      let va: number | string = 0
      let vb: number | string = 0
      if (sortKey === 'name') { va = a.name; vb = b.name }
      else if (sortKey === 'status') { va = a.status; vb = b.status }
      else if (sortKey === 'val_accuracy') { va = a.val_accuracy ?? -Infinity; vb = b.val_accuracy ?? -Infinity }
      else if (sortKey === 'val_sharpe') { va = a.val_sharpe ?? -Infinity; vb = b.val_sharpe ?? -Infinity }
      else if (sortKey === 'test_sharpe') { va = a.test_sharpe ?? -Infinity; vb = b.test_sharpe ?? -Infinity }
      else if (sortKey === 'started_at') { va = a.started_at ?? ''; vb = b.started_at ?? '' }
      if (va < vb) return sortDir === 'asc' ? -1 : 1
      if (va > vb) return sortDir === 'asc' ? 1 : -1
      return 0
    })
    return list
  }, [experiments, statusFilter, sortKey, sortDir])

  const completedWithSharpe = experiments.filter(e => e.status === 'done' && e.test_sharpe !== null)
  const bestSharpe = completedWithSharpe.length > 0
    ? Math.max(...completedWithSharpe.map(e => e.test_sharpe as number)).toFixed(2)
    : '—'

  const runningCount = experiments.filter(e => e.status === 'running').length

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
          { label: 'Running', value: runningCount, color: '#2979ff' },
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
            {isLoading && (
              Array.from({ length: 3 }).map((_, i) => (
                <tr key={i} className="border-t border-[#1e1e1e]">
                  {Array.from({ length: 8 }).map((_, j) => (
                    <td key={j} className="px-4 py-3">
                      <div className="h-4 bg-[#1e1e1e] rounded animate-pulse" />
                    </td>
                  ))}
                </tr>
              ))
            )}
            {!isLoading && filtered.map(e => {
              const isSelected = selected.includes(e.id)
              return (
                <tr
                  key={e.id}
                  className="border-t border-[#1e1e1e] hover:bg-[#0d0d0d] transition-colors"
                  style={{ background: isSelected ? 'rgba(245,166,35,0.06)' : undefined }}
                >
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleSelect(e.id)}
                      className="w-3 h-3 accent-[#f5a623] cursor-pointer"
                    />
                  </td>
                  <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{e.name}</td>
                  <td className="px-4 py-3">
                    <div>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        e.status === 'done'    ? 'bg-[#00c853]/20 text-[#00c853]' :
                        e.status === 'running' ? 'bg-[#2979ff]/20 text-[#2979ff]' :
                        e.status === 'failed'  ? 'bg-[#ff1744]/20 text-[#ff1744]' :
                                                 'bg-[#1e1e1e] text-[#888888]'}`}>
                        {e.status === 'running' ? '● ' : ''}{e.status}
                      </span>
                      {e.status === 'running' && <TrainingProgressBar exp={e} />}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-xs">{e.val_accuracy != null ? `${(e.val_accuracy * 100).toFixed(1)}%` : '—'}</td>
                  <td className="px-4 py-3 text-xs">{e.val_sharpe != null ? e.val_sharpe.toFixed(3) : '—'}</td>
                  <td className="px-4 py-3 text-xs text-[#00c853] font-bold">{e.test_sharpe != null ? e.test_sharpe.toFixed(3) : '—'}</td>
                  <td className="px-4 py-3 text-xs text-[#888888]">{e.started_at ? new Date(e.started_at).toLocaleString() : '—'}</td>
                  <td className="px-4 py-3 text-xs text-[#888888]">{e.completed_at ? new Date(e.completed_at).toLocaleString() : '—'}</td>
                </tr>
              )
            })}
            {!isLoading && filtered.length === 0 && (
              <tr>
                <td colSpan={8} className="px-4 py-10 text-center">
                  <p className="text-xs text-[#555]">
                    {statusFilter !== 'all'
                      ? `No ${statusFilter} experiments found.`
                      : 'No experiments yet. Run: python experiments/run_experiment.py --config lstm_btc_1h.yaml'}
                  </p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
