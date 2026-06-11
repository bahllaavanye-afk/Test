/**
 * ExperimentTracker — experiment list with status, metrics, YAML config viewer,
 * and "Run New Experiment" modal.
 *
 * Calls:
 *   GET  /experiments/          — list of experiments
 *   POST /experiments/          — submit new experiment config (YAML string in body)
 */
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../api/client'

interface Experiment {
  id: string
  name: string
  model_type?: string | null
  status: 'queued' | 'running' | 'done' | 'failed'
  val_sharpe?: number | null
  test_sharpe?: number | null
  started_at?: string | null
  config?: string | null        // YAML string or object
}

// ─── Status Badge ─────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: Experiment['status'] }) {
  const styles: Record<Experiment['status'], { bg: string; color: string; label: string }> = {
    queued:  { bg: '#88888822', color: '#888888', label: 'Queued' },
    running: { bg: '#2979ff22', color: '#2979ff', label: 'Running' },
    done:    { bg: '#00c85322', color: '#00c853', label: 'Done' },
    failed:  { bg: '#ff174422', color: '#ff1744', label: 'Failed' },
  }
  const s = styles[status] ?? styles.queued

  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium"
      style={{ background: s.bg, color: s.color, border: `1px solid ${s.color}33` }}
    >
      {status === 'running' && (
        <span
          className="w-2 h-2 rounded-full inline-block animate-pulse"
          style={{ background: s.color }}
        />
      )}
      {s.label}
    </span>
  )
}

// ─── New Experiment Modal ─────────────────────────────────────────────────────

const DEFAULT_YAML = `name: my_experiment
model: lstm
symbol: BTCUSDT
interval: 1h
seq_len: 60
hidden_size: 128
epochs: 50
learning_rate: 0.001
`

function NewExperimentModal({
  onClose,
  onSubmit,
  isSubmitting,
}: {
  onClose: () => void
  onSubmit: (yaml: string) => void
  isSubmitting: boolean
}) {
  const [yaml, setYaml] = useState(DEFAULT_YAML)

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.75)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="w-full max-w-lg rounded-xl shadow-2xl p-6 space-y-4"
        style={{ background: '#1e2433', border: '1px solid #ffffff1a', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold text-white">Run New Experiment</h3>
          <button
            onClick={onClose}
            className="text-[#555] hover:text-white transition-colors text-lg leading-none"
          >
            ×
          </button>
        </div>

        <div>
          <label className="block text-[10px] text-[#555] uppercase tracking-widest mb-2">
            Experiment Config (YAML)
          </label>
          <textarea
            value={yaml}
            onChange={(e) => setYaml(e.target.value)}
            rows={12}
            className="w-full rounded-lg px-3 py-2 text-xs text-[#ccc] resize-none outline-none focus:ring-1"
            style={{
              background: '#131722',
              border: '1px solid #ffffff1a',
              fontFamily: 'ui-monospace, SFMono-Regular, monospace',
              lineHeight: '1.6',
            }}
            spellCheck={false}
          />
        </div>

        <div className="flex gap-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-sm text-[#888] hover:text-white transition-colors"
            style={{ border: '1px solid #ffffff1a' }}
          >
            Cancel
          </button>
          <button
            onClick={() => onSubmit(yaml)}
            disabled={isSubmitting || yaml.trim() === ''}
            className="px-4 py-2 rounded-lg text-sm font-medium text-white transition-all disabled:opacity-50"
            style={{ background: '#2979ff', border: '1px solid #2979ff88' }}
          >
            {isSubmitting ? (
              <span className="flex items-center gap-2">
                <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Submitting…
              </span>
            ) : 'Submit'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function ExperimentTracker() {
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [showModal, setShowModal] = useState(false)
  const queryClient = useQueryClient()

  const { data: experiments, isLoading, isError } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: async () => {
      const res = await api.get('/experiments/')
      return res.data
    },
    refetchInterval: 15_000,
    retry: false,
  })

  const { mutate: submitExperiment, isPending: isSubmitting } = useMutation({
    mutationFn: async (yaml: string) => {
      const res = await api.post('/experiments/', { config: yaml })
      return res.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
      setShowModal(false)
    },
  })

  function fmtSharpe(v: number | null | undefined): string {
    if (v == null) return '—'
    return v.toFixed(3)
  }

  function fmtDate(s: string | null | undefined): string {
    if (!s) return '—'
    try {
      return new Date(s).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    } catch {
      return s
    }
  }

  function getConfigYaml(exp: Experiment): string {
    if (!exp.config) return '# No config stored'
    if (typeof exp.config === 'string') return exp.config
    return JSON.stringify(exp.config, null, 2)
  }

  const containerStyle = {
    background: '#131722',
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
  }

  return (
    <div style={containerStyle} className="p-4 rounded-lg space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-white uppercase tracking-widest">Experiment Tracker</span>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-white transition-all hover:opacity-90"
          style={{ background: '#2979ff', border: '1px solid #2979ff88' }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Run New Experiment
        </button>
      </div>

      {/* Table */}
      <div className="rounded-lg overflow-hidden" style={{ background: '#1e2433' }}>
        {/* Header row */}
        <div
          className="grid px-4 py-2 border-b border-[#ffffff0d]"
          style={{ gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr 1fr', background: '#131722' }}
        >
          {['Name', 'Model', 'Status', 'Val Sharpe', 'Test Sharpe', 'Started'].map(col => (
            <span key={col} className="text-[10px] text-[#555] uppercase tracking-widest">{col}</span>
          ))}
        </div>

        {isLoading && (
          <div className="flex items-center gap-2 px-4 py-6 text-[#888]">
            <div className="w-4 h-4 border-2 border-[#2979ff] border-t-transparent rounded-full animate-spin" />
            <span className="text-sm">Loading experiments…</span>
          </div>
        )}

        {isError && (
          <div className="px-4 py-6 text-center text-sm text-[#555]">
            Failed to load experiments
          </div>
        )}

        {!isLoading && !isError && (!experiments || experiments.length === 0) && (
          <div className="px-4 py-8 text-center text-sm text-[#555]">
            No experiments yet — run one to get started
          </div>
        )}

        {experiments?.map((exp) => (
          <div key={exp.id}>
            {/* Main row */}
            <div
              className="grid px-4 py-3 border-b border-[#ffffff08] cursor-pointer hover:bg-[#ffffff04] transition-colors items-center"
              style={{ gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr 1fr' }}
              onClick={() => setExpandedId(expandedId === exp.id ? null : exp.id)}
            >
              <div className="flex items-center gap-2">
                <svg
                  width="10"
                  height="10"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#555"
                  strokeWidth="2"
                  className="transition-transform"
                  style={{ transform: expandedId === exp.id ? 'rotate(90deg)' : 'rotate(0deg)' }}
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
                <span className="text-sm text-white truncate">{exp.name}</span>
              </div>
              <span className="text-xs text-[#888]">{exp.model_type ?? '—'}</span>
              <StatusBadge status={exp.status} />
              <span
                className="text-xs font-mono"
                style={{ color: (exp.val_sharpe ?? 0) >= 1.5 ? '#00c853' : (exp.val_sharpe ?? 0) >= 0.5 ? '#f5a623' : '#ff1744' }}
              >
                {fmtSharpe(exp.val_sharpe)}
              </span>
              <span
                className="text-xs font-mono"
                style={{ color: (exp.test_sharpe ?? 0) >= 1.5 ? '#00c853' : (exp.test_sharpe ?? 0) >= 0.5 ? '#f5a623' : '#ff1744' }}
              >
                {fmtSharpe(exp.test_sharpe)}
              </span>
              <span className="text-xs text-[#555]">{fmtDate(exp.started_at)}</span>
            </div>

            {/* Expanded config */}
            {expandedId === exp.id && (
              <div
                className="px-4 py-3 border-b border-[#ffffff08]"
                style={{ background: '#0d1117' }}
              >
                <p className="text-[10px] text-[#555] uppercase tracking-widest mb-2">Config YAML</p>
                <pre
                  className="text-xs text-[#aaa] overflow-x-auto p-3 rounded"
                  style={{ background: '#131722', border: '1px solid #ffffff0d', lineHeight: '1.6' }}
                >
                  {getConfigYaml(exp)}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Modal */}
      {showModal && (
        <NewExperimentModal
          onClose={() => setShowModal(false)}
          onSubmit={submitExperiment}
          isSubmitting={isSubmitting}
        />
      )}
    </div>
  )
}
