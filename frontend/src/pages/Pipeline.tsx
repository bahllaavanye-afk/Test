import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface StageRecord {
  name:          string
  label:         string
  status:        'pending' | 'running' | 'success' | 'failed' | 'skipped'
  started_at?:   string
  completed_at?: string
  duration_s?:   number
  output:        Record<string, unknown>
  error?:        string
  channel?:      string
}

interface PipelineRun {
  run_id:         string
  pipeline:       string
  pipeline_label: string
  desk?:          string
  branch:         string
  triggered_by:   string
  started_at:     string
  completed_at?:  string
  status:         'pending' | 'running' | 'success' | 'failed'
  stages:         StageRecord[]
  run_url:        string
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const STATUS_CONFIG = {
  pending: { color: 'text-gray-400',  bg: 'bg-gray-800/60',   border: 'border-gray-700',    dot: 'bg-gray-500',            label: 'Pending' },
  running: { color: 'text-blue-300',  bg: 'bg-blue-900/30',   border: 'border-blue-500/50', dot: 'bg-blue-400 animate-pulse', label: 'Running' },
  success: { color: 'text-green-300', bg: 'bg-green-900/20',  border: 'border-green-500/40',dot: 'bg-green-400',           label: 'Success' },
  failed:  { color: 'text-red-300',   bg: 'bg-red-900/20',    border: 'border-red-500/40',  dot: 'bg-red-400',             label: 'Failed'  },
  skipped: { color: 'text-gray-500',  bg: 'bg-gray-900/30',   border: 'border-gray-700',    dot: 'bg-gray-600',            label: 'Skipped' },
}

const PIPELINE_ICONS: Record<string, string> = {
  ml_experiments: '🧪',
  desk_trading:   '📈',
  agent_team:     '🤖',
}

function fmt(iso?: string): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function elapsed(startIso?: string, endIso?: string): string {
  if (!startIso) return '—'
  const end  = endIso ? new Date(endIso) : new Date()
  const secs = Math.round((end.getTime() - new Date(startIso).getTime()) / 1000)
  if (secs < 60) return `${secs}s`
  return `${Math.floor(secs / 60)}m ${secs % 60}s`
}

function outputSummary(output: Record<string, unknown>): string {
  return Object.entries(output)
    .slice(0, 4)
    .map(([k, v]) => {
      if (typeof v === 'number') return `${k}=${typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(3) : v}`
      return `${k}=${v}`
    })
    .join('  ')
}

// ─── Stage pill ───────────────────────────────────────────────────────────────

function StagePill({ stage, active }: { stage: StageRecord; active: boolean }) {
  const cfg = STATUS_CONFIG[stage.status] ?? STATUS_CONFIG.pending
  return (
    <div
      className={`relative flex-1 min-w-0 rounded-xl border p-3 transition-all duration-200 cursor-default
        ${cfg.bg} ${cfg.border} ${active ? 'ring-1 ring-white/10' : ''}`}
    >
      {/* Status dot */}
      <div className="flex items-center gap-2 mb-1">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${cfg.dot}`} />
        <span className={`text-xs font-semibold uppercase tracking-wide truncate ${cfg.color}`}>
          {cfg.label}
        </span>
        {stage.duration_s != null && (
          <span className="ml-auto text-xs text-gray-500">{stage.duration_s}s</span>
        )}
      </div>

      {/* Stage name */}
      <p className="text-sm font-medium text-white/90 truncate">{stage.label}</p>

      {/* Output summary */}
      {stage.output && Object.keys(stage.output).length > 0 && (
        <p className="mt-1 text-xs text-gray-400 truncate font-mono">
          {outputSummary(stage.output)}
        </p>
      )}

      {/* Error */}
      {stage.error && (
        <p className="mt-1 text-xs text-red-400 truncate" title={stage.error}>
          {stage.error}
        </p>
      )}

      {/* Channel tag */}
      {stage.channel && (
        <span className="mt-1 inline-block text-xs text-blue-400/70 font-mono">
          {stage.channel}
        </span>
      )}
    </div>
  )
}

// ─── Connector arrow ──────────────────────────────────────────────────────────

function Connector({ done }: { done: boolean }) {
  return (
    <div className="flex items-center justify-center w-5 flex-shrink-0 pt-4">
      <div className={`h-px w-full transition-colors duration-500 ${done ? 'bg-green-500/60' : 'bg-white/10'}`} />
      <svg className={`w-2 h-2 flex-shrink-0 ${done ? 'text-green-500/60' : 'text-white/10'}`} fill="currentColor" viewBox="0 0 8 8">
        <polygon points="0,0 8,4 0,8" />
      </svg>
    </div>
  )
}

// ─── Single run card ──────────────────────────────────────────────────────────

function RunCard({ run, isLatest }: { run: PipelineRun; isLatest: boolean }) {
  const [expanded, setExpanded] = useState(isLatest)
  const cfg = STATUS_CONFIG[run.status] ?? STATUS_CONFIG.pending
  const icon = PIPELINE_ICONS[run.pipeline] ?? '⚡'
  const doneStages = new Set(
    run.stages.filter(s => s.status === 'success').map(s => s.name)
  )

  return (
    <div className={`rounded-2xl border ${cfg.border} bg-white/[0.02] overflow-hidden transition-all`}>
      {/* Header */}
      <button
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/[0.03] transition-colors"
        onClick={() => setExpanded(e => !e)}
      >
        <span className="text-lg">{icon}</span>
        <div className="flex-1 text-left min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-white/90">{run.pipeline_label}</span>
            {run.desk && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-white/10 text-gray-300">
                {run.desk}
              </span>
            )}
            {isLatest && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-blue-900/50 text-blue-300 border border-blue-500/30">
                latest
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 mt-0.5 text-xs text-gray-400">
            <span>{fmt(run.started_at)}</span>
            <span>·</span>
            <span>{elapsed(run.started_at, run.completed_at)}</span>
            <span>·</span>
            <span className="font-mono">{run.branch}</span>
            <span>·</span>
            <span>{run.triggered_by}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${cfg.dot}`} />
            <span className={`text-xs font-medium ${cfg.color}`}>{cfg.label}</span>
          </div>
          {run.run_url && run.run_url.startsWith('https://') && (
            <a
              href={run.run_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-gray-500 hover:text-blue-400 transition-colors"
              onClick={e => e.stopPropagation()}
            >
              ↗
            </a>
          )}
          <svg
            className={`w-4 h-4 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Stage pipeline */}
      {expanded && (
        <div className="px-4 pb-4 pt-1">
          <div className="flex items-start gap-0">
            {run.stages.map((stage, idx) => (
              <div key={stage.name} className="flex items-start flex-1 min-w-0">
                <StagePill
                  stage={stage}
                  active={stage.status === 'running'}
                />
                {idx < run.stages.length - 1 && (
                  <Connector done={doneStages.has(stage.name)} />
                )}
              </div>
            ))}
          </div>

          {/* Progress bar */}
          <div className="mt-3 h-1 rounded-full bg-white/5 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-blue-500 to-green-500 transition-all duration-700"
              style={{
                width: `${run.stages.length
                  ? (run.stages.filter(s => s.status === 'success' || s.status === 'skipped').length
                     / run.stages.length) * 100
                  : 0}%`,
              }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>
              {run.stages.filter(s => s.status === 'success').length} / {run.stages.length} stages complete
            </span>
            <span>
              {run.stages.filter(s => s.status === 'failed').length > 0
                ? `${run.stages.filter(s => s.status === 'failed').length} failed`
                : 'no failures'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Pipeline filter tabs ─────────────────────────────────────────────────────

const FILTERS = [
  { key: '',               label: 'All' },
  { key: 'ml_experiments', label: '🧪 ML Experiments' },
  { key: 'desk_trading',   label: '📈 Desk Trading' },
  { key: 'agent_team',     label: '🤖 Agent Team' },
]

// ─── Main page ────────────────────────────────────────────────────────────────

export default function Pipeline() {
  const [filter, setFilter] = useState('')

  const { data: runs = [], isLoading, error, dataUpdatedAt } = useQuery<PipelineRun[]>({
    queryKey: ['pipeline-status', filter],
    queryFn: () => api
      .get('/pipeline/status', { params: filter ? { pipeline: filter, limit: 30 } : { limit: 30 } })
      .then(r => r.data),
    refetchInterval: 15_000,
    staleTime: 10_000,
  })

  // Identify the latest run per pipeline
  const latestRunIds = new Set<string>()
  const seenPipelines = new Set<string>()
  for (const run of runs) {
    const key = `${run.pipeline}:${run.desk ?? ''}`
    if (!seenPipelines.has(key)) {
      seenPipelines.add(key)
      latestRunIds.add(run.run_id)
    }
  }

  const running = runs.filter(r => r.status === 'running').length
  const failed  = runs.filter(r => r.status === 'failed').length
  const success = runs.filter(r => r.status === 'success').length

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Pipeline Status</h1>
          <p className="text-sm text-gray-400 mt-1">
            Stage-level tracking for ML experiments, desk trading, and agent team runs
          </p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          {running > 0 && (
            <span className="flex items-center gap-1.5 text-blue-300">
              <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
              {running} running
            </span>
          )}
          {failed > 0 && (
            <span className="flex items-center gap-1.5 text-red-400">
              <span className="w-2 h-2 rounded-full bg-red-400" />
              {failed} failed
            </span>
          )}
          {success > 0 && (
            <span className="flex items-center gap-1.5 text-green-400">
              <span className="w-2 h-2 rounded-full bg-green-400" />
              {success} succeeded
            </span>
          )}
          {dataUpdatedAt > 0 && (
            <span className="text-xs text-gray-500">
              updated {new Date(dataUpdatedAt).toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2">
        {FILTERS.map(f => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
              ${filter === f.key
                ? 'bg-white/10 text-white'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Pipeline legend */}
      <div className="flex flex-wrap gap-3 text-xs">
        {Object.entries(STATUS_CONFIG).map(([status, cfg]) => (
          <div key={status} className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${cfg.dot}`} />
            <span className={cfg.color}>{cfg.label}</span>
          </div>
        ))}
        <span className="text-gray-600 ml-2">· auto-refreshes every 15s</span>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-20 rounded-2xl bg-white/[0.02] border border-white/5 animate-pulse" />
          ))}
        </div>
      ) : error ? (
        <div className="rounded-2xl border border-red-500/30 bg-red-900/10 p-6 text-center">
          <p className="text-red-400 font-medium">Failed to load pipeline status</p>
          <p className="text-xs text-gray-500 mt-1">
            Backend API unavailable — pipeline_runs.json may not exist yet. Runs will appear after
            the first GitHub Actions workflow completes.
          </p>
        </div>
      ) : runs.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-12 text-center">
          <div className="text-4xl mb-3">🚀</div>
          <p className="text-white/70 font-medium">No pipeline runs yet</p>
          <p className="text-sm text-gray-500 mt-2">
            Runs appear here once GitHub Actions workflows execute with the pipeline tracker.
            Trigger <code className="text-xs bg-white/10 px-1 rounded">run-experiments.yml</code> or{' '}
            <code className="text-xs bg-white/10 px-1 rounded">desk-trading.yml</code> manually to see stages.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {runs.map(run => (
            <RunCard
              key={run.run_id}
              run={run}
              isLatest={latestRunIds.has(run.run_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
