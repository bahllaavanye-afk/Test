import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface StageMetrics {
  sharpe?: number
  sortino?: number
  win_rate?: number
  max_drawdown?: number
  num_trades?: number
  days_in_stage?: number
  p_value?: number | null
}

interface StageCriteria {
  min_days: number
  min_sharpe: number
  min_win_rate: number
  max_drawdown: number
  min_trades: number
  require_p_value: boolean
}

type CriteriaMap = Record<string, StageCriteria>

interface ReviewEntry {
  ts: string
  stage: string
  transition?: string
  passed?: boolean
  metrics?: StageMetrics
  failures?: string[]
  event?: string
  reason?: string
}

interface StrategyPromotion {
  id: string
  strategy_id: string
  strategy_name: string
  current_stage: 'paper' | 'shadow' | 'staging' | 'live' | 'rejected'
  paper_metrics: StageMetrics
  shadow_metrics: StageMetrics
  staging_metrics: StageMetrics
  live_metrics: StageMetrics
  paper_started_at?: string
  shadow_started_at?: string
  staging_started_at?: string
  live_started_at?: string
  promotion_ready: boolean
  promotion_ready_stage?: string
  awaiting_approval: boolean
  approved_by?: string
  approved_at?: string
  rejection_reason?: string
  last_review_at?: string
  review_history: ReviewEntry[]
  notes?: string
  created_at?: string
}

// ─── Constants ────────────────────────────────────────────────────────────────

const STAGES = ['paper', 'shadow', 'staging', 'live'] as const

const STAGE_CONFIG = {
  paper:    { label: 'Paper',    color: 'text-blue-400',    bg: 'bg-blue-900/30',    border: 'border-blue-500/40',   dot: 'bg-blue-400'   },
  shadow:   { label: 'Shadow',   color: 'text-yellow-400',  bg: 'bg-yellow-900/30',  border: 'border-yellow-500/40', dot: 'bg-yellow-400' },
  staging:  { label: 'Staging',  color: 'text-orange-400',  bg: 'bg-orange-900/30',  border: 'border-orange-500/40', dot: 'bg-orange-400' },
  live:     { label: 'Live',     color: 'text-green-400',   bg: 'bg-green-900/30',   border: 'border-green-500/40',  dot: 'bg-green-400'  },
  rejected: { label: 'Rejected', color: 'text-red-400',     bg: 'bg-red-900/20',     border: 'border-red-500/40',    dot: 'bg-red-400'    },
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt(iso?: string) {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric', year: '2-digit' })
}

function MetricBadge({ label, value, good, fmt: fmtFn }: {
  label: string
  value?: number | null
  good?: boolean
  fmt?: (v: number) => string
}) {
  if (value === undefined || value === null) return null
  const color = good === true ? 'text-green-400' : good === false ? 'text-red-400' : 'text-gray-300'
  const display = fmtFn ? fmtFn(value) : value.toFixed(2)
  return (
    <span className="inline-flex items-center gap-1 bg-white/5 rounded px-2 py-0.5 text-xs">
      <span className="text-gray-500">{label}</span>
      <span className={`font-mono font-semibold ${color}`}>{display}</span>
    </span>
  )
}

function PValueBadge({ p_value }: { p_value?: number | null }) {
  if (p_value === undefined || p_value === null) return null
  const significant = p_value < 0.05
  return (
    <span className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs border ${
      significant
        ? 'bg-purple-900/30 border-purple-500/40 text-purple-300'
        : 'bg-white/5 border-white/10 text-gray-400'
    }`}>
      <span className="text-gray-500">p=</span>
      <span className="font-mono font-semibold">{p_value.toFixed(4)}</span>
      {significant && <span title="Statistically significant (p < 0.05)">★</span>}
    </span>
  )
}

function StageMetricsRow({ metrics }: { metrics: StageMetrics }) {
  if (!metrics || Object.keys(metrics).length === 0) return <span className="text-gray-600 text-xs">No metrics yet</span>
  return (
    <div className="flex flex-wrap gap-1.5 mt-1">
      <MetricBadge label="Sharpe" value={metrics.sharpe} good={metrics.sharpe !== undefined ? metrics.sharpe >= 1.0 : undefined} />
      <MetricBadge label="WinRate" value={metrics.win_rate !== undefined ? metrics.win_rate * 100 : undefined} fmt={v => `${v.toFixed(1)}%`} />
      <MetricBadge label="MaxDD" value={metrics.max_drawdown !== undefined ? metrics.max_drawdown * 100 : undefined} fmt={v => `${v.toFixed(1)}%`} good={metrics.max_drawdown !== undefined ? metrics.max_drawdown > -0.15 : undefined} />
      {metrics.num_trades !== undefined && (
        <span className="inline-flex items-center gap-1 bg-white/5 rounded px-2 py-0.5 text-xs">
          <span className="text-gray-500">Trades</span>
          <span className="font-mono font-semibold text-gray-300">{metrics.num_trades}</span>
        </span>
      )}
      {metrics.days_in_stage !== undefined && (
        <span className="inline-flex items-center gap-1 bg-white/5 rounded px-2 py-0.5 text-xs">
          <span className="text-gray-500">Days</span>
          <span className="font-mono font-semibold text-gray-300">{metrics.days_in_stage}</span>
        </span>
      )}
      <PValueBadge p_value={metrics.p_value} />
    </div>
  )
}

// ─── Criteria Progress Bars ───────────────────────────────────────────────────

function CriteriaProgress({ metrics, criteria }: { metrics: StageMetrics; criteria: StageCriteria }) {
  const checks: Array<{ label: string; pct: number; pass: boolean; text: string }> = []

  if (metrics.sharpe !== undefined) {
    const pct = Math.min(100, (metrics.sharpe / criteria.min_sharpe) * 100)
    checks.push({ label: 'Sharpe', pct, pass: metrics.sharpe >= criteria.min_sharpe, text: `${metrics.sharpe.toFixed(2)} / ${criteria.min_sharpe}` })
  }
  if (metrics.win_rate !== undefined) {
    const pct = Math.min(100, (metrics.win_rate / criteria.min_win_rate) * 100)
    checks.push({ label: 'Win Rate', pct, pass: metrics.win_rate >= criteria.min_win_rate, text: `${(metrics.win_rate * 100).toFixed(1)}% / ${(criteria.min_win_rate * 100).toFixed(0)}%` })
  }
  if (metrics.max_drawdown !== undefined) {
    // max_drawdown is negative; closer to 0 is better; criteria.max_drawdown is e.g. -0.10
    const pct = Math.min(100, (criteria.max_drawdown / Math.min(metrics.max_drawdown, -1e-9)) * 100)
    checks.push({ label: 'Max DD', pct, pass: metrics.max_drawdown >= criteria.max_drawdown, text: `${(metrics.max_drawdown * 100).toFixed(1)}% / ${(criteria.max_drawdown * 100).toFixed(0)}%` })
  }
  if (metrics.days_in_stage !== undefined) {
    const pct = Math.min(100, (metrics.days_in_stage / criteria.min_days) * 100)
    checks.push({ label: 'Days', pct, pass: metrics.days_in_stage >= criteria.min_days, text: `${metrics.days_in_stage} / ${criteria.min_days}` })
  }
  if (metrics.num_trades !== undefined) {
    const pct = Math.min(100, (metrics.num_trades / criteria.min_trades) * 100)
    checks.push({ label: 'Trades', pct, pass: metrics.num_trades >= criteria.min_trades, text: `${metrics.num_trades} / ${criteria.min_trades}` })
  }
  if (criteria.require_p_value) {
    const pv = metrics.p_value
    const pass = pv !== null && pv !== undefined && pv < 0.05
    const pct = pv !== null && pv !== undefined ? Math.min(100, ((0.05 - pv) / 0.05) * 100) : 0
    checks.push({ label: 'ML Sig.', pct: Math.max(0, pct), pass, text: pv != null ? `p=${pv.toFixed(4)}` : 'not computed' })
  }

  if (checks.length === 0) return null

  return (
    <div className="mt-3 space-y-1.5">
      <div className="text-xs font-semibold text-gray-400 mb-2">Promotion Criteria Progress</div>
      {checks.map(({ label, pct, pass, text }) => (
        <div key={label} className="flex items-center gap-2">
          <span className="text-xs text-gray-500 w-16 shrink-0">{label}</span>
          <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${pass ? 'bg-green-500' : 'bg-orange-500'}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className={`text-xs font-mono w-24 text-right shrink-0 ${pass ? 'text-green-400' : 'text-orange-400'}`}>{text}</span>
          <span>{pass ? '✓' : '·'}</span>
        </div>
      ))}
    </div>
  )
}

// ─── Stage Progress Bar ───────────────────────────────────────────────────────

function StageProgress({ current }: { current: string }) {
  const idx = STAGES.indexOf(current as typeof STAGES[number])
  return (
    <div className="flex items-center gap-1 mt-2">
      {STAGES.map((s, i) => {
        const cfg = STAGE_CONFIG[s]
        const active = s === current
        const done = idx > i
        return (
          <div key={s} className="flex items-center gap-1">
            <div className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-all
              ${active ? `${cfg.bg} ${cfg.color} ring-1 ${cfg.border}` : done ? 'bg-white/5 text-gray-400' : 'bg-transparent text-gray-600'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${active ? cfg.dot : done ? 'bg-gray-400' : 'bg-gray-700'}`} />
              {cfg.label}
            </div>
            {i < STAGES.length - 1 && (
              <div className={`h-px w-3 ${done || active ? 'bg-gray-500' : 'bg-gray-700'}`} />
            )}
          </div>
        )
      })}
    </div>
  )
}

// ─── Promotion Card ───────────────────────────────────────────────────────────

const TRANSITION_KEY: Record<string, string> = {
  paper: 'paper_to_shadow',
  shadow: 'shadow_to_staging',
  staging: 'staging_to_live',
}

function PromotionCard({ promo, onApprove, onReject, onReview, criteriaMap }: {
  promo: StrategyPromotion
  onApprove: (id: string) => void
  onReject: (id: string) => void
  onReview: (id: string) => void
  criteriaMap: CriteriaMap
}) {
  const [expanded, setExpanded] = useState(false)
  const cfg = STAGE_CONFIG[promo.current_stage] ?? STAGE_CONFIG.paper

  const currentMetrics = {
    paper: promo.paper_metrics,
    shadow: promo.shadow_metrics,
    staging: promo.staging_metrics,
    live: promo.live_metrics,
    rejected: {},
  }[promo.current_stage] || {}

  return (
    <div className={`rounded-2xl border p-5 transition-all duration-300 ${cfg.bg} ${cfg.border}`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className={`font-bold text-base ${cfg.color}`}>{promo.strategy_name}</span>
            {promo.awaiting_approval && (
              <span className="text-xs px-2 py-0.5 bg-yellow-500/20 border border-yellow-500/40 text-yellow-300 rounded-full animate-pulse">
                Awaiting Approval
              </span>
            )}
            {promo.current_stage === 'live' && (
              <span className="text-xs px-2 py-0.5 bg-green-500/20 border border-green-500/40 text-green-300 rounded-full">
                🟢 LIVE
              </span>
            )}
            {promo.current_stage === 'rejected' && (
              <span className="text-xs px-2 py-0.5 bg-red-500/20 border border-red-500/40 text-red-300 rounded-full">
                Rejected
              </span>
            )}
          </div>
          <div className="text-xs text-gray-500 mt-0.5">Started paper: {fmt(promo.paper_started_at)}</div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2 shrink-0">
          {promo.awaiting_approval && promo.current_stage !== 'live' && promo.current_stage !== 'rejected' && (
            <>
              <button
                onClick={() => onApprove(promo.id)}
                className="px-3 py-1.5 text-xs font-semibold bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors"
              >
                ✓ Promote
              </button>
              <button
                onClick={() => onReject(promo.id)}
                className="px-3 py-1.5 text-xs font-semibold bg-red-900/60 hover:bg-red-800 text-red-300 border border-red-500/40 rounded-lg transition-colors"
              >
                ✕ Reject
              </button>
            </>
          )}
          {promo.current_stage !== 'live' && promo.current_stage !== 'rejected' && !promo.awaiting_approval && (
            <button
              onClick={() => onReview(promo.id)}
              className="px-3 py-1.5 text-xs font-semibold bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10 rounded-lg transition-colors"
            >
              Review Now
            </button>
          )}
          <button
            onClick={() => setExpanded(!expanded)}
            className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            {expanded ? '▲' : '▼'}
          </button>
        </div>
      </div>

      {/* Stage progress */}
      {promo.current_stage !== 'rejected' && <StageProgress current={promo.current_stage} />}

      {/* Current metrics */}
      <div className="mt-3">
        <div className="text-xs text-gray-500 mb-1">Current stage metrics:</div>
        <StageMetricsRow metrics={currentMetrics} />
        {/* Criteria progress bars — only for promotable stages */}
        {TRANSITION_KEY[promo.current_stage] && criteriaMap[TRANSITION_KEY[promo.current_stage]] && (
          <CriteriaProgress
            metrics={currentMetrics}
            criteria={criteriaMap[TRANSITION_KEY[promo.current_stage]]}
          />
        )}
      </div>

      {/* Rejection reason */}
      {promo.rejection_reason && (
        <div className="mt-3 text-xs bg-red-900/20 border border-red-500/30 rounded-lg p-2 text-red-300">
          <span className="font-semibold">Rejected:</span> {promo.rejection_reason}
        </div>
      )}

      {/* Last review */}
      {promo.last_review_at && (
        <div className="mt-2 text-xs text-gray-600">Last reviewed: {fmt(promo.last_review_at)}</div>
      )}

      {/* Expanded: all stage metrics + review history */}
      {expanded && (
        <div className="mt-4 space-y-3 border-t border-white/5 pt-4">
          {/* All stage metrics */}
          <div>
            <div className="text-xs font-semibold text-gray-400 mb-2">All Stage Metrics</div>
            <div className="grid grid-cols-2 gap-2">
              {(['paper', 'shadow', 'staging', 'live'] as const).map(s => {
                const m = { paper: promo.paper_metrics, shadow: promo.shadow_metrics, staging: promo.staging_metrics, live: promo.live_metrics }[s]
                const started = { paper: promo.paper_started_at, shadow: promo.shadow_started_at, staging: promo.staging_started_at, live: promo.live_started_at }[s]
                const scfg = STAGE_CONFIG[s]
                return (
                  <div key={s} className={`rounded-lg p-2 border ${scfg.bg} ${scfg.border}`}>
                    <div className={`text-xs font-semibold ${scfg.color} mb-1`}>{scfg.label}{started ? ` · ${fmt(started)}` : ''}</div>
                    <StageMetricsRow metrics={m || {}} />
                  </div>
                )
              })}
            </div>
          </div>

          {/* Review history */}
          {promo.review_history && promo.review_history.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-gray-400 mb-2">Review History ({promo.review_history.length})</div>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {[...promo.review_history].reverse().map((entry, i) => (
                  <div key={i} className={`text-xs rounded-lg p-2 border ${entry.passed ? 'bg-green-900/10 border-green-500/20' : 'bg-gray-900/30 border-white/5'}`}>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500">{fmt(entry.ts)}</span>
                      <span className={`font-semibold ${entry.passed ? 'text-green-400' : 'text-gray-400'}`}>
                        {entry.event || (entry.passed ? '✓ PASS' : '✗ FAIL')}
                      </span>
                      {entry.transition && <span className="text-gray-600">{entry.transition}</span>}
                    </div>
                    {entry.failures && entry.failures.length > 0 && (
                      <div className="mt-1 text-red-400 text-xs">{entry.failures.join(' · ')}</div>
                    )}
                    {entry.reason && <div className="mt-1 text-gray-400">{entry.reason}</div>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Promotions() {
  const qc = useQueryClient()
  const [rejectTarget, setRejectTarget] = useState<string | null>(null)
  const [rejectReason, setRejectReason] = useState('')
  const [filterStage, setFilterStage] = useState<string>('all')

  const { data: promotions = [], isLoading, error } = useQuery<StrategyPromotion[]>({
    queryKey: ['promotions'],
    queryFn: () => api.get('/api/v1/promotions/').then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: criteriaMap = {} } = useQuery<CriteriaMap>({
    queryKey: ['promotions-criteria'],
    queryFn: () => api.get('/api/v1/promotions/criteria/all').then(r => r.data),
    staleTime: 5 * 60_000,
  })

  const approveMut = useMutation({
    mutationFn: (id: string) => api.post(`/api/v1/promotions/${id}/approve`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['promotions'] }),
  })

  const rejectMut = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      api.post(`/api/v1/promotions/${id}/reject`, { reason }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['promotions'] })
      setRejectTarget(null)
      setRejectReason('')
    },
  })

  const reviewMut = useMutation({
    mutationFn: (id: string) => api.post(`/api/v1/promotions/${id}/review`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['promotions'] }),
  })

  const filtered = filterStage === 'all' ? promotions
    : filterStage === 'pending' ? promotions.filter(p => p.awaiting_approval)
    : promotions.filter(p => p.current_stage === filterStage)

  const pendingCount = promotions.filter(p => p.awaiting_approval).length

  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Strategy Promotion Pipeline</h1>
          <p className="text-sm text-gray-500 mt-1">
            Paper → Shadow → Staging → Live · Holistic review runs daily at 06:00 UTC
          </p>
        </div>
        {pendingCount > 0 && (
          <div className="px-4 py-2 bg-yellow-500/20 border border-yellow-500/40 rounded-xl text-yellow-300 text-sm font-semibold">
            {pendingCount} awaiting approval
          </div>
        )}
      </div>

      {/* Stage criteria reference — live from API */}
      <div className="grid grid-cols-3 gap-3 text-xs">
        {([
          { key: 'paper_to_shadow', label: 'Paper → Shadow' },
          { key: 'shadow_to_staging', label: 'Shadow → Staging' },
          { key: 'staging_to_live', label: 'Staging → Live' },
        ] as const).map(({ key, label }) => {
          const c: StageCriteria | undefined = criteriaMap[key]
          return (
          <div key={key} className="bg-white/3 border border-white/5 rounded-xl p-3">
            <div className="font-semibold text-gray-300 mb-1">{label}</div>
            {c ? (
              <div className="text-gray-500 space-y-0.5">
                <div>Sharpe ≥ {c.min_sharpe} · Win ≥ {(c.min_win_rate * 100).toFixed(0)}%</div>
                <div>DD ≥ {(c.max_drawdown * 100).toFixed(0)}% · {c.min_days}d · {c.min_trades} trades</div>
                {c.require_p_value && <div className="text-purple-400">ML significance required (p &lt; 0.05)</div>}
              </div>
            ) : (
              <div className="text-gray-600">Loading…</div>
            )}
          </div>
          )
        })}
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2 flex-wrap">
        {['all', 'pending', 'paper', 'shadow', 'staging', 'live', 'rejected'].map(f => (
          <button
            key={f}
            onClick={() => setFilterStage(f)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors capitalize
              ${filterStage === f ? 'bg-[#00ff88]/20 text-[#00ff88] border border-[#00ff88]/30' : 'bg-white/5 text-gray-400 border border-white/10 hover:text-gray-200'}`}
          >
            {f === 'pending' ? `⚡ Pending (${pendingCount})` : f === 'all' ? `All (${promotions.length})` : f}
          </button>
        ))}
      </div>

      {/* Loading / error states */}
      {isLoading && (
        <div className="flex items-center justify-center h-32 text-gray-500">
          Loading promotions…
        </div>
      )}
      {error && (
        <div className="bg-red-900/20 border border-red-500/30 rounded-xl p-4 text-red-300 text-sm">
          Failed to load promotions. Is the backend running?
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !error && filtered.length === 0 && (
        <div className="text-center text-gray-600 py-16">
          <div className="text-4xl mb-3">📊</div>
          <div className="font-semibold text-gray-500">No promotions in this filter</div>
          <div className="text-xs mt-1">Register strategies via POST /api/v1/promotions/</div>
        </div>
      )}

      {/* Promotion cards */}
      <div className="space-y-4">
        {filtered.map(p => (
          <PromotionCard
            key={p.id}
            promo={p}
            onApprove={(id) => approveMut.mutate(id)}
            onReject={(id) => { setRejectTarget(id); setRejectReason('') }}
            onReview={(id) => reviewMut.mutate(id)}
            criteriaMap={criteriaMap}
          />
        ))}
      </div>

      {/* Reject modal */}
      {rejectTarget && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-[#1a1d27] border border-white/10 rounded-2xl p-6 w-full max-w-md">
            <h2 className="text-white font-bold text-lg mb-3">Reject Strategy Promotion</h2>
            <p className="text-gray-400 text-sm mb-4">
              Provide a reason for rejection. The strategy will be moved to "rejected" state.
            </p>
            <textarea
              value={rejectReason}
              onChange={e => setRejectReason(e.target.value)}
              placeholder="Reason for rejection…"
              className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-gray-200 placeholder-gray-600 resize-none h-24 focus:outline-none focus:border-white/20"
            />
            <div className="flex gap-3 mt-4">
              <button
                onClick={() => rejectMut.mutate({ id: rejectTarget, reason: rejectReason })}
                disabled={!rejectReason.trim() || rejectMut.isPending}
                className="flex-1 py-2 bg-red-700 hover:bg-red-600 disabled:opacity-50 text-white rounded-lg font-semibold text-sm transition-colors"
              >
                {rejectMut.isPending ? 'Rejecting…' : 'Confirm Reject'}
              </button>
              <button
                onClick={() => setRejectTarget(null)}
                className="flex-1 py-2 bg-white/5 hover:bg-white/10 text-gray-300 rounded-lg font-semibold text-sm transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
