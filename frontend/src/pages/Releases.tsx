import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, ChevronRight, PackageCheck, Zap, Archive, ArrowUpCircle, Eye, GitBranch } from 'lucide-react'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface Release {
  id: string
  model_name: string
  version: string
  artifact_path: string
  framework: string
  n_features: number | null
  seq_len: number | null
  model_params: Record<string, unknown>
  training_config: Record<string, unknown>
  train_metrics: Record<string, number>
  live_metrics: Record<string, number>
  status: 'registered' | 'shadow' | 'challenger' | 'champion' | 'archived'
  traffic_pct: number
  notes: string | null
  promoted_at: string | null
  archived_at: string | null
  created_by: string
  created_at: string
  updated_at: string
}

interface ABStats {
  n_predictions: number
  avg_confidence: number | null
  accuracy: number | null
  avg_latency_ms: number | null
}

interface ABTest {
  champion: Release
  challenger: Release
  champion_stats: ABStats
  challenger_stats: ABStats
  recommendation: 'promote_challenger' | 'keep_champion' | 'insufficient_data'
  min_samples_needed: number
  samples_collected: number
}

interface InferenceLog {
  id: string
  release_id: string
  model_name: string
  version: string
  symbol: string
  ts: string
  prediction: number
  signal: string
  confidence: number
  latency_ms: number
  ab_group: string
  actual_return: number | null
  is_correct: boolean | null
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const STATUS_CONFIG = {
  champion:   { label: 'CHAMPION',   color: '#00c853', bg: '#00c85318', border: '#00c85344' },
  challenger: { label: 'CHALLENGER', color: '#2196F3', bg: '#2196F318', border: '#2196F344' },
  shadow:     { label: 'SHADOW',     color: '#9c27b0', bg: '#9c27b018', border: '#9c27b044' },
  registered: { label: 'REGISTERED', color: '#888888', bg: '#88888818', border: '#88888844' },
  archived:   { label: 'ARCHIVED',   color: '#444444', bg: '#44444418', border: '#44444444' },
}

const FRAMEWORK_COLOR: Record<string, string> = {
  pytorch:   '#ee4c2c',
  xgboost:   '#189ab4',
  lightgbm:  '#30a14e',
  ensemble:  '#f5a623',
}

const num = (v: number | null | undefined, d = 2) => v == null ? '—' : v.toFixed(d)
const pct = (v: number | null | undefined) => v == null ? '—' : `${(v * 100).toFixed(1)}%`
const ms  = (v: number | null | undefined) => v == null ? '—' : `${v.toFixed(1)}ms`

function StatusBadge({ status }: { status: Release['status'] }) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.registered
  return (
    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-widest"
      style={{ color: cfg.color, background: cfg.bg, border: `1px solid ${cfg.border}` }}>
      {cfg.label}
    </span>
  )
}

function FrameworkBadge({ framework }: { framework: string }) {
  const color = FRAMEWORK_COLOR[framework.toLowerCase()] ?? '#888'
  return (
    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded uppercase"
      style={{ color, background: `${color}15`, border: `1px solid ${color}33` }}>
      {framework}
    </span>
  )
}

function MetricPair({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center">
      <span className="text-[10px] text-[#555] uppercase tracking-wider">{label}</span>
      <span className="text-sm font-bold font-mono text-[#e8e8e8]">{value}</span>
    </div>
  )
}

function RecommendationBadge({ rec }: { rec: ABTest['recommendation'] }) {
  if (rec === 'promote_challenger')
    return <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ background: '#00c85320', color: '#00c853', border: '1px solid #00c85344' }}>Promote Challenger ↑</span>
  if (rec === 'keep_champion')
    return <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ background: '#f5a62320', color: '#f5a623', border: '1px solid #f5a62344' }}>Keep Champion</span>
  return <span className="text-xs px-2 py-0.5 rounded text-[#555]" style={{ background: '#1e1e1e', border: '1px solid #333' }}>Collecting data…</span>
}

// ─── Mutation button ──────────────────────────────────────────────────────────

function ActionButton({
  label, icon: Icon, onClick, disabled, color = '#888', size = 'sm',
}: {
  label: string
  icon?: React.ElementType
  onClick: () => void
  disabled?: boolean
  color?: string
  size?: 'xs' | 'sm'
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center gap-1 ${size === 'xs' ? 'text-[10px] px-1.5 py-0.5' : 'text-xs px-2 py-1'} rounded font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed`}
      style={{ color, background: `${color}15`, border: `1px solid ${color}33` }}
    >
      {Icon && <Icon size={size === 'xs' ? 10 : 12} />}
      {label}
    </button>
  )
}

// ─── A/B Test Card ────────────────────────────────────────────────────────────

function ABTestCard({ test, onPromote, onStop }: {
  test: ABTest
  onPromote: (id: string) => void
  onStop: (id: string) => void
}) {
  const progress = Math.min(100, (test.samples_collected / test.min_samples_needed) * 100)

  return (
    <div className="bg-[#111111] border border-[#2196F333] rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <GitBranch size={14} className="text-[#2196F3]" />
          <span className="text-sm font-bold text-[#e8e8e8]">{test.champion.model_name}</span>
          <span className="text-[10px] text-[#555]">A/B Test</span>
        </div>
        <RecommendationBadge rec={test.recommendation} />
      </div>

      {/* Champion vs Challenger */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        {/* Champion */}
        <div className="bg-[#0a0a0a] rounded-lg p-3 border border-[#00c85322]">
          <div className="flex items-center gap-1.5 mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00c853]" />
            <span className="text-[10px] font-bold text-[#00c853] uppercase tracking-wider">Champion</span>
            <span className="text-[10px] text-[#555] font-mono ml-auto">{test.champion.version}</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <MetricPair label="Predictions" value={String(test.champion_stats.n_predictions)} />
            <MetricPair label="Accuracy" value={pct(test.champion_stats.accuracy)} />
            <MetricPair label="Confidence" value={pct(test.champion_stats.avg_confidence)} />
          </div>
          <div className="mt-2 text-[10px] text-[#555]">
            {(100 - test.challenger.traffic_pct).toFixed(0)}% traffic · {ms(test.champion_stats.avg_latency_ms)} avg
          </div>
        </div>
        {/* Challenger */}
        <div className="bg-[#0a0a0a] rounded-lg p-3 border border-[#2196F322]">
          <div className="flex items-center gap-1.5 mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#2196F3]" />
            <span className="text-[10px] font-bold text-[#2196F3] uppercase tracking-wider">Challenger</span>
            <span className="text-[10px] text-[#555] font-mono ml-auto">{test.challenger.version}</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <MetricPair label="Predictions" value={String(test.challenger_stats.n_predictions)} />
            <MetricPair label="Accuracy" value={pct(test.challenger_stats.accuracy)} />
            <MetricPair label="Confidence" value={pct(test.challenger_stats.avg_confidence)} />
          </div>
          <div className="mt-2 text-[10px] text-[#555]">
            {test.challenger.traffic_pct.toFixed(0)}% traffic · {ms(test.challenger_stats.avg_latency_ms)} avg
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-3">
        <div className="flex justify-between text-[10px] text-[#555] mb-1">
          <span>Sample progress</span>
          <span>{test.samples_collected} / {test.min_samples_needed} needed</span>
        </div>
        <div className="h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-500"
            style={{ width: `${progress}%`, background: progress >= 100 ? '#00c853' : '#2196F3' }} />
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <ActionButton
          label="Promote Challenger"
          icon={ArrowUpCircle}
          onClick={() => onPromote(test.challenger.id)}
          color="#00c853"
        />
        <ActionButton
          label="Stop Test"
          icon={Archive}
          onClick={() => onStop(test.challenger.id)}
          color="#ff1744"
        />
      </div>
    </div>
  )
}

// ─── Challenge Dialog ─────────────────────────────────────────────────────────

function ChallengeDialog({
  releaseId,
  onClose,
  onSubmit,
}: {
  releaseId: string
  onClose: () => void
  onSubmit: (id: string, pct: number) => void
}) {
  const [pct, setPct] = useState(10)
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-[#111111] border border-[#2196F3] rounded-xl p-6 w-80" onClick={e => e.stopPropagation()}>
        <h3 className="text-sm font-bold text-[#e8e8e8] mb-1">Start A/B Test</h3>
        <p className="text-xs text-[#555] mb-4">Set challenger traffic split (1–50%)</p>
        <div className="flex items-center gap-3 mb-4">
          <input type="range" min={1} max={50} value={pct} onChange={e => setPct(Number(e.target.value))}
            className="flex-1 accent-[#2196F3]" />
          <span className="text-sm font-bold font-mono text-[#2196F3] w-12 text-right">{pct}%</span>
        </div>
        <p className="text-[10px] text-[#555] mb-4">
          Champion: {100 - pct}% · Challenger: {pct}%
        </p>
        <div className="flex gap-2">
          <button onClick={onClose} className="flex-1 text-xs py-2 rounded border border-[#333] text-[#888] hover:text-[#e8e8e8] transition-colors">Cancel</button>
          <button onClick={() => onSubmit(releaseId, pct)}
            className="flex-1 text-xs py-2 rounded font-bold transition-colors"
            style={{ background: '#2196F3', color: '#fff' }}>
            Start Test
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Inference Log Panel ──────────────────────────────────────────────────────

function InferencePanel({ releaseId, onClose }: { releaseId: string; onClose: () => void }) {
  const { data: logs = [], isLoading } = useQuery<InferenceLog[]>({
    queryKey: ['inferences', releaseId],
    queryFn: () => api.get(`/releases/${releaseId}/inferences`).then(r => r.data),
    refetchInterval: 10_000,
    enabled: !!releaseId,
  })

  return (
    <div className="mt-4 bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#1e1e1e]">
        <span className="text-xs font-semibold text-[#e8e8e8]">Inference Logs</span>
        <button onClick={onClose} className="text-[#555] hover:text-[#888] text-xs">✕ close</button>
      </div>
      {isLoading ? (
        <div className="p-4 text-xs text-[#555] animate-pulse">Loading…</div>
      ) : logs.length === 0 ? (
        <div className="p-4 text-xs text-[#444] text-center">No inferences logged yet</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-[#111111]">
              <tr className="text-[10px] text-[#555]">
                {['Time', 'Symbol', 'Signal', 'Confidence', 'A/B Group', 'Latency', 'Outcome'].map(h => (
                  <th key={h} className="px-3 py-2 text-left uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logs.slice(0, 50).map(log => (
                <tr key={log.id} className="border-t border-[#111111] hover:bg-[#111111] transition-colors">
                  <td className="px-3 py-2 text-[10px] text-[#555] whitespace-nowrap font-mono">
                    {new Date(log.ts).toLocaleTimeString()}
                  </td>
                  <td className="px-3 py-2 text-xs font-mono text-[#e8e8e8]">{log.symbol}</td>
                  <td className="px-3 py-2 text-[10px] font-bold"
                    style={{ color: log.signal === 'buy' ? '#00c853' : log.signal === 'sell' ? '#ff1744' : '#888' }}>
                    {log.signal.toUpperCase()}
                  </td>
                  <td className="px-3 py-2 text-xs font-mono text-[#888]">{pct(log.confidence)}</td>
                  <td className="px-3 py-2">
                    <span className="text-[10px] font-bold"
                      style={{ color: log.ab_group === 'champion' ? '#00c853' : '#2196F3' }}>
                      {log.ab_group.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-[10px] font-mono text-[#555]">{ms(log.latency_ms)}</td>
                  <td className="px-3 py-2 text-xs">
                    {log.is_correct === null ? <span className="text-[#444]">—</span>
                      : log.is_correct
                        ? <span className="text-[#00c853]">✓</span>
                        : <span className="text-[#ff1744]">✗</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Releases() {
  const qc = useQueryClient()
  const [challengeTarget, setChallengeTarget] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<string>('all')

  const { data: releases = [], isLoading, isError } = useQuery<Release[]>({
    queryKey: ['releases'],
    queryFn: () => api.get('/releases/').then(r => r.data),
    refetchInterval: 15_000,
  })

  const { data: abTests = [] } = useQuery<ABTest[]>({
    queryKey: ['ab-tests'],
    queryFn: () => api.get('/releases/ab-tests/active').then(r => r.data),
    refetchInterval: 10_000,
  })

  function mutOpts(invalidateKeys: string[]) {
    return {
      onSuccess: () => {
        invalidateKeys.forEach(k => qc.invalidateQueries({ queryKey: [k] }))
      },
    }
  }

  const promote = useMutation({
    mutationFn: (id: string) => api.post(`/releases/${id}/promote`).then(r => r.data),
    ...mutOpts(['releases', 'ab-tests']),
  })
  const archive = useMutation({
    mutationFn: (id: string) => api.post(`/releases/${id}/archive`).then(r => r.data),
    ...mutOpts(['releases', 'ab-tests']),
  })
  const shadow = useMutation({
    mutationFn: (id: string) => api.post(`/releases/${id}/shadow`).then(r => r.data),
    ...mutOpts(['releases', 'ab-tests']),
  })
  const challenge = useMutation({
    mutationFn: ({ id, pct }: { id: string; pct: number }) =>
      api.post(`/releases/${id}/challenge`, { traffic_pct: pct }).then(r => r.data),
    ...mutOpts(['releases', 'ab-tests']),
  })

  const statuses = ['all', 'champion', 'challenger', 'shadow', 'registered', 'archived']
  const filtered = statusFilter === 'all'
    ? releases
    : releases.filter(r => r.status === statusFilter)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-[#e8e8e8] flex items-center gap-2">
            <PackageCheck size={18} className="text-[#f5a623]" />
            Model Releases
          </h1>
          <p className="text-xs text-[#555] mt-0.5">
            Lifecycle: registered → shadow → challenger → champion → archived
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-[#555]">
          <span className="w-2 h-2 rounded-full bg-[#00c853]" /> {releases.filter(r => r.status === 'champion').length} champion
          <span className="w-2 h-2 rounded-full bg-[#2196F3] ml-2" /> {abTests.length} A/B test{abTests.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Active A/B Tests */}
      {abTests.length > 0 && (
        <div>
          <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider mb-3">Active A/B Tests</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {abTests.map((test, i) => (
              <ABTestCard
                key={i}
                test={test}
                onPromote={id => promote.mutate(id)}
                onStop={id => archive.mutate(id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Releases Table */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-semibold text-[#555] uppercase tracking-wider">All Releases</h2>
          <div className="flex items-center gap-1 bg-[#111111] border border-[#1e1e1e] rounded-lg p-1">
            {statuses.map(s => (
              <button key={s} onClick={() => setStatusFilter(s)}
                className="text-[10px] px-2 py-1 rounded capitalize transition-colors"
                style={{
                  background: statusFilter === s ? '#f5a623' : 'transparent',
                  color: statusFilter === s ? '#000' : '#888',
                }}>
                {s}
              </button>
            ))}
          </div>
        </div>

        {isLoading ? (
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg divide-y divide-[#1e1e1e]">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="flex gap-4 px-4 py-3 animate-pulse">
                <div className="w-32 h-4 bg-[#1e1e1e] rounded" />
                <div className="w-16 h-4 bg-[#1e1e1e] rounded" />
                <div className="w-20 h-4 bg-[#1e1e1e] rounded" />
                <div className="w-24 h-4 bg-[#1e1e1e] rounded" />
              </div>
            ))}
          </div>
        ) : isError ? (
          <div className="flex flex-col items-center py-16 bg-[#111111] border border-[#1e1e1e] rounded-lg space-y-2">
            <p className="text-sm text-[#ff1744]">Failed to load releases</p>
            <p className="text-xs text-[#555]">Check that the backend is running at <code>/api/v1/releases/</code></p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center py-16 bg-[#111111] border border-[#1e1e1e] rounded-lg space-y-3">
            <PackageCheck size={32} className="text-[#333]" />
            <p className="text-sm text-[#555]">No releases yet</p>
            <p className="text-xs text-[#444] max-w-xs text-center">
              Register a model via <code className="text-[#888]">POST /api/v1/releases/</code> after training to start serving it.
            </p>
          </div>
        ) : (
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-[#0a0a0a] border-b border-[#1e1e1e]">
                <tr className="text-[10px] text-[#555] uppercase tracking-wider">
                  {['Model / Version', 'Framework', 'Status', 'Traffic', 'Sharpe', 'Accuracy', 'Features', 'Created', 'Actions'].map(h => (
                    <th key={h} className="px-3 py-2.5 text-left font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map(r => {
                  const isSelected = selectedId === r.id
                  const sharpe = r.train_metrics?.val_sharpe ?? r.train_metrics?.sharpe ?? null
                  const acc = r.train_metrics?.val_accuracy ?? r.train_metrics?.accuracy ?? null

                  return (
                    <>
                      <tr key={r.id}
                        className="border-t border-[#1a1a1a] hover:bg-[#131313] transition-colors cursor-pointer"
                        onClick={() => setSelectedId(isSelected ? null : r.id)}>
                        {/* Model / Version */}
                        <td className="px-3 py-2.5">
                          <div className="flex items-center gap-1.5">
                            {isSelected ? <ChevronDown size={12} className="text-[#555]" /> : <ChevronRight size={12} className="text-[#555]" />}
                            <div>
                              <div className="text-xs font-medium text-[#e8e8e8]">{r.model_name}</div>
                              <div className="text-[10px] font-mono text-[#555]">{r.version}</div>
                            </div>
                          </div>
                        </td>
                        {/* Framework */}
                        <td className="px-3 py-2.5">
                          <FrameworkBadge framework={r.framework} />
                        </td>
                        {/* Status */}
                        <td className="px-3 py-2.5">
                          <StatusBadge status={r.status} />
                        </td>
                        {/* Traffic */}
                        <td className="px-3 py-2.5 text-xs font-mono"
                          style={{ color: r.traffic_pct > 0 ? '#f5a623' : '#444' }}>
                          {r.traffic_pct > 0 ? `${r.traffic_pct.toFixed(0)}%` : '—'}
                        </td>
                        {/* Metrics */}
                        <td className="px-3 py-2.5 text-xs font-mono"
                          style={{ color: sharpe == null ? '#555' : sharpe >= 1.5 ? '#00c853' : sharpe >= 1.0 ? '#f5a623' : '#ff1744' }}>
                          {num(sharpe)}
                        </td>
                        <td className="px-3 py-2.5 text-xs font-mono"
                          style={{ color: acc == null ? '#555' : acc >= 0.6 ? '#00c853' : acc >= 0.5 ? '#f5a623' : '#ff1744' }}>
                          {acc == null ? '—' : `${(acc * 100).toFixed(1)}%`}
                        </td>
                        <td className="px-3 py-2.5 text-xs font-mono text-[#555]">
                          {r.n_features ?? '—'}
                        </td>
                        {/* Created */}
                        <td className="px-3 py-2.5 text-[10px] text-[#555] whitespace-nowrap">
                          {new Date(r.created_at).toLocaleDateString()}
                        </td>
                        {/* Actions */}
                        <td className="px-3 py-2.5" onClick={e => e.stopPropagation()}>
                          <div className="flex items-center gap-1">
                            {r.status === 'registered' && (
                              <>
                                <ActionButton label="Shadow" icon={Eye} size="xs" onClick={() => shadow.mutate(r.id)} color="#9c27b0" />
                                <ActionButton label="Promote" icon={ArrowUpCircle} size="xs" onClick={() => promote.mutate(r.id)} color="#00c853" />
                                <ActionButton label="Archive" size="xs" onClick={() => archive.mutate(r.id)} color="#ff1744" />
                              </>
                            )}
                            {r.status === 'shadow' && (
                              <>
                                <ActionButton label="Challenge" icon={Zap} size="xs" onClick={() => setChallengeTarget(r.id)} color="#2196F3" />
                                <ActionButton label="Promote" icon={ArrowUpCircle} size="xs" onClick={() => promote.mutate(r.id)} color="#00c853" />
                                <ActionButton label="Archive" size="xs" onClick={() => archive.mutate(r.id)} color="#ff1744" />
                              </>
                            )}
                            {r.status === 'challenger' && (
                              <>
                                <ActionButton label="Promote" icon={ArrowUpCircle} size="xs" onClick={() => promote.mutate(r.id)} color="#00c853" />
                                <ActionButton label="Stop Test" size="xs" onClick={() => archive.mutate(r.id)} color="#ff1744" />
                              </>
                            )}
                            {r.status === 'champion' && (
                              <ActionButton label="A/B Test…" icon={GitBranch} size="xs" onClick={() => setChallengeTarget(r.id)} color="#2196F3" disabled />
                            )}
                          </div>
                        </td>
                      </tr>
                      {isSelected && (
                        <tr key={`${r.id}-detail`}>
                          <td colSpan={9} className="px-4 pb-3 bg-[#0a0a0a]">
                            <InferencePanel releaseId={r.id} onClose={() => setSelectedId(null)} />
                          </td>
                        </tr>
                      )}
                    </>
                  )
                })}
              </tbody>
            </table>
            <div className="px-4 py-2 border-t border-[#1e1e1e] text-[10px] text-[#444]">
              {filtered.length} release{filtered.length !== 1 ? 's' : ''} · click a row to view inference logs · refreshes every 15s
            </div>
          </div>
        )}
      </div>

      {/* Challenge Dialog */}
      {challengeTarget && (
        <ChallengeDialog
          releaseId={challengeTarget}
          onClose={() => setChallengeTarget(null)}
          onSubmit={(id, pct) => {
            challenge.mutate({ id, pct })
            setChallengeTarget(null)
          }}
        />
      )}
    </div>
  )
}
