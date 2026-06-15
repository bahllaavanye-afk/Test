/**
 * RiskControls — Per-bot risk rules management
 *
 * Connects to:
 *   GET  /risk/         — dashboard summary (drawdown limit, position limit, circuit breaker)
 *   GET  /risk/rules    — list all risk rules
 *   POST /risk/rules    — create a new rule
 *   DELETE /risk/rules/:id — delete a rule
 *   GET  /risk/events   — recent risk events
 */
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface RiskSummary {
  active_rules: number
  recent_events: number
  circuit_breaker: 'normal' | 'tripped' | 'warning'
  regime: 'bull' | 'sideways' | 'bear'
  max_drawdown_limit_pct: number
  position_limit_pct: number
}

interface RiskRule {
  id: string
  rule_type: string
  threshold: number
  action: string
  is_active: boolean
}

interface RiskEvent {
  id: string
  event_type: string
  details: string | null
  created_at: string
}

interface RuleCreate {
  rule_type: string
  threshold: number
  action: string
}

// ─── Constants ────────────────────────────────────────────────────────────────

const RULE_TYPES: { value: string; label: string; unit: string; description: string }[] = [
  { value: 'max_drawdown_pct', label: 'Max Drawdown %', unit: '%', description: 'Halt when portfolio drawdown exceeds threshold' },
  { value: 'max_position_size_pct', label: 'Max Position Size %', unit: '%', description: 'Reject orders that exceed % of portfolio' },
  { value: 'max_daily_loss_usd', label: 'Max Daily Loss $', unit: '$', description: 'Stop trading when daily loss exceeds amount' },
  { value: 'max_open_positions', label: 'Max Open Positions', unit: '#', description: 'Limit total simultaneous open positions' },
  { value: 'min_win_rate_pct', label: 'Min Win Rate %', unit: '%', description: 'Pause bot if rolling win rate drops below threshold' },
  { value: 'max_loss_streak', label: 'Max Loss Streak', unit: '#', description: 'Halt after N consecutive losing trades' },
  { value: 'max_notional_usd', label: 'Max Notional $', unit: '$', description: 'Cap total notional exposure per symbol' },
  { value: 'max_correlation', label: 'Max Correlation', unit: 'r', description: 'Block trade if position correlation exceeds value' },
]

const ACTIONS: { value: string; label: string; color: string }[] = [
  { value: 'alert', label: 'Alert Only', color: '#f5a623' },
  { value: 'halt_bot', label: 'Halt Bot', color: '#ff9800' },
  { value: 'halt_all', label: 'Halt All Trading', color: '#ff1744' },
  { value: 'reduce_position', label: 'Reduce Position 50%', color: '#ff6b35' },
  { value: 'close_all', label: 'Close All Positions', color: '#ff1744' },
]

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ruleTypeLabel(type: string): string {
  return RULE_TYPES.find(r => r.value === type)?.label ?? type
}

function actionLabel(action: string): string {
  return ACTIONS.find(a => a.value === action)?.label ?? action
}

function actionColor(action: string): string {
  return ACTIONS.find(a => a.value === action)?.color ?? '#888'
}

function regimeColor(regime: string): string {
  return regime === 'bull' ? '#00c853' : regime === 'bear' ? '#ff1744' : '#f5a623'
}

function cbColor(status: string): string {
  return status === 'normal' ? '#00c853' : status === 'tripped' ? '#ff1744' : '#f5a623'
}

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return iso
  }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div style={{
      background: '#111', border: '1px solid #1e1e1e', borderRadius: 8,
      padding: '16px 20px', flex: 1, minWidth: 140,
    }}>
      <div style={{ color: '#888', fontSize: 11, fontFamily: 'Inter, sans-serif', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ color: color ?? '#e8e8e8', fontSize: 22, fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, lineHeight: 1 }}>
        {value}
      </div>
      {sub && (
        <div style={{ color: '#555', fontSize: 11, fontFamily: 'Inter, sans-serif', marginTop: 4 }}>{sub}</div>
      )}
    </div>
  )
}

function RuleRow({ rule, onDelete }: { rule: RiskRule; onDelete: (id: string) => void }) {
  const [confirming, setConfirming] = useState(false)

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr 100px 160px 90px 80px',
      gap: 12, alignItems: 'center',
      padding: '12px 16px',
      borderBottom: '1px solid #1a1a1a',
      transition: 'background 0.15s',
    }}
      onMouseEnter={e => (e.currentTarget.style.background = '#141414')}
      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
    >
      {/* Rule type */}
      <div>
        <div style={{ color: '#e8e8e8', fontSize: 13, fontFamily: 'Inter, sans-serif', fontWeight: 500 }}>
          {ruleTypeLabel(rule.rule_type)}
        </div>
        <div style={{ color: '#555', fontSize: 11, fontFamily: 'Inter, sans-serif', marginTop: 2 }}>
          {rule.rule_type}
        </div>
      </div>

      {/* Threshold */}
      <div style={{ color: '#f5a623', fontFamily: 'JetBrains Mono, monospace', fontSize: 14, fontWeight: 700 }}>
        {rule.threshold}
      </div>

      {/* Action badge */}
      <div>
        <span style={{
          background: actionColor(rule.action) + '22',
          color: actionColor(rule.action),
          border: `1px solid ${actionColor(rule.action)}44`,
          borderRadius: 4, padding: '2px 8px',
          fontSize: 11, fontFamily: 'Inter, sans-serif', fontWeight: 600,
        }}>
          {actionLabel(rule.action)}
        </span>
      </div>

      {/* Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{
          width: 8, height: 8, borderRadius: '50%',
          background: rule.is_active ? '#00c853' : '#333',
        }} />
        <span style={{ color: rule.is_active ? '#00c853' : '#555', fontSize: 12, fontFamily: 'Inter, sans-serif' }}>
          {rule.is_active ? 'Active' : 'Off'}
        </span>
      </div>

      {/* Delete */}
      <div>
        {confirming ? (
          <div style={{ display: 'flex', gap: 6 }}>
            <button
              onClick={() => { onDelete(rule.id); setConfirming(false) }}
              style={{
                background: '#ff174422', color: '#ff1744', border: '1px solid #ff174444',
                borderRadius: 4, padding: '3px 8px', cursor: 'pointer', fontSize: 11,
              }}
            >
              Confirm
            </button>
            <button
              onClick={() => setConfirming(false)}
              style={{
                background: 'transparent', color: '#888', border: '1px solid #2a2a2a',
                borderRadius: 4, padding: '3px 8px', cursor: 'pointer', fontSize: 11,
              }}
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirming(true)}
            style={{
              background: 'transparent', color: '#555', border: '1px solid #2a2a2a',
              borderRadius: 4, padding: '3px 10px', cursor: 'pointer', fontSize: 11,
              transition: 'all 0.15s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.color = '#ff1744'
              e.currentTarget.style.borderColor = '#ff174444'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.color = '#555'
              e.currentTarget.style.borderColor = '#2a2a2a'
            }}
          >
            Delete
          </button>
        )}
      </div>
    </div>
  )
}

function AddRuleForm({ onAdd }: { onAdd: (rule: RuleCreate) => void }) {
  const [ruleType, setRuleType] = useState('max_drawdown_pct')
  const [threshold, setThreshold] = useState('')
  const [action, setAction] = useState('alert')
  const [error, setError] = useState('')

  const selectedMeta = RULE_TYPES.find(r => r.value === ruleType)

  function handleSubmit() {
    const val = parseFloat(threshold)
    if (isNaN(val) || val <= 0) {
      setError('Enter a valid positive threshold')
      return
    }
    setError('')
    onAdd({ rule_type: ruleType, threshold: val, action })
    setThreshold('')
  }

  const inputStyle: React.CSSProperties = {
    background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 6,
    color: '#e8e8e8', fontSize: 13, fontFamily: 'Inter, sans-serif',
    padding: '8px 12px', outline: 'none', width: '100%', boxSizing: 'border-box',
  }

  return (
    <div style={{ padding: '20px', borderTop: '1px solid #1e1e1e' }}>
      <div style={{ color: '#888', fontSize: 11, fontFamily: 'Inter, sans-serif', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
        Add Risk Rule
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 140px 180px 140px', gap: 10, alignItems: 'end' }}>
        {/* Rule type */}
        <div>
          <div style={{ color: '#555', fontSize: 11, marginBottom: 4, fontFamily: 'Inter, sans-serif' }}>Rule Type</div>
          <select value={ruleType} onChange={e => setRuleType(e.target.value)} style={inputStyle}>
            {RULE_TYPES.map(r => (
              <option key={r.value} value={r.value}>{r.label}</option>
            ))}
          </select>
          {selectedMeta && (
            <div style={{ color: '#444', fontSize: 10, marginTop: 3, fontFamily: 'Inter, sans-serif' }}>
              {selectedMeta.description}
            </div>
          )}
        </div>

        {/* Threshold */}
        <div>
          <div style={{ color: '#555', fontSize: 11, marginBottom: 4, fontFamily: 'Inter, sans-serif' }}>
            Threshold {selectedMeta ? `(${selectedMeta.unit})` : ''}
          </div>
          <input
            type="number"
            value={threshold}
            onChange={e => setThreshold(e.target.value)}
            placeholder="e.g. 10"
            style={inputStyle}
          />
        </div>

        {/* Action */}
        <div>
          <div style={{ color: '#555', fontSize: 11, marginBottom: 4, fontFamily: 'Inter, sans-serif' }}>Action</div>
          <select value={action} onChange={e => setAction(e.target.value)} style={inputStyle}>
            {ACTIONS.map(a => (
              <option key={a.value} value={a.value}>{a.label}</option>
            ))}
          </select>
        </div>

        {/* Button */}
        <div>
          {error && <div style={{ color: '#ff1744', fontSize: 11, marginBottom: 4 }}>{error}</div>}
          <button
            onClick={handleSubmit}
            style={{
              background: '#f5a623', color: '#000', border: 'none', borderRadius: 6,
              padding: '9px 20px', fontWeight: 700, fontSize: 13, cursor: 'pointer',
              fontFamily: 'Inter, sans-serif', width: '100%', transition: 'opacity 0.15s',
            }}
            onMouseEnter={e => (e.currentTarget.style.opacity = '0.85')}
            onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
          >
            Add Rule
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function RiskControls() {
  const qc = useQueryClient()

  const { data: summary, isLoading: summaryLoading } = useQuery<RiskSummary>({
    queryKey: ['risk-summary'],
    queryFn: () => api.get('/risk/').then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: rules = [], isLoading: rulesLoading, error: rulesError } = useQuery<RiskRule[]>({
    queryKey: ['risk-rules'],
    queryFn: () => api.get('/risk/rules').then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: events = [], isLoading: eventsLoading } = useQuery<RiskEvent[]>({
    queryKey: ['risk-events'],
    queryFn: () => api.get('/risk/events?limit=10').then(r => r.data),
    refetchInterval: 30_000,
  })

  const createRule = useMutation({
    mutationFn: (body: RuleCreate) => api.post('/risk/rules', body).then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['risk-rules'] })
      qc.invalidateQueries({ queryKey: ['risk-summary'] })
    },
  })

  const deleteRule = useMutation({
    mutationFn: (id: string) => api.delete(`/risk/rules/${id}`).then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['risk-rules'] })
      qc.invalidateQueries({ queryKey: ['risk-summary'] })
    },
  })

  const pageStyle: React.CSSProperties = {
    background: '#0a0a0a',
    minHeight: '100vh',
    padding: '24px 28px',
    fontFamily: 'Inter, sans-serif',
    color: '#e8e8e8',
  }

  // ── Loading / Error ──────────────────────────────────────────────────────

  if (rulesLoading) {
    return (
      <div style={pageStyle}>
        <div style={{ color: '#555', fontSize: 13, marginTop: 60, textAlign: 'center' }}>
          Loading risk controls…
        </div>
      </div>
    )
  }

  if (rulesError) {
    return (
      <div style={pageStyle}>
        <div style={{
          background: '#ff174411', border: '1px solid #ff174433', borderRadius: 8,
          padding: 16, color: '#ff1744', fontSize: 13, marginTop: 60,
        }}>
          Failed to load risk rules: {(rulesError as Error).message}
        </div>
      </div>
    )
  }

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div style={pageStyle}>
      {/* Page header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#e8e8e8' }}>
          Risk Controls
        </h1>
        <div style={{ color: '#555', fontSize: 13, marginTop: 4 }}>
          Per-strategy guardrails — drawdown limits, position caps, circuit breakers
        </div>
      </div>

      {/* Summary KPI bar */}
      {summaryLoading ? (
        <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
          {[...Array(5)].map((_, i) => (
            <div key={i} style={{ flex: 1, minWidth: 140, height: 72, background: '#111', borderRadius: 8, border: '1px solid #1e1e1e' }} />
          ))}
        </div>
      ) : summary ? (
        <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
          <StatCard
            label="Circuit Breaker"
            value={summary.circuit_breaker.toUpperCase()}
            color={cbColor(summary.circuit_breaker)}
          />
          <StatCard
            label="Market Regime"
            value={summary.regime.toUpperCase()}
            color={regimeColor(summary.regime)}
          />
          <StatCard
            label="Active Rules"
            value={summary.active_rules}
            sub="enforced guardrails"
          />
          <StatCard
            label="Max Drawdown Limit"
            value={`${summary.max_drawdown_limit_pct}%`}
            color="#f5a623"
          />
          <StatCard
            label="Max Position Size"
            value={`${summary.position_limit_pct}%`}
            color="#f5a623"
            sub="of portfolio"
          />
        </div>
      ) : null}

      {/* Two-column layout: rules left, events right */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16, alignItems: 'start' }}>

        {/* Rules panel */}
        <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 10 }}>
          {/* Header */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '16px 20px', borderBottom: '1px solid #1e1e1e',
          }}>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600 }}>Risk Rules</div>
              <div style={{ color: '#555', fontSize: 12, marginTop: 2 }}>
                {rules.length} rule{rules.length !== 1 ? 's' : ''} configured
              </div>
            </div>
            <div style={{
              background: createRule.isPending ? '#1a1a1a' : '#f5a62322',
              color: createRule.isPending ? '#555' : '#f5a623',
              border: `1px solid ${createRule.isPending ? '#2a2a2a' : '#f5a62344'}`,
              borderRadius: 6, padding: '4px 12px', fontSize: 12, fontWeight: 600,
            }}>
              {createRule.isPending ? 'Saving…' : `${rules.filter(r => r.is_active).length} Active`}
            </div>
          </div>

          {/* Column headers */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 100px 160px 90px 80px',
            gap: 12, padding: '8px 16px',
            color: '#555', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em',
            borderBottom: '1px solid #1a1a1a',
          }}>
            <div>Rule Type</div>
            <div>Threshold</div>
            <div>Action</div>
            <div>Status</div>
            <div />
          </div>

          {/* Rule rows */}
          {rules.length === 0 ? (
            <div style={{ padding: '32px 20px', textAlign: 'center', color: '#444', fontSize: 13 }}>
              No risk rules configured yet. Add one below.
            </div>
          ) : (
            rules.map(rule => (
              <RuleRow
                key={rule.id}
                rule={rule}
                onDelete={(id) => deleteRule.mutate(id)}
              />
            ))
          )}

          {/* Add rule form */}
          <AddRuleForm onAdd={(body) => createRule.mutate(body)} />

          {/* Mutation error */}
          {(createRule.isError || deleteRule.isError) && (
            <div style={{
              margin: '0 20px 16px',
              background: '#ff174411', border: '1px solid #ff174433',
              borderRadius: 6, padding: '10px 14px', color: '#ff1744', fontSize: 12,
            }}>
              {(createRule.error as Error | null)?.message
                ?? (deleteRule.error as Error | null)?.message
                ?? 'Operation failed'}
            </div>
          )}
        </div>

        {/* Events panel */}
        <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 10 }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid #1e1e1e' }}>
            <div style={{ fontSize: 15, fontWeight: 600 }}>Recent Risk Events</div>
            <div style={{ color: '#555', fontSize: 12, marginTop: 2 }}>Last 10 triggered events</div>
          </div>

          {eventsLoading ? (
            <div style={{ padding: 20 }}>
              {[...Array(5)].map((_, i) => (
                <div key={i} style={{ height: 40, background: '#1a1a1a', borderRadius: 6, marginBottom: 8 }} />
              ))}
            </div>
          ) : events.length === 0 ? (
            <div style={{ padding: '32px 20px', textAlign: 'center', color: '#444', fontSize: 13 }}>
              No risk events — system operating normally
            </div>
          ) : (
            <div style={{ padding: '8px 0' }}>
              {events.map(ev => (
                <div
                  key={ev.id}
                  style={{ padding: '10px 20px', borderBottom: '1px solid #1a1a1a' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{
                      background: '#ff174411', color: '#ff1744',
                      border: '1px solid #ff174433', borderRadius: 4,
                      padding: '2px 7px', fontSize: 11, fontWeight: 600,
                    }}>
                      {ev.event_type}
                    </span>
                    <span style={{ color: '#444', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}>
                      {fmtDate(ev.created_at)}
                    </span>
                  </div>
                  {ev.details && (
                    <div style={{ color: '#666', fontSize: 12, marginTop: 4, fontFamily: 'Inter, sans-serif' }}>
                      {ev.details}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

      </div>

      {/* Preset rule templates */}
      <div style={{ marginTop: 16 }}>
        <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 10, padding: '16px 20px' }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Quick Presets</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {[
              { label: 'Conservative', rules: [
                  { rule_type: 'max_drawdown_pct', threshold: 5, action: 'halt_all' },
                  { rule_type: 'max_position_size_pct', threshold: 5, action: 'alert' },
                  { rule_type: 'max_daily_loss_usd', threshold: 500, action: 'halt_all' },
              ]},
              { label: 'Moderate', rules: [
                  { rule_type: 'max_drawdown_pct', threshold: 10, action: 'halt_all' },
                  { rule_type: 'max_position_size_pct', threshold: 10, action: 'alert' },
                  { rule_type: 'max_daily_loss_usd', threshold: 1000, action: 'halt_bot' },
              ]},
              { label: 'Aggressive', rules: [
                  { rule_type: 'max_drawdown_pct', threshold: 20, action: 'alert' },
                  { rule_type: 'max_position_size_pct', threshold: 20, action: 'alert' },
                  { rule_type: 'max_loss_streak', threshold: 5, action: 'halt_bot' },
              ]},
            ].map(preset => (
              <button
                key={preset.label}
                onClick={() => preset.rules.forEach(r => createRule.mutate(r))}
                disabled={createRule.isPending}
                style={{
                  background: '#1a1a1a', color: '#e8e8e8', border: '1px solid #2a2a2a',
                  borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12,
                  fontFamily: 'Inter, sans-serif', transition: 'all 0.15s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = '#f5a623'
                  e.currentTarget.style.color = '#f5a623'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = '#2a2a2a'
                  e.currentTarget.style.color = '#e8e8e8'
                }}
              >
                {preset.label} Preset ({preset.rules.length} rules)
              </button>
            ))}
          </div>
          <div style={{ color: '#444', fontSize: 11, marginTop: 8 }}>
            Presets add multiple rules at once — you can delete individual rules after.
          </div>
        </div>
      </div>

    </div>
  )
}
