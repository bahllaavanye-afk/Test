import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../api/client'

interface RiskLimit {
  label: string
  current: number   // 0-100 representing current utilization %
  limit: number     // The limit value shown as context
  unit: string
  criticalAt: number // threshold % at which we show red
}

function RiskGauge({ value, label }: { value: number; label: string }) {
  // value 0-100 percent of drawdown utilization
  const color = value >= 80 ? '#ff1744' : value >= 50 ? '#f5a623' : '#00c853'
  const label2 = value >= 80 ? 'CRITICAL' : value >= 50 ? 'ELEVATED' : 'NORMAL'

  return (
    <div className="bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#888888] uppercase tracking-wider">{label}</span>
        <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ color, background: `${color}20` }}>
          {label2}
        </span>
      </div>
      {/* Gauge track */}
      <div className="risk-gauge-track">
        <div
          className="risk-gauge-needle"
          style={{ left: `${Math.min(Math.max(value, 2), 98)}%` }}
        />
      </div>
      <div className="flex justify-between text-[9px] text-[#555555] font-mono">
        <span>0%</span>
        <span className="font-bold" style={{ color }}>{value.toFixed(1)}%</span>
        <span>100%</span>
      </div>
    </div>
  )
}

function RiskLimitBar({ label, currentPct, limitLabel, color }: { label: string; currentPct: number; limitLabel: string; color: string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[#888888]">{label}</span>
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold" style={{ color }}>{currentPct.toFixed(0)}%</span>
          <span className="text-[#555555] text-[10px]">/ {limitLabel}</span>
        </div>
      </div>
      <div className="h-2 bg-[#1a1a1a] rounded-full overflow-hidden border border-[#1e1e1e]">
        <div
          className="h-full rounded-full transition-all duration-700 relative overflow-hidden"
          style={{
            width: `${Math.min(currentPct, 100)}%`,
            background: currentPct >= 80
              ? 'linear-gradient(90deg, #ff1744, #ff4444)'
              : currentPct >= 50
              ? 'linear-gradient(90deg, #f5a623, #ffc107)'
              : 'linear-gradient(90deg, #00c853, #00e676)',
          }}
        />
      </div>
    </div>
  )
}

export default function RiskManager() {
  const qc = useQueryClient()
  const { data: rules } = useQuery({ queryKey: ['risk-rules'], queryFn: () => api.get('/risk/rules').then(r => r.data) })
  const { data: events } = useQuery({ queryKey: ['risk-events'], queryFn: () => api.get('/risk/events').then(r => r.data), refetchInterval: 10_000 })
  const { data: perf } = useQuery({ queryKey: ['performance'], queryFn: () => api.get('/analytics/performance').then(r => r.data), refetchInterval: 30_000 })

  const [form, setForm] = useState({ rule_type: 'max_drawdown', threshold: '0.10', action: 'halt_all' })
  const addRule = useMutation({
    mutationFn: () => api.post('/risk/rules', { ...form, threshold: parseFloat(form.threshold) }).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['risk-rules'] }),
  })

  const RULE_TYPES = ['max_drawdown', 'arb_drawdown', 'max_position', 'daily_loss_limit', 'correlation_limit']
  const ACTIONS = ['alert', 'halt_bucket', 'halt_all']

  // Compute drawdown utilization from perf data
  const currentDrawdown: number = perf?.max_drawdown != null ? Math.abs(perf.max_drawdown) : 0
  const globalDrawdownPct = Math.min((currentDrawdown / 10) * 100, 100) // 10% = 100% utilization
  const arbDrawdownPct = Math.min((currentDrawdown / 5) * 100, 100)    // 5% arb limit

  const DEFAULT_LIMITS = [
    { label: 'Max Position Size', value: '5% equity', color: '#f5a623' },
    { label: 'Global Circuit Breaker', value: '10% drawdown → halt all', color: '#ff1744' },
    { label: 'Arb Circuit Breaker', value: '5% drawdown → halt arb', color: '#ff1744' },
    { label: 'Cluster Limit', value: '30% equity/cluster', color: '#888888' },
    { label: 'Capital Split', value: '70% arb / 30% directional', color: '#2979ff' },
  ]

  const ACTION_COLORS: Record<string, string> = {
    halt_all: '#ff1744',
    halt_bucket: '#f5a623',
    alert: '#888888',
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-bold">Risk Manager</h1>
        <p className="text-xs text-[#888888] mt-0.5">Real-time risk monitoring with Kelly sizing, correlation limits, and circuit breakers.</p>
      </div>

      {/* Risk gauges row */}
      <div className="grid grid-cols-2 gap-4">
        <RiskGauge value={globalDrawdownPct} label="Global Drawdown Utilization (10% limit)" />
        <RiskGauge value={arbDrawdownPct} label="Arb Strategy Drawdown Utilization (5% limit)" />
      </div>

      {/* Risk limit progress bars */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-4">
        <h2 className="text-sm font-semibold text-white">Risk Limit Utilization</h2>
        <RiskLimitBar
          label="Global Circuit Breaker"
          currentPct={globalDrawdownPct}
          limitLabel="10% drawdown"
          color={globalDrawdownPct >= 80 ? '#ff1744' : globalDrawdownPct >= 50 ? '#f5a623' : '#00c853'}
        />
        <RiskLimitBar
          label="Arb Circuit Breaker"
          currentPct={arbDrawdownPct}
          limitLabel="5% drawdown"
          color={arbDrawdownPct >= 80 ? '#ff1744' : arbDrawdownPct >= 50 ? '#f5a623' : '#00c853'}
        />
        <RiskLimitBar
          label="Capital Allocation (Arb)"
          currentPct={70}
          limitLabel="70% cap"
          color="#2979ff"
        />
        <RiskLimitBar
          label="Capital Allocation (Directional)"
          currentPct={30}
          limitLabel="30% cap"
          color="#f5a623"
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Default Limits */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
          <h2 className="text-sm font-semibold">Default Limits</h2>
          {DEFAULT_LIMITS.map(({ label, value, color }) => (
            <div key={label} className="flex justify-between items-start text-xs border-b border-[#1e1e1e] pb-2 last:border-0">
              <span className="text-[#888888] flex-1 mr-2">{label}</span>
              <span className="font-mono text-right" style={{ color }}>{value}</span>
            </div>
          ))}
        </div>

        {/* Active Rules */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
          <h2 className="text-sm font-semibold">Active Rules</h2>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {rules?.map((r: any) => (
              <div key={r.id} className="flex justify-between text-xs p-2 bg-[#0a0a0a] rounded">
                <span className="font-mono">{r.rule_type}</span>
                <span className="text-[#f5a623]">{(typeof r.threshold === 'number' ? r.threshold * 100 : 0).toFixed(0)}%</span>
                <span className={`${r.action === 'halt_all' ? 'text-[#ff1744]' : r.action === 'halt_bucket' ? 'text-[#f5a623]' : 'text-[#888888]'}`}>{r.action}</span>
              </div>
            )) : (
              <p className="text-xs text-[#555555] text-center py-3">No custom rules. Using defaults.</p>
            )}
          </div>
          <div className="border-t border-[#1e1e1e] pt-3 space-y-2">
            <select value={form.rule_type} onChange={e => setForm(f => ({ ...f, rule_type: e.target.value }))}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs focus:outline-none focus:border-[#f5a623]/40 transition-colors">
              {RULE_TYPES.map(t => <option key={t}>{t}</option>)}
            </select>
            <input type="number" step="0.01" min="0" max="1" value={form.threshold}
              onChange={e => setForm(f => ({ ...f, threshold: e.target.value }))}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs focus:outline-none focus:border-[#f5a623]/40 transition-colors"
              placeholder="Threshold (0-1)" />
            <select value={form.action} onChange={e => setForm(f => ({ ...f, action: e.target.value }))}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs focus:outline-none focus:border-[#f5a623]/40 transition-colors">
              {ACTIONS.map(a => <option key={a}>{a}</option>)}
            </select>
            <button onClick={() => addRule.mutate()} disabled={addRule.isPending}
              className="w-full font-bold py-1.5 rounded text-xs transition-all duration-200 hover:opacity-90 active:scale-95 disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #f5a623, #e09520)', color: '#000' }}>
              ADD RULE
            </button>
          </div>
        </div>

        {/* Risk Events */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold">Risk Events</h2>
            {Array.isArray(events) && events.length > 0 && (
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-[#ff1744]/20 text-[#ff1744]">
                {events.length}
              </span>
            )}
          </div>
          <div className="space-y-2 overflow-y-auto max-h-64">
            {Array.isArray(events) && events.length > 0 ? events.map((e: any) => (
              <div key={e.id} className="text-xs p-2.5 bg-[#ff1744]/8 border border-[#ff1744]/20 rounded animate-fade-in">
                <div className="flex justify-between mb-1">
                  <span className="text-[#ff1744] font-semibold">{e.event_type}</span>
                  <span className="text-[#888888]">{new Date(e.created_at).toLocaleTimeString()}</span>
                </div>
                {e.details && <p className="text-[#555555] text-[10px] truncate">{typeof e.details === 'string' ? e.details : JSON.stringify(e.details)}</p>}
              </div>
            )) : (
              <div className="flex flex-col items-center justify-center py-6 text-center space-y-2">
                <div className="w-8 h-8 rounded-full bg-[#00c853]/10 border border-[#00c853]/30 flex items-center justify-center">
                  <span className="text-[#00c853] text-sm">✓</span>
                </div>
                <p className="text-xs text-[#00c853]">No risk events</p>
                <p className="text-[10px] text-[#555555]">System operating normally</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
