import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../api/client'

export default function RiskManager() {
  const qc = useQueryClient()
  const { data: rules } = useQuery({ queryKey: ['risk-rules'], queryFn: () => api.get('/risk/rules').then(r => r.data) })
  const { data: events } = useQuery({ queryKey: ['risk-events'], queryFn: () => api.get('/risk/events').then(r => r.data), refetchInterval: 10_000 })
  const [form, setForm] = useState({ rule_type: 'max_drawdown', threshold: '0.10', action: 'halt_all' })
  const addRule = useMutation({
    mutationFn: () => api.post('/risk/rules', { ...form, threshold: parseFloat(form.threshold) }).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['risk-rules'] }),
  })

  const RULE_TYPES = ['max_drawdown', 'arb_drawdown', 'max_position', 'daily_loss_limit', 'correlation_limit']
  const ACTIONS = ['alert', 'halt_bucket', 'halt_all']

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-bold">Risk Manager</h1>
      <p className="text-xs text-[#888888]">Real-time risk monitoring with Kelly sizing, correlation limits, and circuit breakers.</p>

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
          <h2 className="text-sm font-semibold">Default Limits</h2>
          {[
            ['Max Position Size', '5% equity', '#f5a623'],
            ['Global Circuit Breaker', '10% drawdown → halt all', '#ff1744'],
            ['Arb Circuit Breaker', '5% drawdown → halt arb', '#ff1744'],
            ['Cluster Limit', '30% equity per correlation cluster', '#888888'],
            ['Capital Split', '70% arb / 30% directional', '#2979ff'],
          ].map(([label, value, color]) => (
            <div key={label as string} className="flex justify-between text-xs border-b border-[#1e1e1e] pb-2 last:border-0">
              <span className="text-[#888888]">{label}</span>
              <span style={{ color: color as string }}>{value}</span>
            </div>
          ))}
        </div>

        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
          <h2 className="text-sm font-semibold">Active Rules</h2>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {rules?.map((r: any) => (
              <div key={r.id} className="flex justify-between text-xs p-2 bg-[#0a0a0a] rounded">
                <span className="font-mono">{r.rule_type}</span>
                <span className="text-[#f5a623]">{(r.threshold * 100).toFixed(0)}%</span>
                <span className={`${r.action === 'halt_all' ? 'text-[#ff1744]' : r.action === 'halt_bucket' ? 'text-[#f5a623]' : 'text-[#888888]'}`}>{r.action}</span>
              </div>
            ))}
          </div>
          <div className="border-t border-[#1e1e1e] pt-3 space-y-2">
            <select value={form.rule_type} onChange={e => setForm(f => ({ ...f, rule_type: e.target.value }))}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs">
              {RULE_TYPES.map(t => <option key={t}>{t}</option>)}
            </select>
            <input type="number" step="0.01" min="0" max="1" value={form.threshold}
              onChange={e => setForm(f => ({ ...f, threshold: e.target.value }))}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs"
              placeholder="Threshold (0-1)" />
            <select value={form.action} onChange={e => setForm(f => ({ ...f, action: e.target.value }))}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs">
              {ACTIONS.map(a => <option key={a}>{a}</option>)}
            </select>
            <button onClick={() => addRule.mutate()} className="w-full bg-[#f5a623] text-black font-bold py-1.5 rounded text-xs hover:bg-[#e09520] transition-colors">
              ADD RULE
            </button>
          </div>
        </div>

        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <h2 className="text-sm font-semibold mb-3">Risk Events</h2>
          <div className="space-y-2 overflow-y-auto max-h-64">
            {events?.map((e: any) => (
              <div key={e.id} className="text-xs p-2.5 bg-[#ff1744]/10 border border-[#ff1744]/20 rounded">
                <div className="flex justify-between mb-1">
                  <span className="text-[#ff1744] font-semibold">{e.event_type}</span>
                  <span className="text-[#888888]">{new Date(e.created_at).toLocaleTimeString()}</span>
                </div>
                {e.details && <p className="text-[#888888]">{JSON.stringify(e.details)}</p>}
              </div>
            ))}
            {!events?.length && <p className="text-xs text-[#888888]">No risk events. System operating normally.</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
