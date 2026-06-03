import { useState, useEffect, useRef } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

const STATUS_STEPS = ['queued', 'loading_data', 'running', 'computing_metrics', 'done']

function getStepIndex(status: string): number {
  if (status === 'completed') return STATUS_STEPS.indexOf('done')
  return STATUS_STEPS.indexOf(status)
}

function isTerminal(status: string): boolean {
  return status === 'done' || status === 'completed' || status === 'failed'
}

function RunProgressBar({ status }: { status: string }) {
  const stepIdx = getStepIndex(status)
  const isFailed = status === 'failed'
  const isDone = status === 'done' || status === 'completed'
  const progress = isDone ? 100 : isFailed ? 0 : Math.max(10, Math.round(((stepIdx + 1) / STATUS_STEPS.length) * 100))

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[#888888] uppercase tracking-wider text-[10px]">Progress</span>
        <span className={`font-mono font-bold text-[10px] ${isDone ? 'text-[#00c853]' : isFailed ? 'text-[#ff1744]' : 'text-[#f5a623]'}`}>
          {isDone ? '100%' : isFailed ? 'FAILED' : `${progress}%`}
        </span>
      </div>
      <div className="h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full relative overflow-hidden transition-all duration-700 ${isDone ? 'bg-[#00c853]' : isFailed ? 'bg-[#ff1744]' : 'bg-[#f5a623]'}`}
          style={{ width: `${progress}%` }}
        >
          {!isDone && !isFailed && <div className="absolute inset-0 progress-shimmer" />}
        </div>
      </div>
      <div className="flex justify-between px-0.5">
        {STATUS_STEPS.map((step, i) => {
          const isActive = i === stepIdx && !isDone && !isFailed
          const isComplete = i < stepIdx || isDone
          return (
            <div key={step} className="flex flex-col items-center gap-0.5" title={step}>
              <div
                className={`w-1.5 h-1.5 rounded-full ${isActive ? 'step-pulse bg-[#f5a623]' : isComplete ? 'bg-[#00c853]' : 'bg-[#1e1e1e]'}`}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function BacktestLab() {
  const qc = useQueryClient()
  const today = new Date().toISOString().slice(0, 10)
  const [form, setForm] = useState({
    strategy_name: 'momentum',
    symbol: 'SPY',
    interval: '1d',
    start_date: '2021-01-01',
    end_date: today,
    initial_equity: 100000,
  })

  // Track the active run ID for polling
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const pollIntervalRef = useRef<number | null>(null)

  const { data: runs, refetch: refetchRuns } = useQuery({
    queryKey: ['backtests'],
    queryFn: () => api.get('/backtests/').then(r => r.data),
  })

  // Poll specific run by ID every 3s until terminal
  useEffect(() => {
    if (!activeRunId) return
    if (pollIntervalRef.current !== null) window.clearInterval(pollIntervalRef.current)

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const result = await api.get(`/backtests/${activeRunId}`).then(r => r.data)
        // Update the backtests list cache
        qc.setQueryData(['backtests'], (old: any[]) => {
          if (!Array.isArray(old)) return old
          const idx = old.findIndex((r: any) => r.id === activeRunId)
          if (idx === -1) return [result, ...old]
          const updated = [...old]
          updated[idx] = result
          return updated
        })
        if (isTerminal(result.status)) {
          window.clearInterval(pollIntervalRef.current!)
          pollIntervalRef.current = null
          setActiveRunId(null)
          refetchRuns()
        }
      } catch {
        // ignore transient errors
      }
    }, 3_000)

    return () => {
      if (pollIntervalRef.current !== null) window.clearInterval(pollIntervalRef.current)
    }
  }, [activeRunId, qc, refetchRuns])

  const { data: strategies } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => api.get('/strategies/').then(r => r.data),
    staleTime: 300_000,
    retry: false,
  })

  const runMutation = useMutation({
    mutationFn: () => api.post('/backtests/', form).then(r => r.data),
    onSuccess: (newRun: any) => {
      // Add the new run to the list immediately
      qc.setQueryData(['backtests'], (old: any[]) => {
        if (!Array.isArray(old)) return [newRun]
        return [newRun, ...old]
      })
      // Start polling the specific run ID
      setActiveRunId(newRun.id)
    },
  })

  const field = (k: string, v: any) => setForm(f => ({ ...f, [k]: v }))

  const STRATEGIES = ['momentum', 'mean_reversion', 'rsi_macd', 'breakout', 'supertrend', 'pairs_trading', 'ml_momentum', 'ml_mean_reversion', 'ml_breakout', 'ensemble']
  const strategyCount = Array.isArray(strategies) ? strategies.length : STRATEGIES.length

  // Date range presets
  const DATE_PRESETS = [
    { label: '1Y', start: new Date(Date.now() - 365 * 86400_000).toISOString().slice(0, 10) },
    { label: '2Y', start: new Date(Date.now() - 2 * 365 * 86400_000).toISOString().slice(0, 10) },
    { label: '3Y', start: new Date(Date.now() - 3 * 365 * 86400_000).toISOString().slice(0, 10) },
    { label: 'All', start: '2020-01-01' },
  ]

  const runsArr: any[] = Array.isArray(runs) ? runs : []
  const runningJob = runsArr.find(r => r.status === 'running' || r.status === 'queued' || r.status === 'loading_data' || r.status === 'computing_metrics')
  const completedRuns = runsArr.filter(r => r.status === 'done' || r.status === 'completed')
  const bestSharpe = completedRuns.reduce((best: number | null, r: any) => {
    if (r.sharpe != null) return best === null ? r.sharpe : Math.max(best, r.sharpe)
    return best
  }, null)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold">Backtest Lab</h1>
          <p className="text-xs text-[#888888] mt-0.5">Walk-forward validated · no in-sample overfitting</p>
        </div>
        {bestSharpe !== null && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-[#111111] border border-[#1e1e1e] rounded-lg glow-green-pulse">
            <span className="text-xs text-[#888888]">Best Sharpe</span>
            <span className="text-sm font-black font-mono text-[#00c853]">{bestSharpe.toFixed(3)}</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Config panel */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#f5a623]">Configure Run</h2>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-[#888888]">Strategy</label>
              <span className="text-[10px] text-[#555555] font-mono">{strategyCount} available</span>
            </div>
            <select value={form.strategy_name} onChange={e => field('strategy_name', e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs hover:border-[#2a2a2a] focus:border-[#f5a623]/40 focus:outline-none transition-colors">
              {STRATEGIES.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>

          {[['Symbol', 'symbol', 'text', 'SPY'], ['Interval', 'interval', 'text', '1d']].map(([label, key, type, ph]) => (
            <div key={key}>
              <label className="text-xs text-[#888888]">{label}</label>
              <input type={type} value={(form as any)[key]} onChange={e => field(key, e.target.value)} placeholder={ph}
                className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs mt-1 hover:border-[#2a2a2a] focus:border-[#f5a623]/40 focus:outline-none transition-colors" />
            </div>
          ))}

          {/* Date range presets */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-[#888888]">Date Range</label>
              <div className="flex gap-1">
                {DATE_PRESETS.map(p => (
                  <button
                    key={p.label}
                    onClick={() => { field('start_date', p.start); field('end_date', today) }}
                    className={`px-2 py-0.5 text-[10px] rounded transition-colors ${
                      form.start_date === p.start && form.end_date === today
                        ? 'bg-[#f5a623] text-black font-bold'
                        : 'bg-[#1e1e1e] text-[#888888] hover:bg-[#2a2a2a] hover:text-[#e8e8e8]'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-[#555555]">Start</label>
                <input type="date" value={form.start_date} onChange={e => field('start_date', e.target.value)}
                  className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs mt-0.5 hover:border-[#2a2a2a] focus:border-[#f5a623]/40 focus:outline-none transition-colors" />
              </div>
              <div>
                <label className="text-[10px] text-[#555555]">End</label>
                <input type="date" value={form.end_date} onChange={e => field('end_date', e.target.value)}
                  className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs mt-0.5 hover:border-[#2a2a2a] focus:border-[#f5a623]/40 focus:outline-none transition-colors" />
              </div>
            </div>
          </div>

          {runningJob ? (
            <div className="bg-[#0a0a0a] border border-[#f5a623]/30 rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#f5a623] font-semibold">Running: {runningJob.strategy_name}</span>
                <span className="font-mono text-[#888888]">{runningJob.symbol}</span>
              </div>
              <RunProgressBar status={runningJob.status} />
            </div>
          ) : (
            <button onClick={() => runMutation.mutate()} disabled={runMutation.isPending}
              className="w-full font-bold py-2 rounded text-xs transition-all duration-200 hover:opacity-90 active:scale-95 disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #f5a623, #e09520)', color: '#000' }}>
              {runMutation.isPending ? 'Submitting...' : '▶ RUN BACKTEST'}
            </button>
          )}
        </div>

        {/* Run history */}
        <div className="col-span-2 bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold">Run History</h2>
            <span className="text-[10px] text-[#555555] font-mono">{runsArr.length} runs</span>
          </div>
          <div className="space-y-2">
            {runs?.map((r: any) => (
              <div key={r.id} className="flex justify-between items-center text-xs p-2.5 bg-[#0a0a0a] rounded border border-[#1e1e1e]">
                <span className="font-mono text-[#f5a623]">{r.strategy_name}</span>
                <span>{r.symbol} / {r.interval}</span>
                <span className={`px-2 py-0.5 rounded text-xs ${r.status === 'done' || r.status === 'completed' ? 'bg-[#00c853]/20 text-[#00c853]' : r.status === 'running' ? 'bg-[#f5a623]/20 text-[#f5a623]' : r.status === 'failed' ? 'bg-[#ff1744]/20 text-[#ff1744]' : 'bg-[#1e1e1e] text-[#888888]'}`}>
                  {r.status}
                </span>
                {r.sharpe != null && <span className="text-[#00c853]">Sharpe: {r.sharpe?.toFixed(3)}</span>}
                {r.max_drawdown != null && <span className="text-[#ff1744]">DD: {(r.max_drawdown * 100).toFixed(1)}%</span>}
                {r.total_return != null && <span className="text-[#2979ff]">Ret: {(r.total_return * 100).toFixed(1)}%</span>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
