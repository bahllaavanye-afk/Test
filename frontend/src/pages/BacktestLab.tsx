import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

export default function BacktestLab() {
  const qc = useQueryClient()
  const [form, setForm] = useState({
    strategy_name: 'momentum',
    symbol: 'SPY',
    interval: '1d',
    start_date: '2021-01-01',
    end_date: '2024-01-01',
    initial_equity: 100000,
  })

  const { data: runs } = useQuery({
    queryKey: ['backtests'],
    queryFn: () => api.get('/backtests/').then(r => r.data),
    refetchInterval: 5_000,
  })

  const runMutation = useMutation({
    mutationFn: () => api.post('/backtests/', form).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['backtests'] }),
  })

  const field = (k: string, v: any) => setForm(f => ({ ...f, [k]: v }))

  const STRATEGIES = ['momentum', 'mean_reversion', 'rsi_macd', 'breakout', 'supertrend', 'pairs_trading', 'ml_momentum', 'ml_mean_reversion', 'ml_breakout', 'ensemble']

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-bold">Backtest Lab</h1>
      <div className="grid grid-cols-3 gap-6">
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
          <h2 className="text-sm font-semibold text-[#f5a623]">Configure Run</h2>

          <div>
            <label className="text-xs text-[#888888]">Strategy</label>
            <select value={form.strategy_name} onChange={e => field('strategy_name', e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs mt-1">
              {STRATEGIES.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>

          {[['Symbol', 'symbol', 'text', 'SPY'], ['Interval', 'interval', 'text', '1d']].map(([label, key, type, ph]) => (
            <div key={key}>
              <label className="text-xs text-[#888888]">{label}</label>
              <input type={type} value={(form as any)[key]} onChange={e => field(key, e.target.value)} placeholder={ph}
                className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs mt-1" />
            </div>
          ))}

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-[#888888]">Start</label>
              <input type="date" value={form.start_date} onChange={e => field('start_date', e.target.value)}
                className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs mt-1" />
            </div>
            <div>
              <label className="text-xs text-[#888888]">End</label>
              <input type="date" value={form.end_date} onChange={e => field('end_date', e.target.value)}
                className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs mt-1" />
            </div>
          </div>

          <button onClick={() => runMutation.mutate()} disabled={runMutation.isPending}
            className="w-full bg-[#f5a623] text-black font-bold py-2 rounded text-xs hover:bg-[#e09520] disabled:opacity-50 transition-colors">
            {runMutation.isPending ? '⏳ Running...' : '▶ RUN BACKTEST'}
          </button>
        </div>

        <div className="col-span-2 bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <h2 className="text-sm font-semibold mb-3">Run History</h2>
          <div className="space-y-2">
            {runs?.map((r: any) => (
              <div key={r.id} className="flex justify-between items-center text-xs p-2.5 bg-[#0a0a0a] rounded border border-[#1e1e1e]">
                <span className="font-mono text-[#f5a623]">{r.strategy_name}</span>
                <span>{r.symbol} / {r.interval}</span>
                <span className={`px-2 py-0.5 rounded text-xs ${r.status === 'done' ? 'bg-[#00c853]/20 text-[#00c853]' : r.status === 'running' ? 'bg-[#f5a623]/20 text-[#f5a623]' : 'bg-[#1e1e1e] text-[#888888]'}`}>
                  {r.status}
                </span>
                {r.sharpe != null && <span className="text-[#00c853]">Sharpe: {r.sharpe?.toFixed(3)}</span>}
                {r.max_drawdown != null && <span className="text-[#ff1744]">DD: {(r.max_drawdown * 100).toFixed(1)}%</span>}
                {r.total_return != null && <span className="text-[#2979ff]">Ret: {(r.total_return * 100).toFixed(1)}%</span>}
              </div>
            ))}
            {!runs?.length && <p className="text-xs text-[#888888]">No backtests yet. Configure and run one above.</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
