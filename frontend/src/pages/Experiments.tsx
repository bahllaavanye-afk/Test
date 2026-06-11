import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

interface Experiment {
  id: string
  name: string
  status: 'done' | 'running' | 'failed' | 'pending' | 'queued'
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

interface TrainRequest {
  model_name: string
  symbol: string
  interval: string
}

interface TrainResponse {
  experiment_id: string
  name: string
  status: string
  model_name: string
  symbol: string
  interval: string
}

const MODEL_OPTIONS = ['lstm', 'xgboost', 'lorentzian', 'ensemble', 'lightgbm']
const SYMBOL_OPTIONS = ['BTC/USDT', 'ETH/USDT', 'SPY', 'QQQ', 'AAPL', 'TSLA']
const INTERVAL_OPTIONS = ['1h', '4h', '1d', '15m']

function TrainModelForm({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()
  const [modelName, setModelName] = useState(MODEL_OPTIONS[0])
  const [symbol, setSymbol] = useState(SYMBOL_OPTIONS[0])
  const [interval, setInterval] = useState(INTERVAL_OPTIONS[0])
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)

  const mutation = useMutation<TrainResponse, Error, TrainRequest>({
    mutationFn: (req: TrainRequest) =>
      api.post('/ml/train', req).then(r => r.data),
    onSuccess: (data) => {
      setToast({ type: 'success', message: `Training queued: ${data.name} (ID: ${data.experiment_id.slice(0, 8)})` })
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
      setTimeout(() => {
        setToast(null)
        onClose()
      }, 2500)
    },
    onError: (err: Error) => {
      setToast({ type: 'error', message: err.message || 'Failed to queue training job' })
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({ model_name: modelName, symbol, interval })
  }

  return (
    <div className="bg-[#111111] border border-[#f5a623]/30 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-[#e8e8e8]">Train Model</h2>
        <button onClick={onClose} className="text-[#555] hover:text-[#888] transition-colors text-xs">
          Cancel
        </button>
      </div>

      {toast && (
        <div
          className="px-3 py-2 rounded text-xs font-medium"
          style={{
            background: toast.type === 'success' ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)',
            border: `1px solid ${toast.type === 'success' ? 'rgba(0,200,83,0.3)' : 'rgba(255,23,68,0.3)'}`,
            color: toast.type === 'success' ? '#00c853' : '#ff1744',
          }}
        >
          {toast.message}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="block text-[10px] text-[#555] uppercase tracking-wider mb-1">Model</label>
            <select
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623]/50"
            >
              {MODEL_OPTIONS.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-[#555] uppercase tracking-wider mb-1">Symbol</label>
            <select
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623]/50"
            >
              {SYMBOL_OPTIONS.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-[#555] uppercase tracking-wider mb-1">Interval</label>
            <select
              value={interval}
              onChange={e => setInterval(e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623]/50"
            >
              {INTERVAL_OPTIONS.map(i => (
                <option key={i} value={i}>{i}</option>
              ))}
            </select>
          </div>
        </div>
        <button
          type="submit"
          disabled={mutation.isPending}
          className="px-4 py-2 rounded text-xs font-bold bg-[#f5a623] text-black hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {mutation.isPending ? 'Queuing...' : 'Queue Training Job'}
        </button>
      </form>
    </div>
  )
}

interface TrainRequest {
  model_name: string
  symbol: string
  interval: string
}

interface TrainResponse {
  experiment_id: string
  name: string
  status: string
  model_name: string
  symbol: string
  interval: string
}

const MODEL_OPTIONS = ['lstm', 'xgboost', 'lorentzian', 'ensemble', 'lightgbm']
const SYMBOL_OPTIONS = ['BTC/USDT', 'ETH/USDT', 'SPY', 'QQQ', 'AAPL', 'TSLA']
const INTERVAL_OPTIONS = ['1h', '4h', '1d', '15m']

function TrainModelForm({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()
  const [modelName, setModelName] = useState(MODEL_OPTIONS[0])
  const [symbol, setSymbol] = useState(SYMBOL_OPTIONS[0])
  const [interval, setInterval] = useState(INTERVAL_OPTIONS[0])
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)

  const mutation = useMutation<TrainResponse, Error, TrainRequest>({
    mutationFn: (req: TrainRequest) =>
      api.post('/ml/train', req).then(r => r.data),
    onSuccess: (data) => {
      setToast({ type: 'success', message: `Training queued: ${data.name} (ID: ${data.experiment_id.slice(0, 8)})` })
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
      setTimeout(() => {
        setToast(null)
        onClose()
      }, 2500)
    },
    onError: (err: Error) => {
      setToast({ type: 'error', message: err.message || 'Failed to queue training job' })
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({ model_name: modelName, symbol, interval })
  }

  return (
    <div className="bg-[#111111] border border-[#f5a623]/30 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-[#e8e8e8]">Train Model</h2>
        <button onClick={onClose} className="text-[#555] hover:text-[#888] transition-colors text-xs">
          Cancel
        </button>
      </div>

      {toast && (
        <div
          className="px-3 py-2 rounded text-xs font-medium"
          style={{
            background: toast.type === 'success' ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)',
            border: `1px solid ${toast.type === 'success' ? 'rgba(0,200,83,0.3)' : 'rgba(255,23,68,0.3)'}`,
            color: toast.type === 'success' ? '#00c853' : '#ff1744',
          }}
        >
          {toast.message}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="block text-[10px] text-[#555] uppercase tracking-wider mb-1">Model</label>
            <select
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623]/50"
            >
              {MODEL_OPTIONS.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-[#555] uppercase tracking-wider mb-1">Symbol</label>
            <select
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623]/50"
            >
              {SYMBOL_OPTIONS.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-[#555] uppercase tracking-wider mb-1">Interval</label>
            <select
              value={interval}
              onChange={e => setInterval(e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623]/50"
            >
              {INTERVAL_OPTIONS.map(i => (
                <option key={i} value={i}>{i}</option>
              ))}
            </select>
          </div>
        </div>
        <button
          type="submit"
          disabled={mutation.isPending}
          className="px-4 py-2 rounded text-xs font-bold bg-[#f5a623] text-black hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {mutation.isPending ? 'Queuing...' : 'Queue Training Job'}
        </button>
      </form>
    </div>
  )
}

export default function Experiments() {
  const [showTrainForm, setShowTrainForm] = useState(false)

  const { data: exps } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: () => api.get('/experiments/').then(r => r.data),
    refetchInterval: 5_000,
  })

  const experiments: Experiment[] = exps ?? []

  const completedWithSharpe = experiments.filter(
    e => e.status === 'done' && e.test_sharpe !== null
  )
  const bestSharpe = completedWithSharpe.length > 0
    ? Math.max(...completedWithSharpe.map(e => e.test_sharpe as number)).toFixed(2)
    : '—'

  const runningCount = experiments.filter(e => e.status === 'running').length

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">ML Experiments</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowTrainForm(v => !v)}
            className="px-3 py-1.5 rounded text-xs font-bold bg-[#f5a623] text-black hover:opacity-90 transition-opacity"
          >
            {showTrainForm ? 'Cancel' : 'Train Model'}
          </button>
          <span className="text-xs text-[#888888]">Auto-refreshes every 5s · PyTorch Lightning</span>
        </div>
      </div>

      {showTrainForm && (
        <TrainModelForm onClose={() => setShowTrainForm(false)} />
      )}

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

      {/* Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-[#0a0a0a]">
            <tr className="text-xs text-[#888888]">
              {['Name', 'Status', 'Val Acc', 'Val Sharpe', 'Test Sharpe', 'Started', 'Completed'].map(h => (
                <th key={h} className="text-left px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {experiments.map(e => (
              <tr key={e.id} className="border-t border-[#1e1e1e] hover:bg-[#111111]/50 transition-colors">
                <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{e.name}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    e.status === 'done' ? 'bg-[#00c853]/20 text-[#00c853]' :
                    e.status === 'running' ? 'bg-[#2979ff]/20 text-[#2979ff]' :
                    e.status === 'failed' ? 'bg-[#ff1744]/20 text-[#ff1744]' :
                    e.status === 'queued' ? 'bg-[#f5a623]/20 text-[#f5a623]' :
                    'bg-[#1e1e1e] text-[#888888]'}`}>
                    {e.status === 'running' ? '● ' : ''}{e.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-[#e8e8e8]">{e.val_accuracy != null ? `${(e.val_accuracy * 100).toFixed(1)}%` : '—'}</td>
                <td className="px-4 py-3 text-xs text-[#e8e8e8]">{e.val_sharpe != null ? e.val_sharpe.toFixed(3) : '—'}</td>
                <td className="px-4 py-3 text-xs text-[#00c853] font-bold">{e.test_sharpe != null ? e.test_sharpe.toFixed(3) : '—'}</td>
                <td className="px-4 py-3 text-xs text-[#888888]">{e.started_at ? new Date(e.started_at).toLocaleString() : '—'}</td>
                <td className="px-4 py-3 text-xs text-[#888888]">{e.completed_at ? new Date(e.completed_at).toLocaleString() : '—'}</td>
              </tr>
            ))}
            {experiments.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center">
                  <p className="text-sm text-[#888888]">No experiments yet</p>
                  <p className="text-xs text-[#555] mt-1">
                    Click "Train Model" above to queue your first training run, or run:
                    python experiments/run_experiment.py --config lstm_btc_1h.yaml
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
