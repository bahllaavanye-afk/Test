import { useQuery, useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../api/client'

interface MLModel {
  id: string
  model_type: string
  symbol: string | null
  interval: string | null
  val_accuracy: number | null
  val_sharpe: number | null
  is_active: boolean
  // Backend emits `trained_at`; older payloads used `created_at`. Accept either.
  created_at?: string | null
  trained_at?: string | null
}

// ── Defensive helpers ────────────────────────────────────────────────────────
// API fields can arrive as undefined/null/strings. Coerce defensively so a
// single bad field never throws during render and blanks the whole page.
const num = (v: unknown): number =>
  typeof v === 'number' && Number.isFinite(v) ? v : 0

const fmtNum = (v: unknown, digits = 3): string => num(v).toFixed(digits)

const asArray = <T,>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : [])

const fmtDate = (s: string | null | undefined): string => {
  if (!s) return '—'
  const d = new Date(s)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleDateString()
}

interface Experiment {
  id: string
  name: string
  status: string
  val_accuracy: number | null
  val_sharpe: number | null
  test_sharpe: number | null
  config: Record<string, unknown>
  started_at: string | null
  completed_at: string | null
}

function StatCard({ label, value, color = '#f5a623', sub }: { label: string; value: string | number; color?: string; sub?: string }) {
  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
      <p className="text-xs text-[#888888] mb-1">{label}</p>
      <p className="text-xl font-bold font-mono" style={{ color }}>{value}</p>
      {sub && <p className="text-xs text-[#555] mt-1">{sub}</p>}
    </div>
  )
}

export default function MLInsights() {
  const {
    data: modelsData,
    isLoading: modelsLoading,
    error: modelsError,
  } = useQuery<MLModel[]>({
    queryKey: ['ml-models'],
    queryFn: () => api.get('/ml/models').then(r => r.data),
    refetchInterval: 30_000,
  })

  const {
    data: expsData,
    isLoading: expsLoading,
    error: expsError,
  } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: () => api.get('/experiments/').then(r => r.data),
    refetchInterval: 10_000,
  })

  // Never assume the API returned an array — an error body or paginated object
  // would otherwise throw on .filter/.map and blank the entire page.
  const models = asArray<MLModel>(modelsData)
  const exps = asArray<Experiment>(expsData)

  const activeModels = models.filter(m => m.is_active)
  const completedExps = exps.filter(e => e.status === 'done')
  const bestSharpe = completedExps.length > 0
    ? Math.max(...completedExps.map(e => num(e.test_sharpe)))
    : 0
  const avgAccuracy = completedExps.length > 0
    ? completedExps.reduce((sum, e) => sum + num(e.val_accuracy), 0) / completedExps.length
    : 0

  const [optimizeResult, setOptimizeResult] = useState<Record<string, unknown> | null>(null)
  const optimizeMutation = useMutation({
    mutationFn: (symbol: string) =>
      api.post('/ml/ensemble/optimize-weights', { symbol, n_splits: 5, lookback_days: 365 }).then(r => r.data),
    onSuccess: (data) => setOptimizeResult(data),
  })

  const MODEL_TYPE_LABELS: Record<string, string> = {
    lstm: 'LSTM (Bidirectional + Attention)',
    transformer: 'Temporal Fusion Transformer',
    xgboost: 'XGBoost + Optuna HPO',
    lightgbm: 'LightGBM',
    lorentzian_knn: 'Lorentzian KNN',
    ensemble: 'Weighted Ensemble',
    a3c_lstm: 'A3C-LSTM (RL)',
    ppo: 'PPO Execution Agent',
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold">ML Insights</h1>
          <p className="text-xs text-[#888888] mt-0.5">Model registry · experiment results · signal confidence</p>
        </div>
        <span className="text-xs text-[#888888]">Live from model registry</span>
      </div>

      {(modelsError || expsError) && (
        <div className="bg-[#ff1744]/10 border border-[#ff1744]/40 rounded-lg p-3 text-xs text-[#ff1744]">
          {modelsError && <p>Failed to load models: {(modelsError as Error)?.message ?? 'unknown error'}</p>}
          {expsError && <p>Failed to load experiments: {(expsError as Error)?.message ?? 'unknown error'}</p>}
        </div>
      )}

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Registered Models" value={models.length} color="#2196F3" sub={`${activeModels.length} active`} />
        <StatCard label="Completed Experiments" value={completedExps.length} color="#00c853" sub={`of ${exps.length} total`} />
        <StatCard label="Best Test Sharpe" value={fmtNum(bestSharpe)} color="#f5a623" sub="highest across all runs" />
        <StatCard label="Avg Val Accuracy" value={avgAccuracy > 0 ? `${(avgAccuracy * 100).toFixed(1)}%` : '—'} color="#9C27B0" sub="completed runs" />
      </div>

      {/* Model Registry */}
      <div>
        <h2 className="text-sm font-semibold mb-3 text-[#888888] uppercase tracking-wider">Model Registry</h2>
        {modelsLoading ? (
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6 text-center text-[#555] text-sm">
            Loading models…
          </div>
        ) : models.length === 0 ? (
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6 text-center">
            <div>No models trained yet. Use the Experiments tab to train your first model.</div>
          </div>
        ) : (
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-[#0a0a0a]">
                <tr className="text-xs text-[#888888]">
                  {['Type', 'Symbol', 'Interval', 'Val Accuracy', 'Val Sharpe', 'Status', 'Registered'].map(h => (
                    <th key={h} className="text-left px-4 py-3">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.id} className="border-t border-[#1e1e1e] hover:bg-[#1a1a1a] transition-colors">
                    <td className="px-4 py-3 text-xs text-[#e8e8e8]">
                      <span className="font-mono">{m.model_type}</span>
                      <span className="ml-2 text-[#555] hidden xl:inline">
                        {MODEL_TYPE_LABELS[m.model_type] ? `· ${MODEL_TYPE_LABELS[m.model_type]}` : ''}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs font-mono text-[#888888]">{m.symbol ?? '—'}</td>
                    <td className="px-4 py-3 text-xs text-[#888888]">{m.interval ?? '—'}</td>
                    <td className="px-4 py-3 text-xs">
                      {m.val_accuracy != null
                        ? <span style={{ color: num(m.val_accuracy) > 0.55 ? '#00c853' : '#888' }}>{(num(m.val_accuracy) * 100).toFixed(1)}%</span>
                        : <span className="text-[#555]">—</span>}
                    </td>
                    <td className="px-4 py-3 text-xs font-bold">
                      {m.val_sharpe != null
                        ? <span style={{ color: num(m.val_sharpe) > 1.5 ? '#00c853' : num(m.val_sharpe) > 0.8 ? '#f5a623' : '#888' }}>
                            {fmtNum(m.val_sharpe)}
                          </span>
                        : <span className="text-[#555]">—</span>}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        m.is_active ? 'bg-[#00c853]/15 text-[#00c853]' : 'bg-[#1e1e1e] text-[#555]'
                      }`}>
                        {m.is_active ? '● live' : 'inactive'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-[#555]">
                      {fmtDate(m.trained_at ?? m.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Top Experiments by Test Sharpe */}
      <div>
        <h2 className="text-sm font-semibold mb-3 text-[#888888] uppercase tracking-wider">
          Top Experiments — by Test Sharpe
        </h2>
        {expsLoading ? (
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6 text-center text-[#555] text-sm">
            Loading experiments…
          </div>
        ) : completedExps.length === 0 ? (
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6 text-center">
            <p className="text-[#555] text-sm">No completed experiments yet.</p>
            <p className="text-[#444] text-xs mt-2">
              Run: <code className="text-[#f5a623]">python experiments/run_experiment.py --config lstm_btc_1h.yaml</code>
            </p>
          </div>
        ) : (
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-[#0a0a0a]">
                <tr className="text-xs text-[#888888]">
                  {['Experiment', 'Model', 'Val Acc', 'Val Sharpe', 'Test Sharpe', 'Completed'].map(h => (
                    <th key={h} className="text-left px-4 py-3">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...completedExps]
                  .sort((a, b) => num(b.test_sharpe) - num(a.test_sharpe))
                  .slice(0, 20)
                  .map(e => (
                    <tr key={e.id} className="border-t border-[#1e1e1e] hover:bg-[#1a1a1a] transition-colors">
                      <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{e.name}</td>
                      <td className="px-4 py-3 text-xs text-[#888888]">
                        {(e.config as any)?.model ?? '—'}
                      </td>
                      <td className="px-4 py-3 text-xs">
                        {e.val_accuracy != null ? `${(num(e.val_accuracy) * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-4 py-3 text-xs">
                        {e.val_sharpe != null ? fmtNum(e.val_sharpe) : '—'}
                      </td>
                      <td className="px-4 py-3 text-xs font-bold">
                        {e.test_sharpe != null
                          ? <span style={{ color: num(e.test_sharpe) > 1.5 ? '#00c853' : num(e.test_sharpe) > 0.8 ? '#f5a623' : '#888' }}>
                              {fmtNum(e.test_sharpe)}
                            </span>
                          : <span className="text-[#555]">—</span>}
                      </td>
                      <td className="px-4 py-3 text-xs text-[#555]">
                        {fmtDate(e.completed_at)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Ensemble Weight Optimizer */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3 text-[#888888] uppercase tracking-wider">
          Walk-Forward Ensemble Optimization
        </h2>
        <p className="text-xs text-[#888888] mb-3">
          Runs SLSQP walk-forward optimization across 5 folds to find the Sharpe-maximising blend
          of LSTM · XGBoost · Lorentzian weights. Updates in-memory immediately.
        </p>
        <div className="flex items-center gap-3">
          <button
            onClick={() => optimizeMutation.mutate('SPY')}
            disabled={optimizeMutation.isPending || !activeModels.length}
            className="px-4 py-2 text-xs font-semibold rounded border border-[#f5a623] text-[#f5a623] hover:bg-[#f5a623]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {optimizeMutation.isPending ? 'Optimising…' : 'Optimize Weights (SPY, 1y)'}
          </button>
          {!activeModels.length && (
            <span className="text-xs text-[#555]">No active models — train first</span>
          )}
          {optimizeMutation.isError && (
            <span className="text-xs text-[#ff1744]">
              {(optimizeMutation.error as Error)?.message ?? 'Optimization failed'}
            </span>
          )}
        </div>
        {optimizeResult && (
          <div className="mt-3 border border-[#1e1e1e] rounded p-3 text-xs font-mono">
            <p className="text-[#888888] mb-2">
              Optimal weights for <span className="text-[#f5a623]">{String((optimizeResult as Record<string, unknown>).symbol)}</span>
              {' · '}
              {String((optimizeResult as Record<string, unknown>).n_splits)} folds
            </p>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries((optimizeResult.optimal_weights as Record<string, unknown>) ?? {}).map(([model, w]) => (
                <div key={model} className="bg-[#0a0a0a] rounded p-2">
                  <p className="text-[#888888]">{model}</p>
                  <p className="text-[#00c853] font-bold">{(num(w) * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Training Guide */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3 text-[#888888] uppercase tracking-wider">Free GPU Training</h2>
        <div className="grid grid-cols-3 gap-4 text-xs">
          {[
            { platform: 'Kaggle', quota: '30 GPU hrs/week · T4/P100', notebook: 'notebooks/train_lstm.ipynb', color: '#20BEFF' },
            { platform: 'Google Colab', quota: 'Free T4 · limited runtime', notebook: 'notebooks/train_xgboost.ipynb', color: '#F9AB00' },
            { platform: 'Lightning.AI', quota: '22 hrs/month · A10G', notebook: 'notebooks/train_transformer.ipynb', color: '#792EE5' },
          ].map(({ platform, quota, notebook, color }) => (
            <div key={platform} className="border border-[#1e1e1e] rounded p-3">
              <p className="font-bold mb-1" style={{ color }}>{platform}</p>
              <p className="text-[#888888] mb-2">{quota}</p>
              <code className="text-[#f5a623] text-[10px]">{notebook}</code>
            </div>
          ))}
        </div>
        <p className="text-xs text-[#555] mt-3">
          After training: save <code className="text-[#888]">model.pt + scaler.pkl</code> → upload to{' '}
          <code className="text-[#888]">backend/models_artifacts/</code> → activate via the registry above.
        </p>
      </div>
    </div>
  )
}
