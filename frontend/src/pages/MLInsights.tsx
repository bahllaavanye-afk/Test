import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

interface MLModel {
  id: string
  model_type: string
  symbol: string | null
  interval: string | null
  val_accuracy: number | null
  val_sharpe: number | null
  is_active: boolean
  created_at: string
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
  const { data: models = [], isLoading: modelsLoading } = useQuery<MLModel[]>({
    queryKey: ['ml-models'],
    queryFn: () => api.get('/ml/models').then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: exps = [], isLoading: expsLoading } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: () => api.get('/experiments/').then(r => r.data),
    refetchInterval: 10_000,
  })

  const activeModels = models.filter(m => m.is_active)
  const completedExps = exps.filter(e => e.status === 'done')
  const bestSharpe = completedExps.length > 0
    ? Math.max(...completedExps.map(e => e.test_sharpe ?? 0))
    : 0
  const avgAccuracy = completedExps.length > 0
    ? completedExps.reduce((sum, e) => sum + (e.val_accuracy ?? 0), 0) / completedExps.length
    : 0

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

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Registered Models" value={models.length} color="#2196F3" sub={`${activeModels.length} active`} />
        <StatCard label="Completed Experiments" value={completedExps.length} color="#00c853" sub={`of ${exps.length} total`} />
        <StatCard label="Best Test Sharpe" value={bestSharpe.toFixed(3)} color="#f5a623" sub="highest across all runs" />
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
                        ? <span style={{ color: m.val_accuracy > 0.55 ? '#00c853' : '#888' }}>{(m.val_accuracy * 100).toFixed(1)}%</span>
                        : <span className="text-[#555]">—</span>}
                    </td>
                    <td className="px-4 py-3 text-xs font-bold">
                      {m.val_sharpe != null
                        ? <span style={{ color: m.val_sharpe > 1.5 ? '#00c853' : m.val_sharpe > 0.8 ? '#f5a623' : '#888' }}>
                            {m.val_sharpe.toFixed(3)}
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
                      {new Date(m.created_at).toLocaleDateString()}
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
                  .sort((a, b) => (b.test_sharpe ?? 0) - (a.test_sharpe ?? 0))
                  .slice(0, 20)
                  .map(e => (
                    <tr key={e.id} className="border-t border-[#1e1e1e] hover:bg-[#1a1a1a] transition-colors">
                      <td className="px-4 py-3 text-xs font-mono text-[#e8e8e8]">{e.name}</td>
                      <td className="px-4 py-3 text-xs text-[#888888]">
                        {(e.config as any)?.model ?? '—'}
                      </td>
                      <td className="px-4 py-3 text-xs">
                        {e.val_accuracy != null ? `${(e.val_accuracy * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-4 py-3 text-xs">
                        {e.val_sharpe?.toFixed(3) ?? '—'}
                      </td>
                      <td className="px-4 py-3 text-xs font-bold">
                        {e.test_sharpe != null
                          ? <span style={{ color: e.test_sharpe > 1.5 ? '#00c853' : e.test_sharpe > 0.8 ? '#f5a623' : '#888' }}>
                              {e.test_sharpe.toFixed(3)}
                            </span>
                          : <span className="text-[#555]">—</span>}
                      </td>
                      <td className="px-4 py-3 text-xs text-[#555]">
                        {e.completed_at ? new Date(e.completed_at).toLocaleDateString() : '—'}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
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
