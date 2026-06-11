/**
 * FeatureImportance — horizontal bar chart of top-10 ML feature importances.
 * Uses inline div widths — no recharts needed.
 *
 * Props: { modelName?: string }
 * Calls GET /ml/models/{modelName}/feature-importance or /ml/feature-importance
 */
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

interface FeatureItem {
  feature: string
  importance: number
}

interface FeatureImportanceProps {
  modelName?: string
}

export default function FeatureImportance({ modelName }: FeatureImportanceProps) {
  const endpoint = modelName
    ? `/ml/models/${encodeURIComponent(modelName)}/feature-importance`
    : '/ml/feature-importance'

  const { data, isLoading, isError } = useQuery<FeatureItem[]>({
    queryKey: ['feature-importance', modelName],
    queryFn: async () => {
      const res = await api.get(endpoint)
      return res.data
    },
    retry: false,
    staleTime: 60_000,
  })

  const containerStyle = {
    background: '#131722',
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
  }

  if (isLoading) {
    return (
      <div style={containerStyle} className="p-4 rounded-lg">
        <div className="flex items-center gap-2 text-[#888]">
          <div className="w-4 h-4 border-2 border-[#2979ff] border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Loading feature importance…</span>
        </div>
      </div>
    )
  }

  if (isError || !data || data.length === 0) {
    return (
      <div style={containerStyle} className="p-4 rounded-lg">
        <div
          className="rounded-lg p-8 flex flex-col items-center gap-3 text-center"
          style={{ background: '#1e2433' }}
        >
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
            <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
            <line x1="12" y1="22.08" x2="12" y2="12" />
          </svg>
          <p className="text-white font-medium">Train a model first to see feature importance</p>
          {modelName && (
            <p className="text-sm text-[#555]">
              Model: <span className="text-[#2979ff]">{modelName}</span>
            </p>
          )}
        </div>
      </div>
    )
  }

  // Sort descending, take top 10
  const sorted = [...data]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10)

  const maxImportance = sorted[0]?.importance ?? 1

  return (
    <div style={containerStyle} className="p-4 rounded-lg">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-xs font-medium text-white uppercase tracking-widest">
          Feature Importance
        </span>
        {modelName && (
          <span
            className="text-[11px] px-2 py-0.5 rounded"
            style={{ background: '#2979ff22', color: '#2979ff', border: '1px solid #2979ff44' }}
          >
            {modelName}
          </span>
        )}
      </div>

      <div className="space-y-2">
        {sorted.map((item, idx) => {
          const pct = maxImportance > 0 ? (item.importance / maxImportance) * 100 : 0
          return (
            <div key={item.feature} className="space-y-0.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className="text-[10px] font-mono"
                    style={{ color: '#555', minWidth: '1rem' }}
                  >
                    {idx + 1}
                  </span>
                  <span className="text-xs text-[#ccc] font-mono">{item.feature}</span>
                </div>
                <span className="text-xs font-mono text-[#2979ff]">
                  {(item.importance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-5 rounded overflow-hidden" style={{ background: '#0a0a0a' }}>
                <div
                  className="h-full rounded transition-all duration-500"
                  style={{
                    width: `${pct}%`,
                    background: 'linear-gradient(90deg, #1a47cc, #2979ff)',
                    boxShadow: pct > 50 ? '0 0 8px rgba(41,121,255,0.4)' : 'none',
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
