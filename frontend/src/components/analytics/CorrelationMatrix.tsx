import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

// ─── Types ───────────────────────────────────────────────────────────────────

interface CorrelationData {
  symbols: string[]
  matrix: number[][]
  computed_at: string
  error?: string
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function correlationToColor(value: number, isDiagonal: boolean): string {
  if (isDiagonal) return 'rgba(80, 80, 80, 0.5)'
  if (value >= 0) {
    // 0 → white (#ffffff), +1 → deep green (#00c853)
    const opacity = Math.abs(value)
    return `rgba(0, 200, 83, ${opacity.toFixed(3)})`
  } else {
    // 0 → white (#ffffff), -1 → deep red (#ff1744)
    const opacity = Math.abs(value)
    return `rgba(255, 23, 68, ${opacity.toFixed(3)})`
  }
}

function correlationLabel(value: number): string {
  const abs = Math.abs(value)
  if (abs >= 0.9) return 'extremely high'
  if (abs >= 0.7) return 'high'
  if (abs >= 0.5) return 'moderate'
  if (abs >= 0.3) return 'low'
  return 'negligible'
}

function correlationDirection(value: number): string {
  if (value > 0.05) return 'positive'
  if (value < -0.05) return 'negative'
  return 'near-zero'
}

function correlationTooltip(sym1: string, sym2: string, value: number): string {
  const pct = Math.abs(Math.round(value * 100))
  const label = correlationLabel(value)
  const dir = correlationDirection(value)

  if (value > 0.05) {
    return `${sym1} vs ${sym2}: ${value.toFixed(2)} (${label} ${dir} correlation — these positions move together ${pct}% of the time)`
  } else if (value < -0.05) {
    return `${sym1} vs ${sym2}: ${value.toFixed(2)} (${label} ${dir} correlation — these positions tend to move in opposite directions)`
  }
  return `${sym1} vs ${sym2}: ${value.toFixed(2)} (${label} correlation — no meaningful relationship)`
}

// ─── Subcomponents ───────────────────────────────────────────────────────────

function LoadingSkeleton({ size }: { size: number }) {
  return (
    <div className="overflow-x-auto">
      <div className="inline-grid gap-1" style={{ gridTemplateColumns: `auto repeat(${size}, minmax(52px, 1fr))` }}>
        {/* Header row */}
        <div className="w-12 h-8" />
        {Array.from({ length: size }).map((_, i) => (
          <div key={i} className="h-8 rounded bg-[#1e1e1e] animate-pulse" />
        ))}
        {/* Data rows */}
        {Array.from({ length: size }).map((_, row) => (
          <>
            <div key={`lbl-${row}`} className="w-12 h-8 rounded bg-[#1e1e1e] animate-pulse my-0.5" />
            {Array.from({ length: size }).map((_, col) => (
              <div key={`cell-${row}-${col}`} className="h-8 rounded bg-[#1e1e1e] animate-pulse my-0.5" />
            ))}
          </>
        ))}
      </div>
    </div>
  )
}

function CorrelationCell({
  value,
  rowSymbol,
  colSymbol,
  isDiagonal,
}: {
  value: number
  rowSymbol: string
  colSymbol: string
  isDiagonal: boolean
}) {
  const [hovered, setHovered] = useState(false)
  const bg = correlationToColor(value, isDiagonal)
  const textColor = isDiagonal
    ? '#888888'
    : Math.abs(value) > 0.5
    ? '#ffffff'
    : '#cccccc'

  return (
    <div
      className="relative flex items-center justify-center rounded text-[11px] font-mono font-bold select-none cursor-default transition-all"
      style={{
        backgroundColor: bg,
        color: textColor,
        minWidth: 52,
        height: 36,
        border: hovered && !isDiagonal ? '1px solid rgba(255,255,255,0.3)' : '1px solid transparent',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {isDiagonal ? '1.00' : value.toFixed(2)}

      {hovered && !isDiagonal && (
        <div
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-20 pointer-events-none"
          style={{ width: 240 }}
        >
          <div className="bg-[#1e1e1e] border border-[#333333] rounded px-3 py-2 text-[11px] text-[#e8e8e8] leading-relaxed shadow-xl">
            {correlationTooltip(rowSymbol, colSymbol, value)}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function CorrelationMatrix({
  accountId,
  days = 30,
}: {
  accountId?: string
  days?: number
}) {
  const params = new URLSearchParams({ days: String(days) })
  if (accountId) params.set('account_id', accountId)

  const { data, isLoading, isError } = useQuery<CorrelationData>({
    queryKey: ['correlation', accountId, days],
    queryFn: () => api.get(`/analytics/correlation?${params.toString()}`).then(r => r.data),
    refetchInterval: 5 * 60 * 1000, // every 5 minutes
    retry: 2,
  })

  // ─── Empty state ──────────────────────────────────────────────────────────
  if (!isLoading && !isError && data && data.matrix.length === 0) {
    return (
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6">
        <h2 className="text-sm font-semibold text-white mb-1">Correlation Matrix</h2>
        <p className="text-xs text-[#888888] mb-6">Pairwise Pearson correlation of daily returns</p>
        <div className="flex flex-col items-center justify-center py-8 text-center space-y-3">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
            <rect x="3" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="3" width="7" height="7" rx="1" />
            <rect x="3" y="14" width="7" height="7" rx="1" />
            <rect x="14" y="14" width="7" height="7" rx="1" />
          </svg>
          <p className="text-sm text-[#888888]">No positions to correlate</p>
          <p className="text-xs text-[#555555]">
            Open positions to see correlation analysis — or price data is temporarily unavailable.
          </p>
        </div>
      </div>
    )
  }

  // ─── Error state ──────────────────────────────────────────────────────────
  if (isError) {
    return (
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6">
        <h2 className="text-sm font-semibold text-white mb-4">Correlation Matrix</h2>
        <p className="text-xs text-[#ff1744]">Failed to load correlation data. Check API connectivity.</p>
      </div>
    )
  }

  // ─── Loading state ────────────────────────────────────────────────────────
  if (isLoading || !data) {
    return (
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-white">Correlation Matrix</h2>
            <p className="text-xs text-[#888888] mt-0.5">Loading price data from Alpaca…</p>
          </div>
        </div>
        <LoadingSkeleton size={6} />
      </div>
    )
  }

  const { symbols, matrix, computed_at, error: dataError } = data

  // ─── API data error (credentials / no data) ───────────────────────────────
  if (dataError) {
    return (
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6">
        <h2 className="text-sm font-semibold text-white mb-1">Correlation Matrix</h2>
        <p className="text-xs text-[#888888] mb-4">Pairwise Pearson correlation of daily returns</p>
        <div className="bg-[#1a1a1a] border border-[#ff1744]/20 rounded p-3">
          <p className="text-xs text-[#ff1744]">{dataError}</p>
        </div>
      </div>
    )
  }

  // ─── Concentration risk warnings ──────────────────────────────────────────
  const CONCENTRATION_THRESHOLD = 0.85
  const warnings: { sym1: string; sym2: string; value: number }[] = []
  for (let i = 0; i < symbols.length; i++) {
    for (let j = i + 1; j < symbols.length; j++) {
      const val = matrix[i]?.[j] ?? 0
      if (val > CONCENTRATION_THRESHOLD) {
        warnings.push({ sym1: symbols[i], sym2: symbols[j], value: val })
      }
    }
  }

  const computedAtStr = computed_at
    ? new Date(computed_at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    : '—'

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <h2 className="text-sm font-semibold text-white">Correlation Matrix</h2>
          <p className="text-xs text-[#888888] mt-0.5">
            Pairwise Pearson correlation · {days}-day daily returns · refreshes every 5 min
          </p>
        </div>
        <div className="text-[10px] text-[#555555] font-mono">
          As of {computedAtStr}
        </div>
      </div>

      {/* Concentration risk warnings */}
      {warnings.length > 0 && (
        <div className="mb-4 space-y-1.5">
          {warnings.map(w => (
            <div
              key={`${w.sym1}-${w.sym2}`}
              className="flex items-center gap-2 bg-[#1a1100] border border-[#f5a623]/30 rounded px-3 py-2"
            >
              <span className="text-[#f5a623] text-sm">⚠</span>
              <p className="text-xs text-[#f5a623]">
                Concentration risk:{' '}
                <span className="font-bold">{w.sym1} + {w.sym2}</span> are{' '}
                <span className="font-bold">{Math.round(w.value * 100)}% correlated</span> — these positions move almost identically.
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-6 mb-4 text-[10px] text-[#555555]">
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(255, 23, 68, 0.9)' }} />
          <span>−1.0 (inverse)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(200, 200, 200, 0.1)', border: '1px solid #333' }} />
          <span>0.0 (uncorrelated)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(0, 200, 83, 0.9)' }} />
          <span>+1.0 (perfectly correlated)</span>
        </div>
      </div>

      {/* Matrix table */}
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1">
          <thead>
            <tr>
              {/* Empty top-left corner */}
              <th className="w-14" />
              {symbols.map(sym => (
                <th
                  key={sym}
                  className="text-[10px] font-bold text-[#888888] text-center pb-1 font-mono"
                  style={{ minWidth: 52 }}
                >
                  {sym}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {symbols.map((rowSym, rowIdx) => (
              <tr key={rowSym}>
                {/* Row label */}
                <td className="text-[10px] font-bold text-[#888888] text-right pr-2 font-mono whitespace-nowrap">
                  {rowSym}
                </td>
                {symbols.map((colSym, colIdx) => {
                  const val = matrix[rowIdx]?.[colIdx] ?? 1.0
                  const isDiag = rowIdx === colIdx
                  return (
                    <td key={colSym} className="p-0">
                      <CorrelationCell
                        value={val}
                        rowSymbol={rowSym}
                        colSymbol={colSym}
                        isDiagonal={isDiag}
                      />
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer note */}
      <p className="text-[10px] text-[#444444] mt-3">
        Default symbols shown when no open positions detected. Hover any cell for interpretation.
        Correlations &gt; 0.85 trigger concentration risk warnings above.
      </p>
    </div>
  )
}
