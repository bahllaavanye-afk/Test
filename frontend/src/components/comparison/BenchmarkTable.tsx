/**
 * BenchmarkTable — benchmark comparison table.
 * Bloomberg dark theme: bg #131722, card #1e2433
 *
 * Props: { benchmarks: Array<{name, sharpe, annual_return_pct, max_drawdown_pct, color}> }
 */

interface Benchmark {
  name: string
  sharpe: number | null
  annual_return_pct: number | null
  max_drawdown_pct: number | null
  color?: string
}

interface BenchmarkTableProps {
  benchmarks: Benchmark[]
}

function sharpeColor(v: number | null): string {
  if (v == null) return '#888888'
  if (v >= 1.5) return '#00c853'
  if (v >= 0.5) return '#f5a623'
  return '#ff1744'
}

function fmt(v: number | null, suffix = ''): string {
  if (v == null) return '—'
  return `${v >= 0 ? '' : ''}${v.toFixed(2)}${suffix}`
}

const TARGET_ROW: Benchmark = {
  name: 'QuantEdge Target',
  sharpe: 2.0,
  annual_return_pct: 25.0,
  max_drawdown_pct: -15.0,
  color: '#f5a623',
}

export default function BenchmarkTable({ benchmarks }: BenchmarkTableProps) {
  const rows = [TARGET_ROW, ...benchmarks]

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ background: '#1e2433', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}
    >
      {/* Table header */}
      <div
        className="grid grid-cols-4 px-4 py-2 border-b border-[#ffffff0d]"
        style={{ background: '#131722' }}
      >
        {['Name', 'Annual Return', 'Sharpe', 'Max Drawdown'].map((col) => (
          <span key={col} className="text-xs text-[#555] uppercase tracking-widest">
            {col}
          </span>
        ))}
      </div>

      {/* Rows */}
      {rows.map((row, idx) => {
        const isTarget = idx === 0
        const rowBg = isTarget ? 'rgba(245,166,35,0.04)' : idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)'

        return (
          <div
            key={row.name}
            className="grid grid-cols-4 px-4 py-3 border-b border-[#ffffff0d] items-center"
            style={{ background: rowBg }}
          >
            {/* Name */}
            <div className="flex items-center gap-2">
              {row.color && (
                <span
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: row.color }}
                />
              )}
              <span
                className="text-sm font-medium"
                style={{ color: isTarget ? '#f5a623' : 'white' }}
              >
                {isTarget ? (
                  <>
                    {row.name}
                    <span
                      className="ml-2 text-[10px] px-1.5 py-0.5 rounded"
                      style={{ background: '#f5a62322', color: '#f5a623', border: '1px solid #f5a62344' }}
                    >
                      TARGET
                    </span>
                  </>
                ) : row.name}
              </span>
            </div>

            {/* Annual Return */}
            <span
              className="text-sm font-mono"
              style={{ color: (row.annual_return_pct ?? 0) >= 0 ? '#00c853' : '#ff1744' }}
            >
              {isTarget ? `>${fmt(row.annual_return_pct, '%')}` : fmt(row.annual_return_pct, '%')}
            </span>

            {/* Sharpe */}
            <span
              className="text-sm font-mono font-semibold"
              style={{ color: sharpeColor(row.sharpe) }}
            >
              {isTarget ? `>${fmt(row.sharpe)}` : fmt(row.sharpe)}
            </span>

            {/* Max Drawdown */}
            <span
              className="text-sm font-mono"
              style={{ color: '#ff1744' }}
            >
              {isTarget ? `>${fmt(row.max_drawdown_pct, '%')}` : fmt(row.max_drawdown_pct, '%')}
            </span>
          </div>
        )
      })}

      {rows.length === 1 && (
        <div className="px-4 py-6 text-center text-sm text-[#555]">
          No benchmark data provided
        </div>
      )}

      {/* Legend */}
      <div className="px-4 py-3 flex items-center gap-4" style={{ borderTop: '1px solid #ffffff0d' }}>
        <span className="text-[10px] text-[#555] uppercase tracking-widest">Sharpe legend:</span>
        {[
          { label: '≥ 1.5', color: '#00c853' },
          { label: '≥ 0.5', color: '#f5a623' },
          { label: '< 0.5', color: '#ff1744' },
        ].map(({ label, color }) => (
          <span key={label} className="flex items-center gap-1 text-[11px]" style={{ color }}>
            <span className="w-2 h-2 rounded-full inline-block" style={{ background: color }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  )
}
