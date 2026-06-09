/**
 * HeatmapCalendar — 12-month return heatmap grid.
 * Green (#00c853) for positive months, red (#ff1744) for negative.
 * Intensity proportional to absolute return (opacity 0.3–1.0).
 *
 * Props: { data: Array<{month: string, return_pct: number}> }
 * month format: "YYYY-MM" or "Jan", "Feb", etc.
 */

interface MonthData {
  month: string
  return_pct: number
}

interface HeatmapCalendarProps {
  data: MonthData[]
}

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function parseMonthIndex(month: string): number {
  // Try "YYYY-MM" format
  const isoMatch = month.match(/^\d{4}-(\d{2})$/)
  if (isoMatch) return parseInt(isoMatch[1], 10) - 1

  // Try month name
  const nameIdx = MONTH_NAMES.findIndex(m => month.toLowerCase().startsWith(m.toLowerCase()))
  if (nameIdx !== -1) return nameIdx

  // Try month number as string
  const num = parseInt(month, 10)
  if (!isNaN(num) && num >= 1 && num <= 12) return num - 1

  return -1
}

function getColor(returnPct: number, maxAbs: number): string {
  if (maxAbs === 0) return 'rgba(100,100,100,0.3)'
  const intensity = Math.min(1.0, Math.max(0.3, 0.3 + (Math.abs(returnPct) / maxAbs) * 0.7))
  if (returnPct >= 0) {
    return `rgba(0,200,83,${intensity.toFixed(2)})`
  }
  return `rgba(255,23,68,${intensity.toFixed(2)})`
}

export default function HeatmapCalendar({ data }: HeatmapCalendarProps) {
  // Build a map from month index → data
  const monthMap = new Map<number, MonthData>()
  for (const entry of data) {
    const idx = parseMonthIndex(entry.month)
    if (idx >= 0 && idx < 12) {
      monthMap.set(idx, entry)
    }
  }

  const maxAbs = data.length > 0
    ? Math.max(...data.map(d => Math.abs(d.return_pct)))
    : 0

  return (
    <div
      className="rounded-lg p-4"
      style={{ background: '#1e2433', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}
    >
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-medium text-white uppercase tracking-widest">Monthly Returns</span>
        {data.length > 0 && (
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 text-[11px] text-[#00c853]">
              <span className="w-2 h-2 rounded-sm inline-block bg-[#00c853]" /> Positive
            </span>
            <span className="flex items-center gap-1 text-[11px] text-[#ff1744]">
              <span className="w-2 h-2 rounded-sm inline-block bg-[#ff1744]" /> Negative
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-6 gap-2 sm:grid-cols-12">
        {MONTH_NAMES.map((name, idx) => {
          const entry = monthMap.get(idx)
          const bg = entry ? getColor(entry.return_pct, maxAbs) : 'rgba(255,255,255,0.05)'
          const textColor = entry
            ? (entry.return_pct >= 0 ? '#00c853' : '#ff1744')
            : '#555555'

          return (
            <div
              key={name}
              className="rounded flex flex-col items-center justify-center py-3 cursor-default transition-opacity hover:opacity-80"
              style={{ background: bg }}
              title={entry ? `${name}: ${entry.return_pct >= 0 ? '+' : ''}${entry.return_pct.toFixed(2)}%` : `${name}: no data`}
            >
              <span className="text-[10px] text-[#aaa] mb-1">{name}</span>
              {entry ? (
                <span className="text-xs font-semibold" style={{ color: textColor }}>
                  {entry.return_pct >= 0 ? '+' : ''}{entry.return_pct.toFixed(1)}%
                </span>
              ) : (
                <span className="text-[10px] text-[#444]">—</span>
              )}
            </div>
          )
        })}
      </div>

      {data.length === 0 && (
        <p className="text-center text-sm text-[#555] mt-4">No monthly return data available</p>
      )}
    </div>
  )
}
