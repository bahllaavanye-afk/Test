/**
 * MarketHeatmap — S&P 500 sector heatmap, TradingView-style.
 * Each box is a sector ETF; color = % change (green/red gradient);
 * size = market cap weight.
 */
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'

interface SectorData {
  sector: string
  etf: string
  change_pct: number
  weight: number      // 0-1 relative size
  top_mover: string
  top_mover_pct: number
}

const FALLBACK_SECTORS: SectorData[] = [
  { sector: 'Technology',    etf: 'XLK', change_pct: 0, weight: 0.29, top_mover: 'NVDA', top_mover_pct: 0 },
  { sector: 'Healthcare',    etf: 'XLV', change_pct: 0, weight: 0.13, top_mover: 'LLY',  top_mover_pct: 0 },
  { sector: 'Financials',    etf: 'XLF', change_pct: 0, weight: 0.13, top_mover: 'JPM',  top_mover_pct: 0 },
  { sector: 'Consumer Disc', etf: 'XLY', change_pct: 0, weight: 0.10, top_mover: 'AMZN', top_mover_pct: 0 },
  { sector: 'Comm. Services',etf: 'XLC', change_pct: 0, weight: 0.09, top_mover: 'META', top_mover_pct: 0 },
  { sector: 'Industrials',   etf: 'XLI', change_pct: 0, weight: 0.08, top_mover: 'RTX',  top_mover_pct: 0 },
  { sector: 'Consumer Stapl',etf: 'XLP', change_pct: 0, weight: 0.06, top_mover: 'KO',   top_mover_pct: 0 },
  { sector: 'Energy',        etf: 'XLE', change_pct: 0, weight: 0.04, top_mover: 'XOM',  top_mover_pct: 0 },
  { sector: 'Utilities',     etf: 'XLU', change_pct: 0, weight: 0.03, top_mover: 'NEE',  top_mover_pct: 0 },
  { sector: 'Real Estate',   etf: 'XLRE', change_pct: 0, weight: 0.02, top_mover: 'AMT', top_mover_pct: 0 },
  { sector: 'Materials',     etf: 'XLB', change_pct: 0, weight: 0.02, top_mover: 'LIN',  top_mover_pct: 0 },
]

function changeColor(pct: number): string {
  const intensity = Math.min(Math.abs(pct) / 3, 1)  // saturate at ±3%
  if (pct > 0) {
    const g = Math.round(100 + intensity * 155)
    return `rgba(0, ${g}, 50, ${0.15 + intensity * 0.50})`
  } else {
    const r = Math.round(100 + intensity * 155)
    return `rgba(${r}, 20, 20, ${0.15 + intensity * 0.50})`
  }
}

function textColor(pct: number): string {
  return pct >= 0 ? '#00ff88' : '#ff4d4d'
}

interface HeatmapBoxProps {
  d: SectorData
  onClick?: () => void
}

function HeatmapBox({ d, onClick }: HeatmapBoxProps) {
  return (
    <button
      onClick={onClick}
      className="relative flex flex-col items-center justify-center p-2 rounded-md border border-[#1e1e2e] transition-transform hover:scale-[1.02] hover:z-10 cursor-pointer text-center"
      style={{ backgroundColor: changeColor(d.change_pct) }}
    >
      <span className="text-[11px] font-semibold text-white truncate w-full">{d.etf}</span>
      <span className="text-[10px] text-[#8a8a9a] truncate w-full leading-tight">{d.sector}</span>
      <span className={`text-[13px] font-bold font-mono mt-1 ${textColor(d.change_pct)}`}>
        {d.change_pct >= 0 ? '+' : ''}{d.change_pct.toFixed(2)}%
      </span>
      <span className="text-[9px] text-[#6a6a8a] mt-0.5">
        {d.top_mover} {d.top_mover_pct >= 0 ? '+' : ''}{d.top_mover_pct.toFixed(1)}%
      </span>
    </button>
  )
}

interface MarketHeatmapProps {
  className?: string
  onSelectSector?: (etf: string) => void
}

export function MarketHeatmap({ className = '', onSelectSector }: MarketHeatmapProps) {
  const { data: sectors = FALLBACK_SECTORS } = useQuery<SectorData[]>({
    queryKey: ['sector-heatmap'],
    queryFn: async () => {
      const { data } = await api.get('/market-data/sector-heatmap')
      return Array.isArray(data) ? data : FALLBACK_SECTORS
    },
    refetchInterval: 60_000,
    staleTime: 55_000,
    placeholderData: FALLBACK_SECTORS,
  })

  // Sort by weight descending so bigger sectors take more visual space
  const sorted = [...sectors].sort((a, b) => b.weight - a.weight)
  const avgChange = sectors.reduce((s, d) => s + d.change_pct * d.weight, 0)
  const marketUp = avgChange >= 0

  return (
    <div className={`bg-[#0d0d14] border border-[#1e1e2e] rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1e1e2e]">
        <span className="text-[11px] font-semibold text-[#8a8a9a] uppercase tracking-wider">
          S&P 500 Sectors
        </span>
        <span className={`text-[12px] font-mono font-bold ${marketUp ? 'text-[#00ff88]' : 'text-[#ff4d4d]'}`}>
          Market avg {marketUp ? '+' : ''}{avgChange.toFixed(2)}%
        </span>
      </div>

      {/* Grid — proportional sizing via flex-wrap */}
      <div className="p-2 grid gap-1.5" style={{
        gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
      }}>
        {sorted.map(d => (
          <HeatmapBox
            key={d.etf}
            d={d}
            onClick={() => onSelectSector?.(d.etf)}
          />
        ))}
      </div>
    </div>
  )
}
