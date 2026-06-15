/**
 * WatchlistPanel — TradingView-style live watchlist with mini sparklines.
 * Shows symbol, last price, change%, mini trend line, volume rank.
 * Updates every 5 seconds via the /api/v1/market-data/quotes endpoint.
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'

interface Quote {
  symbol: string
  last: number
  change_pct: number
  volume: number
  volume_rank: number          // 0-1 relative to 30-day avg
  history: number[]            // last 20 close prices for sparkline
  bid?: number
  ask?: number
}

const DEFAULT_SYMBOLS = [
  'SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META',
  'BTC-USD', 'ETH-USD', 'SOL-USD',
]

// Tiny sparkline drawn on a <canvas>
function Sparkline({ prices, positive }: { prices: number[]; positive: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas || prices.length < 2) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const w = canvas.width, h = canvas.height
    ctx.clearRect(0, 0, w, h)
    const min = Math.min(...prices), max = Math.max(...prices)
    const range = max - min || 1
    const step = w / (prices.length - 1)
    ctx.beginPath()
    prices.forEach((p, i) => {
      const x = i * step
      const y = h - ((p - min) / range) * (h - 2) - 1
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.strokeStyle = positive ? '#00ff88' : '#ff4d4d'
    ctx.lineWidth = 1.5
    ctx.stroke()
  }, [prices, positive])
  return <canvas ref={ref} width={64} height={28} className="opacity-80" />
}

// Volume bar visual
function VolumeBar({ rank }: { rank: number }) {
  return (
    <div className="flex items-center gap-1">
      {[0.2, 0.4, 0.6, 0.8, 1.0].map((threshold, i) => (
        <div
          key={i}
          className={`w-1 rounded-sm transition-all ${
            rank >= threshold ? 'bg-[#00bfff]' : 'bg-[#2a2a3a]'
          }`}
          style={{ height: `${(i + 1) * 3 + 4}px` }}
        />
      ))}
    </div>
  )
}

async function fetchQuotes(symbols: string[]): Promise<Quote[]> {
  try {
    const { data } = await api.get('/market-data/quotes', {
      params: { symbols: symbols.join(',') },
    })
    return Array.isArray(data) ? data : []
  } catch {
    return []
  }
}

interface WatchlistPanelProps {
  symbols?: string[]
  className?: string
  onSelectSymbol?: (symbol: string) => void
}

export function WatchlistPanel({
  symbols = DEFAULT_SYMBOLS,
  className = '',
  onSelectSymbol,
}: WatchlistPanelProps) {
  const [selected, setSelected] = useState<string>(symbols[0])
  const [filter, setFilter] = useState('')
  const [sortBy, setSortBy] = useState<'symbol' | 'change' | 'volume'>('change')

  const { data: quotes = [] } = useQuery({
    queryKey: ['watchlist-quotes', symbols],
    queryFn: () => fetchQuotes(symbols),
    refetchInterval: 5000,
    staleTime: 4000,
  })

  const sorted = [...quotes]
    .filter(q => q.symbol.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'change') return Math.abs(b.change_pct) - Math.abs(a.change_pct)
      if (sortBy === 'volume') return b.volume_rank - a.volume_rank
      return a.symbol.localeCompare(b.symbol)
    })

  const handleSelect = useCallback((sym: string) => {
    setSelected(sym)
    onSelectSymbol?.(sym)
  }, [onSelectSymbol])

  return (
    <div className={`flex flex-col h-full bg-[#0d0d14] border border-[#1e1e2e] rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1e1e2e]">
        <span className="text-[11px] font-semibold text-[#8a8a9a] uppercase tracking-wider">Watchlist</span>
        <div className="flex gap-1">
          {(['symbol', 'change', 'volume'] as const).map(k => (
            <button
              key={k}
              onClick={() => setSortBy(k)}
              className={`text-[10px] px-2 py-0.5 rounded transition-colors ${
                sortBy === k
                  ? 'bg-[#1a1a2e] text-[#00ff88]'
                  : 'text-[#5a5a7a] hover:text-[#8a8a9a]'
              }`}
            >
              {k === 'symbol' ? 'A-Z' : k === 'change' ? '% Chg' : 'Vol'}
            </button>
          ))}
        </div>
      </div>

      {/* Search */}
      <div className="px-2 py-1.5 border-b border-[#1e1e2e]">
        <input
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="Search..."
          className="w-full bg-[#111120] text-[12px] text-white placeholder-[#4a4a5a] px-2 py-1 rounded border border-[#2a2a3a] focus:border-[#00ff88] focus:outline-none"
        />
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {sorted.length === 0 && (
          <div className="flex items-center justify-center h-20 text-[11px] text-[#4a4a5a]">
            Loading quotes…
          </div>
        )}
        {sorted.map(q => {
          const pos = q.change_pct >= 0
          const isSelected = selected === q.symbol
          return (
            <button
              key={q.symbol}
              onClick={() => handleSelect(q.symbol)}
              className={`w-full flex items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-[#111120] ${
                isSelected ? 'bg-[#111120] border-l-2 border-[#00ff88]' : 'border-l-2 border-transparent'
              }`}
            >
              {/* Symbol + last */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-[12px] font-medium text-white truncate">{q.symbol}</span>
                  <span className="text-[12px] font-mono text-white">
                    {q.last?.toFixed(q.last < 10 ? 4 : 2)}
                  </span>
                </div>
                <div className="flex items-center justify-between mt-0.5">
                  <VolumeBar rank={q.volume_rank ?? 0} />
                  <span className={`text-[11px] font-mono ${pos ? 'text-[#00ff88]' : 'text-[#ff4d4d]'}`}>
                    {pos ? '+' : ''}{q.change_pct?.toFixed(2)}%
                  </span>
                </div>
              </div>
              {/* Sparkline */}
              <Sparkline prices={q.history ?? []} positive={pos} />
            </button>
          )
        })}
      </div>
    </div>
  )
}
