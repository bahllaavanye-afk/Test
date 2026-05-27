import { useEffect, useRef, useState } from 'react'
import { useWebSocket } from '../../hooks/useWebSocket'

interface SymbolPrice {
  symbol: string
  last: number | null
  bid: number | null
  ask: number | null
  changePct: number | null
  flash: 'up' | 'down' | null
}

function initPrices(symbols: string[]): Record<string, SymbolPrice> {
  return Object.fromEntries(
    symbols.map((s) => [
      s,
      { symbol: s, last: null, bid: null, ask: null, changePct: null, flash: null },
    ])
  )
}

interface LivePriceTickerProps {
  symbols: string[]
}

export function LivePriceTicker({ symbols }: LivePriceTickerProps) {
  const [prices, setPrices] = useState<Record<string, SymbolPrice>>(() => initPrices(symbols))
  const flashTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  const { lastMessage, connected } = useWebSocket('/ws/prices', true)

  // Update prices when a WS message arrives
  useEffect(() => {
    if (!lastMessage) return
    if (lastMessage.type !== 'price_update') return

    const { symbol, last, bid, ask, change_pct } = lastMessage as {
      type: string
      symbol: string
      last: number
      bid: number
      ask: number
      change_pct: number
    }

    if (!symbol) return

    setPrices((prev) => {
      const existing = prev[symbol]
      const prevLast = existing?.last ?? null
      let flash: 'up' | 'down' | null = null
      if (prevLast !== null && last !== null) {
        if (last > prevLast) flash = 'up'
        else if (last < prevLast) flash = 'down'
      }

      return {
        ...prev,
        [symbol]: { symbol, last, bid, ask, changePct: change_pct ?? null, flash },
      }
    })

    // Clear flash after 600 ms
    if (flashTimers.current[symbol]) clearTimeout(flashTimers.current[symbol])
    flashTimers.current[symbol] = setTimeout(() => {
      setPrices((prev) =>
        prev[symbol] ? { ...prev, [symbol]: { ...prev[symbol], flash: null } } : prev
      )
    }, 600)
  }, [lastMessage])

  // Reset when symbol list changes
  useEffect(() => {
    setPrices(initPrices(symbols))
  }, [symbols])

  const formatPrice = (v: number | null) =>
    v == null ? '—' : v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

  const formatChange = (v: number | null) => {
    if (v == null) return null
    const sign = v >= 0 ? '+' : ''
    return `${sign}${v.toFixed(2)}%`
  }

  return (
    <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-[#1e1e1e]">
        <span
          className={`inline-block w-1.5 h-1.5 rounded-full ${connected ? 'bg-[#00c853]' : 'bg-[#ff1744]'}`}
        />
        <span className="text-[10px] text-[#555555] uppercase tracking-widest font-mono">
          Live Prices
        </span>
        {!connected && (
          <span className="text-[10px] text-[#ff1744] font-mono ml-auto">Reconnecting…</span>
        )}
      </div>

      {/* Scrollable row of tickers */}
      <div className="flex items-center gap-0 overflow-x-auto scrollbar-hide py-2 px-2">
        {symbols.map((sym) => {
          const p = prices[sym]
          const changeStr = formatChange(p?.changePct ?? null)
          const isUp = (p?.changePct ?? 0) >= 0
          const changeColor = isUp ? '#00c853' : '#ff1744'
          const arrowChar = isUp ? '▲' : '▼'

          // Flash background classes
          let flashBg = 'bg-transparent'
          if (p?.flash === 'up') flashBg = 'bg-[#00c853]/20'
          else if (p?.flash === 'down') flashBg = 'bg-[#ff1744]/20'

          return (
            <div
              key={sym}
              className={`flex-shrink-0 flex items-center gap-2 px-3 py-1 rounded-md mx-0.5 transition-colors duration-300 ${flashBg}`}
            >
              {/* Symbol */}
              <span className="text-xs font-bold text-[#f5a623] font-mono">{sym}</span>

              {/* Last price */}
              <span
                className={`text-sm font-bold font-mono transition-colors duration-300 ${
                  p?.flash === 'up'
                    ? 'text-[#00c853]'
                    : p?.flash === 'down'
                    ? 'text-[#ff1744]'
                    : 'text-white'
                }`}
              >
                ${formatPrice(p?.last ?? null)}
              </span>

              {/* Bid / Ask spread */}
              {p?.bid != null && p?.ask != null && (
                <span className="text-[10px] text-[#555555] font-mono hidden sm:inline">
                  {formatPrice(p.bid)} / {formatPrice(p.ask)}
                </span>
              )}

              {/* Change % */}
              {changeStr && (
                <span
                  className="text-xs font-mono font-semibold flex items-center gap-0.5"
                  style={{ color: changeColor }}
                >
                  <span className="text-[9px]">{arrowChar}</span>
                  {changeStr}
                </span>
              )}

              {/* Separator */}
              <span className="text-[#1e1e1e] select-none ml-1">|</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default LivePriceTicker
