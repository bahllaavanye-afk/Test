import { useMemo } from 'react'

interface Candle {
  open: number
  high: number
  low: number
  close: number
  volume: number
  time: number
}

function generateCandles(seed: number, basePrice: number, count: number, volatility: number): Candle[] {
  const candles: Candle[] = []
  let price = basePrice
  let rng = seed

  function rand() {
    rng = (rng * 1664525 + 1013904223) & 0xffffffff
    return (rng >>> 0) / 0xffffffff
  }

  const now = Date.now()
  const intervalMs = 4 * 60 * 60 * 1000 // 4h candles

  for (let i = 0; i < count; i++) {
    const drift = (rand() - 0.48) * volatility
    const open = price
    const bodySize = (rand() - 0.5) * volatility * 1.5
    const close = open + bodySize + drift
    const wickTop = rand() * volatility * 0.8
    const wickBottom = rand() * volatility * 0.8
    const high = Math.max(open, close) + wickTop
    const low = Math.min(open, close) - wickBottom
    const volume = rand() * 1000 + 200

    candles.push({ open, high, low, close, volume, time: now - (count - i) * intervalMs })
    price = close
  }
  return candles
}

interface Props {
  symbol?: string
  height?: number
}

export default function MockCandlestickChart({ symbol = 'BTC/USD', height = 480 }: Props) {
  const isCrypto = symbol.includes('BTC') || symbol.includes('ETH') || symbol.includes('BNB') || symbol.includes('SOL') || symbol.includes('BINANCE')
  const basePrice = symbol.includes('BTC') ? 67450
    : symbol.includes('ETH') ? 3820
    : symbol.includes('SOL') ? 178
    : symbol.includes('SPY') || symbol.includes('AAPL') ? 189
    : symbol.includes('MSFT') ? 415
    : symbol.includes('GOOGL') ? 175
    : symbol.includes('AMZN') ? 185
    : symbol.includes('QQQ') ? 468
    : 200

  const volatility = isCrypto ? basePrice * 0.012 : basePrice * 0.007
  const seed = symbol.split('').reduce((acc, c) => acc + c.charCodeAt(0), 42)
  const candles = useMemo(() => generateCandles(seed, basePrice, 100, volatility), [seed, basePrice, volatility])

  const W = 900
  const CHART_H = height - 100
  const VOLUME_H = 60
  const PAD_LEFT = 12
  const PAD_RIGHT = 48
  const PAD_TOP = 16
  const PAD_BOTTOM = 8
  const chartW = W - PAD_LEFT - PAD_RIGHT

  const allHighs = candles.map(c => c.high)
  const allLows = candles.map(c => c.low)
  const priceMin = Math.min(...allLows)
  const priceMax = Math.max(...allHighs)
  const priceRange = priceMax - priceMin || 1

  const maxVol = Math.max(...candles.map(c => c.volume))

  const candleW = chartW / candles.length
  const bodyW = Math.max(candleW * 0.6, 2)

  function px(price: number) {
    return PAD_TOP + (1 - (price - priceMin) / priceRange) * (CHART_H - PAD_TOP - PAD_BOTTOM)
  }

  function cx(i: number) {
    return PAD_LEFT + (i + 0.5) * candleW
  }

  // Y axis price levels
  const priceLevels = Array.from({ length: 6 }, (_, i) => priceMin + (priceRange * i) / 5)
  // X axis time labels
  const timeLabels = [0, 20, 40, 60, 80, 99].map(i => ({
    i,
    label: new Date(candles[i]?.time ?? 0).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  }))

  const lastCandle = candles[candles.length - 1]
  const firstCandle = candles[0]
  const totalPnlPct = ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100
  const isUp = lastCandle.close >= lastCandle.open
  const lastColor = isUp ? '#00c853' : '#ff1744'

  // Compute a simple moving average (20 period)
  const sma20: number[] = []
  for (let i = 0; i < candles.length; i++) {
    if (i < 19) { sma20.push(NaN); continue }
    const avg = candles.slice(i - 19, i + 1).reduce((a, c) => a + c.close, 0) / 20
    sma20.push(avg)
  }

  const smaPath = sma20
    .map((v, i) => (isNaN(v) ? null : `${i === 19 ? 'M' : 'L'} ${cx(i).toFixed(1)} ${px(v).toFixed(1)}`))
    .filter(Boolean)
    .join(' ')

  // Format price nicely
  function fmt(p: number) {
    if (p >= 1000) return p.toLocaleString('en-US', { maximumFractionDigits: 0 })
    return p.toFixed(2)
  }

  const displaySymbol = symbol.replace('BINANCE:', '').replace('NASDAQ:', '').replace('NYSE:', '')

  return (
    <div
      className="relative bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg overflow-hidden select-none"
      style={{ height }}
    >
      {/* Header bar */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-[#1e1e1e]">
        <span className="text-sm font-bold text-[#e8e8e8] font-mono">{displaySymbol}</span>
        <span className="text-xs text-[#888888]">4H</span>
        <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: '#1e1e1e', color: '#888888' }}>MOCK DATA</span>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-[#888888]">O <span className="text-[#e8e8e8] font-mono">{fmt(lastCandle.open)}</span></span>
          <span className="text-xs text-[#888888]">H <span className="text-[#00c853] font-mono">{fmt(lastCandle.high)}</span></span>
          <span className="text-xs text-[#888888]">L <span className="text-[#ff1744] font-mono">{fmt(lastCandle.low)}</span></span>
          <span className="text-xs text-[#888888]">C <span style={{ color: lastColor }} className="font-mono font-bold">{fmt(lastCandle.close)}</span></span>
          <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ color: totalPnlPct >= 0 ? '#00c853' : '#ff1744', background: totalPnlPct >= 0 ? '#00c85318' : '#ff174418' }}>
            {totalPnlPct >= 0 ? '+' : ''}{totalPnlPct.toFixed(2)}%
          </span>
        </div>
      </div>

      {/* SVG chart */}
      <svg
        viewBox={`0 0 ${W} ${CHART_H + VOLUME_H + 24}`}
        width="100%"
        height={height - 44}
        preserveAspectRatio="none"
        style={{ display: 'block' }}
      >
        {/* Background grid */}
        {priceLevels.map((p, idx) => (
          <line
            key={idx}
            x1={PAD_LEFT} y1={px(p).toFixed(1)}
            x2={W - PAD_RIGHT} y2={px(p).toFixed(1)}
            stroke="#1e1e1e" strokeWidth="0.5"
          />
        ))}

        {/* Price labels on right axis */}
        {priceLevels.map((p, idx) => (
          <text
            key={idx}
            x={W - PAD_RIGHT + 4}
            y={px(p) + 4}
            fill="#555555"
            fontSize="9"
            fontFamily="monospace"
          >
            {fmt(p)}
          </text>
        ))}

        {/* SMA 20 */}
        {smaPath && (
          <path d={smaPath} fill="none" stroke="#f5a623" strokeWidth="1" strokeOpacity="0.6" />
        )}

        {/* Candles */}
        {candles.map((c, i) => {
          const x = cx(i)
          const color = c.close >= c.open ? '#00c853' : '#ff1744'
          const bodyTop = px(Math.max(c.open, c.close))
          const bodyBot = px(Math.min(c.open, c.close))
          const bodyHeight = Math.max(bodyBot - bodyTop, 1)
          const halfBody = bodyW / 2

          return (
            <g key={i}>
              {/* Wick */}
              <line
                x1={x} y1={px(c.high).toFixed(1)}
                x2={x} y2={px(c.low).toFixed(1)}
                stroke={color} strokeWidth="0.8" opacity="0.9"
              />
              {/* Body */}
              <rect
                x={(x - halfBody).toFixed(1)}
                y={bodyTop.toFixed(1)}
                width={bodyW.toFixed(1)}
                height={bodyHeight.toFixed(1)}
                fill={color}
                opacity="0.9"
              />
            </g>
          )
        })}

        {/* Current price horizontal line */}
        <line
          x1={PAD_LEFT} y1={px(lastCandle.close).toFixed(1)}
          x2={W - PAD_RIGHT} y2={px(lastCandle.close).toFixed(1)}
          stroke={lastColor} strokeWidth="0.8" strokeDasharray="3,3" opacity="0.7"
        />
        <rect x={W - PAD_RIGHT} y={px(lastCandle.close) - 8} width={PAD_RIGHT - 2} height={16} fill={lastColor} rx="2" opacity="0.9" />
        <text x={W - PAD_RIGHT + 3} y={px(lastCandle.close) + 4} fill="#000" fontSize="8" fontFamily="monospace" fontWeight="bold">
          {fmt(lastCandle.close)}
        </text>

        {/* Volume bars */}
        {candles.map((c, i) => {
          const x = cx(i)
          const color = c.close >= c.open ? '#00c853' : '#ff1744'
          const barH = (c.volume / maxVol) * VOLUME_H * 0.9
          const yTop = CHART_H + VOLUME_H - barH
          return (
            <rect
              key={i}
              x={(x - bodyW / 2).toFixed(1)}
              y={yTop.toFixed(1)}
              width={bodyW.toFixed(1)}
              height={barH.toFixed(1)}
              fill={color}
              opacity="0.3"
            />
          )
        })}

        {/* Volume label */}
        <text x={PAD_LEFT + 2} y={CHART_H + 10} fill="#444" fontSize="8" fontFamily="monospace">VOL</text>

        {/* X axis time labels */}
        {timeLabels.map(({ i, label }) => (
          <text
            key={i}
            x={cx(i).toFixed(1)}
            y={CHART_H + VOLUME_H + 16}
            fill="#555555"
            fontSize="8"
            fontFamily="monospace"
            textAnchor="middle"
          >
            {label}
          </text>
        ))}

        {/* SMA legend */}
        <line x1={PAD_LEFT + 4} y1={PAD_TOP + 6} x2={PAD_LEFT + 20} y2={PAD_TOP + 6} stroke="#f5a623" strokeWidth="1.5" />
        <text x={PAD_LEFT + 24} y={PAD_TOP + 10} fill="#f5a623" fontSize="9" fontFamily="monospace" opacity="0.8">SMA20</text>
      </svg>
    </div>
  )
}
