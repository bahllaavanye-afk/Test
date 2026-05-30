/**
 * LWEquityCurve — Lightweight Charts equity curve with optional trade markers.
 *
 * Props:
 *   data     — array of { time: number (unix seconds), value: number }
 *   markers  — optional array of trade objects from GET /api/v1/trades
 *              { time: number, side: "buy" | "sell", price: number, size: number }
 */
import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'

export interface TradeMarker {
  time: number        // unix epoch seconds
  side: 'buy' | 'sell'
  price: number
  size: number
}

interface LWEquityCurveProps {
  data: { time: number; value: number }[]
  markers?: TradeMarker[]
  height?: number
}

export default function LWEquityCurve({ data, markers = [], height = 280 }: LWEquityCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#111111' },
        textColor: '#888888',
      },
      grid: {
        vertLines: { color: '#1e1e1e' },
        horzLines: { color: '#1e1e1e' },
      },
      crosshair: {
        vertLine: { color: '#333333', labelBackgroundColor: '#1e1e1e' },
        horzLine: { color: '#333333', labelBackgroundColor: '#1e1e1e' },
      },
      rightPriceScale: {
        borderColor: '#1e1e1e',
        textColor: '#888888',
      },
      timeScale: {
        borderColor: '#1e1e1e',
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height,
    })

    const series = chart.addAreaSeries({
      lineColor: '#f5a623',
      topColor: 'rgba(245, 166, 35, 0.18)',
      bottomColor: 'rgba(245, 166, 35, 0.01)',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
    })

    if (data.length > 0) {
      series.setData(
        data.map(d => ({ time: d.time as any, value: d.value }))
      )
    }

    // Build series markers from trade data
    if (markers.length > 0) {
      const seriesMarkers = markers
        .filter(m => m.time != null && m.side != null)
        .map(m => ({
          time: m.time as any,
          position: m.side === 'buy' ? ('belowBar' as const) : ('aboveBar' as const),
          shape: m.side === 'buy' ? ('arrowUp' as const) : ('arrowDown' as const),
          color: m.side === 'buy' ? '#00c853' : '#ff1744',
          text: `${m.side.toUpperCase()} ×${m.size}`,
          size: 1,
        }))
        // Sort by time ascending (required by Lightweight Charts)
        .sort((a, b) => (a.time as number) - (b.time as number))

      series.setMarkers(seriesMarkers)
    }

    // Fit content
    chart.timeScale().fitContent()

    // Resize observer
    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
    }
  }, [data, markers, height])

  if (data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center space-y-2" style={{ height }}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        <p className="text-sm text-[#888888]">No equity curve data yet</p>
        <p className="text-xs text-[#555]">Connect Alpaca to begin paper trading.</p>
      </div>
    )
  }

  return <div ref={containerRef} style={{ height }} />
}
