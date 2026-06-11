/**
 * LWDrawdown — drawdown chart using lightweight-charts AreaSeries.
 * Red fill (#ff174422), red line (#ff1744), Y-axis capped at 0.
 *
 * Props: { data: Array<{time: string, value: number}> }
 */
import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'

interface LWDrawdownProps {
  data: { time: string; value: number }[]
  height?: number
}

export default function LWDrawdown({ data, height = 220 }: LWDrawdownProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1e2433' },
        textColor: '#888888',
      },
      grid: {
        vertLines: { color: '#ffffff0d' },
        horzLines: { color: '#ffffff0d' },
      },
      crosshair: {
        vertLine: { color: '#333333', labelBackgroundColor: '#1e2433' },
        horzLine: { color: '#333333', labelBackgroundColor: '#1e2433' },
      },
      rightPriceScale: {
        borderColor: '#ffffff1a',
        textColor: '#888888',
        // Format as percentage
        mode: 1,
      },
      timeScale: {
        borderColor: '#ffffff1a',
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height,
    })

    const series = chart.addAreaSeries({
      lineColor: '#ff1744',
      topColor: 'rgba(255,23,68,0.08)',
      bottomColor: 'rgba(255,23,68,0.22)',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      title: 'Drawdown %',
      // Invert fill — drawdown is negative so topColor is lighter
    })

    // Ensure all values are ≤ 0
    const clampedData = data.map(d => ({
      time: d.time as any,
      value: Math.min(0, d.value),
    }))

    series.setData(clampedData)

    // Add a zero line
    series.createPriceLine({
      price: 0,
      color: '#ffffff22',
      lineWidth: 1,
      lineStyle: 0,
      axisLabelVisible: false,
      title: '',
    })

    chart.timeScale().fitContent()

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
  }, [data, height])

  if (data.length === 0) {
    return (
      <div
        className="rounded-lg flex flex-col items-center justify-center gap-2"
        style={{ background: '#1e2433', height }}
      >
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#444" strokeWidth="1.5">
          <polyline points="22 17 13.5 8.5 8.5 13.5 2 7" />
          <polyline points="16 17 22 17 22 11" />
        </svg>
        <p className="text-sm text-[#555]">No drawdown data</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg overflow-hidden" style={{ background: '#1e2433' }}>
      <div className="px-4 pt-3 pb-1 flex items-center justify-between">
        <span className="text-xs font-medium text-white uppercase tracking-widest">Drawdown</span>
        <span className="text-xs text-[#ff1744]">Max: {Math.min(...data.map(d => d.value)).toFixed(2)}%</span>
      </div>
      <div ref={containerRef} style={{ height }} />
    </div>
  )
}
