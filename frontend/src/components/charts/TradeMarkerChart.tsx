/**
 * TradeMarkerChart — Lightweight Charts candlestick with buy/sell trade overlays.
 *
 * Props:
 *   symbol  — ticker symbol (e.g. "SPY")
 *   height  — chart height in pixels (default 360)
 *
 * Fetches OHLCV via GET /api/v1/market-data/bars?symbol=...
 * Fetches trades via GET /api/v1/trades?limit=200&symbol=...
 * Overlays green upward triangle markers for buys and red downward triangle markers for sells.
 * Shows P&L tooltip on hover over each marker.
 */
import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  CandlestickData,
} from 'lightweight-charts'
import api from '../../api/client'

// ── Types ────────────────────────────────────────────────────────────────────

interface TradeRecord {
  id: string
  symbol: string
  side: string
  avg_fill_price: number | null
  entry_price: number | null
  exit_price: number | null
  realized_pnl: number | null
  quantity: number
  opened_at: string | null
  closed_at: string | null
  strategy_name: string | null
}

interface OHLCVBar {
  time: number   // unix epoch seconds
  open: number
  high: number
  low: number
  close: number
}

interface TooltipState {
  visible: boolean
  x: number
  y: number
  side: string
  price: number
  pnl: number | null
  qty: number
  strategy: string | null
  time: string
}

interface TradeMarkerChartProps {
  symbol: string
  height?: number
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function isoToUnix(iso: string | null | undefined): number | null {
  if (!iso) return null
  const ms = new Date(iso).getTime()
  return isNaN(ms) ? null : Math.floor(ms / 1000)
}

function formatPnl(pnl: number | null): string {
  if (pnl == null) return '—'
  const sign = pnl >= 0 ? '+' : ''
  return `${sign}$${pnl.toFixed(2)}`
}

// ── Component ────────────────────────────────────────────────────────────────

export default function TradeMarkerChart({ symbol, height = 360 }: TradeMarkerChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false, x: 0, y: 0, side: '', price: 0, pnl: null, qty: 0, strategy: null, time: '',
  })

  // Fetch OHLCV bars for the symbol (last 200 daily candles)
  const { data: bars, isLoading: barsLoading, error: barsError } = useQuery<OHLCVBar[]>({
    queryKey: ['market-data-bars', symbol],
    queryFn: () =>
      api.get('/market-data/bars', { params: { symbol, timeframe: '1Day', limit: 200 } })
        .then(r => r.data),
    staleTime: 60_000,
  })

  // Fetch recent trades filtered by symbol
  const { data: trades, isLoading: tradesLoading } = useQuery<TradeRecord[]>({
    queryKey: ['trades-for-chart', symbol],
    queryFn: () =>
      api.get('/trades/', { params: { limit: 200, symbol } })
        .then(r => r.data),
    staleTime: 30_000,
  })

  // Build / update chart when data changes
  useEffect(() => {
    if (!containerRef.current) return
    if (!bars || bars.length === 0) return

    // Create chart once
    if (!chartRef.current) {
      chartRef.current = createChart(containerRef.current, {
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

      candleSeriesRef.current = chartRef.current.addCandlestickSeries({
        upColor: '#00c853',
        downColor: '#ff1744',
        borderUpColor: '#00c853',
        borderDownColor: '#ff1744',
        wickUpColor: '#00c853',
        wickDownColor: '#ff1744',
      })
    } else {
      chartRef.current.applyOptions({ height })
    }

    // Set OHLCV data — lightweight-charts requires sorted ascending by time
    const sortedBars = [...bars].sort((a, b) => a.time - b.time)
    candleSeriesRef.current!.setData(
      sortedBars.map(b => ({
        time: b.time as any,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      }))
    )

    // Build trade markers
    if (trades && trades.length > 0) {
      const sortedBarTimes = sortedBars.map(b => b.time)
      const barTimeSet = new Set(sortedBarTimes)

      // Annotate each marker with extra data for the tooltip
      type MarkerWithMeta = {
        time: number
        position: 'belowBar' | 'aboveBar'
        shape: 'arrowUp' | 'arrowDown'
        color: string
        text: string
        size: number
        _trade: TradeRecord
        _price: number
      }

      const markerData: MarkerWithMeta[] = trades
        .filter(t => t.symbol === symbol)
        .map((t): MarkerWithMeta | null => {
          const rawTime = t.side === 'buy'
            ? isoToUnix(t.opened_at)
            : isoToUnix(t.closed_at ?? t.opened_at)
          if (rawTime == null) return null

          // Snap to nearest available bar time (floor)
          let snapped = sortedBarTimes[0]
          for (const bt of sortedBarTimes) {
            if (bt <= rawTime) snapped = bt
          }
          if (!barTimeSet.has(snapped)) return null

          const price =
            t.avg_fill_price ??
            (t.side === 'buy' ? t.entry_price : t.exit_price) ??
            0

          return {
            time: snapped,
            position: t.side === 'buy' ? 'belowBar' : 'aboveBar',
            shape: t.side === 'buy' ? 'arrowUp' : 'arrowDown',
            color: t.side === 'buy' ? '#00c853' : '#ff1744',
            text: t.strategy_name ?? (t.side === 'buy' ? 'BUY' : 'SELL'),
            size: 2,
            _trade: t,
            _price: price,
          }
        })
        .filter((m): m is MarkerWithMeta => m !== null)
        .sort((a, b) => a.time - b.time)

      // Strip the custom fields before passing to the chart library
      const lwMarkers = markerData.map(({ _trade: _t, _price: _p, ...rest }) => ({
        ...rest,
        time: rest.time as any,
      }))
      candleSeriesRef.current!.setMarkers(lwMarkers)

      // Crosshair hover → show tooltip when near a marker
      chartRef.current!.subscribeCrosshairMove(param => {
        if (!param.point || !param.time) {
          setTooltip(prev => ({ ...prev, visible: false }))
          return
        }
        const hoverTime = param.time as number
        const nearby = markerData.find(m => Math.abs(m.time - hoverTime) < 86400)
        if (nearby && containerRef.current) {
          setTooltip({
            visible: true,
            x: param.point.x ?? 0,
            y: param.point.y ?? 0,
            side: nearby._trade.side,
            price: nearby._price,
            pnl: nearby._trade.realized_pnl,
            qty: nearby._trade.quantity,
            strategy: nearby._trade.strategy_name,
            time: new Date(nearby.time * 1000).toLocaleDateString(),
          })
        } else {
          setTooltip(prev => ({ ...prev, visible: false }))
        }
      })
    } else {
      candleSeriesRef.current!.setMarkers([])
    }

    chartRef.current!.timeScale().fitContent()
  }, [bars, trades, symbol, height])

  // Destroy chart on unmount
  useEffect(() => {
    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
        candleSeriesRef.current = null
      }
    }
  }, [])

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return
    const el = containerRef.current
    const ro = new ResizeObserver(() => {
      if (chartRef.current && el) {
        chartRef.current.applyOptions({ width: el.clientWidth })
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const isLoading = barsLoading || tradesLoading

  if (isLoading) {
    return (
      <div
        className="flex items-center justify-center bg-[#111111] border border-[#1e1e1e] rounded"
        style={{ height }}
      >
        <div className="space-y-2 w-full px-6">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-4 bg-[#1e1e1e] rounded animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  if (barsError) {
    return (
      <div
        className="flex items-center justify-center bg-[#111111] border border-[#1e1e1e] rounded text-[#888888] text-sm"
        style={{ height }}
      >
        Failed to load chart data for {symbol}
      </div>
    )
  }

  if (!bars || bars.length === 0) {
    return (
      <div
        className="flex flex-col items-center justify-center bg-[#111111] border border-[#1e1e1e] rounded space-y-2"
        style={{ height }}
      >
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        <p className="text-sm text-[#888888]">No market data for {symbol}</p>
      </div>
    )
  }

  return (
    <div className="relative">
      <div ref={containerRef} style={{ height }} />

      {/* P&L Tooltip */}
      {tooltip.visible && (
        <div
          className="absolute z-50 pointer-events-none"
          style={{ left: tooltip.x + 16, top: Math.max(0, tooltip.y - 90) }}
        >
          <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 shadow-xl min-w-[160px]">
            <div className="flex items-center gap-2 mb-1">
              <span
                className="text-xs font-bold uppercase"
                style={{ color: tooltip.side === 'buy' ? '#00c853' : '#ff1744' }}
              >
                {tooltip.side === 'buy' ? '▲ BUY' : '▼ SELL'}
              </span>
              <span className="text-xs text-[#555]">{tooltip.time}</span>
            </div>
            <div className="text-xs text-[#888888]">
              Fill:{' '}
              <span className="text-[#e8e8e8] font-mono">${tooltip.price.toFixed(2)}</span>
            </div>
            <div className="text-xs text-[#888888]">
              Qty: <span className="text-[#e8e8e8] font-mono">{tooltip.qty}</span>
            </div>
            <div className="text-xs text-[#888888]">
              P&amp;L:{' '}
              <span
                className="font-mono font-bold"
                style={{
                  color:
                    tooltip.pnl == null ? '#888' : tooltip.pnl >= 0 ? '#00c853' : '#ff1744',
                }}
              >
                {formatPnl(tooltip.pnl)}
              </span>
            </div>
            {tooltip.strategy && (
              <div className="text-[10px] text-[#555] mt-1 truncate">{tooltip.strategy}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
