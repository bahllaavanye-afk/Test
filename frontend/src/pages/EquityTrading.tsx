import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import TVAdvancedChart from '../components/charts/TVAdvancedChart'
import AdvancedOrderForm from '../components/trading/AdvancedOrderForm'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────
type Interval = '1' | '5' | '15' | '60' | '240' | 'D' | 'W'

const SYMBOLS = [
  { tv: 'NYSE:SPY',      label: 'SPY' },
  { tv: 'NASDAQ:AAPL',   label: 'AAPL' },
  { tv: 'NASDAQ:MSFT',   label: 'MSFT' },
  { tv: 'NASDAQ:NVDA',   label: 'NVDA' },
  { tv: 'NASDAQ:GOOGL',  label: 'GOOGL' },
  { tv: 'NASDAQ:AMZN',   label: 'AMZN' },
  { tv: 'NASDAQ:META',   label: 'META' },
  { tv: 'NASDAQ:TSLA',   label: 'TSLA' },
  { tv: 'NYSE:QQQ',      label: 'QQQ' },
]

const INTERVALS: { label: string; value: Interval }[] = [
  { label: '1m',  value: '1' },
  { label: '5m',  value: '5' },
  { label: '15m', value: '15' },
  { label: '1h',  value: '60' },
  { label: '4h',  value: '240' },
  { label: '1D',  value: 'D' },
  { label: '1W',  value: 'W' },
]

const STUDIES_PRESETS: Record<string, string[]> = {
  default: ['Volume@tv-basicstudies', 'MACD@tv-basicstudies', 'RSI@tv-basicstudies'],
  momentum: ['Volume@tv-basicstudies', 'RSI@tv-basicstudies', 'MACD@tv-basicstudies', 'MOM@tv-basicstudies'],
  bands: ['Volume@tv-basicstudies', 'BB@tv-basicstudies', 'ATR@tv-basicstudies'],
  trend: ['Volume@tv-basicstudies', 'MAExp@tv-basicstudies', 'MAExp@tv-basicstudies'],
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function pnlColor(v: number | null | undefined) {
  if (v == null) return '#555'
  return v >= 0 ? '#00c853' : '#ff1744'
}

function fmtPnl(v: number | null | undefined, prefix = '$') {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : '-'}${prefix}${Math.abs(v).toFixed(2)}`
}

function fmtPct(v: number | null | undefined) {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`
}

// ─── Positions Table ──────────────────────────────────────────────────────────
function PositionsPanel({ onSymbolClick }: { onSymbolClick: (sym: string) => void }) {
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['positions-terminal'],
    queryFn: () => api.get('/positions/').then(r => r.data),
    refetchInterval: 5_000,
  })
  const positions: any[] = Array.isArray(data) ? data : []

  const totalUnrealPnl = positions.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1e1e1e]">
        <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">Positions ({positions.length})</span>
        <div className="flex items-center gap-2">
          {positions.length > 0 && (
            <span className="text-xs font-bold font-mono" style={{ color: pnlColor(totalUnrealPnl) }}>
              {fmtPnl(totalUnrealPnl)} unrealized
            </span>
          )}
          <button onClick={() => refetch()} className="text-[10px] text-[#444] hover:text-[#888] transition-colors">↻</button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-3 space-y-2">
            {[1, 2].map(i => <div key={i} className="h-10 bg-[#1a1a1a] rounded animate-pulse" />)}
          </div>
        ) : positions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center px-4">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#333" strokeWidth="1.5">
              <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>
            </svg>
            <p className="text-xs text-[#444]">No open positions</p>
            <p className="text-[10px] text-[#333]">Enter an order to open a position</p>
          </div>
        ) : (
          <div className="divide-y divide-[#111]">
            {positions.map((pos: any) => {
              const pnl = pos.unrealized_pnl ?? null
              const pct = pos.current_price != null && pos.avg_cost > 0
                ? ((pos.current_price - pos.avg_cost) / pos.avg_cost * 100) * (pos.side === 'short' ? -1 : 1)
                : null
              const sym = pos.symbol ?? '?'
              return (
                <div
                  key={pos.id}
                  className="px-3 py-2.5 hover:bg-[#111] transition-colors cursor-pointer"
                  onClick={() => {
                    // Find matching TV symbol or build one
                    const found = SYMBOLS.find(s => s.label === sym)
                    onSymbolClick(found ? found.tv : `NASDAQ:${sym}`)
                  }}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="text-[9px] font-bold px-1 rounded"
                        style={{
                          background: pos.side === 'long' ? 'rgba(0,200,83,0.15)' : 'rgba(255,23,68,0.15)',
                          color: pos.side === 'long' ? '#00c853' : '#ff1744',
                        }}
                      >
                        {(pos.side ?? 'LONG').toUpperCase()}
                      </span>
                      <span className="text-xs font-bold text-[#e8e8e8] font-mono">{sym}</span>
                    </div>
                    <span className="text-xs font-bold font-mono" style={{ color: pnlColor(pnl) }}>
                      {fmtPnl(pnl)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-[#555]">
                      {pos.quantity} @ ${(pos.avg_cost ?? 0).toFixed(2)}
                    </span>
                    <span style={{ color: pnlColor(pct) }}>{fmtPct(pct)}</span>
                  </div>
                  {pos.current_price != null && (
                    <div className="text-[10px] text-[#444] mt-0.5">
                      Current: ${Number(pos.current_price).toFixed(2)}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Orders Feed ──────────────────────────────────────────────────────────────
function OrdersFeed() {
  const { data, isLoading } = useQuery({
    queryKey: ['orders-terminal'],
    queryFn: () => api.get('/orders/?limit=30').then(r => r.data),
    refetchInterval: 3_000,
  })
  const orders: any[] = Array.isArray(data) ? data : []

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-[#1e1e1e]">
        <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">Order History</span>
      </div>
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-3 space-y-1">
            {[1, 2, 3].map(i => <div key={i} className="h-8 bg-[#1a1a1a] rounded animate-pulse" />)}
          </div>
        ) : orders.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-xs text-[#444]">No orders placed yet</p>
          </div>
        ) : (
          <div className="divide-y divide-[#111]">
            {orders.map((o: any, i: number) => {
              const isBuy = o.side === 'buy'
              const ts = o.created_at ? new Date(o.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'
              return (
                <div key={o.id ?? i} className="px-3 py-2 flex items-center justify-between hover:bg-[#111] transition-colors">
                  <div className="flex items-center gap-2">
                    <span
                      className="text-[9px] font-bold px-1 py-0.5 rounded min-w-[28px] text-center"
                      style={{
                        background: isBuy ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)',
                        color: isBuy ? '#00c853' : '#ff1744',
                      }}
                    >
                      {isBuy ? 'BUY' : 'SELL'}
                    </span>
                    <div>
                      <p className="text-xs font-mono text-[#e8e8e8]">{o.symbol} <span className="text-[#555]">×{o.quantity}</span></p>
                      <p className="text-[10px] text-[#444]">{o.order_type} · {o.execution_algo ?? 'auto'}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p
                      className="text-[10px] font-bold"
                      style={{
                        color: o.status === 'filled' ? '#00c853'
                             : o.status === 'cancelled' ? '#ff1744'
                             : '#f5a623',
                      }}
                    >
                      {(o.status ?? '—').toUpperCase()}
                    </p>
                    <p className="text-[9px] text-[#444]">{ts}</p>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}


// ─── Account Summary Bar ──────────────────────────────────────────────────────
function AccountBar() {
  const { data: accounts } = useQuery({
    queryKey: ['accounts-bar'],
    queryFn: () => api.get('/accounts/').then(r => r.data),
    refetchInterval: 15_000,
  })
  const { data: perf } = useQuery({
    queryKey: ['performance'],
    queryFn: () => api.get('/analytics/performance').then(r => r.data),
    refetchInterval: 30_000,
  })
  const accts: any[] = Array.isArray(accounts) ? accounts : []
  const totalEquity = accts.reduce((s, a) => s + (a.equity ?? 0), 0)
  const totalPnl = perf?.total_pnl ?? null

  if (accts.length === 0) return null

  return (
    <div className="flex items-center gap-6 px-4 py-2 bg-[#0d0d0d] border-b border-[#1e1e1e] text-xs">
      {accts.map((a: any) => (
        <div key={a.id} className="flex items-center gap-1.5">
          <span className="text-[#444]">{a.label}:</span>
          <span className="font-mono font-bold text-[#e8e8e8]">
            {a.equity != null ? `$${Number(a.equity).toLocaleString()}` : '—'}
          </span>
          <span className="text-[10px] px-1 rounded" style={{
            background: a.mode === 'live' ? 'rgba(255,23,68,0.15)' : 'rgba(245,166,35,0.15)',
            color: a.mode === 'live' ? '#ff1744' : '#f5a623',
          }}>{a.mode?.toUpperCase()}</span>
        </div>
      ))}
      {totalPnl != null && (
        <>
          <div className="h-3 w-px bg-[#1e1e1e]" />
          <div className="flex items-center gap-1.5">
            <span className="text-[#444]">P&L:</span>
            <span className="font-mono font-bold" style={{ color: pnlColor(totalPnl) }}>
              {fmtPnl(totalPnl)}
            </span>
          </div>
        </>
      )}
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function EquityTrading() {
  const [activeSymbol, setActiveSymbol] = useState('NASDAQ:AAPL')
  const [interval, setInterval] = useState<Interval>('60')
  const [studyPreset, setStudyPreset] = useState<keyof typeof STUDIES_PRESETS>('default')
  const [rightTab, setRightTab] = useState<'order' | 'positions' | 'orders'>('order')

  const tickerSymbol = useMemo(
    () => activeSymbol.split(':')[1] ?? activeSymbol,
    [activeSymbol]
  )

  return (
    <div className="flex flex-col h-[calc(100vh-56px)] bg-[#0a0a0a]">
      <AccountBar />

      {/* ── Symbol + Interval Bar ── */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[#1e1e1e] bg-[#0d0d0d] flex-wrap">
        {/* Symbol picker */}
        <div className="flex items-center gap-1 flex-wrap">
          {SYMBOLS.map(s => (
            <button
              key={s.tv}
              onClick={() => setActiveSymbol(s.tv)}
              className="text-xs px-2 py-1 rounded font-mono font-bold transition-all duration-100"
              style={{
                background: activeSymbol === s.tv ? '#f5a623' : '#111',
                color: activeSymbol === s.tv ? '#000' : '#666',
                border: `1px solid ${activeSymbol === s.tv ? '#f5a623' : '#1e1e1e'}`,
              }}
            >
              {s.label}
            </button>
          ))}
          {/* Custom symbol input */}
          <input
            className="bg-[#111] border border-[#1e1e1e] rounded px-2 py-1 text-xs font-mono text-white w-28 focus:outline-none focus:border-[#333] uppercase"
            placeholder="EXCHANGE:SYM"
            onKeyDown={e => {
              if (e.key === 'Enter') {
                const val = (e.target as HTMLInputElement).value.trim().toUpperCase()
                if (val) setActiveSymbol(val)
              }
            }}
          />
        </div>

        <div className="h-4 w-px bg-[#1e1e1e] mx-1" />

        {/* Timeframe */}
        <div className="flex items-center gap-1">
          {INTERVALS.map(iv => (
            <button
              key={iv.value}
              onClick={() => setInterval(iv.value)}
              className="text-xs px-2 py-1 rounded transition-all"
              style={{
                background: interval === iv.value ? '#2196f3' : 'transparent',
                color: interval === iv.value ? '#fff' : '#555',
              }}
            >
              {iv.label}
            </button>
          ))}
        </div>

        <div className="h-4 w-px bg-[#1e1e1e] mx-1" />

        {/* Study preset */}
        <div className="flex items-center gap-1">
          {(Object.keys(STUDIES_PRESETS) as (keyof typeof STUDIES_PRESETS)[]).map(p => (
            <button
              key={p}
              onClick={() => setStudyPreset(p)}
              className="text-[10px] px-1.5 py-0.5 rounded capitalize transition-colors"
              style={{
                background: studyPreset === p ? '#9c27b0' : 'transparent',
                color: studyPreset === p ? '#fff' : '#444',
              }}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* ── Main Layout: Chart + Right Panel ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chart takes most of the space */}
        <div className="flex-1 min-w-0 p-0">
          <TVAdvancedChart
            symbol={activeSymbol}
            interval={interval}
            height={undefined as any}
            studies={STUDIES_PRESETS[studyPreset]}
            showWatchlist={true}
            showDetails={true}
          />
        </div>

        {/* ── Right Panel ── */}
        <div className="w-72 flex-shrink-0 flex flex-col bg-[#0d0d0d] border-l border-[#1e1e1e]">
          {/* Tab bar */}
          <div className="flex border-b border-[#1e1e1e]">
            {([
              { key: 'order',     label: 'Order Entry' },
              { key: 'positions', label: 'Positions' },
              { key: 'orders',    label: 'History' },
            ] as { key: typeof rightTab; label: string }[]).map(tab => (
              <button
                key={tab.key}
                onClick={() => setRightTab(tab.key)}
                className="flex-1 py-2 text-[10px] font-medium uppercase tracking-wider transition-colors"
                style={{
                  color: rightTab === tab.key ? '#f5a623' : '#444',
                  borderBottom: rightTab === tab.key ? '2px solid #f5a623' : '2px solid transparent',
                  background: 'transparent',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto">
            {rightTab === 'order' && (
              <AdvancedOrderForm
                defaultSymbol={tickerSymbol}
                onSuccess={() => setRightTab('positions')}
              />
            )}
            {rightTab === 'positions' && <PositionsPanel onSymbolClick={setActiveSymbol} />}
            {rightTab === 'orders' && <OrdersFeed />}
          </div>
        </div>
      </div>
    </div>
  )
}
