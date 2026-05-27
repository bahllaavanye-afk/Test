import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import TVAdvancedChart from '../components/charts/TVAdvancedChart'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────
type Side = 'buy' | 'sell'
type OrdType = 'market' | 'limit' | 'stop' | 'stop_limit'
type Algo = 'auto' | 'market' | 'limit_first' | 'twap' | 'vwap'
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

const ALGO_INFO: Record<Algo, { label: string; desc: string; color: string }> = {
  auto:        { label: 'Auto',        desc: 'Smart route: size-based selection',        color: '#888' },
  market:      { label: 'Market',      desc: 'Immediate fill at best available price',   color: '#ff1744' },
  limit_first: { label: 'Limit-First', desc: 'Limit → market fallback (saves ~5 bps)',  color: '#00c853' },
  twap:        { label: 'TWAP',        desc: 'Time-slice over 30min — low market impact',color: '#2196f3' },
  vwap:        { label: 'VWAP',        desc: 'Volume-weighted — tracks market rhythm',   color: '#9c27b0' },
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

// ─── Order Entry ──────────────────────────────────────────────────────────────
function OrderEntry({ defaultSymbol }: { defaultSymbol: string }) {
  const qc = useQueryClient()
  const [sym, setSym] = useState(defaultSymbol.split(':')[1] ?? defaultSymbol)
  const [side, setSide] = useState<Side>('buy')
  const [type, setType] = useState<OrdType>('limit')
  const [qty, setQty] = useState('1')
  const [limit, setLimit] = useState('')
  const [stop, setStop] = useState('')
  const [algo, setAlgo] = useState<Algo>('limit_first')
  const [notional, setNotional] = useState('')
  const [useNotional, setUseNotional] = useState(false)

  const { data: accounts } = useQuery({
    queryKey: ['accounts-order'],
    queryFn: () => api.get('/accounts/').then(r => r.data),
    staleTime: 60_000,
  })
  const accountList: any[] = Array.isArray(accounts) ? accounts : []
  const [accountId, setAccountId] = useState('')

  const mutation = useMutation({
    mutationFn: () => {
      if (!accountId) throw new Error('Select an account first')
      return api.post('/orders/', {
        symbol: sym.toUpperCase(),
        side,
        order_type: type,
        quantity: useNotional ? undefined : parseFloat(qty) || 1,
        notional: useNotional ? parseFloat(notional) : undefined,
        limit_price: (type === 'limit' || type === 'stop_limit') && limit ? parseFloat(limit) : null,
        stop_price: (type === 'stop' || type === 'stop_limit') && stop ? parseFloat(stop) : null,
        execution_algo: algo,
        account_id: accountId,
      }).then(r => r.data)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['orders-terminal'] })
      qc.invalidateQueries({ queryKey: ['positions-terminal'] })
    },
  })

  const needsLimit = type === 'limit' || type === 'stop_limit'
  const needsStop  = type === 'stop'  || type === 'stop_limit'

  return (
    <div className="p-3 space-y-2.5">
      {/* Symbol */}
      <div>
        <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Symbol</p>
        <input
          value={sym}
          onChange={e => setSym(e.target.value.toUpperCase())}
          className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2.5 py-1.5 text-sm font-mono font-bold text-white focus:outline-none focus:border-[#333] transition-colors uppercase"
          placeholder="AAPL"
        />
      </div>

      {/* Account */}
      {accountList.length > 0 ? (
        <div>
          <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Account</p>
          <select
            value={accountId}
            onChange={e => setAccountId(e.target.value)}
            className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none"
          >
            <option value="">Select account…</option>
            {accountList.map((a: any) => (
              <option key={a.id} value={a.id}>{a.label} ({a.mode})</option>
            ))}
          </select>
        </div>
      ) : (
        <div className="bg-[#0d0d0d] border border-[#f5a623]/20 rounded-lg px-2.5 py-2">
          <p className="text-[10px] text-[#f5a623] font-medium">No broker account connected</p>
          <p className="text-[9px] text-[#555] mt-0.5">Add Alpaca API keys in Settings to place real orders.</p>
        </div>
      )}

      {/* Side */}
      <div className="grid grid-cols-2 gap-1.5">
        {(['buy', 'sell'] as Side[]).map(s => (
          <button
            key={s}
            onClick={() => setSide(s)}
            className="py-2 rounded-lg text-xs font-bold tracking-wide transition-all duration-150"
            style={{
              background: side === s
                ? (s === 'buy' ? '#00c853' : '#ff1744')
                : '#1a1a1a',
              color: side === s ? (s === 'buy' ? '#000' : '#fff') : '#555',
              border: `1px solid ${side === s ? (s === 'buy' ? '#00c853' : '#ff1744') : '#222'}`,
            }}
          >
            {s === 'buy' ? '▲ BUY / LONG' : '▼ SELL / SHORT'}
          </button>
        ))}
      </div>

      {/* Order type */}
      <div>
        <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Order type</p>
        <div className="grid grid-cols-4 gap-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-1">
          {(['market', 'limit', 'stop', 'stop_limit'] as OrdType[]).map(t => (
            <button
              key={t}
              onClick={() => setType(t)}
              className="py-1 rounded text-[10px] font-medium transition-colors"
              style={{
                background: type === t ? '#1e1e1e' : 'transparent',
                color: type === t ? '#e8e8e8' : '#555',
              }}
            >
              {t === 'stop_limit' ? 'STP-LMT' : t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Qty / Notional toggle */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <p className="text-[10px] text-[#555] uppercase tracking-wider">{useNotional ? 'Notional ($)' : 'Shares / Qty'}</p>
          <button
            onClick={() => setUseNotional(v => !v)}
            className="text-[9px] text-[#444] hover:text-[#888] transition-colors"
          >
            Switch to {useNotional ? 'shares' : 'notional'}
          </button>
        </div>
        {useNotional ? (
          <input
            type="number"
            min="1"
            value={notional}
            onChange={e => setNotional(e.target.value)}
            className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-[#333]"
            placeholder="1000.00"
          />
        ) : (
          <input
            type="number"
            min="0.001"
            step="any"
            value={qty}
            onChange={e => setQty(e.target.value)}
            className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-[#333]"
            placeholder="1"
          />
        )}
      </div>

      {/* Limit price */}
      {needsLimit && (
        <div>
          <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Limit price</p>
          <input
            type="number"
            step="0.01"
            value={limit}
            onChange={e => setLimit(e.target.value)}
            className="w-full bg-[#0a0a0a] border border-[#2196f3]/40 rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-[#2196f3]/80 transition-colors"
            placeholder="0.00"
          />
        </div>
      )}

      {/* Stop price */}
      {needsStop && (
        <div>
          <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Stop price</p>
          <input
            type="number"
            step="0.01"
            value={stop}
            onChange={e => setStop(e.target.value)}
            className="w-full bg-[#0a0a0a] border border-[#f5a623]/40 rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-[#f5a623]/80 transition-colors"
            placeholder="0.00"
          />
        </div>
      )}

      {/* Execution algo */}
      <div>
        <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Execution</p>
        <div className="space-y-1">
          {(Object.keys(ALGO_INFO) as Algo[]).map(a => (
            <button
              key={a}
              onClick={() => setAlgo(a)}
              className="w-full text-left px-2.5 py-1.5 rounded-lg text-xs transition-all"
              style={{
                background: algo === a ? '#111' : 'transparent',
                border: `1px solid ${algo === a ? ALGO_INFO[a].color + '44' : '#1a1a1a'}`,
                color: algo === a ? ALGO_INFO[a].color : '#555',
              }}
            >
              <span className="font-bold">{ALGO_INFO[a].label}</span>
              <span className="text-[9px] ml-2 opacity-70">{ALGO_INFO[a].desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending || !accountId}
        className="w-full py-2.5 rounded-lg text-sm font-bold tracking-wide transition-all duration-150 active:scale-[0.98]"
        style={{
          background: side === 'buy'
            ? (mutation.isPending ? '#1a2a1a' : 'linear-gradient(135deg, #00c853, #00a843)')
            : (mutation.isPending ? '#2a1a1a' : 'linear-gradient(135deg, #ff1744, #c62828)'),
          color: '#fff',
          opacity: !accountId ? 0.4 : 1,
          cursor: !accountId ? 'not-allowed' : 'pointer',
        }}
      >
        {mutation.isPending ? 'Submitting…' : side === 'buy' ? `▲ BUY ${sym || '—'}` : `▼ SELL ${sym || '—'}`}
      </button>

      {mutation.isError && (
        <div className="bg-[#ff1744]/10 border border-[#ff1744]/20 rounded-lg px-2.5 py-2">
          <p className="text-xs text-[#ff1744]">{String((mutation.error as any)?.response?.data?.detail ?? mutation.error)}</p>
        </div>
      )}
      {mutation.isSuccess && (
        <div className="bg-[#00c853]/10 border border-[#00c853]/20 rounded-lg px-2.5 py-2">
          <p className="text-xs text-[#00c853]">Order submitted successfully</p>
        </div>
      )}
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
          <div className="flex-1 overflow-hidden">
            {rightTab === 'order' && <OrderEntry defaultSymbol={tickerSymbol} />}
            {rightTab === 'positions' && <PositionsPanel onSymbolClick={setActiveSymbol} />}
            {rightTab === 'orders' && <OrdersFeed />}
          </div>
        </div>
      </div>
    </div>
  )
}
