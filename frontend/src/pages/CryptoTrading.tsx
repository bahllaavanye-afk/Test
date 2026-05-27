import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import TVAdvancedChart from '../components/charts/TVAdvancedChart'
import AdvancedOrderForm from '../components/trading/AdvancedOrderForm'
import api from '../api/client'

type Interval = '1' | '5' | '15' | '60' | '240' | 'D'

// Alpaca crypto pairs — traded via Alpaca, charted via Coinbase/CRYPTO feed
const CRYPTO_PAIRS = [
  { tv: 'COINBASE:BTCUSD',  alpaca: 'BTC/USD',  label: 'BTC',  color: '#f7931a' },
  { tv: 'COINBASE:ETHUSD',  alpaca: 'ETH/USD',  label: 'ETH',  color: '#627eea' },
  { tv: 'COINBASE:SOLUSD',  alpaca: 'SOL/USD',  label: 'SOL',  color: '#9945ff' },
  { tv: 'COINBASE:DOGEUSD', alpaca: 'DOGE/USD', label: 'DOGE', color: '#c2a633' },
  { tv: 'COINBASE:LTCUSD',  alpaca: 'LTC/USD',  label: 'LTC',  color: '#bfbbbb' },
  { tv: 'COINBASE:BCHUSD',  alpaca: 'BCH/USD',  label: 'BCH',  color: '#8dc351' },
  { tv: 'COINBASE:UNIUSD',  alpaca: 'UNI/USD',  label: 'UNI',  color: '#ff007a' },
  { tv: 'COINBASE:LINKUSD', alpaca: 'LINK/USD', label: 'LINK', color: '#2a5ada' },
]

const CRYPTO_WATCHLIST = CRYPTO_PAIRS.map(p => p.tv)
  .concat(['COINBASE:AVAXUSD', 'COINBASE:MATICUSD', 'COINBASE:XRPUSD'])

const INTERVALS: { label: string; value: Interval }[] = [
  { label: '1m',  value: '1' },
  { label: '5m',  value: '5' },
  { label: '15m', value: '15' },
  { label: '1h',  value: '60' },
  { label: '4h',  value: '240' },
  { label: '1D',  value: 'D' },
]

const CRYPTO_STUDIES = [
  'Volume@tv-basicstudies',
  'RSI@tv-basicstudies',
  'MACD@tv-basicstudies',
  'BB@tv-basicstudies',
]

function pnlColor(v: number | null | undefined) {
  if (v == null) return '#555'
  return v >= 0 ? '#00c853' : '#ff1744'
}

// ─── Arb Scanner ──────────────────────────────────────────────────────────────
function ArbScanner() {
  const { data, isLoading } = useQuery({
    queryKey: ['arb-opportunities'],
    queryFn: () => api.get('/analytics/arb-opportunities').then(r => r.data).catch(() => null),
    refetchInterval: 10_000,
  })

  const ops: any[] = Array.isArray(data) ? data : []

  return (
    <div className="bg-[#0d0d0d] border-t border-[#1e1e1e]">
      <div className="px-3 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#f5a623] animate-pulse" />
          <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">Spread Scanner</span>
        </div>
        <span className="text-[10px] text-[#333]">Alpaca crypto spreads</span>
      </div>
      <div className="px-3 pb-2">
        {isLoading ? (
          <div className="h-6 bg-[#1a1a1a] rounded animate-pulse" />
        ) : ops.length > 0 ? (
          <div className="space-y-1">
            {ops.map((op: any, i: number) => (
              <div key={i} className="flex items-center justify-between text-xs bg-[#111] rounded px-2 py-1.5">
                <span className="font-mono text-[#e8e8e8]">{op.path ?? 'BTC/USD spread'}</span>
                <span className="font-bold" style={{ color: (op.spread ?? 0) > 0 ? '#00c853' : '#ff1744' }}>
                  {op.spread != null ? `${(op.spread * 100).toFixed(3)}%` : '—'}
                </span>
                <span className="text-[10px] text-[#444]">{op.status ?? 'monitoring'}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[10px] text-[#333] italic">
            Monitors BTC/ETH/SOL bid-ask spreads on Alpaca. Signals fire when spread &gt; 0.15%.
          </p>
        )}
      </div>
    </div>
  )
}

// ─── Crypto Positions ─────────────────────────────────────────────────────────
function CryptoPositions() {
  const { data } = useQuery({
    queryKey: ['positions-crypto'],
    queryFn: () => api.get('/positions/').then(r => r.data),
    refetchInterval: 5_000,
    select: (d: any[]) => d.filter(p =>
      ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC']
        .some(c => p.symbol?.includes(c))
    ),
  })
  const positions: any[] = Array.isArray(data) ? data : []

  return (
    <div>
      <div className="px-3 py-2 border-b border-[#1e1e1e]">
        <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">Crypto Positions ({positions.length})</span>
      </div>
      {positions.length === 0 ? (
        <div className="px-3 py-6 text-center">
          <p className="text-xs text-[#444]">No crypto positions open</p>
        </div>
      ) : (
        <div className="divide-y divide-[#111]">
          {positions.map((p: any) => {
            const pnl = p.unrealized_pnl ?? null
            return (
              <div key={p.id} className="px-3 py-2">
                <div className="flex justify-between items-center mb-0.5">
                  <span className="text-xs font-bold font-mono text-[#e8e8e8]">{p.symbol}</span>
                  <span className="text-xs font-bold font-mono" style={{ color: pnlColor(pnl) }}>
                    {pnl != null ? `${pnl >= 0 ? '+' : '-'}$${Math.abs(pnl).toFixed(2)}` : '—'}
                  </span>
                </div>
                <div className="flex justify-between text-[10px] text-[#444]">
                  <span>{p.quantity} @ ${Number(p.avg_cost ?? 0).toFixed(4)}</span>
                  {p.current_price && <span>Now: ${Number(p.current_price).toFixed(4)}</span>}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── Crypto Orders Feed ───────────────────────────────────────────────────────
function CryptoOrders() {
  const { data } = useQuery({
    queryKey: ['orders-crypto'],
    queryFn: () => api.get('/orders/?limit=20').then(r => r.data),
    refetchInterval: 5_000,
  })
  const orders: any[] = (Array.isArray(data) ? data : []).filter((o: any) =>
    ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'USDT'].some(c => o.symbol?.includes(c))
  )

  return (
    <div>
      <div className="px-3 py-2 border-b border-[#1e1e1e]">
        <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">Crypto Orders</span>
      </div>
      {orders.length === 0 ? (
        <div className="px-3 py-6 text-center"><p className="text-xs text-[#444]">No crypto orders yet</p></div>
      ) : (
        <div className="divide-y divide-[#111]">
          {orders.map((o: any, i: number) => (
            <div key={o.id ?? i} className="px-3 py-2 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="text-[9px] font-bold px-1 rounded"
                  style={{ background: o.side === 'buy' ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)', color: o.side === 'buy' ? '#00c853' : '#ff1744' }}>
                  {o.side?.toUpperCase()}
                </span>
                <span className="text-xs font-mono text-[#e8e8e8]">{o.symbol}</span>
              </div>
              <span className="text-[10px]"
                style={{ color: o.status === 'filled' ? '#00c853' : o.status === 'cancelled' ? '#ff1744' : '#f5a623' }}>
                {o.status}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function CryptoTrading() {
  const [activePair, setActivePair] = useState('COINBASE:BTCUSD')
  const [interval, setInterval] = useState<Interval>('60')
  const [rightTab, setRightTab] = useState<'order' | 'positions' | 'orders'>('order')

  return (
    <div className="flex flex-col h-[calc(100vh-56px)] bg-[#0a0a0a]">
      {/* ── Pair + Interval Bar ── */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[#1e1e1e] bg-[#0d0d0d] flex-wrap">
        {CRYPTO_PAIRS.map(pair => (
          <button
            key={pair.tv}
            onClick={() => setActivePair(pair.tv)}
            className="text-xs px-2.5 py-1 rounded font-mono font-bold transition-all"
            style={{
              background: activePair === pair.tv ? pair.color + '22' : '#111',
              color: activePair === pair.tv ? pair.color : '#555',
              border: `1px solid ${activePair === pair.tv ? pair.color + '66' : '#1e1e1e'}`,
            }}
          >
            {pair.label}
          </button>
        ))}

        <div className="h-4 w-px bg-[#1e1e1e] mx-1" />

        {INTERVALS.map(iv => (
          <button
            key={iv.value}
            onClick={() => setInterval(iv.value)}
            className="text-xs px-2 py-1 rounded transition-all"
            style={{ background: interval === iv.value ? '#f5a623' : 'transparent', color: interval === iv.value ? '#000' : '#555' }}
          >
            {iv.label}
          </button>
        ))}
      </div>

      {/* ── Main Layout ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chart */}
        <div className="flex flex-col flex-1 min-w-0">
          <div className="flex-1">
            <TVAdvancedChart
              symbol={activePair}
              interval={interval}
              height={undefined as any}
              studies={CRYPTO_STUDIES}
              watchlist={CRYPTO_WATCHLIST}
              showWatchlist={true}
            />
          </div>
          <ArbScanner />
        </div>

        {/* Right Panel */}
        <div className="w-64 flex-shrink-0 flex flex-col bg-[#0d0d0d] border-l border-[#1e1e1e]">
          <div className="flex border-b border-[#1e1e1e]">
            {([
              { key: 'order',     label: 'Order' },
              { key: 'positions', label: 'Positions' },
              { key: 'orders',    label: 'History' },
            ] as { key: typeof rightTab; label: string }[]).map(tab => (
              <button
                key={tab.key}
                onClick={() => setRightTab(tab.key)}
                className="flex-1 py-2 text-[10px] font-medium uppercase tracking-wider transition-colors"
                style={{
                  color: rightTab === tab.key ? '#f7931a' : '#444',
                  borderBottom: rightTab === tab.key ? '2px solid #f7931a' : '2px solid transparent',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="flex-1 overflow-y-auto">
            {rightTab === 'order' && (
              <AdvancedOrderForm
                defaultSymbol={CRYPTO_PAIRS.find(p => p.tv === activePair)?.alpaca ?? activePair.replace('COINBASE:', '').replace('/', '')}
                onSuccess={() => setRightTab('positions')}
              />
            )}
            {rightTab === 'positions' && <CryptoPositions />}
            {rightTab === 'orders'    && <CryptoOrders />}
          </div>
        </div>
      </div>
    </div>
  )
}
