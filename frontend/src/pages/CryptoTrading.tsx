import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import TVAdvancedChart from '../components/charts/TVAdvancedChart'
import api from '../api/client'

type Interval = '1' | '5' | '15' | '60' | '240' | 'D'
type Side = 'buy' | 'sell'
type OrdType = 'market' | 'limit'

const CRYPTO_PAIRS = [
  { tv: 'BINANCE:BTCUSDT',  label: 'BTC/USDT',  color: '#f7931a' },
  { tv: 'BINANCE:ETHUSDT',  label: 'ETH/USDT',  color: '#627eea' },
  { tv: 'BINANCE:SOLUSDT',  label: 'SOL/USDT',  color: '#9945ff' },
  { tv: 'BINANCE:BNBUSDT',  label: 'BNB/USDT',  color: '#f3ba2f' },
  { tv: 'COINBASE:BTCUSD',  label: 'BTC/USD',   color: '#0052ff' },
  { tv: 'COINBASE:ETHUSD',  label: 'ETH/USD',   color: '#627eea' },
  { tv: 'BINANCE:XRPUSDT',  label: 'XRP/USDT',  color: '#346aa9' },
  { tv: 'BINANCE:ADAUSDT',  label: 'ADA/USDT',  color: '#0033ad' },
]

const CRYPTO_WATCHLIST = CRYPTO_PAIRS.map(p => p.tv)
  .concat(['BINANCE:DOGEUSDT', 'BINANCE:AVAXUSDT', 'BINANCE:DOTUSDT', 'BINANCE:MATICUSDT'])

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
          <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">Triangular Arb Scanner</span>
        </div>
        <span className="text-[10px] text-[#333]">BTC→ETH→USDT→BTC</span>
      </div>
      <div className="px-3 pb-2">
        {isLoading ? (
          <div className="h-6 bg-[#1a1a1a] rounded animate-pulse" />
        ) : ops.length > 0 ? (
          <div className="space-y-1">
            {ops.map((op: any, i: number) => (
              <div key={i} className="flex items-center justify-between text-xs bg-[#111] rounded px-2 py-1.5">
                <span className="font-mono text-[#e8e8e8]">{op.path ?? 'BTC→ETH→USDT→BTC'}</span>
                <span className="font-bold" style={{ color: (op.spread ?? 0) > 0 ? '#00c853' : '#ff1744' }}>
                  {op.spread != null ? `${(op.spread * 100).toFixed(3)}%` : '—'}
                </span>
                <span className="text-[10px] text-[#444]">{op.status ?? 'monitoring'}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[10px] text-[#333] italic">
            Strategy monitors BTC/ETH/USDT mismatches automatically. Signals fire when spread &gt; 0.15%.
          </p>
        )}
      </div>
    </div>
  )
}

// ─── Crypto Order Entry ───────────────────────────────────────────────────────
function CryptoOrderEntry({ defaultPair }: { defaultPair: string }) {
  const qc = useQueryClient()
  const sym = defaultPair.replace('BINANCE:', '').replace('COINBASE:', '')
  const [side, setSide] = useState<Side>('buy')
  const [type, setType] = useState<OrdType>('limit')
  const [qty, setQty] = useState('')
  const [limit, setLimit] = useState('')
  const [notional, setNotional] = useState('')
  const [useNotional, setUseNotional] = useState(true)

  const { data: accounts } = useQuery({
    queryKey: ['accounts-crypto'],
    queryFn: () => api.get('/accounts/').then(r => r.data),
    staleTime: 60_000,
  })
  const accts: any[] = Array.isArray(accounts) ? accounts : []
  const [accountId, setAccountId] = useState('')

  const mutation = useMutation({
    mutationFn: () => {
      if (!accountId) throw new Error('No account selected')
      return api.post('/orders/', {
        symbol: sym,
        side,
        order_type: type,
        quantity: useNotional ? undefined : parseFloat(qty) || 0.001,
        notional: useNotional ? parseFloat(notional) : undefined,
        limit_price: type === 'limit' && limit ? parseFloat(limit) : null,
        execution_algo: 'limit_first',
        account_id: accountId,
      }).then(r => r.data)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['orders-crypto'] })
      qc.invalidateQueries({ queryKey: ['positions-crypto'] })
    },
  })

  return (
    <div className="p-3 space-y-2.5">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-sm font-bold font-mono text-[#f5a623]">{sym}</span>
        <span className="text-[10px] text-[#444]">CRYPTO</span>
      </div>

      {accts.length > 0 ? (
        <select
          value={accountId}
          onChange={e => setAccountId(e.target.value)}
          className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none"
        >
          <option value="">Select account…</option>
          {accts.map((a: any) => (
            <option key={a.id} value={a.id}>{a.label}</option>
          ))}
        </select>
      ) : (
        <div className="bg-[#0d0d0d] border border-[#f5a623]/20 rounded px-2 py-1.5">
          <p className="text-[9px] text-[#f5a623]">Connect Binance or Alpaca account in Settings</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-1.5">
        {(['buy', 'sell'] as Side[]).map(s => (
          <button
            key={s}
            onClick={() => setSide(s)}
            className="py-1.5 rounded-lg text-xs font-bold transition-all"
            style={{
              background: side === s ? (s === 'buy' ? '#00c853' : '#ff1744') : '#1a1a1a',
              color: side === s ? (s === 'buy' ? '#000' : '#fff') : '#555',
              border: `1px solid ${side === s ? (s === 'buy' ? '#00c853' : '#ff1744') : '#222'}`,
            }}
          >
            {s === 'buy' ? '▲ BUY' : '▼ SELL'}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-1">
        {(['market', 'limit'] as OrdType[]).map(t => (
          <button
            key={t}
            onClick={() => setType(t)}
            className="py-1 rounded text-[10px] font-medium transition-colors"
            style={{ background: type === t ? '#1e1e1e' : 'transparent', color: type === t ? '#e8e8e8' : '#555' }}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      <div>
        <div className="flex justify-between mb-1">
          <span className="text-[10px] text-[#555] uppercase tracking-wider">{useNotional ? 'Notional ($)' : 'Amount'}</span>
          <button onClick={() => setUseNotional(v => !v)} className="text-[9px] text-[#444] hover:text-[#888]">
            Switch to {useNotional ? 'amount' : 'notional'}
          </button>
        </div>
        <input
          type="number"
          value={useNotional ? notional : qty}
          onChange={e => useNotional ? setNotional(e.target.value) : setQty(e.target.value)}
          className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-[#333]"
          placeholder={useNotional ? '100.00' : '0.001'}
          step={useNotional ? '1' : '0.00001'}
        />
      </div>

      {type === 'limit' && (
        <div>
          <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Limit price (USDT)</p>
          <input
            type="number"
            step="0.01"
            value={limit}
            onChange={e => setLimit(e.target.value)}
            className="w-full bg-[#0a0a0a] border border-[#2196f3]/40 rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none"
            placeholder="0.00"
          />
        </div>
      )}

      <button
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending || !accountId}
        className="w-full py-2.5 rounded-lg text-sm font-bold transition-all active:scale-[0.98]"
        style={{
          background: side === 'buy' ? 'linear-gradient(135deg, #00c853, #00a843)' : 'linear-gradient(135deg, #ff1744, #c62828)',
          color: '#fff',
          opacity: !accountId ? 0.4 : 1,
        }}
      >
        {mutation.isPending ? 'Placing…' : `${side === 'buy' ? '▲ BUY' : '▼ SELL'} ${sym}`}
      </button>

      {mutation.isError && <p className="text-xs text-[#ff1744]">{String((mutation.error as any)?.response?.data?.detail ?? mutation.error)}</p>}
      {mutation.isSuccess && <p className="text-xs text-[#00c853]">Order placed ✓</p>}
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
  const [activePair, setActivePair] = useState('BINANCE:BTCUSDT')
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
            {rightTab === 'order'     && <CryptoOrderEntry defaultPair={activePair} />}
            {rightTab === 'positions' && <CryptoPositions />}
            {rightTab === 'orders'    && <CryptoOrders />}
          </div>
        </div>
      </div>
    </div>
  )
}
