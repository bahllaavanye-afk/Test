import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import TVAdvancedChart from '../components/charts/TVAdvancedChart'
import OrderForm from '../components/trading/OrderForm'
import api from '../api/client'

export default function EquityTrading() {
  const [symbol, setSymbol] = useState('NASDAQ:AAPL')
  const { data: orders, isLoading } = useQuery({
    queryKey: ['orders'],
    queryFn: () => api.get('/orders/').then(r => r.data),
    refetchInterval: 5_000,
  })
  const { data: positions } = useQuery({
    queryKey: ['positions'],
    queryFn: () => api.get('/positions/').then(r => r.data),
    refetchInterval: 10_000,
  })

  const SYMBOLS = ['NASDAQ:AAPL', 'NASDAQ:MSFT', 'NASDAQ:GOOGL', 'NASDAQ:AMZN', 'NYSE:SPY', 'NYSE:QQQ']

  return (
    <div className="flex gap-4 h-[calc(100vh-100px)]">
      <div className="flex-1 flex flex-col gap-3">
        <div className="flex gap-2 items-center">
          {SYMBOLS.map(s => (
            <button key={s} onClick={() => setSymbol(s)}
              className={`text-xs px-2 py-1 rounded transition-colors ${symbol === s ? 'bg-[#f5a623] text-black' : 'bg-[#1e1e1e] text-[#888888] hover:text-white'}`}>
              {s.split(':')[1]}
            </button>
          ))}
          <input value={symbol} onChange={e => setSymbol(e.target.value)}
            className="bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1 text-xs w-36 ml-2"
            placeholder="Custom symbol" />
        </div>
        <div className="flex-1">
          <TVAdvancedChart symbol={symbol} height={480} />
        </div>
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
          <h3 className="text-xs text-[#888888] uppercase mb-2">Open Positions</h3>
          <div className="grid grid-cols-5 text-xs text-[#888888] mb-1 px-1">
            {['Symbol', 'Qty', 'Avg Entry', 'P&L', 'Side'].map(h => <span key={h}>{h}</span>)}
          </div>
          <div className="space-y-1 max-h-28 overflow-y-auto">
            {positions?.map((p: any) => (
              <div key={p.id} className="grid grid-cols-5 text-xs px-1 py-1 bg-[#0a0a0a] rounded">
                <span className="font-mono">{p.symbol}</span>
                <span>{p.quantity}</span>
                <span>${p.avg_cost?.toFixed(2)}</span>
                <span className={p.unrealized_pnl >= 0 ? 'text-[#00c853]' : 'text-[#ff1744]'}>
                  ${p.unrealized_pnl?.toFixed(2) ?? '—'}
                </span>
                <span className={p.side === 'long' ? 'text-[#00c853]' : 'text-[#ff1744]'}>{p.side}</span>
              </div>
            ))}
            {!positions?.length && <p className="text-[#888888] text-xs px-1">No open positions</p>}
          </div>
        </div>
      </div>

      <div className="w-64 flex flex-col gap-3">
        <OrderForm symbol={symbol.split(':')[1] || symbol} />
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3 flex-1">
          <h3 className="text-xs text-[#888888] uppercase mb-2">Recent Orders</h3>
          <div className="space-y-1.5 overflow-y-auto max-h-48">
            {isLoading && <p className="text-xs text-[#888888]">Loading...</p>}
            {orders?.slice(0, 15).map((o: any) => (
              <div key={o.id} className="text-xs p-1.5 bg-[#0a0a0a] rounded flex justify-between">
                <span className={o.side === 'buy' ? 'text-[#00c853]' : 'text-[#ff1744]'}>{o.side.toUpperCase()}</span>
                <span className="font-mono">{o.symbol}</span>
                <span className="text-[#888888]">{o.status}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
