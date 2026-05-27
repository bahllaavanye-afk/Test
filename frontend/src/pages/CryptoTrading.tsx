import { useState } from 'react'
import MockCandlestickChart from '../components/charts/MockCandlestickChart'
import OrderForm from '../components/trading/OrderForm'

export default function CryptoTrading() {
  const [pair, setPair] = useState('BINANCE:BTCUSDT')
  const PAIRS = ['BINANCE:BTCUSDT', 'BINANCE:ETHUSDT', 'BINANCE:BNBUSDT', 'BINANCE:SOLUSDT']

  return (
    <div className="flex gap-4 h-[calc(100vh-100px)]">
      <div className="flex-1 flex flex-col gap-3">
        <div className="flex gap-2 items-center">
          {PAIRS.map(p => (
            <button key={p} onClick={() => setPair(p)}
              className={`text-xs px-2 py-1 rounded ${pair === p ? 'bg-[#f5a623] text-black' : 'bg-[#1e1e1e] text-[#888888]'}`}>
              {p.replace('BINANCE:', '')}
            </button>
          ))}
        </div>
        <div className="flex-1">
          <MockCandlestickChart symbol={pair} height={520} />
        </div>
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
          <h3 className="text-xs text-[#888888] uppercase mb-2">Triangular Arb Opportunities</h3>
          <p className="text-xs text-[#888888]">Strategy monitors BTC→ETH→USDT→BTC mismatches automatically. Signals appear in alerts when spread &gt;0.15%.</p>
        </div>
      </div>
      <div className="w-64">
        <OrderForm symbol={pair.split(':')[1] || pair} />
      </div>
    </div>
  )
}
