import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { submitOrder } from '../../api/orders'

interface Props {
  symbol?: string
  accountId?: string
}

export default function OrderForm({ symbol = 'AAPL', accountId = '' }: Props) {
  const qc = useQueryClient()
  const [form, setForm] = useState({
    symbol,
    side: 'buy',
    order_type: 'market',
    quantity: 1,
    limit_price: '',
    execution_algo: 'auto',
  })

  const mutation = useMutation({
    mutationFn: () => submitOrder({ ...form, account_id: accountId, limit_price: form.limit_price ? parseFloat(form.limit_price) : null }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['orders'] }),
  })

  const field = (key: string, value: any) => setForm(f => ({ ...f, [key]: value }))

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
      <h3 className="text-[#f5a623] font-semibold text-xs uppercase tracking-wider">Order Entry</h3>

      <input className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs" value={form.symbol} onChange={e => field('symbol', e.target.value)} placeholder="Symbol" />

      <div className="grid grid-cols-2 gap-2">
        {['buy', 'sell'].map(s => (
          <button key={s} onClick={() => field('side', s)} className={`py-1.5 rounded text-xs font-semibold transition-colors ${form.side === s ? (s === 'buy' ? 'bg-[#00c853] text-black' : 'bg-[#ff1744] text-white') : 'bg-[#1e1e1e] text-[#888888]'}`}>
            {s.toUpperCase()}
          </button>
        ))}
      </div>

      <select className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs" value={form.order_type} onChange={e => field('order_type', e.target.value)}>
        {['market', 'limit', 'stop'].map(t => <option key={t}>{t}</option>)}
      </select>

      <input type="number" className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs" value={form.quantity} onChange={e => field('quantity', parseFloat(e.target.value))} placeholder="Quantity" />

      {form.order_type === 'limit' && (
        <input type="number" className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs" value={form.limit_price} onChange={e => field('limit_price', e.target.value)} placeholder="Limit Price" />
      )}

      <select className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1.5 text-xs" value={form.execution_algo} onChange={e => field('execution_algo', e.target.value)}>
        {['auto', 'market', 'limit_first', 'twap'].map(a => <option key={a}>{a}</option>)}
      </select>

      <button onClick={() => mutation.mutate()} disabled={mutation.isPending} className="w-full bg-[#f5a623] text-black font-bold py-2 rounded text-xs hover:bg-[#e09520] transition-colors disabled:opacity-50">
        {mutation.isPending ? 'Submitting...' : 'SUBMIT ORDER'}
      </button>

      {mutation.isError && <p className="text-[#ff1744] text-xs">{String(mutation.error)}</p>}
      {mutation.isSuccess && <p className="text-[#00c853] text-xs">Order submitted!</p>}
    </div>
  )
}
