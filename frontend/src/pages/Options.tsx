import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ──────────────────────────────────────────────────────────────────

interface OptionContract {
  symbol: string
  underlying_symbol: string
  expiration_date: string
  strike_price: number
  option_type: 'call' | 'put'
  bid: number | null
  ask: number | null
  mid: number | null
  last: number | null
  volume: number | null
  open_interest: number | null
  implied_volatility: number | null
  delta: number | null
  gamma: number | null
  theta: number | null
  vega: number | null
  rho: number | null
}

interface ExpirationsResponse {
  underlying: string
  expirations: string[]
}

// ── Constants ──────────────────────────────────────────────────────────────

const QUICK_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ', 'TSLA']

const STRATEGIES = [
  { label: 'Covered Call', side: 'sell' as const, type: 'call' as const, note: 'Sell OTM call against long position' },
  { label: 'Cash Secured Put', side: 'sell' as const, type: 'put' as const, note: 'Sell OTM put to enter long at discount' },
  { label: 'Iron Condor', side: 'sell' as const, type: null, note: 'Sell OTM put spread + call spread' },
  { label: 'Long Call', side: 'buy' as const, type: 'call' as const, note: 'Buy ITM call for leveraged upside' },
  { label: 'Long Put', side: 'buy' as const, type: 'put' as const, note: 'Buy OTM put for downside protection' },
]

// ── Helpers ────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, decimals = 2): string {
  if (v == null) return '—'
  return v.toFixed(decimals)
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  return (v * 100).toFixed(1) + '%'
}

function dte(expiry: string): number {
  return Math.max(0, Math.round((new Date(expiry).getTime() - Date.now()) / 86_400_000))
}

function maxLossLabel(side: 'buy' | 'sell', type: 'call' | 'put', mid: number | null, strike: number): string {
  if (mid == null) return '—'
  const premium = mid * 100
  if (side === 'buy' && type === 'call') return `$${premium.toFixed(2)} (premium)`
  if (side === 'buy' && type === 'put') return `$${premium.toFixed(2)} (premium)`
  if (side === 'sell' && type === 'call') return 'Unlimited'
  if (side === 'sell' && type === 'put') return `$${(strike * 100 - premium).toFixed(2)}`
  return '—'
}

function maxGainLabel(side: 'buy' | 'sell', type: 'call' | 'put', mid: number | null, strike: number): string {
  if (mid == null) return '—'
  const premium = mid * 100
  if (side === 'buy' && type === 'call') return 'Unlimited'
  if (side === 'buy' && type === 'put') return `$${(strike * 100 - premium).toFixed(2)}`
  if (side === 'sell' && type === 'call') return `$${premium.toFixed(2)}`
  if (side === 'sell' && type === 'put') return `$${premium.toFixed(2)}`
  return '—'
}

// ── Sub-components ─────────────────────────────────────────────────────────

function SkeletonRows({ count = 8 }: { count?: number }) {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <tr key={i} className="animate-pulse">
          {Array.from({ length: 15 }).map((_, j) => (
            <td key={j} className="px-2 py-1.5">
              <div className="h-3 bg-[#1e1e1e] rounded w-full" />
            </td>
          ))}
        </tr>
      ))}
    </>
  )
}

function GreekCard({ contract }: { contract: OptionContract }) {
  const d = dte(contract.expiration_date)
  const spread = contract.ask != null && contract.bid != null ? contract.ask - contract.bid : null
  return (
    <div className="bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-3 grid grid-cols-4 gap-2">
      {[
        { label: 'Δ Delta', value: fmt(contract.delta, 3) },
        { label: 'Γ Gamma', value: fmt(contract.gamma, 4) },
        { label: 'Θ Theta', value: fmt(contract.theta, 3) },
        { label: 'ν Vega', value: fmt(contract.vega, 3) },
        { label: 'IV', value: fmtPct(contract.implied_volatility) },
        { label: 'DTE', value: `${d}d` },
        { label: 'Spread', value: spread != null ? `$${spread.toFixed(2)}` : '—' },
        { label: 'Mid', value: contract.mid != null ? `$${contract.mid.toFixed(2)}` : '—' },
      ].map(({ label, value }) => (
        <div key={label} className="text-center">
          <div className="text-[10px] text-[#888888]">{label}</div>
          <div className="text-xs font-mono font-semibold text-white">{value}</div>
        </div>
      ))}
    </div>
  )
}

interface OrderPanelProps {
  contract: OptionContract
  onClose: () => void
}

function OrderPanel({ contract, onClose }: OrderPanelProps) {
  const [side, setSide] = useState<'buy' | 'sell'>('buy')
  const [orderType, setOrderType] = useState<'market' | 'limit'>('limit')
  const [qty, setQty] = useState(1)
  const [limitPrice, setLimitPrice] = useState<string>(contract.mid != null ? contract.mid.toFixed(2) : '')
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  const contractVal = parseFloat(limitPrice || '0') * 100

  async function handleSubmit() {
    setSubmitting(true)
    setResult(null)
    try {
      // Orders endpoint expects account_id — we'll skip account selection for now
      // and just show a confirmation. Full integration would require account_id.
      await new Promise(r => setTimeout(r, 300))
      setResult(`Order preview: ${side.toUpperCase()} ${qty}x ${contract.symbol} @ ${orderType === 'market' ? 'MKT' : '$' + limitPrice}`)
    } catch {
      setResult('Order submission failed.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="font-mono text-xs text-[#888888]">{contract.symbol}</span>
          <div className="text-sm font-semibold text-white">
            ${contract.strike_price} {contract.option_type.toUpperCase()} · {contract.expiration_date}
          </div>
        </div>
        <button onClick={onClose} className="text-[#888888] hover:text-white text-lg leading-none">&times;</button>
      </div>

      <div className="grid grid-cols-3 gap-1 text-xs">
        <div className="bg-[#0a0a0a] rounded p-1.5 text-center">
          <div className="text-[#888888]">Bid</div>
          <div className="text-[#ff1744] font-mono">{contract.bid != null ? `$${contract.bid.toFixed(2)}` : '—'}</div>
        </div>
        <div className="bg-[#0a0a0a] rounded p-1.5 text-center">
          <div className="text-[#888888]">Mid</div>
          <div className="text-white font-mono">{contract.mid != null ? `$${contract.mid.toFixed(2)}` : '—'}</div>
        </div>
        <div className="bg-[#0a0a0a] rounded p-1.5 text-center">
          <div className="text-[#888888]">Ask</div>
          <div className="text-[#00c853] font-mono">{contract.ask != null ? `$${contract.ask.toFixed(2)}` : '—'}</div>
        </div>
      </div>

      {/* Buy / Sell toggle */}
      <div className="flex gap-1">
        {(['buy', 'sell'] as const).map(s => (
          <button
            key={s}
            onClick={() => setSide(s)}
            className={`flex-1 py-1.5 rounded text-xs font-semibold transition-colors ${
              side === s
                ? s === 'buy' ? 'bg-[#00c853] text-black' : 'bg-[#ff1744] text-white'
                : 'bg-[#1e1e1e] text-[#888888]'
            }`}
          >
            {s.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Qty */}
      <div>
        <label className="text-[10px] text-[#888888] uppercase">Contracts (×100 shares)</label>
        <input
          type="number"
          min={1}
          value={qty}
          onChange={e => setQty(Math.max(1, parseInt(e.target.value) || 1))}
          className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1 text-xs font-mono mt-0.5"
        />
      </div>

      {/* Order type */}
      <div>
        <label className="text-[10px] text-[#888888] uppercase">Order Type</label>
        <div className="flex gap-1 mt-0.5">
          {(['limit', 'market'] as const).map(ot => (
            <button
              key={ot}
              onClick={() => setOrderType(ot)}
              className={`flex-1 py-1 rounded text-xs transition-colors ${
                orderType === ot ? 'bg-[#f5a623] text-black font-semibold' : 'bg-[#1e1e1e] text-[#888888]'
              }`}
            >
              {ot.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {orderType === 'limit' && (
        <div>
          <label className="text-[10px] text-[#888888] uppercase">Limit Price (per share)</label>
          <input
            type="number"
            step="0.01"
            value={limitPrice}
            onChange={e => setLimitPrice(e.target.value)}
            className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1 text-xs font-mono mt-0.5"
          />
        </div>
      )}

      {/* Max loss / Max gain */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-[#0a0a0a] rounded p-1.5">
          <div className="text-[#888888] text-[10px]">Max Loss</div>
          <div className="text-[#ff1744] font-mono">{maxLossLabel(side, contract.option_type, contract.mid, contract.strike_price)}</div>
        </div>
        <div className="bg-[#0a0a0a] rounded p-1.5">
          <div className="text-[#888888] text-[10px]">Max Gain</div>
          <div className="text-[#00c853] font-mono">{maxGainLabel(side, contract.option_type, contract.mid, contract.strike_price)}</div>
        </div>
      </div>

      {contractVal > 0 && (
        <div className="text-[10px] text-[#888888]">
          Total premium: <span className="text-white font-mono">${(contractVal * qty).toFixed(2)}</span> ({qty} contract{qty > 1 ? 's' : ''} × 100 shares)
        </div>
      )}

      {result && (
        <div className="text-xs text-[#f5a623] bg-[#f5a623]/10 rounded p-2">{result}</div>
      )}

      <button
        onClick={handleSubmit}
        disabled={submitting}
        className="w-full py-2 bg-[#f5a623] text-black text-xs font-semibold rounded hover:bg-[#f5a623]/90 disabled:opacity-50 transition-colors"
      >
        {submitting ? 'Submitting…' : `${side.toUpperCase()} ${qty} Contract${qty > 1 ? 's' : ''}`}
      </button>
    </div>
  )
}

// ── Main Page ──────────────────────────────────────────────────────────────

export default function Options() {
  const [underlying, setUnderlying] = useState('AAPL')
  const [customInput, setCustomInput] = useState('')
  const [selectedExpiry, setSelectedExpiry] = useState<string | null>(null)
  const [selectedContract, setSelectedContract] = useState<OptionContract | null>(null)
  const [straddleStrike, setStraddleStrike] = useState<string>('')
  const [straddleExpiry, setStraddleExpiry] = useState<string>('')

  // Expirations
  const { data: expData } = useQuery<ExpirationsResponse>({
    queryKey: ['options-expirations', underlying],
    queryFn: () => api.get(`/options/expirations/${underlying}`).then(r => r.data),
    staleTime: 60_000,
  })
  const expirations = expData?.expirations ?? []

  // Auto-select first expiry when symbol changes
  const effectiveExpiry = selectedExpiry ?? expirations[0] ?? null

  // Chain
  const { data: chain = [], isLoading: chainLoading, error: chainError } = useQuery<OptionContract[]>({
    queryKey: ['options-chain', underlying, effectiveExpiry],
    queryFn: () =>
      api.get(`/options/chain/${underlying}`, {
        params: effectiveExpiry ? { expiration: effectiveExpiry } : {},
      }).then(r => r.data),
    enabled: !!underlying,
    staleTime: 30_000,
  })

  // Spot price from market data
  const { data: quote } = useQuery<{ last: number | null }>({
    queryKey: ['quote', underlying],
    queryFn: () => api.get(`/market-data/quote/${underlying}`).then(r => r.data),
    staleTime: 10_000,
    refetchInterval: 15_000,
  })
  const spotPrice = quote?.last ?? null

  // Separate calls and puts, sorted by strike
  const calls = useMemo(
    () => chain.filter(c => c.option_type === 'call').sort((a, b) => a.strike_price - b.strike_price),
    [chain]
  )
  const puts = useMemo(
    () => chain.filter(c => c.option_type === 'put').sort((a, b) => a.strike_price - b.strike_price),
    [chain]
  )

  // Unique strikes across both sides
  const strikes = useMemo(() => {
    const s = new Set([...calls.map(c => c.strike_price), ...puts.map(c => c.strike_price)])
    return Array.from(s).sort((a, b) => a - b)
  }, [calls, puts])

  // Build lookup maps by strike
  const callByStrike = useMemo(() => new Map(calls.map(c => [c.strike_price, c])), [calls])
  const putByStrike = useMemo(() => new Map(puts.map(c => [c.strike_price, c])), [puts])

  function isATM(strike: number): boolean {
    if (spotPrice == null) return false
    return Math.abs(strike - spotPrice) / spotPrice < 0.01
  }

  function isITMCall(strike: number): boolean {
    return spotPrice != null && strike < spotPrice
  }

  function isITMPut(strike: number): boolean {
    return spotPrice != null && strike > spotPrice
  }

  // Straddle calculator
  const straddleStrikeNum = parseFloat(straddleStrike)
  const straddleCallContract = !isNaN(straddleStrikeNum)
    ? chain.find(c => c.option_type === 'call' && c.strike_price === straddleStrikeNum && (!straddleExpiry || c.expiration_date === straddleExpiry))
    : null
  const straddlePutContract = !isNaN(straddleStrikeNum)
    ? chain.find(c => c.option_type === 'put' && c.strike_price === straddleStrikeNum && (!straddleExpiry || c.expiration_date === straddleExpiry))
    : null
  const straddleCost = (straddleCallContract?.mid ?? 0) + (straddlePutContract?.mid ?? 0)
  const straddleBreakevenPct = spotPrice && straddleCost > 0 ? (straddleCost / spotPrice) * 100 : null

  function applyStrategy(strat: typeof STRATEGIES[0]) {
    setSelectedContract(null)
    // If a specific type is requested, find a good candidate from current chain
    if (strat.type) {
      const filtered = chain.filter(c => c.option_type === strat.type)
      if (filtered.length > 0) {
        const nearest = filtered.reduce((best, c) => {
          const d = Math.abs((c.delta ?? 0) - (strat.side === 'buy' ? 0.70 : 0.30))
          const bd = Math.abs((best.delta ?? 0) - (strat.side === 'buy' ? 0.70 : 0.30))
          return d < bd ? c : best
        })
        setSelectedContract(nearest)
      }
    }
  }

  return (
    <div className="space-y-4 min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-white">Options Chain</h1>
        {spotPrice != null && (
          <div className="text-sm font-mono text-[#888888]">
            <span className="text-white font-semibold">{underlying}</span>
            {' '}<span className="text-[#00c853]">${spotPrice.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Symbol selector */}
      <div className="flex flex-wrap gap-2 items-center">
        {QUICK_SYMBOLS.map(s => (
          <button
            key={s}
            onClick={() => { setUnderlying(s); setSelectedExpiry(null); setSelectedContract(null) }}
            className={`px-3 py-1 rounded text-xs font-mono font-semibold transition-colors ${
              underlying === s
                ? 'bg-[#f5a623] text-black'
                : 'bg-[#111111] border border-[#1e1e1e] text-[#888888] hover:text-white'
            }`}
          >
            {s}
          </button>
        ))}
        <div className="flex gap-1">
          <input
            value={customInput}
            onChange={e => setCustomInput(e.target.value.toUpperCase())}
            onKeyDown={e => {
              if (e.key === 'Enter' && customInput.trim()) {
                setUnderlying(customInput.trim())
                setSelectedExpiry(null)
                setSelectedContract(null)
              }
            }}
            placeholder="Custom…"
            className="bg-[#111111] border border-[#1e1e1e] rounded px-2 py-1 text-xs font-mono w-20"
          />
          <button
            onClick={() => { if (customInput.trim()) { setUnderlying(customInput.trim()); setSelectedExpiry(null); setSelectedContract(null) } }}
            className="bg-[#1e1e1e] hover:bg-[#2a2a2a] text-[#888888] hover:text-white px-2 py-1 rounded text-xs"
          >
            Go
          </button>
        </div>
      </div>

      {/* Strategy quick-select */}
      <div className="flex flex-wrap gap-1.5">
        {STRATEGIES.map(strat => (
          <button
            key={strat.label}
            onClick={() => applyStrategy(strat)}
            title={strat.note}
            className="px-2.5 py-1 bg-[#111111] border border-[#1e1e1e] rounded text-[10px] text-[#888888] hover:text-[#f5a623] hover:border-[#f5a623]/40 transition-colors"
          >
            {strat.label}
          </button>
        ))}
      </div>

      {/* Expiration tabs */}
      {expirations.length > 0 && (
        <div className="flex gap-1 flex-wrap">
          {expirations.slice(0, 12).map(exp => (
            <button
              key={exp}
              onClick={() => { setSelectedExpiry(exp); setSelectedContract(null) }}
              className={`px-2.5 py-1 rounded text-[10px] font-mono transition-colors ${
                effectiveExpiry === exp
                  ? 'bg-[#f5a623]/20 text-[#f5a623] border border-[#f5a623]/40'
                  : 'bg-[#111111] border border-[#1e1e1e] text-[#888888] hover:text-white'
              }`}
            >
              {exp} <span className="text-[#555]">({dte(exp)}d)</span>
            </button>
          ))}
        </div>
      )}

      {/* Main content */}
      <div className="flex gap-4">
        {/* Chain Table */}
        <div className="flex-1 overflow-auto">
          {chainError ? (
            <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-8 text-center">
              <div className="text-[#888888] text-sm">Options data unavailable — check Alpaca subscription level</div>
              <div className="text-[10px] text-[#555] mt-1">Alpaca requires options account approval to access options market data.</div>
            </div>
          ) : (
            <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
              <table className="w-full text-[11px] font-mono">
                <thead>
                  <tr className="border-b border-[#1e1e1e]">
                    {/* Calls side */}
                    <th className="text-[#00c853] text-center py-2 px-2 bg-[#001a00]/30" colSpan={7}>CALLS</th>
                    {/* Strike */}
                    <th className="text-center py-2 px-2 text-[#f5a623] bg-[#111111]">STRIKE</th>
                    {/* Puts side */}
                    <th className="text-[#ff1744] text-center py-2 px-2 bg-[#1a0000]/30" colSpan={7}>PUTS</th>
                  </tr>
                  <tr className="text-[#555] border-b border-[#1e1e1e] text-[10px]">
                    {['Bid', 'Ask', 'IV%', 'Delta', 'Volume', 'OI', 'Last'].map(h => (
                      <th key={`c-${h}`} className="py-1 px-2 text-right font-normal bg-[#001a00]/10">{h}</th>
                    ))}
                    <th className="py-1 px-2 text-center font-normal"></th>
                    {['Bid', 'Ask', 'IV%', 'Delta', 'Volume', 'OI', 'Last'].map(h => (
                      <th key={`p-${h}`} className="py-1 px-2 text-right font-normal bg-[#1a0000]/10">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {chainLoading ? (
                    <SkeletonRows count={10} />
                  ) : strikes.length === 0 ? (
                    <tr>
                      <td colSpan={15} className="text-center text-[#555] py-8">
                        No contracts found for selected parameters.
                      </td>
                    </tr>
                  ) : (
                    strikes.map(strike => {
                      const call = callByStrike.get(strike)
                      const put = putByStrike.get(strike)
                      const atm = isATM(strike)
                      const itmCall = isITMCall(strike)
                      const itmPut = isITMPut(strike)
                      return (
                        <tr
                          key={strike}
                          className={`border-b border-[#1e1e1e]/50 ${atm ? 'border-[#f5a623]/30 border-b-2' : ''}`}
                        >
                          {/* Call cells */}
                          {call ? (
                            <td
                              colSpan={7}
                              className={`cursor-pointer ${itmCall ? 'bg-[#001a00]' : ''} ${selectedContract?.symbol === call.symbol ? 'ring-1 ring-inset ring-[#f5a623]' : ''} hover:bg-[#001a00]/80`}
                              onClick={() => setSelectedContract(selectedContract?.symbol === call.symbol ? null : call)}
                            >
                              <div className="grid grid-cols-7 divide-x divide-[#1e1e1e]/30">
                                <div className="px-2 py-1.5 text-right text-[#ff1744]">{fmt(call.bid)}</div>
                                <div className="px-2 py-1.5 text-right text-[#00c853]">{fmt(call.ask)}</div>
                                <div className="px-2 py-1.5 text-right text-[#888]">{fmtPct(call.implied_volatility)}</div>
                                <div className="px-2 py-1.5 text-right text-[#888]">{fmt(call.delta, 3)}</div>
                                <div className="px-2 py-1.5 text-right text-[#888]">{call.volume ?? '—'}</div>
                                <div className="px-2 py-1.5 text-right text-[#888]">{call.open_interest ?? '—'}</div>
                                <div className="px-2 py-1.5 text-right">{fmt(call.last)}</div>
                              </div>
                            </td>
                          ) : (
                            <td colSpan={7} className="px-2 py-1.5 text-center text-[#1e1e1e]">—</td>
                          )}

                          {/* Strike center */}
                          <td className={`px-3 py-1.5 text-center font-semibold ${atm ? 'text-[#f5a623] bg-[#f5a623]/5' : 'text-[#888]'}`}>
                            {strike}
                          </td>

                          {/* Put cells */}
                          {put ? (
                            <td
                              colSpan={7}
                              className={`cursor-pointer ${itmPut ? 'bg-[#1a0000]' : ''} ${selectedContract?.symbol === put.symbol ? 'ring-1 ring-inset ring-[#f5a623]' : ''} hover:bg-[#1a0000]/80`}
                              onClick={() => setSelectedContract(selectedContract?.symbol === put.symbol ? null : put)}
                            >
                              <div className="grid grid-cols-7 divide-x divide-[#1e1e1e]/30">
                                <div className="px-2 py-1.5 text-right text-[#ff1744]">{fmt(put.bid)}</div>
                                <div className="px-2 py-1.5 text-right text-[#00c853]">{fmt(put.ask)}</div>
                                <div className="px-2 py-1.5 text-right text-[#888]">{fmtPct(put.implied_volatility)}</div>
                                <div className="px-2 py-1.5 text-right text-[#888]">{fmt(put.delta, 3)}</div>
                                <div className="px-2 py-1.5 text-right text-[#888]">{put.volume ?? '—'}</div>
                                <div className="px-2 py-1.5 text-right text-[#888]">{put.open_interest ?? '—'}</div>
                                <div className="px-2 py-1.5 text-right">{fmt(put.last)}</div>
                              </div>
                            </td>
                          ) : (
                            <td colSpan={7} className="px-2 py-1.5 text-center text-[#1e1e1e]">—</td>
                          )}
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Right panel: Greeks + Order */}
        {selectedContract && (
          <div className="w-72 flex-none space-y-3">
            <GreekCard contract={selectedContract} />
            <OrderPanel contract={selectedContract} onClose={() => setSelectedContract(null)} />
          </div>
        )}
      </div>

      {/* Straddle Calculator */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h3 className="text-xs text-[#888888] uppercase mb-3">Straddle Calculator</h3>
        <div className="flex gap-3 items-end flex-wrap">
          <div>
            <label className="text-[10px] text-[#555] uppercase block mb-1">Strike</label>
            <select
              value={straddleStrike}
              onChange={e => setStraddleStrike(e.target.value)}
              className="bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1 text-xs font-mono"
            >
              <option value="">Select strike…</option>
              {strikes.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-[#555] uppercase block mb-1">Expiry</label>
            <select
              value={straddleExpiry}
              onChange={e => setStraddleExpiry(e.target.value)}
              className="bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1 text-xs font-mono"
            >
              <option value="">Any</option>
              {expirations.map(e => <option key={e} value={e}>{e}</option>)}
            </select>
          </div>

          {straddleCost > 0 ? (
            <div className="flex gap-4 items-center">
              <div>
                <div className="text-[10px] text-[#888888]">Straddle Cost</div>
                <div className="font-mono text-white font-semibold">${(straddleCost * 100).toFixed(2)} <span className="text-[#555] font-normal">per contract</span></div>
              </div>
              {straddleBreakevenPct != null && (
                <div>
                  <div className="text-[10px] text-[#888888]">Breakeven Move</div>
                  <div className="font-mono text-[#f5a623] font-semibold">±{straddleBreakevenPct.toFixed(2)}%</div>
                </div>
              )}
              <div className="text-[10px] text-[#555] max-w-[140px]">
                Call mid: {straddleCallContract?.mid != null ? `$${straddleCallContract.mid.toFixed(2)}` : '—'}{' '}
                + Put mid: {straddlePutContract?.mid != null ? `$${straddlePutContract.mid.toFixed(2)}` : '—'}
              </div>
            </div>
          ) : straddleStrike ? (
            <div className="text-xs text-[#555]">No matching contracts for this strike/expiry.</div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
