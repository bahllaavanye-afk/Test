import { useState, useEffect, useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────
type Side = 'buy' | 'sell'
type OrdType = 'market' | 'limit' | 'stop' | 'stop_limit'
type TIF = 'gtc' | 'day' | 'ioc' | 'fok'
type Algo = 'auto' | 'market' | 'limit_first' | 'twap' | 'vwap'
type SizeMode = 'shares' | 'notional' | 'pct_equity'

interface Props {
  defaultSymbol?: string
  defaultSide?: Side
  onSuccess?: () => void
}

const TIF_INFO: Record<TIF, string> = {
  gtc:  'Good Till Cancelled — persists until filled or cancelled',
  day:  'Day — cancels at market close if unfilled',
  ioc:  'Immediate or Cancel — fills what it can, cancels rest',
  fok:  'Fill or Kill — fill entirely or cancel',
}

const ALGO_INFO: Record<Algo, { label: string; desc: string; color: string }> = {
  auto:        { label: 'Auto',         desc: 'Smart routing based on size',        color: '#888' },
  limit_first: { label: 'Limit-First',  desc: 'Post limit → market fallback. ~5bps saved', color: '#00c853' },
  twap:        { label: 'TWAP 30min',   desc: 'Slice order over 30 minutes',        color: '#2196f3' },
  vwap:        { label: 'VWAP',         desc: 'Track volume-weighted price',         color: '#9c27b0' },
  market:      { label: 'Market',       desc: 'Immediate fill — widest spread',      color: '#ff1744' },
}

// ─── Risk/Reward Calculator ────────────────────────────────────────────────────
function RiskRewardDisplay({
  side, entry, sl, tp, qty, notional,
}: {
  side: Side; entry: number | null; sl: number | null; tp: number | null;
  qty: number | null; notional: number | null;
}) {
  if (!entry || (!sl && !tp)) return null

  const dir = side === 'buy' ? 1 : -1
  const riskPts  = sl  ? Math.abs(entry - sl)  : null
  const rewardPts = tp ? Math.abs(tp - entry)   : null
  const shares = qty ?? (notional && entry ? notional / entry : null)

  const riskDollars   = riskPts   && shares ? riskPts   * shares : null
  const rewardDollars = rewardPts && shares ? rewardPts * shares : null
  const rrRatio = riskDollars && rewardDollars ? rewardDollars / riskDollars : null

  const slValid = sl != null && (side === 'buy' ? sl < entry : sl > entry)
  const tpValid = tp != null && (side === 'buy' ? tp > entry : tp < entry)

  return (
    <div className="bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-2.5 space-y-1.5">
      <p className="text-[10px] text-[#444] uppercase tracking-wider font-medium">Risk / Reward</p>
      {sl != null && (
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-[#555]">Stop-Loss distance</span>
          <div className="flex items-center gap-2">
            {!slValid && <span className="text-[10px] text-[#ff1744]">⚠ wrong side</span>}
            <span className="text-xs font-mono text-[#ff1744]">
              {riskPts != null ? `$${riskPts.toFixed(2)}` : '—'}
              {riskDollars != null ? ` = -$${riskDollars.toFixed(0)}` : ''}
            </span>
          </div>
        </div>
      )}
      {tp != null && (
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-[#555]">Take-Profit target</span>
          <div className="flex items-center gap-2">
            {!tpValid && <span className="text-[10px] text-[#ff1744]">⚠ wrong side</span>}
            <span className="text-xs font-mono text-[#00c853]">
              {rewardPts != null ? `$${rewardPts.toFixed(2)}` : '—'}
              {rewardDollars != null ? ` = +$${rewardDollars.toFixed(0)}` : ''}
            </span>
          </div>
        </div>
      )}
      {rrRatio != null && (
        <div className="flex justify-between items-center pt-1 border-t border-[#1e1e1e]">
          <span className="text-[10px] text-[#555] font-medium">Risk:Reward ratio</span>
          <span
            className="text-sm font-bold font-mono"
            style={{ color: rrRatio >= 2 ? '#00c853' : rrRatio >= 1 ? '#f5a623' : '#ff1744' }}
          >
            1:{rrRatio.toFixed(2)}
            {rrRatio >= 2 ? ' ✓' : rrRatio < 1 ? ' ✗' : ''}
          </span>
        </div>
      )}
    </div>
  )
}

// ─── Collapsible Section ──────────────────────────────────────────────────────
function Section({ title, children, defaultOpen = false, badge }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean; badge?: string
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-[#1e1e1e] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 bg-[#0d0d0d] hover:bg-[#111] transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-[#888] uppercase tracking-wider">{title}</span>
          {badge && <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#f5a623]/15 text-[#f5a623] font-bold">{badge}</span>}
        </div>
        <span className="text-[#444] text-xs">{open ? '▲' : '▼'}</span>
      </button>
      {open && <div className="p-3 space-y-2.5 bg-[#080808]">{children}</div>}
    </div>
  )
}

// ─── Number Input ──────────────────────────────────────────────────────────────
function NumInput({
  label, value, onChange, placeholder, step = '0.01', color,
  hint, onPctChange, currentPrice,
}: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string; step?: string; color?: string;
  hint?: string; onPctChange?: (pct: string) => void; currentPrice?: number | null;
}) {
  const [pctMode, setPctMode] = useState(false)
  const [pctVal, setPctVal] = useState('')

  function applyPct(pct: string) {
    if (!currentPrice || !pct) return
    const p = parseFloat(pct)
    if (isNaN(p)) return
    onPctChange?.(String((currentPrice * (1 + p / 100)).toFixed(2)))
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-[#555] uppercase tracking-wider">{label}</span>
        {onPctChange && currentPrice && (
          <button onClick={() => setPctMode(m => !m)} className="text-[9px] text-[#444] hover:text-[#888]">
            {pctMode ? '$ price' : '% offset'}
          </button>
        )}
      </div>
      {pctMode && onPctChange && currentPrice ? (
        <div className="flex gap-1">
          <input
            type="number" step="0.1" value={pctVal}
            onChange={e => { setPctVal(e.target.value); applyPct(e.target.value) }}
            className="flex-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-[#333]"
            placeholder="-2.0"
          />
          <span className="text-xs text-[#444] self-center">%</span>
        </div>
      ) : (
        <input
          type="number" step={step} value={value}
          onChange={e => onChange(e.target.value)}
          className="w-full bg-[#0a0a0a] rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none transition-colors"
          style={{ border: `1px solid ${color ? color + '44' : '#1e1e1e'}` }}
          placeholder={placeholder}
        />
      )}
      {hint && <p className="text-[9px] text-[#333] mt-0.5">{hint}</p>}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function AdvancedOrderForm({ defaultSymbol = 'AAPL', defaultSide = 'buy', onSuccess }: Props) {
  const qc = useQueryClient()

  // Core fields
  const [sym, setSym]       = useState(defaultSymbol)
  const [side, setSide]     = useState<Side>(defaultSide)
  const [type, setType]     = useState<OrdType>('limit')
  const [tif, setTif]       = useState<TIF>('gtc')
  const [algo, setAlgo]     = useState<Algo>('limit_first')
  const [accountId, setAccountId] = useState('')

  // Size
  const [sizeMode, setSizeMode] = useState<SizeMode>('shares')
  const [qty, setQty]         = useState('')
  const [notional, setNotional] = useState('')
  const [equityPct, setEquityPct] = useState('')

  // Prices
  const [limitPrice, setLimitPrice]   = useState('')
  const [stopPrice, setStopPrice]     = useState('')
  const [takeProfit, setTakeProfit]   = useState('')
  const [stopLoss, setStopLoss]       = useState('')
  const [trailingPct, setTrailingPct] = useState('')
  const [useTrailing, setUseTrailing] = useState(false)

  // Quote (for % offset helpers)
  const [livePrice, setLivePrice] = useState<number | null>(null)

  useEffect(() => {
    setSym(defaultSymbol)
  }, [defaultSymbol])

  useEffect(() => {
    setSide(defaultSide)
  }, [defaultSide])

  const { data: accounts } = useQuery({
    queryKey: ['accounts-adv'],
    queryFn: () => api.get('/accounts/').then(r => r.data),
    staleTime: 60_000,
  })
  const accts: any[] = Array.isArray(accounts) ? accounts : []

  const { data: quote } = useQuery({
    queryKey: ['quote', sym],
    queryFn: () => api.get(`/market-data/quote/${sym}`).then(r => r.data).catch(() => null),
    refetchInterval: 5_000,
    enabled: sym.length >= 2,
  })
  useEffect(() => {
    if (quote?.last) setLivePrice(quote.last)
  }, [quote])

  const { data: acctEquity } = useQuery({
    queryKey: ['account-equity', accountId],
    queryFn: () => api.get(`/accounts/${accountId}/equity`).then(r => r.data).catch(() => null),
    enabled: !!accountId,
    refetchInterval: 30_000,
  })

  // Derive qty from equity%
  const derivedQty = useMemo(() => {
    if (sizeMode === 'pct_equity' && equityPct && acctEquity?.equity && livePrice) {
      return String(((acctEquity.equity * parseFloat(equityPct) / 100) / livePrice).toFixed(4))
    }
    return null
  }, [sizeMode, equityPct, acctEquity, livePrice])

  const effectiveQty = sizeMode === 'shares' ? qty : (sizeMode === 'pct_equity' ? (derivedQty ?? '') : null)
  const effectiveNotional = sizeMode === 'notional' ? notional : null

  const isBracket = !!(takeProfit || stopLoss || (useTrailing && trailingPct))
  const needsLimit = type === 'limit' || type === 'stop_limit'
  const needsStop  = type === 'stop'  || type === 'stop_limit'

  const entryPrice = limitPrice ? parseFloat(limitPrice) : livePrice

  const mutation = useMutation({
    mutationFn: async () => {
      if (!accountId) throw new Error('Select an account')
      const payload: any = {
        symbol: sym.replace(/^(NASDAQ:|NYSE:|BINANCE:)/, '').toUpperCase(),
        side,
        order_type: useTrailing ? 'trailing_stop' : type,
        time_in_force: tif,
        execution_algo: algo,
        account_id: accountId,
      }

      if (effectiveQty) payload.quantity = parseFloat(effectiveQty)
      if (effectiveNotional) payload.notional = parseFloat(effectiveNotional)
      if (needsLimit && limitPrice) payload.limit_price = parseFloat(limitPrice)
      if (needsStop  && stopPrice)  payload.stop_price  = parseFloat(stopPrice)
      if (takeProfit)    payload.take_profit_price = parseFloat(takeProfit)
      if (stopLoss)      payload.stop_loss_price   = parseFloat(stopLoss)
      if (useTrailing && trailingPct) payload.trailing_stop_pct = parseFloat(trailingPct)

      const endpoint = isBracket ? '/orders/bracket' : '/orders/'
      return api.post(endpoint, payload).then(r => r.data)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['orders-terminal'] })
      qc.invalidateQueries({ queryKey: ['orders-crypto'] })
      qc.invalidateQueries({ queryKey: ['positions-terminal'] })
      onSuccess?.()
    },
  })

  return (
    <div className="space-y-2.5 p-3 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 200px)' }}>
      {/* Symbol + live price */}
      <div className="flex items-center gap-2">
        <input
          value={sym}
          onChange={e => setSym(e.target.value.toUpperCase().replace(/^(NASDAQ:|NYSE:)/, ''))}
          className="flex-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2.5 py-2 text-sm font-mono font-bold text-white focus:outline-none focus:border-[#333] uppercase"
          placeholder="AAPL"
        />
        {livePrice && (
          <div className="text-right">
            <p className="text-sm font-bold font-mono text-[#e8e8e8]">${livePrice.toFixed(2)}</p>
            <p className="text-[9px] text-[#444]">live</p>
          </div>
        )}
      </div>

      {/* Account selector */}
      {accts.length > 0 ? (
        <select
          value={accountId}
          onChange={e => setAccountId(e.target.value)}
          className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-2 py-1.5 text-xs text-[#e8e8e8] focus:outline-none"
        >
          <option value="">Select account…</option>
          {accts.map((a: any) => (
            <option key={a.id} value={a.id}>{a.label} · {a.mode?.toUpperCase()}</option>
          ))}
        </select>
      ) : (
        <div className="border border-[#f5a623]/25 rounded-lg px-2.5 py-2 bg-[#0d0d0d]">
          <p className="text-[10px] text-[#f5a623]">No broker account connected</p>
          <p className="text-[9px] text-[#444] mt-0.5">Add Alpaca API keys in Settings.</p>
        </div>
      )}

      {/* Account equity display */}
      {acctEquity && (
        <div className="grid grid-cols-3 gap-1 text-center">
          {[
            { label: 'Equity', value: `$${Number(acctEquity.equity ?? 0).toLocaleString('en', { maximumFractionDigits: 0 })}` },
            { label: 'Cash', value: `$${Number(acctEquity.cash ?? 0).toLocaleString('en', { maximumFractionDigits: 0 })}` },
            { label: 'Buy Power', value: `$${Number(acctEquity.buying_power ?? 0).toLocaleString('en', { maximumFractionDigits: 0 })}` },
          ].map(item => (
            <div key={item.label} className="bg-[#0d0d0d] border border-[#1a1a1a] rounded p-1.5">
              <p className="text-[9px] text-[#444] uppercase">{item.label}</p>
              <p className="text-[10px] font-mono font-bold text-[#e8e8e8]">{item.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Buy / Sell */}
      <div className="grid grid-cols-2 gap-1.5">
        {(['buy', 'sell'] as Side[]).map(s => (
          <button
            key={s}
            onClick={() => setSide(s)}
            className="py-2.5 rounded-lg text-xs font-bold tracking-widest transition-all duration-100"
            style={{
              background: side === s ? (s === 'buy' ? '#00c853' : '#ff1744') : '#131313',
              color: side === s ? (s === 'buy' ? '#000' : '#fff') : '#444',
              border: `1px solid ${side === s ? (s === 'buy' ? '#00c853' : '#ff1744') : '#1e1e1e'}`,
            }}
          >
            {s === 'buy' ? '▲ BUY' : '▼ SELL'}
          </button>
        ))}
      </div>

      {/* Order type */}
      <div className="grid grid-cols-4 gap-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-0.5">
        {(['market', 'limit', 'stop', 'stop_limit'] as OrdType[]).map(t => (
          <button
            key={t}
            onClick={() => setType(t)}
            className="py-1.5 rounded text-[10px] font-medium transition-colors"
            style={{ background: type === t ? '#1e1e1e' : 'transparent', color: type === t ? '#e8e8e8' : '#444' }}
          >
            {t === 'stop_limit' ? 'STP-LMT' : t.slice(0, 1).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Limit price */}
      {needsLimit && (
        <NumInput
          label="Limit price"
          value={limitPrice} onChange={setLimitPrice}
          placeholder={livePrice?.toFixed(2) ?? '0.00'}
          color="#2196f3"
          hint="Order fills at this price or better"
          onPctChange={setLimitPrice}
          currentPrice={livePrice}
        />
      )}

      {/* Stop price */}
      {needsStop && (
        <NumInput
          label="Stop trigger price"
          value={stopPrice} onChange={setStopPrice}
          placeholder="0.00"
          color="#f5a623"
          hint="Order activates when price crosses this"
          onPctChange={setStopPrice}
          currentPrice={livePrice}
        />
      )}

      {/* Size */}
      <Section title="Position Size" defaultOpen={true}>
        <div className="grid grid-cols-3 gap-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded p-0.5">
          {(['shares', 'notional', 'pct_equity'] as SizeMode[]).map(m => (
            <button key={m} onClick={() => setSizeMode(m)}
              className="py-1 rounded text-[10px] font-medium transition-colors capitalize"
              style={{ background: sizeMode === m ? '#1e1e1e' : 'transparent', color: sizeMode === m ? '#e8e8e8' : '#444' }}>
              {m === 'pct_equity' ? '% equity' : m}
            </button>
          ))}
        </div>
        {sizeMode === 'shares' && (
          <NumInput label="Shares / contracts" value={qty} onChange={setQty} placeholder="1" step="1" />
        )}
        {sizeMode === 'notional' && (
          <NumInput label="Dollar amount" value={notional} onChange={setNotional} placeholder="1000.00" hint={livePrice ? `≈ ${(parseFloat(notional || '0') / livePrice).toFixed(3)} shares` : undefined} />
        )}
        {sizeMode === 'pct_equity' && (
          <>
            <NumInput label="% of account equity" value={equityPct} onChange={setEquityPct} placeholder="2.0" step="0.1" hint={derivedQty ? `≈ ${derivedQty} shares` : 'Connect account to compute'} />
          </>
        )}
      </Section>

      {/* Bracket Orders: SL + TP */}
      <Section title="Stop-Loss / Take-Profit" badge={isBracket ? 'BRACKET' : undefined} defaultOpen={false}>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] text-[#555]">Trailing stop instead of fixed SL</span>
          <button
            onClick={() => setUseTrailing(t => !t)}
            className="relative h-4 w-7 rounded-full transition-colors"
            style={{ background: useTrailing ? '#f5a623' : '#1e1e1e' }}
          >
            <span className="absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform"
              style={{ transform: useTrailing ? 'translateX(14px)' : 'translateX(2px)' }} />
          </button>
        </div>

        {!useTrailing ? (
          <NumInput
            label="Stop-Loss price"
            value={stopLoss} onChange={setStopLoss}
            placeholder={livePrice ? (side === 'buy' ? (livePrice * 0.97).toFixed(2) : (livePrice * 1.03).toFixed(2)) : '0.00'}
            color="#ff1744"
            hint={side === 'buy' ? 'Must be below entry' : 'Must be above entry'}
            onPctChange={setStopLoss}
            currentPrice={livePrice}
          />
        ) : (
          <NumInput
            label="Trailing stop %"
            value={trailingPct} onChange={setTrailingPct}
            placeholder="2.0"
            step="0.1"
            color="#ff1744"
            hint="Locks in profits as price moves in your favor"
          />
        )}

        <NumInput
          label="Take-Profit price"
          value={takeProfit} onChange={setTakeProfit}
          placeholder={livePrice ? (side === 'buy' ? (livePrice * 1.05).toFixed(2) : (livePrice * 0.95).toFixed(2)) : '0.00'}
          color="#00c853"
          hint={side === 'buy' ? 'Must be above entry' : 'Must be below entry'}
          onPctChange={setTakeProfit}
          currentPrice={livePrice}
        />

        <RiskRewardDisplay
          side={side}
          entry={entryPrice}
          sl={stopLoss ? parseFloat(stopLoss) : null}
          tp={takeProfit ? parseFloat(takeProfit) : null}
          qty={effectiveQty ? parseFloat(effectiveQty) : null}
          notional={effectiveNotional ? parseFloat(effectiveNotional) : null}
        />
      </Section>

      {/* Time-in-force + Execution Algo */}
      <Section title="Execution Settings" defaultOpen={false}>
        <div>
          <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Time-in-Force</p>
          <div className="grid grid-cols-4 gap-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded p-0.5">
            {(['gtc', 'day', 'ioc', 'fok'] as TIF[]).map(t => (
              <button key={t} onClick={() => setTif(t)}
                className="py-1.5 rounded text-[10px] uppercase font-bold transition-colors"
                style={{ background: tif === t ? '#1e1e1e' : 'transparent', color: tif === t ? '#e8e8e8' : '#444' }}
                title={TIF_INFO[t]}>
                {t}
              </button>
            ))}
          </div>
          <p className="text-[9px] text-[#333] mt-1">{TIF_INFO[tif]}</p>
        </div>

        <div>
          <p className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Execution Algorithm</p>
          <div className="space-y-1">
            {(Object.keys(ALGO_INFO) as Algo[]).map(a => (
              <button key={a} onClick={() => setAlgo(a)}
                className="w-full text-left px-2 py-1.5 rounded-lg text-xs transition-all"
                style={{
                  background: algo === a ? '#111' : 'transparent',
                  border: `1px solid ${algo === a ? ALGO_INFO[a].color + '44' : '#1a1a1a'}`,
                  color: algo === a ? ALGO_INFO[a].color : '#555',
                }}>
                <span className="font-bold">{ALGO_INFO[a].label}</span>
                <span className="text-[9px] ml-2 opacity-60">{ALGO_INFO[a].desc}</span>
              </button>
            ))}
          </div>
        </div>
      </Section>

      {/* Submit */}
      <div className="pt-1">
        <button
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending || !accountId || (!effectiveQty && !effectiveNotional)}
          className="w-full py-3 rounded-lg text-sm font-bold tracking-wider transition-all duration-150 active:scale-[0.98]"
          style={{
            background: !accountId || (!effectiveQty && !effectiveNotional)
              ? '#111'
              : side === 'buy'
              ? 'linear-gradient(135deg, #00c853, #00a843)'
              : 'linear-gradient(135deg, #ff1744, #c62828)',
            color: !accountId ? '#333' : '#fff',
            boxShadow: accountId ? (side === 'buy' ? '0 0 20px rgba(0,200,83,0.2)' : '0 0 20px rgba(255,23,68,0.2)') : 'none',
          }}
        >
          {mutation.isPending
            ? 'Placing order…'
            : isBracket
            ? `${side === 'buy' ? '▲ BUY' : '▼ SELL'} ${sym} (BRACKET)`
            : `${side === 'buy' ? '▲ BUY' : '▼ SELL'} ${sym}`}
        </button>

        {isBracket && (
          <p className="text-[10px] text-[#555] text-center mt-1.5">
            3 linked orders: entry + stop-loss + take-profit
          </p>
        )}
      </div>

      {mutation.isError && (
        <div className="bg-[#ff1744]/8 border border-[#ff1744]/25 rounded-lg px-3 py-2">
          <p className="text-xs text-[#ff1744]">{String((mutation.error as any)?.response?.data?.detail ?? mutation.error)}</p>
        </div>
      )}
      {mutation.isSuccess && (
        <div className="bg-[#00c853]/8 border border-[#00c853]/25 rounded-lg px-3 py-2">
          <p className="text-xs text-[#00c853]">{isBracket ? 'Bracket order placed — SL + TP linked ✓' : 'Order submitted ✓'}</p>
        </div>
      )}
    </div>
  )
}
