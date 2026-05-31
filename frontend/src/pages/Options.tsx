import { useState, useMemo } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../api/client'
import PortfolioGreeks from '../components/options/PortfolioGreeks'
import OptionsPCRPanel from '../components/strategies/OptionsPCRPanel'

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

interface IVRankData {
  symbol: string
  iv_rank: number | null
  hv_30: number | null
  hv_iv_ratio: number | null
  regime: string
  trade_signal: string
  iv_percentile?: number | null
  current_iv?: number | null
  error?: string
}

interface RulesValidationRequest {
  account_id?: string
  symbol: string
  option_symbol: string
  expiration_date: string
  side: 'buy' | 'sell'
  quantity: number
  credit_received: number
  delta: number
  strategy_type: 'csp' | 'covered_call' | 'iron_condor' | 'long_call' | 'long_put'
}

interface RuleDetail {
  value: number | null
  target: string
  status: 'ok' | 'warn' | 'error'
  max?: number | null
}

interface RulesValidationResponse {
  is_valid: boolean
  warnings: string[]
  errors: string[]
  rules: {
    dte: RuleDetail
    delta: RuleDetail
    iv_rank: RuleDetail
    position_size: RuleDetail & { max: number | null }
  }
  profit_target_price: number | null
  stop_loss_price: number | null
  exit_before_date: string | null
  max_profit: number
  max_loss_if_stopped: number | null
  dte: number
}

// ── Constants ──────────────────────────────────────────────────────────────

const QUICK_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ', 'TSLA']

const STRATEGIES = [
  { label: 'Covered Call', side: 'sell' as const, type: 'call' as const, strategyType: 'covered_call' as const, note: 'Sell OTM call against long position' },
  { label: 'Cash Secured Put', side: 'sell' as const, type: 'put' as const, strategyType: 'csp' as const, note: 'Sell OTM put to enter long at discount' },
  { label: 'Iron Condor', side: 'sell' as const, type: null, strategyType: 'iron_condor' as const, note: 'Sell OTM put spread + call spread' },
  { label: 'Long Call', side: 'buy' as const, type: 'call' as const, strategyType: 'long_call' as const, note: 'Buy ITM call for leveraged upside' },
  { label: 'Long Put', side: 'buy' as const, type: 'put' as const, strategyType: 'long_put' as const, note: 'Buy OTM put for downside protection' },
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

function fmtDate(dateStr: string | null): string {
  if (!dateStr) return '—'
  try {
    return new Date(dateStr + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  } catch {
    return dateStr
  }
}

function statusIcon(status: 'ok' | 'warn' | 'error'): string {
  if (status === 'ok') return '✓'
  if (status === 'warn') return '⚠'
  return '✗'
}

function statusColor(status: 'ok' | 'warn' | 'error'): string {
  if (status === 'ok') return '#00c853'
  if (status === 'warn') return '#f5a623'
  return '#ff1744'
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

// ── IV Rank Bar ────────────────────────────────────────────────────────────

function IVRankBar({ symbol }: { symbol: string }) {
  const { data, isLoading } = useQuery<IVRankData>({
    queryKey: ['iv-rank', symbol],
    queryFn: () => api.get(`/market-data/iv-rank/${symbol}`).then(r => r.data),
    staleTime: 60_000,
    refetchInterval: 120_000,
  })

  if (isLoading) {
    return (
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg px-4 py-2.5 animate-pulse">
        <div className="h-4 bg-[#1e1e1e] rounded w-48" />
      </div>
    )
  }

  if (!data || data.iv_rank == null) {
    return (
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg px-4 py-2.5">
        <span className="text-[10px] text-[#555]">IV Rank unavailable for {symbol}</span>
      </div>
    )
  }

  const rank = data.iv_rank
  const isHigh = rank >= 50
  const isMedium = rank >= 30 && rank < 50

  const regimeColor = isHigh ? '#00c853' : isMedium ? '#888888' : '#f5a623'
  const regimeLabel = isHigh
    ? 'HIGH IV — Good for selling'
    : isMedium
    ? 'MEDIUM IV — Neutral'
    : 'LOW IV — Good for buying'

  // bar color gradient from red (0) → amber (50) → green (80+)
  const barColor = rank >= 50 ? '#00c853' : rank >= 30 ? '#f5a623' : '#ff1744'

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg px-4 py-2.5">
      <div className="flex items-center gap-4 flex-wrap">
        {/* IV Rank bar */}
        <div className="flex items-center gap-2 min-w-[220px]">
          <span className="text-[10px] text-[#555] uppercase tracking-wider whitespace-nowrap">IV Rank</span>
          <div className="flex-1 h-2 bg-[#1e1e1e] rounded-full overflow-hidden min-w-[80px]">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(rank, 100)}%`, backgroundColor: barColor }}
            />
          </div>
          <span className="text-xs font-mono font-semibold text-white whitespace-nowrap">
            {rank.toFixed(0)}/100
          </span>
        </div>

        {/* Regime label */}
        <span
          className="text-[10px] font-semibold"
          style={{ color: regimeColor }}
        >
          {regimeLabel}
        </span>

        {/* HV/IV ratio */}
        {data.hv_iv_ratio != null && (
          <div className="text-[10px] text-[#555]">
            <span className="text-[#888]">HV/IV: </span>
            <span className="font-mono text-white">{data.hv_iv_ratio.toFixed(2)}×</span>
            <span className="ml-1 text-[#555]">
              {data.hv_iv_ratio > 1 ? '(options rich)' : '(options cheap)'}
            </span>
          </div>
        )}

        {/* IV Percentile if present */}
        {data.iv_percentile != null && (
          <div className="text-[10px] text-[#555]">
            <span className="text-[#888]">IV Pct: </span>
            <span className="font-mono text-white">{data.iv_percentile.toFixed(0)}%</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Rules Panel ────────────────────────────────────────────────────────────

interface RulesPanelProps {
  contract: OptionContract
  side: 'buy' | 'sell'
  qty: number
  underlying: string
  strategyType: 'csp' | 'covered_call' | 'iron_condor' | 'long_call' | 'long_put'
}

function RulesPanel({ contract, side, qty, underlying, strategyType }: RulesPanelProps) {
  const creditReceived = side === 'sell' && contract.mid != null ? contract.mid : 0

  const { data, isLoading } = useQuery<RulesValidationResponse>({
    queryKey: ['options-rules', contract.symbol, side, qty, strategyType],
    queryFn: () => {
      const body: RulesValidationRequest = {
        symbol: underlying,
        option_symbol: contract.symbol,
        expiration_date: contract.expiration_date,
        side,
        quantity: qty,
        credit_received: creditReceived,
        delta: contract.delta ?? 0,
        strategy_type: strategyType,
      }
      return api.post('/options/rules/validate', body).then(r => r.data)
    },
    enabled: qty > 0,
    staleTime: 30_000,
  })

  if (isLoading) {
    return (
      <div className="bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-3 space-y-2 animate-pulse">
        <div className="h-3 bg-[#1e1e1e] rounded w-32" />
        <div className="h-3 bg-[#1e1e1e] rounded w-full" />
        <div className="h-3 bg-[#1e1e1e] rounded w-3/4" />
      </div>
    )
  }

  if (!data) return null

  const ruleItems = [
    { key: 'DTE', detail: data.rules.dte },
    { key: 'Delta', detail: data.rules.delta },
    { key: 'IV Rank', detail: data.rules.iv_rank },
    { key: 'Size', detail: data.rules.position_size },
  ]

  return (
    <div className="bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg p-3 space-y-2">
      <div className="text-[10px] text-[#555] uppercase tracking-wider">Rules Check</div>

      {/* Rule rows */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-1">
        {ruleItems.map(({ key, detail }) => (
          <div key={key} className="flex items-center gap-1.5 text-[11px]">
            <span
              className="font-semibold text-[11px]"
              style={{ color: statusColor(detail.status) }}
            >
              {statusIcon(detail.status)}
            </span>
            <span className="text-[#888]">{key}:</span>
            <span className="font-mono text-white text-[10px]">
              {detail.value != null ? String(detail.value) : '?'}
            </span>
            <span className="text-[10px] text-[#444]">({detail.target})</span>
          </div>
        ))}
      </div>

      {/* Exit levels */}
      {(data.profit_target_price != null || data.stop_loss_price != null) && (
        <div className="pt-1 border-t border-[#1e1e1e] text-[10px] text-[#555] space-y-0.5">
          {data.profit_target_price != null && (
            <div>
              <span className="text-[#888]">Profit target: </span>
              <span className="font-mono text-[#00c853]">${data.profit_target_price.toFixed(2)}</span>
            </div>
          )}
          {data.stop_loss_price != null && (
            <div>
              <span className="text-[#888]">Stop loss: </span>
              <span className="font-mono text-[#ff1744]">${data.stop_loss_price.toFixed(2)}</span>
            </div>
          )}
          {data.exit_before_date && (
            <div>
              <span className="text-[#888]">Exit by: </span>
              <span className="font-mono text-[#f5a623]">{fmtDate(data.exit_before_date)}</span>
              <span className="text-[#444] ml-1">(21 DTE)</span>
            </div>
          )}
          {data.max_profit > 0 && (
            <div>
              <span className="text-[#888]">Max profit: </span>
              <span className="font-mono text-[#00c853]">+${data.max_profit.toFixed(2)}</span>
              {data.max_loss_if_stopped != null && (
                <span className="ml-2 text-[#888]">
                  Max loss (stopped): <span className="font-mono text-[#ff1744]">${data.max_loss_if_stopped.toFixed(2)}</span>
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Errors */}
      {data.errors.length > 0 && (
        <div className="space-y-1 pt-1 border-t border-[#1e1e1e]">
          {data.errors.map((e, i) => (
            <div key={i} className="text-[10px] text-[#ff1744]">✗ {e}</div>
          ))}
        </div>
      )}

      {/* Warnings (compact) */}
      {data.warnings.length > 0 && (
        <div className="space-y-1">
          {data.warnings.map((w, i) => (
            <div key={i} className="text-[10px] text-[#f5a623]">⚠ {w}</div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Order Panel ────────────────────────────────────────────────────────────

interface OrderPanelProps {
  contract: OptionContract
  underlying: string
  onClose: () => void
}

function OrderPanel({ contract, underlying, onClose }: OrderPanelProps) {
  const [side, setSide] = useState<'buy' | 'sell'>('buy')
  const [orderType, setOrderType] = useState<'market' | 'limit'>('limit')
  const [qty, setQty] = useState(1)
  const [limitPrice, setLimitPrice] = useState<string>(contract.mid != null ? contract.mid.toFixed(2) : '')
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [showRules, setShowRules] = useState(true)

  // Infer strategy type from side + option type
  const strategyType: 'csp' | 'covered_call' | 'iron_condor' | 'long_call' | 'long_put' =
    side === 'sell' && contract.option_type === 'put'
      ? 'csp'
      : side === 'sell' && contract.option_type === 'call'
      ? 'covered_call'
      : side === 'buy' && contract.option_type === 'call'
      ? 'long_call'
      : 'long_put'

  // Rules validation query to drive submit button label
  const { data: rulesData } = useQuery<RulesValidationResponse>({
    queryKey: ['options-rules', contract.symbol, side, qty, strategyType],
    queryFn: () => {
      const creditReceived = side === 'sell' && contract.mid != null ? contract.mid : 0
      const body: RulesValidationRequest = {
        symbol: underlying,
        option_symbol: contract.symbol,
        expiration_date: contract.expiration_date,
        side,
        quantity: qty,
        credit_received: creditReceived,
        delta: contract.delta ?? 0,
        strategy_type: strategyType,
      }
      return api.post('/options/rules/validate', body).then(r => r.data)
    },
    enabled: qty > 0,
    staleTime: 30_000,
  })

  const contractVal = parseFloat(limitPrice || '0') * 100

  async function handleSubmit() {
    setSubmitting(true)
    setResult(null)
    try {
      await new Promise(r => setTimeout(r, 300))
      setResult(`Order preview: ${side.toUpperCase()} ${qty}x ${contract.symbol} @ ${orderType === 'market' ? 'MKT' : '$' + limitPrice}`)
    } catch {
      setResult('Order submission failed.')
    } finally {
      setSubmitting(false)
    }
  }

  const submitLabel = rulesData
    ? rulesData.is_valid
      ? `Trade within rules ✓ — ${side.toUpperCase()} ${qty}`
      : `Trade outside rules ⚠ — ${side.toUpperCase()} ${qty}`
    : `${side.toUpperCase()} ${qty} Contract${qty > 1 ? 's' : ''}`

  const submitBgClass = rulesData && !rulesData.is_valid
    ? 'bg-[#ff1744]/80 hover:bg-[#ff1744]'
    : 'bg-[#f5a623] hover:bg-[#f5a623]/90'

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

      {/* Rules validation panel */}
      <div>
        <button
          onClick={() => setShowRules(v => !v)}
          className="text-[10px] text-[#555] hover:text-[#888] w-full text-left flex items-center gap-1 mb-1"
        >
          <span>{showRules ? '▾' : '▸'}</span>
          <span className="uppercase tracking-wider">Rules Validation</span>
          {rulesData && (
            <span
              className="ml-1 font-semibold"
              style={{ color: rulesData.is_valid ? '#00c853' : '#f5a623' }}
            >
              {rulesData.is_valid ? '✓ Pass' : `⚠ ${rulesData.errors.length + rulesData.warnings.length} issue${rulesData.errors.length + rulesData.warnings.length !== 1 ? 's' : ''}`}
            </span>
          )}
        </button>
        {showRules && qty > 0 && (
          <RulesPanel
            contract={contract}
            side={side}
            qty={qty}
            underlying={underlying}
            strategyType={strategyType}
          />
        )}
      </div>

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
        className={`w-full py-2 text-xs font-semibold rounded disabled:opacity-50 transition-colors ${
          rulesData && !rulesData.is_valid
            ? 'bg-[#f5a623]/80 text-black hover:bg-[#f5a623]'
            : 'bg-[#f5a623] text-black hover:bg-[#f5a623]/90'
        }`}
      >
        {submitting ? 'Submitting…' : submitLabel}
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
  const [showPortfolioGreeks, setShowPortfolioGreeks] = useState(false)

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

      {/* IV Rank Bar */}
      <IVRankBar symbol={underlying} />

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
                    <th className="text-[#00c853] text-center py-2 px-2 bg-[#001a00]/30" colSpan={7}>CALLS</th>
                    <th className="text-center py-2 px-2 text-[#f5a623] bg-[#111111]">STRIKE</th>
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

                          <td className={`px-3 py-1.5 text-center font-semibold ${atm ? 'text-[#f5a623] bg-[#f5a623]/5' : 'text-[#888]'}`}>
                            {strike}
                          </td>

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

        {/* Right panel: Greeks + Order + Portfolio Greeks */}
        {selectedContract && (
          <div className="w-72 flex-none space-y-3">
            <GreekCard contract={selectedContract} />
            <OrderPanel
              contract={selectedContract}
              underlying={underlying}
              onClose={() => setSelectedContract(null)}
            />

            {/* Portfolio Greeks collapsible */}
            <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
              <button
                onClick={() => setShowPortfolioGreeks(v => !v)}
                className="w-full px-4 py-2.5 flex items-center justify-between text-left hover:bg-[#1a1a1a] transition-colors"
              >
                <span className="text-[10px] text-[#555] uppercase tracking-wider">Portfolio Greeks</span>
                <span className="text-[#444] text-xs">{showPortfolioGreeks ? '▾' : '▸'}</span>
              </button>
              {showPortfolioGreeks && (
                <div className="px-4 pb-4">
                  <PortfolioGreeks />
                </div>
              )}
            </div>
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

      {/* Portfolio Greeks section (bottom, always visible when no contract selected) */}
      {!selectedContract && (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
          <button
            onClick={() => setShowPortfolioGreeks(v => !v)}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-[#1a1a1a] transition-colors"
          >
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-white">Portfolio Greeks</span>
              <span className="text-[10px] text-[#555]">Options-level risk summary</span>
            </div>
            <span className="text-[#444] text-sm">{showPortfolioGreeks ? '▾' : '▸'}</span>
          </button>
          {showPortfolioGreeks && (
            <div className="px-4 pb-4">
              <PortfolioGreeks />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
