/**
 * FundingRateMonitor — Crypto perpetual futures funding rate arbitrage dashboard
 *
 * Connects to:
 *   GET /analytics/funding-rates?symbols=BTCUSDT,ETHUSDT,...
 *
 * Funding rate arb: when |rate| > 10bps per 8h (≈ 109% annualized):
 *   rate > 0  → longs pay shorts → short perp + long spot (delta-neutral carry)
 *   rate < 0  → shorts pay longs → long perp + short spot (delta-neutral carry)
 */
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

// ─── Types ────────────────────────────────────────────────────────────────────

interface FundingHistory {
  funding_time: string | null
  rate: number | null
  rate_pct: number | null
}

interface FundingSymbol {
  symbol: string
  base_asset: string
  mark_price: number | null
  index_price: number | null
  last_funding_rate: number | null
  last_funding_rate_pct: number | null
  predicted_rate: number | null
  next_funding_time: string | null
  avg_rate_8h: number | null
  rate_annualized_pct: number | null
  signal: string
  history: FundingHistory[]
}

interface ArbOpportunity {
  symbol: string
  base_asset: string
  rate_pct: number
  annualized_pct: number | null
  direction: 'short_perp' | 'long_perp'
  signal: string
}

interface FundingResponse {
  symbols: FundingSymbol[]
  arb_opportunities: ArbOpportunity[]
  total_symbols: number
  extreme_count: number
  computed_at: string
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function rateColor(rate: number | null): string {
  if (rate === null) return '#555'
  const abs = Math.abs(rate)
  if (abs >= 0.001) return rate > 0 ? '#ff1744' : '#00c853'   // extreme
  if (abs >= 0.0003) return rate > 0 ? '#ff9800' : '#4caf50'  // moderate
  return '#888' // neutral
}

function fmtRate(rate: number | null): string {
  if (rate === null) return '—'
  return `${(rate * 100).toFixed(4)}%`
}

function fmtAnnualized(pct: number | null): string {
  if (pct === null) return '—'
  return `${pct.toFixed(2)}%`
}

function fmtPrice(p: number | null): string {
  if (p === null) return '—'
  return p >= 1 ? p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : p.toFixed(6)
}

function signalBadge(signal: string): { label: string; bg: string; color: string } {
  switch (signal) {
    case 'sell_perp_buy_spot': return { label: 'Short Perp / Long Spot', bg: '#ff174411', color: '#ff1744' }
    case 'buy_perp_sell_spot': return { label: 'Long Perp / Short Spot', bg: '#00c85311', color: '#00c853' }
    case 'slight_long_bias': return { label: 'Slight Long Bias', bg: '#ff980011', color: '#ff9800' }
    case 'slight_short_bias': return { label: 'Slight Short Bias', bg: '#4caf5011', color: '#4caf50' }
    default: return { label: 'Neutral', bg: '#33333322', color: '#555' }
  }
}

function nextFundingCountdown(isoTime: string | null): string {
  if (!isoTime) return '—'
  try {
    const ms = new Date(isoTime).getTime() - Date.now()
    if (ms <= 0) return 'Now'
    const h = Math.floor(ms / 3_600_000)
    const m = Math.floor((ms % 3_600_000) / 60_000)
    return `${h}h ${m}m`
  } catch {
    return '—'
  }
}

function MiniSparkline({ history }: { history: FundingHistory[] }) {
  const rates = history.map(h => h.rate ?? 0)
  if (rates.length < 2) return <span style={{ color: '#333' }}>—</span>

  const max = Math.max(...rates.map(Math.abs))
  const W = 80, H = 28, n = rates.length
  const mid = H / 2

  const pts = rates.map((r, i) => {
    const x = (i / (n - 1)) * W
    const y = mid - (max > 0 ? (r / max) * (mid - 2) : 0)
    return `${x},${y}`
  }).join(' ')

  const lastRate = rates[rates.length - 1]
  const strokeColor = lastRate > 0 ? '#ff1744' : lastRate < 0 ? '#00c853' : '#555'

  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      <line x1={0} y1={mid} x2={W} y2={mid} stroke="#2a2a2a" strokeWidth={1} />
      <polyline
        points={pts}
        fill="none"
        stroke={strokeColor}
        strokeWidth={1.5}
        strokeLinejoin="round"
        opacity={0.9}
      />
    </svg>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div style={{
      background: '#111', border: '1px solid #1e1e1e', borderRadius: 8,
      padding: '14px 18px', flex: 1, minWidth: 120,
    }}>
      <div style={{ color: '#888', fontSize: 11, fontFamily: 'Inter, sans-serif', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ color: color ?? '#e8e8e8', fontSize: 20, fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, lineHeight: 1 }}>
        {value}
      </div>
      {sub && <div style={{ color: '#444', fontSize: 11, marginTop: 4, fontFamily: 'Inter, sans-serif' }}>{sub}</div>}
    </div>
  )
}

function ArbCard({ opp }: { opp: ArbOpportunity }) {
  const badge = signalBadge(opp.signal)
  return (
    <div style={{
      background: opp.direction === 'short_perp' ? '#ff174408' : '#00c85308',
      border: `1px solid ${opp.direction === 'short_perp' ? '#ff174430' : '#00c85330'}`,
      borderRadius: 8, padding: '12px 16px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: 14, color: '#e8e8e8' }}>
          {opp.base_asset}
        </span>
        <span style={{
          background: badge.bg, color: badge.color, border: `1px solid ${badge.color}44`,
          borderRadius: 4, padding: '2px 7px', fontSize: 10, fontWeight: 600,
        }}>
          {badge.label}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 16 }}>
        <div>
          <div style={{ color: '#555', fontSize: 10, marginBottom: 2 }}>Rate/8h</div>
          <div style={{ color: rateColor(opp.rate_pct / 100), fontFamily: 'JetBrains Mono, monospace', fontSize: 14, fontWeight: 700 }}>
            {opp.rate_pct > 0 ? '+' : ''}{opp.rate_pct.toFixed(4)}%
          </div>
        </div>
        <div>
          <div style={{ color: '#555', fontSize: 10, marginBottom: 2 }}>Annualized</div>
          <div style={{ color: rateColor(opp.rate_pct / 100), fontFamily: 'JetBrains Mono, monospace', fontSize: 14, fontWeight: 700 }}>
            {fmtAnnualized(opp.annualized_pct)}
          </div>
        </div>
      </div>
    </div>
  )
}

function FundingRow({ sym }: { sym: FundingSymbol }) {
  const badge = signalBadge(sym.signal)
  const rc = rateColor(sym.last_funding_rate)

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '100px 110px 100px 120px 120px 90px 90px 80px',
        gap: 8, alignItems: 'center',
        padding: '10px 16px',
        borderBottom: '1px solid #1a1a1a',
        transition: 'background 0.12s',
      }}
      onMouseEnter={e => (e.currentTarget.style.background = '#141414')}
      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
    >
      {/* Asset */}
      <div>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: 13, color: '#e8e8e8' }}>
          {sym.base_asset}
        </div>
        <div style={{ color: '#444', fontSize: 10 }}>PERP</div>
      </div>

      {/* Mark price */}
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#e8e8e8' }}>
        ${fmtPrice(sym.mark_price)}
      </div>

      {/* Current rate */}
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: rc, fontWeight: 700 }}>
        {sym.last_funding_rate !== null ? (
          <>
            {sym.last_funding_rate > 0 ? '+' : ''}{fmtRate(sym.last_funding_rate)}
          </>
        ) : '—'}
      </div>

      {/* Predicted rate */}
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: rateColor(sym.predicted_rate) }}>
        {sym.predicted_rate !== null ? (
          <>{sym.predicted_rate > 0 ? '+' : ''}{fmtRate(sym.predicted_rate)}</>
        ) : '—'}
      </div>

      {/* Annualized */}
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: rc }}>
        {fmtAnnualized(sym.rate_annualized_pct)}
      </div>

      {/* Next funding */}
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#888' }}>
        {nextFundingCountdown(sym.next_funding_time)}
      </div>

      {/* Signal badge */}
      <div>
        <span style={{
          background: badge.bg, color: badge.color,
          border: `1px solid ${badge.color}44`,
          borderRadius: 4, padding: '2px 6px', fontSize: 10, fontWeight: 600,
          whiteSpace: 'nowrap',
        }}>
          {sym.signal === 'neutral' ? 'Neutral' : sym.signal.includes('sell') ? '↓ Perp' : '↑ Perp'}
        </span>
      </div>

      {/* Sparkline */}
      <div>
        <MiniSparkline history={sym.history} />
      </div>
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

const FILTER_OPTIONS = [
  { key: 'all', label: 'All' },
  { key: 'arb', label: 'Arb Opportunities' },
  { key: 'positive', label: 'Positive Rate' },
  { key: 'negative', label: 'Negative Rate' },
]

export default function FundingRateMonitor() {
  const [filter, setFilter] = useState('all')
  const [sortBy, setSortBy] = useState<'rate' | 'annualized' | 'asset'>('rate')

  const { data, isLoading, error, dataUpdatedAt } = useQuery<FundingResponse>({
    queryKey: ['funding-rates'],
    queryFn: () => api.get('/analytics/funding-rates').then(r => r.data),
    refetchInterval: 30_000, // refresh every 30s — funding changes every 8h but bots watch continuously
  })

  const filtered = useMemo(() => {
    if (!data) return []
    let syms = [...data.symbols]
    if (filter === 'arb') syms = syms.filter(s => s.last_funding_rate !== null && Math.abs(s.last_funding_rate) >= 0.001)
    if (filter === 'positive') syms = syms.filter(s => (s.last_funding_rate ?? 0) > 0)
    if (filter === 'negative') syms = syms.filter(s => (s.last_funding_rate ?? 0) < 0)

    syms.sort((a, b) => {
      if (sortBy === 'rate') return Math.abs(b.last_funding_rate ?? 0) - Math.abs(a.last_funding_rate ?? 0)
      if (sortBy === 'annualized') return Math.abs(b.rate_annualized_pct ?? 0) - Math.abs(a.rate_annualized_pct ?? 0)
      return a.base_asset.localeCompare(b.base_asset)
    })
    return syms
  }, [data, filter, sortBy])

  const pageStyle: React.CSSProperties = {
    background: '#0a0a0a', minHeight: '100vh',
    padding: '24px 28px', fontFamily: 'Inter, sans-serif', color: '#e8e8e8',
  }

  if (isLoading) {
    return (
      <div style={pageStyle}>
        <div style={{ color: '#555', fontSize: 13, marginTop: 60, textAlign: 'center' }}>
          Fetching funding rates from Binance…
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={pageStyle}>
        <div style={{
          background: '#ff174411', border: '1px solid #ff174433', borderRadius: 8,
          padding: 16, color: '#ff1744', fontSize: 13, marginTop: 40,
        }}>
          Failed to load funding rates: {(error as Error).message}
        </div>
      </div>
    )
  }

  const lastUpdate = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : '—'

  return (
    <div style={pageStyle}>
      {/* Header */}
      <div style={{ marginBottom: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#e8e8e8' }}>
            Funding Rate Monitor
          </h1>
          <div style={{ color: '#555', fontSize: 13, marginTop: 4 }}>
            Perpetual futures funding arbitrage — Binance USDT-M perps · Updated {lastUpdate}
          </div>
        </div>
        <div style={{
          background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 6,
          padding: '6px 12px', fontSize: 11, color: '#555',
        }}>
          Next refresh in 30s
        </div>
      </div>

      {/* KPI cards */}
      {data && (
        <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
          <StatCard
            label="Total Symbols"
            value={data.total_symbols}
            sub="Binance USDT-M perps"
          />
          <StatCard
            label="Arb Opportunities"
            value={data.arb_opportunities.length}
            sub="|rate| ≥ 10bps per 8h"
            color={data.arb_opportunities.length > 0 ? '#f5a623' : '#00c853'}
          />
          <StatCard
            label="Extreme Rates"
            value={data.extreme_count}
            sub="highest alpha carry trades"
            color={data.extreme_count > 0 ? '#ff1744' : '#888'}
          />
          {data.symbols[0] && (
            <StatCard
              label={`${data.symbols[0].base_asset} Rate/8h`}
              value={fmtRate(data.symbols[0].last_funding_rate)}
              sub={`≈ ${fmtAnnualized(data.symbols[0].rate_annualized_pct)} annualized`}
              color={rateColor(data.symbols[0].last_funding_rate)}
            />
          )}
          {data.symbols[1] && (
            <StatCard
              label={`${data.symbols[1].base_asset} Rate/8h`}
              value={fmtRate(data.symbols[1].last_funding_rate)}
              sub={`≈ ${fmtAnnualized(data.symbols[1].rate_annualized_pct)} annualized`}
              color={rateColor(data.symbols[1].last_funding_rate)}
            />
          )}
        </div>
      )}

      {/* Arb opportunities strip */}
      {data && data.arb_opportunities.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{
            color: '#888', fontSize: 11, textTransform: 'uppercase',
            letterSpacing: '0.06em', marginBottom: 8,
          }}>
            Active Arb Opportunities
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {data.arb_opportunities.slice(0, 6).map(opp => (
              <ArbCard key={opp.symbol} opp={opp} />
            ))}
          </div>
        </div>
      )}

      {/* Info box: what is funding rate arb */}
      <div style={{
        background: '#111', border: '1px solid #1e1e1e', borderRadius: 8,
        padding: '12px 16px', marginBottom: 16,
        display: 'flex', gap: 24, flexWrap: 'wrap',
      }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ color: '#f5a623', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>How Funding Rate Arb Works</div>
          <div style={{ color: '#555', fontSize: 12, lineHeight: 1.6 }}>
            Perp funding resets every 8h. When rate &gt; 0: longs pay shorts →
            go <span style={{ color: '#ff1744' }}>short perp</span> +{' '}
            <span style={{ color: '#00c853' }}>long spot</span> (delta-neutral carry).
            Annualized yield = rate × 3 × 365.
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ color: '#f5a623', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>Risk Considerations</div>
          <div style={{ color: '#555', fontSize: 12, lineHeight: 1.6 }}>
            Execution risk on spot leg, liquidation on perp, basis risk if spot/perp diverge.
            QuantEdge targets |rate| &gt; 10bps per 8h (≈100% annualized) for minimum viability.
          </div>
        </div>
      </div>

      {/* Main table */}
      <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 10 }}>
        {/* Table controls */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '14px 16px', borderBottom: '1px solid #1e1e1e', flexWrap: 'wrap', gap: 10,
        }}>
          {/* Filters */}
          <div style={{ display: 'flex', gap: 4 }}>
            {FILTER_OPTIONS.map(f => (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                style={{
                  background: filter === f.key ? '#f5a623' : 'transparent',
                  color: filter === f.key ? '#000' : '#888',
                  border: `1px solid ${filter === f.key ? '#f5a623' : '#2a2a2a'}`,
                  borderRadius: 5, padding: '4px 11px', cursor: 'pointer',
                  fontSize: 12, fontFamily: 'Inter, sans-serif', fontWeight: filter === f.key ? 700 : 400,
                  transition: 'all 0.15s',
                }}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* Sort */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: '#555', fontSize: 11 }}>Sort by:</span>
            {(['rate', 'annualized', 'asset'] as const).map(s => (
              <button
                key={s}
                onClick={() => setSortBy(s)}
                style={{
                  background: sortBy === s ? '#1a1a1a' : 'transparent',
                  color: sortBy === s ? '#f5a623' : '#555',
                  border: `1px solid ${sortBy === s ? '#f5a623' : '#2a2a2a'}`,
                  borderRadius: 5, padding: '3px 9px', cursor: 'pointer', fontSize: 11,
                }}
              >
                {s === 'rate' ? 'Rate' : s === 'annualized' ? 'Annualized' : 'Asset'}
              </button>
            ))}
          </div>
        </div>

        {/* Column headers */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '100px 110px 100px 120px 120px 90px 90px 80px',
          gap: 8, padding: '8px 16px',
          color: '#444', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em',
          borderBottom: '1px solid #1a1a1a',
        }}>
          <div>Asset</div>
          <div>Mark Price</div>
          <div>Rate/8h</div>
          <div>Predicted</div>
          <div>Annualized</div>
          <div>Next Fund.</div>
          <div>Signal</div>
          <div>Rate History</div>
        </div>

        {/* Rows */}
        {filtered.length === 0 ? (
          <div style={{ padding: '32px', textAlign: 'center', color: '#444', fontSize: 13 }}>
            No symbols match the current filter.
          </div>
        ) : (
          filtered.map(sym => <FundingRow key={sym.symbol} sym={sym} />)
        )}
      </div>

      {/* Methodology note */}
      <div style={{ marginTop: 12, color: '#333', fontSize: 11 }}>
        Data source: Binance USDT-M Perpetual Futures public API · Funding events every 8h (00:00, 08:00, 16:00 UTC) ·
        Rate/8h shown as % · Annualized = rate × 3 × 365 · Arb threshold: |rate| ≥ 0.10% per 8h
      </div>
    </div>
  )
}
