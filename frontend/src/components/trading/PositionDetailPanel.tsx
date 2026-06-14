/**
 * PositionDetailPanel — Option Alpha-style position details slide-in panel.
 * Shows Greeks estimates, P&L payoff chart, exit options editor.
 */
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { X, Edit2, Save, RotateCcw, AlertTriangle } from 'lucide-react'
import api from '../../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Position {
  id: string | null
  symbol: string
  quantity: number
  avg_cost: number
  current_price: number | null
  unrealized_pnl: number | null
  side: string
}

interface ExitConfig {
  symbol: string
  strategy_name: string
  strategy_type: string
  risk_bucket: string
  exit_strategies_active: string[]
  entry_price: number | null
  stop_loss: number | null
  take_profit: number | null
  peak_price: number | null
  bars_held: number
  pnl_pct: number | null
  stored_at: string | null
  profit_target_pct?: number | null
  stop_loss_pct?: number | null
  trailing_stop_pct?: number | null
  expiration_days?: number | null
  pricing_method?: string | null
  bid_ask_guard?: boolean | null
  notes?: string | null
  tags?: string[] | null
}

interface ExitOptionsUpdate {
  stop_loss?: number | null
  take_profit?: number | null
  profit_target_pct?: number | null
  stop_loss_pct?: number | null
  trailing_stop_pct?: number | null
  expiration_days?: number | null
  pricing_method?: string | null
  bid_ask_guard?: boolean | null
  notes?: string | null
  tags?: string[] | null
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const fmtP = (v: number | null | undefined) =>
  v == null ? '—' : `$${Math.abs(v).toFixed(2)}`
const fmtPct = (v: number | null | undefined) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`
const fmtN = (v: number | null | undefined, d = 4) =>
  v == null ? '—' : v.toFixed(d)
const pnlColor = (v: number | null | undefined) =>
  v == null ? '#555' : v >= 0 ? '#00c853' : '#ff1744'

/** Estimate basic Greeks from position data (no options chain needed). */
function estimateGreeks(pos: Position, exitConfig: ExitConfig | null) {
  const price = pos.current_price ?? pos.avg_cost
  const entry = exitConfig?.entry_price ?? pos.avg_cost
  const pnlPct = price && entry ? ((price - entry) / entry) : 0
  // Equity Greeks approximations for pedagogical display
  const delta = pos.side === 'long' ? 1.0 : -1.0
  const gamma = 0.0
  const theta = -0.02 // ~$0.02/day decay proxy
  const vega = 0.0
  const alpha = pnlPct > 0.05 ? 0.12 : pnlPct < -0.05 ? -0.08 : 0.03
  return { delta, gamma, theta, vega, alpha }
}

/** Generate a simple P&L payoff curve for equity (linear with stop/target). */
function buildPayoffCurve(
  entryPrice: number,
  stopLoss: number | null,
  takeProfit: number | null,
  quantity: number,
  side: string
): Array<{ price: number; pnl: number }> {
  const lo = stopLoss ? Math.min(stopLoss, entryPrice * 0.8) : entryPrice * 0.8
  const hi = takeProfit ? Math.max(takeProfit, entryPrice * 1.2) : entryPrice * 1.2
  const steps = 40
  const pts: Array<{ price: number; pnl: number }> = []
  for (let i = 0; i <= steps; i++) {
    const p = lo + (hi - lo) * (i / steps)
    const rawPnl = (p - entryPrice) * quantity * (side === 'long' ? 1 : -1)
    // Apply hard cap at stop/profit
    let pnl = rawPnl
    if (stopLoss && side === 'long' && p <= stopLoss)
      pnl = (stopLoss - entryPrice) * quantity * -1
    if (stopLoss && side === 'short' && p >= stopLoss)
      pnl = (stopLoss - entryPrice) * quantity
    if (takeProfit && side === 'long' && p >= takeProfit)
      pnl = (takeProfit - entryPrice) * quantity
    if (takeProfit && side === 'short' && p <= takeProfit)
      pnl = (entryPrice - takeProfit) * quantity
    pts.push({ price: p, pnl })
  }
  return pts
}

// ── Payoff Chart ──────────────────────────────────────────────────────────────

function PayoffChart({
  position,
  exitConfig,
}: {
  position: Position
  exitConfig: ExitConfig | null
}) {
  const entry = exitConfig?.entry_price ?? position.avg_cost
  const stop = exitConfig?.stop_loss ?? null
  const profit = exitConfig?.take_profit ?? null
  const qty = Math.abs(position.quantity)

  const points = buildPayoffCurve(entry, stop, profit, qty, position.side)
  if (!points.length) return null

  const W = 360
  const H = 120
  const PAD = { l: 48, r: 12, t: 12, b: 24 }

  const prices = points.map(p => p.price)
  const pnls = points.map(p => p.pnl)
  const minP = Math.min(...prices)
  const maxP = Math.max(...prices)
  const minPnl = Math.min(...pnls, 0)
  const maxPnl = Math.max(...pnls, 0)

  const scaleX = (p: number) =>
    PAD.l + ((p - minP) / (maxP - minP || 1)) * (W - PAD.l - PAD.r)
  const scaleY = (v: number) =>
    PAD.t + ((maxPnl - v) / (maxPnl - minPnl || 1)) * (H - PAD.t - PAD.b)
  const zeroY = scaleY(0)

  // Build SVG path
  const path = points
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(pt.price).toFixed(1)} ${scaleY(pt.pnl).toFixed(1)}`)
    .join(' ')

  // Fill areas
  const profitPoints = points.filter(pt => pt.pnl >= 0)
  const lossPoints = points.filter(pt => pt.pnl <= 0)

  const profitFill = profitPoints.length
    ? profitPoints
        .map((pt, i) => {
          if (i === 0)
            return `M ${scaleX(pt.price).toFixed(1)} ${zeroY.toFixed(1)} L ${scaleX(pt.price).toFixed(1)} ${scaleY(pt.pnl).toFixed(1)}`
          return `L ${scaleX(pt.price).toFixed(1)} ${scaleY(pt.pnl).toFixed(1)}`
        })
        .concat([`L ${scaleX(profitPoints[profitPoints.length - 1].price).toFixed(1)} ${zeroY.toFixed(1)} Z`])
        .join(' ')
    : ''

  const lossFill = lossPoints.length
    ? lossPoints
        .map((pt, i) => {
          if (i === 0)
            return `M ${scaleX(pt.price).toFixed(1)} ${zeroY.toFixed(1)} L ${scaleX(pt.price).toFixed(1)} ${scaleY(pt.pnl).toFixed(1)}`
          return `L ${scaleX(pt.price).toFixed(1)} ${scaleY(pt.pnl).toFixed(1)}`
        })
        .concat([`L ${scaleX(lossPoints[lossPoints.length - 1].price).toFixed(1)} ${zeroY.toFixed(1)} Z`])
        .join(' ')
    : ''

  const curP = position.current_price ?? entry
  const curX = scaleX(curP)
  const curPnl = points.reduce((a, b) =>
    Math.abs(b.price - curP) < Math.abs(a.price - curP) ? b : a
  ).pnl
  const curY = scaleY(curPnl)

  return (
    <svg width={W} height={H} style={{ display: 'block', width: '100%', height: H }}>
      {/* Zero line */}
      <line x1={PAD.l} y1={zeroY} x2={W - PAD.r} y2={zeroY} stroke="#2a2a2a" strokeWidth={1} />

      {/* Profit fill */}
      {points.some(p => p.pnl > 0) && (
        <path d={profitFill} fill="rgba(0,200,83,0.15)" />
      )}
      {/* Loss fill */}
      {points.some(p => p.pnl < 0) && (
        <path d={lossFill} fill="rgba(255,23,68,0.15)" />
      )}

      {/* Main curve */}
      <path d={path} fill="none" stroke="#f5a623" strokeWidth={1.5} />

      {/* Entry price marker */}
      {entry && (() => {
        const ex = scaleX(entry)
        return (
          <g>
            <line x1={ex} y1={PAD.t} x2={ex} y2={H - PAD.b} stroke="#2196F3" strokeWidth={1} strokeDasharray="3,3" />
            <text x={ex + 3} y={PAD.t + 10} fill="#2196F3" fontSize={9} fontFamily="JetBrains Mono, monospace">entry</text>
          </g>
        )
      })()}

      {/* Current price dot */}
      <circle cx={curX} cy={curY} r={4} fill={curPnl >= 0 ? '#00c853' : '#ff1744'} />
      <line x1={curX} y1={PAD.t} x2={curX} y2={H - PAD.b} stroke={curPnl >= 0 ? '#00c853' : '#ff1744'} strokeWidth={1} strokeDasharray="2,2" opacity={0.5} />

      {/* Stop loss line */}
      {stop && (() => {
        const sx = scaleX(stop)
        return <line x1={sx} y1={PAD.t} x2={sx} y2={H - PAD.b} stroke="#ff1744" strokeWidth={1} strokeDasharray="4,2" />
      })()}
      {/* Take profit line */}
      {profit && (() => {
        const px2 = scaleX(profit)
        return <line x1={px2} y1={PAD.t} x2={px2} y2={H - PAD.b} stroke="#00c853" strokeWidth={1} strokeDasharray="4,2" />
      })()}

      {/* Y axis labels */}
      <text x={PAD.l - 4} y={PAD.t + 6} fill="#555" fontSize={8} textAnchor="end" fontFamily="JetBrains Mono, monospace">
        +{Math.abs(maxPnl).toFixed(0)}
      </text>
      <text x={PAD.l - 4} y={H - PAD.b} fill="#555" fontSize={8} textAnchor="end" fontFamily="JetBrains Mono, monospace">
        -{Math.abs(minPnl).toFixed(0)}
      </text>
      <text x={PAD.l - 4} y={zeroY + 3} fill="#555" fontSize={8} textAnchor="end" fontFamily="JetBrains Mono, monospace">0</text>
    </svg>
  )
}

// ── Metric Cell ───────────────────────────────────────────────────────────────

function MetricCell({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ padding: '10px 12px', background: '#161616', borderRadius: 6, border: '1px solid #1e1e1e' }}>
      <div style={{ fontSize: 9, color: '#555', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', marginBottom: 4, textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{ fontSize: 14, fontWeight: 700, color: color || '#e8e8e8', fontFamily: 'JetBrains Mono, monospace' }}>
        {value}
      </div>
    </div>
  )
}

// ── Exit Options Form ─────────────────────────────────────────────────────────

function ExitOptionsForm({
  symbol,
  config,
  onClose,
}: {
  symbol: string
  config: ExitConfig | null
  onClose: () => void
}) {
  const qc = useQueryClient()
  const [form, setForm] = useState<ExitOptionsUpdate>({
    profit_target_pct: config?.profit_target_pct ?? null,
    stop_loss_pct: config?.stop_loss_pct ?? null,
    take_profit: config?.take_profit ?? null,
    stop_loss: config?.stop_loss ?? null,
    trailing_stop_pct: config?.trailing_stop_pct ?? null,
    expiration_days: config?.expiration_days ?? null,
    pricing_method: config?.pricing_method ?? 'Normal',
    bid_ask_guard: config?.bid_ask_guard ?? true,
    notes: config?.notes ?? '',
    tags: config?.tags ?? [],
  })

  const mutation = useMutation({
    mutationFn: (data: ExitOptionsUpdate) =>
      api.patch(`/positions/${symbol}/exit-config`, data).then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['exit-config', symbol] })
      onClose()
    },
  })

  const set = (k: keyof ExitOptionsUpdate, v: unknown) =>
    setForm(f => ({ ...f, [k]: v }))

  const inputStyle: React.CSSProperties = {
    background: '#1a1a1a',
    border: '1px solid #2a2a2a',
    borderRadius: 4,
    color: '#e8e8e8',
    fontFamily: 'JetBrains Mono, monospace',
    fontSize: 12,
    padding: '5px 8px',
    width: '100%',
    outline: 'none',
  }

  const labelStyle: React.CSSProperties = {
    fontSize: 10,
    color: '#555',
    fontFamily: 'JetBrains Mono, monospace',
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    marginBottom: 4,
    display: 'block',
  }

  const PRICING_OPTIONS = ['Off', 'Normal', 'Fast', 'Patient', 'Market']

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999,
      }}
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <div style={{ background: '#111', border: '1px solid #2a2a2a', borderRadius: 12, width: 480, maxHeight: '85vh', overflow: 'auto', padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h3 style={{ margin: 0, color: '#f5a623', fontSize: 15, fontFamily: 'JetBrains Mono, monospace' }}>
            Edit Exit Options — {symbol}
          </h3>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', padding: 4 }}>
            <X size={16} />
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {/* Profit Taking % */}
          <div>
            <label style={labelStyle}>Profit Target %</label>
            <input
              type="number" step="0.5"
              value={form.profit_target_pct ?? ''}
              onChange={e => set('profit_target_pct', e.target.value ? +e.target.value : null)}
              placeholder="e.g. 50"
              style={inputStyle}
            />
          </div>

          {/* Profit Target $ */}
          <div>
            <label style={labelStyle}>Profit Target $</label>
            <input
              type="number" step="0.01"
              value={form.take_profit ?? ''}
              onChange={e => set('take_profit', e.target.value ? +e.target.value : null)}
              placeholder="e.g. 1.25"
              style={inputStyle}
            />
          </div>

          {/* Pricing method */}
          <div>
            <label style={labelStyle}>Pricing Method</label>
            <select
              value={form.pricing_method ?? 'Normal'}
              onChange={e => set('pricing_method', e.target.value)}
              style={{ ...inputStyle, cursor: 'pointer' }}
            >
              {PRICING_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>

          {/* Stop Loss % */}
          <div>
            <label style={labelStyle}>Stop Loss %</label>
            <input
              type="number" step="0.5"
              value={form.stop_loss_pct ?? ''}
              onChange={e => set('stop_loss_pct', e.target.value ? +e.target.value : null)}
              placeholder="e.g. 200"
              style={inputStyle}
            />
          </div>

          {/* Stop Loss $ */}
          <div>
            <label style={labelStyle}>Stop Loss $</label>
            <input
              type="number" step="0.01"
              value={form.stop_loss ?? ''}
              onChange={e => set('stop_loss', e.target.value ? +e.target.value : null)}
              placeholder="price level"
              style={inputStyle}
            />
          </div>

          {/* Trailing Stop % */}
          <div>
            <label style={labelStyle}>Trailing Stop %</label>
            <input
              type="number" step="0.5"
              value={form.trailing_stop_pct ?? ''}
              onChange={e => set('trailing_stop_pct', e.target.value ? +e.target.value : null)}
              placeholder="e.g. 5"
              style={inputStyle}
            />
          </div>

          {/* Expiration Days */}
          <div>
            <label style={labelStyle}>Expiration (DTE)</label>
            <input
              type="number" step="1"
              value={form.expiration_days ?? ''}
              onChange={e => set('expiration_days', e.target.value ? +e.target.value : null)}
              placeholder="days to expiry"
              style={inputStyle}
            />
          </div>

          {/* Bid/Ask Guard */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, paddingTop: 20 }}>
            <input
              type="checkbox"
              id="bid_ask_guard"
              checked={form.bid_ask_guard ?? true}
              onChange={e => set('bid_ask_guard', e.target.checked)}
              style={{ accentColor: '#f5a623', width: 14, height: 14 }}
            />
            <label htmlFor="bid_ask_guard" style={{ ...labelStyle, margin: 0, cursor: 'pointer' }}>
              Bid/Ask Guard
            </label>
          </div>
        </div>

        {/* Tags */}
        <div style={{ marginTop: 12 }}>
          <label style={labelStyle}>Tags (comma-separated)</label>
          <input
            type="text"
            value={(form.tags ?? []).join(', ')}
            onChange={e => set('tags', e.target.value.split(',').map(t => t.trim()).filter(Boolean))}
            placeholder="earnings, high-iv, monthly"
            style={inputStyle}
          />
        </div>

        {/* Notes */}
        <div style={{ marginTop: 12 }}>
          <label style={labelStyle}>Notes</label>
          <textarea
            value={form.notes ?? ''}
            onChange={e => set('notes', e.target.value)}
            placeholder="Trade notes…"
            rows={3}
            style={{ ...inputStyle, resize: 'vertical' }}
          />
        </div>

        {/* Error */}
        {mutation.isError && (
          <div style={{ marginTop: 10, color: '#ff1744', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}>
            <AlertTriangle size={12} style={{ marginRight: 6, verticalAlign: 'middle' }} />
            Failed to save exit options
          </div>
        )}

        {/* Actions */}
        <div style={{ display: 'flex', gap: 10, marginTop: 20, justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            style={{ padding: '7px 18px', borderRadius: 6, border: '1px solid #2a2a2a', background: 'none', color: '#888', fontSize: 12, cursor: 'pointer' }}
          >
            Cancel
          </button>
          <button
            onClick={() => mutation.mutate(form)}
            disabled={mutation.isPending}
            style={{
              padding: '7px 18px', borderRadius: 6, border: 'none',
              background: '#f5a623', color: '#0a0a0a', fontSize: 12, fontWeight: 700,
              cursor: mutation.isPending ? 'wait' : 'pointer',
              display: 'flex', alignItems: 'center', gap: 6,
            }}
          >
            <Save size={12} />
            {mutation.isPending ? 'Saving…' : 'Save Options'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Main Panel ────────────────────────────────────────────────────────────────

interface Props {
  position: Position
  onClose: () => void
}

export default function PositionDetailPanel({ position, onClose }: Props) {
  const [editingExits, setEditingExits] = useState(false)

  const { data: exitConfig, isLoading: loadingExit } = useQuery<ExitConfig>({
    queryKey: ['exit-config', position.symbol],
    queryFn: () =>
      api.get(`/positions/${position.symbol}/exit-config`).then(r => r.data),
    retry: false,
  })

  const greeks = estimateGreeks(position, exitConfig ?? null)
  const entry = exitConfig?.entry_price ?? position.avg_cost
  const price = position.current_price ?? entry
  const pnl = position.unrealized_pnl ?? (price - entry) * position.quantity
  const pnlPct = entry > 0 ? ((price - entry) / entry) * 100 : 0
  const dit = exitConfig?.bars_held ?? 0
  const maxProfit = exitConfig?.take_profit != null
    ? (exitConfig.take_profit - entry) * Math.abs(position.quantity)
    : null
  const maxLoss = exitConfig?.stop_loss != null
    ? (entry - exitConfig.stop_loss) * Math.abs(position.quantity)
    : null
  const rewardRisk = maxProfit != null && maxLoss != null && maxLoss !== 0
    ? maxProfit / maxLoss
    : null
  const capitalAtRisk = Math.abs(position.quantity) * entry
  const otmPct = exitConfig?.take_profit != null
    ? ((exitConfig.take_profit - price) / price) * 100
    : null

  return (
    <>
      {/* Overlay */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
          zIndex: 900, animation: 'fadeIn 0.15s ease',
        }}
      />

      {/* Panel */}
      <div
        style={{
          position: 'fixed', top: 0, right: 0, bottom: 0, width: 480,
          background: '#0f0f0f', borderLeft: '1px solid #1e1e1e',
          zIndex: 901, overflow: 'auto',
          animation: 'slideInRight 0.2s ease',
          display: 'flex', flexDirection: 'column',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '16px 20px', borderBottom: '1px solid #1e1e1e',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          position: 'sticky', top: 0, background: '#0f0f0f', zIndex: 2,
        }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#e8e8e8', fontFamily: 'JetBrains Mono, monospace' }}>
              {position.symbol}
              <span style={{
                marginLeft: 10, fontSize: 11, padding: '2px 8px', borderRadius: 3,
                background: position.side === 'long' ? 'rgba(0,200,83,0.15)' : 'rgba(255,23,68,0.15)',
                color: position.side === 'long' ? '#00c853' : '#ff1744',
              }}>
                {position.side.toUpperCase()}
              </span>
            </div>
            <div style={{ fontSize: 11, color: '#555', fontFamily: 'JetBrains Mono, monospace', marginTop: 2 }}>
              {exitConfig?.strategy_name ?? 'position'} · {Math.abs(position.quantity)} shares
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button
              onClick={() => setEditingExits(true)}
              style={{
                padding: '5px 12px', borderRadius: 5, border: '1px solid #2a2a2a',
                background: '#161616', color: '#f5a623', fontSize: 11,
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5,
                fontFamily: 'JetBrains Mono, monospace',
              }}
            >
              <Edit2 size={11} /> Edit Exits
            </button>
            <button
              onClick={onClose}
              style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', padding: 4 }}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* P&L Payoff Chart */}
          <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 8, padding: 16 }}>
            <div style={{ fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', marginBottom: 10, textTransform: 'uppercase' }}>
              P&L at Expiration
            </div>
            <PayoffChart position={position} exitConfig={exitConfig ?? null} />
            <div style={{ display: 'flex', gap: 16, marginTop: 10, fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace' }}>
              <span><span style={{ color: '#2196F3' }}>──</span> Entry</span>
              <span><span style={{ color: '#f5a623' }}>──</span> Curve</span>
              {exitConfig?.stop_loss && <span><span style={{ color: '#ff1744' }}>- -</span> Stop</span>}
              {exitConfig?.take_profit && <span><span style={{ color: '#00c853' }}>- -</span> Target</span>}
            </div>
          </div>

          {/* P&L Summary */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: '#1e1e1e', borderRadius: 8, overflow: 'hidden', border: '1px solid #1e1e1e' }}>
            <MetricCell label="Total P/L" value={`${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}`} color={pnlColor(pnl)} />
            <MetricCell label="Return %" value={fmtPct(pnlPct)} color={pnlColor(pnlPct)} />
            <MetricCell label="Market Value" value={`$${(price * Math.abs(position.quantity)).toFixed(2)}`} />
            <MetricCell label="Capital at Risk" value={`$${capitalAtRisk.toFixed(2)}`} />
            <MetricCell label="Entry Price" value={fmtP(entry)} />
            <MetricCell label="Current Price" value={fmtP(price)} />
          </div>

          {/* Greeks */}
          <div>
            <div style={{ fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', marginBottom: 10, textTransform: 'uppercase' }}>
              Greeks (Estimated)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
              <MetricCell label="Delta" value={fmtN(greeks.delta, 3)} color={greeks.delta > 0 ? '#00c853' : '#ff1744'} />
              <MetricCell label="Gamma" value={fmtN(greeks.gamma, 4)} />
              <MetricCell label="Theta" value={fmtN(greeks.theta, 4)} color="#ff1744" />
              <MetricCell label="Vega" value={fmtN(greeks.vega, 4)} />
              <MetricCell label="Alpha" value={fmtN(greeks.alpha * 100, 2) + '%'} color={pnlColor(greeks.alpha)} />
              <MetricCell label="DIT" value={dit.toString()} />
            </div>
          </div>

          {/* Risk Metrics */}
          <div>
            <div style={{ fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', marginBottom: 10, textTransform: 'uppercase' }}>
              Risk / Reward
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <MetricCell
                label="Max Profit"
                value={maxProfit != null ? `$${maxProfit.toFixed(2)}` : '∞'}
                color="#00c853"
              />
              <MetricCell
                label="Max Loss"
                value={maxLoss != null ? `-$${maxLoss.toFixed(2)}` : 'unlimited'}
                color="#ff1744"
              />
              <MetricCell
                label="Reward / Risk"
                value={rewardRisk != null ? `${rewardRisk.toFixed(2)}x` : '—'}
                color={rewardRisk != null && rewardRisk >= 1 ? '#00c853' : '#888'}
              />
              <MetricCell
                label="OTM to Target"
                value={otmPct != null ? fmtPct(otmPct) : '—'}
              />
            </div>
          </div>

          {/* Exit Options Summary */}
          <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 8, padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div style={{ fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                Exit Options
              </div>
              <button
                onClick={() => setEditingExits(true)}
                style={{ background: 'none', border: 'none', color: '#f5a623', fontSize: 11, cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace', display: 'flex', alignItems: 'center', gap: 4 }}
              >
                <Edit2 size={10} /> Edit
              </button>
            </div>
            {loadingExit ? (
              <div style={{ color: '#555', fontSize: 12 }}>Loading…</div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
                {[
                  { label: 'Profit Target', value: exitConfig?.take_profit ? fmtP(exitConfig.take_profit) : exitConfig?.profit_target_pct ? `${exitConfig.profit_target_pct}%` : 'Not set' },
                  { label: 'Stop Loss', value: exitConfig?.stop_loss ? fmtP(exitConfig.stop_loss) : exitConfig?.stop_loss_pct ? `${exitConfig.stop_loss_pct}%` : 'Not set' },
                  { label: 'Trailing Stop', value: exitConfig?.trailing_stop_pct ? `${exitConfig.trailing_stop_pct}%` : 'Off' },
                  { label: 'Expiration', value: exitConfig?.expiration_days ? `${exitConfig.expiration_days} DTE` : 'None' },
                  { label: 'Pricing', value: exitConfig?.pricing_method ?? 'Normal' },
                  { label: 'Bid/Ask Guard', value: exitConfig?.bid_ask_guard ? 'On' : 'Off' },
                ].map(({ label, value }) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #1a1a1a' }}>
                    <span style={{ color: '#555', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>{label}</span>
                    <span style={{ color: '#e8e8e8', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>{value}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Active exit strategies */}
            {exitConfig?.exit_strategies_active?.length ? (
              <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {exitConfig.exit_strategies_active.map(s => (
                  <span key={s} style={{
                    padding: '2px 8px', borderRadius: 3, fontSize: 10,
                    background: 'rgba(245,166,35,0.1)', color: '#f5a623',
                    fontFamily: 'JetBrains Mono, monospace',
                  }}>
                    {s.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            ) : null}
          </div>

          {/* Notes & Tags */}
          {(exitConfig?.notes || exitConfig?.tags?.length) ? (
            <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 8, padding: 16 }}>
              {exitConfig.tags?.length ? (
                <div style={{ marginBottom: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {exitConfig.tags.map(t => (
                    <span key={t} style={{ padding: '2px 8px', borderRadius: 3, fontSize: 10, background: '#1e1e1e', color: '#888', fontFamily: 'JetBrains Mono, monospace' }}>
                      #{t}
                    </span>
                  ))}
                </div>
              ) : null}
              {exitConfig.notes ? (
                <div style={{ fontSize: 12, color: '#888', lineHeight: 1.5 }}>
                  {exitConfig.notes}
                </div>
              ) : null}
            </div>
          ) : null}

          {/* Close position button */}
          <button
            style={{
              width: '100%', padding: '12px', borderRadius: 8,
              border: '1px solid #ff1744', background: 'rgba(255,23,68,0.1)',
              color: '#ff1744', fontSize: 13, fontWeight: 700,
              cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace',
              letterSpacing: '0.04em', marginBottom: 8,
            }}
          >
            CLOSE POSITION
          </button>
        </div>
      </div>

      {/* Exit Options Modal */}
      {editingExits && (
        <ExitOptionsForm
          symbol={position.symbol}
          config={exitConfig ?? null}
          onClose={() => setEditingExits(false)}
        />
      )}

      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
    </>
  )
}
