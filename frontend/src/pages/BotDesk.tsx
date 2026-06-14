/**
 * BotDesk — Options Alpha-style command center.
 * Shows all bots grouped by desk with P&L, win rate, and live status.
 * Each desk is a "trading pod" that runs 24/7 via APScheduler — no LLM needed.
 */
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import {
  Bot, TrendingUp, Bitcoin, Layers, BarChart2, Globe, Coins, Circle,
  Play, Pause, RefreshCw, ChevronDown, ChevronRight, Zap, Brain, GitBranch,
  DollarSign, Target, Activity, AlertTriangle,
} from 'lucide-react'

// ── Types ────────────────────────────────────────────────────────────────────

interface BotSummary {
  id: string
  name: string
  symbol: string
  market_type: string
  desk: string
  signal_source: string
  ml_model_name: string | null
  ml_confidence_threshold: number | null
  is_enabled: boolean
  run_count: number
  last_run_at: string | null
  last_signal: string | null
  last_result: Record<string, unknown> | null
  open_positions: number
  trades_30d: number
  pnl_30d: number
}

const DESK_META: Record<string, { label: string; icon: React.ElementType; color: string }> = {
  equity:      { label: 'Equity',      icon: TrendingUp,  color: '#2196F3' },
  crypto:      { label: 'Crypto',      icon: Bitcoin,     color: '#f5a623' },
  options:     { label: 'Options',     icon: Layers,      color: '#9C27B0' },
  futures:     { label: 'Futures',     icon: BarChart2,   color: '#FF5722' },
  fx:          { label: 'FX',          icon: Globe,       color: '#009688' },
  commodities: { label: 'Commodities', icon: Coins,       color: '#FF9800' },
  polymarket:  { label: 'Polymarket',  icon: Target,      color: '#E91E63' },
}

const SIGNAL_SOURCE_BADGES: Record<string, { label: string; color: string }> = {
  rule_based: { label: 'Rule-Based', color: '#2196F3' },
  ml_signal:  { label: 'ML Signal',  color: '#9C27B0' },
  hybrid:     { label: 'Hybrid',     color: '#00BCD4' },
}

const SIGNAL_COLORS: Record<string, string> = {
  buy:  '#00c853',
  sell: '#ff1744',
  hold: '#888',
  alert: '#f5a623',
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtPnl(v: number): string {
  return (v >= 0 ? '+' : '') + v.toFixed(2)
}

function fmtAgo(ts: string | null): string {
  if (!ts) return 'never'
  const diff = (Date.now() - new Date(ts).getTime()) / 1000
  if (diff < 60) return `${Math.round(diff)}s ago`
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`
  return `${Math.round(diff / 86400)}d ago`
}

// ── DeskCard ─────────────────────────────────────────────────────────────────

function DeskCard({
  desk,
  bots,
  onToggle,
}: {
  desk: string
  bots: BotSummary[]
  onToggle: (id: string, enabled: boolean) => void
}) {
  const [expanded, setExpanded] = useState(true)
  const meta = DESK_META[desk] ?? { label: desk, icon: Bot, color: '#888' }
  const DeskIcon = meta.icon

  const enabled = bots.filter(b => b.is_enabled).length
  const totalPnl = bots.reduce((s, b) => s + (b.pnl_30d || 0), 0)
  const totalTrades = bots.reduce((s, b) => s + (b.trades_30d || 0), 0)
  const openPos = bots.reduce((s, b) => s + (b.open_positions || 0), 0)

  return (
    <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 10, overflow: 'hidden', marginBottom: 16 }}>
      {/* Desk Header */}
      <div
        onClick={() => setExpanded(e => !e)}
        style={{
          display: 'flex', alignItems: 'center', gap: 12, padding: '12px 16px',
          cursor: 'pointer', background: '#161616', borderBottom: expanded ? '1px solid #1e1e1e' : 'none',
        }}
      >
        <DeskIcon size={16} color={meta.color} />
        <span style={{ fontWeight: 700, color: '#e8e8e8', fontSize: 14 }}>{meta.label} Desk</span>
        <span style={{ fontSize: 12, color: '#666' }}>{bots.length} bots · {enabled} active</span>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 24, alignItems: 'center' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: '#666' }}>30d P&L</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: totalPnl >= 0 ? '#00c853' : '#ff1744', fontFamily: 'JetBrains Mono, monospace' }}>
              {fmtPnl(totalPnl)}%
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: '#666' }}>Trades</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#e8e8e8', fontFamily: 'JetBrains Mono, monospace' }}>{totalTrades}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: '#666' }}>Open</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: openPos > 0 ? '#f5a623' : '#888', fontFamily: 'JetBrains Mono, monospace' }}>{openPos}</div>
          </div>
          {expanded ? <ChevronDown size={14} color="#666" /> : <ChevronRight size={14} color="#666" />}
        </div>
      </div>

      {/* Bot Rows */}
      {expanded && (
        <div>
          {bots.map(bot => (
            <BotRow key={bot.id} bot={bot} onToggle={onToggle} />
          ))}
        </div>
      )}
    </div>
  )
}

// ── BotRow ───────────────────────────────────────────────────────────────────

function BotRow({ bot, onToggle }: { bot: BotSummary; onToggle: (id: string, enabled: boolean) => void }) {
  const srcBadge = SIGNAL_SOURCE_BADGES[bot.signal_source] ?? { label: bot.signal_source, color: '#888' }

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 90px 110px 80px 80px 70px 70px 80px',
        gap: 8,
        padding: '10px 16px',
        borderBottom: '1px solid #1a1a1a',
        alignItems: 'center',
        fontSize: 12,
        opacity: bot.is_enabled ? 1 : 0.55,
      }}
    >
      {/* Name + symbol */}
      <div>
        <div style={{ color: '#e8e8e8', fontWeight: 600, marginBottom: 2 }}>{bot.name}</div>
        <div style={{ color: '#666', fontSize: 11 }}>{bot.symbol} · {bot.market_type}</div>
      </div>

      {/* Signal source */}
      <div>
        <span style={{
          background: srcBadge.color + '22', color: srcBadge.color,
          border: `1px solid ${srcBadge.color}44`, borderRadius: 4,
          padding: '2px 6px', fontSize: 10, fontWeight: 700,
          display: 'inline-flex', alignItems: 'center', gap: 3,
        }}>
          {bot.signal_source === 'ml_signal' || bot.signal_source === 'hybrid'
            ? <Brain size={9} />
            : <GitBranch size={9} />}
          {srcBadge.label}
        </span>
      </div>

      {/* Last signal */}
      <div style={{ color: SIGNAL_COLORS[bot.last_signal ?? 'hold'] ?? '#888', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
        {bot.last_signal ?? '—'}
        {bot.last_signal && (
          <div style={{ color: '#555', fontSize: 10, fontWeight: 400 }}>{fmtAgo(bot.last_run_at)}</div>
        )}
      </div>

      {/* Open positions */}
      <div style={{ textAlign: 'center', color: bot.open_positions > 0 ? '#f5a623' : '#555', fontFamily: 'JetBrains Mono, monospace' }}>
        {bot.open_positions > 0 ? (
          <span style={{ background: '#f5a62322', border: '1px solid #f5a62344', borderRadius: 4, padding: '2px 6px' }}>
            {bot.open_positions}
          </span>
        ) : '—'}
      </div>

      {/* 30d trades */}
      <div style={{ textAlign: 'center', color: '#888', fontFamily: 'JetBrains Mono, monospace' }}>
        {bot.trades_30d}
      </div>

      {/* 30d P&L */}
      <div style={{ textAlign: 'right', color: (bot.pnl_30d ?? 0) >= 0 ? '#00c853' : '#ff1744', fontFamily: 'JetBrains Mono, monospace', fontWeight: 700 }}>
        {fmtPnl(bot.pnl_30d ?? 0)}%
      </div>

      {/* Run count */}
      <div style={{ textAlign: 'center', color: '#555', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
        {bot.run_count}↻
      </div>

      {/* Toggle */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button
          onClick={() => onToggle(bot.id, !bot.is_enabled)}
          style={{
            background: bot.is_enabled ? '#00c85322' : '#33333344',
            border: `1px solid ${bot.is_enabled ? '#00c85355' : '#333'}`,
            borderRadius: 20, cursor: 'pointer', padding: '4px 10px',
            color: bot.is_enabled ? '#00c853' : '#555', fontSize: 11, fontWeight: 700,
            display: 'flex', alignItems: 'center', gap: 4,
          }}
        >
          {bot.is_enabled ? <Play size={10} /> : <Pause size={10} />}
          {bot.is_enabled ? 'ON' : 'OFF'}
        </button>
      </div>
    </div>
  )
}

// ── Stats Bar ─────────────────────────────────────────────────────────────────

function StatsBar({ bots }: { bots: BotSummary[] }) {
  const totalEnabled = bots.filter(b => b.is_enabled).length
  const totalOpen = bots.reduce((s, b) => s + (b.open_positions || 0), 0)
  const totalTrades = bots.reduce((s, b) => s + (b.trades_30d || 0), 0)
  const totalPnl = bots.reduce((s, b) => s + (b.pnl_30d || 0), 0)
  const mlBots = bots.filter(b => b.signal_source !== 'rule_based').length

  const stats = [
    { label: 'Active Bots',   value: `${totalEnabled}/${bots.length}`, icon: Activity, color: '#00c853' },
    { label: 'Open Positions', value: totalOpen,                         icon: Circle,   color: '#f5a623' },
    { label: 'Trades (30d)',   value: totalTrades,                       icon: RefreshCw, color: '#2196F3' },
    { label: 'P&L (30d)',      value: `${fmtPnl(totalPnl)}%`,           icon: DollarSign, color: totalPnl >= 0 ? '#00c853' : '#ff1744' },
    { label: 'ML/Hybrid Bots', value: mlBots,                           icon: Brain,    color: '#9C27B0' },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12, marginBottom: 24 }}>
      {stats.map(({ label, value, icon: Icon, color }) => (
        <div key={label} style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 8, padding: '12px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <Icon size={14} color={color} />
            <span style={{ fontSize: 11, color: '#666' }}>{label}</span>
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#e8e8e8', fontFamily: 'JetBrains Mono, monospace' }}>{value}</div>
        </div>
      ))}
    </div>
  )
}

// ── Architecture Info ─────────────────────────────────────────────────────────

function ArchitecturePanel() {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 10, marginBottom: 16 }}>
      <div
        onClick={() => setOpen(o => !o)}
        style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '12px 16px', cursor: 'pointer' }}
      >
        <Zap size={14} color="#f5a623" />
        <span style={{ color: '#e8e8e8', fontWeight: 600, fontSize: 13 }}>How Autopilot Works</span>
        <AlertTriangle size={12} color="#00c853" style={{ marginLeft: 4 }} />
        <span style={{ color: '#00c853', fontSize: 11 }}>24/7 — No LLM Needed</span>
        <span style={{ marginLeft: 'auto', color: '#555', fontSize: 11 }}>{open ? '▲ hide' : '▼ show'}</span>
      </div>
      {open && (
        <div style={{ padding: '0 16px 16px', fontSize: 12, color: '#888', lineHeight: 1.6 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            {[
              {
                title: '🔄 APScheduler Loop',
                color: '#2196F3',
                lines: [
                  'BotRunner loads all enabled bots at startup',
                  'Schedules each bot at its configured interval (1m–1d)',
                  'Runs even when Claude is offline or out of tokens',
                  'Interval map: 1m, 5m, 15m, 30m, 1h, 4h, 1d',
                ],
              },
              {
                title: '📊 Signal Sources',
                color: '#9C27B0',
                lines: [
                  'Rule-Based: RSI, MACD, BB, EMA, ADX, Stochastic, CCI',
                  'ML Signal: LSTM/XGBoost/Lorentzian confidence ≥ threshold',
                  'Hybrid: rule conditions pass AND ML confirms direction',
                  'Fails open if ML model unavailable (rules still trade)',
                ],
              },
              {
                title: '🏛️ Multi-Desk Execution',
                color: '#f5a623',
                lines: [
                  'Same signal engine across all desks (equity/crypto/options/futures/fx/commodities)',
                  'Paper orders → Trade records on TP/SL hit',
                  'RL execution agent minimizes slippage (TWAP fallback)',
                  'Position monitor checks exits every 5 minutes',
                ],
              },
            ].map(({ title, color, lines }) => (
              <div key={title} style={{ background: '#0a0a0a', borderRadius: 6, padding: 12, borderLeft: `3px solid ${color}` }}>
                <div style={{ color, fontWeight: 700, marginBottom: 8, fontSize: 12 }}>{title}</div>
                {lines.map(l => (
                  <div key={l} style={{ color: '#777', fontSize: 11, marginBottom: 4 }}>• {l}</div>
                ))}
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12, padding: '8px 12px', background: '#0a0a0a', borderRadius: 6, border: '1px solid #1e1e1e' }}>
            <span style={{ color: '#555', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
              Pipeline: Market Data → Indicator Engine → Signal Source (rule/ML/hybrid) → Risk Gate → Paper Order → Position Monitor → Trade Record → P&L
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function BotDesk() {
  const queryClient = useQueryClient()
  const [filterDesk, setFilterDesk] = useState<string>('all')

  const { data: summary = [], isLoading, error } = useQuery<BotSummary[]>({
    queryKey: ['bots-summary'],
    queryFn: () => api.get('/bots/summary/all').then((r: { data: BotSummary[] }) => r.data),
    refetchInterval: 30_000,
  })

  const toggleMut = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      api.patch(`/bots/${id}`, { is_enabled: enabled }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bots-summary'] }),
  })

  const onToggle = (id: string, enabled: boolean) => toggleMut.mutate({ id, enabled })

  // Group by desk
  const desks = [...new Set(summary.map(b => b.desk || 'equity'))].sort()
  const byDesk: Record<string, BotSummary[]> = {}
  for (const desk of desks) {
    byDesk[desk] = summary.filter(b => (b.desk || 'equity') === desk)
  }

  const filtered = filterDesk === 'all' ? desks : desks.filter(d => d === filterDesk)

  if (isLoading) return (
    <div style={{ padding: 32, color: '#555', fontFamily: 'JetBrains Mono, monospace' }}>Loading desks…</div>
  )
  if (error) return (
    <div style={{ padding: 32, color: '#ff1744' }}>Failed to load bot desk: {error instanceof Error ? error.message : 'Unknown error'}</div>
  )

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <Bot size={20} color="#f5a623" />
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#e8e8e8' }}>Bot Command Center</h1>
        <span style={{ fontSize: 11, color: '#00c853', background: '#00c85322', border: '1px solid #00c85344', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>
          AUTOPILOT
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button
            onClick={() => queryClient.invalidateQueries({ queryKey: ['bots-summary'] })}
            style={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 6, padding: '6px 12px', color: '#888', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      {/* Stats bar */}
      {summary.length > 0 && <StatsBar bots={summary} />}

      {/* Architecture explanation */}
      <ArchitecturePanel />

      {/* Desk filter tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {['all', ...desks].map(desk => {
          const meta = DESK_META[desk]
          const active = filterDesk === desk
          return (
            <button
              key={desk}
              onClick={() => setFilterDesk(desk)}
              style={{
                background: active ? (meta?.color ?? '#f5a623') + '22' : '#111',
                border: `1px solid ${active ? (meta?.color ?? '#f5a623') + '66' : '#1e1e1e'}`,
                borderRadius: 6, padding: '6px 14px', cursor: 'pointer',
                color: active ? (meta?.color ?? '#f5a623') : '#888',
                fontSize: 12, fontWeight: active ? 700 : 400,
                display: 'flex', alignItems: 'center', gap: 6,
              }}
            >
              {meta && <meta.icon size={12} />}
              {desk === 'all' ? 'All Desks' : (meta?.label ?? desk)}
              <span style={{ fontSize: 10, opacity: 0.7 }}>
                {desk === 'all' ? summary.length : (byDesk[desk]?.length ?? 0)}
              </span>
            </button>
          )
        })}
      </div>

      {/* Column headers */}
      {summary.length > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 90px 110px 80px 80px 70px 70px 80px',
          gap: 8, padding: '6px 16px', fontSize: 10, color: '#444', marginBottom: 4, fontWeight: 700,
        }}>
          <div>BOT / SYMBOL</div>
          <div>SOURCE</div>
          <div>LAST SIGNAL</div>
          <div style={{ textAlign: 'center' }}>OPEN POS</div>
          <div style={{ textAlign: 'center' }}>TRADES 30D</div>
          <div style={{ textAlign: 'right' }}>P&L 30D</div>
          <div style={{ textAlign: 'center' }}>RUNS</div>
          <div style={{ textAlign: 'right' }}>STATUS</div>
        </div>
      )}

      {/* Desk cards */}
      {summary.length === 0 ? (
        <div style={{
          background: '#111', border: '1px solid #1e1e1e', borderRadius: 10,
          padding: 48, textAlign: 'center', color: '#555',
        }}>
          <Bot size={32} color="#333" style={{ marginBottom: 12 }} />
          <div style={{ fontSize: 14, marginBottom: 8 }}>No bots configured yet</div>
          <div style={{ fontSize: 12 }}>Go to Bot Builder to create your first automated trading bot</div>
        </div>
      ) : (
        filtered.map(desk => (
          <DeskCard
            key={desk}
            desk={desk}
            bots={byDesk[desk] ?? []}
            onToggle={onToggle}
          />
        ))
      )}
    </div>
  )
}
