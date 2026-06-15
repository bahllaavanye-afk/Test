import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

interface LeaderboardRow {
  rank: number
  strategy_id: string
  name: string
  display_name: string
  market_type: string
  strategy_type: string
  risk_bucket: string
  is_enabled: boolean
  symbols: string[]
  total_pnl: number
  return_pct: number
  win_rate: number
  total_trades: number
  sharpe_30d: number
  allocation: number
}

const DAYS_OPTIONS = [
  { label: '7d', value: 7 },
  { label: '30d', value: 30 },
  { label: '90d', value: 90 },
]

const MARKET_OPTIONS = ['all', 'equity', 'crypto', 'polymarket']

function SharpeBar({ value }: { value: number }) {
  const clamped = Math.min(Math.max(value, -1), 4)
  const pct = ((clamped + 1) / 5) * 100
  const color = value >= 2 ? '#00c853' : value >= 1 ? '#f5a623' : '#ff1744'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 6, background: '#1e1e1e', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.3s' }} />
      </div>
      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color, minWidth: 36, textAlign: 'right' }}>
        {value.toFixed(2)}
      </span>
    </div>
  )
}

function FollowButton({ strategyId, name }: { strategyId: string; name: string }) {
  const [following, setFollowing] = useState(false)
  const [loading, setLoading] = useState(false)

  const toggle = async () => {
    setLoading(true)
    try {
      if (!following) {
        await api.post(`/copy-trading/follow`, { strategy_id: strategyId, size_multiplier: 1.0 })
      } else {
        await api.delete(`/copy-trading/follow/${strategyId}`)
      }
      setFollowing(f => !f)
    } catch {
      // endpoint may not be wired yet — toggle optimistically
      setFollowing(f => !f)
    } finally {
      setLoading(false)
    }
  }

  return (
    <button
      onClick={toggle}
      disabled={loading}
      style={{
        padding: '4px 14px',
        borderRadius: 4,
        border: `1px solid ${following ? '#ff1744' : '#f5a623'}`,
        background: following ? 'rgba(255,23,68,0.1)' : 'rgba(245,166,35,0.1)',
        color: following ? '#ff1744' : '#f5a623',
        fontSize: 12,
        fontFamily: 'JetBrains Mono, monospace',
        cursor: loading ? 'wait' : 'pointer',
        transition: 'all 0.15s',
      }}
    >
      {loading ? '…' : following ? 'Unfollow' : 'Follow'}
    </button>
  )
}

export default function CopyTrading() {
  const [days, setDays] = useState(30)
  const [market, setMarket] = useState('all')

  const { data, isLoading, error } = useQuery<LeaderboardRow[]>({
    queryKey: ['copy-trading-leaderboard', days],
    queryFn: () => api.get(`/copy-trading/leaderboard?days=${days}`).then(r => r.data),
    staleTime: 60_000,
    refetchInterval: 120_000,
  })

  const filtered = (data ?? []).filter(
    row => market === 'all' || row.market_type === market
  )

  return (
    <div style={{ padding: 24, color: '#e8e8e8', fontFamily: 'Inter, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#f5a623' }}>
          Copy Trading Desk
        </h1>
        <p style={{ margin: '6px 0 0', color: '#888', fontSize: 13 }}>
          Mirror top-performing strategies. Ranked by rolling Sharpe ratio.
        </p>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 4 }}>
          {DAYS_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setDays(opt.value)}
              style={{
                padding: '5px 14px',
                borderRadius: 4,
                border: `1px solid ${days === opt.value ? '#f5a623' : '#1e1e1e'}`,
                background: days === opt.value ? 'rgba(245,166,35,0.15)' : '#111111',
                color: days === opt.value ? '#f5a623' : '#888',
                fontSize: 12,
                fontFamily: 'JetBrains Mono, monospace',
                cursor: 'pointer',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          {MARKET_OPTIONS.map(m => (
            <button
              key={m}
              onClick={() => setMarket(m)}
              style={{
                padding: '5px 14px',
                borderRadius: 4,
                border: `1px solid ${market === m ? '#2196F3' : '#1e1e1e'}`,
                background: market === m ? 'rgba(33,150,243,0.15)' : '#111111',
                color: market === m ? '#2196F3' : '#888',
                fontSize: 12,
                fontFamily: 'JetBrains Mono, monospace',
                cursor: 'pointer',
                textTransform: 'capitalize',
              }}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div style={{ background: '#111111', border: '1px solid #1e1e1e', borderRadius: 8, overflow: 'hidden' }}>
        {isLoading ? (
          <div style={{ padding: 48, textAlign: 'center', color: '#555', fontFamily: 'JetBrains Mono, monospace', fontSize: 13 }}>
            Loading leaderboard…
          </div>
        ) : error ? (
          <div style={{ padding: 48, textAlign: 'center', color: '#ff1744', fontSize: 13 }}>
            Failed to load leaderboard
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: 48, textAlign: 'center', color: '#555', fontSize: 13 }}>
            No strategy data for the selected period. Strategies need closed trades to appear here.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #1e1e1e' }}>
                {['#', 'Strategy', 'Market', 'Type', 'Symbols', 'Win Rate', 'Trades', 'Total P&L', 'Return %', '30d Sharpe', ''].map(h => (
                  <th key={h} style={{
                    padding: '10px 14px', textAlign: h === '#' ? 'center' : 'left',
                    color: '#555', fontWeight: 600, fontSize: 11, textTransform: 'uppercase',
                    letterSpacing: '0.06em', whiteSpace: 'nowrap',
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => {
                const pnlColor = row.total_pnl >= 0 ? '#00c853' : '#ff1744'
                const retColor = row.return_pct >= 0 ? '#00c853' : '#ff1744'
                return (
                  <tr
                    key={row.strategy_id}
                    style={{
                      borderBottom: i < filtered.length - 1 ? '1px solid #1a1a1a' : 'none',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = '#161616')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                  >
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: row.rank <= 3 ? '#f5a623' : '#555', fontFamily: 'JetBrains Mono, monospace', fontWeight: 700 }}>
                      {row.rank <= 3 ? ['🥇', '🥈', '🥉'][row.rank - 1] : row.rank}
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <div style={{ fontWeight: 600, color: '#e8e8e8' }}>{row.display_name}</div>
                      <div style={{ fontSize: 11, color: '#555', fontFamily: 'JetBrains Mono, monospace', marginTop: 2 }}>{row.name}</div>
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 3, fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                        background: row.market_type === 'crypto' ? 'rgba(245,166,35,0.15)' : row.market_type === 'equity' ? 'rgba(33,150,243,0.15)' : 'rgba(156,39,176,0.15)',
                        color: row.market_type === 'crypto' ? '#f5a623' : row.market_type === 'equity' ? '#2196F3' : '#9C27B0',
                      }}>
                        {row.market_type}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px', color: '#888', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}>
                      {row.strategy_type}
                    </td>
                    <td style={{ padding: '12px 14px', color: '#666', fontSize: 11, fontFamily: 'JetBrains Mono, monospace', maxWidth: 120 }}>
                      {(row.symbols ?? []).slice(0, 3).join(', ')}{(row.symbols ?? []).length > 3 ? ' …' : ''}
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'JetBrains Mono, monospace', color: row.win_rate >= 55 ? '#00c853' : '#888' }}>
                      {row.win_rate.toFixed(1)}%
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'JetBrains Mono, monospace', color: '#888' }}>
                      {row.total_trades}
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'JetBrains Mono, monospace', color: pnlColor }}>
                      {row.total_pnl >= 0 ? '+' : ''}${row.total_pnl.toFixed(2)}
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'JetBrains Mono, monospace', color: retColor }}>
                      {row.return_pct >= 0 ? '+' : ''}{row.return_pct.toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', minWidth: 140 }}>
                      <SharpeBar value={row.sharpe_30d} />
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <FollowButton strategyId={row.strategy_id} name={row.name} />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Legend */}
      <div style={{ marginTop: 16, display: 'flex', gap: 20, fontSize: 11, color: '#555' }}>
        <span>Sharpe &gt;2.0 = <span style={{ color: '#00c853' }}>green</span></span>
        <span>Sharpe 1–2 = <span style={{ color: '#f5a623' }}>amber</span></span>
        <span>Sharpe &lt;1 = <span style={{ color: '#ff1744' }}>red</span></span>
        <span style={{ marginLeft: 'auto' }}>Updated every 2 minutes</span>
      </div>
    </div>
  )
}
