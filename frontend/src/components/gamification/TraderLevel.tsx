import { useQuery } from '@tanstack/react-query'
import { Trophy, Flame, Target, TrendingUp, Zap, Lock } from 'lucide-react'
import api from '../../api/client'

/**
 * Gamified trader-progress card — inspired by Robinhood's streaks and Binance's
 * reward hub. Everything here is DERIVED FROM REAL performance data
 * (/analytics/performance): XP = number of closed trades, level from XP
 * thresholds, badges unlocked by real win-rate / Sharpe / PnL. No fabricated
 * numbers — with zero trades it shows an honest "begin" state.
 */

interface Performance {
  total_trades: number
  total_pnl: number
  win_rate: number
  sharpe_ratio: number | null
  max_drawdown: number | null
}

// Level curve: XP needed scales so early levels come fast, later ones grind.
const LEVELS = [0, 10, 25, 50, 100, 200, 400, 750, 1500, 3000]
const LEVEL_TITLES = [
  'Rookie', 'Apprentice', 'Trader', 'Analyst', 'Strategist',
  'Quant', 'Portfolio Mgr', 'Desk Head', 'Partner', 'Legend',
]

function levelFromXp(xp: number): { level: number; title: string; floor: number; next: number | null } {
  let level = 1
  for (let i = 0; i < LEVELS.length; i++) {
    if (xp >= LEVELS[i]) level = i + 1
  }
  const idx = Math.min(level - 1, LEVELS.length - 1)
  const floor = LEVELS[idx]
  const next = idx + 1 < LEVELS.length ? LEVELS[idx + 1] : null
  return { level, title: LEVEL_TITLES[idx], floor, next }
}

interface Badge {
  id: string
  label: string
  icon: typeof Trophy
  unlocked: boolean
  hint: string
}

function badgesFor(p: Performance): Badge[] {
  return [
    { id: 'first', label: 'First Blood', icon: Zap, unlocked: p.total_trades >= 1, hint: 'Close your first trade' },
    { id: 'profitable', label: 'In the Green', icon: TrendingUp, unlocked: p.total_pnl > 0, hint: 'Reach positive total PnL' },
    { id: 'sharp', label: 'Sharp Shooter', icon: Target, unlocked: (p.sharpe_ratio ?? 0) >= 1, hint: 'Achieve Sharpe ≥ 1.0' },
    { id: 'consistent', label: 'Edge Found', icon: Flame, unlocked: p.win_rate >= 0.5 && p.total_trades >= 10, hint: 'Win rate ≥ 50% over 10+ trades' },
    { id: 'veteran', label: 'Centurion', icon: Trophy, unlocked: p.total_trades >= 100, hint: 'Close 100 trades' },
  ]
}

export default function TraderLevel() {
  const { data, isLoading, error } = useQuery<Performance>({
    queryKey: ['analytics-performance-gamify'],
    queryFn: () => api.get('/analytics/performance').then((r) => r.data),
    staleTime: 60_000,
    retry: false,
  })

  if (isLoading) {
    return <div className="kpi-card animate-pulse h-[120px]" aria-busy="true" />
  }
  if (error || !data) {
    return null // non-critical widget — fail silent rather than break the dashboard
  }

  const xp = data.total_trades
  const { level, title, floor, next } = levelFromXp(xp)
  const progress = next ? Math.min(100, Math.round(((xp - floor) / (next - floor)) * 100)) : 100
  const badges = badgesFor(data)
  const unlockedCount = badges.filter((b) => b.unlocked).length

  return (
    <div className="kpi-card relative overflow-hidden">
      {/* Glow accent */}
      <div
        className="absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl opacity-30 pointer-events-none"
        style={{ background: 'radial-gradient(circle, #00ff88, transparent 70%)' }}
        aria-hidden="true"
      />
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div
            className="w-11 h-11 rounded-xl flex items-center justify-center font-bold text-lg shrink-0"
            style={{ background: 'linear-gradient(135deg, #00ff88, #00d4ff)', color: '#06120c' }}
          >
            {level}
          </div>
          <div>
            <p className="font-bold text-[15px] leading-tight">{title}</p>
            <p className="text-[10px] text-[var(--muted)] uppercase tracking-widest">Trader Level {level}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="mono-num text-[15px] font-bold text-[#00ff88]">{xp} XP</p>
          <p className="text-[10px] text-[var(--muted)]">{unlockedCount}/{badges.length} badges</p>
        </div>
      </div>

      {/* XP progress bar */}
      <div className="mb-3">
        <div className="h-2 rounded-full bg-white/[0.06] overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${progress}%`, background: 'linear-gradient(90deg, #00ff88, #00d4ff)' }}
          />
        </div>
        <p className="text-[10px] text-[var(--muted)] mt-1">
          {next ? `${next - xp} trades to Level ${level + 1}` : 'Max level reached 🏆'}
        </p>
      </div>

      {/* Badge row */}
      <div className="flex items-center gap-2 flex-wrap">
        {badges.map((b) => {
          const Icon = b.unlocked ? b.icon : Lock
          return (
            <div
              key={b.id}
              title={b.unlocked ? b.label : `Locked — ${b.hint}`}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[10px] font-semibold transition-colors ${
                b.unlocked
                  ? 'border-[#00ff88]/30 bg-[#00ff88]/10 text-[#00ff88]'
                  : 'border-white/[0.06] bg-white/[0.02] text-[#666] '
              }`}
            >
              <Icon size={12} aria-hidden="true" />
              <span className="hidden sm:inline">{b.label}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
