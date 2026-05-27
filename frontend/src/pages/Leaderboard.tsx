import { useState, useMemo } from 'react'

type StrategyType = 'MANUAL' | 'ML' | 'ARB'

interface Strategy {
  id: string
  name: string
  type: StrategyType
  active: boolean
  sharpe: number
  annualReturn: number
  maxDd: number
  winRate: number
  trades: number
  avgHold: string
  lastSignal: string
  lastSignalTime: string
  lastConfidence: number
  seed: number
}

const ALL_STRATEGIES: Strategy[] = [
  { id: 'pairs_trading', name: 'Pairs Trading', type: 'MANUAL', active: true, sharpe: 1.82, annualReturn: 31.2, maxDd: -4.1, winRate: 81, trades: 243, avgHold: '4h 12m', lastSignal: 'LONG AAPL/MSFT spread', lastSignalTime: '2:41 PM', lastConfidence: 91, seed: 11 },
  { id: 'triangular_arb', name: 'Triangular Arb', type: 'ARB', active: true, sharpe: 2.41, annualReturn: 18.9, maxDd: -1.2, winRate: 94, trades: 1847, avgHold: '3m 8s', lastSignal: 'BTC→ETH→USDT cycle', lastSignalTime: '2:44 PM', lastConfidence: 99, seed: 7 },
  { id: 'poly_binary_arb', name: 'Poly Binary Arb', type: 'ARB', active: true, sharpe: 2.87, annualReturn: 22.3, maxDd: -0.4, winRate: 98, trades: 412, avgHold: '18m', lastSignal: 'YES token mispricing 2.1%', lastSignalTime: '2:43 PM', lastConfidence: 98, seed: 3 },
  { id: 'ensemble', name: 'Ensemble Model', type: 'ML', active: true, sharpe: 1.89, annualReturn: 38.7, maxDd: -6.8, winRate: 76, trades: 89, avgHold: '3d 4h', lastSignal: 'LONG SPY stacking signal', lastSignalTime: '2:38 PM', lastConfidence: 88, seed: 17 },
  { id: 'ml_pca_arb', name: 'ML PCA Arb', type: 'ML', active: true, sharpe: 1.94, annualReturn: 31.4, maxDd: -4.8, winRate: 82, trades: 167, avgHold: '1d 2h', lastSignal: 'Short PC1 factor loading', lastSignalTime: '2:31 PM', lastConfidence: 84, seed: 23 },
  { id: 'pca_stat_arb', name: 'PCA Stat Arb', type: 'MANUAL', active: true, sharpe: 1.67, annualReturn: 27.8, maxDd: -5.2, winRate: 79, trades: 189, avgHold: '6h 44m', lastSignal: 'LONG QQQ vs SPY divergence', lastSignalTime: '2:27 PM', lastConfidence: 86, seed: 29 },
  { id: 'ml_momentum', name: 'ML Momentum', type: 'ML', active: true, sharpe: 1.67, annualReturn: 34.2, maxDd: -7.1, winRate: 74, trades: 118, avgHold: '2d 6h', lastSignal: 'LONG AAPL momentum signal', lastSignalTime: '2:38 PM', lastConfidence: 87, seed: 37 },
  { id: 'ml_breakout', name: 'ML Breakout', type: 'ML', active: true, sharpe: 1.43, annualReturn: 29.1, maxDd: -8.4, winRate: 69, trades: 112, avgHold: '1d 8h', lastSignal: 'LONG GOOGL breakout above $180', lastSignalTime: '2:29 PM', lastConfidence: 83, seed: 41 },
  { id: 'ml_mean_reversion', name: 'ML Mean Rev.', type: 'ML', active: true, sharpe: 1.31, annualReturn: 24.8, maxDd: -9.2, winRate: 72, trades: 87, avgHold: '8h 30m', lastSignal: 'SHORT AMZN reversion target $183', lastSignalTime: '2:21 PM', lastConfidence: 79, seed: 43 },
  { id: 'momentum', name: 'Momentum Alpha', type: 'MANUAL', active: true, sharpe: 1.24, annualReturn: 28.4, maxDd: -8.2, winRate: 71, trades: 142, avgHold: '3d 12h', lastSignal: 'LONG SPY cross above 20-EMA', lastSignalTime: '2:14 PM', lastConfidence: 87, seed: 53 },
  { id: 'lorentzian_knn', name: 'Lorentzian KNN', type: 'ML', active: true, sharpe: 1.31, annualReturn: 23.4, maxDd: -10.1, winRate: 67, trades: 201, avgHold: '14h', lastSignal: 'LONG MSFT KNN cluster signal', lastSignalTime: '2:11 PM', lastConfidence: 74, seed: 59 },
  { id: 'breakout', name: 'Breakout Scanner', type: 'MANUAL', active: true, sharpe: 0.97, annualReturn: 21.7, maxDd: -9.8, winRate: 65, trades: 134, avgHold: '1d 6h', lastSignal: 'LONG QQQ channel breakout', lastSignalTime: '2:03 PM', lastConfidence: 78, seed: 61 },
  { id: 'mean_reversion', name: 'Mean Reversion', type: 'MANUAL', active: false, sharpe: 1.08, annualReturn: 19.3, maxDd: -11.4, winRate: 68, trades: 98, avgHold: '12h 20m', lastSignal: 'LONG IWM z-score neg 1.8sigma', lastSignalTime: '11:42 AM', lastConfidence: 81, seed: 67 },
  { id: 'supertrend', name: 'SuperTrend', type: 'MANUAL', active: true, sharpe: 0.91, annualReturn: 16.8, maxDd: -12.3, winRate: 63, trades: 156, avgHold: '2d 4h', lastSignal: 'LONG SPY supertrend flip', lastSignalTime: '1:52 PM', lastConfidence: 71, seed: 71 },
  { id: 'rl_trader', name: 'RL Trader (PPO)', type: 'ML', active: false, sharpe: 1.12, annualReturn: 19.8, maxDd: -11.2, winRate: 61, trades: 143, avgHold: '1d 18h', lastSignal: 'LONG QQQ policy gradient', lastSignalTime: '9:15 AM', lastConfidence: 68, seed: 73 },
  { id: 'rsi_macd', name: 'RSI + MACD', type: 'MANUAL', active: true, sharpe: 0.79, annualReturn: 14.2, maxDd: -13.1, winRate: 62, trades: 187, avgHold: '6h 45m', lastSignal: 'SHORT QQQ MACD bearish cross', lastSignalTime: '1:31 PM', lastConfidence: 74, seed: 79 },
  { id: 'low_volatility', name: 'Low Volatility', type: 'MANUAL', active: false, sharpe: 0.73, annualReturn: 11.2, maxDd: -7.9, winRate: 58, trades: 67, avgHold: '5d 2h', lastSignal: 'LONG USMV rebalance signal', lastSignalTime: 'Yesterday', lastConfidence: 63, seed: 83 },
]

function seededRand(seed: number) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
}

function Sparkline({ seed, color, width = 140, height = 40 }: { seed: number; color: string; width?: number; height?: number }) {
  const rand = seededRand(seed)
  const points: number[] = []
  let v = 100
  for (let i = 0; i < 30; i++) {
    v += (rand() - 0.46) * 4
    points.push(v)
  }
  const min = Math.min(...points)
  const max = Math.max(...points)
  const range = max - min || 1
  const step = width / (points.length - 1)
  const toY = (val: number) => height - 2 - ((val - min) / range) * (height - 4)
  const pts = points.map((p, i) => `${(i * step).toFixed(1)},${toY(p).toFixed(1)}`)
  const linePath = `M ${pts.join(' L ')}`
  const areaPath = `M 0,${height} L ${linePath.slice(2)} L ${((points.length - 1) * step).toFixed(1)},${height} Z`
  return (
    <svg width={width} height={height} style={{ overflow: 'visible', display: 'block' }}>
      <defs>
        <linearGradient id={`sg-${seed}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#sg-${seed})`} />
      <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function TypeBadge({ type }: { type: StrategyType }) {
  const map: Record<StrategyType, { bg: string; color: string }> = {
    MANUAL: { bg: 'rgba(41,121,255,0.12)', color: '#2979ff' },
    ML: { bg: 'rgba(156,39,176,0.12)', color: '#ce93d8' },
    ARB: { bg: 'rgba(0,200,83,0.12)', color: '#00c853' },
  }
  const style = map[type]
  return (
    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded tracking-widest"
      style={{ background: style.bg, color: style.color, border: `1px solid ${style.color}30` }}>
      {type}
    </span>
  )
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={e => { e.stopPropagation(); onChange(!checked) }}
      className="relative inline-flex h-5 w-9 items-center rounded-full transition-all duration-300 focus:outline-none flex-shrink-0"
      style={{ background: checked ? '#00c853' : '#1e1e1e', border: `1px solid ${checked ? '#00c853' : '#333'}`, boxShadow: checked ? '0 0 8px rgba(0,200,83,0.4)' : 'none' }}
    >
      <span className="inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform duration-300"
        style={{ transform: checked ? 'translateX(18px)' : 'translateX(2px)', boxShadow: '0 1px 3px rgba(0,0,0,0.5)' }} />
    </button>
  )
}

function StatCell({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <p className="text-[9px] text-[#444] uppercase tracking-wider font-medium">{label}</p>
      <p className="text-xs font-bold font-mono" style={{ color: color ?? '#e8e8e8' }}>{value}</p>
    </div>
  )
}

function StrategyCard({ strategy, toggled, onToggle, rank }: {
  strategy: Strategy; toggled: boolean; onToggle: (id: string, val: boolean) => void; rank: number
}) {
  const isActive = toggled
  const returnColor = strategy.annualReturn >= 25 ? '#00c853' : strategy.annualReturn >= 15 ? '#f5a623' : '#888'
  const sharpeColor = strategy.sharpe >= 1.5 ? '#00c853' : strategy.sharpe >= 1.0 ? '#f5a623' : '#ff1744'
  const sparkColor = isActive ? (strategy.annualReturn >= 20 ? '#00c853' : '#f5a623') : '#444'
  const borderColor = !isActive ? '#1a1a1a' : strategy.sharpe >= 1.5 ? 'rgba(0,200,83,0.25)' : 'rgba(245,166,35,0.2)'
  const isSignalBuy = strategy.lastSignal.startsWith('LONG')
  return (
    <div className="group relative flex flex-col bg-[#111111] rounded-xl overflow-hidden transition-all duration-300 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-black/50"
      style={{ border: `1px solid ${borderColor}` }}>
      {rank <= 3 && isActive && (
        <div className="absolute top-0 right-0 w-7 h-7 flex items-center justify-center text-[10px] font-black rounded-bl-lg"
          style={{ background: rank === 1 ? '#f5a623' : rank === 2 ? '#9e9e9e' : '#cd7f32', color: '#000' }}>
          #{rank}
        </div>
      )}
      <div className="px-4 pt-4 pb-3 flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <div className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: isActive ? '#00c853' : '#333', boxShadow: isActive ? '0 0 6px rgba(0,200,83,0.7)' : 'none' }} />
            <h3 className="text-sm font-bold text-[#e8e8e8] truncate">{strategy.name}</h3>
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            <TypeBadge type={strategy.type} />
            <span className="text-[9px] font-medium px-1.5 py-0.5 rounded"
              style={{ background: isActive ? 'rgba(0,200,83,0.08)' : '#161616', color: isActive ? '#00c853' : '#444' }}>
              {isActive ? '● ACTIVE' : '○ INACTIVE'}
            </span>
          </div>
        </div>
        <Toggle checked={isActive} onChange={v => onToggle(strategy.id, v)} />
      </div>
      <div className="px-4 pb-3 flex items-end justify-between gap-3">
        <Sparkline seed={strategy.seed} color={sparkColor} width={120} height={38} />
        <div className="text-right flex-shrink-0">
          <p className="text-[9px] text-[#444] uppercase">Annual</p>
          <p className="text-base font-bold font-mono" style={{ color: returnColor }}>+{strategy.annualReturn.toFixed(1)}%</p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-px bg-[#161616] border-t border-b border-[#1a1a1a]">
        <div className="bg-[#111111] px-3 py-2.5"><StatCell label="Sharpe" value={strategy.sharpe.toFixed(2)} color={sharpeColor} /></div>
        <div className="bg-[#111111] px-3 py-2.5"><StatCell label="Max DD" value={`${strategy.maxDd.toFixed(1)}%`} color="#ff1744" /></div>
        <div className="bg-[#111111] px-3 py-2.5"><StatCell label="Win Rate" value={`${strategy.winRate}%`} color={strategy.winRate >= 70 ? '#00c853' : strategy.winRate >= 60 ? '#f5a623' : '#888'} /></div>
        <div className="bg-[#111111] px-3 py-2.5"><StatCell label="Trades" value={strategy.trades.toLocaleString()} /></div>
        <div className="bg-[#111111] px-3 py-2.5"><StatCell label="Avg Hold" value={strategy.avgHold} /></div>
        <div className="bg-[#111111] px-3 py-2.5"><StatCell label="Type" value={strategy.type} color={strategy.type === 'ARB' ? '#00c853' : strategy.type === 'ML' ? '#ce93d8' : '#2979ff'} /></div>
      </div>
      <div className="px-4 py-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded flex-shrink-0"
            style={{ background: isSignalBuy ? 'rgba(0,200,83,0.12)' : 'rgba(255,23,68,0.12)', color: isSignalBuy ? '#00c853' : '#ff1744', border: `1px solid ${isSignalBuy ? 'rgba(0,200,83,0.25)' : 'rgba(255,23,68,0.25)'}` }}>
            {isSignalBuy ? '▲' : '▼'}
          </span>
          <span className="text-[10px] text-[#888] truncate">{strategy.lastSignal}</span>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className="text-[9px] text-[#444]">{strategy.lastSignalTime}</span>
          <span className="text-[9px] font-mono text-[#2979ff]">{strategy.lastConfidence}%</span>
        </div>
      </div>
    </div>
  )
}

type FilterType = 'ALL' | 'MANUAL' | 'ML' | 'ARB'
type SortKey = 'sharpe' | 'return' | 'winrate' | 'trades'

export default function Leaderboard() {
  const [toggleMap, setToggleMap] = useState<Record<string, boolean>>(
    Object.fromEntries(ALL_STRATEGIES.map(s => [s.id, s.active]))
  )
  const [filterType, setFilterType] = useState<FilterType>('ALL')
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'inactive'>('all')
  const [sortBy, setSortBy] = useState<SortKey>('sharpe')
  const [search, setSearch] = useState('')

  function handleToggle(id: string, val: boolean) {
    setToggleMap(prev => ({ ...prev, [id]: val }))
  }

  const activeCount = Object.values(toggleMap).filter(Boolean).length

  const filtered = useMemo(() => {
    let list = [...ALL_STRATEGIES]
    if (search) {
      const q = search.toLowerCase()
      list = list.filter(s => s.name.toLowerCase().includes(q) || s.type.toLowerCase().includes(q))
    }
    if (filterType !== 'ALL') list = list.filter(s => s.type === filterType)
    if (filterStatus === 'active') list = list.filter(s => toggleMap[s.id])
    if (filterStatus === 'inactive') list = list.filter(s => !toggleMap[s.id])
    list.sort((a, b) => {
      if (sortBy === 'sharpe') return b.sharpe - a.sharpe
      if (sortBy === 'return') return b.annualReturn - a.annualReturn
      if (sortBy === 'winrate') return b.winRate - a.winRate
      if (sortBy === 'trades') return b.trades - a.trades
      return 0
    })
    return list
  }, [filterType, filterStatus, sortBy, search, toggleMap])

  return (
    <div className="space-y-5 animate-fadein">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-black text-[#e8e8e8] tracking-tight">Strategy Leaderboard</h1>
          <p className="text-xs text-[#555] mt-1">{activeCount}/{ALL_STRATEGIES.length} active</p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search strategies..."
            className="bg-[#111111] border border-[#1e1e1e] rounded-lg px-3 py-1.5 text-xs text-[#e8e8e8] w-40 focus:outline-none focus:border-[#f5a623]/40 transition-colors" />
          <div className="flex items-center gap-0.5 bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-1">
            {(['ALL', 'MANUAL', 'ML', 'ARB'] as FilterType[]).map(f => (
              <button key={f} onClick={() => setFilterType(f)} className="text-[10px] px-2.5 py-1 rounded font-bold transition-all duration-150"
                style={{ background: filterType === f ? '#f5a623' : 'transparent', color: filterType === f ? '#000' : '#555' }}>
                {f}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-0.5 bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-1">
            {(['all', 'active', 'inactive'] as const).map(f => (
              <button key={f} onClick={() => setFilterStatus(f)} className="text-[10px] px-2.5 py-1 rounded capitalize font-medium transition-all duration-150"
                style={{ background: filterStatus === f ? '#1e1e1e' : 'transparent', color: filterStatus === f ? '#e8e8e8' : '#555' }}>
                {f}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-0.5 bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-1">
            {([['sharpe', 'Sharpe'], ['return', 'Return'], ['winrate', 'Win%'], ['trades', 'Trades']] as [SortKey, string][]).map(([k, label]) => (
              <button key={k} onClick={() => setSortBy(k)} className="text-[10px] px-2.5 py-1 rounded font-medium transition-all duration-150"
                style={{ background: sortBy === k ? '#2979ff' : 'transparent', color: sortBy === k ? '#fff' : '#555' }}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="qe-card px-4 py-3">
          <p className="text-[10px] text-[#555] uppercase">Best Sharpe</p>
          <p className="text-sm font-bold text-[#00c853] font-mono">2.87 <span className="text-[10px] text-[#555] font-normal">poly_binary_arb</span></p>
        </div>
        <div className="qe-card px-4 py-3">
          <p className="text-[10px] text-[#555] uppercase">Best Return</p>
          <p className="text-sm font-bold text-[#f5a623] font-mono">+38.7% <span className="text-[10px] text-[#555] font-normal">ensemble</span></p>
        </div>
        <div className="qe-card px-4 py-3">
          <p className="text-[10px] text-[#555] uppercase">Best Win Rate</p>
          <p className="text-sm font-bold text-[#2979ff] font-mono">98% <span className="text-[10px] text-[#555] font-normal">poly_binary_arb</span></p>
        </div>
        <div className="qe-card px-4 py-3">
          <p className="text-[10px] text-[#555] uppercase">Lowest Max DD</p>
          <p className="text-sm font-bold text-[#ff1744] font-mono">-0.4% <span className="text-[10px] text-[#555] font-normal">poly_binary_arb</span></p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((s, i) => (
          <StrategyCard key={s.id} strategy={s} toggled={toggleMap[s.id] ?? s.active} onToggle={handleToggle} rank={i + 1} />
        ))}
        {filtered.length === 0 && (
          <div className="col-span-3 py-20 text-center">
            <p className="text-[#444] text-sm">No strategies match your filters.</p>
          </div>
        )}
      </div>
    </div>
  )
}
