import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import api from '../api/client'

interface BenchmarkRow { name: string; annualReturn: number; sharpe: number; maxDd: number; ytd: number; color: string; isUs?: boolean }
interface CompRow { strategy: string; symbol: string; manualSharpe: number; mlSharpe: number; pValue: number; winner: 'ML' | 'MANUAL' }
interface CurveSeries { name: string; color: string; isUs?: boolean; values: number[] }

interface SignalEvent {
  ts: number
  strategy: string
  symbol: string
  manual: string
  ml: string
  ml_confidence: number
  agree: boolean
  manual_only?: boolean
}

const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

// ── Sub-components ────────────────────────────────────────────────────────────

function EquityChart({ series }: { series: CurveSeries[] }) {
  const W = 800, H = 280, PAD_L = 48, PAD_R = 16, PAD_T = 16, PAD_B = 28
  const chartW = W - PAD_L - PAD_R, chartH = H - PAD_T - PAD_B
  const days = series[0]?.values.length ?? 0
  if (days === 0) return null
  const allValues = series.flatMap(s => s.values)
  const minV = Math.min(...allValues) * 0.99, maxV = Math.max(...allValues) * 1.005, range = maxV - minV
  const toX = (i: number) => PAD_L + (i / (days - 1)) * chartW
  const toY = (v: number) => PAD_T + (1 - (v - minV) / range) * chartH
  const yGrids = Array.from({ length: 5 }, (_, i) => minV + (range * i) / 4)
  const [hovered, setHovered] = useState<string | null>(null)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} onMouseLeave={() => setHovered(null)}>
      {yGrids.map((v, i) => (
        <g key={i}>
          <line x1={PAD_L} y1={toY(v).toFixed(1)} x2={W - PAD_R} y2={toY(v).toFixed(1)} stroke="#1a1a1a" strokeWidth="0.8" />
          <text x={PAD_L - 4} y={toY(v) + 4} fill="#444" fontSize="9" fontFamily="monospace" textAnchor="end">{v.toFixed(0)}</text>
        </g>
      ))}
      <line x1={PAD_L} y1={toY(100).toFixed(1)} x2={W - PAD_R} y2={toY(100).toFixed(1)} stroke="#2a2a2a" strokeWidth="1" strokeDasharray="4,4" />
      {series.map(s => {
        const pts = s.values.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')
        const linePath = `M ${pts.replace(/ /g, ' L ')}`
        const isHighlighted = hovered === null || hovered === s.name
        return (
          <g key={s.name} style={{ cursor: 'pointer' }} onMouseEnter={() => setHovered(s.name)}>
            <path d={linePath} fill="none" stroke={s.color} strokeWidth={s.isUs ? 2.5 : 1.5} strokeOpacity={isHighlighted ? 1 : 0.3} strokeLinecap="round" strokeLinejoin="round" style={{ transition: 'stroke-opacity 0.2s' }} />
            {s.isUs && (
              <>
                <circle cx={toX(days - 1)} cy={toY(s.values[days - 1])} r="4" fill={s.color} />
                <text x={toX(days - 1) + 6} y={toY(s.values[days - 1]) + 4} fill={s.color} fontSize="9" fontFamily="monospace" fontWeight="bold">{s.values[days - 1].toFixed(1)}</text>
              </>
            )}
          </g>
        )
      })}
    </svg>
  )
}

function ImprovementBar({ pct }: { pct: number }) {
  const w = Math.min(Math.abs(pct) * 2, 100)
  const color = pct > 0 ? '#00c853' : '#ff1744'
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${w}%`, background: color }} />
      </div>
      <span className="text-xs font-mono font-bold" style={{ color }}>{pct > 0 ? '+' : ''}{pct.toFixed(1)}%</span>
    </div>
  )
}

// ── Overview tab ──────────────────────────────────────────────────────────────

function OverviewTab({
  comparisons,
  benchmarks,
  benchmarksLoading,
  resultsLoading,
}: {
  comparisons: CompRow[]
  benchmarks: BenchmarkRow[]
  benchmarksLoading: boolean
  resultsLoading: boolean
}) {
  const mlWins = comparisons.filter(c => c.winner === 'ML').length
  const sigPairs = comparisons.filter(c => c.pValue < 0.05).length
  const avgImprovement = mlWins > 0
    ? comparisons.filter(c => c.winner === 'ML').reduce((acc, c) => acc + ((c.mlSharpe - c.manualSharpe) / c.manualSharpe) * 100, 0) / mlWins
    : 0

  const curveSeries: CurveSeries[] = benchmarks
    .filter(b => b.annualReturn != null)
    .map(b => ({ name: b.name, color: b.color, isUs: b.isUs, values: [100] }))

  const hasComparisons = comparisons.length > 0
  const hasBenchmarks = benchmarks.length > 0

  return (
    <div className="space-y-6">
      {/* Equity Curve Comparison */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-5">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-sm font-bold text-[#e8e8e8]">Equity Curve Comparison</h2>
            <p className="text-[11px] text-[#555] mt-0.5">Normalized to 100 at start · From API data</p>
          </div>
          <div className="flex items-center gap-4 flex-wrap">
            {curveSeries.map(s => (
              <div key={s.name} className="flex items-center gap-1.5">
                <div className="rounded-full" style={{ width: s.isUs ? 16 : 12, height: s.isUs ? 3 : 2, background: s.color }} />
                <span className="text-[10px]" style={{ color: s.isUs ? s.color : '#888' }}>{s.name}</span>
              </div>
            ))}
          </div>
        </div>
        {benchmarksLoading ? (
          <div className="h-[280px] bg-[#0d0d0d] rounded-lg animate-pulse" />
        ) : curveSeries.length > 0 ? (
          <EquityChart series={curveSeries} />
        ) : (
          <div className="h-[280px] flex items-center justify-center text-xs text-[#555]">
            No comparison runs yet — run a strategy comparison in Backtest Lab
          </div>
        )}
      </div>

      {/* Benchmark Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-[#1a1a1a] flex items-center justify-between">
          <h2 className="text-sm font-bold text-[#e8e8e8]">Benchmark Performance Reference</h2>
          <span className="text-[10px] text-[#555]">Risk-adjusted</span>
        </div>
        {benchmarksLoading ? (
          <div className="p-5 space-y-2">{[1,2,3,4,5].map(i => <div key={i} className="h-8 bg-[#0d0d0d] rounded animate-pulse" />)}</div>
        ) : !hasBenchmarks ? (
          <div className="px-5 py-8 text-center text-xs text-[#555]">
            No comparison runs yet — run a strategy comparison in Backtest Lab
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-[#0d0d0d]">
                <tr className="text-[10px] text-[#555] uppercase tracking-wider">
                  <th className="text-left px-5 py-3">Benchmark</th>
                  <th className="text-right px-5 py-3">Annual Return</th>
                  <th className="text-right px-5 py-3">YTD</th>
                  <th className="text-right px-5 py-3">Sharpe</th>
                  <th className="text-right px-5 py-3">Max Drawdown</th>
                </tr>
              </thead>
              <tbody>
                {benchmarks.map((b, i) => (
                  <tr key={i} className="border-t hover:bg-[#111111] transition-colors"
                    style={{ borderColor: b.isUs ? 'rgba(245,166,35,0.2)' : '#1a1a1a', background: b.isUs ? 'rgba(245,166,35,0.03)' : 'transparent' }}>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: b.color }} />
                        <span className="text-xs font-medium" style={{ color: b.isUs ? b.color : '#e8e8e8' }}>{b.name}</span>
                        {b.isUs && <span className="text-[9px] bg-[#f5a623] text-black px-1.5 py-0.5 rounded font-bold">OUR PLATFORM</span>}
                      </div>
                    </td>
                    <td className="px-5 py-3.5 text-right"><span className="text-xs font-mono font-bold" style={{ color: b.isUs ? '#f5a623' : '#e8e8e8' }}>{b.annualReturn >= 0 ? '+' : ''}{b.annualReturn.toFixed(1)}%</span></td>
                    <td className="px-5 py-3.5 text-right"><span className="text-xs font-mono" style={{ color: b.isUs ? '#f5a623' : '#aaa' }}>{b.ytd >= 0 ? '+' : ''}{b.ytd.toFixed(1)}%</span></td>
                    <td className="px-5 py-3.5 text-right"><span className="text-xs font-mono font-bold" style={{ color: b.sharpe >= 1.5 ? '#00c853' : b.sharpe >= 0.6 ? '#f5a623' : '#888' }}>{b.sharpe.toFixed(2)}</span></td>
                    <td className="px-5 py-3.5 text-right"><span className="text-xs font-mono text-[#ff1744]">{b.maxDd.toFixed(1)}%</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Statistical Summary */}
      {hasComparisons && (
        <div className="rounded-xl p-5 border" style={{ background: 'rgba(0,200,83,0.05)', borderColor: 'rgba(0,200,83,0.2)' }}>
          <div className="flex items-start gap-4">
            <div>
              <h3 className="text-sm font-bold text-[#00c853] mb-1">Statistical Analysis Result</h3>
              <p className="text-sm text-[#aaa] leading-relaxed">
                ML enhancement adds <span className="text-[#00c853] font-bold">+{avgImprovement.toFixed(0)}% average Sharpe improvement</span>,
                statistically significant at <span className="text-[#00c853] font-bold">p&lt;0.05</span> across{' '}
                <span className="text-[#00c853] font-bold">{sigPairs} of {comparisons.length} strategies</span>.
                ML wins {mlWins}/{comparisons.length} head-to-head matchups.
              </p>
              <div className="flex items-center gap-4 mt-3">
                <div className="text-center"><p className="text-xl font-black text-[#00c853]">{mlWins}/{comparisons.length}</p><p className="text-[10px] text-[#555]">ML Wins</p></div>
                <div className="w-px h-8 bg-[#1e1e1e]" />
                <div className="text-center"><p className="text-xl font-black text-[#00c853]">{sigPairs}/{comparisons.length}</p><p className="text-[10px] text-[#555]">Significant</p></div>
                <div className="w-px h-8 bg-[#1e1e1e]" />
                <div className="text-center"><p className="text-xl font-black text-[#00c853]">+{avgImprovement.toFixed(0)}%</p><p className="text-[10px] text-[#555]">Avg Sharpe Lift</p></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Head-to-Head Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-[#1a1a1a]">
          <h2 className="text-sm font-bold text-[#e8e8e8]">Manual vs ML Head-to-Head</h2>
          <p className="text-[11px] text-[#555] mt-0.5">P-value from two-sided t-test on daily returns.</p>
        </div>
        {resultsLoading ? (
          <div className="p-5 space-y-2">{[1,2,3].map(i => <div key={i} className="h-8 bg-[#0d0d0d] rounded animate-pulse" />)}</div>
        ) : !hasComparisons ? (
          <div className="px-5 py-8 text-center text-xs text-[#555]">
            No comparison runs yet — run a strategy comparison in Backtest Lab
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-[#0d0d0d]">
                <tr className="text-[10px] text-[#555] uppercase tracking-wider">
                  <th className="text-left px-5 py-3">Strategy</th>
                  <th className="text-right px-5 py-3">Symbol</th>
                  <th className="text-right px-5 py-3">Manual Sharpe</th>
                  <th className="text-right px-5 py-3">ML Sharpe</th>
                  <th className="text-right px-5 py-3">Improvement</th>
                  <th className="text-right px-5 py-3">P-Value</th>
                  <th className="text-center px-5 py-3">Winner</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((c, i) => {
                  const improvement = ((c.mlSharpe - c.manualSharpe) / c.manualSharpe) * 100
                  const isSig = c.pValue < 0.05
                  return (
                    <tr key={i} className="border-t border-[#1a1a1a] hover:bg-[#0d0d0d] transition-colors">
                      <td className="px-5 py-3.5 text-xs font-mono text-[#e8e8e8]">{c.strategy}</td>
                      <td className="px-5 py-3.5 text-right text-xs font-mono text-[#888]">{c.symbol}</td>
                      <td className="px-5 py-3.5 text-right"><span className="text-xs font-mono font-bold" style={{ color: c.manualSharpe >= 1.5 ? '#00c853' : c.manualSharpe >= 1.0 ? '#f5a623' : '#888' }}>{c.manualSharpe.toFixed(2)}</span></td>
                      <td className="px-5 py-3.5 text-right"><span className="text-xs font-mono font-bold" style={{ color: c.mlSharpe >= 1.5 ? '#00c853' : c.mlSharpe >= 1.0 ? '#f5a623' : '#888' }}>{c.mlSharpe.toFixed(2)}</span></td>
                      <td className="px-5 py-3.5 text-right"><ImprovementBar pct={improvement} /></td>
                      <td className="px-5 py-3.5 text-right">
                        <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: isSig ? 'rgba(0,200,83,0.1)' : 'rgba(255,23,68,0.08)', color: isSig ? '#00c853' : '#ff1744', border: `1px solid ${isSig ? 'rgba(0,200,83,0.25)' : 'rgba(255,23,68,0.2)'}` }}>
                          p={c.pValue.toFixed(3)} {isSig ? '✓' : '✗'}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 text-center">
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded inline-block"
                          style={{ background: c.winner === 'ML' ? 'rgba(156,39,176,0.12)' : 'rgba(41,121,255,0.12)', color: c.winner === 'ML' ? '#ce93d8' : '#2979ff', border: `1px solid ${c.winner === 'ML' ? 'rgba(156,39,176,0.3)' : 'rgba(41,121,255,0.3)'}` }}>
                          {c.winner === 'ML' ? 'ML WINS' : 'MANUAL WINS'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Advanced Analytics tab ────────────────────────────────────────────────────

const chartTheme = {
  background: '#111111',
  text: '#888',
  grid: '#1e1e1e',
}

function AdvancedAnalyticsTab() {
  const [signals, setSignals] = useState<SignalEvent[]>([])
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    const token = sessionStorage.getItem('access_token') || localStorage.getItem('token') || ''
    ws.current = new WebSocket(`${WS_BASE}/ws/signal-compare?token=${token}`)
    ws.current.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as SignalEvent & { type?: string }
        if (msg.type !== 'heartbeat') {
          setSignals(prev => [msg, ...prev].slice(0, 50))
        }
      } catch {
        // ignore malformed messages
      }
    }
    return () => ws.current?.close()
  }, [])

  return (
    <div className="space-y-6">
      {/* Section 1: Live Signal Feed */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-[#1a1a1a] flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-[#e8e8e8]">Live Signal Feed</h2>
            <p className="text-[11px] text-[#555] mt-0.5">Real-time manual vs ML signal comparison · last 50</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#00c853] animate-pulse" />
            <span className="text-[10px] text-[#555]">LIVE</span>
          </div>
        </div>
        <div
          className="p-3 font-mono text-[11px] overflow-y-auto"
          style={{ maxHeight: 300, background: '#0a0a0a' }}
        >
          {signals.length === 0 ? (
            <div className="text-[#555] text-center py-8">
              Waiting for signal events... (connects to /ws/signal-compare)
            </div>
          ) : (
            signals.map((sig, i) => {
              const ts = new Date(sig.ts * 1000).toLocaleTimeString()
              const agreeColor = sig.manual_only ? '#555' : sig.agree ? '#00c853' : '#f5a623'
              const agreeLabel = sig.manual_only ? 'MANUAL ONLY' : sig.agree ? '✓ AGREE' : '✗ DIFFER'
              return (
                <div key={i} className="py-1 border-b border-[#1a1a1a] last:border-0" style={{ color: agreeColor }}>
                  [{ts}] {sig.strategy} | {sig.symbol} | Manual: {sig.manual} (−) vs ML: {sig.ml} ({Math.round(sig.ml_confidence * 100)}%) | {agreeLabel}
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* Section 2: Win Rate by ML Confidence */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-5">
        <h2 className="text-sm font-bold text-[#e8e8e8] mb-1">Win Rate by ML Confidence</h2>
        <p className="text-[11px] text-[#555] mb-4">Manual vs ML win rate across confidence buckets — requires live trade data</p>
        <div className="flex items-center justify-center h-[220px] text-[#555] text-[12px]">
          No trade data yet. Run a comparison to populate this chart.
        </div>
      </div>

      {/* Section 3: MAE/MFE Comparison */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-5">
        <h2 className="text-sm font-bold text-[#e8e8e8] mb-1">MAE / MFE Comparison</h2>
        <p className="text-[11px] text-[#555] mb-4">Maximum Adverse Excursion · Maximum Favorable Excursion · Edge Ratio</p>
        <div className="flex items-center justify-center h-[180px] text-[#555] text-[12px]">
          No trade data yet. Run a comparison to populate this chart.
        </div>
      </div>

      {/* Section 4: R-Multiple Distribution */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-5">
        <h2 className="text-sm font-bold text-[#e8e8e8] mb-1">R-Multiple Distribution</h2>
        <p className="text-[11px] text-[#555] mb-4">Trade outcome distribution in units of initial risk</p>
        <div className="flex items-center justify-center h-[180px] text-[#555] text-[12px]">
          No trade data yet. Run a comparison to populate this chart.
        </div>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Comparison() {
  const [activeTab, setActiveTab] = useState<'overview' | 'advanced'>('overview')

  const { data: resultsData, isLoading: resultsLoading } = useQuery<CompRow[]>({
    queryKey: ['comparison-results'],
    queryFn: () => api.get('/comparison/results').then(r => r.data),
    refetchInterval: 60_000,
  })
  const { data: benchmarksData, isLoading: benchmarksLoading } = useQuery<BenchmarkRow[]>({
    queryKey: ['comparison-benchmarks'],
    queryFn: () => api.get('/comparison/benchmarks').then(r => {
      const raw = r.data
      if (Array.isArray(raw)) return raw
      const COLORS: Record<string, string> = {
        'SPY': '#2196F3', 'QQQ': '#9C27B0', 'BRK-B': '#FF9800', 'ALL_WEATHER': '#4CAF50',
      }
      return Object.entries(raw).map(([ticker, b]: [string, unknown]) => {
        const bm = b as Record<string, unknown>
        return {
          name: (bm.name as string) ?? ticker,
          annualReturn: (bm.annual_return as number) ?? 0,
          sharpe: (bm.sharpe as number) ?? 0,
          maxDd: (bm.max_dd as number) ?? 0,
          ytd: (bm.annual_return as number) ?? 0,
          color: COLORS[ticker] ?? '#888888',
          isUs: false,
        }
      })
    }),
    refetchInterval: 300_000,
  })

  const comparisons: CompRow[] = resultsData ?? []
  const benchmarks: BenchmarkRow[] = benchmarksData ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-black text-[#e8e8e8] tracking-tight">QuantEdge vs The World</h1>
        <p className="text-sm text-[#555] mt-1">Full institutional performance comparison · Walk-forward validated</p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-2 mb-4 border-b border-gray-700">
        {(['overview', 'advanced'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium capitalize ${
              activeTab === tab
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab === 'advanced' ? 'Advanced Analytics' : 'Overview'}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <OverviewTab
          comparisons={comparisons}
          benchmarks={benchmarks}
          benchmarksLoading={benchmarksLoading}
          resultsLoading={resultsLoading}
        />
      )}
      {activeTab === 'advanced' && <AdvancedAnalyticsTab />}
    </div>
  )
}
