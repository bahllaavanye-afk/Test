import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSelector, useDispatch } from 'react-redux'
import api from '../api/client'
import { RegimeIndicator } from '../components/risk/RegimeIndicator'
import { selectTradingMode, setMode } from '../store/slices/tradingModeSlice'
import LiveChartPlaceholder from '../components/charts/MockCandlestickChart'

function vixColor(vix: number | null | undefined): string {
  if (vix == null) return '#888888'
  if (vix > 30) return '#ff1744'
  if (vix > 20) return '#f5a623'
  return '#00c853'
}

function biasColor(bias: string | undefined): string {
  if (bias === 'risk_on') return '#00c853'
  if (bias === 'risk_off') return '#ff1744'
  return '#f5a623'
}

function MetricCard({ label, value, sub, color = '#f5a623' }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 transition-all duration-200 hover:border-[#2e2e2e] hover:bg-[#151515]">
      <p className="text-[#888888] text-xs uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold mt-1 transition-colors" style={{ color }}>{value}</p>
      {sub && <p className="text-[#888888] text-xs mt-1">{sub}</p>}
    </div>
  )
}

function ConfirmLiveModal({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) {
  const [input, setInput] = useState('')
  const valid = input.trim() === 'CONFIRM LIVE'
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-[#111111] border border-[#ff1744]/40 rounded-xl p-6 w-full max-w-md shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <span className="w-3 h-3 rounded-full bg-[#ff1744] animate-pulse inline-block" />
          <h2 className="text-[#ff1744] font-bold text-base">Switch to Live Trading</h2>
        </div>
        <p className="text-[#888888] text-sm mb-2">
          You are about to switch to <span className="text-[#ff1744] font-bold">LIVE trading mode</span>.
          Real money will be used. Strategies will execute against live markets.
        </p>
        <ul className="text-xs text-[#888888] mb-4 space-y-1 list-disc list-inside">
          <li>All active strategies will trade with real capital</li>
          <li>Orders will be sent to live broker connections</li>
          <li>Risk limits and position sizing apply immediately</li>
        </ul>
        <p className="text-xs text-[#888888] mb-2">Type <span className="text-white font-mono font-bold">CONFIRM LIVE</span> to proceed:</p>
        <input autoFocus value={input} onChange={e => setInput(e.target.value)}
          className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-3 py-2 text-sm font-mono text-white mb-4 focus:outline-none focus:border-[#ff1744]/60"
          placeholder="CONFIRM LIVE" />
        <div className="flex gap-3">
          <button onClick={onCancel} className="flex-1 px-4 py-2 rounded bg-[#1e1e1e] text-[#888888] text-sm hover:bg-[#2e2e2e] transition-colors">Cancel</button>
          <button onClick={() => valid && onConfirm()} disabled={!valid} className="flex-1 px-4 py-2 rounded text-sm font-bold transition-all duration-200"
            style={{ background: valid ? '#ff1744' : '#3a1a1e', color: valid ? '#fff' : '#666', cursor: valid ? 'pointer' : 'not-allowed' }}>
            Switch to Live
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const dispatch = useDispatch()
  const mode = useSelector(selectTradingMode)
  const [showLiveModal, setShowLiveModal] = useState(false)
  const [chartSymbol, setChartSymbol] = useState('NYSE:SPY')

  const { data: perf } = useQuery({ queryKey: ['performance'], queryFn: () => api.get('/analytics/performance').then(r => r.data), refetchInterval: 30_000 })
  const { data: positions } = useQuery({ queryKey: ['positions'], queryFn: () => api.get('/positions/').then(r => r.data), refetchInterval: 10_000 })
  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: () => api.get('/strategies/').then(r => r.data) })
  const { data: macro } = useQuery({ queryKey: ['macro'], queryFn: () => api.get('/analytics/macro').then(r => r.data), refetchInterval: 300_000 })
  const { data: sentiment } = useQuery({ queryKey: ['sentiment'], queryFn: () => api.get('/analytics/sentiment').then(r => r.data), refetchInterval: 600_000 })
  const { data: agentStatus } = useQuery({ queryKey: ['agents-status'], queryFn: () => api.get('/agents/status').then(r => r.data), refetchInterval: 15_000 })
  const { data: accounts } = useQuery({ queryKey: ['accounts'], queryFn: () => api.get('/accounts/').then(r => r.data), refetchInterval: 30_000 })

  const activeCount = Array.isArray(strategies) ? strategies.filter((s: any) => s.is_active || s.is_enabled).length : 0
  const totalPnl = perf?.total_pnl ?? 0
  const noAccountConnected = !accounts || (Array.isArray(accounts) && accounts.length === 0)
  const agentList: any[] = Array.isArray(agentStatus?.agents) ? agentStatus.agents : []

  const isLive = mode === 'live'
  const isPaper = mode === 'paper'
  const CHART_SYMBOLS = ['NYSE:SPY', 'NASDAQ:AAPL', 'NASDAQ:MSFT', 'NASDAQ:QQQ']

  return (
    <div className="space-y-5">
      {showLiveModal && <ConfirmLiveModal onConfirm={() => { dispatch(setMode('live')); setShowLiveModal(false) }} onCancel={() => setShowLiveModal(false)} />}

      <div className={`rounded-lg px-4 py-3 flex items-center justify-between transition-all duration-500 ${isLive ? 'bg-[#ff1744]/10 border border-[#ff1744]/40' : 'bg-[#f5a623]/10 border border-[#f5a623]/30'}`}>
        <div className="flex items-center gap-3">
          <span className="w-3 h-3 rounded-full inline-block"
            style={{ background: isLive ? '#ff1744' : '#f5a623', boxShadow: isLive ? '0 0 8px #ff1744' : '0 0 8px #f5a623', animation: isLive ? 'pulse 1s infinite' : 'none' }} />
          <div>
            <p className="text-sm font-bold" style={{ color: isLive ? '#ff1744' : '#f5a623' }}>
              {isLive ? 'LIVE TRADING — REAL MONEY AT RISK' : 'PAPER TRADING MODE'}
            </p>
            <p className="text-xs text-[#888888] mt-0.5">
              {isLive ? 'Strategies are executing against live markets. Monitor positions closely.' : 'All orders are simulated. No real capital at risk. Run paper for 2 weeks before going live.'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isPaper && (
            <button onClick={() => setShowLiveModal(true)} className="px-3 py-1.5 rounded text-xs font-bold text-white transition-all duration-200 hover:opacity-90 active:scale-95"
              style={{ background: 'linear-gradient(135deg, #ff1744, #c62828)' }}>Switch to Live Trading</button>
          )}
          {isLive && (
            <button onClick={() => dispatch(setMode('paper'))} className="px-3 py-1.5 rounded text-xs font-bold text-black transition-all duration-200 hover:opacity-90 active:scale-95"
              style={{ background: '#f5a623' }}>Switch to Paper</button>
          )}
        </div>
      </div>

      {noAccountConnected && (
        <div className="bg-[#111111] border border-[#f5a623]/30 rounded-lg p-4 flex items-center gap-3">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f5a623" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
          </svg>
          <div>
            <p className="text-sm text-[#f5a623] font-semibold">Connect your Alpaca account to see live P&amp;L</p>
            <p className="text-xs text-[#888888] mt-0.5">No broker account detected. <a href="/settings" className="text-[#f5a623] underline">Add API keys in Settings</a> to start paper trading.</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Total P&L" value={perf ? `$${totalPnl.toFixed(2)}` : '—'} sub={perf ? `${perf.total_trades ?? 0} trades` : 'Connect Alpaca to see P&L'} color={perf ? (totalPnl >= 0 ? '#00c853' : '#ff1744') : '#555555'} />
        <MetricCard label="Open Positions" value={Array.isArray(positions) ? String(positions.length) : '—'} sub="live positions" color="#2979ff" />
        <MetricCard label="Active Strategies" value={String(activeCount)} sub="running 24/7" color="#f5a623" />
        <MetricCard label="Target Sharpe" value=">2.0" sub="vs SPY 0.47" color="#9C27B0" />
      </div>

      <RegimeIndicator />

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 flex flex-col gap-2">
          <div className="flex gap-2">
            {CHART_SYMBOLS.map(s => (
              <button key={s} onClick={() => setChartSymbol(s)}
                className="text-xs px-2 py-1 rounded transition-colors"
                style={{ background: chartSymbol === s ? '#f5a623' : '#1e1e1e', color: chartSymbol === s ? '#000' : '#888' }}>
                {s.split(':')[1]}
              </button>
            ))}
          </div>
          <LiveChartPlaceholder symbol={chartSymbol} height={400} />
        </div>

        <div className="space-y-3">
          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
            <h3 className="text-xs text-[#888888] uppercase tracking-wider mb-3">Account Summary</h3>
            {noAccountConnected ? (
              <div className="text-center py-4 space-y-2">
                <p className="text-xs text-[#888888]">No account connected</p>
                <a href="/settings" className="text-xs text-[#f5a623] underline">Add API keys in Settings</a>
              </div>
            ) : (
              <div className="space-y-2">
                {(Array.isArray(accounts) ? accounts : [accounts]).filter(Boolean).map((acc: any, i: number) => (
                  <div key={acc?.id ?? i} className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-[#888888]">{acc?.broker ?? 'Account'}</span>
                      <span className="text-xs font-mono font-bold" style={{ color: (acc?.total_pnl ?? 0) >= 0 ? '#00c853' : '#ff1744' }}>
                        {acc?.total_pnl != null ? `${acc.total_pnl >= 0 ? '+' : ''}$${acc.total_pnl.toFixed(2)}` : '---'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-[#555]">Equity</span>
                      <span className="text-xs font-mono text-[#e8e8e8]">
                        {acc?.equity != null ? `$${Number(acc.equity).toLocaleString()}` : '---'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
            <h3 className="text-xs text-[#888888] uppercase tracking-wider mb-3">
              Agent Team ({agentList.length > 0 ? agentList.length : '---'})
            </h3>
            {agentList.length === 0 ? (
              <p className="text-xs text-[#555]">No agent status. Start the backend to see agent health.</p>
            ) : (
              <div className="space-y-1.5">
                {agentList.map((agent: any) => {
                  const isRunning = agent.running ?? false
                  return (
                    <div key={agent.name} className="flex items-center gap-2 py-1">
                      <span className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{ background: isRunning ? '#00c853' : '#ff1744', boxShadow: isRunning ? '0 0 5px #00c853' : 'none' }} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-[#e8e8e8] truncate">{agent.name}</span>
                          <span className="text-[10px] text-[#555] ml-1 flex-shrink-0">{agent.total_runs ?? 0} runs</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] text-[#555] truncate">{agent.role ?? ''}</span>
                          <span className="text-[10px] text-[#444] flex-shrink-0">
                            {agent.last_run ? new Date(agent.last_run).toLocaleTimeString() : '---'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 hover:border-[#2e2e2e] transition-colors">
          <h3 className="text-xs text-[#888888] uppercase tracking-wider mb-3">Macro Signals</h3>
          {!macro ? (
            <div className="space-y-2">{[1,2,3,4].map(i => <div key={i} className="h-5 bg-[#1e1e1e] rounded animate-pulse" />)}</div>
          ) : (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">VIX Level</span>
                <span className="text-sm font-bold font-mono" style={{ color: vixColor(macro.vix) }}>
                  {macro.vix != null ? macro.vix.toFixed(2) : '---'}
                  {macro.signals?.vix_regime && <span className="ml-1 text-xs font-normal">({macro.signals.vix_regime})</span>}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Yield Curve (10Y-2Y)</span>
                <span className="text-sm font-mono" style={{ color: macro.signals?.yield_curve_inverted ? '#ff1744' : '#00c853' }}>
                  {macro.yield_spread_10y2y != null ? `${(macro.yield_spread_10y2y * 100).toFixed(0)} bps` : '---'}
                  {macro.signals?.yield_curve_inverted != null && <span className="ml-1 text-xs">({macro.signals.yield_curve_inverted ? 'INVERTED' : 'normal'})</span>}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Macro Bias</span>
                <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ color: biasColor(macro.macro_bias), background: `${biasColor(macro.macro_bias)}20` }}>
                  {macro.macro_bias?.replace('_', ' ').toUpperCase() ?? '---'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-[#888888]">Macro Score</span>
                <span className="text-sm font-bold font-mono" style={{ color: biasColor(macro.macro_bias) }}>
                  {macro.macro_score != null ? (macro.macro_score > 0 ? `+${macro.macro_score}` : macro.macro_score) : '---'} / 3
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 hover:border-[#2e2e2e] transition-colors">
          <h3 className="text-xs text-[#888888] uppercase tracking-wider mb-3">Reddit Buzz (WSB)</h3>
          {!sentiment ? (
            <div className="space-y-2">{[1,2,3,4,5].map(i => <div key={i} className="h-5 bg-[#1e1e1e] rounded animate-pulse" />)}</div>
          ) : sentiment.error ? (
            <p className="text-xs text-[#888888]">Sentiment unavailable</p>
          ) : (Array.isArray(sentiment.results) && sentiment.results.length === 0) ? (
            <p className="text-xs text-[#555]">No sentiment data available</p>
          ) : (
            <div className="space-y-2">
              {(sentiment.results ?? []).slice(0, 5).map((item: any, i: number) => {
                const maxMentions = sentiment.results?.[0]?.mentions_24h ?? 1
                const pct = Math.round(((item.mentions_24h ?? 0) / Math.max(maxMentions, 1)) * 100)
                return (
                  <div key={item.ticker ?? i} className="space-y-0.5">
                    <div className="flex justify-between text-xs">
                      <span className="font-bold text-[#f5a623]">{item.ticker}</span>
                      <span className="text-[#888888]">{item.mentions_24h?.toLocaleString()} mentions</span>
                    </div>
                    <div className="h-1.5 bg-[#1e1e1e] rounded-full overflow-hidden">
                      <div className="h-full bg-[#f5a623] rounded-full transition-all duration-500" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      <style>{`@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
    </div>
  )
}
