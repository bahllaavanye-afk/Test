import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

export default function OptionsFlow() {
  const [unusualOnly, setUnusualOnly] = useState(false)
  const [wheelTickers, setWheelTickers] = useState('AAPL,MSFT,NVDA,AMD,SPY')

  const { data: flow = [], isLoading } = useQuery({
    queryKey: ['options-flow', unusualOnly],
    queryFn: () => api.get(`/options/flow?unusual_only=${unusualOnly}`).then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: pcr } = useQuery({
    queryKey: ['put-call-ratio'],
    queryFn: () => api.get('/options/put-call-ratio').then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: wheel = [] } = useQuery({
    queryKey: ['wheel', wheelTickers],
    queryFn: () => api.get(`/options/wheel?tickers=${wheelTickers}`).then(r => r.data),
    refetchInterval: 60_000,
  })

  const { data: calendar = [] } = useQuery({
    queryKey: ['macro-calendar'],
    queryFn: () => api.get('/options/macro-calendar?days_ahead=60').then(r => r.data),
    refetchInterval: 3_600_000,
  })

  const { data: nextFomc } = useQuery({
    queryKey: ['next-fomc'],
    queryFn: () => api.get('/options/next-fomc').then(r => r.data),
    refetchInterval: 3_600_000,
  })

  const sentimentColor = (s: string) =>
    s === 'bullish' ? 'text-[#00c853]' : s === 'bearish' ? 'text-[#ff1744]' : 'text-[#888888]'

  const categoryColor = (cat: string) => {
    const map: Record<string, string> = {
      fomc: '#f5a623', cpi: '#2196f3', ppi: '#9c27b0', nfp: '#00c853', gdp: '#ff9800',
    }
    return map[cat] || '#888888'
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-white">Options &amp; Macro</h1>
        {nextFomc && (
          <div className="flex items-center gap-2 bg-[#111111] border border-[#f5a623]/40 rounded px-3 py-1.5">
            <span className="text-[#f5a623] text-xs font-semibold">Next FOMC</span>
            <span className="text-white text-xs">{nextFomc.date}</span>
            <span className="text-[#888888] text-xs">({nextFomc.days_away}d away)</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        {/* Put/Call Ratio */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
          <div className="text-xs text-[#888888] uppercase mb-2">Put/Call Ratio</div>
          {pcr ? (
            <>
              <div className={`text-2xl font-bold ${sentimentColor(pcr.sentiment)}`}>{pcr.ratio}</div>
              <div className="text-xs text-[#888888] mt-1">
                Calls: {pcr.calls?.toLocaleString()} · Puts: {pcr.puts?.toLocaleString()}
              </div>
              <div className={`text-xs mt-1 font-semibold ${sentimentColor(pcr.sentiment)}`}>
                {pcr.sentiment?.toUpperCase()}
              </div>
            </>
          ) : <div className="text-[#888888] text-xs">Loading...</div>}
        </div>

        {/* Unusual Activity Count */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
          <div className="text-xs text-[#888888] uppercase mb-2">Unusual Activity</div>
          <div className="text-2xl font-bold text-[#f5a623]">
            {(flow as any[]).filter((f: any) => f.is_unusual).length}
          </div>
          <div className="text-xs text-[#888888] mt-1">of {(flow as any[]).length} total flows</div>
          <div className="text-xs text-[#f5a623] mt-1 font-semibold">SCANNING LIVE</div>
        </div>

        {/* Wheel Opportunities */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
          <div className="text-xs text-[#888888] uppercase mb-2">Wheel Opportunities</div>
          <div className="text-2xl font-bold text-[#00c853]">{(wheel as any[]).length}</div>
          <div className="text-xs text-[#888888] mt-1">
            Best yield: {(wheel as any[])[0]?.annualized_yield?.toFixed(1)}% ann.
          </div>
          <div className="text-xs text-[#00c853] mt-1 font-semibold">HIGH IV RANK</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Options Flow Table */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs text-[#888888] uppercase">Options Flow</h3>
            <button
              onClick={() => setUnusualOnly(!unusualOnly)}
              className={`text-xs px-2 py-0.5 rounded ${unusualOnly ? 'bg-[#f5a623] text-black' : 'bg-[#1e1e1e] text-[#888888]'}`}
            >
              {unusualOnly ? 'Unusual Only' : 'All Flows'}
            </button>
          </div>
          <div className="grid grid-cols-6 text-xs text-[#888888] mb-1 px-1">
            {['Ticker', 'Type', 'Strike', 'Expiry', 'Premium', 'IV%'].map(h => <span key={h}>{h}</span>)}
          </div>
          <div className="space-y-0.5 max-h-64 overflow-y-auto">
            {isLoading && <p className="text-xs text-[#888888] px-1">Loading...</p>}
            {(flow as any[]).slice(0, 20).map((f: any, i: number) => (
              <div key={i} className={`grid grid-cols-6 text-xs px-1 py-1 rounded ${f.is_unusual ? 'bg-[#f5a623]/10 border border-[#f5a623]/20' : 'bg-[#0a0a0a]'}`}>
                <span className="font-mono font-semibold text-white">{f.ticker}</span>
                <span className={f.option_type === 'call' ? 'text-[#00c853]' : 'text-[#ff1744]'}>
                  {f.option_type?.toUpperCase()}
                </span>
                <span className="font-mono">${f.strike}</span>
                <span className="text-[#888888]">{f.expiry?.slice(5)}</span>
                <span className="font-mono">${(f.premium / 1000).toFixed(0)}K</span>
                <span className={f.iv_percentile > 70 ? 'text-[#f5a623]' : 'text-[#888888]'}>
                  {f.iv_percentile?.toFixed(0)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Wheel Strategy */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs text-[#888888] uppercase">Wheel Strategy (CSP)</h3>
          </div>
          <div className="flex gap-1 mb-2">
            <input
              value={wheelTickers}
              onChange={e => setWheelTickers(e.target.value)}
              className="flex-1 bg-[#0a0a0a] border border-[#1e1e1e] rounded px-2 py-1 text-xs"
              placeholder="AAPL,MSFT,NVDA"
            />
          </div>
          <div className="grid grid-cols-5 text-xs text-[#888888] mb-1 px-1">
            {['Ticker', 'Strike', 'DTE', 'Yield%', 'IV Rank'].map(h => <span key={h}>{h}</span>)}
          </div>
          <div className="space-y-0.5 max-h-56 overflow-y-auto">
            {(wheel as any[]).map((w: any, i: number) => (
              <div key={i} className="grid grid-cols-5 text-xs px-1 py-1 bg-[#0a0a0a] rounded">
                <span className="font-mono font-semibold text-white">{w.ticker}</span>
                <span className="font-mono">${w.strike}</span>
                <span className="text-[#888888]">{w.expiry ? Math.round((new Date(w.expiry).getTime() - Date.now()) / 86400000) : '—'}d</span>
                <span className="text-[#00c853] font-semibold">{w.annualized_yield?.toFixed(1)}%</span>
                <span className={w.iv_rank > 70 ? 'text-[#f5a623]' : 'text-[#888888]'}>{w.iv_rank?.toFixed(0)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Macro Calendar */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-3">
        <h3 className="text-xs text-[#888888] uppercase mb-2">Macro Event Calendar (60 days)</h3>
        <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
          {(calendar as any[]).map((ev: any, i: number) => (
            <div key={i} className="flex items-start gap-2 bg-[#0a0a0a] rounded p-2">
              <div className="text-center min-w-[48px]">
                <div className="text-[10px] text-[#888888]">{ev.date?.slice(5)}</div>
                <div className={`text-[10px] font-semibold ${ev.days_away <= 7 ? 'text-[#ff1744]' : ev.days_away <= 21 ? 'text-[#f5a623]' : 'text-[#888888]'}`}>
                  {ev.days_away}d
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1">
                  <span className="text-[10px] font-bold px-1 rounded" style={{ color: categoryColor(ev.category), backgroundColor: categoryColor(ev.category) + '20' }}>
                    {ev.category?.toUpperCase()}
                  </span>
                  {ev.importance === 'high' && <span className="text-[10px] text-[#ff1744]">★</span>}
                </div>
                <div className="text-xs text-white mt-0.5 truncate">{ev.title}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
