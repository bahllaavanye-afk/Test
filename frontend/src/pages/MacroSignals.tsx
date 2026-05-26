import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../api/client'

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

function spreadColor(spread: number | null | undefined): string {
  if (spread == null) return '#888888'
  if (spread < 0) return '#ff1744'
  if (spread < 0.5) return '#f5a623'
  return '#00c853'
}

type SortKey = 'ticker' | 'mentions_24h' | 'upvotes' | 'rank_24h_ago'
type SortDir = 'asc' | 'desc'

export default function MacroSignals() {
  const [sortKey, setSortKey] = useState<SortKey>('mentions_24h')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const { data: macro, isLoading: macroLoading } = useQuery({
    queryKey: ['macro'],
    queryFn: () => api.get('/analytics/macro').then(r => r.data),
    refetchInterval: 300_000,
  })

  const { data: sentiment, isLoading: sentimentLoading } = useQuery({
    queryKey: ['sentiment'],
    queryFn: () => api.get('/analytics/sentiment').then(r => r.data),
    refetchInterval: 600_000,
  })

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sortedSentiment = [...(sentiment?.results ?? [])].sort((a: any, b: any) => {
    const av = a[sortKey] ?? 0
    const bv = b[sortKey] ?? 0
    if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
    return sortDir === 'asc' ? av - bv : bv - av
  })

  const spread = macro?.yield_spread_10y2y
  const spreadBps = spread != null ? Math.round(spread * 100) : null

  function macroInterpretation(): string {
    if (!macro) return 'Loading macro data...'
    const parts: string[] = []

    if (macro.vix != null) {
      if (macro.vix > 30) parts.push(`VIX at ${macro.vix.toFixed(1)} signals extreme fear — options are expensive, expect elevated realized volatility and potential for sharp reversals.`)
      else if (macro.vix > 20) parts.push(`VIX at ${macro.vix.toFixed(1)} is elevated — markets are cautious, risk premiums are higher than average.`)
      else parts.push(`VIX at ${macro.vix.toFixed(1)} is complacent — markets are calm, risk premiums are compressed, favorable for carry and trend strategies.`)
    }

    if (spread != null) {
      if (spread < 0) parts.push(`The yield curve is inverted (10Y-2Y = ${spreadBps} bps), historically a recession warning 12-18 months ahead. This typically pressures bank earnings and signals tight financial conditions.`)
      else if (spread < 0.5) parts.push(`The yield curve is flat (10Y-2Y = ${spreadBps} bps), suggesting growth uncertainty and limited term premium.`)
      else parts.push(`The yield curve is positively sloped (10Y-2Y = ${spreadBps} bps), supporting bank net interest margins and reflecting growth expectations.`)
    }

    if (macro.hy_credit_spread != null) {
      if (macro.hy_credit_spread > 6) parts.push(`High-yield spreads at ${macro.hy_credit_spread.toFixed(1)}% signal significant credit stress — risk assets likely face headwinds.`)
      else if (macro.hy_credit_spread > 5) parts.push(`High-yield spreads at ${macro.hy_credit_spread.toFixed(1)}% are elevated, indicating some credit market stress.`)
      else parts.push(`High-yield spreads at ${macro.hy_credit_spread.toFixed(1)}% are contained — credit markets are not flashing systemic risk.`)
    }

    if (macro.macro_bias) {
      const bias = macro.macro_bias.replace('_', ' ').toUpperCase()
      parts.push(`Overall macro bias: ${bias} (score ${macro.macro_score > 0 ? '+' : ''}${macro.macro_score}/3). ${macro.macro_bias === 'risk_on' ? 'Conditions favor momentum, carry, and directional strategies.' : macro.macro_bias === 'risk_off' ? 'Conditions favor defensive, arb, and reduced position sizing.' : 'Mixed signals — favor diversified exposure with tight risk controls.'}`)
    }

    return parts.join(' ') || 'Insufficient data to generate interpretation.'
  }

  const SortIcon = ({ col }: { col: SortKey }) => (
    <span className="ml-1 text-[#555]">
      {sortKey === col ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}
    </span>
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">Macro &amp; Sentiment</h1>
        <span className="text-xs text-[#888888]">
          FRED: 5min cache · Sentiment: 10min cache
        </span>
      </div>

      {/* Row 1: KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* VIX Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <p className="text-[#888888] text-xs uppercase tracking-wider">VIX Level</p>
          {macroLoading ? (
            <p className="text-[#888888] text-sm mt-2">Loading...</p>
          ) : (
            <>
              <p className="text-3xl font-bold mt-1 font-mono" style={{ color: vixColor(macro?.vix) }}>
                {macro?.vix != null ? macro.vix.toFixed(2) : '—'}
              </p>
              <p className="text-xs mt-1" style={{ color: vixColor(macro?.vix) }}>
                {macro?.signals?.vix_regime?.toUpperCase() ?? '—'}
              </p>
            </>
          )}
        </div>

        {/* 10Y-2Y Spread Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <p className="text-[#888888] text-xs uppercase tracking-wider">10Y-2Y Spread</p>
          {macroLoading ? (
            <p className="text-[#888888] text-sm mt-2">Loading...</p>
          ) : (
            <>
              <p className="text-3xl font-bold mt-1 font-mono" style={{ color: spreadColor(spread) }}>
                {spreadBps != null ? `${spreadBps > 0 ? '+' : ''}${spreadBps} bps` : '—'}
              </p>
              <p className="text-xs mt-1" style={{ color: spreadColor(spread) }}>
                {macro?.signals?.yield_curve_inverted != null
                  ? macro.signals.yield_curve_inverted ? 'INVERTED' : 'NORMAL'
                  : '—'}
              </p>
            </>
          )}
        </div>

        {/* Macro Bias Card */}
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
          <p className="text-[#888888] text-xs uppercase tracking-wider">Macro Bias</p>
          {macroLoading ? (
            <p className="text-[#888888] text-sm mt-2">Loading...</p>
          ) : (
            <>
              <p className="text-3xl font-bold mt-1" style={{ color: biasColor(macro?.macro_bias) }}>
                {macro?.macro_bias?.replace('_', ' ').toUpperCase() ?? '—'}
              </p>
              <p className="text-xs mt-1 text-[#888888]">
                Score: {macro?.macro_score != null ? (macro.macro_score > 0 ? `+${macro.macro_score}` : macro.macro_score) : '—'} / 3
              </p>
            </>
          )}
        </div>
      </div>

      {/* Row 2: Full FRED Data Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-semibold">FRED Economic Indicators</h2>
        </div>
        {macroLoading ? (
          <p className="text-[#888888] text-sm p-4">Loading FRED data...</p>
        ) : !macro ? (
          <p className="text-[#888888] text-sm p-4">FRED data unavailable</p>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[#1e1e1e]">
                <th className="text-left px-4 py-2 text-[#888888] font-normal">Indicator</th>
                <th className="text-right px-4 py-2 text-[#888888] font-normal">Value</th>
                <th className="text-right px-4 py-2 text-[#888888] font-normal">Signal</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  label: 'VIX (CBOE Volatility Index)',
                  value: macro.vix != null ? macro.vix.toFixed(2) : '—',
                  signal: macro.signals?.vix_regime?.toUpperCase(),
                  color: vixColor(macro.vix),
                },
                {
                  label: '10Y-2Y Yield Spread',
                  value: spreadBps != null ? `${spreadBps > 0 ? '+' : ''}${spreadBps} bps` : '—',
                  signal: macro.signals?.yield_curve_signal?.replace('_', ' ').toUpperCase(),
                  color: spreadColor(macro.yield_spread_10y2y),
                },
                {
                  label: 'Fed Funds Rate (DFF)',
                  value: macro.fed_funds_rate != null ? `${macro.fed_funds_rate.toFixed(2)}%` : '—',
                  signal: macro.fed_funds_rate != null ? (macro.fed_funds_rate > 5 ? 'RESTRICTIVE' : macro.fed_funds_rate > 2 ? 'NEUTRAL' : 'ACCOMMODATIVE') : '—',
                  color: macro.fed_funds_rate != null ? (macro.fed_funds_rate > 5 ? '#ff1744' : macro.fed_funds_rate > 2 ? '#f5a623' : '#00c853') : '#888888',
                },
                {
                  label: 'HY Credit Spread (BAMLH0A0HYM2)',
                  value: macro.hy_credit_spread != null ? `${macro.hy_credit_spread.toFixed(2)}%` : '—',
                  signal: macro.signals?.credit_stress != null ? (macro.signals.credit_stress ? 'STRESS' : 'CONTAINED') : '—',
                  color: macro.signals?.credit_stress ? '#ff1744' : '#00c853',
                },
                {
                  label: 'USD Broad Dollar Index',
                  value: macro.usd_index != null ? macro.usd_index.toFixed(2) : '—',
                  signal: '—',
                  color: '#888888',
                },
              ].map((row, i) => (
                <tr key={i} className="border-b border-[#1e1e1e] last:border-0 hover:bg-[#1a1a1a]">
                  <td className="px-4 py-2.5 text-[#e8e8e8]">{row.label}</td>
                  <td className="px-4 py-2.5 text-right font-mono font-bold" style={{ color: row.color }}>
                    {row.value}
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    {row.signal && row.signal !== '—' ? (
                      <span
                        className="px-2 py-0.5 rounded text-xs font-semibold"
                        style={{ color: row.color, background: `${row.color}20` }}
                      >
                        {row.signal}
                      </span>
                    ) : (
                      <span className="text-[#555]">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {macro?.fetched_at && (
          <p className="text-[#555] text-xs px-4 py-2 border-t border-[#1e1e1e]">
            Last updated: {new Date(macro.fetched_at).toLocaleTimeString()}
          </p>
        )}
      </div>

      {/* Row 3: Reddit Sentiment Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e] flex items-center justify-between">
          <h2 className="text-sm font-semibold">Reddit WallStreetBets Sentiment</h2>
          {sentiment?.source && (
            <span className="text-xs text-[#555]">via {sentiment.source}</span>
          )}
        </div>
        {sentimentLoading ? (
          <p className="text-[#888888] text-sm p-4">Loading sentiment data...</p>
        ) : !sentiment || sentiment.error ? (
          <p className="text-[#888888] text-sm p-4">Sentiment data unavailable</p>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[#1e1e1e]">
                {(
                  [
                    { key: 'ticker', label: 'Ticker' },
                    { key: 'mentions_24h', label: '24h Mentions' },
                    { key: 'upvotes', label: 'Upvotes' },
                    { key: 'rank_24h_ago', label: 'Rank 24h Ago' },
                  ] as { key: SortKey; label: string }[]
                ).map(col => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    className={`px-4 py-2 text-[#888888] font-normal cursor-pointer hover:text-[#e8e8e8] select-none ${col.key === 'ticker' ? 'text-left' : 'text-right'}`}
                  >
                    {col.label}
                    <SortIcon col={col.key} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedSentiment.map((item: any, i: number) => (
                <tr key={item.ticker ?? i} className="border-b border-[#1e1e1e] last:border-0 hover:bg-[#1a1a1a]">
                  <td className="px-4 py-2.5 font-bold text-[#f5a623]">{item.ticker ?? '—'}</td>
                  <td className="px-4 py-2.5 text-right font-mono text-[#e8e8e8]">
                    {item.mentions_24h?.toLocaleString() ?? '—'}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-[#888888]">
                    {item.upvotes?.toLocaleString() ?? '—'}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-[#888888]">
                    {item.rank_24h_ago != null ? `#${item.rank_24h_ago}` : '—'}
                  </td>
                </tr>
              ))}
              {sortedSentiment.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-4 py-4 text-center text-[#555]">
                    No sentiment data available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
        {sentiment?.fetched_at && (
          <p className="text-[#555] text-xs px-4 py-2 border-t border-[#1e1e1e]">
            Last updated: {new Date(sentiment.fetched_at).toLocaleTimeString()}
          </p>
        )}
      </div>

      {/* Row 4: Macro Interpretation */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3">Macro Interpretation</h2>
        <p className="text-sm text-[#c0c0c0] leading-relaxed">
          {macroLoading ? (
            <span className="text-[#888888]">Generating interpretation...</span>
          ) : (
            macroInterpretation()
          )}
        </p>
        <div className="mt-3 pt-3 border-t border-[#1e1e1e] grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="text-xs text-[#888888]">
            <span className="font-semibold text-[#e8e8e8]">Arbitrage strategies</span> — {macro?.macro_bias === 'risk_off' ? 'Prioritize (70%+ allocation)' : 'Maintain standard allocation'}
          </div>
          <div className="text-xs text-[#888888]">
            <span className="font-semibold text-[#e8e8e8]">ML/directional strategies</span> — {macro?.macro_bias === 'risk_off' ? 'Reduce exposure, tighten stops' : macro?.macro_bias === 'risk_on' ? 'Increase exposure within limits' : 'Standard 30% allocation'}
          </div>
          <div className="text-xs text-[#888888]">
            <span className="font-semibold text-[#e8e8e8]">Position sizing</span> — {macro?.vix != null && macro.vix > 30 ? 'Reduce via Kelly fraction (high vol)' : macro?.vix != null && macro.vix > 20 ? 'Slightly reduced (elevated vol)' : 'Standard Kelly sizing'}
          </div>
        </div>
      </div>
    </div>
  )
}
