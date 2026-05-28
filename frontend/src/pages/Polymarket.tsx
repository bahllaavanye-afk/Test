import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../api/client'

type FilterType = 'all' | 'politics' | 'crypto' | 'sports' | 'economics'
type SortType = 'volume' | 'liquidity' | 'recent'

interface Market {
  id: string
  title: string
  yes_price: number
  no_price: number
  volume_24h: number
  liquidity: number
  category: string
  end_date: string
}

const DEMO_MARKETS: Market[] = [
  { id: 'fed-rate-2025', title: 'Will Fed cut rates before Sep 2025?', yes_price: 0.68, no_price: 0.30, volume_24h: 245000, liquidity: 890000, category: 'economics', end_date: '2025-09-01' },
  { id: 'btc-100k-2025', title: 'BTC above $100k by end of 2025?', yes_price: 0.45, no_price: 0.53, volume_24h: 1200000, liquidity: 3400000, category: 'crypto', end_date: '2025-12-31' },
  { id: 'sp500-5500', title: 'S&P 500 above 5500 EOY 2025?', yes_price: 0.71, no_price: 0.27, volume_24h: 580000, liquidity: 1200000, category: 'economics', end_date: '2025-12-31' },
  { id: 'eth-merge-staking', title: 'ETH staking yield above 5% in Q3 2025?', yes_price: 0.38, no_price: 0.60, volume_24h: 89000, liquidity: 340000, category: 'crypto', end_date: '2025-09-30' },
  { id: 'trump-2026', title: 'Will US avoid recession in 2026?', yes_price: 0.44, no_price: 0.52, volume_24h: 320000, liquidity: 760000, category: 'economics', end_date: '2026-12-31' },
]

const CATEGORY_COLORS: Record<string, string> = {
  economics: '#f5a623',
  crypto: '#00c853',
  politics: '#2979ff',
  sports: '#e040fb',
  other: '#888888',
}

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`
  return `$${n.toFixed(0)}`
}

export default function Polymarket() {
  const [filter, setFilter] = useState<FilterType>('all')
  const [sortBy, setSortBy] = useState<SortType>('volume')

  const { data: apiMarkets = [], isLoading } = useQuery<Market[]>({
    queryKey: ['polymarkets', filter, sortBy],
    queryFn: () => api.get(`/market_data/polymarket?filter=${filter}&sort=${sortBy}`).then(r => r.data),
    refetchInterval: 30_000,
    retry: 1,
  })

  const tradeMutation = useMutation({
    mutationFn: (market: Market) =>
      api.post('/orders/', {
        symbol: market.id,
        side: 'buy',
        qty: 1,
        market_type: 'polymarket',
      }).then(r => r.data),
  })

  const markets: Market[] = apiMarkets.length === 0 && !isLoading ? DEMO_MARKETS : apiMarkets

  const filteredMarkets = markets.filter(m =>
    filter === 'all' ? true : m.category === filter
  )

  const sortedMarkets = [...filteredMarkets].sort((a, b) => {
    if (sortBy === 'volume') return b.volume_24h - a.volume_24h
    if (sortBy === 'liquidity') return b.liquidity - a.liquidity
    return new Date(b.end_date).getTime() - new Date(a.end_date).getTime()
  })

  // Arb opportunities: YES + NO price < $0.97
  const arbOpportunities = (markets as any[]).filter((m: any) => {
    const yes = m.yes_price || 0
    const no = m.no_price || 0
    return yes + no < 0.97 && yes + no > 0.01
  })

  const { data: positions = [] } = useQuery<any[]>({
    queryKey: ['positions', 'polymarket'],
    queryFn: () => api.get('/positions/?market_type=polymarket').then(r => r.data),
    refetchInterval: 15_000,
    retry: 1,
  })

  const FILTERS: { value: FilterType; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'politics', label: 'Politics' },
    { value: 'crypto', label: 'Crypto' },
    { value: 'sports', label: 'Sports' },
    { value: 'economics', label: 'Economics' },
  ]

  const usingDemo = apiMarkets.length === 0 && !isLoading

  return (
    <div className="space-y-6">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex-1 min-w-0">
          <h1 className="text-xl font-bold text-white">Prediction Markets</h1>
          <p className="text-xs text-[#888888] mt-0.5">Polymarket CLOB · Polygon blockchain · live order book</p>
        </div>

        {/* Filter tabs */}
        <div className="flex items-center gap-1 bg-[#111111] border border-[#1e1e1e] rounded-lg p-1">
          {FILTERS.map(f => (
            <button
              key={f.value}
              onClick={() => setFilter(f.value)}
              className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                filter === f.value
                  ? 'bg-[#f5a623] text-black font-semibold'
                  : 'text-[#888888] hover:text-white'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Sort dropdown */}
        <select
          value={sortBy}
          onChange={e => setSortBy(e.target.value as SortType)}
          className="bg-[#111111] border border-[#1e1e1e] text-xs text-[#e8e8e8] rounded-lg px-3 py-2 focus:outline-none focus:border-[#f5a623]"
        >
          <option value="volume">Sort: Volume</option>
          <option value="liquidity">Sort: Liquidity</option>
          <option value="recent">Sort: End Date</option>
        </select>
      </div>

      {/* Demo banner */}
      {usingDemo && (
        <div className="border border-[#1e1e1e] bg-[#111111] rounded-lg px-4 py-3 text-xs text-[#888888]">
          Connect <span className="text-[#f5a623] font-mono">POLYMARKET_PRIVATE_KEY</span> in{' '}
          <span className="font-mono">.env</span> to see live markets. Paper trading shows simulated markets.
        </div>
      )}

      {/* Arb Alert banner */}
      {arbOpportunities.length > 0 && (
        <div className="border border-[#f5a623]/50 bg-[#f5a623]/10 rounded-lg px-4 py-3 flex items-center gap-2">
          <span className="text-[#f5a623] font-bold text-sm">⚡</span>
          <span className="text-[#f5a623] text-sm font-medium">
            {arbOpportunities.length} arbitrage {arbOpportunities.length === 1 ? 'opportunity' : 'opportunities'} —{' '}
            YES+NO &lt; $0.97, risk-free profit available
          </span>
          <span className="ml-auto text-[10px] text-[#f5a623]/50 font-mono hidden md:block">
            {arbOpportunities.map((m: any) => m.id).join(' · ')}
          </span>
        </div>
      )}

      {/* Loading skeleton */}
      {isLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 animate-pulse">
              <div className="h-4 bg-[#1e1e1e] rounded w-3/4 mb-3" />
              <div className="flex gap-2 mb-4">
                <div className="flex-1 h-16 bg-[#1e1e1e] rounded-lg" />
                <div className="flex-1 h-16 bg-[#1e1e1e] rounded-lg" />
              </div>
              <div className="h-3 bg-[#1e1e1e] rounded w-full" />
            </div>
          ))}
        </div>
      )}

      {/* Market cards grid */}
      {!isLoading && sortedMarkets.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {sortedMarkets.map(market => {
            const sum = market.yes_price + market.no_price
            const isArb = sum < 0.97 && sum > 0.01
            const catColor = CATEGORY_COLORS[market.category] ?? CATEGORY_COLORS.other
            const savings = ((0.97 - sum) * 100).toFixed(1)

            return (
              <div
                key={market.id}
                className={`bg-[#111111] rounded-lg p-4 flex flex-col gap-3 transition-all ${
                  isArb
                    ? 'border border-[#f5a623]/60 shadow-[0_0_16px_rgba(245,166,35,0.12)]'
                    : 'border border-[#1e1e1e] hover:border-[#333333]'
                }`}
              >
                {/* Title row */}
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-semibold text-white leading-tight line-clamp-2 flex-1">
                    {market.title}
                  </p>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {isArb && (
                      <span className="px-1.5 py-0.5 text-[10px] font-bold bg-[#f5a623] text-black rounded uppercase tracking-wide">
                        ARB
                      </span>
                    )}
                    <span
                      className="px-1.5 py-0.5 text-[10px] font-semibold rounded capitalize"
                      style={{ backgroundColor: `${catColor}22`, color: catColor }}
                    >
                      {market.category}
                    </span>
                  </div>
                </div>

                {/* YES / NO price buttons */}
                <div className="flex gap-2">
                  <button className="flex-1 bg-[#00c853]/10 border border-[#00c853]/30 hover:bg-[#00c853]/20 rounded-lg py-3 text-center transition-colors">
                    <div className="text-[10px] text-[#888888] mb-1 font-medium uppercase tracking-wide">YES</div>
                    <div className="text-2xl font-black text-[#00c853]">
                      {(market.yes_price * 100).toFixed(0)}¢
                    </div>
                  </button>
                  <button className="flex-1 bg-[#ff1744]/10 border border-[#ff1744]/30 hover:bg-[#ff1744]/20 rounded-lg py-3 text-center transition-colors">
                    <div className="text-[10px] text-[#888888] mb-1 font-medium uppercase tracking-wide">NO</div>
                    <div className="text-2xl font-black text-[#ff1744]">
                      {(market.no_price * 100).toFixed(0)}¢
                    </div>
                  </button>
                </div>

                {/* Stats row */}
                <div className="flex items-center gap-4 text-xs text-[#888888]">
                  <div>
                    <span className="text-[#555555]">Vol 24h </span>
                    <span className="text-[#e8e8e8] font-mono">{formatCompact(market.volume_24h)}</span>
                  </div>
                  <div>
                    <span className="text-[#555555]">Liq </span>
                    <span className="text-[#e8e8e8] font-mono">{formatCompact(market.liquidity)}</span>
                  </div>
                  <div className="ml-auto">
                    <span className="text-[#555555]">Ends </span>
                    <span className="text-[#e8e8e8]">
                      {new Date(market.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' })}
                    </span>
                  </div>
                </div>

                {/* Action row */}
                <div className="flex items-center justify-between pt-1 border-t border-[#1e1e1e]">
                  {isArb ? (
                    <span className="text-xs text-[#f5a623]/80 font-mono">
                      Sum {(sum * 100).toFixed(0)}¢ · save {savings}¢
                    </span>
                  ) : (
                    <span />
                  )}
                  <button
                    onClick={() => tradeMutation.mutate(market)}
                    disabled={tradeMutation.isPending}
                    className="px-4 py-1.5 text-xs font-bold rounded-lg border border-[#f5a623]/40 text-[#f5a623] hover:bg-[#f5a623]/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {tradeMutation.isPending ? 'PLACING...' : 'TRADE'}
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && sortedMarkets.length === 0 && !usingDemo && (
        <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg px-6 py-12 text-center">
          <p className="text-[#888888] text-sm">No markets match the current filter.</p>
          <p className="text-xs text-[#555555] mt-2">Try switching to "All" to see all available markets.</p>
        </div>
      )}

      {/* Open Positions from Polymarket */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-sm font-semibold text-white mb-3">Open Positions — Polymarket</h2>
        {positions.length === 0 ? (
          <p className="text-xs text-[#888888] text-center py-6">
            No open Polymarket positions. Use the TRADE button above to enter a position.
          </p>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[#1e1e1e] text-[#888888]">
                <th className="text-left py-2">Market</th>
                <th className="text-right py-2">Side</th>
                <th className="text-right py-2">Qty</th>
                <th className="text-right py-2">Avg Price</th>
                <th className="text-right py-2">Unrealized P&L</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos: any, i: number) => (
                <tr key={i} className="border-b border-[#1e1e1e] last:border-0 hover:bg-[#1e1e1e]/30">
                  <td className="py-2 font-mono text-[#e8e8e8] max-w-[200px] truncate">{pos.symbol}</td>
                  <td className={`py-2 text-right font-semibold ${pos.side === 'buy' ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
                    {pos.side?.toUpperCase()}
                  </td>
                  <td className="py-2 text-right text-[#e8e8e8]">{pos.qty}</td>
                  <td className="py-2 text-right font-mono text-[#888888]">${(pos.avg_price ?? 0).toFixed(3)}</td>
                  <td className={`py-2 text-right font-medium ${(pos.unrealized_pnl ?? 0) >= 0 ? 'text-[#00c853]' : 'text-[#ff1744]'}`}>
                    {(pos.unrealized_pnl ?? 0) >= 0 ? '+' : ''}${(pos.unrealized_pnl ?? 0).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
