interface Props {
  symbol?: string
  height?: number
}

export default function LiveChartPlaceholder({ symbol = 'BTC/USD', height = 480 }: Props) {
  const displaySymbol = symbol.replace('BINANCE:', '').replace('NASDAQ:', '').replace('NYSE:', '')

  return (
    <div
      className="relative bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg overflow-hidden flex flex-col items-center justify-center select-none"
      style={{ height }}
    >
      <div className="text-center space-y-3 px-8">
        <p className="text-2xl font-bold font-mono text-[#e8e8e8]">{displaySymbol}</p>
        <div className="w-16 h-16 mx-auto rounded-full bg-[#1e1e1e] flex items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
        </div>
        <p className="text-sm font-semibold text-[#888888]">
          Connect your broker API for live {displaySymbol} chart
        </p>
        <p className="text-xs text-[#555]">
          Add your Alpaca or Binance API key in Settings to stream real-time price data.
        </p>
        <a
          href="/settings"
          className="inline-block mt-2 px-4 py-2 rounded-lg text-xs font-bold text-black transition-all duration-200 hover:opacity-90"
          style={{ background: '#f5a623' }}
        >
          Add API Keys in Settings
        </a>
      </div>
    </div>
  )
}
