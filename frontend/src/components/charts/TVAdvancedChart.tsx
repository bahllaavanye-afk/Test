import { useEffect, useRef } from 'react'

interface Props {
  symbol?: string
  interval?: string
  height?: number
  studies?: string[]
  watchlist?: string[]
  showWatchlist?: boolean
  showDetails?: boolean
}

const DEFAULT_STUDIES = [
  'Volume@tv-basicstudies',
  'MACD@tv-basicstudies',
  'RSI@tv-basicstudies',
]

const EQUITY_WATCHLIST = [
  'NASDAQ:AAPL', 'NASDAQ:MSFT', 'NASDAQ:NVDA', 'NASDAQ:GOOGL',
  'NASDAQ:AMZN', 'NASDAQ:META', 'NASDAQ:TSLA', 'NYSE:SPY',
  'NYSE:QQQ', 'NYSE:IWM', 'NYSE:GLD', 'NYSE:TLT',
]

export default function TVAdvancedChart({
  symbol = 'NASDAQ:AAPL',
  interval = '60',
  height = 520,
  studies = DEFAULT_STUDIES,
  watchlist = EQUITY_WATCHLIST,
  showWatchlist = true,
  showDetails = true,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    containerRef.current.innerHTML = ''

    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'
    script.type = 'text/javascript'
    script.async = true
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol,
      interval,
      timezone: 'America/New_York',
      theme: 'dark',
      style: '1',
      locale: 'en',
      backgroundColor: '#0a0a0a',
      gridColor: '#1e1e1e',
      hide_side_toolbar: false,
      allow_symbol_change: true,
      withdateranges: true,
      show_popup_button: true,
      popup_width: '1200',
      popup_height: '700',
      details: showDetails,
      hotlist: true,
      calendar: false,
      watchlist: showWatchlist ? watchlist : undefined,
      studies,
      support_host: 'https://www.tradingview.com',
    })

    containerRef.current.appendChild(script)
  }, [symbol, interval])

  return (
    <div
      className="tradingview-widget-container rounded-lg overflow-hidden border border-[#1e1e1e]"
      ref={containerRef}
      style={{ height, width: '100%' }}
    >
      <div className="tradingview-widget-container__widget" style={{ height: '100%', width: '100%' }} />
    </div>
  )
}

// Named export alias so consumers can use either:
//   import TVAdvancedChart from '...'           (default)
//   import { TVAdvancedChart } from '...'       (named)
export { TVAdvancedChart };
