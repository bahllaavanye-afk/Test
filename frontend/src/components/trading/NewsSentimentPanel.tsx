import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

interface NewsItem {
  id: number | string | null
  headline: string
  summary: string
  source: string
  url: string
  author: string
  created_at: string | null
  updated_at: string | null
  symbols: string[]
  sentiment_score: number
}

interface NewsResponse {
  news: NewsItem[]
  error?: string
  data_source?: string
}

interface Props {
  symbols: string[]
}

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return ''
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}

function SentimentBar({ score }: { score: number }) {
  const isPositive = score >= 0
  const width = `${Math.abs(score) * 100}%`
  return (
    <div className="h-0.5 bg-[#1e1e1e] rounded-full overflow-hidden mt-1.5">
      <div
        className="h-full rounded-full transition-all duration-300"
        style={{
          width,
          background: isPositive ? '#00c853' : '#ff1744',
          marginLeft: isPositive ? undefined : 'auto',
        }}
      />
    </div>
  )
}

export default function NewsSentimentPanel({ symbols }: Props) {
  const symbolsStr = symbols.join(',')

  const { data, isLoading, isError } = useQuery<NewsResponse>({
    queryKey: ['market-news', symbolsStr],
    queryFn: () =>
      api
        .get(`/market-data/news?symbols=${encodeURIComponent(symbolsStr)}&limit=20`)
        .then((r) => r.data)
        .catch(() => ({ news: [], data_source: 'unavailable' })),
    refetchInterval: 60_000,
    staleTime: 30_000,
  })

  const news: NewsItem[] = data?.news ?? []
  const avgSentiment =
    news.length > 0
      ? news.reduce((sum, item) => sum + item.sentiment_score, 0) / news.length
      : 0

  const dotColor =
    avgSentiment > 0.2 ? '#00c853' : avgSentiment < -0.2 ? '#ff1744' : '#555555'

  if (isError || data?.data_source === 'unavailable') {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center gap-2 px-3 py-2 border-b border-[#1e1e1e]">
          <span
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{ background: '#555555' }}
          />
          <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">
            Market News
          </span>
        </div>
        <div className="flex items-center justify-center flex-1 px-4">
          <p className="text-xs text-[#444] text-center">News feed unavailable</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[#1e1e1e] flex-shrink-0">
        <span
          className="w-2 h-2 rounded-full flex-shrink-0 transition-colors duration-500"
          style={{ background: dotColor }}
        />
        <span className="text-[10px] text-[#555] uppercase tracking-wider font-medium">
          Market News
        </span>
        {news.length > 0 && (
          <span className="ml-auto text-[9px] text-[#333]">{news.length} articles</span>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto divide-y divide-[#111]">
        {isLoading ? (
          <div className="p-3 space-y-3">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="space-y-1.5">
                <div className="h-3 bg-[#1a1a1a] rounded animate-pulse w-5/6" />
                <div className="h-2.5 bg-[#1a1a1a] rounded animate-pulse w-3/4" />
                <div className="h-0.5 bg-[#1a1a1a] rounded animate-pulse w-1/3" />
              </div>
            ))}
          </div>
        ) : news.length === 0 ? (
          <div className="flex items-center justify-center h-full px-4">
            <p className="text-xs text-[#444]">No recent news</p>
          </div>
        ) : (
          news.map((item, idx) => (
            <a
              key={item.id ?? idx}
              href={item.url || '#'}
              target="_blank"
              rel="noopener noreferrer"
              className="block px-3 py-2.5 hover:bg-[#111] transition-colors cursor-pointer"
            >
              {/* Headline */}
              <p
                className="text-sm text-[#e8e8e8] leading-snug"
                style={{
                  display: '-webkit-box',
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden',
                }}
              >
                {item.headline}
              </p>

              {/* Meta: source + time */}
              <div className="flex items-center gap-1.5 mt-1">
                <span className="text-[10px] text-[#444]">{item.source}</span>
                {item.source && item.created_at && (
                  <span className="text-[10px] text-[#333]">·</span>
                )}
                <span className="text-[10px] text-[#444]">{timeAgo(item.created_at)}</span>
              </div>

              {/* Sentiment bar */}
              <SentimentBar score={item.sentiment_score} />

              {/* Symbol tags */}
              {item.symbols.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1.5">
                  {item.symbols.slice(0, 5).map((sym) => (
                    <span
                      key={sym}
                      className="text-[9px] px-1 py-0.5 rounded font-mono font-bold"
                      style={{
                        background: 'rgba(245,166,35,0.12)',
                        color: '#f5a623',
                      }}
                    >
                      {sym}
                    </span>
                  ))}
                  {item.symbols.length > 5 && (
                    <span className="text-[9px] text-[#333]">+{item.symbols.length - 5}</span>
                  )}
                </div>
              )}
            </a>
          ))
        )}
      </div>
    </div>
  )
}
