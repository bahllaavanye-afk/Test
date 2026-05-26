import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

const CATEGORIES = ['All', 'order', 'signal', 'risk', 'experiment', 'system'] as const
type Category = typeof CATEGORIES[number]

const BADGE_COLORS: Record<string, string> = {
  order: '#2196f3',
  signal: '#f5a623',
  risk: '#ff1744',
  experiment: '#9c27b0',
  system: '#888888',
}

function relativeTime(ts: string): string {
  const diff = Math.floor((Date.now() - new Date(ts).getTime()) / 1000)
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

export default function Activity() {
  const [activeCategory, setActiveCategory] = useState<Category>('All')

  const { data: events } = useQuery({
    queryKey: ['activity'],
    queryFn: () => api.get('/notifications/activity?limit=100').then(r => r.data),
    refetchInterval: 2000,
  })

  const { data: stats } = useQuery({
    queryKey: ['activity-stats'],
    queryFn: () => api.get('/notifications/stats').then(r => r.data),
  })

  const filtered: any[] = (events ?? []).filter((e: any) =>
    activeCategory === 'All' || e.category === activeCategory
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">Activity Feed</h1>
        <span className="text-xs text-[#888888]">Auto-refreshes every 2s</span>
      </div>

      {/* Stats Card */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4">
        <h2 className="text-xs text-[#888888] uppercase tracking-wider mb-3">Event Stats</h2>
        <div className="flex flex-wrap gap-4">
          <div>
            <p className="text-xs text-[#888888]">Total Events</p>
            <p className="text-2xl font-bold text-[#f5a623]">{stats?.total ?? 0}</p>
          </div>
          {Object.entries(BADGE_COLORS).map(([cat, color]) => (
            <div key={cat}>
              <p className="text-xs text-[#888888] capitalize">{cat}</p>
              <p className="text-2xl font-bold" style={{ color }}>{stats?.by_category?.[cat] ?? 0}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Category Filter */}
      <div className="flex gap-2 flex-wrap">
        {CATEGORIES.map(cat => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className="px-3 py-1 rounded text-xs font-medium border transition-colors"
            style={
              activeCategory === cat
                ? {
                    backgroundColor: cat === 'All' ? '#f5a623' : BADGE_COLORS[cat],
                    borderColor: cat === 'All' ? '#f5a623' : BADGE_COLORS[cat],
                    color: '#0a0a0a',
                  }
                : {
                    backgroundColor: 'transparent',
                    borderColor: '#1e1e1e',
                    color: cat === 'All' ? '#e8e8e8' : BADGE_COLORS[cat],
                  }
            }
          >
            {cat === 'All' ? 'All' : cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>

      {/* Event List */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="divide-y divide-[#1e1e1e] max-h-[600px] overflow-y-auto">
          {filtered.length === 0 && (
            <div className="px-4 py-8 text-center text-xs text-[#888888]">
              No events yet — system activity will appear here.
            </div>
          )}
          {filtered.map((event: any, i: number) => (
            <div key={event.id ?? i} className="flex items-start gap-3 px-4 py-3 hover:bg-[#0a0a0a] transition-colors">
              <span className="text-xs text-[#888888] whitespace-nowrap mt-0.5 w-16 shrink-0">
                {event.timestamp ? relativeTime(event.timestamp) : '—'}
              </span>
              <span
                className="text-xs font-medium px-2 py-0.5 rounded shrink-0"
                style={{
                  backgroundColor: `${BADGE_COLORS[event.category] ?? '#888888'}22`,
                  color: BADGE_COLORS[event.category] ?? '#888888',
                }}
              >
                {event.category ?? 'unknown'}
              </span>
              <span className="text-xs text-[#e8e8e8] flex-1 min-w-0 break-words">
                {event.summary ?? event.message ?? JSON.stringify(event)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
