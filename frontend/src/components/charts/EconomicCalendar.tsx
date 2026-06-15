/**
 * EconomicCalendar — upcoming high-impact macro events.
 * Data from FRED / public economic calendar APIs (no auth).
 * Shows: event name, impact level (high/medium/low), country flag,
 * actual vs forecast vs previous, and countdown to release.
 */
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'

interface EconEvent {
  id: string
  name: string
  country: string
  country_flag: string
  impact: 'high' | 'medium' | 'low'
  scheduled_at: string   // ISO
  actual: string | null
  forecast: string | null
  previous: string | null
}

const IMPACT_STYLE: Record<string, string> = {
  high:   'text-[#ff4d4d] bg-[#ff4d4d]/10 border-[#ff4d4d]/30',
  medium: 'text-[#ffa726] bg-[#ffa726]/10 border-[#ffa726]/30',
  low:    'text-[#8a8a9a] bg-[#8a8a9a]/10 border-[#8a8a9a]/30',
}

function formatCountdown(iso: string): string {
  const diff = new Date(iso).getTime() - Date.now()
  if (diff < 0) return 'Released'
  const h = Math.floor(diff / 3_600_000)
  const m = Math.floor((diff % 3_600_000) / 60_000)
  if (h > 48) return `${Math.floor(h / 24)}d`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

interface EventRowProps { event: EconEvent }

function EventRow({ event: e }: EventRowProps) {
  const released = !!e.actual
  const beat = released && e.forecast
    ? parseFloat(e.actual ?? '0') > parseFloat(e.forecast)
    : null

  return (
    <div className={`flex items-start gap-2 px-3 py-2 border-b border-[#1e1e2e] last:border-0 ${
      released ? 'opacity-60' : ''
    }`}>
      {/* Impact + time */}
      <div className="w-12 flex flex-col items-center gap-0.5 mt-0.5 shrink-0">
        <span className={`text-[9px] font-semibold px-1 py-0.5 rounded border ${IMPACT_STYLE[e.impact]}`}>
          {e.impact.toUpperCase()[0]}
        </span>
        <span className="text-[9px] text-[#5a5a7a]">{formatTime(e.scheduled_at)}</span>
      </div>

      {/* Event info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1">
          <span className="text-[10px]">{e.country_flag}</span>
          <span className="text-[11px] font-medium text-white truncate">{e.name}</span>
        </div>
        <div className="flex items-center gap-2 mt-0.5">
          {e.actual && (
            <span className={`text-[10px] font-mono font-semibold ${
              beat === true ? 'text-[#00ff88]' : beat === false ? 'text-[#ff4d4d]' : 'text-white'
            }`}>
              A: {e.actual}
            </span>
          )}
          {e.forecast && (
            <span className="text-[10px] font-mono text-[#6a6a8a]">F: {e.forecast}</span>
          )}
          {e.previous && (
            <span className="text-[10px] font-mono text-[#5a5a5a]">P: {e.previous}</span>
          )}
          {!released && (
            <span className="text-[10px] text-[#7c3aed] ml-auto">{formatCountdown(e.scheduled_at)}</span>
          )}
        </div>
      </div>
    </div>
  )
}

export function EconomicCalendar({ className = '' }: { className?: string }) {
  const { data: events = [], isLoading } = useQuery<EconEvent[]>({
    queryKey: ['economic-calendar'],
    queryFn: async () => {
      const { data } = await api.get('/market-data/economic-calendar')
      return Array.isArray(data) ? data : []
    },
    refetchInterval: 300_000,
    staleTime: 290_000,
  })

  const upcoming = events.filter(e => !e.actual).slice(0, 8)
  const recent   = events.filter(e => e.actual).slice(0, 4)

  return (
    <div className={`bg-[#0d0d14] border border-[#1e1e2e] rounded-lg overflow-hidden flex flex-col ${className}`}>
      {/* Header */}
      <div className="px-3 py-2 border-b border-[#1e1e2e]">
        <span className="text-[11px] font-semibold text-[#8a8a9a] uppercase tracking-wider">
          Economic Calendar
        </span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {isLoading && (
          <div className="flex items-center justify-center h-16 text-[11px] text-[#4a4a5a]">
            Loading events…
          </div>
        )}
        {!isLoading && events.length === 0 && (
          <div className="flex items-center justify-center h-16 text-[11px] text-[#4a4a5a]">
            No events today
          </div>
        )}
        {upcoming.map(e => <EventRow key={e.id} event={e} />)}
        {recent.length > 0 && (
          <>
            <div className="px-3 py-1 text-[9px] text-[#4a4a5a] uppercase tracking-wider bg-[#111120]">
              Released
            </div>
            {recent.map(e => <EventRow key={e.id} event={e} />)}
          </>
        )}
      </div>
    </div>
  )
}
