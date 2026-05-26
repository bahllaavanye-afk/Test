import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

const DEFAULT_CATEGORIES = ['orders', 'fills', 'signals', 'decisions', 'risk']

function todayStr(): string {
  return new Date().toISOString().slice(0, 10)
}

export default function Archive() {
  const [selectedDate, setSelectedDate] = useState<string>(todayStr())
  const [selectedCategory, setSelectedCategory] = useState<string>('orders')

  const { data: archiveIndex } = useQuery({
    queryKey: ['archive-index'],
    queryFn: () => api.get('/archive/index').then(r => r.data),
  })

  const categories: string[] = archiveIndex?.categories ?? DEFAULT_CATEGORIES

  const { data: records, isFetching } = useQuery({
    queryKey: ['archive', selectedCategory, selectedDate],
    queryFn: () =>
      api.get(`/archive/${selectedCategory}?date=${selectedDate}&limit=200`).then(r => r.data),
    enabled: !!selectedDate && !!selectedCategory,
  })

  const rows: any[] = records ?? []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[#e8e8e8]">Trade Archive</h1>
        <span className="text-xs text-[#888888]">Historical record store</span>
      </div>

      {/* Controls */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg p-4 flex flex-wrap gap-4 items-center">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-[#888888] uppercase tracking-wider">Date</label>
          <input
            type="date"
            value={selectedDate}
            onChange={e => setSelectedDate(e.target.value)}
            className="bg-[#0a0a0a] border border-[#1e1e1e] rounded px-3 py-1.5 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623] transition-colors"
            max={todayStr()}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-[#888888] uppercase tracking-wider">Category</label>
          <div className="flex gap-2 flex-wrap">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className="px-3 py-1.5 rounded text-xs font-medium border transition-colors capitalize"
                style={
                  selectedCategory === cat
                    ? { backgroundColor: '#f5a623', borderColor: '#f5a623', color: '#0a0a0a' }
                    : { backgroundColor: 'transparent', borderColor: '#1e1e1e', color: '#888888' }
                }
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Records Table */}
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
        <div className="p-3 border-b border-[#1e1e1e] flex items-center justify-between">
          <h2 className="text-sm font-semibold capitalize">{selectedCategory} — {selectedDate}</h2>
          <div className="flex items-center gap-2">
            {isFetching && (
              <span className="text-xs text-[#f5a623]">Loading…</span>
            )}
            <span className="text-xs text-[#888888]">{rows.length} records</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-[#0a0a0a]">
              <tr className="text-xs text-[#888888]">
                <th className="text-left px-4 py-3 w-40">Timestamp</th>
                <th className="text-left px-4 py-3 w-36">Event Type</th>
                <th className="text-left px-4 py-3">Summary</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row: any, i: number) => (
                <tr key={row.id ?? i} className="border-t border-[#1e1e1e] hover:bg-[#0a0a0a] transition-colors align-top">
                  <td className="px-4 py-3 text-xs text-[#888888] whitespace-nowrap">
                    {row.timestamp ? new Date(row.timestamp).toLocaleString() : '—'}
                  </td>
                  <td className="px-4 py-3 text-xs font-mono text-[#f5a623] whitespace-nowrap">
                    {row.event_type ?? row.type ?? '—'}
                  </td>
                  <td className="px-4 py-3 text-xs text-[#e8e8e8] font-mono">
                    {typeof row.summary === 'string'
                      ? row.summary
                      : row.summary != null
                      ? (
                        <pre className="whitespace-pre-wrap break-all text-[10px] leading-relaxed text-[#aaaaaa]">
                          {JSON.stringify(row.summary, null, 2)}
                        </pre>
                      )
                      : (
                        <pre className="whitespace-pre-wrap break-all text-[10px] leading-relaxed text-[#aaaaaa]">
                          {JSON.stringify(
                            Object.fromEntries(
                              Object.entries(row).filter(([k]) => !['id', 'timestamp', 'event_type', 'type'].includes(k))
                            ),
                            null,
                            2
                          )}
                        </pre>
                      )}
                  </td>
                </tr>
              ))}
              {rows.length === 0 && !isFetching && (
                <tr>
                  <td colSpan={3} className="px-4 py-8 text-center text-xs text-[#888888]">
                    No records found for {selectedCategory} on {selectedDate}.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
