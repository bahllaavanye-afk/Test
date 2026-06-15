/**
 * AlertCenter — TradingView-style price alert management.
 * Create alerts on price levels, % moves, or indicator crossovers.
 * Alerts fire via WebSocket and appear as toast notifications.
 */
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'

type AlertType = 'price_above' | 'price_below' | 'change_pct' | 'rsi_overbought' | 'rsi_oversold'

interface Alert {
  id: string
  symbol: string
  type: AlertType
  value: number
  note: string
  active: boolean
  triggered_at: string | null
  created_at: string
}

const TYPE_LABELS: Record<AlertType, string> = {
  price_above:    'Price ≥',
  price_below:    'Price ≤',
  change_pct:     '% Move',
  rsi_overbought: 'RSI ≥',
  rsi_oversold:   'RSI ≤',
}

interface AlertCardProps {
  alert: Alert
  onDelete: (id: string) => void
}

function AlertCard({ alert, onDelete }: AlertCardProps) {
  const fired = !!alert.triggered_at
  return (
    <div className={`flex items-center justify-between px-3 py-2 rounded-md border ${
      fired ? 'border-[#00ff88]/30 bg-[#00ff88]/5' : 'border-[#1e1e2e] bg-[#0d0d14]'
    }`}>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-semibold text-white">{alert.symbol}</span>
          <span className="text-[10px] text-[#8a8a9a]">{TYPE_LABELS[alert.type]} {alert.value}</span>
          {fired && <span className="text-[9px] bg-[#00ff88]/20 text-[#00ff88] px-1.5 py-0.5 rounded">TRIGGERED</span>}
        </div>
        {alert.note && (
          <span className="text-[10px] text-[#5a5a7a] truncate block">{alert.note}</span>
        )}
      </div>
      <button
        onClick={() => onDelete(alert.id)}
        className="ml-2 text-[#5a5a7a] hover:text-[#ff4d4d] text-[14px] transition-colors"
        aria-label="Delete alert"
      >
        ×
      </button>
    </div>
  )
}

export function AlertCenter({ className = '' }: { className?: string }) {
  const qc = useQueryClient()
  const [symbol, setSymbol]  = useState('SPY')
  const [type, setType]      = useState<AlertType>('price_above')
  const [value, setValue]    = useState('')
  const [note, setNote]      = useState('')
  const [expanded, setExpanded] = useState(false)

  const { data: alerts = [] } = useQuery<Alert[]>({
    queryKey: ['alerts'],
    queryFn: async () => {
      const { data } = await api.get('/alerts')
      return Array.isArray(data) ? data : []
    },
    refetchInterval: 10_000,
  })

  const create = useMutation({
    mutationFn: async () => api.post('/alerts', {
      symbol: symbol.toUpperCase(),
      type, value: parseFloat(value), note,
    }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['alerts'] })
      setValue('')
      setNote('')
      setExpanded(false)
    },
  })

  const del = useMutation({
    mutationFn: (id: string) => api.delete(`/alerts/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts'] }),
  })

  const active = alerts.filter(a => a.active && !a.triggered_at)
  const fired  = alerts.filter(a => a.triggered_at)

  return (
    <div className={`bg-[#0d0d14] border border-[#1e1e2e] rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1e1e2e]">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold text-[#8a8a9a] uppercase tracking-wider">Alerts</span>
          {active.length > 0 && (
            <span className="text-[9px] bg-[#7c3aed]/20 text-[#a78bfa] px-1.5 py-0.5 rounded-full">
              {active.length} active
            </span>
          )}
          {fired.length > 0 && (
            <span className="text-[9px] bg-[#00ff88]/20 text-[#00ff88] px-1.5 py-0.5 rounded-full">
              {fired.length} fired
            </span>
          )}
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-[11px] text-[#00ff88] hover:text-white transition-colors"
        >
          + New
        </button>
      </div>

      {/* Create form */}
      {expanded && (
        <div className="px-3 py-2 border-b border-[#1e1e2e] bg-[#111120]">
          <div className="flex gap-2 mb-2">
            <input
              value={symbol}
              onChange={e => setSymbol(e.target.value.toUpperCase())}
              placeholder="Symbol"
              className="w-20 bg-[#0d0d14] text-[11px] text-white placeholder-[#4a4a5a] px-2 py-1 rounded border border-[#2a2a3a] focus:border-[#00ff88] focus:outline-none"
            />
            <select
              value={type}
              onChange={e => setType(e.target.value as AlertType)}
              className="flex-1 bg-[#0d0d14] text-[11px] text-white px-2 py-1 rounded border border-[#2a2a3a] focus:border-[#00ff88] focus:outline-none"
            >
              {Object.entries(TYPE_LABELS).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
            <input
              value={value}
              onChange={e => setValue(e.target.value)}
              placeholder="Value"
              type="number"
              className="w-20 bg-[#0d0d14] text-[11px] text-white placeholder-[#4a4a5a] px-2 py-1 rounded border border-[#2a2a3a] focus:border-[#00ff88] focus:outline-none"
            />
          </div>
          <input
            value={note}
            onChange={e => setNote(e.target.value)}
            placeholder="Note (optional)"
            className="w-full bg-[#0d0d14] text-[11px] text-white placeholder-[#4a4a5a] px-2 py-1 rounded border border-[#2a2a3a] focus:border-[#00ff88] focus:outline-none mb-2"
          />
          <button
            onClick={() => create.mutate()}
            disabled={!symbol || !value || create.isPending}
            className="w-full text-[11px] bg-[#00ff88]/10 text-[#00ff88] border border-[#00ff88]/30 rounded py-1.5 hover:bg-[#00ff88]/20 disabled:opacity-40 transition-colors"
          >
            {create.isPending ? 'Creating…' : 'Create Alert'}
          </button>
        </div>
      )}

      {/* Alert list */}
      <div className="flex-1 overflow-y-auto max-h-60 px-2 py-1.5 space-y-1">
        {alerts.length === 0 && (
          <div className="text-center text-[11px] text-[#4a4a5a] py-4">
            No alerts yet. Click + New to create one.
          </div>
        )}
        {[...active, ...fired].map(a => (
          <AlertCard key={a.id} alert={a} onDelete={id => del.mutate(id)} />
        ))}
      </div>
    </div>
  )
}
