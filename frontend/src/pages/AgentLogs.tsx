import { useState, useEffect, useRef, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Flame, Pause, Play, X, CheckCircle } from 'lucide-react'
import { api } from '../api/client'
import { getToken } from '../api/client'
import { wsBase } from '../utils/endpoints'

// ── Types ─────────────────────────────────────────────────────────────────────

interface AgentLog {
  id: string
  employee_id: string | null
  agent_type: string | null
  action: string
  tool_used: string | null
  input_summary: string | null
  output_summary: string | null
  duration_ms: number | null
  status: string
  error_message: string | null
  anomaly_score: number
  is_anomaly: boolean
  reviewed_by: string | null
  reviewed_at: string | null
  review_note: string | null
  strategy_name: string | null
  symbol: string | null
  account_id: string | null
  created_at: string | null
}

interface LogStats {
  hours: number
  total_actions: number
  error_count: number
  anomaly_count: number
  active_employees: number
  by_employee: Record<string, number>
  by_action: Record<string, number>
  by_status: Record<string, number>
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { color: string; label: string }> = {
    ok: { color: '#00c853', label: 'OK' },
    warning: { color: '#f5a623', label: 'WARN' },
    error: { color: '#ff1744', label: 'ERR' },
  }
  const s = map[status] || { color: '#888', label: status.toUpperCase() }
  return (
    <span
      style={{
        color: s.color,
        border: `1px solid ${s.color}40`,
        background: `${s.color}18`,
        padding: '1px 6px',
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: 1,
        fontFamily: 'JetBrains Mono, monospace',
      }}
    >
      {s.label}
    </span>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function AgentLogs() {
  const queryClient = useQueryClient()
  const [selectedLog, setSelectedLog] = useState<AgentLog | null>(null)
  const [logs, setLogs] = useState<AgentLog[]>([])
  const [paused, setPaused] = useState(false)
  const pausedRef = useRef(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reviewNoteRef = useRef('')

  // Filters
  const [filterEmployee, setFilterEmployee] = useState('')
  const [filterAgentType, setFilterAgentType] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [filterAnomalyOnly, setFilterAnomalyOnly] = useState(false)

  pausedRef.current = paused

  // ── REST: initial load ───────────────────────────────────────────────────
  const params = new URLSearchParams()
  params.set('limit', '100')
  if (filterEmployee) params.set('employee_id', filterEmployee)
  if (filterAgentType) params.set('agent_type', filterAgentType)
  if (filterStatus) params.set('status', filterStatus)
  if (filterAnomalyOnly) params.set('anomaly_only', 'true')

  const logsQuery = useQuery<AgentLog[]>({
    queryKey: ['agent-logs', filterEmployee, filterAgentType, filterStatus, filterAnomalyOnly],
    queryFn: () => api.get(`/agent-logs/?${params.toString()}`).then(r => r.data),
    staleTime: 10_000,
  })

  // ── REST: stats ──────────────────────────────────────────────────────────
  const statsQuery = useQuery<LogStats>({
    queryKey: ['agent-logs-stats'],
    queryFn: () => api.get('/agent-logs/stats').then(r => r.data),
    staleTime: 30_000,
    refetchInterval: 30_000,
  })

  // Sync REST data into local state on load/filter change
  useEffect(() => {
    if (logsQuery.data) {
      setLogs(logsQuery.data)
    }
  }, [logsQuery.data])

  // ── WebSocket ────────────────────────────────────────────────────────────
  const connectWs = useCallback(() => {
    const token = getToken()
    if (!token) return
    const ws = new WebSocket(`${wsBase()}/ws/agent-logs?token=${encodeURIComponent(token)}`)
    wsRef.current = ws

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg.type === 'heartbeat') return
        if (msg.type !== 'agent_log') return
        if (pausedRef.current) return

        const newLog: AgentLog = {
          id: msg.id,
          employee_id: msg.employee_id,
          agent_type: msg.agent_type,
          action: msg.action,
          tool_used: msg.tool_used,
          input_summary: msg.input_summary,
          output_summary: msg.output_summary,
          duration_ms: msg.duration_ms,
          status: msg.status,
          error_message: msg.error_message,
          anomaly_score: msg.anomaly_score,
          is_anomaly: msg.is_anomaly,
          reviewed_by: null,
          reviewed_at: null,
          review_note: null,
          strategy_name: msg.strategy_name,
          symbol: msg.symbol,
          account_id: msg.account_id,
          created_at: msg.created_at,
        }

        // Apply client-side filters
        if (filterEmployee && newLog.employee_id !== filterEmployee) return
        if (filterAgentType && newLog.agent_type !== filterAgentType) return
        if (filterStatus && newLog.status !== filterStatus) return
        if (filterAnomalyOnly && !newLog.is_anomaly) return

        setLogs(prev => [newLog, ...prev].slice(0, 500))
        // Invalidate stats periodically
        queryClient.invalidateQueries({ queryKey: ['agent-logs-stats'] })
      } catch {
        // ignore parse errors
      }
    }

    ws.onerror = () => {}
    ws.onclose = () => {
      // Reconnect after 5s
      setTimeout(() => {
        if (wsRef.current === ws) connectWs()
      }, 5000)
    }
  }, [filterEmployee, filterAgentType, filterStatus, filterAnomalyOnly, queryClient])

  useEffect(() => {
    connectWs()
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
      }
    }
  }, [connectWs])

  // ── Review mutation ──────────────────────────────────────────────────────
  const reviewMutation = useMutation({
    mutationFn: ({ logId, note, resolved }: { logId: string; note: string; resolved: boolean }) =>
      api.post(`/agent-logs/${logId}/review`, { note, resolved }).then(r => r.data),
    onSuccess: (data) => {
      setLogs(prev =>
        prev.map(l =>
          l.id === data.id
            ? { ...l, reviewed_by: data.reviewed_by, reviewed_at: data.reviewed_at, review_note: data.review_note, is_anomaly: data.is_anomaly }
            : l
        )
      )
      if (selectedLog?.id === data.id) {
        setSelectedLog(prev => prev ? { ...prev, ...data } : null)
      }
    },
  })

  const stats = statsQuery.data
  const statItems = [
    { label: 'Total Actions (24h)', value: stats?.total_actions ?? '—', color: '#e8e8e8' },
    { label: 'Errors', value: stats?.error_count ?? '—', color: '#ff1744' },
    { label: 'Anomalies', value: stats?.anomaly_count ?? '—', color: '#f5a623' },
    { label: 'Active Employees', value: stats?.active_employees ?? '—', color: '#00c853' },
  ]

  return (
    <div style={{ background: '#0a0a0a', minHeight: '100vh', padding: '20px 24px', fontFamily: 'JetBrains Mono, monospace' }}>
      {/* Title */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ color: '#e8e8e8', fontSize: 20, fontWeight: 700, margin: 0 }}>
          Agent Activity Logs
        </h1>
        <p style={{ color: '#555', fontSize: 12, margin: '4px 0 0' }}>
          Real-time stream of every agent and employee action
        </p>
      </div>

      {/* Stats bar */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        {statItems.map(s => (
          <div
            key={s.label}
            style={{
              background: '#111',
              border: '1px solid #1e1e1e',
              borderRadius: 8,
              padding: '12px 20px',
              minWidth: 140,
              flex: '1 1 140px',
            }}
          >
            <div style={{ fontSize: 11, color: '#555', marginBottom: 4 }}>{s.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          placeholder="Employee ID"
          value={filterEmployee}
          onChange={e => setFilterEmployee(e.target.value)}
          style={inputStyle}
        />
        <select
          value={filterAgentType}
          onChange={e => setFilterAgentType(e.target.value)}
          style={inputStyle}
        >
          <option value="">All Agent Types</option>
          <option value="strategy">strategy</option>
          <option value="ml">ml</option>
          <option value="execution">execution</option>
          <option value="risk">risk</option>
          <option value="human">human</option>
          <option value="system">system</option>
        </select>
        <select
          value={filterStatus}
          onChange={e => setFilterStatus(e.target.value)}
          style={inputStyle}
        >
          <option value="">All Statuses</option>
          <option value="ok">ok</option>
          <option value="warning">warning</option>
          <option value="error">error</option>
        </select>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#888', fontSize: 12, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={filterAnomalyOnly}
            onChange={e => setFilterAnomalyOnly(e.target.checked)}
          />
          Anomaly only
        </label>
        <button
          onClick={() => setPaused(p => !p)}
          style={{
            marginLeft: 'auto',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 14px',
            background: paused ? '#f5a62320' : '#1e1e1e',
            border: `1px solid ${paused ? '#f5a623' : '#333'}`,
            borderRadius: 6,
            color: paused ? '#f5a623' : '#888',
            cursor: 'pointer',
            fontSize: 12,
          }}
        >
          {paused ? <Play size={14} /> : <Pause size={14} />}
          {paused ? 'Resume' : 'Pause'}
        </button>
      </div>

      {/* Live feed table */}
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1, overflowX: 'auto' }}>
          {logsQuery.isLoading && (
            <div style={{ color: '#555', fontSize: 12, padding: 20 }}>Loading…</div>
          )}
          {logsQuery.isError && (
            <div style={{ color: '#ff1744', fontSize: 12, padding: 20 }}>Failed to load logs.</div>
          )}
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #1e1e1e' }}>
                {['Time', 'Employee', 'Agent Type', 'Action', 'Tool', 'Duration', 'Status', 'Anomaly'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '6px 10px', color: '#555', fontWeight: 600, whiteSpace: 'nowrap' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logs.map(log => (
                <tr
                  key={log.id}
                  onClick={() => setSelectedLog(log)}
                  style={{
                    borderBottom: '1px solid #111',
                    cursor: 'pointer',
                    background: selectedLog?.id === log.id ? '#1a1a1a' : 'transparent',
                    transition: 'background 0.1s',
                  }}
                  onMouseEnter={e => { if (selectedLog?.id !== log.id) (e.currentTarget as HTMLElement).style.background = '#141414' }}
                  onMouseLeave={e => { if (selectedLog?.id !== log.id) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
                >
                  <td style={{ padding: '5px 10px', color: '#555', whiteSpace: 'nowrap' }}>
                    {formatTime(log.created_at)}
                  </td>
                  <td style={{ padding: '5px 10px', color: '#e8e8e8', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {log.employee_id || '—'}
                  </td>
                  <td style={{ padding: '5px 10px', color: '#888' }}>
                    {log.agent_type || '—'}
                  </td>
                  <td style={{ padding: '5px 10px', color: '#e8e8e8', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {log.action}
                  </td>
                  <td style={{ padding: '5px 10px', color: '#888' }}>
                    {log.tool_used || '—'}
                  </td>
                  <td style={{ padding: '5px 10px', color: '#555', whiteSpace: 'nowrap' }}>
                    {log.duration_ms != null ? `${log.duration_ms}ms` : '—'}
                  </td>
                  <td style={{ padding: '5px 10px' }}>
                    <StatusBadge status={log.status} />
                  </td>
                  <td style={{ padding: '5px 10px', textAlign: 'center' }}>
                    {log.is_anomaly && <Flame size={14} color="#f5a623" />}
                  </td>
                </tr>
              ))}
              {logs.length === 0 && !logsQuery.isLoading && (
                <tr>
                  <td colSpan={8} style={{ padding: 30, textAlign: 'center', color: '#333' }}>
                    No log entries yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Detail panel */}
        {selectedLog && (
          <div
            style={{
              width: 340,
              minWidth: 320,
              background: '#111',
              border: '1px solid #1e1e1e',
              borderRadius: 8,
              padding: 16,
              fontSize: 12,
              alignSelf: 'flex-start',
              position: 'sticky',
              top: 20,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ color: '#e8e8e8', fontWeight: 700, fontSize: 13 }}>Log Detail</span>
              <button
                onClick={() => setSelectedLog(null)}
                style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', padding: 0 }}
              >
                <X size={16} />
              </button>
            </div>

            <DetailRow label="ID" value={selectedLog.id.slice(0, 8) + '…'} />
            <DetailRow label="Time" value={formatTime(selectedLog.created_at)} />
            <DetailRow label="Employee" value={selectedLog.employee_id || '—'} />
            <DetailRow label="Agent Type" value={selectedLog.agent_type || '—'} />
            <DetailRow label="Action" value={selectedLog.action} />
            <DetailRow label="Tool" value={selectedLog.tool_used || '—'} />
            <DetailRow label="Duration" value={selectedLog.duration_ms != null ? `${selectedLog.duration_ms}ms` : '—'} />
            <DetailRow label="Status" value={<StatusBadge status={selectedLog.status} />} />
            <DetailRow label="Anomaly Score" value={selectedLog.anomaly_score.toFixed(2)} />
            {selectedLog.strategy_name && <DetailRow label="Strategy" value={selectedLog.strategy_name} />}
            {selectedLog.symbol && <DetailRow label="Symbol" value={selectedLog.symbol} />}

            {selectedLog.input_summary && (
              <div style={{ marginTop: 10 }}>
                <div style={{ color: '#555', marginBottom: 3 }}>Input</div>
                <div style={{ color: '#888', background: '#0d0d0d', padding: '6px 8px', borderRadius: 4, wordBreak: 'break-word' }}>
                  {selectedLog.input_summary}
                </div>
              </div>
            )}
            {selectedLog.output_summary && (
              <div style={{ marginTop: 10 }}>
                <div style={{ color: '#555', marginBottom: 3 }}>Output</div>
                <div style={{ color: '#888', background: '#0d0d0d', padding: '6px 8px', borderRadius: 4, wordBreak: 'break-word' }}>
                  {selectedLog.output_summary}
                </div>
              </div>
            )}
            {selectedLog.error_message && (
              <div style={{ marginTop: 10 }}>
                <div style={{ color: '#ff1744', marginBottom: 3 }}>Error</div>
                <div style={{ color: '#ff174480', background: '#0d0d0d', padding: '6px 8px', borderRadius: 4, wordBreak: 'break-word' }}>
                  {selectedLog.error_message}
                </div>
              </div>
            )}

            {/* Review section */}
            <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid #1e1e1e' }}>
              <div style={{ color: '#555', marginBottom: 8, fontWeight: 600 }}>Review</div>
              {selectedLog.reviewed_by ? (
                <div style={{ color: '#00c853', fontSize: 11 }}>
                  <CheckCircle size={12} style={{ display: 'inline', marginRight: 4 }} />
                  Reviewed by {selectedLog.reviewed_by}
                  {selectedLog.review_note && (
                    <div style={{ color: '#888', marginTop: 4 }}>{selectedLog.review_note}</div>
                  )}
                </div>
              ) : (
                <div style={{ color: '#555', fontSize: 11 }}>Not yet reviewed.</div>
              )}

              <textarea
                placeholder="Review note…"
                defaultValue=""
                onChange={e => { reviewNoteRef.current = e.target.value }}
                style={{
                  marginTop: 8,
                  width: '100%',
                  background: '#0d0d0d',
                  border: '1px solid #222',
                  borderRadius: 4,
                  color: '#e8e8e8',
                  fontSize: 11,
                  padding: '6px 8px',
                  resize: 'vertical',
                  minHeight: 56,
                  boxSizing: 'border-box',
                  fontFamily: 'JetBrains Mono, monospace',
                }}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button
                  onClick={() =>
                    reviewMutation.mutate({
                      logId: selectedLog.id,
                      note: reviewNoteRef.current,
                      resolved: true,
                    })
                  }
                  disabled={reviewMutation.isPending}
                  style={{
                    flex: 1,
                    padding: '6px',
                    background: '#00c85320',
                    border: '1px solid #00c853',
                    borderRadius: 4,
                    color: '#00c853',
                    cursor: 'pointer',
                    fontSize: 11,
                  }}
                >
                  Mark Resolved
                </button>
                <button
                  onClick={() =>
                    reviewMutation.mutate({
                      logId: selectedLog.id,
                      note: reviewNoteRef.current,
                      resolved: false,
                    })
                  }
                  disabled={reviewMutation.isPending}
                  style={{
                    flex: 1,
                    padding: '6px',
                    background: '#f5a62320',
                    border: '1px solid #f5a623',
                    borderRadius: 4,
                    color: '#f5a623',
                    cursor: 'pointer',
                    fontSize: 11,
                  }}
                >
                  Flag Anomaly
                </button>
              </div>
              {reviewMutation.isError && (
                <div style={{ color: '#ff1744', fontSize: 11, marginTop: 6 }}>
                  {(reviewMutation.error as Error)?.message?.includes('403')
                    ? 'Superuser access required.'
                    : 'Review failed.'}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Small helpers ─────────────────────────────────────────────────────────────

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', gap: 8, marginBottom: 5, alignItems: 'flex-start' }}>
      <span style={{ color: '#555', minWidth: 96, flexShrink: 0 }}>{label}</span>
      <span style={{ color: '#e8e8e8', wordBreak: 'break-all' }}>{value}</span>
    </div>
  )
}

const inputStyle: React.CSSProperties = {
  background: '#111',
  border: '1px solid #1e1e1e',
  borderRadius: 6,
  color: '#e8e8e8',
  fontSize: 12,
  padding: '6px 10px',
  fontFamily: 'JetBrains Mono, monospace',
  outline: 'none',
}
