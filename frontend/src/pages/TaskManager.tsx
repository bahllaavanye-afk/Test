/**
 * TaskManager — Employee task dispatch board (Kanban-style).
 * Route: /tasks
 */
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, X, ChevronDown, ChevronUp } from 'lucide-react'
import api from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

type TaskStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
type TaskPriority = 'low' | 'medium' | 'high' | 'urgent'

interface Task {
  id: string
  title: string
  description: string | null
  task_type: string
  assigned_to: string | null
  assigned_by: string | null
  status: TaskStatus
  priority: TaskPriority
  params: Record<string, unknown>
  result: Record<string, unknown> | null
  error_message: string | null
  progress_pct: number
  created_at: string
  started_at: string | null
  completed_at: string | null
}

interface Employee {
  display: string
  skills: string[]
}

type EmployeeMap = Record<string, Employee>

// ── Constants ─────────────────────────────────────────────────────────────────

const TASK_TYPES = [
  'analyze', 'backtest', 'retrain', 'risk_check', 'optimize',
  'alpha_mining', 'predict', 'evaluate', 'fetch_ohlcv', 'execute',
  'twap', 'vwap', 'factor_research', 'paper_review', 'compare',
]

const STATUS_COLUMNS: TaskStatus[] = ['queued', 'running', 'done', 'failed']

const STATUS_COLORS: Record<TaskStatus, { bg: string; text: string; border: string }> = {
  queued:    { bg: 'rgba(245,166,35,0.08)',  text: '#f5a623', border: 'rgba(245,166,35,0.3)' },
  running:   { bg: 'rgba(33,150,243,0.08)',  text: '#2196F3', border: 'rgba(33,150,243,0.3)' },
  done:      { bg: 'rgba(0,200,83,0.08)',    text: '#00c853', border: 'rgba(0,200,83,0.3)' },
  failed:    { bg: 'rgba(255,23,68,0.08)',   text: '#ff1744', border: 'rgba(255,23,68,0.3)' },
  cancelled: { bg: 'rgba(136,136,136,0.08)', text: '#888',    border: 'rgba(136,136,136,0.3)' },
}

const PRIORITY_COLORS: Record<TaskPriority, string> = {
  low: '#555',
  medium: '#888',
  high: '#f5a623',
  urgent: '#ff1744',
}

const EMPLOYEE_ICONS: Record<string, string> = {
  strategy_agent: '📈',
  ml_agent: '🧠',
  risk_agent: '🛡️',
  data_agent: '📊',
  execution_agent: '⚡',
  research_agent: '🔬',
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtTs(ts: string) {
  const d = new Date(ts)
  const now = Date.now()
  const diffMs = now - d.getTime()
  if (diffMs < 60_000) return 'just now'
  if (diffMs < 3_600_000) return `${Math.floor(diffMs / 60_000)}m ago`
  if (diffMs < 86_400_000) return `${Math.floor(diffMs / 3_600_000)}h ago`
  return d.toLocaleDateString()
}

// ── Employee Card ─────────────────────────────────────────────────────────────

function EmployeeCard({
  id,
  employee,
  tasks,
  onSelect,
  selected,
}: {
  id: string
  employee: Employee
  tasks: Task[]
  onSelect: (id: string) => void
  selected: boolean
}) {
  const activeTasks = tasks.filter(t => t.assigned_to === id && (t.status === 'queued' || t.status === 'running')).length

  return (
    <button
      onClick={() => onSelect(id)}
      className="w-full text-left p-3 rounded-xl border transition-all duration-200"
      style={{
        background: selected ? 'rgba(245,166,35,0.08)' : '#111111',
        border: `1px solid ${selected ? 'rgba(245,166,35,0.4)' : '#1e1e1e'}`,
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">{EMPLOYEE_ICONS[id] ?? '🤖'}</span>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-bold text-[#e8e8e8] truncate">{employee.display}</p>
          {activeTasks > 0 && (
            <span className="text-[10px] font-mono text-[#2196F3]">{activeTasks} active</span>
          )}
        </div>
      </div>
      <div className="flex flex-wrap gap-1">
        {employee.skills.slice(0, 3).map(skill => (
          <span key={skill} className="text-[9px] px-1.5 py-0.5 rounded bg-[#1e1e1e] text-[#555] font-mono">
            {skill}
          </span>
        ))}
        {employee.skills.length > 3 && (
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#1e1e1e] text-[#555]">+{employee.skills.length - 3}</span>
        )}
      </div>
    </button>
  )
}

// ── Task Card ─────────────────────────────────────────────────────────────────

function TaskCard({ task, employees }: { task: Task; employees: EmployeeMap | null }) {
  const [expanded, setExpanded] = useState(false)
  const sc = STATUS_COLORS[task.status]
  const priorityColor = PRIORITY_COLORS[task.priority]
  const empName = task.assigned_to ? (employees?.[task.assigned_to]?.display ?? task.assigned_to) : 'Unassigned'

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-3 hover:border-[#2a2a2a] transition-colors">
      <div className="flex items-start gap-2 mb-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded"
              style={{ background: 'rgba(245,166,35,0.1)', color: '#f5a623', border: '1px solid rgba(245,166,35,0.2)' }}>
              {task.task_type}
            </span>
            <span className="text-[10px] font-bold" style={{ color: priorityColor }}>
              {task.priority.toUpperCase()}
            </span>
          </div>
          <p className="text-xs font-semibold text-[#e8e8e8] truncate">{task.title}</p>
        </div>
        <button onClick={() => setExpanded(e => !e)} className="text-[#555] hover:text-[#888] transition-colors flex-shrink-0 mt-0.5">
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[#555]">
          {EMPLOYEE_ICONS[task.assigned_to ?? ''] ?? '🤖'} {empName}
        </span>
        <span className="text-[10px] text-[#444] font-mono">{fmtTs(task.created_at)}</span>
      </div>

      {task.status === 'running' && task.progress_pct > 0 && (
        <div className="mt-2">
          <div className="h-1 bg-[#1e1e1e] rounded-full overflow-hidden">
            <div className="h-full bg-[#2196F3] rounded-full transition-all duration-500"
              style={{ width: `${task.progress_pct}%` }} />
          </div>
          <span className="text-[9px] text-[#555] font-mono">{task.progress_pct.toFixed(0)}%</span>
        </div>
      )}

      {expanded && (
        <div className="mt-3 pt-3 border-t border-[#1e1e1e] space-y-2">
          {task.description && (
            <p className="text-[10px] text-[#888]">{task.description}</p>
          )}
          {Object.keys(task.params ?? {}).length > 0 && (
            <div>
              <p className="text-[9px] text-[#555] uppercase tracking-wider mb-1">Params</p>
              <pre className="text-[9px] font-mono text-[#888] bg-[#0d0d0d] rounded p-2 overflow-auto max-h-24">
                {JSON.stringify(task.params, null, 2)}
              </pre>
            </div>
          )}
          {task.result && (
            <div>
              <p className="text-[9px] text-[#555] uppercase tracking-wider mb-1">Result</p>
              <pre className="text-[9px] font-mono text-[#00c853] bg-[#0d0d0d] rounded p-2 overflow-auto max-h-24">
                {JSON.stringify(task.result, null, 2)}
              </pre>
            </div>
          )}
          {task.error_message && (
            <div>
              <p className="text-[9px] text-[#555] uppercase tracking-wider mb-1">Error</p>
              <p className="text-[10px] text-[#ff1744] font-mono bg-[#0d0d0d] rounded p-2">{task.error_message}</p>
            </div>
          )}
          {task.started_at && (
            <p className="text-[9px] text-[#555]">Started: {fmtTs(task.started_at)}</p>
          )}
          {task.completed_at && (
            <p className="text-[9px] text-[#555]">Completed: {fmtTs(task.completed_at)}</p>
          )}
        </div>
      )}
    </div>
  )
}

// ── New Task Modal ────────────────────────────────────────────────────────────

function NewTaskModal({
  employees,
  onClose,
  onCreated,
}: {
  employees: EmployeeMap | null
  onClose: () => void
  onCreated: () => void
}) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [taskType, setTaskType] = useState(TASK_TYPES[0])
  const [assignedTo, setAssignedTo] = useState('')
  const [priority, setPriority] = useState<TaskPriority>('medium')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!title.trim()) { setError('Title is required'); return }
    setSubmitting(true)
    setError(null)
    try {
      await api.post('/tasks/', {
        title: title.trim(),
        description: description.trim() || null,
        task_type: taskType,
        assigned_to: assignedTo || null,
        priority,
        params: {},
      })
      onCreated()
      onClose()
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string }
      setError(err.response?.data?.detail ?? err.message ?? 'Failed to create task')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-[#111111] border border-[#1e1e1e] rounded-2xl w-full max-w-lg shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#1e1e1e]">
          <h2 className="text-sm font-bold text-[#e8e8e8]">Dispatch New Task</h2>
          <button onClick={onClose} className="text-[#555] hover:text-[#888] transition-colors">
            <X size={16} />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {error && (
            <div className="text-xs text-[#ff1744] bg-[#ff1744]/10 border border-[#ff1744]/30 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <div>
            <label className="text-[10px] text-[#555] uppercase tracking-wider mb-1 block">Title *</label>
            <input
              value={title}
              onChange={e => setTitle(e.target.value)}
              className="w-full bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg px-3 py-2 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623]/50 font-mono"
              placeholder="e.g. Retrain LSTM model on BTC/USD"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-[#555] uppercase tracking-wider mb-1 block">Task Type</label>
              <select
                value={taskType}
                onChange={e => setTaskType(e.target.value)}
                className="w-full bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg px-3 py-2 text-xs text-[#e8e8e8] focus:outline-none focus:border-[#f5a623]/50"
              >
                {TASK_TYPES.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-[10px] text-[#555] uppercase tracking-wider mb-1 block">Priority</label>
              <select
                value={priority}
                onChange={e => setPriority(e.target.value as TaskPriority)}
                className="w-full bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-[#f5a623]/50"
                style={{ color: PRIORITY_COLORS[priority] }}
              >
                {(['low', 'medium', 'high', 'urgent'] as TaskPriority[]).map(p => (
                  <option key={p} value={p} style={{ color: PRIORITY_COLORS[p] }}>
                    {p.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="text-[10px] text-[#555] uppercase tracking-wider mb-1 block">Assign to Employee</label>
            <div className="grid grid-cols-2 gap-2">
              {employees ? Object.entries(employees).map(([id, emp]) => (
                <button
                  key={id}
                  onClick={() => setAssignedTo(assignedTo === id ? '' : id)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg border transition-all text-left"
                  style={{
                    background: assignedTo === id ? 'rgba(245,166,35,0.08)' : '#0d0d0d',
                    border: `1px solid ${assignedTo === id ? 'rgba(245,166,35,0.4)' : '#1e1e1e'}`,
                  }}
                >
                  <span>{EMPLOYEE_ICONS[id] ?? '🤖'}</span>
                  <div>
                    <p className="text-[10px] font-semibold text-[#e8e8e8]">{emp.display}</p>
                    <p className="text-[9px] text-[#555]">{emp.skills.slice(0, 2).join(', ')}</p>
                  </div>
                </button>
              )) : (
                <p className="text-xs text-[#555]">Loading employees...</p>
              )}
            </div>
          </div>

          <div>
            <label className="text-[10px] text-[#555] uppercase tracking-wider mb-1 block">Description</label>
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              rows={3}
              className="w-full bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg px-3 py-2 text-xs text-[#888] focus:outline-none focus:border-[#f5a623]/50 resize-none"
              placeholder="Optional: describe what this task should accomplish..."
            />
          </div>
        </div>

        <div className="flex gap-3 px-5 pb-5">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-lg bg-[#1e1e1e] text-xs text-[#888] hover:bg-[#2a2a2a] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting || !title.trim()}
            className="flex-1 py-2 rounded-lg text-xs font-bold transition-all"
            style={{
              background: (!submitting && title.trim()) ? 'linear-gradient(135deg, #f5a623, #e0921a)' : '#2a2a2a',
              color: (!submitting && title.trim()) ? '#000' : '#555',
              cursor: (!submitting && title.trim()) ? 'pointer' : 'not-allowed',
            }}
          >
            {submitting ? 'Dispatching...' : 'Dispatch Task'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function TaskManager() {
  const [showModal, setShowModal] = useState(false)
  const [selectedEmployee, setSelectedEmployee] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const { data: tasks, isLoading: tasksLoading } = useQuery<Task[]>({
    queryKey: ['tasks'],
    queryFn: () => api.get('/tasks/?limit=200').then(r => r.data),
    refetchInterval: 10_000,
  })

  const { data: employees } = useQuery<EmployeeMap>({
    queryKey: ['task-employees'],
    queryFn: () => api.get('/tasks/employees').then(r => r.data),
    staleTime: 300_000,
  })

  const filteredTasks = (tasks ?? []).filter(t =>
    !selectedEmployee || t.assigned_to === selectedEmployee
  )

  return (
    <div className="space-y-5">
      {showModal && (
        <NewTaskModal
          employees={employees ?? null}
          onClose={() => setShowModal(false)}
          onCreated={() => queryClient.invalidateQueries({ queryKey: ['tasks'] })}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-[#e8e8e8]">Task Manager</h1>
          <p className="text-xs text-[#555] mt-0.5">Employee dispatch & automation board</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold text-black transition-all hover:opacity-90 active:scale-95"
          style={{ background: 'linear-gradient(135deg, #f5a623, #e0921a)' }}
        >
          <Plus size={14} />
          New Task
        </button>
      </div>

      <div className="flex gap-5">
        {/* Left panel — Employee Roster */}
        <div className="w-56 flex-shrink-0 space-y-2">
          <p className="text-[10px] text-[#555] uppercase tracking-wider px-1">Employees</p>
          <button
            onClick={() => setSelectedEmployee(null)}
            className="w-full text-left px-3 py-2 rounded-xl border transition-all text-xs font-semibold"
            style={{
              background: !selectedEmployee ? 'rgba(245,166,35,0.08)' : '#111111',
              border: `1px solid ${!selectedEmployee ? 'rgba(245,166,35,0.4)' : '#1e1e1e'}`,
              color: !selectedEmployee ? '#f5a623' : '#888',
            }}
          >
            All Employees
          </button>
          {employees ? (
            Object.entries(employees).map(([id, emp]) => (
              <EmployeeCard
                key={id}
                id={id}
                employee={emp}
                tasks={tasks ?? []}
                selected={selectedEmployee === id}
                onSelect={id => setSelectedEmployee(prev => prev === id ? null : id)}
              />
            ))
          ) : (
            <div className="space-y-2">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="h-16 bg-[#111111] border border-[#1e1e1e] rounded-xl animate-pulse" />
              ))}
            </div>
          )}
        </div>

        {/* Right panel — Kanban board */}
        <div className="flex-1 grid grid-cols-4 gap-3">
          {STATUS_COLUMNS.map(status => {
            const colTasks = filteredTasks.filter(t => t.status === status)
            const sc = STATUS_COLORS[status]
            return (
              <div key={status} className="flex flex-col gap-2">
                {/* Column header */}
                <div className="flex items-center justify-between px-2 py-1.5 rounded-lg border"
                  style={{ background: sc.bg, border: `1px solid ${sc.border}` }}>
                  <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: sc.text }}>
                    {status}
                  </span>
                  <span className="text-[10px] font-mono" style={{ color: sc.text }}>
                    {colTasks.length}
                  </span>
                </div>

                {/* Task cards */}
                <div className="space-y-2 min-h-32">
                  {tasksLoading ? (
                    <div className="h-16 bg-[#111111] border border-[#1e1e1e] rounded-xl animate-pulse" />
                  ) : colTasks.length === 0 ? (
                    <div className="h-16 flex items-center justify-center border border-dashed border-[#1e1e1e] rounded-xl">
                      <span className="text-[10px] text-[#333]">Empty</span>
                    </div>
                  ) : (
                    colTasks.map(task => (
                      <TaskCard key={task.id} task={task} employees={employees ?? null} />
                    ))
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
