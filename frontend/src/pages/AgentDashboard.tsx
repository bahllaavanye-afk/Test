import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ──────────────────────────────────────────────────────────────────

interface AgentRoster {
  name: string
  role: string
  runs: number
  successes: number
  last_success: string | null
  last_summary: string
  last_seen: string | null
  is_online: boolean
}

interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
  agent?: string
  ts?: string
}

interface Task {
  agent: string
  description: string
  priority: string
  claimed_at: string
  created_by?: string
}

interface Memory {
  active_agents: Record<string, { last_seen: string; role: string }>
  improvement_stats: Record<string, { runs: number; successes: number; last_summary?: string }>
  failure_traces: { agent: string; what_failed: string; error: string; timestamp: string }[]
  peer_learnings: string[]
  platform_metrics: Record<string, unknown>
  last_updated: string | null
}

// ── Helpers ────────────────────────────────────────────────────────────────

const AGENT_EMOJI: Record<string, string> = {
  // Automation agents
  continuous_improver: '🔧', signal_runner: '📡', quick_backtest: '⚡',
  peer_reviewer: '👁️', frontend_design: '🎨', token_monitor: '🪙',
  strategy_generator: '🧠', free_agent_engineer: '🤖', desk_trader: '📊',
  system_watchdog: '🛡️', ml_trainer: '🏋️', standup_agent: '📢',
  investor_pipeline: '💼', run_experiments: '🔬',
  algo_agent: '🎲', self_improver: '♻️', research_scientist: '🔭',
  modeling_engineer: '⚙️',
  // QuantEdge employees
  vp_eng: '🏗️', alpha_dir: '📈', ml_lead: '🧬', risk_eng: '🛡️',
  backend_lead: '⚙️', qa_dir: '✅', devops_dir: '🚀', exec_eng: '⚡',
  poly_desk: '🎯', ml_researcher: '🔬', vp_research: '🔭',
  quant_researcher: '📊', cro: '⚖️', frontend_lead: '🎨',
  equity_lead: '📈', fixed_income_desk: '💵', macro_researcher: '🌐',
  stat_arb_desk: '⚖️', vol_trader: '🌊', momentum_quant: '🚀',
  alt_data_lead: '🛰️', model_validator: '🔍', feature_engineer: '🧮',
  crypto_quant: '₿', derivatives_desk: '📐', arb_trader: '🔄',
  portfolio_manager: '💼', market_maker: '📋', regime_analyst: '🔮',
  backtest_engineer: '📉', data_engineer_2: '🗄️', infra_lead: '☁️',
}

// Employee domain and LLM assignment (mirrors deep_code_review.py AGENTS)
const EMPLOYEE_DOMAIN: Record<string, { domain: string; llm: string; channel: string }> = {
  alpha_dir:    { domain: 'strategies',      llm: 'gemini',     channel: '#desk-research' },
  ml_lead:      { domain: 'ml-models',       llm: 'sambanova',  channel: '#ml-research' },
  exec_eng:     { domain: 'execution',       llm: 'cerebras',   channel: '#desk-equities' },
  risk_eng:     { domain: 'risk',            llm: 'groq',       channel: '#risk' },
  backend_lead: { domain: 'api-backend',     llm: 'deepseek',   channel: '#engineering' },
  devops_dir:   { domain: 'tasks-scheduler', llm: 'together',   channel: '#engineering' },
  frontend_lead:{ domain: 'frontend',        llm: 'hyperbolic', channel: '#frontend' },
  vp_eng:       { domain: 'infrastructure',  llm: 'nvidia_nim', channel: '#engineering' },
  cro:          { domain: 'synthesis',       llm: 'gemini',     channel: '#engineering' },
}

function timeAgo(iso: string | null): string {
  if (!iso) return 'never'
  const diff = Date.now() - new Date(iso).getTime()
  const m = Math.floor(diff / 60000)
  if (m < 2) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

function statusColor(agent: AgentRoster): string {
  if (!agent.last_seen) return '#444'
  const diff = Date.now() - new Date(agent.last_seen).getTime()
  if (diff < 10 * 60 * 1000) return '#00c853'  // < 10min = green
  if (diff < 2 * 60 * 60 * 1000) return '#f5a623'  // < 2h = amber
  return '#ff1744'  // stale = red
}

// ── Sub-components ─────────────────────────────────────────────────────────

function AgentCard({
  agent, selected, onClick,
}: { agent: AgentRoster; selected: boolean; onClick: () => void }) {
  const color = statusColor(agent)
  const sr = agent.runs > 0 ? Math.round((agent.successes / agent.runs) * 100) : 0
  return (
    <button
      onClick={onClick}
      style={{
        width: '100%', textAlign: 'left', padding: '10px 12px',
        background: selected ? '#1a2a1a' : '#0f0f0f',
        border: `1px solid ${selected ? '#00c853' : '#1e1e1e'}`,
        borderRadius: 8, cursor: 'pointer', transition: 'all 0.15s',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 16 }}>{AGENT_EMOJI[agent.name] ?? '🤖'}</span>
        <span style={{ color: '#e8e8e8', fontSize: 12, fontWeight: 600, fontFamily: 'JetBrains Mono, monospace' }}>
          {agent.name}
        </span>
        <span style={{
          marginLeft: 'auto', width: 8, height: 8, borderRadius: '50%',
          background: color, boxShadow: `0 0 6px ${color}`,
        }} />
      </div>
      <div style={{ color: '#888', fontSize: 11, marginBottom: 6, lineHeight: 1.4 }}>
        {agent.role.slice(0, 60)}{agent.role.length > 60 ? '…' : ''}
      </div>
      <div style={{ display: 'flex', gap: 12, fontSize: 10, color: '#666' }}>
        <span>runs: <span style={{ color: '#aaa' }}>{agent.runs}</span></span>
        <span>sr: <span style={{ color: sr > 70 ? '#00c853' : sr > 40 ? '#f5a623' : '#ff1744' }}>{sr}%</span></span>
        <span style={{ marginLeft: 'auto' }}>{timeAgo(agent.last_seen)}</span>
      </div>
    </button>
  )
}

function ChatPanel({ agentName }: { agentName: string }) {
  const [msgs, setMsgs] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setMsgs([{
      role: 'assistant',
      content: `Hi, I'm **${agentName}**. I'm the ${AGENT_EMOJI[agentName] ?? '🤖'} agent on QuantEdge. Ask me anything about my domain, the platform architecture, strategies, or assign me a task.`,
      agent: agentName,
    }])
  }, [agentName])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [msgs])

  async function send() {
    if (!input.trim() || sending) return
    const userMsg = input.trim()
    setInput('')
    setMsgs(m => [...m, { role: 'user', content: userMsg }])
    setSending(true)
    try {
      const history = msgs.map(m => ({ role: m.role, content: m.content }))
      const res = await api.post('/agents/chat', {
        agent: agentName, message: userMsg, history,
      })
      setMsgs(m => [...m, {
        role: 'assistant', content: res.data.reply,
        agent: agentName, ts: res.data.timestamp,
      }])
    } catch {
      setMsgs(m => [...m, {
        role: 'assistant',
        content: '⚠️ LLM unavailable — set GROQ_API_KEY_1 or GEMINI_API_KEY_1 in backend env.',
        agent: agentName,
      }])
    } finally {
      setSending(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px', borderBottom: '1px solid #1e1e1e',
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <span style={{ fontSize: 20 }}>{AGENT_EMOJI[agentName] ?? '🤖'}</span>
        <div>
          <div style={{ color: '#e8e8e8', fontWeight: 600, fontSize: 13, fontFamily: 'JetBrains Mono, monospace' }}>
            {agentName}
          </div>
          <div style={{ color: '#666', fontSize: 11 }}>Powered by Groq / DeepSeek / Gemini (free)</div>
        </div>
        <div style={{
          marginLeft: 'auto', fontSize: 10, color: '#666',
          background: '#1a1a1a', padding: '2px 8px', borderRadius: 12,
          border: '1px solid #2a2a2a',
        }}>
          Works offline from Claude
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {msgs.map((m, i) => (
          <div key={i} style={{
            display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start',
          }}>
            <div style={{
              maxWidth: '80%', padding: '10px 14px', borderRadius: 12,
              background: m.role === 'user' ? '#1a2a3a' : '#111',
              border: `1px solid ${m.role === 'user' ? '#2a3a4a' : '#1e1e1e'}`,
              color: '#e8e8e8', fontSize: 13, lineHeight: 1.6,
              fontFamily: m.role === 'assistant' ? 'JetBrains Mono, monospace' : 'inherit',
            }}>
              {m.role === 'assistant' && (
                <div style={{ color: '#f5a623', fontSize: 11, marginBottom: 6, fontWeight: 600 }}>
                  {AGENT_EMOJI[agentName]} {agentName}
                </div>
              )}
              <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
            </div>
          </div>
        ))}
        {sending && (
          <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
            <div style={{
              padding: '10px 14px', borderRadius: 12, background: '#111',
              border: '1px solid #1e1e1e', color: '#666', fontSize: 13,
              fontFamily: 'JetBrains Mono, monospace',
            }}>
              <span style={{ animation: 'pulse 1s infinite' }}>thinking…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: '12px 16px', borderTop: '1px solid #1e1e1e',
        display: 'flex', gap: 10,
      }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder={`Ask ${agentName} anything, or assign a task…`}
          style={{
            flex: 1, background: '#111', border: '1px solid #2a2a2a',
            borderRadius: 8, padding: '10px 14px', color: '#e8e8e8', fontSize: 13,
            outline: 'none', fontFamily: 'JetBrains Mono, monospace',
          }}
        />
        <button
          onClick={send}
          disabled={sending || !input.trim()}
          style={{
            padding: '10px 20px', background: '#00c853', color: '#000',
            border: 'none', borderRadius: 8, cursor: 'pointer',
            fontWeight: 700, fontSize: 13, opacity: sending || !input.trim() ? 0.4 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  )
}

function TaskPanel({
  roster, tasks, onAssign, onDelete,
}: {
  roster: AgentRoster[]
  tasks: Record<string, Task>
  onAssign: (t: { task_id: string; description: string; assigned_to: string; priority: string }) => void
  onDelete: (id: string) => void
}) {
  const [tid, setTid] = useState('')
  const [desc, setDesc] = useState('')
  const [agent, setAgent] = useState(roster[0]?.name ?? '')
  const [priority, setPriority] = useState('normal')

  function submit() {
    if (!tid.trim() || !desc.trim()) return
    onAssign({ task_id: tid.trim(), description: desc.trim(), assigned_to: agent, priority })
    setTid(''); setDesc('')
  }

  const inputStyle = {
    background: '#111', border: '1px solid #2a2a2a', borderRadius: 6,
    padding: '8px 12px', color: '#e8e8e8', fontSize: 12,
    fontFamily: 'JetBrains Mono, monospace', width: '100%',
  }

  return (
    <div style={{ padding: 16 }}>
      <div style={{ color: '#888', fontSize: 11, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
        Assign Task
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
        <input
          placeholder="task-id (e.g. fix-risk-engine)"
          value={tid}
          onChange={e => setTid(e.target.value)}
          style={inputStyle}
        />
        <textarea
          placeholder="Task description…"
          value={desc}
          onChange={e => setDesc(e.target.value)}
          rows={3}
          style={{ ...inputStyle, resize: 'vertical' }}
        />
        <div style={{ display: 'flex', gap: 8 }}>
          <select
            value={agent}
            onChange={e => setAgent(e.target.value)}
            style={{ ...inputStyle, flex: 2 }}
          >
            {roster.map(a => (
              <option key={a.name} value={a.name}>
                {AGENT_EMOJI[a.name] ?? '🤖'} {a.name}
              </option>
            ))}
          </select>
          <select
            value={priority}
            onChange={e => setPriority(e.target.value)}
            style={{ ...inputStyle, flex: 1 }}
          >
            <option value="low">Low</option>
            <option value="normal">Normal</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
        <button
          onClick={submit}
          style={{
            background: '#f5a623', color: '#000', border: 'none',
            borderRadius: 6, padding: '10px', fontWeight: 700, cursor: 'pointer', fontSize: 12,
          }}
        >
          Assign Task →
        </button>
      </div>

      {/* Active tasks */}
      <div style={{ color: '#888', fontSize: 11, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
        Active Tasks ({Object.keys(tasks).length})
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 300, overflowY: 'auto' }}>
        {Object.entries(tasks).map(([id, task]) => (
          <div key={id} style={{
            background: '#111', border: '1px solid #1e1e1e', borderRadius: 6,
            padding: '10px 12px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ color: '#f5a623', fontSize: 11, fontFamily: 'JetBrains Mono, monospace', fontWeight: 600 }}>
                {id}
              </span>
              <span style={{
                fontSize: 10, padding: '1px 6px', borderRadius: 10,
                background: task.priority === 'critical' ? '#ff1744' : task.priority === 'high' ? '#f5a623' : '#1e1e1e',
                color: task.priority === 'critical' || task.priority === 'high' ? '#000' : '#888',
              }}>
                {task.priority}
              </span>
              <button
                onClick={() => onDelete(id)}
                style={{
                  marginLeft: 'auto', background: 'none', border: '1px solid #2a2a2a',
                  color: '#666', borderRadius: 4, padding: '2px 6px', cursor: 'pointer', fontSize: 10,
                }}
              >
                ✕
              </button>
            </div>
            <div style={{ color: '#aaa', fontSize: 11, marginBottom: 4 }}>{task.description}</div>
            <div style={{ color: '#555', fontSize: 10 }}>
              {AGENT_EMOJI[task.agent] ?? '🤖'} {task.agent} · {timeAgo(task.claimed_at)}
            </div>
          </div>
        ))}
        {Object.keys(tasks).length === 0 && (
          <div style={{ color: '#444', fontSize: 12, textAlign: 'center', padding: '20px 0' }}>
            No active tasks
          </div>
        )}
      </div>
    </div>
  )
}

interface LLMStatus {
  configured_providers: string[]
  autonomous_llm_active: boolean
  egress_hosts_required: string[]
  active_egress_hosts: string[]
  note: string
}

function LLMStatusPanel() {
  const { data, isLoading, error } = useQuery<LLMStatus>({
    queryKey: ['agents', 'llm-status'],
    queryFn: () => api.get('/agents/llm-status').then(r => r.data),
    refetchInterval: 60_000,
  })

  const PROVIDER_INFO: Record<string, { label: string; host: string; link: string; color: string }> = {
    groq:     { label: 'Groq', host: 'api.groq.com',                           link: 'console.groq.com',           color: '#f97316' },
    deepseek: { label: 'DeepSeek', host: 'api.deepseek.com',                   link: 'platform.deepseek.com',      color: '#3b82f6' },
    gemini:   { label: 'Gemini', host: 'generativelanguage.googleapis.com',    link: 'aistudio.google.com',        color: '#4285f4' },
  }

  const ALL_PROVIDERS = ['groq', 'deepseek', 'gemini']

  if (isLoading) return (
    <div style={{ padding: 24, color: '#555', fontSize: 12, textAlign: 'center' }}>Loading LLM status…</div>
  )
  if (error) return (
    <div style={{ padding: 24, color: '#ff1744', fontSize: 12 }}>Failed to load LLM status</div>
  )

  const configured = data?.configured_providers ?? []
  const active = data?.autonomous_llm_active ?? false

  return (
    <div style={{ height: '100%', overflowY: 'auto', padding: 16 }}>
      {/* Status banner */}
      <div style={{
        padding: '10px 14px', borderRadius: 8, marginBottom: 16,
        background: active ? '#0a1a0a' : '#1a0a0a',
        border: `1px solid ${active ? '#00c853' : '#ff1744'}`,
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <div style={{
          width: 10, height: 10, borderRadius: '50%',
          background: active ? '#00c853' : '#ff1744',
          boxShadow: `0 0 8px ${active ? '#00c853' : '#ff1744'}`,
        }} />
        <div>
          <div style={{ color: active ? '#00c853' : '#ff1744', fontWeight: 700, fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}>
            {active ? 'AUTONOMOUS LLM ACTIVE' : 'RULE ENGINE MODE (no API key)'}
          </div>
          <div style={{ color: '#666', fontSize: 10, marginTop: 2 }}>
            {active
              ? `${configured.length} provider(s) configured — LLM reasoning layer is live`
              : 'Agents run rule-based analysis — set a free API key for LLM depth'}
          </div>
        </div>
      </div>

      {/* Provider grid */}
      <div style={{ color: '#555', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
        Free LLM Providers (priority order)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 20 }}>
        {ALL_PROVIDERS.map((p, idx) => {
          const info = PROVIDER_INFO[p]
          const isOn = configured.includes(p)
          return (
            <div key={p} style={{
              padding: '10px 12px', borderRadius: 6,
              background: isOn ? '#0a160a' : '#111',
              border: `1px solid ${isOn ? '#1a3a1a' : '#1e1e1e'}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{
                  fontSize: 10, fontWeight: 700,
                  color: '#444', width: 14, textAlign: 'right',
                  fontFamily: 'JetBrains Mono, monospace',
                }}>{idx + 1}</span>
                <span style={{ color: info.color, fontWeight: 700, fontSize: 12 }}>{info.label}</span>
                <span style={{
                  marginLeft: 'auto',
                  fontSize: 10, padding: '2px 8px', borderRadius: 10,
                  background: isOn ? '#00c853' : '#1e1e1e',
                  color: isOn ? '#000' : '#555',
                  fontWeight: isOn ? 700 : 400,
                }}>
                  {isOn ? '✓ configured' : 'not set'}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 8, marginTop: 6, fontSize: 10, color: '#444', paddingLeft: 22 }}>
                <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>{info.host}</span>
                <span>·</span>
                <span style={{ color: '#555' }}>{info.link}</span>
              </div>
              {!isOn && (
                <div style={{ paddingLeft: 22, marginTop: 4, fontSize: 10, color: '#555', fontFamily: 'JetBrains Mono, monospace' }}>
                  Set {p.toUpperCase()}_API_KEY in Render → Environment
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Egress note */}
      <div style={{ color: '#555', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
        Required Egress Hosts
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 20 }}>
        {(data?.egress_hosts_required ?? ['api.groq.com', 'api.deepseek.com', 'generativelanguage.googleapis.com']).map(h => (
          <div key={h} style={{
            padding: '6px 10px', borderRadius: 4,
            background: (data?.active_egress_hosts ?? []).includes(h) ? '#0a160a' : '#111',
            border: '1px solid #1a1a1a',
            fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
            color: (data?.active_egress_hosts ?? []).includes(h) ? '#69f0ae' : '#555',
          }}>
            {h}
          </div>
        ))}
        <div style={{ color: '#444', fontSize: 10, marginTop: 4 }}>
          Add these to Render → Settings → Outbound Networking allowlist
        </div>
      </div>

      {/* How it works */}
      <div style={{ color: '#555', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
        How It Works
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11, color: '#666' }}>
        {[
          { step: '1', label: 'Rule Engine', desc: 'Always runs — zero config needed', color: '#00c853' },
          { step: '2', label: 'Free LLM', desc: 'Enhances with language model depth (Groq → DeepSeek → Gemini)', color: '#f5a623' },
          { step: '3', label: 'Agent Bus', desc: 'Broadcasts findings to all desks via Redis Streams', color: '#2196F3' },
          { step: '4', label: 'Slack', desc: 'Posts standups + alerts when SLACK_BOT_TOKEN is set', color: '#9c27b0' },
        ].map(({ step, label, desc, color }) => (
          <div key={step} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
            <span style={{
              width: 18, height: 18, borderRadius: '50%', background: '#1a1a1a',
              color, fontSize: 10, fontWeight: 700, display: 'flex', alignItems: 'center',
              justifyContent: 'center', flexShrink: 0, fontFamily: 'JetBrains Mono, monospace',
            }}>{step}</span>
            <div>
              <span style={{ color: '#aaa', fontWeight: 600 }}>{label}</span>
              <span style={{ color: '#555' }}> — {desc}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

interface CodeReview {
  domain: string
  employee: string
  provider: string
  grade: string
  date: string
  top_priority: string
}

function CodeReviewPanel() {
  const { data } = useQuery<{ reviews: CodeReview[] }>({
    queryKey: ['agents', 'code-reviews'],
    queryFn: () => api.get('/agents/code-reviews').then(r => r.data),
    refetchInterval: 300_000,
  })

  const reviews = data?.reviews ?? []
  const gradeColor = (g: string) => {
    if (!g || g === '?') return '#555'
    if (g.startsWith('A')) return '#00c853'
    if (g.startsWith('B')) return '#69f0ae'
    if (g.startsWith('C')) return '#f5a623'
    if (g.startsWith('D')) return '#ff6d00'
    return '#ff1744'
  }

  return (
    <div style={{ height: '100%', overflowY: 'auto', padding: 16 }}>
      <div style={{ color: '#888', fontSize: 11, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
        Employee Code Reviews · {reviews.length > 0 ? `Last: ${reviews[0]?.date ?? '—'}` : 'No reviews yet'}
      </div>
      {Object.entries(EMPLOYEE_DOMAIN).filter(([k]) => k !== 'cro').map(([emp, info]) => {
        const review = reviews.find(r => r.employee === emp || r.domain === info.domain)
        return (
          <div key={emp} style={{
            padding: '10px 12px', marginBottom: 8, background: '#111',
            borderRadius: 6, border: '1px solid #1e1e1e',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ fontSize: 14 }}>{AGENT_EMOJI[emp] ?? '🤖'}</span>
              <span style={{ color: '#e8e8e8', fontSize: 11, fontWeight: 600, fontFamily: 'JetBrains Mono, monospace' }}>
                {emp}
              </span>
              <span style={{ color: '#555', fontSize: 10 }}>→ {info.domain}</span>
              <span style={{
                marginLeft: 'auto',
                fontSize: 14, fontWeight: 800,
                color: gradeColor(review?.grade ?? ''),
                fontFamily: 'JetBrains Mono, monospace',
              }}>
                {review?.grade ?? '—'}
              </span>
            </div>
            <div style={{ display: 'flex', gap: 8, fontSize: 10, color: '#555' }}>
              <span>via <span style={{ color: '#f5a623' }}>{info.llm}</span></span>
              <span style={{ color: info.channel.includes('research') ? '#9c27b0' : '#2196F3' }}>{info.channel}</span>
            </div>
            {review?.top_priority && (
              <div style={{ marginTop: 6, color: '#888', fontSize: 10, lineHeight: 1.4 }}>
                {review.top_priority.slice(0, 90)}{review.top_priority.length > 90 ? '…' : ''}
              </div>
            )}
          </div>
        )
      })}
      {reviews.length === 0 && (
        <div style={{ color: '#444', fontSize: 12, textAlign: 'center', padding: '30px 0' }}>
          Run deep-code-review workflow to see employee grades
        </div>
      )}
    </div>
  )
}

function MemoryPanel({ memory, skills }: { memory: Memory | null; skills: string[] }) {
  const [tab, setTab] = useState<'skills' | 'failures' | 'learnings' | 'metrics'>('skills')

  const tabStyle = (t: string) => ({
    padding: '6px 12px', fontSize: 11, cursor: 'pointer',
    background: tab === t ? '#1a1a1a' : 'transparent',
    color: tab === t ? '#f5a623' : '#666',
    border: 'none', borderBottom: `2px solid ${tab === t ? '#f5a623' : 'transparent'}`,
    fontFamily: 'JetBrains Mono, monospace',
  })

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        display: 'flex', borderBottom: '1px solid #1e1e1e',
        padding: '0 12px',
      }}>
        {(['skills', 'learnings', 'failures', 'metrics'] as const).map(t => (
          <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
        {tab === 'skills' && (
          <>
            <div style={{ color: '#888', fontSize: 11, marginBottom: 10 }}>
              Voyager skill library · {skills.length} patterns
            </div>
            {skills.slice().reverse().map((s, i) => (
              <div key={i} style={{
                padding: '6px 10px', marginBottom: 4, background: '#111',
                borderRadius: 4, border: '1px solid #1a1a1a',
                color: '#aaa', fontSize: 11, lineHeight: 1.5,
                fontFamily: 'JetBrains Mono, monospace',
              }}>
                <span style={{ color: '#444', marginRight: 8 }}>{skills.length - i}.</span>
                {s}
              </div>
            ))}
          </>
        )}

        {tab === 'learnings' && (
          <>
            <div style={{ color: '#888', fontSize: 11, marginBottom: 10 }}>
              Peer learnings (Reflexion) · last 20
            </div>
            {(memory?.peer_learnings ?? []).slice().reverse().map((l, i) => (
              <div key={i} style={{
                padding: '6px 10px', marginBottom: 4, background: '#111',
                borderRadius: 4, border: '1px solid #1a1a1a',
                color: '#aaa', fontSize: 11, lineHeight: 1.5,
                fontFamily: 'JetBrains Mono, monospace',
              }}>
                {l}
              </div>
            ))}
          </>
        )}

        {tab === 'failures' && (
          <>
            <div style={{ color: '#888', fontSize: 11, marginBottom: 10 }}>
              Failure traces (last 20) — used to generate new skills
            </div>
            {(memory?.failure_traces ?? []).slice().reverse().map((f, i) => (
              <div key={i} style={{
                padding: '8px 10px', marginBottom: 6, background: '#111',
                borderRadius: 4, border: '1px solid #2a1a1a',
              }}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
                  <span style={{ color: '#ff1744', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}>
                    {AGENT_EMOJI[f.agent] ?? '🤖'} {f.agent}
                  </span>
                  <span style={{ color: '#444', fontSize: 10 }}>{timeAgo(f.timestamp)}</span>
                </div>
                <div style={{ color: '#cc6666', fontSize: 11 }}>{f.what_failed}</div>
                <div style={{ color: '#555', fontSize: 10, marginTop: 2 }}>{f.error}</div>
              </div>
            ))}
          </>
        )}

        {tab === 'metrics' && memory && (
          <>
            <div style={{ color: '#888', fontSize: 11, marginBottom: 10 }}>Platform metrics</div>
            {Object.entries(memory.improvement_stats).map(([agent, s]) => {
              const stat = s as { runs: number; successes: number; last_summary?: string }
              const sr = stat.runs > 0 ? Math.round((stat.successes / stat.runs) * 100) : 0
              return (
                <div key={agent} style={{
                  padding: '8px 10px', marginBottom: 6, background: '#111',
                  borderRadius: 4, border: '1px solid #1e1e1e',
                }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 4, alignItems: 'center' }}>
                    <span style={{ color: '#e8e8e8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}>
                      {AGENT_EMOJI[agent] ?? '🤖'} {agent}
                    </span>
                    <span style={{ marginLeft: 'auto', color: sr > 70 ? '#00c853' : sr > 40 ? '#f5a623' : '#ff1744', fontSize: 11 }}>
                      {sr}% SR
                    </span>
                    <span style={{ color: '#555', fontSize: 10 }}>
                      {stat.runs} runs
                    </span>
                  </div>
                  {stat.last_summary && (
                    <div style={{ color: '#555', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}>
                      {stat.last_summary.slice(0, 100)}
                    </div>
                  )}
                </div>
              )
            })}
          </>
        )}
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function AgentDashboard() {
  const [selectedAgent, setSelectedAgent] = useState('free_agent_engineer')
  const [rightTab, setRightTab] = useState<'memory' | 'tasks' | 'reviews' | 'llm'>('reviews')
  const qc = useQueryClient()

  const { data: roster = [] } = useQuery<AgentRoster[]>({
    queryKey: ['agents', 'roster'],
    queryFn: () => api.get('/agents/roster').then(r => r.data),
    refetchInterval: 30_000,
  })

  const { data: memory } = useQuery<Memory>({
    queryKey: ['agents', 'memory'],
    queryFn: () => api.get('/agents/memory').then(r => r.data),
    refetchInterval: 60_000,
  })

  const { data: skillsData } = useQuery<{ skills: string[]; total: number }>({
    queryKey: ['agents', 'skills'],
    queryFn: () => api.get('/agents/skills').then(r => r.data),
    refetchInterval: 120_000,
  })

  const { data: tasksData } = useQuery<{ active: Record<string, Task> }>({
    queryKey: ['agents', 'tasks'],
    queryFn: () => api.get('/agents/tasks').then(r => r.data),
    refetchInterval: 30_000,
  })

  const assignTask = useMutation({
    mutationFn: (body: { task_id: string; description: string; assigned_to: string; priority: string }) =>
      api.post('/agents/tasks', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agents', 'tasks'] }),
  })

  const deleteTask = useMutation({
    mutationFn: (id: string) => api.delete(`/agents/tasks/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agents', 'tasks'] }),
  })

  // Stats bar
  const totalRuns = roster.reduce((s, a) => s + a.runs, 0)
  const totalSuccess = roster.reduce((s, a) => s + a.successes, 0)
  const onlineCount = roster.filter(a => {
    if (!a.last_seen) return false
    return Date.now() - new Date(a.last_seen).getTime() < 2 * 60 * 60 * 1000
  }).length

  const tabBtnStyle = (active: boolean) => ({
    padding: '6px 16px', fontSize: 11, cursor: 'pointer',
    background: active ? '#1a1a1a' : 'transparent',
    color: active ? '#f5a623' : '#555',
    border: 'none', borderBottom: `2px solid ${active ? '#f5a623' : 'transparent'}`,
    fontFamily: 'JetBrains Mono, monospace',
  })

  return (
    <div style={{
      height: '100%', display: 'flex', flexDirection: 'column',
      background: '#0a0a0a', color: '#e8e8e8',
      fontFamily: 'Inter, sans-serif',
    }}>
      {/* Header */}
      <div style={{
        padding: '14px 20px', borderBottom: '1px solid #1e1e1e',
        display: 'flex', alignItems: 'center', gap: 16,
        background: '#0d0d0d',
      }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: '#e8e8e8' }}>
            🤖 Agent Command Center
          </div>
          <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>
            30+ employees · 8 LLM providers · 24/7 code review · assign tasks · shared memory
          </div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 24 }}>
          {[
            { label: 'Agents', value: `${onlineCount}/${roster.length}`, color: '#00c853' },
            { label: 'Total Runs', value: totalRuns.toLocaleString(), color: '#f5a623' },
            { label: 'Success Rate', value: totalRuns > 0 ? `${Math.round((totalSuccess / totalRuns) * 100)}%` : '—', color: '#2196F3' },
            { label: 'Skills', value: skillsData?.total ?? 0, color: '#9c27b0' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ textAlign: 'right' }}>
              <div style={{ color, fontSize: 18, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace' }}>
                {value}
              </div>
              <div style={{ color: '#555', fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                {label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Left: Agent roster */}
        <div style={{
          width: 260, borderRight: '1px solid #1e1e1e', overflowY: 'auto',
          padding: 12, display: 'flex', flexDirection: 'column', gap: 6,
        }}>
          <div style={{ color: '#555', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            Agents ({roster.length})
          </div>
          {roster.map(a => (
            <AgentCard
              key={a.name}
              agent={a}
              selected={selectedAgent === a.name}
              onClick={() => setSelectedAgent(a.name)}
            />
          ))}
        </div>

        {/* Center: Chat */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: '1px solid #1e1e1e' }}>
          <ChatPanel key={selectedAgent} agentName={selectedAgent} />
        </div>

        {/* Right: Memory + Tasks */}
        <div style={{ width: 340, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', borderBottom: '1px solid #1e1e1e', flexWrap: 'wrap' }}>
            <button style={tabBtnStyle(rightTab === 'llm')} onClick={() => setRightTab('llm')}>
              LLM
            </button>
            <button style={tabBtnStyle(rightTab === 'reviews')} onClick={() => setRightTab('reviews')}>
              Reviews
            </button>
            <button style={tabBtnStyle(rightTab === 'memory')} onClick={() => setRightTab('memory')}>
              Memory
            </button>
            <button style={tabBtnStyle(rightTab === 'tasks')} onClick={() => setRightTab('tasks')}>
              Tasks ({Object.keys(tasksData?.active ?? {}).length})
            </button>
          </div>
          <div style={{ flex: 1, overflow: 'hidden' }}>
            {rightTab === 'llm' ? (
              <LLMStatusPanel />
            ) : rightTab === 'reviews' ? (
              <CodeReviewPanel />
            ) : rightTab === 'memory' ? (
              <MemoryPanel
                memory={memory ?? null}
                skills={skillsData?.skills ?? []}
              />
            ) : (
              <TaskPanel
                roster={roster}
                tasks={tasksData?.active ?? {}}
                onAssign={assignTask.mutate}
                onDelete={deleteTask.mutate}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
