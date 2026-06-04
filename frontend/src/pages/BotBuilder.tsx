import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Bot,
  Plus,
  Play,
  Trash2,
  ToggleLeft,
  ToggleRight,
  ChevronDown,
  ChevronUp,
  Zap,
  Settings,
  List,
  Loader2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Minus,
} from 'lucide-react'
import {
  botsApi,
  BotCreate,
  BotOut,
  BotTemplate,
  TriggerConfig,
  ConditionConfig,
  ActionConfig,
  ExitRuleConfig,
} from '../api/bots'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ViewMode = 'build' | 'list'

interface SectionProps {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}

// ---------------------------------------------------------------------------
// Collapsible Section
// ---------------------------------------------------------------------------

function Section({ title, children, defaultOpen = true }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-[#1e1e1e] rounded-lg overflow-hidden mb-3">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-[#111111] hover:bg-[#1a1a1a] transition-colors text-left"
      >
        <span className="text-sm font-semibold text-[#e8e8e8] tracking-wide uppercase">{title}</span>
        {open ? <ChevronUp size={14} className="text-[#888]" /> : <ChevronDown size={14} className="text-[#888]" />}
      </button>
      {open && <div className="p-4 bg-[#0d0d0d]">{children}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Field helpers
// ---------------------------------------------------------------------------

function Label({ children }: { children: React.ReactNode }) {
  return <label className="block text-xs text-[#888] mb-1 font-mono uppercase tracking-widest">{children}</label>
}

function Input({
  value,
  onChange,
  placeholder,
  type = 'text',
  step,
}: {
  value: string | number
  onChange: (v: string) => void
  placeholder?: string
  type?: string
  step?: string
}) {
  return (
    <input
      type={type}
      step={step}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded px-3 py-2 text-sm text-[#e8e8e8] font-mono placeholder-[#444] focus:outline-none focus:border-[#f5a623] transition-colors"
    />
  )
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string
  onChange: (v: string) => void
  options: { label: string; value: string }[]
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded px-3 py-2 text-sm text-[#e8e8e8] font-mono focus:outline-none focus:border-[#f5a623] transition-colors"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

// ---------------------------------------------------------------------------
// Default form state
// ---------------------------------------------------------------------------

function defaultForm(): BotCreate {
  return {
    name: '',
    description: '',
    symbol: 'SPY',
    market_type: 'equity',
    trigger: { type: 'schedule', interval: '1h' },
    conditions: [],
    condition_logic: 'ALL',
    action: { type: 'open_long', size_pct: 5 },
    exit_rules: [],
    template_id: null,
  }
}

// ---------------------------------------------------------------------------
// Signal badge
// ---------------------------------------------------------------------------

function SignalBadge({ signal }: { signal: string | null }) {
  if (!signal) return <span className="text-[#555] text-xs font-mono">—</span>
  const map: Record<string, string> = {
    buy: 'bg-[#00c853]/20 text-[#00c853] border border-[#00c853]/30',
    sell: 'bg-[#ff1744]/20 text-[#ff1744] border border-[#ff1744]/30',
    hold: 'bg-[#555]/20 text-[#888] border border-[#555]/30',
    alert: 'bg-[#f5a623]/20 text-[#f5a623] border border-[#f5a623]/30',
  }
  const cls = map[signal.toLowerCase()] || map.hold
  return (
    <span className={`text-xs font-mono px-2 py-0.5 rounded ${cls}`}>
      {signal.toUpperCase()}
    </span>
  )
}

// ---------------------------------------------------------------------------
// BotBuilder page
// ---------------------------------------------------------------------------

export default function BotBuilder() {
  const queryClient = useQueryClient()
  const [view, setView] = useState<ViewMode>('build')
  const [form, setForm] = useState<BotCreate>(defaultForm())
  const [editingId, setEditingId] = useState<string | null>(null)
  const [runResults, setRunResults] = useState<Record<string, { fired: boolean; reason: string; signal: string }>>({})
  const [formError, setFormError] = useState<string | null>(null)
  const [formSuccess, setFormSuccess] = useState<string | null>(null)

  // Queries
  const { data: templates = {}, isLoading: loadingTemplates } = useQuery({
    queryKey: ['bot-templates'],
    queryFn: botsApi.getTemplates,
  })

  const { data: bots = [], isLoading: loadingBots, error: botsError } = useQuery({
    queryKey: ['bots'],
    queryFn: botsApi.list,
  })

  // Mutations
  const createMutation = useMutation({
    mutationFn: botsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bots'] })
      setForm(defaultForm())
      setEditingId(null)
      setFormSuccess('Bot created successfully.')
      setTimeout(() => setFormSuccess(null), 3000)
    },
    onError: (err: Error) => setFormError(err.message),
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<BotOut> }) => botsApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bots'] })
      setForm(defaultForm())
      setEditingId(null)
      setFormSuccess('Bot updated.')
      setTimeout(() => setFormSuccess(null), 3000)
    },
    onError: (err: Error) => setFormError(err.message),
  })

  const deleteMutation = useMutation({
    mutationFn: botsApi.delete,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bots'] }),
  })

  const toggleMutation = useMutation({
    mutationFn: botsApi.toggle,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bots'] }),
  })

  const runMutation = useMutation({
    mutationFn: botsApi.run,
    onSuccess: (result, id) => {
      setRunResults((prev) => ({ ...prev, [id]: result }))
      queryClient.invalidateQueries({ queryKey: ['bots'] })
    },
  })

  // ---------------------------------------------------------------------------
  // Template loading
  // ---------------------------------------------------------------------------

  function loadTemplate(key: string) {
    const t: BotTemplate = templates[key]
    if (!t) return
    setForm({
      name: t.name,
      description: t.description,
      symbol: t.symbol,
      market_type: t.market_type,
      trigger: t.trigger,
      conditions: t.conditions,
      condition_logic: t.condition_logic,
      action: t.action,
      exit_rules: t.exit_rules,
      template_id: key,
    })
    setEditingId(null)
    setView('build')
  }

  function loadBotForEdit(bot: BotOut) {
    setForm({
      name: bot.name,
      description: bot.description,
      symbol: bot.symbol,
      market_type: bot.market_type,
      trigger: bot.trigger,
      conditions: bot.conditions,
      condition_logic: bot.condition_logic,
      action: bot.action,
      exit_rules: bot.exit_rules,
      template_id: bot.template_id,
    })
    setEditingId(bot.id)
    setView('build')
  }

  // ---------------------------------------------------------------------------
  // Submit
  // ---------------------------------------------------------------------------

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setFormError(null)
    if (!form.name.trim()) { setFormError('Name is required.'); return }
    if (!form.symbol.trim()) { setFormError('Symbol is required.'); return }
    if (editingId) {
      updateMutation.mutate({ id: editingId, data: form })
    } else {
      createMutation.mutate(form)
    }
  }

  // ---------------------------------------------------------------------------
  // Condition helpers
  // ---------------------------------------------------------------------------

  function addCondition() {
    setForm((f) => ({
      ...f,
      conditions: [
        ...f.conditions,
        { type: 'indicator' as const, indicator: 'rsi', period: 14, operator: '<', value: 30 },
      ],
    }))
  }

  function updateCondition(i: number, partial: Partial<ConditionConfig>) {
    setForm((f) => {
      const conds = [...f.conditions]
      conds[i] = { ...conds[i], ...partial }
      return { ...f, conditions: conds }
    })
  }

  function removeCondition(i: number) {
    setForm((f) => ({ ...f, conditions: f.conditions.filter((_, idx) => idx !== i) }))
  }

  // ---------------------------------------------------------------------------
  // Exit rule helpers
  // ---------------------------------------------------------------------------

  function addExitRule() {
    setForm((f) => ({
      ...f,
      exit_rules: [...f.exit_rules, { type: 'stop_loss' as const, value: 2 }],
    }))
  }

  function updateExitRule(i: number, partial: Partial<ExitRuleConfig>) {
    setForm((f) => {
      const rules = [...f.exit_rules]
      rules[i] = { ...rules[i], ...partial }
      return { ...f, exit_rules: rules }
    })
  }

  function removeExitRule(i: number) {
    setForm((f) => ({ ...f, exit_rules: f.exit_rules.filter((_, idx) => idx !== i) }))
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="h-full flex flex-col bg-[#0a0a0a] text-[#e8e8e8]">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[#1e1e1e]">
        <div className="flex items-center gap-3">
          <Bot size={20} className="text-[#f5a623]" />
          <h1 className="text-lg font-semibold text-[#e8e8e8]">Bot Builder</h1>
          <span className="text-xs text-[#555] font-mono">24/7 Automated</span>
        </div>
        {/* Tab toggle */}
        <div className="flex bg-[#111111] border border-[#1e1e1e] rounded-lg overflow-hidden">
          <button
            onClick={() => setView('build')}
            className={`px-4 py-2 text-xs font-mono flex items-center gap-1.5 transition-colors ${
              view === 'build'
                ? 'bg-[#f5a623]/10 text-[#f5a623]'
                : 'text-[#888] hover:text-[#e8e8e8]'
            }`}
          >
            <Settings size={13} />
            Build
          </button>
          <button
            onClick={() => setView('list')}
            className={`px-4 py-2 text-xs font-mono flex items-center gap-1.5 transition-colors ${
              view === 'list'
                ? 'bg-[#f5a623]/10 text-[#f5a623]'
                : 'text-[#888] hover:text-[#e8e8e8]'
            }`}
          >
            <List size={13} />
            My Bots
            {bots.length > 0 && (
              <span className="bg-[#f5a623]/20 text-[#f5a623] text-xs px-1.5 py-0.5 rounded font-mono">
                {bots.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-hidden flex">
        {view === 'build' ? (
          <BuildView
            form={form}
            setForm={setForm}
            templates={templates}
            loadingTemplates={loadingTemplates}
            loadTemplate={loadTemplate}
            editingId={editingId}
            formError={formError}
            formSuccess={formSuccess}
            setFormError={setFormError}
            handleSubmit={handleSubmit}
            isPending={createMutation.isPending || updateMutation.isPending}
            addCondition={addCondition}
            updateCondition={updateCondition}
            removeCondition={removeCondition}
            addExitRule={addExitRule}
            updateExitRule={updateExitRule}
            removeExitRule={removeExitRule}
            onCancelEdit={() => { setEditingId(null); setForm(defaultForm()) }}
          />
        ) : (
          <ListView
            bots={bots}
            isLoading={loadingBots}
            error={botsError as Error | null}
            runResults={runResults}
            onEdit={loadBotForEdit}
            onDelete={(id) => deleteMutation.mutate(id)}
            onToggle={(id) => toggleMutation.mutate(id)}
            onRun={(id) => runMutation.mutate(id)}
            runningId={runMutation.isPending ? runMutation.variables ?? null : null}
          />
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Build view
// ---------------------------------------------------------------------------

interface BuildViewProps {
  form: BotCreate
  setForm: React.Dispatch<React.SetStateAction<BotCreate>>
  templates: Record<string, BotTemplate>
  loadingTemplates: boolean
  loadTemplate: (key: string) => void
  editingId: string | null
  formError: string | null
  formSuccess: string | null
  setFormError: (e: string | null) => void
  handleSubmit: (e: React.FormEvent) => void
  isPending: boolean
  addCondition: () => void
  updateCondition: (i: number, p: Partial<ConditionConfig>) => void
  removeCondition: (i: number) => void
  addExitRule: () => void
  updateExitRule: (i: number, p: Partial<ExitRuleConfig>) => void
  removeExitRule: (i: number) => void
  onCancelEdit: () => void
}

function BuildView({
  form,
  setForm,
  templates,
  loadingTemplates,
  loadTemplate,
  editingId,
  formError,
  formSuccess,
  setFormError,
  handleSubmit,
  isPending,
  addCondition,
  updateCondition,
  removeCondition,
  addExitRule,
  updateExitRule,
  removeExitRule,
  onCancelEdit,
}: BuildViewProps) {
  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Left: Templates (30%) */}
      <div className="w-80 flex-shrink-0 border-r border-[#1e1e1e] overflow-y-auto p-4">
        <p className="text-xs text-[#888] font-mono uppercase tracking-widest mb-3">Templates</p>
        {loadingTemplates ? (
          <div className="flex items-center gap-2 text-[#555] text-xs font-mono">
            <Loader2 size={12} className="animate-spin" />
            Loading…
          </div>
        ) : (
          <>
            <button
              onClick={() => setForm({
                name: '',
                description: '',
                symbol: 'SPY',
                market_type: 'equity',
                trigger: { type: 'schedule', interval: '1h' },
                conditions: [],
                condition_logic: 'ALL',
                action: { type: 'open_long', size_pct: 5 },
                exit_rules: [],
                template_id: null,
              })}
              className="w-full mb-2 px-3 py-2 border border-dashed border-[#2a2a2a] rounded-lg text-xs text-[#888] hover:border-[#f5a623] hover:text-[#f5a623] transition-colors font-mono"
            >
              + Start from scratch
            </button>
            <div className="space-y-2">
              {Object.entries(templates).map(([key, t]) => (
                <button
                  key={key}
                  onClick={() => loadTemplate(key)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg border transition-colors ${
                    form.template_id === key
                      ? 'border-[#f5a623] bg-[#f5a623]/5'
                      : 'border-[#1e1e1e] hover:border-[#2a2a2a] bg-[#111111]'
                  }`}
                >
                  <div className="text-xs font-semibold text-[#e8e8e8] mb-0.5">{t.name}</div>
                  <div className="text-xs text-[#666] line-clamp-2">{t.description}</div>
                  <div className="flex items-center gap-2 mt-1.5">
                    <span className="text-xs font-mono text-[#f5a623]">{t.symbol}</span>
                    <span className="text-xs text-[#555]">{t.trigger.interval || t.trigger.type}</span>
                    <span className="text-xs text-[#555]">{t.action.type}</span>
                  </div>
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Center: Form (70%) */}
      <div className="flex-1 overflow-y-auto p-4">
        <form onSubmit={handleSubmit}>
          {/* Basic Info */}
          <Section title="Basic Info">
            <div className="grid grid-cols-2 gap-3">
              <div className="col-span-2">
                <Label>Bot Name</Label>
                <Input
                  value={form.name}
                  onChange={(v) => setForm((f) => ({ ...f, name: v }))}
                  placeholder="My RSI Bot"
                />
              </div>
              <div>
                <Label>Symbol</Label>
                <Input
                  value={form.symbol}
                  onChange={(v) => setForm((f) => ({ ...f, symbol: v.toUpperCase() }))}
                  placeholder="SPY"
                />
              </div>
              <div>
                <Label>Market Type</Label>
                <Select
                  value={form.market_type}
                  onChange={(v) => setForm((f) => ({ ...f, market_type: v }))}
                  options={[
                    { label: 'Equity', value: 'equity' },
                    { label: 'Crypto', value: 'crypto' },
                    { label: 'Polymarket', value: 'polymarket' },
                  ]}
                />
              </div>
              <div className="col-span-2">
                <Label>Description</Label>
                <Input
                  value={form.description}
                  onChange={(v) => setForm((f) => ({ ...f, description: v }))}
                  placeholder="What does this bot do?"
                />
              </div>
            </div>
          </Section>

          {/* Trigger */}
          <Section title="Trigger">
            <TriggerEditor
              trigger={form.trigger}
              onChange={(t: TriggerConfig) => setForm((f) => ({ ...f, trigger: t }))}
            />
          </Section>

          {/* Conditions */}
          <Section title="Conditions">
            <div className="flex items-center gap-3 mb-3">
              <span className="text-xs text-[#888] font-mono">Logic:</span>
              {(['ALL', 'ANY'] as const).map((opt) => (
                <button
                  key={opt}
                  type="button"
                  onClick={() => setForm((f) => ({ ...f, condition_logic: opt }))}
                  className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
                    form.condition_logic === opt
                      ? 'bg-[#f5a623]/20 text-[#f5a623] border border-[#f5a623]/40'
                      : 'bg-[#1a1a1a] text-[#888] border border-[#2a2a2a] hover:text-[#e8e8e8]'
                  }`}
                >
                  {opt}
                </button>
              ))}
            </div>
            <div className="space-y-2">
              {form.conditions.map((cond, i) => (
                <ConditionRow
                  key={i}
                  cond={cond}
                  onChange={(p) => updateCondition(i, p)}
                  onRemove={() => removeCondition(i)}
                />
              ))}
            </div>
            <button
              type="button"
              onClick={addCondition}
              className="mt-2 flex items-center gap-1.5 text-xs text-[#f5a623] hover:text-[#f5a623]/80 font-mono transition-colors"
            >
              <Plus size={12} />
              Add Condition
            </button>
          </Section>

          {/* Action */}
          <Section title="Action">
            <ActionEditor
              action={form.action}
              onChange={(a: ActionConfig) => setForm((f) => ({ ...f, action: a }))}
            />
          </Section>

          {/* Exit Rules */}
          <Section title="Exit Rules" defaultOpen={false}>
            <div className="space-y-2">
              {form.exit_rules.map((rule, i) => (
                <ExitRuleRow
                  key={i}
                  rule={rule}
                  onChange={(p) => updateExitRule(i, p)}
                  onRemove={() => removeExitRule(i)}
                />
              ))}
            </div>
            <button
              type="button"
              onClick={addExitRule}
              className="mt-2 flex items-center gap-1.5 text-xs text-[#f5a623] hover:text-[#f5a623]/80 font-mono transition-colors"
            >
              <Plus size={12} />
              Add Exit Rule
            </button>
          </Section>

          {/* Error / Success */}
          {formError && (
            <div className="flex items-center gap-2 p-3 bg-[#ff1744]/10 border border-[#ff1744]/30 rounded-lg mb-3 text-xs text-[#ff1744] font-mono">
              <AlertCircle size={13} />
              {formError}
              <button type="button" onClick={() => setFormError(null)} className="ml-auto">
                <XCircle size={13} />
              </button>
            </div>
          )}
          {formSuccess && (
            <div className="flex items-center gap-2 p-3 bg-[#00c853]/10 border border-[#00c853]/30 rounded-lg mb-3 text-xs text-[#00c853] font-mono">
              <CheckCircle2 size={13} />
              {formSuccess}
            </div>
          )}

          {/* Submit */}
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={isPending}
              className="flex items-center gap-2 px-5 py-2.5 bg-[#f5a623] text-[#0a0a0a] rounded-lg text-sm font-semibold hover:bg-[#f5a623]/90 transition-colors disabled:opacity-50"
            >
              {isPending ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
              {editingId ? 'Update Bot' : 'Create Bot'}
            </button>
            {editingId && (
              <button
                type="button"
                onClick={onCancelEdit}
                className="px-4 py-2.5 text-sm text-[#888] hover:text-[#e8e8e8] transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Trigger editor
// ---------------------------------------------------------------------------

function TriggerEditor({ trigger, onChange }: { trigger: TriggerConfig; onChange: (t: TriggerConfig) => void }) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Trigger Type</Label>
        <Select
          value={trigger.type}
          onChange={(v) => onChange({ ...trigger, type: v as TriggerConfig['type'] })}
          options={[
            { label: 'Schedule (time-based)', value: 'schedule' },
            { label: 'Price Cross', value: 'price_cross' },
            { label: 'Indicator Signal', value: 'indicator' },
          ]}
        />
      </div>
      {trigger.type === 'schedule' && (
        <div>
          <Label>Interval</Label>
          <Select
            value={trigger.interval || '1h'}
            onChange={(v) => onChange({ ...trigger, interval: v })}
            options={[
              { label: 'Every 1 minute', value: '1m' },
              { label: 'Every 5 minutes', value: '5m' },
              { label: 'Every 15 minutes', value: '15m' },
              { label: 'Every 1 hour', value: '1h' },
              { label: 'Every 4 hours', value: '4h' },
              { label: 'Daily', value: '1d' },
            ]}
          />
        </div>
      )}
      {trigger.type === 'price_cross' && (
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Price Level</Label>
            <Input
              type="number"
              step="0.01"
              value={trigger.price_level ?? ''}
              onChange={(v) => onChange({ ...trigger, price_level: parseFloat(v) || undefined })}
              placeholder="100000"
            />
          </div>
          <div>
            <Label>Direction</Label>
            <Select
              value={trigger.direction || 'above'}
              onChange={(v) => onChange({ ...trigger, direction: v })}
              options={[
                { label: 'Crosses Above', value: 'above' },
                { label: 'Crosses Below', value: 'below' },
              ]}
            />
          </div>
        </div>
      )}
      {trigger.type === 'indicator' && (
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Indicator</Label>
            <Select
              value={trigger.indicator || 'rsi'}
              onChange={(v) => onChange({ ...trigger, indicator: v })}
              options={[
                { label: 'RSI', value: 'rsi' },
                { label: 'MACD', value: 'macd' },
                { label: 'Bollinger Bands', value: 'bb' },
                { label: 'SMA', value: 'sma' },
                { label: 'EMA', value: 'ema' },
              ]}
            />
          </div>
          <div>
            <Label>Period</Label>
            <Input
              type="number"
              value={trigger.indicator_period ?? 14}
              onChange={(v) => onChange({ ...trigger, indicator_period: parseInt(v) || 14 })}
            />
          </div>
          <div>
            <Label>Operator</Label>
            <Select
              value={trigger.indicator_operator || '<'}
              onChange={(v) => onChange({ ...trigger, indicator_operator: v })}
              options={[
                { label: '<', value: '<' },
                { label: '>', value: '>' },
                { label: 'Crosses Above', value: 'crosses_above' },
                { label: 'Crosses Below', value: 'crosses_below' },
              ]}
            />
          </div>
          <div>
            <Label>Value</Label>
            <Input
              type="number"
              step="0.01"
              value={trigger.indicator_value ?? ''}
              onChange={(v) => onChange({ ...trigger, indicator_value: parseFloat(v) || undefined })}
              placeholder="30"
            />
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Condition row
// ---------------------------------------------------------------------------

function ConditionRow({
  cond,
  onChange,
  onRemove,
}: {
  cond: ConditionConfig
  onChange: (p: Partial<ConditionConfig>) => void
  onRemove: () => void
}) {
  return (
    <div className="flex items-start gap-2 p-3 bg-[#111111] border border-[#1e1e1e] rounded-lg">
      <div className="flex-1 grid grid-cols-4 gap-2">
        <div>
          <Label>Type</Label>
          <Select
            value={cond.type}
            onChange={(v) => onChange({ type: v as ConditionConfig['type'] })}
            options={[
              { label: 'Indicator', value: 'indicator' },
              { label: 'Price vs MA', value: 'price_vs_ma' },
              { label: 'PnL %', value: 'pnl' },
              { label: 'Time Window', value: 'time_window' },
              { label: 'Has Position', value: 'position_exists' },
              { label: 'No Position', value: 'no_position' },
            ]}
          />
        </div>
        {cond.type === 'indicator' && (
          <>
            <div>
              <Label>Indicator</Label>
              <Select
                value={cond.indicator || 'rsi'}
                onChange={(v) => onChange({ indicator: v })}
                options={[
                  { label: 'RSI', value: 'rsi' },
                  { label: 'MACD', value: 'macd' },
                  { label: 'BB', value: 'bb' },
                  { label: 'SMA', value: 'sma' },
                  { label: 'EMA', value: 'ema' },
                  { label: 'EMA Cross', value: 'ema_cross' },
                ]}
              />
            </div>
            <div>
              <Label>Operator</Label>
              <Select
                value={cond.operator || '<'}
                onChange={(v) => onChange({ operator: v })}
                options={[
                  { label: '<', value: '<' },
                  { label: '>', value: '>' },
                  { label: '==', value: '==' },
                  { label: 'Bullish Cross', value: 'bullish_cross' },
                  { label: 'Bearish Cross', value: 'bearish_cross' },
                  { label: 'Below Lower BB', value: 'price_below_lower' },
                  { label: 'Above Upper BB', value: 'price_above_upper' },
                ]}
              />
            </div>
            <div>
              <Label>Value</Label>
              <Input
                type="number"
                step="0.1"
                value={cond.value ?? ''}
                onChange={(v) => onChange({ value: parseFloat(v) || undefined })}
                placeholder="30"
              />
            </div>
          </>
        )}
        {cond.type === 'price_vs_ma' && (
          <>
            <div>
              <Label>MA Period</Label>
              <Input
                type="number"
                value={cond.ma_period ?? 50}
                onChange={(v) => onChange({ ma_period: parseInt(v) || 50 })}
              />
            </div>
            <div>
              <Label>Operator</Label>
              <Select
                value={cond.operator || '>'}
                onChange={(v) => onChange({ operator: v })}
                options={[
                  { label: 'Price > MA', value: '>' },
                  { label: 'Price < MA', value: '<' },
                ]}
              />
            </div>
          </>
        )}
        {cond.type === 'pnl' && (
          <>
            <div>
              <Label>Operator</Label>
              <Select
                value={cond.operator || '<'}
                onChange={(v) => onChange({ operator: v })}
                options={[
                  { label: '<', value: '<' },
                  { label: '>', value: '>' },
                ]}
              />
            </div>
            <div>
              <Label>PnL %</Label>
              <Input
                type="number"
                step="0.1"
                value={cond.pnl_pct ?? ''}
                onChange={(v) => onChange({ pnl_pct: parseFloat(v) || undefined })}
                placeholder="-2.0"
              />
            </div>
          </>
        )}
        {cond.type === 'time_window' && (
          <>
            <div>
              <Label>Start (ET)</Label>
              <Input
                value={cond.start_time || '09:30'}
                onChange={(v) => onChange({ start_time: v })}
                placeholder="09:30"
              />
            </div>
            <div>
              <Label>End (ET)</Label>
              <Input
                value={cond.end_time || '16:00'}
                onChange={(v) => onChange({ end_time: v })}
                placeholder="16:00"
              />
            </div>
          </>
        )}
      </div>
      <button
        type="button"
        onClick={onRemove}
        className="mt-5 text-[#555] hover:text-[#ff1744] transition-colors"
      >
        <Minus size={14} />
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Action editor
// ---------------------------------------------------------------------------

function ActionEditor({ action, onChange }: { action: ActionConfig; onChange: (a: ActionConfig) => void }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <div className="col-span-2">
        <Label>Action Type</Label>
        <Select
          value={action.type}
          onChange={(v) => onChange({ ...action, type: v as ActionConfig['type'] })}
          options={[
            { label: 'Open Long', value: 'open_long' },
            { label: 'Open Short', value: 'open_short' },
            { label: 'Close Position', value: 'close_position' },
            { label: 'Send Alert', value: 'send_alert' },
            { label: 'Reduce Position', value: 'reduce_position' },
          ]}
        />
      </div>
      {action.type !== 'send_alert' && action.type !== 'close_position' && (
        <>
          <div>
            <Label>Position Size (%)</Label>
            <Input
              type="number"
              step="0.5"
              value={action.size_pct ?? 5}
              onChange={(v) => onChange({ ...action, size_pct: parseFloat(v) || 5 })}
              placeholder="5"
            />
          </div>
          <div>
            <Label>Stop Loss (%)</Label>
            <Input
              type="number"
              step="0.5"
              value={action.stop_loss_pct ?? ''}
              onChange={(v) => onChange({ ...action, stop_loss_pct: parseFloat(v) || undefined })}
              placeholder="2"
            />
          </div>
          <div>
            <Label>Take Profit (%)</Label>
            <Input
              type="number"
              step="0.5"
              value={action.take_profit_pct ?? ''}
              onChange={(v) => onChange({ ...action, take_profit_pct: parseFloat(v) || undefined })}
              placeholder="5"
            />
          </div>
          <div>
            <Label>Trailing Stop (%)</Label>
            <Input
              type="number"
              step="0.5"
              value={action.trailing_stop_pct ?? ''}
              onChange={(v) => onChange({ ...action, trailing_stop_pct: parseFloat(v) || undefined })}
              placeholder="1.5"
            />
          </div>
        </>
      )}
      {action.type === 'send_alert' && (
        <div className="col-span-2">
          <Label>Alert Message</Label>
          <Input
            value={action.alert_message || ''}
            onChange={(v) => onChange({ ...action, alert_message: v })}
            placeholder="Enter alert message..."
          />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Exit rule row
// ---------------------------------------------------------------------------

function ExitRuleRow({
  rule,
  onChange,
  onRemove,
}: {
  rule: ExitRuleConfig
  onChange: (p: Partial<ExitRuleConfig>) => void
  onRemove: () => void
}) {
  return (
    <div className="flex items-start gap-2 p-3 bg-[#111111] border border-[#1e1e1e] rounded-lg">
      <div className="flex-1 grid grid-cols-3 gap-2">
        <div>
          <Label>Type</Label>
          <Select
            value={rule.type}
            onChange={(v) => onChange({ type: v as ExitRuleConfig['type'] })}
            options={[
              { label: 'Take Profit', value: 'take_profit' },
              { label: 'Stop Loss', value: 'stop_loss' },
              { label: 'Trailing Stop', value: 'trailing_stop' },
              { label: 'Time Exit', value: 'time_exit' },
              { label: 'Indicator', value: 'indicator' },
            ]}
          />
        </div>
        {(rule.type === 'take_profit' || rule.type === 'stop_loss' || rule.type === 'trailing_stop') && (
          <div>
            <Label>Value (%)</Label>
            <Input
              type="number"
              step="0.5"
              value={rule.value ?? ''}
              onChange={(v) => onChange({ value: parseFloat(v) || undefined })}
              placeholder="2"
            />
          </div>
        )}
        {rule.type === 'time_exit' && (
          <div>
            <Label>After (hours)</Label>
            <Input
              type="number"
              value={rule.hours ?? ''}
              onChange={(v) => onChange({ hours: parseInt(v) || undefined })}
              placeholder="24"
            />
          </div>
        )}
        {rule.type === 'indicator' && (
          <>
            <div>
              <Label>Indicator</Label>
              <Select
                value={rule.indicator || 'rsi'}
                onChange={(v) => onChange({ indicator: v })}
                options={[
                  { label: 'RSI', value: 'rsi' },
                  { label: 'MACD', value: 'macd' },
                  { label: 'BB', value: 'bb' },
                  { label: 'EMA Cross', value: 'ema_cross' },
                ]}
              />
            </div>
            <div>
              <Label>Operator / Value</Label>
              <Input
                type="number"
                step="0.1"
                value={rule.indicator_value ?? ''}
                onChange={(v) => onChange({ indicator_value: parseFloat(v) || undefined })}
                placeholder="60"
              />
            </div>
          </>
        )}
      </div>
      <button
        type="button"
        onClick={onRemove}
        className="mt-5 text-[#555] hover:text-[#ff1744] transition-colors"
      >
        <Minus size={14} />
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// List view
// ---------------------------------------------------------------------------

interface ListViewProps {
  bots: BotOut[]
  isLoading: boolean
  error: Error | null
  runResults: Record<string, { fired: boolean; reason: string; signal: string }>
  onEdit: (bot: BotOut) => void
  onDelete: (id: string) => void
  onToggle: (id: string) => void
  onRun: (id: string) => void
  runningId: string | null
}

function ListView({
  bots,
  isLoading,
  error,
  runResults,
  onEdit,
  onDelete,
  onToggle,
  onRun,
  runningId,
}: ListViewProps) {
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-[#555]">
        <Loader2 size={20} className="animate-spin mr-2" />
        <span className="text-sm font-mono">Loading bots…</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex items-center gap-2 text-[#ff1744] text-sm font-mono">
          <AlertCircle size={16} />
          {error.message}
        </div>
      </div>
    )
  }

  if (bots.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
        <div className="text-6xl mb-4">🤖</div>
        <p className="text-[#e8e8e8] font-semibold mb-2">No bots yet.</p>
        <p className="text-[#555] text-sm font-mono">
          Start with a template on the Build tab to create your first automated bot.
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="text-[#555] border-b border-[#1e1e1e]">
              <th className="text-left py-2 px-3">Name</th>
              <th className="text-left py-2 px-3">Symbol</th>
              <th className="text-left py-2 px-3">Status</th>
              <th className="text-left py-2 px-3">Last Signal</th>
              <th className="text-left py-2 px-3">Runs</th>
              <th className="text-left py-2 px-3">Last Run</th>
              <th className="text-left py-2 px-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {bots.map((bot) => {
              const rr = runResults[bot.id]
              return (
                <tr
                  key={bot.id}
                  className="border-b border-[#111111] hover:bg-[#111111] transition-colors"
                >
                  <td className="py-2.5 px-3">
                    <div className="text-[#e8e8e8] font-semibold">{bot.name}</div>
                    {bot.template_id && (
                      <div className="text-[#555] text-xs">{bot.template_id}</div>
                    )}
                  </td>
                  <td className="py-2.5 px-3">
                    <span className="text-[#f5a623]">{bot.symbol}</span>
                    <span className="text-[#555] ml-1">{bot.market_type}</span>
                  </td>
                  <td className="py-2.5 px-3">
                    <button
                      onClick={() => onToggle(bot.id)}
                      className="flex items-center gap-1.5 transition-colors"
                    >
                      {bot.is_enabled ? (
                        <>
                          <ToggleRight size={16} className="text-[#00c853]" />
                          <span className="text-[#00c853]">ON</span>
                        </>
                      ) : (
                        <>
                          <ToggleLeft size={16} className="text-[#555]" />
                          <span className="text-[#555]">OFF</span>
                        </>
                      )}
                    </button>
                  </td>
                  <td className="py-2.5 px-3">
                    <SignalBadge signal={rr?.signal ?? bot.last_signal} />
                  </td>
                  <td className="py-2.5 px-3 text-[#888]">{bot.run_count}</td>
                  <td className="py-2.5 px-3 text-[#555]">
                    {bot.last_run_at
                      ? new Date(bot.last_run_at).toLocaleString(undefined, { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' })
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => onRun(bot.id)}
                        disabled={runningId === bot.id}
                        title="Run now"
                        className="text-[#888] hover:text-[#f5a623] transition-colors disabled:opacity-50"
                      >
                        {runningId === bot.id ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <Play size={14} />
                        )}
                      </button>
                      <button
                        onClick={() => onEdit(bot)}
                        title="Edit"
                        className="text-[#888] hover:text-[#e8e8e8] transition-colors"
                      >
                        <Settings size={14} />
                      </button>
                      <button
                        onClick={() => {
                          if (confirm(`Delete bot "${bot.name}"?`)) onDelete(bot.id)
                        }}
                        title="Delete"
                        className="text-[#888] hover:text-[#ff1744] transition-colors"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    {rr && (
                      <div
                        className={`text-xs mt-1 ${rr.fired ? 'text-[#00c853]' : 'text-[#555]'}`}
                        title={rr.reason}
                      >
                        {rr.fired ? '✓ Fired' : '○ No signal'}
                      </div>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
