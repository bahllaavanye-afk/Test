import api from './client'

export interface TriggerConfig {
  type: 'schedule' | 'price_cross' | 'indicator'
  interval?: string
  price_level?: number
  direction?: string
  indicator?: string
  indicator_period?: number
  indicator_operator?: string
  indicator_value?: number
}

export interface ConditionConfig {
  type: 'indicator' | 'price_vs_ma' | 'pnl' | 'time_window' | 'position_exists' | 'no_position'
  indicator?: string
  period?: number
  operator?: string
  value?: number
  ma_period?: number
  start_time?: string
  end_time?: string
  pnl_pct?: number
}

export interface ActionConfig {
  type: 'open_long' | 'open_short' | 'close_position' | 'send_alert' | 'reduce_position'
  size_pct?: number
  stop_loss_pct?: number
  take_profit_pct?: number
  trailing_stop_pct?: number
  alert_message?: string
  reduce_by_pct?: number
}

export interface ExitRuleConfig {
  type: 'take_profit' | 'stop_loss' | 'trailing_stop' | 'time_exit' | 'indicator'
  value?: number
  hours?: number
  indicator?: string
  period?: number
  operator?: string
  indicator_value?: number
}

export interface BotOut {
  id: string
  name: string
  description: string
  symbol: string
  market_type: string
  trigger: TriggerConfig
  conditions: ConditionConfig[]
  condition_logic: string
  action: ActionConfig
  exit_rules: ExitRuleConfig[]
  is_enabled: boolean
  is_archived?: boolean
  archived_at?: string | null
  run_count: number
  last_run_at: string | null
  last_signal: string | null
  last_result: Record<string, unknown> | null
  template_id: string | null
  created_at: string
}

export interface BotCreate {
  name: string
  description: string
  symbol: string
  market_type: string
  trigger: TriggerConfig
  conditions: ConditionConfig[]
  condition_logic: string
  action: ActionConfig
  exit_rules: ExitRuleConfig[]
  template_id?: string | null
}

export interface BotTemplate {
  name: string
  description: string
  symbol: string
  market_type: string
  trigger: TriggerConfig
  conditions: ConditionConfig[]
  condition_logic: string
  action: ActionConfig
  exit_rules: ExitRuleConfig[]
}

export interface BotRunResult {
  fired: boolean
  reason: string
  signal: string
  orders_created: string[]
  details: Record<string, unknown>
}

export const botsApi = {
  getTemplates: () =>
    api.get<Record<string, BotTemplate>>('/bots/templates').then((r) => r.data),

  list: () =>
    api.get<BotOut[]>('/bots/').then((r) => r.data),

  listArchived: () =>
    api.get<BotOut[]>('/bots/?archived=true').then((r) => r.data),

  get: (id: string) =>
    api.get<BotOut>(`/bots/${id}`).then((r) => r.data),

  create: (data: BotCreate) =>
    api.post<BotOut>('/bots/', data).then((r) => r.data),

  update: (id: string, data: Partial<BotOut>) =>
    api.patch<BotOut>(`/bots/${id}`, data).then((r) => r.data),

  // Archive (soft-delete): preserves the bot row, config, and linked trades.
  archive: (id: string) =>
    api.delete(`/bots/${id}`),

  // Restore an archived bot back to the active list (returns it disabled).
  restore: (id: string) =>
    api.post<BotOut>(`/bots/${id}/restore`).then((r) => r.data),

  toggle: (id: string) =>
    api.post<BotOut>(`/bots/${id}/toggle`).then((r) => r.data),

  run: (id: string) =>
    api.post<BotRunResult>(`/bots/${id}/run`).then((r) => r.data),
}
