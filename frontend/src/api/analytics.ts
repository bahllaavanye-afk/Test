import api from './client'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface BenchmarkEntry {
  name: string
  sharpe: number
  annual_return: number
  max_dd: number
  beating_sharpe: boolean | null
  beating_return: boolean | null
  sharpe_delta: number | null
  return_delta: number | null
}

export interface CompetitionReport {
  quantedge: {
    sharpe: number
    annual_return_pct: number
    max_drawdown_pct: number
    data_available: boolean
  }
  target: {
    sharpe: number
    annual_return: number
    max_dd: number
  }
  benchmarks: Record<string, BenchmarkEntry>
  benchmarks_beaten: number
  total_benchmarks: number
  rank_summary: string
  computed_at: string
}

// ─── API functions ────────────────────────────────────────────────────────────

export const getCompetitionReport = (): Promise<CompetitionReport> =>
  api.get('/analytics/competition-report').then(r => r.data)
