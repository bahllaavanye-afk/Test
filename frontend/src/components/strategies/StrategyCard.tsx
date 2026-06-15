/**
 * StrategyCard — displays a single strategy's key metrics with an optional
 * ML Enhancement row showing ML Sharpe and the percentage delta vs manual Sharpe.
 */

export interface Strategy {
  id: string
  name: string
  symbol: string
  status: 'active' | 'paper' | 'disabled'
  sharpe: number
  annual_return: number
  max_drawdown: number
  win_rate?: number
  ml_sharpe?: number
}

interface StrategyCardProps {
  strategy: Strategy
}

function statusColor(status: Strategy['status']): string {
  switch (status) {
    case 'active': return '#00c853'
    case 'paper': return '#f5a623'
    case 'disabled': return '#555'
  }
}

function sharpeColor(v: number): string {
  if (v >= 1.5) return '#00c853'
  if (v >= 1.0) return '#f5a623'
  return '#888'
}

export function StrategyCard({ strategy }: StrategyCardProps) {
  const {
    name,
    symbol,
    status,
    sharpe,
    annual_return,
    max_drawdown,
    win_rate,
    ml_sharpe,
  } = strategy

  return (
    <div className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-4 hover:border-[#2a2a2a] transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-sm font-bold text-[#e8e8e8]">{name}</h3>
          <span className="text-[11px] font-mono text-[#888]">{symbol}</span>
        </div>
        <span
          className="text-[9px] font-bold uppercase px-2 py-0.5 rounded"
          style={{
            color: statusColor(status),
            background: `${statusColor(status)}18`,
            border: `1px solid ${statusColor(status)}40`,
          }}
        >
          {status}
        </span>
      </div>

      {/* Metrics */}
      <div className="space-y-1.5">
        {/* Annual Return */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Annual Return</span>
          <span
            className="font-mono font-bold"
            style={{ color: annual_return >= 0 ? '#00c853' : '#ff1744' }}
          >
            {annual_return >= 0 ? '+' : ''}{annual_return.toFixed(1)}%
          </span>
        </div>

        {/* Max Drawdown */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Max Drawdown</span>
          <span className="font-mono text-[#ff1744]">{max_drawdown.toFixed(1)}%</span>
        </div>

        {/* Win Rate */}
        {win_rate != null && (
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Win Rate</span>
            <span className="font-mono text-[#e8e8e8]">{win_rate.toFixed(1)}%</span>
          </div>
        )}

        {/* Sharpe */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Sharpe</span>
          <span className="font-mono font-bold" style={{ color: sharpeColor(sharpe) }}>
            {sharpe.toFixed(2)}
          </span>
        </div>

        {/* ML Enhancement row */}
        {ml_sharpe != null && (
          <div className="flex items-center justify-between text-xs mt-1">
            <span className="text-gray-500">ML Sharpe</span>
            <div className="flex items-center gap-1">
              <span className="text-white">{ml_sharpe.toFixed(2)}</span>
              {ml_sharpe > sharpe ? (
                <span className="text-green-400 bg-green-900/30 px-1.5 py-0.5 rounded text-xs">
                  +{((ml_sharpe - sharpe) / Math.abs(sharpe) * 100).toFixed(0)}%
                </span>
              ) : (
                <span className="text-red-400 bg-red-900/30 px-1.5 py-0.5 rounded text-xs">
                  {((ml_sharpe - sharpe) / Math.abs(sharpe) * 100).toFixed(0)}%
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default StrategyCard
