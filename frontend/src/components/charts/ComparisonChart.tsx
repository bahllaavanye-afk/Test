/**
 * ComparisonChart — overlaid equity curves for manual vs ML vs benchmarks.
 * Pure SVG, no external chart library required. Bloomberg dark theme.
 */
import { useState } from 'react'

export interface CurveSeries {
  name: string
  color: string
  /** starting value = 100, subsequent values normalised to that base */
  values: number[]
  /** thicker stroke + label for the strategies we own */
  isPrimary?: boolean
}

interface Props {
  series: CurveSeries[]
  height?: number
  /** optional x-axis labels (dates or periods) */
  labels?: string[]
}

const W = 860
const PAD_L = 52
const PAD_R = 20
const PAD_T = 18
const PAD_B = 30

export function ComparisonChart({ series, height = 300, labels }: Props) {
  const [hovered, setHovered] = useState<string | null>(null)

  const chartW = W - PAD_L - PAD_R
  const chartH = height - PAD_T - PAD_B

  const days = series[0]?.values.length ?? 0
  if (days === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#555', fontFamily: 'monospace', fontSize: 13 }}>
        No equity curve data
      </div>
    )
  }

  const allValues = series.flatMap(s => s.values)
  const minV = Math.min(...allValues) * 0.99
  const maxV = Math.max(...allValues) * 1.005
  const range = maxV - minV

  const toX = (i: number) => PAD_L + (i / Math.max(days - 1, 1)) * chartW
  const toY = (v: number) => PAD_T + (1 - (v - minV) / range) * chartH

  const yTicks = 5
  const yGrids = Array.from({ length: yTicks }, (_, i) => minV + (range * i) / (yTicks - 1))

  // X-axis sample labels (show ~6 evenly spaced)
  const xSamples = labels && labels.length > 0
    ? Array.from({ length: 6 }, (_, i) => Math.round(i * (days - 1) / 5))
    : []

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} width="100%" style={{ display: 'block' }} onMouseLeave={() => setHovered(null)}>
        {/* Y grid */}
        {yGrids.map((v, i) => (
          <g key={i}>
            <line x1={PAD_L} y1={toY(v).toFixed(1)} x2={W - PAD_R} y2={toY(v).toFixed(1)} stroke="#1e2433" strokeWidth="1" />
            <text x={PAD_L - 5} y={toY(v) + 4} fill="#555" fontSize="9" fontFamily="monospace" textAnchor="end">
              {v.toFixed(0)}
            </text>
          </g>
        ))}
        {/* Baseline at 100 */}
        <line x1={PAD_L} y1={toY(100).toFixed(1)} x2={W - PAD_R} y2={toY(100).toFixed(1)} stroke="#333" strokeWidth="1" strokeDasharray="4 3" />

        {/* X-axis labels */}
        {xSamples.map(idx => labels && (
          <text key={idx} x={toX(idx)} y={height - 6} fill="#444" fontSize="8" fontFamily="monospace" textAnchor="middle">
            {labels[idx]}
          </text>
        ))}

        {/* Curves */}
        {series.map(s => {
          const pts = s.values.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' L ')
          const linePath = `M ${pts}`
          const isActive = hovered === null || hovered === s.name
          const lastVal = s.values[days - 1]
          return (
            <g key={s.name} style={{ cursor: 'pointer' }} onMouseEnter={() => setHovered(s.name)}>
              <path
                d={linePath}
                fill="none"
                stroke={s.color}
                strokeWidth={s.isPrimary ? 2.5 : 1.5}
                strokeOpacity={isActive ? 1 : 0.2}
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ transition: 'stroke-opacity 0.15s' }}
              />
              {s.isPrimary && (
                <>
                  <circle cx={toX(days - 1)} cy={toY(lastVal)} r="4" fill={s.color} opacity={isActive ? 1 : 0.2} />
                  <text
                    x={toX(days - 1) + 7}
                    y={toY(lastVal) + 4}
                    fill={s.color}
                    fontSize="9"
                    fontFamily="monospace"
                    fontWeight="bold"
                    opacity={isActive ? 1 : 0.2}
                  >
                    {lastVal.toFixed(1)}
                  </text>
                </>
              )}
            </g>
          )
        })}
      </svg>

      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px 20px', marginTop: 8, paddingLeft: PAD_L }}>
        {series.map(s => (
          <button
            key={s.name}
            onClick={() => setHovered(hovered === s.name ? null : s.name)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer', padding: 0,
              display: 'flex', alignItems: 'center', gap: 5,
              opacity: hovered === null || hovered === s.name ? 1 : 0.4,
              transition: 'opacity 0.15s',
            }}
          >
            <span style={{ display: 'inline-block', width: 24, height: 2, background: s.color, borderRadius: 1 }} />
            <span style={{ color: '#aaa', fontSize: 11, fontFamily: 'monospace' }}>{s.name}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

export default ComparisonChart
