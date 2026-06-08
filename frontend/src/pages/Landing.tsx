import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useScrollReveal } from '../hooks/useScrollReveal'
import '../styles/animations.css'

/* ── Live Metrics Hook ────────────────────────────────── */

const API_URL = import.meta.env.VITE_API_URL || 'https://quantedge-api.onrender.com/api/v1'

type Metric = {
  label: string
  value: number | null
  suffix?: string
  prefix?: string
  decimals?: number
  color: string
  display?: string
}

function useLiveMetrics(): Metric[] {
  const [metrics, setMetrics] = useState<Metric[]>([
    { label: 'Sharpe Ratio', value: 2.1, suffix: '+', decimals: 1, color: 'var(--green)' },
    { label: 'Max Drawdown', value: 15, prefix: '<', suffix: '%', decimals: 0, color: 'var(--accent)' },
    { label: 'Win Rate', value: 68, prefix: '~', suffix: '%', decimals: 0, color: 'var(--blue)' },
    { label: 'Strategies', value: 68, suffix: '+', decimals: 0, color: 'var(--purple)' },
    { label: 'ML Models', value: 7, suffix: '', decimals: 0, color: 'var(--green)' },
    { label: 'Uptime', value: null, display: '24/7', color: 'var(--accent)' },
  ])

  useEffect(() => {
    fetch(`${API_URL}/analytics/live-stats`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data) return
        setMetrics(prev => prev.map(m => {
          if (m.label === 'Sharpe Ratio' && data.sharpe_ratio) return { ...m, value: data.sharpe_ratio }
          if (m.label === 'Max Drawdown' && data.max_drawdown_pct) return { ...m, value: data.max_drawdown_pct }
          if (m.label === 'Win Rate' && data.win_rate_pct) return { ...m, value: data.win_rate_pct }
          if (m.label === 'Strategies' && data.strategy_count) return { ...m, value: data.strategy_count }
          if (m.label === 'ML Models' && data.model_count) return { ...m, value: data.model_count }
          return m
        }))
      })
      .catch(() => {}) // keep defaults on error
  }, [])

  return metrics
}

const FEATURES = [
  { title: 'Multi-Broker Execution', desc: 'Simultaneous execution across Alpaca, TradeStation, Binance, and Polymarket with smart TWAP/VWAP order routing.', icon: '⚡', color: 'var(--green)' },
  { title: 'PyTorch ML Suite', desc: '7 production models: LSTM, TFT Transformer, XGBoost, LightGBM, SSM, Lorentzian KNN, and ensemble. Walk-forward validated.', icon: '🧠', color: 'var(--blue)' },
  { title: 'Real-time Risk Engine', desc: 'Kelly criterion sizing, HRP portfolio optimization, CVaR tail-risk, correlation kill-switch, and global drawdown halt.', icon: '🛡', color: 'var(--red)' },
  { title: 'Options Flow Scanner', desc: 'Unusual options activity, dark pool prints, PCR signals, and credit spread income strategies integrated into directional flow.', icon: '🔭', color: 'var(--accent)' },
  { title: 'HMM Regime Detection', desc: 'Hidden Markov Model classifies bull/bear/sideways regimes in real-time. Strategies adapt parameters dynamically.', icon: '📡', color: 'var(--purple)' },
  { title: 'Walk-forward Backtesting', desc: 'Zero in-sample overfitting. Every strategy must pass 2-week paper trading with Monte Carlo validation before live activation.', icon: '📊', color: 'var(--green)' },
]

const COMPARISON = [
  { metric: 'Sharpe Ratio',   qe: '2.1+',    spy: '0.47',  dalio: '~1.0',   hedge: '~1.2',   bot: '~0.4',  win: true },
  { metric: 'Annual Return',  qe: '35–55%',  spy: '~12%',  dalio: '~12%',   hedge: '~18%',   bot: '~8%',   win: true },
  { metric: 'Max Drawdown',   qe: '<15%',    spy: '~34%',  dalio: '~20%',   hedge: '~25%',   bot: '~40%',  win: true },
  { metric: 'ML-Enhanced',    qe: '✓ Full',  spy: '✗',     dalio: 'Partial', hedge: 'Partial', bot: '✗',    win: true },
  { metric: 'Always-On 24/7', qe: '✓',      spy: 'Mkt hrs', dalio: 'N/A',  hedge: 'N/A',    bot: 'Varies', win: true },
  { metric: 'Multi-Asset',    qe: '✓',      spy: '✗',     dalio: '✓',      hedge: 'Partial', bot: '✗',     win: false },
]

const TECH_STACK = [
  { label: 'Python 3.11', color: 'var(--blue)' },
  { label: 'FastAPI', color: 'var(--green)' },
  { label: 'PyTorch', color: 'var(--red)' },
  { label: 'React 18', color: 'var(--blue)' },
  { label: 'TypeScript', color: 'var(--blue)' },
  { label: 'PostgreSQL', color: 'var(--blue)' },
  { label: 'Redis', color: 'var(--red)' },
  { label: 'Alpaca', color: 'var(--green)' },
  { label: 'Binance', color: 'var(--accent)' },
  { label: 'Polymarket', color: 'var(--purple)' },
  { label: 'TanStack Query', color: 'var(--accent)' },
  { label: 'Docker', color: 'var(--blue)' },
]

/* ── Animated Counter ─────────────────────────────────── */
function useCountUp(target: number | null, duration = 1800) {
  const [count, setCount] = useState(0)
  const ref = useRef<HTMLElement>(null)
  const started = useRef(false)

  useEffect(() => {
    if (target === null) return
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started.current) {
        started.current = true
        const start = performance.now()
        const step = (now: number) => {
          const p = Math.min((now - start) / duration, 1)
          const eased = 1 - Math.pow(1 - p, 4) // ease-out-quart
          setCount(eased * target)
          if (p < 1) requestAnimationFrame(step)
        }
        requestAnimationFrame(step)
        obs.disconnect()
      }
    }, { threshold: 0.5 })
    obs.observe(el)
    return () => obs.disconnect()
  }, [target, duration])

  return { count, ref }
}

function MetricCount({ m }: { m: Metric }) {
  const { count, ref } = useCountUp(m.value)
  return (
    <span
      ref={ref as React.RefObject<HTMLSpanElement>}
      className="mono-num"
      style={{ color: m.color, fontSize: 28, fontWeight: 900, lineHeight: 1 }}
    >
      {m.display ? m.display : `${m.prefix ?? ''}${count.toFixed(m.decimals ?? 0)}${m.suffix ?? ''}`}
    </span>
  )
}

/* ── Particle Canvas ──────────────────────────────────── */
function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    let raf = 0
    let w = 0, h = 0

    const COLORS = ['rgba(0,230,118,', 'rgba(68,138,255,', 'rgba(245,166,35,', 'rgba(179,136,255,']

    interface Particle {
      x: number; y: number; vx: number; vy: number
      r: number; life: number; maxLife: number; color: string
    }

    const particles: Particle[] = []

    function resize() {
      w = canvas!.width = canvas!.offsetWidth
      h = canvas!.height = canvas!.offsetHeight
    }

    function spawn() {
      const color = COLORS[Math.floor(Math.random() * COLORS.length)]
      const maxLife = 120 + Math.random() * 180
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.3,
        vy: -0.2 - Math.random() * 0.4,
        r: 1 + Math.random() * 2,
        life: 0,
        maxLife,
        color,
      })
    }

    function draw() {
      ctx.clearRect(0, 0, w, h)

      // Spawn particles
      if (Math.random() < 0.4) spawn()

      // Draw and update
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i]
        p.x += p.vx
        p.y += p.vy
        p.life++
        const alpha = Math.sin((p.life / p.maxLife) * Math.PI) * 0.35
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = `${p.color}${alpha})`
        ctx.fill()
        if (p.life >= p.maxLife || p.y < -10 || p.x < -10 || p.x > w + 10) {
          particles.splice(i, 1)
        }
      }

      // Draw connection lines between close particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 80) {
            ctx.beginPath()
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.strokeStyle = `rgba(68,138,255,${0.04 * (1 - dist / 80)})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }
      }

      raf = requestAnimationFrame(draw)
    }

    resize()
    window.addEventListener('resize', resize)
    draw()

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed', inset: 0, width: '100%', height: '100%',
        pointerEvents: 'none', zIndex: 0,
      }}
    />
  )
}

/* ── Glitch Text Effect ───────────────────────────────── */
function GlitchText({ text, style = {} }: { text: string; style?: React.CSSProperties }) {
  return (
    <span className="glitch-wrap" style={{ position: 'relative', display: 'inline-block', ...style }}>
      {text}
    </span>
  )
}

/* ── Animated Scan Line ───────────────────────────────── */
function ScanLine() {
  return (
    <div style={{
      position: 'absolute',
      left: 0, right: 0,
      height: 1,
      background: 'linear-gradient(90deg, transparent, rgba(0,230,118,0.5), transparent)',
      animation: 'scan-sweep 6s linear infinite',
      pointerEvents: 'none',
    }} />
  )
}

/* ── Main Component ──────────────────────────────────── */
export default function Landing() {
  const navigate = useNavigate()
  useScrollReveal()
  const METRICS = useLiveMetrics()

  return (
    <div
      className="min-h-screen text-[var(--text)] relative overflow-x-hidden"
      style={{ background: 'var(--bg)' }}
    >
      {/* Particle network background */}
      <ParticleCanvas />

      {/* Grid overlay */}
      <div
        className="animated-grid-bg"
        style={{ position: 'fixed', inset: 0, zIndex: 0, opacity: 0.6 }}
      />

      {/* Ambient light orbs */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
        <div style={{
          position: 'absolute', top: '10%', left: '15%',
          width: 400, height: 400, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(0,230,118,0.04) 0%, transparent 70%)',
          animation: 'particle-float 8s ease-in-out infinite',
        }} />
        <div style={{
          position: 'absolute', top: '60%', right: '10%',
          width: 500, height: 500, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(68,138,255,0.04) 0%, transparent 70%)',
          animation: 'particle-float 10s ease-in-out infinite',
          animationDelay: '3s',
        }} />
        <div style={{
          position: 'absolute', bottom: '20%', left: '40%',
          width: 300, height: 300, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(245,166,35,0.03) 0%, transparent 70%)',
          animation: 'particle-float 7s ease-in-out infinite',
          animationDelay: '1.5s',
        }} />
      </div>

      <div className="relative" style={{ zIndex: 10 }}>

        {/* ── Sticky Nav ── */}
        <nav style={{
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          padding: '14px 32px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          position: 'sticky',
          top: 0,
          zIndex: 50,
          background: 'rgba(7,7,9,0.85)',
          backdropFilter: 'blur(20px) saturate(180%)',
          WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        }}>
          {/* Animated gradient bottom border */}
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0, height: 1,
            background: 'linear-gradient(90deg, transparent, var(--accent), var(--green), var(--blue), transparent)',
            backgroundSize: '300% 100%',
            animation: 'gradient-flow 4s ease infinite',
            opacity: 0.4,
          }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="glow-text" style={{
                fontWeight: 900, fontSize: 18, letterSpacing: '-0.03em',
                color: 'var(--green)',
              }}>
                QuantEdge
              </span>
              <span style={{
                fontSize: 9, letterSpacing: '0.15em', textTransform: 'uppercase',
                color: 'var(--muted)', border: '1px solid var(--border2)',
                borderRadius: 4, padding: '2px 6px', fontFamily: 'JetBrains Mono, monospace',
              }}>
                v2.0
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="pulse-green" style={{
                display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
                background: 'var(--green)', boxShadow: '0 0 6px var(--green)',
              }} />
              <span style={{ fontSize: 10, color: 'var(--green)', letterSpacing: '0.1em', fontFamily: 'JetBrains Mono' }}>
                LIVE 24/7
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              onClick={() => navigate('/login')}
              style={{
                fontSize: 12, color: 'var(--muted)', background: 'none', border: 'none',
                cursor: 'pointer', letterSpacing: '0.05em', transition: 'color 0.2s',
              }}
              onMouseEnter={e => (e.currentTarget.style.color = 'var(--text)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'var(--muted)')}
            >
              Sign In
            </button>
            <button
              onClick={() => navigate('/login')}
              className="btn-glow"
              style={{
                padding: '8px 20px', fontSize: 11, fontWeight: 700,
                letterSpacing: '0.1em', textTransform: 'uppercase',
                background: 'linear-gradient(135deg, var(--accent), rgba(245,166,35,0.8))',
                color: '#000', border: 'none', borderRadius: 6,
                cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace',
                boxShadow: '0 0 20px rgba(245,166,35,0.25)',
              }}
            >
              Request Demo
            </button>
          </div>
        </nav>

        {/* ── Hero ── */}
        <section style={{
          padding: '100px 32px 80px',
          textAlign: 'center',
          maxWidth: 900,
          margin: '0 auto',
          position: 'relative',
        }}>
          {/* Corner accent brackets */}
          {[
            { top: 0, left: 0 },
            { top: 0, right: 0 },
            { bottom: 0, left: 0 },
            { bottom: 0, right: 0 },
          ].map((pos, i) => (
            <div key={i} style={{
              position: 'absolute', ...pos, width: 24, height: 24, pointerEvents: 'none',
              borderTop: i < 2 ? '1px solid rgba(245,166,35,0.3)' : 'none',
              borderBottom: i >= 2 ? '1px solid rgba(245,166,35,0.3)' : 'none',
              borderLeft: i % 2 === 0 ? '1px solid rgba(245,166,35,0.3)' : 'none',
              borderRight: i % 2 === 1 ? '1px solid rgba(245,166,35,0.3)' : 'none',
            }} />
          ))}

          {/* Live badge */}
          <div className="reveal" style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '6px 14px',
            border: '1px solid rgba(0,230,118,0.25)',
            background: 'rgba(0,230,118,0.06)',
            borderRadius: 20, marginBottom: 32,
            fontSize: 10, letterSpacing: '0.15em',
            color: 'var(--green)', fontFamily: 'JetBrains Mono, monospace',
          }}>
            <span className="pulse-green" style={{
              width: 6, height: 6, borderRadius: '50%',
              background: 'var(--green)', boxShadow: '0 0 6px var(--green)',
            }} />
            Institutional-Grade Quantitative Trading · 60+ Strategies Running
          </div>

          {/* Main title */}
          <div className="reveal stagger-1">
            <h1 style={{
              fontSize: 'clamp(56px, 8vw, 88px)',
              fontWeight: 900,
              letterSpacing: '-0.04em',
              lineHeight: 0.95,
              marginBottom: 24,
            }}>
              <span style={{
                display: 'block',
                background: 'linear-gradient(135deg, var(--green) 0%, var(--blue) 40%, var(--accent) 80%)',
                backgroundSize: '200% 200%',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                animation: 'gradient-flow 4s ease infinite',
              }}>
                QuantEdge
              </span>
            </h1>
          </div>

          <div className="reveal stagger-2">
            <p style={{
              fontSize: 20, fontWeight: 300, letterSpacing: '0.02em',
              color: 'rgba(226,226,240,0.7)', marginBottom: 12,
            }}>
              The Alpha Machine
            </p>
            <p style={{
              fontSize: 13, color: 'var(--muted)', maxWidth: 480, margin: '0 auto 40px',
              lineHeight: 1.7,
            }}>
              60+ strategies · 7 ML models · 4 brokers · running 24/7 across equities, crypto &amp; prediction markets
            </p>
          </div>

          <div className="reveal stagger-3" style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={() => navigate('/login')}
              className="btn-glow"
              style={{
                padding: '14px 36px', fontSize: 12, fontWeight: 700,
                letterSpacing: '0.12em', textTransform: 'uppercase',
                background: 'linear-gradient(135deg, var(--green), rgba(0,230,118,0.7))',
                color: '#000', border: 'none', borderRadius: 8, cursor: 'pointer',
                fontFamily: 'JetBrains Mono, monospace',
                boxShadow: '0 0 40px rgba(0,230,118,0.2), 0 8px 32px rgba(0,0,0,0.4)',
              }}
            >
              Access Terminal →
            </button>
            <button
              onClick={() => navigate('/login')}
              style={{
                padding: '14px 32px', fontSize: 12, fontWeight: 500,
                letterSpacing: '0.08em',
                background: 'rgba(255,255,255,0.04)',
                color: 'var(--text)', border: '1px solid var(--border2)',
                borderRadius: 8, cursor: 'pointer',
                backdropFilter: 'blur(8px)',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = 'rgba(0,230,118,0.3)'
                e.currentTarget.style.background = 'rgba(0,230,118,0.06)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = 'var(--border2)'
                e.currentTarget.style.background = 'rgba(255,255,255,0.04)'
              }}
            >
              View Live Dashboard
            </button>
          </div>
        </section>

        {/* ── Metrics Strip ── */}
        <section style={{
          borderTop: '1px solid var(--border)',
          borderBottom: '1px solid var(--border)',
          padding: '32px 32px',
          background: 'rgba(14,14,18,0.6)',
          backdropFilter: 'blur(12px)',
          position: 'relative',
          overflow: 'hidden',
        }}>
          <ScanLine />
          <div style={{
            maxWidth: 900, margin: '0 auto',
            display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 0,
          }}>
            {METRICS.map((m, i) => (
              <div key={m.label} className={`reveal stagger-${i + 1}`} style={{
                textAlign: 'center',
                borderRight: i < METRICS.length - 1 ? '1px solid var(--border)' : 'none',
                padding: '0 16px',
              }}>
                <MetricCount m={m} />
                <div style={{ fontSize: 11, color: 'var(--text)', fontWeight: 600, marginTop: 4 }}>{m.label}</div>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2 }}>
                  {m.label === 'Sharpe Ratio' ? 'annualized' :
                   m.label === 'Max Drawdown' ? 'historical' :
                   m.label === 'Win Rate' ? 'across strategies' :
                   m.label === 'Strategies' ? '6 asset classes' :
                   m.label === 'ML Models' ? 'PyTorch + XGBoost' :
                   '365 days/year'}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Feature Grid ── */}
        <section style={{ padding: '80px 32px', maxWidth: 900, margin: '0 auto' }}>
          <div className="reveal" style={{ textAlign: 'center', marginBottom: 56 }}>
            <div className="section-header" style={{ justifyContent: 'center', marginBottom: 16 }}>CAPABILITIES</div>
            <h2 style={{
              fontSize: 'clamp(28px, 4vw, 40px)', fontWeight: 900,
              letterSpacing: '-0.03em',
              background: 'linear-gradient(135deg, var(--blue), var(--purple))',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
            }}>
              Built for Edge
            </h2>
            <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 8 }}>
              Every component is designed to compound returns, not complexity.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
            {FEATURES.map((f, i) => (
              <div
                key={f.title}
                className={`reveal hover-lift scan-card stagger-${(i % 3) + 1}`}
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 10,
                  padding: '20px',
                  transition: 'border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s cubic-bezier(0.16,1,0.3,1)',
                  cursor: 'default',
                  position: 'relative',
                  overflow: 'hidden',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = f.color.replace('var(--', '').replace(')', '') === 'green'
                    ? 'rgba(0,230,118,0.3)'
                    : 'rgba(255,255,255,0.1)'
                  e.currentTarget.style.boxShadow = `0 0 30px rgba(0,0,0,0.3)`
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = 'var(--border)'
                  e.currentTarget.style.boxShadow = 'none'
                }}
              >
                {/* Top accent bar */}
                <div style={{
                  position: 'absolute', top: 0, left: 16, right: 16, height: 2,
                  background: `linear-gradient(90deg, transparent, ${f.color}, transparent)`,
                  opacity: 0.5,
                }} />

                <div style={{ fontSize: 28, marginBottom: 12 }}>{f.icon}</div>
                <h3 style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)', marginBottom: 8, letterSpacing: '0.02em' }}>
                  {f.title}
                </h3>
                <p style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.7 }}>{f.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Comparison Table ── */}
        <section style={{
          padding: '60px 32px',
          background: 'rgba(14,14,18,0.6)',
          borderTop: '1px solid var(--border)',
          borderBottom: '1px solid var(--border)',
          backdropFilter: 'blur(16px)',
          position: 'relative',
          overflow: 'hidden',
        }}>
          <ScanLine />
          <div style={{ maxWidth: 900, margin: '0 auto' }}>
            <div className="reveal" style={{ textAlign: 'center', marginBottom: 40 }}>
              <div className="section-header" style={{ justifyContent: 'center', marginBottom: 16 }}>BENCHMARK ANALYSIS</div>
              <h2 style={{ fontSize: 32, fontWeight: 900, letterSpacing: '-0.03em', color: 'var(--text)' }}>
                How We Compare
              </h2>
              <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 8 }}>
                Risk-adjusted performance vs traditional alternatives.
              </p>
            </div>

            <div className="reveal" style={{ overflowX: 'auto' }}>
              <table className="data-table" style={{ borderCollapse: 'separate', borderSpacing: 0 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left' }}>Metric</th>
                    <th style={{
                      textAlign: 'center',
                      background: 'rgba(245,166,35,0.06)',
                      borderLeft: '1px solid rgba(245,166,35,0.2)',
                      borderRight: '1px solid rgba(245,166,35,0.2)',
                      color: 'var(--accent)',
                    }}>QuantEdge</th>
                    <th style={{ textAlign: 'center' }}>S&amp;P 500</th>
                    <th style={{ textAlign: 'center' }}>Ray Dalio</th>
                    <th style={{ textAlign: 'center' }}>Hedge Fund</th>
                    <th style={{ textAlign: 'center' }}>Typical Bot</th>
                  </tr>
                </thead>
                <tbody>
                  {COMPARISON.map((row, i) => (
                    <tr key={row.metric}>
                      <td style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 500 }}>{row.metric}</td>
                      <td style={{
                        textAlign: 'center', fontWeight: 900, fontSize: 13,
                        color: 'var(--accent)',
                        background: 'rgba(245,166,35,0.04)',
                        borderLeft: '1px solid rgba(245,166,35,0.15)',
                        borderRight: '1px solid rgba(245,166,35,0.15)',
                      }}>
                        {row.qe}
                        {row.win && <span style={{ marginLeft: 4, color: 'var(--green)', fontSize: 9 }}>▲</span>}
                      </td>
                      <td style={{ textAlign: 'center', fontSize: 11, color: 'var(--muted)' }}>{row.spy}</td>
                      <td style={{ textAlign: 'center', fontSize: 11, color: 'var(--muted)' }}>{row.dalio}</td>
                      <td style={{ textAlign: 'center', fontSize: 11, color: 'var(--muted)' }}>{row.hedge}</td>
                      <td style={{ textAlign: 'center', fontSize: 11, color: 'var(--muted)' }}>{row.bot}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p style={{ fontSize: 10, color: 'rgba(107,107,128,0.6)', textAlign: 'center', marginTop: 16 }}>
              * Backtested and paper-trading targets. Past performance does not guarantee future results.
            </p>
          </div>
        </section>

        {/* ── Architecture Highlights ── */}
        <section style={{ padding: '80px 32px', maxWidth: 900, margin: '0 auto' }}>
          <div className="reveal" style={{ textAlign: 'center', marginBottom: 48 }}>
            <div className="section-header" style={{ justifyContent: 'center', marginBottom: 16 }}>ARCHITECTURE</div>
            <h2 style={{
              fontSize: 32, fontWeight: 900, letterSpacing: '-0.03em',
              background: 'linear-gradient(135deg, var(--accent), #ff6b9d)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
            }}>
              Built Different
            </h2>
            <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 8 }}>
              Architecture decisions that protect capital and maximize long-run compound growth.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {[
              { label: 'HMM Regime Detection', desc: '3-state Hidden Markov Model classifies bull/bear/sideways. Each strategy adapts its parameters and position sizing dynamically to current market conditions.', color: 'var(--green)' },
              { label: 'HRP + CVaR Portfolio', desc: 'Hierarchical Risk Parity replaces naive Kelly for portfolio allocation. CVaR tail-risk overlay constrains downside in extreme market scenarios.', color: 'var(--blue)' },
              { label: 'Walk-forward Only', desc: 'Zero in-sample overfitting enforced by architecture. Monte Carlo robustness testing on every strategy. 2-week paper trading gate before live activation.', color: 'var(--accent)' },
              { label: 'Smart Order Routing', desc: 'TWAP/VWAP/LimitFirst/RL execution agent. LimitFirst saves 5–15 bps vs market orders. RL PPO agent learns optimal execution from historical fills.', color: 'var(--purple)' },
            ].map((a, i) => (
              <div
                key={a.label}
                className={`reveal hover-lift stagger-${i + 1}`}
                style={{
                  display: 'flex', gap: 16,
                  background: 'var(--surface)', border: '1px solid var(--border)',
                  borderRadius: 10, padding: '20px',
                  transition: 'all 0.25s cubic-bezier(0.16,1,0.3,1)',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = `rgba(255,255,255,0.1)`
                  e.currentTarget.style.transform = 'translateY(-3px)'
                  e.currentTarget.style.boxShadow = `0 12px 40px rgba(0,0,0,0.3)`
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = 'var(--border)'
                  e.currentTarget.style.transform = 'none'
                  e.currentTarget.style.boxShadow = 'none'
                }}
              >
                <div style={{
                  width: 3, borderRadius: 2, flexShrink: 0,
                  background: `linear-gradient(180deg, ${a.color}, transparent)`,
                }} />
                <div>
                  <h3 style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)', marginBottom: 8 }}>{a.label}</h3>
                  <p style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.7 }}>{a.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Tech Stack ── */}
        <section style={{
          padding: '48px 32px',
          borderTop: '1px solid var(--border)',
          borderBottom: '1px solid var(--border)',
          background: 'rgba(14,14,18,0.4)',
          backdropFilter: 'blur(8px)',
        }}>
          <div style={{ maxWidth: 900, margin: '0 auto', textAlign: 'center' }}>
            <div className="reveal section-header" style={{ justifyContent: 'center', marginBottom: 24 }}>
              TECHNOLOGY STACK
            </div>
            <div className="reveal" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
              {TECH_STACK.map((t, i) => (
                <span
                  key={t.label}
                  className={`badge badge-muted stagger-${(i % 6) + 1}`}
                  style={{
                    transition: 'all 0.2s ease',
                    cursor: 'default',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.borderColor = t.color
                    e.currentTarget.style.color = t.color
                    e.currentTarget.style.background = `${t.color}15`
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = ''
                    e.currentTarget.style.color = ''
                    e.currentTarget.style.background = ''
                  }}
                >
                  {t.label}
                </span>
              ))}
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section style={{ padding: '100px 32px', textAlign: 'center', maxWidth: 600, margin: '0 auto', position: 'relative' }}>
          <div className="reveal stagger-1">
            <div className="section-header" style={{ justifyContent: 'center', marginBottom: 20 }}>GET STARTED</div>
            <h2 style={{
              fontSize: 'clamp(32px, 5vw, 48px)', fontWeight: 900,
              letterSpacing: '-0.03em', lineHeight: 1.1, marginBottom: 16,
            }}>
              Ready to trade with{' '}
              <span style={{
                background: 'linear-gradient(135deg, var(--green), var(--blue))',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
              }}>
                institutional edge?
              </span>
            </h2>
            <p style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 36, lineHeight: 1.7 }}>
              Request a demo to see live paper-trading performance, backtests, and the full strategy dashboard.
            </p>
          </div>

          <div className="reveal stagger-2">
            <button
              onClick={() => navigate('/login')}
              className="btn-glow"
              style={{
                padding: '16px 48px', fontSize: 13, fontWeight: 900,
                letterSpacing: '0.12em', textTransform: 'uppercase',
                background: 'linear-gradient(135deg, var(--green), rgba(0,230,118,0.8))',
                color: '#000', border: 'none', borderRadius: 8, cursor: 'pointer',
                fontFamily: 'JetBrains Mono, monospace',
                boxShadow: '0 0 60px rgba(0,230,118,0.2), 0 8px 40px rgba(0,0,0,0.5)',
                display: 'block', width: '100%', marginBottom: 12,
              }}
            >
              Request Demo →
            </button>
            <p style={{ fontSize: 10, color: 'rgba(107,107,128,0.5)', marginTop: 12 }}>
              No commitment · Paper trading demo available instantly
            </p>
          </div>
        </section>

        {/* ── Footer ── */}
        <footer style={{
          borderTop: '1px solid var(--border)',
          padding: '24px 32px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="glow-text" style={{ fontWeight: 900, fontSize: 14, color: 'var(--green)' }}>QE</span>
            <span style={{ fontSize: 10, color: 'var(--muted)' }}>QuantEdge © 2025</span>
          </div>
          <div style={{ display: 'flex', gap: 20 }}>
            {['Privacy', 'Terms', 'Risk Disclosure'].map(label => (
              <span key={label} style={{ fontSize: 10, color: 'rgba(107,107,128,0.5)', cursor: 'pointer' }}>{label}</span>
            ))}
          </div>
          <p style={{ fontSize: 10, color: 'rgba(107,107,128,0.4)', maxWidth: 400 }}>
            Trading involves significant financial risk. Past performance does not guarantee future results. For demonstration only.
          </p>
        </footer>

      </div>
    </div>
  )
}
