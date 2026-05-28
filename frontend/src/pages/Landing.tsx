import { useNavigate } from 'react-router-dom'
import { GradientText } from '../components/ui/GradientText'
import { AnimatedCounter } from '../components/ui/AnimatedCounter'
import { LiveIndicator } from '../components/ui/LiveIndicator'
import '../styles/animations.css'

const METRICS = [
  { label: 'Sharpe Ratio', value: '>2.0', sub: 'annualized', numeric: 2.0, decimals: 1, prefix: '' },
  { label: 'Max Drawdown', value: '<15%', sub: 'historical', numeric: 15, decimals: 0, prefix: '<', suffix: '%' },
  { label: 'Win Rate', value: '~68%', sub: 'across strategies', numeric: 68, decimals: 0, prefix: '~', suffix: '%' },
  { label: 'Uptime', value: '24/7', sub: '365 days', numeric: null, prefix: '' },
  { label: 'Strategies', value: '48', sub: '41 manual + 7 ML', numeric: 48, decimals: 0, prefix: '' },
  { label: 'ML Models', value: '7', sub: 'PyTorch + XGBoost', numeric: 7, decimals: 0, prefix: '' },
]

const FEATURES = [
  {
    title: 'Multi-Broker Execution',
    desc: 'Simultaneous execution across Alpaca, TradeStation, Binance, and Polymarket. Smart order routing with TWAP/VWAP slippage minimization.',
    icon: '⚡',
  },
  {
    title: 'PyTorch ML Suite',
    desc: '5 production models: LSTM trend, XGBoost classifier, regime detector, sentiment NLP, and volatility forecaster. Walk-forward validated.',
    icon: '🧠',
  },
  {
    title: 'Real-time Risk Engine',
    desc: 'Kelly criterion position sizing, correlation kill-switch, per-strategy circuit breakers, and global 10% drawdown halt.',
    icon: '🛡',
  },
  {
    title: 'Options Flow Scanner',
    desc: 'Unusual options activity detection, dark pool prints, and put/call ratio signals integrated into directional strategies.',
    icon: '🔭',
  },
  {
    title: 'FOMC Macro Calendar',
    desc: 'Fed meeting, CPI, NFP, and earnings event risk management. Automatic position sizing reduction into high-impact events.',
    icon: '📅',
  },
  {
    title: 'Walk-forward Backtesting',
    desc: 'Zero in-sample overfitting policy enforced by architecture. Every strategy must pass 2-week paper trading before live activation.',
    icon: '📊',
  },
]

const COMPARISON = [
  { metric: 'Sharpe Ratio', quantedge: '2.1+', spy: '0.8', dalio: '~1.0', hedge: '~1.2', bot: '~0.5', highlight: true },
  { metric: 'Annual Return', quantedge: '35–55%', spy: '~12%', dalio: '~12%', hedge: '~18%', bot: '~8%', highlight: false },
  { metric: 'Max Drawdown', quantedge: '<15%', spy: '~34%', dalio: '~20%', hedge: '~25%', bot: '~40%', highlight: false },
  { metric: 'ML-Enhanced', quantedge: '✓', spy: '✗', dalio: 'Partial', hedge: 'Partial', bot: '✗', highlight: false },
  { metric: 'Always-On', quantedge: '✓', spy: 'Market hrs', dalio: 'N/A', hedge: 'N/A', bot: 'Varies', highlight: false },
  { metric: 'Multi-Asset', quantedge: '✓', spy: '✗', dalio: '✓', hedge: 'Partial', bot: '✗', highlight: false },
]

const ARCH_HIGHLIGHTS = [
  {
    label: 'Regime Detection',
    desc: 'Hidden Markov Model classifies bull/bear/sideways market regimes. Each strategy adapts parameters dynamically to current conditions.',
  },
  {
    label: 'Correlation Kill-switch',
    desc: 'Max 30% capital per correlated cluster. Prevents factor blow-up during systemic events like 2020 COVID crash or 2022 rate shock.',
  },
  {
    label: 'Kelly Criterion Sizing',
    desc: '25% fractional Kelly across all strategies. Mathematically optimal position sizing for maximum long-run compound growth.',
  },
  {
    label: 'TWAP/VWAP Routing',
    desc: 'LimitFirst policy saves 5–15 bps vs market orders on average. Iceberg execution for block trades to minimize market impact.',
  },
]

const TECH_STACK = [
  'Python / FastAPI',
  'React 18 + TypeScript',
  'PyTorch',
  'Redis (Upstash)',
  'Supabase / PostgreSQL',
  'Alpaca Markets',
  'Binance',
  'Polymarket CLOB',
  'Docker',
  'TanStack Query',
]

export default function Landing() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-[#0a0d12] text-white relative overflow-x-hidden">
      {/* Animated grid background */}
      <div className="fixed inset-0 pointer-events-none animated-grid-bg z-0" aria-hidden="true" />

      {/* Floating particle decorations */}
      <div className="fixed top-1/4 left-1/4 w-64 h-64 rounded-full pointer-events-none z-0 opacity-5 animate-particle-float"
        style={{ background: 'radial-gradient(circle, #00ff88 0%, transparent 70%)', animationDelay: '0s' }}
        aria-hidden="true"
      />
      <div className="fixed top-2/3 right-1/4 w-96 h-96 rounded-full pointer-events-none z-0 opacity-5 animate-particle-float"
        style={{ background: 'radial-gradient(circle, #00d4ff 0%, transparent 70%)', animationDelay: '2s' }}
        aria-hidden="true"
      />
      <div className="fixed top-1/2 right-1/3 w-48 h-48 rounded-full pointer-events-none z-0 opacity-4 animate-particle-float"
        style={{ background: 'radial-gradient(circle, #6366f1 0%, transparent 70%)', animationDelay: '4s' }}
        aria-hidden="true"
      />

      <div className="relative z-10">
      {/* Sticky nav */}
      <nav className="border-b border-white/[0.06] px-8 py-4 flex items-center justify-between sticky top-0 glass-panel z-20">
        <div className="flex items-center gap-3">
          <span className="font-black text-xl tracking-tight">
            <GradientText gradient="green-blue">QuantEdge</GradientText>
          </span>
          <span className="text-[10px] text-[#555555] border border-[#1e1e1e] rounded px-1.5 py-0.5 font-mono">BETA</span>
          <LiveIndicator label="LIVE" color="#00ff88" />
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/login')}
            className="text-sm text-[#888888] hover:text-white transition-colors"
          >
            Sign In
          </button>
          <button
            onClick={() => navigate('/login')}
            className="px-4 py-2 text-sm font-bold rounded-lg transition-all hover:scale-105"
            style={{
              background: 'linear-gradient(135deg, #00ff88, #00d4ff)',
              color: '#000',
              boxShadow: '0 0 20px rgba(0,255,136,0.25)',
            }}
          >
            Request Demo
          </button>
        </div>
      </nav>

      {/* Hero */}
      <section className="px-8 pt-24 pb-20 text-center max-w-5xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 text-xs border border-[#00ff88]/30 bg-[#00ff88]/10 text-[#00ff88] rounded-full mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00ff88] animate-pulse" />
          Live 24/7 · Institutional-Grade Quant Trading
        </div>
        <h1 className="text-7xl font-black tracking-tight mb-5 leading-none">
          <GradientText gradient="green-blue">QuantEdge</GradientText>
        </h1>
        <p className="text-2xl text-[#aaaaaa] font-light mb-4">
          Institutional-Grade Quantitative Trading Platform
        </p>
        <p className="text-base text-[#666666] max-w-xl mx-auto mb-12">
          9 strategies + 5 ML models running 24/7 across equities, crypto &amp; prediction markets
        </p>
        <div className="flex items-center justify-center gap-4 flex-wrap">
          <button
            onClick={() => navigate('/login')}
            className="px-8 py-3.5 font-bold rounded-xl transition-all hover:scale-105 text-sm text-black"
            style={{
              background: 'linear-gradient(135deg, #00ff88, #00d4ff)',
              boxShadow: '0 0 30px rgba(0,255,136,0.3)',
            }}
          >
            Request Demo →
          </button>
          <button
            onClick={() => navigate('/login')}
            className="px-8 py-3.5 border border-white/[0.12] text-[#aaaaaa] font-medium rounded-xl hover:border-[#00ff88]/40 hover:text-white transition-colors text-sm backdrop-blur-sm"
          >
            View Live Dashboard
          </button>
        </div>
      </section>

      {/* Key metrics bar */}
      <section className="border-y border-white/[0.06] py-8 relative">
        <div
          className="absolute inset-0 animate-gradient opacity-30"
          style={{
            backgroundImage: 'linear-gradient(135deg, rgba(0,255,136,0.05), rgba(0,212,255,0.05), rgba(99,102,241,0.05))',
            backgroundSize: '300% 300%',
          }}
        />
        <div className="max-w-5xl mx-auto px-8 relative">
          <div className="grid grid-cols-3 md:grid-cols-6 gap-6 text-center">
            {METRICS.map((m, i) => (
              <div key={m.label} className={`${i < METRICS.length - 1 ? 'md:border-r border-white/[0.06]' : ''} pr-6`}>
                <div className="text-2xl font-black">
                  {m.numeric !== null ? (
                    <AnimatedCounter
                      value={m.numeric as number}
                      prefix={m.prefix}
                      suffix={'suffix' in m ? (m.suffix as string) : ''}
                      decimals={'decimals' in m ? (m.decimals as number) : 0}
                      duration={1500}
                    />
                  ) : (
                    <span className="text-[#00ff88]">{m.value}</span>
                  )}
                </div>
                <div className="text-xs font-semibold text-white mt-1">{m.label}</div>
                <div className="text-[10px] text-[#555555] mt-0.5">{m.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Feature grid 3×2 */}
      <section className="py-24 px-8 max-w-5xl mx-auto">
        <div className="text-center mb-14">
          <h2 className="text-3xl font-black mb-3">
            <GradientText gradient="blue-purple">Built for Edge</GradientText>
          </h2>
          <p className="text-sm text-[#666666]">Every component is designed to compound returns, not complexity.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map(f => (
            <div
              key={f.title}
              className="glass-card p-5 hover:scale-[1.02] transition-all duration-300 cursor-default group hover:shadow-[0_0_24px_rgba(0,255,136,0.08)]"
            >
              <div className="text-3xl mb-4">{f.icon}</div>
              <h3 className="text-sm font-bold text-white mb-2 group-hover:text-[#00ff88] transition-colors">
                {f.title}
              </h3>
              <p className="text-xs text-[#666666] leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Comparison table */}
      <section className="py-20 px-8 border-y border-white/[0.06] relative">
        <div className="absolute inset-0 glass-panel opacity-60 pointer-events-none" aria-hidden="true" />
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12 relative">
            <h2 className="text-3xl font-black mb-3">How We Compare</h2>
            <p className="text-sm text-[#666666]">Risk-adjusted performance vs traditional alternatives.</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-separate border-spacing-0">
              <thead>
                <tr className="border-b border-[#1e1e1e]">
                  <th className="text-left py-3 pr-4 text-[#888888] font-medium text-xs w-36">Metric</th>
                  <th className="py-3 px-4 text-center bg-[#f5a623]/5 border border-[#f5a623]/20 rounded-t-lg font-bold text-[#f5a623] text-xs">
                    QuantEdge
                  </th>
                  <th className="py-3 px-4 text-center text-[#888888] font-medium text-xs">S&amp;P 500</th>
                  <th className="py-3 px-4 text-center text-[#888888] font-medium text-xs">Ray Dalio</th>
                  <th className="py-3 px-4 text-center text-[#888888] font-medium text-xs">Top Hedge Fund</th>
                  <th className="py-3 px-4 text-center text-[#888888] font-medium text-xs">Typical Bot</th>
                </tr>
              </thead>
              <tbody>
                {COMPARISON.map((row, i) => (
                  <tr key={row.metric} className={`border-b border-[#1e1e1e] ${i % 2 === 0 ? '' : 'bg-[#0a0a0a]/40'}`}>
                    <td className="py-3 pr-4 text-xs text-[#aaaaaa] font-medium">{row.metric}</td>
                    <td className="py-3 px-4 text-center font-black text-[#f5a623] text-sm bg-[#f5a623]/5">
                      {row.quantedge}
                    </td>
                    <td className="py-3 px-4 text-center text-[#666666] text-xs">{row.spy}</td>
                    <td className="py-3 px-4 text-center text-[#666666] text-xs">{row.dalio}</td>
                    <td className="py-3 px-4 text-center text-[#666666] text-xs">{row.hedge}</td>
                    <td className="py-3 px-4 text-center text-[#666666] text-xs">{row.bot}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-[#444444] text-center mt-5">
            * QuantEdge figures are backtested and paper-trading targets. Past performance does not guarantee future results.
          </p>
        </div>
      </section>

      {/* Built different — architecture highlights */}
      <section className="py-24 px-8 max-w-5xl mx-auto">
        <div className="text-center mb-14">
          <h2 className="text-3xl font-black mb-3">
            <GradientText gradient="amber-pink">Built Different</GradientText>
          </h2>
          <p className="text-sm text-[#666666]">Architecture decisions that protect capital and maximize long-run compound growth.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {ARCH_HIGHLIGHTS.map(a => (
            <div key={a.label} className="glass-card flex gap-4 p-5 hover:scale-[1.01] transition-all duration-300 group hover:shadow-[0_0_24px_rgba(0,212,255,0.08)]">
              <div className="w-1 rounded-full bg-gradient-to-b from-[#00ff88] to-[#00d4ff] shrink-0 self-stretch" />
              <div>
                <h3 className="text-sm font-bold text-white mb-1.5 group-hover:text-[#00d4ff] transition-colors">{a.label}</h3>
                <p className="text-xs text-[#666666] leading-relaxed">{a.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Tech stack section */}
      <section className="py-14 px-8 border-y border-white/[0.06] relative">
        <div className="absolute inset-0 glass-panel opacity-50 pointer-events-none" aria-hidden="true" />
        <div className="max-w-5xl mx-auto text-center relative">
          <h3 className="text-xs font-bold text-[#555555] uppercase tracking-widest mb-8">Technology Stack</h3>
          <div className="flex flex-wrap items-center justify-center gap-3">
            {TECH_STACK.map(t => (
              <span
                key={t}
                className="px-3 py-1.5 border border-white/[0.08] rounded-full text-xs text-[#666666] hover:border-[#00ff88]/30 hover:text-[#aaaaaa] transition-all duration-200 backdrop-blur-sm cursor-default"
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-28 px-8 text-center max-w-2xl mx-auto">
        <h2 className="text-4xl font-black mb-4 leading-tight">
          Ready to trade with{' '}
          <GradientText gradient="green-blue">institutional edge?</GradientText>
        </h2>
        <p className="text-sm text-[#666666] mb-10 max-w-md mx-auto leading-relaxed">
          Request a demo to see live paper-trading performance, backtests, and the full strategy dashboard.
        </p>
        <button
          onClick={() => navigate('/login')}
          className="px-12 py-4 font-black rounded-xl transition-all hover:scale-105 text-base text-black"
          style={{
            background: 'linear-gradient(135deg, #00ff88, #00d4ff)',
            boxShadow: '0 0 40px rgba(0,255,136,0.25)',
          }}
        >
          Request Demo →
        </button>
        <p className="text-xs text-[#333333] mt-6">No commitment. Paper trading demo available instantly.</p>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[0.06] py-8 px-8 text-center">
        <div className="flex items-center justify-center gap-6 text-[10px] text-[#333333] mb-3">
          <span>Privacy Policy</span>
          <span>Terms of Service</span>
          <span>Risk Disclosure</span>
        </div>
        <p className="text-[10px] text-[#2a2a2a]">
          QuantEdge © 2025 — Quantitative trading involves significant financial risk. Past performance does not guarantee future results. For demonstration purposes only.
        </p>
      </footer>
      </div>
    </div>
  )
}
