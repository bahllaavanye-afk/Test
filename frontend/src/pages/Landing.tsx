import { useNavigate } from 'react-router-dom'

const METRICS = [
  { label: 'Sharpe Ratio', value: '>2.0', sub: 'annualized' },
  { label: 'Max Drawdown', value: '<15%', sub: 'historical' },
  { label: 'Win Rate', value: '~68%', sub: 'across strategies' },
  { label: 'Uptime', value: '24/7', sub: '365 days' },
  { label: 'Strategies', value: '9', sub: 'live + paper' },
  { label: 'ML Models', value: '5', sub: 'PyTorch' },
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
    <div className="min-h-screen bg-[#0a0a0a] text-white">

      {/* Sticky nav */}
      <nav className="border-b border-[#1e1e1e] px-8 py-4 flex items-center justify-between sticky top-0 bg-[#0a0a0a]/95 backdrop-blur-sm z-10">
        <div className="flex items-center gap-2">
          <span
            className="font-black text-xl tracking-tight text-transparent bg-clip-text"
            style={{ backgroundImage: 'linear-gradient(135deg, #f5a623, #ffcc70)' }}
          >
            QuantEdge
          </span>
          <span className="text-[10px] text-[#555555] border border-[#1e1e1e] rounded px-1.5 py-0.5 font-mono">BETA</span>
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
            className="px-4 py-2 text-sm bg-[#f5a623] text-black font-bold rounded-lg hover:bg-[#e8971a] transition-colors"
          >
            Request Demo
          </button>
        </div>
      </nav>

      {/* Hero */}
      <section className="px-8 pt-24 pb-20 text-center max-w-5xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 text-xs border border-[#f5a623]/30 bg-[#f5a623]/10 text-[#f5a623] rounded-full mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-[#f5a623] animate-pulse" />
          Live 24/7 · Institutional-Grade Quant Trading
        </div>
        <h1 className="text-7xl font-black tracking-tight mb-5 leading-none">
          <span
            className="text-transparent bg-clip-text"
            style={{ backgroundImage: 'linear-gradient(135deg, #f5a623 0%, #ffcc70 50%, #f5a623 100%)' }}
          >
            QuantEdge
          </span>
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
            className="px-8 py-3.5 bg-[#f5a623] text-black font-bold rounded-xl hover:bg-[#e8971a] transition-all hover:scale-105 text-sm shadow-[0_0_30px_rgba(245,166,35,0.3)]"
          >
            Request Demo →
          </button>
          <button
            onClick={() => navigate('/login')}
            className="px-8 py-3.5 border border-[#333333] text-[#aaaaaa] font-medium rounded-xl hover:border-[#f5a623]/50 hover:text-white transition-colors text-sm"
          >
            View Live Dashboard
          </button>
        </div>
      </section>

      {/* Key metrics bar */}
      <section className="border-y border-[#1e1e1e] bg-[#111111] py-8">
        <div className="max-w-5xl mx-auto px-8">
          <div className="grid grid-cols-3 md:grid-cols-6 gap-6 text-center">
            {METRICS.map((m, i) => (
              <div key={m.label} className={`${i < METRICS.length - 1 ? 'md:border-r border-[#1e1e1e]' : ''} pr-6`}>
                <div className="text-2xl font-black text-[#f5a623]">{m.value}</div>
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
          <h2 className="text-3xl font-black mb-3">Built for Edge</h2>
          <p className="text-sm text-[#666666]">Every component is designed to compound returns, not complexity.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map(f => (
            <div
              key={f.title}
              className="bg-[#111111] border border-[#1e1e1e] rounded-xl p-5 hover:border-[#f5a623]/30 transition-all hover:shadow-[0_0_20px_rgba(245,166,35,0.05)] group"
            >
              <div className="text-3xl mb-4">{f.icon}</div>
              <h3 className="text-sm font-bold text-white mb-2 group-hover:text-[#f5a623] transition-colors">
                {f.title}
              </h3>
              <p className="text-xs text-[#666666] leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Comparison table */}
      <section className="py-20 px-8 bg-[#111111] border-y border-[#1e1e1e]">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
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
          <h2 className="text-3xl font-black mb-3">Built Different</h2>
          <p className="text-sm text-[#666666]">Architecture decisions that protect capital and maximize long-run compound growth.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {ARCH_HIGHLIGHTS.map(a => (
            <div key={a.label} className="flex gap-4 bg-[#111111] border border-[#1e1e1e] rounded-xl p-5 hover:border-[#f5a623]/20 transition-colors group">
              <div className="w-1 rounded-full bg-gradient-to-b from-[#f5a623] to-[#f5a623]/30 shrink-0 self-stretch" />
              <div>
                <h3 className="text-sm font-bold text-white mb-1.5 group-hover:text-[#f5a623] transition-colors">{a.label}</h3>
                <p className="text-xs text-[#666666] leading-relaxed">{a.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Tech stack section */}
      <section className="py-14 px-8 border-y border-[#1e1e1e] bg-[#111111]">
        <div className="max-w-5xl mx-auto text-center">
          <h3 className="text-xs font-bold text-[#555555] uppercase tracking-widest mb-8">Technology Stack</h3>
          <div className="flex flex-wrap items-center justify-center gap-3">
            {TECH_STACK.map(t => (
              <span
                key={t}
                className="px-3 py-1.5 border border-[#1e1e1e] rounded-full text-xs text-[#666666] hover:border-[#f5a623]/30 hover:text-[#aaaaaa] transition-colors"
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
          <span
            className="text-transparent bg-clip-text"
            style={{ backgroundImage: 'linear-gradient(135deg, #f5a623, #ffcc70)' }}
          >
            institutional edge?
          </span>
        </h2>
        <p className="text-sm text-[#666666] mb-10 max-w-md mx-auto leading-relaxed">
          Request a demo to see live paper-trading performance, backtests, and the full strategy dashboard.
        </p>
        <button
          onClick={() => navigate('/login')}
          className="px-12 py-4 bg-[#f5a623] text-black font-black rounded-xl hover:bg-[#e8971a] transition-all hover:scale-105 text-base shadow-[0_0_40px_rgba(245,166,35,0.25)]"
        >
          Request Demo →
        </button>
        <p className="text-xs text-[#333333] mt-6">No commitment. Paper trading demo available instantly.</p>
      </section>

      {/* Footer */}
      <footer className="border-t border-[#1e1e1e] py-8 px-8 text-center">
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
  )
}
