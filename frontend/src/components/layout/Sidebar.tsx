import { NavLink } from 'react-router-dom'
import { LayoutDashboard, TrendingUp, Bitcoin, BarChart2, FlaskConical, Beaker, LineChart, Shield, Activity, Trophy, DollarSign, Archive, Monitor, Layers, Globe, Coins, CandlestickChart, BrainCircuit, GitBranch, PackageCheck, Bot, Users } from 'lucide-react'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/equity', icon: TrendingUp, label: 'Equity' },
  { to: '/crypto', icon: Bitcoin, label: 'Crypto' },
  { to: '/comparison', icon: BarChart2, label: 'Comparison' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtest' },
  { to: '/experiments', icon: Beaker, label: 'Experiments' },
  { to: '/ml-insights', icon: BrainCircuit, label: 'ML Insights' },
  { to: '/bots', icon: Bot, label: 'Bot Builder' },
  { to: '/agents', icon: Users, label: 'Agent Command' },
  { to: '/analytics', icon: LineChart, label: 'Analytics' },
  { to: '/risk', icon: Shield, label: 'Risk' },
  { to: '/activity', icon: Activity, label: 'Activity' },
  { to: '/leaderboard', icon: Trophy, label: 'Leaderboard' },
  { to: '/pnl', icon: DollarSign, label: 'P&L' },
  { to: '/archive', icon: Archive, label: 'Archive' },
  { to: '/system', icon: Monitor, label: 'System Monitor' },
  { to: '/options-chain', icon: CandlestickChart, label: 'Options Chain' },
  { to: '/options', icon: Layers, label: 'Options & Macro' },
  { to: '/macro', icon: Globe, label: 'Macro & Sentiment' },
  { to: '/polymarket', icon: Coins, label: 'Polymarket' },
  { to: '/pipeline', icon: GitBranch, label: 'Pipeline' },
  { to: '/releases', icon: PackageCheck, label: 'Model Releases' },
]

export default function Sidebar() {
  return (
    <aside className="w-16 glass-panel border-r border-white/[0.06] flex flex-col items-center py-4 gap-1 relative z-10 overflow-visible">
      <div
        className="font-bold text-lg mb-6 text-transparent bg-clip-text"
        style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
      >
        Q
      </div>
      {NAV.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            `sidebar-nav-item w-10 h-10 flex items-center justify-center rounded-lg transition-all duration-200 ${
              isActive
                ? 'bg-[#00ff88]/10 text-[#00ff88] shadow-[0_0_12px_rgba(0,255,136,0.20)]'
                : 'text-[#888888] hover:text-[#e8e8e8] hover:bg-white/[0.06] hover:shadow-[0_0_8px_rgba(0,212,255,0.10)]'
            }`
          }
        >
          <Icon size={18} />
          <span className="sidebar-tooltip">{label}</span>
        </NavLink>
      ))}
    </aside>
  )
}
