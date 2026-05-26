import { NavLink } from 'react-router-dom'
import { LayoutDashboard, TrendingUp, Bitcoin, BarChart2, FlaskConical, Beaker, LineChart, Shield, Activity, Trophy, DollarSign, Archive, Monitor, Layers, Globe, Coins } from 'lucide-react'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/equity', icon: TrendingUp, label: 'Equity' },
  { to: '/crypto', icon: Bitcoin, label: 'Crypto' },
  { to: '/comparison', icon: BarChart2, label: 'Comparison' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtest' },
  { to: '/experiments', icon: Beaker, label: 'Experiments' },
  { to: '/analytics', icon: LineChart, label: 'Analytics' },
  { to: '/risk', icon: Shield, label: 'Risk' },
  { to: '/activity', icon: Activity, label: 'Activity' },
  { to: '/leaderboard', icon: Trophy, label: 'Leaderboard' },
  { to: '/pnl', icon: DollarSign, label: 'P&L' },
  { to: '/archive', icon: Archive, label: 'Archive' },
  { to: '/system', icon: Monitor, label: 'System Monitor' },
  { to: '/options', icon: Layers, label: 'Options & Macro' },
  { to: '/macro', icon: Globe, label: 'Macro & Sentiment' },
  { to: '/polymarket', icon: Coins, label: 'Polymarket' },
]

export default function Sidebar() {
  return (
    <aside className="w-16 bg-[#111111] border-r border-[#1e1e1e] flex flex-col items-center py-4 gap-1">
      <div className="text-[#f5a623] font-bold text-lg mb-6">Q</div>
      {NAV.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            `w-10 h-10 flex items-center justify-center rounded-lg transition-colors ${
              isActive ? 'bg-[#f5a623]/20 text-[#f5a623]' : 'text-[#888888] hover:text-[#e8e8e8] hover:bg-[#1e1e1e]'
            }`
          }
          title={label}
        >
          <Icon size={18} />
        </NavLink>
      ))}
    </aside>
  )
}
