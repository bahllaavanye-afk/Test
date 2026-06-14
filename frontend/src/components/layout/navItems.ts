import {
  LayoutDashboard, TrendingUp, Bitcoin, BarChart2, FlaskConical, Beaker,
  LineChart, Shield, Activity, Trophy, DollarSign, Archive, Monitor, Layers,
  Globe, Coins, CandlestickChart, BrainCircuit, GitBranch, PackageCheck, Bot,
  Users, ScanSearch, Rocket, Building2, Terminal, type LucideIcon,
} from 'lucide-react'

export interface NavItem {
  to: string
  icon: LucideIcon
  label: string
  /** Short label for the mobile bottom bar. */
  short?: string
}

/** Full navigation — used by the desktop sidebar and the mobile "More" sheet. */
export const NAV: NavItem[] = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard', short: 'Home' },
  { to: '/equity', icon: TrendingUp, label: 'Equity', short: 'Equity' },
  { to: '/crypto', icon: Bitcoin, label: 'Crypto', short: 'Crypto' },
  { to: '/comparison', icon: BarChart2, label: 'Comparison' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtest' },
  { to: '/experiments', icon: Beaker, label: 'Experiments' },
  { to: '/ml-insights', icon: BrainCircuit, label: 'ML Insights' },
  { to: '/bots', icon: Bot, label: 'Bot Builder' },
  { to: '/bot-desk', icon: Building2, label: 'Bot Desk' },
  { to: '/agents', icon: Users, label: 'Agent Command' },
  { to: '/scanners', icon: ScanSearch, label: 'Scanners' },
  { to: '/analytics', icon: LineChart, label: 'Analytics', short: 'Stats' },
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
  { to: '/promotions', icon: Rocket, label: 'Strategy Promotions' },
  { to: '/releases', icon: PackageCheck, label: 'Model Releases' },
  { to: '/agent-logs', icon: Terminal, label: 'Agent Logs' },
]

/** Primary destinations shown directly in the mobile bottom tab bar. */
export const PRIMARY_NAV: NavItem[] = [
  NAV[0], // Dashboard
  NAV[1], // Equity
  NAV[2], // Crypto
  NAV[10], // Analytics
]
