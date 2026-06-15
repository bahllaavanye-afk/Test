import {
  LayoutDashboard, TrendingUp, Bitcoin, BarChart2, FlaskConical, Beaker,
  LineChart, Shield, Activity, Trophy, DollarSign, Archive, Monitor, Layers,
  Globe, Coins, CandlestickChart, BrainCircuit, GitBranch, PackageCheck, Bot,
  Users, ScanSearch, Rocket, Building2, Terminal, BarChart3, ListCollapse,
  ClipboardList, Copy, PieChart, ShieldAlert, Percent,
  type LucideIcon,
} from 'lucide-react'

export interface NavItem {
  to: string
  icon: LucideIcon
  label: string
  /** Short label for the mobile bottom bar. */
  short?: string
}

export interface NavGroup {
  emoji: string
  label: string
  items: NavItem[]
}

/** Full navigation — used by the desktop sidebar and the mobile "More" sheet. */
export const NAV: NavItem[] = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard', short: 'Home' },
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
  { to: '/bot-dashboard', icon: BarChart3, label: 'Bot Dashboard' },
  { to: '/positions', icon: ListCollapse, label: 'Positions Hub' },
  { to: '/tasks', icon: ClipboardList, label: 'Task Manager' },
  { to: '/copy-trading', icon: Copy, label: 'Copy Trading' },
  { to: '/attribution', icon: PieChart, label: 'Attribution' },
  { to: '/risk-controls', icon: ShieldAlert, label: 'Risk Controls' },
  { to: '/funding-rates', icon: Percent, label: 'Funding Rates' },
]

/** Grouped navigation for the sidebar. */
export const NAV_GROUPS: NavGroup[] = [
  {
    emoji: '🤖',
    label: 'BOTS',
    items: [
      { to: '/bot-dashboard', icon: BarChart3, label: 'Bot Dashboard' },
      { to: '/bots', icon: Bot, label: 'Bot Builder' },
      { to: '/bot-desk', icon: Building2, label: 'Bot Desk' },
      { to: '/agents', icon: Users, label: 'Agent Command' },
      { to: '/agent-logs', icon: Terminal, label: 'Agent Logs' },
      { to: '/scanners', icon: ScanSearch, label: 'Scanners' },
      { to: '/tasks', icon: ClipboardList, label: 'Task Manager' },
    ],
  },
  {
    emoji: '📊',
    label: 'MARKETS',
    items: [
      { to: '/dashboard', icon: LayoutDashboard, label: 'Overview', short: 'Home' },
      { to: '/equity', icon: TrendingUp, label: 'Equity Trading', short: 'Equity' },
      { to: '/crypto', icon: Bitcoin, label: 'Crypto Trading', short: 'Crypto' },
      { to: '/options-chain', icon: CandlestickChart, label: 'Options Chain' },
      { to: '/macro', icon: Globe, label: 'Macro & Sentiment' },
      { to: '/polymarket', icon: Coins, label: 'Polymarket' },
      { to: '/positions', icon: ListCollapse, label: 'Positions Hub' },
    ],
  },
  {
    emoji: '📈',
    label: 'ANALYTICS',
    items: [
      { to: '/analytics', icon: LineChart, label: 'Analytics', short: 'Stats' },
      { to: '/backtest', icon: FlaskConical, label: 'Backtest Lab' },
      { to: '/comparison', icon: BarChart2, label: 'Comparison' },
      { to: '/leaderboard', icon: Trophy, label: 'Leaderboard' },
      { to: '/pnl', icon: DollarSign, label: 'P&L' },
      { to: '/attribution', icon: PieChart, label: 'Attribution' },
      { to: '/ml-insights', icon: BrainCircuit, label: 'ML Insights' },
    ],
  },
  {
    emoji: '⚠️',
    label: 'RISK',
    items: [
      { to: '/risk-controls', icon: ShieldAlert, label: 'Risk Controls' },
      { to: '/risk', icon: Shield, label: 'Risk Manager' },
      { to: '/funding-rates', icon: Percent, label: 'Funding Rates' },
    ],
  },
  {
    emoji: '⚙️',
    label: 'SETTINGS',
    items: [
      { to: '/system', icon: Monitor, label: 'System Monitor' },
      { to: '/pipeline', icon: GitBranch, label: 'Pipeline' },
      { to: '/releases', icon: PackageCheck, label: 'Model Releases' },
    ],
  },
]

/** Primary destinations shown directly in the mobile bottom tab bar. */
export const PRIMARY_NAV: NavItem[] = [
  { to: '/bot-dashboard', icon: BarChart3, label: 'Bots', short: 'Bots' },
  { to: '/dashboard', icon: LayoutDashboard, label: 'Markets', short: 'Markets' },
  { to: '/analytics', icon: LineChart, label: 'Analytics', short: 'Stats' },
  { to: '/risk-controls', icon: ShieldAlert, label: 'Risk', short: 'Risk' },
]
