import {
  LayoutDashboard, TrendingUp, Bitcoin, BarChart2, FlaskConical, Beaker,
  LineChart, Shield, Activity, Trophy, DollarSign, Archive, Monitor, Layers,
  Globe, Coins, CandlestickChart, BrainCircuit, GitBranch, PackageCheck, Bot,
  Users, ScanSearch, Rocket, Building2, Terminal, BarChart3, ListCollapse,
  ClipboardList, Copy, PieChart, ShieldAlert, Percent, Settings,
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
  label: string
  items: NavItem[]
}

/** Grouped navigation — used by sidebar and mobile "More" sheet. */
export const NAV_GROUPS: NavGroup[] = [
  {
    label: 'BOTS',
    items: [
      { to: '/bot-dashboard', icon: BarChart3, label: 'Bot Dashboard', short: 'Bots' },
      { to: '/bots', icon: Bot, label: 'Bot Builder' },
      { to: '/bot-desk', icon: Building2, label: 'Bot Desk' },
      { to: '/copy-trading', icon: Copy, label: 'Copy Trading' },
      { to: '/agents', icon: Users, label: 'Agent Command' },
      { to: '/tasks', icon: ClipboardList, label: 'Task Manager' },
    ],
  },
  {
    label: 'MARKETS',
    items: [
      { to: '/dashboard', icon: LayoutDashboard, label: 'Overview', short: 'Home' },
      { to: '/equity', icon: TrendingUp, label: 'Equity', short: 'Stocks' },
      { to: '/crypto', icon: Bitcoin, label: 'Crypto', short: 'Crypto' },
      { to: '/polymarket', icon: Coins, label: 'Polymarket' },
      { to: '/options', icon: Layers, label: 'Options' },
      { to: '/options-chain', icon: CandlestickChart, label: 'Options Chain' },
      { to: '/macro', icon: Globe, label: 'Macro' },
      { to: '/funding-rates', icon: Percent, label: 'Funding Rates' },
      { to: '/scanners', icon: ScanSearch, label: 'Scanners' },
      { to: '/positions', icon: ListCollapse, label: 'Positions' },
    ],
  },
  {
    label: 'ANALYTICS',
    items: [
      { to: '/analytics', icon: LineChart, label: 'Analytics', short: 'Stats' },
      { to: '/backtest', icon: FlaskConical, label: 'Backtest' },
      { to: '/comparison', icon: BarChart2, label: 'Comparison' },
      { to: '/experiments', icon: Beaker, label: 'Experiments' },
      { to: '/ml-insights', icon: BrainCircuit, label: 'ML Insights' },
      { to: '/attribution', icon: PieChart, label: 'Attribution' },
      { to: '/leaderboard', icon: Trophy, label: 'Leaderboard' },
      { to: '/pnl', icon: DollarSign, label: 'P&L' },
      { to: '/activity', icon: Activity, label: 'Activity' },
    ],
  },
  {
    label: 'RISK',
    items: [
      { to: '/risk', icon: Shield, label: 'Risk Manager' },
      { to: '/risk-controls', icon: ShieldAlert, label: 'Risk Controls' },
    ],
  },
  {
    label: 'SYSTEM',
    items: [
      { to: '/pipeline', icon: GitBranch, label: 'Pipeline' },
      { to: '/promotions', icon: Rocket, label: 'Promotions' },
      { to: '/releases', icon: PackageCheck, label: 'Releases' },
      { to: '/agent-logs', icon: Terminal, label: 'Agent Logs' },
      { to: '/system', icon: Monitor, label: 'System Monitor' },
      { to: '/archive', icon: Archive, label: 'Archive' },
    ],
  },
]

/** Flat nav list (all items) — used by legacy consumers */
export const NAV: NavItem[] = NAV_GROUPS.flatMap(g => g.items)

/** Primary destinations shown in the mobile bottom tab bar. */
export const PRIMARY_NAV: NavItem[] = [
  NAV_GROUPS[0].items[0], // Bot Dashboard
  NAV_GROUPS[1].items[1], // Equity
  NAV_GROUPS[1].items[2], // Crypto
  NAV_GROUPS[2].items[0], // Analytics
]
