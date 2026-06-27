import { lazy, Suspense, useEffect } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { useSelector, useDispatch } from 'react-redux'
import AppShell from './components/layout/AppShell'
import { ErrorBoundary } from './components/ErrorBoundary'
// Public pages — eagerly loaded (smallest possible initial bundle)
import Login from './pages/Login'
import Landing from './pages/Landing'
import LiveSnapshot from './pages/LiveSnapshot'
import { selectIsAuthenticated, selectExpiredAt, sessionExpired } from './store/slices/authSlice'

// All protected pages — lazy-loaded so each becomes a separate Vite chunk.
// Users only download the chunk for the page they actually visit.
const Dashboard = lazy(() => import('./pages/Dashboard'))
const EquityTrading = lazy(() => import('./pages/EquityTrading'))
const CryptoTrading = lazy(() => import('./pages/CryptoTrading'))
const Comparison = lazy(() => import('./pages/Comparison'))
const BacktestLab = lazy(() => import('./pages/BacktestLab'))
const Experiments = lazy(() => import('./pages/Experiments'))
const Analytics = lazy(() => import('./pages/Analytics'))
const RiskManager = lazy(() => import('./pages/RiskManager'))
const Activity = lazy(() => import('./pages/Activity'))
const Leaderboard = lazy(() => import('./pages/Leaderboard'))
const PnL = lazy(() => import('./pages/PnL'))
const Archive = lazy(() => import('./pages/Archive'))
const SystemMonitor = lazy(() => import('./pages/SystemMonitor'))
const OptionsFlow = lazy(() => import('./pages/OptionsFlow'))
const Options = lazy(() => import('./pages/Options'))
const MacroSignals = lazy(() => import('./pages/MacroSignals'))
const Polymarket = lazy(() => import('./pages/Polymarket'))
const MLInsights = lazy(() => import('./pages/MLInsights'))
const Pipeline = lazy(() => import('./pages/Pipeline'))
const Releases = lazy(() => import('./pages/Releases'))
const BotBuilder = lazy(() => import('./pages/BotBuilder'))
const BotDesk = lazy(() => import('./pages/BotDesk'))
const AgentDashboard = lazy(() => import('./pages/AgentDashboard'))
const Scanners = lazy(() => import('./pages/Scanners'))
const Promotions = lazy(() => import('./pages/Promotions'))
const AgentLogs = lazy(() => import('./pages/AgentLogs'))
const BotDashboard = lazy(() => import('./pages/BotDashboard'))
const PositionsHub = lazy(() => import('./pages/PositionsHub'))
const BotDetail = lazy(() => import('./pages/BotDetail'))
const TaskManager = lazy(() => import('./pages/TaskManager'))
const CopyTrading = lazy(() => import('./pages/CopyTrading'))
const PerformanceAttribution = lazy(() => import('./pages/PerformanceAttribution'))
const RiskControls = lazy(() => import('./pages/RiskControls'))
const FundingRateMonitor = lazy(() => import('./pages/FundingRateMonitor'))
const GoogleCallback = lazy(() => import('./pages/GoogleCallback'))

function PageLoader() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '100%', color: '#555', fontFamily: 'JetBrains Mono, monospace', fontSize: 13,
    }}>
      Loading…
    </div>
  )
}

function RequireAuth({ children }: { children: React.ReactNode }) {
  // Auth disabled — all pages public for demo/investor access
  void useSelector(selectIsAuthenticated)
  return <>{children}</>
}

function SessionExpiryHandler() {
  const navigate = useNavigate()
  const dispatch = useDispatch()
  const expiredAt = useSelector(selectExpiredAt)

  useEffect(() => {
    const handleExpiry = () => {
      dispatch(sessionExpired())
    }
    window.addEventListener('sessionExpired', handleExpiry)
    return () => window.removeEventListener('sessionExpired', handleExpiry)
  }, [dispatch])

  useEffect(() => {
    if (expiredAt) navigate('/login', { replace: true })
  }, [expiredAt, navigate])

  return null
}

export default function App() {
  return (
    <ErrorBoundary>
    <Suspense fallback={<PageLoader />}>
      <SessionExpiryHandler />
      <Routes>
        <Route path="/landing" element={<Landing />} />
        <Route path="/live" element={<LiveSnapshot />} />
        <Route path="/login" element={<Login />} />
        <Route path="/auth/google/callback" element={<GoogleCallback />} />
        <Route path="/" element={<AppShell />}>
          <Route index element={<Navigate to="/bot-dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="equity" element={<EquityTrading />} />
          <Route path="crypto" element={<CryptoTrading />} />
          <Route path="comparison" element={<Comparison />} />
          <Route path="backtest" element={<BacktestLab />} />
          <Route path="experiments" element={<Experiments />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="risk" element={<RiskManager />} />
          <Route path="activity" element={<Activity />} />
          <Route path="leaderboard" element={<Leaderboard />} />
          <Route path="pnl" element={<PnL />} />
          <Route path="archive" element={<Archive />} />
          <Route path="system" element={<SystemMonitor />} />
          <Route path="options" element={<OptionsFlow />} />
          <Route path="options-chain" element={<Options />} />
          <Route path="macro" element={<MacroSignals />} />
          <Route path="polymarket" element={<Polymarket />} />
          <Route path="ml-insights" element={<MLInsights />} />
          <Route path="pipeline" element={<Pipeline />} />
          <Route path="releases" element={<Releases />} />
          <Route path="bots" element={<BotBuilder />} />
          <Route path="bot-builder" element={<BotBuilder />} />
          <Route path="bot-desk" element={<BotDesk />} />
          <Route path="agents" element={<AgentDashboard />} />
          <Route path="agent-dashboard" element={<AgentDashboard />} />
          <Route path="scanners" element={<Scanners />} />
          <Route path="promotions" element={<Promotions />} />
          <Route path="agent-logs" element={<AgentLogs />} />
          <Route path="bot-dashboard" element={<BotDashboard />} />
          <Route path="bot-dashboard/:botId" element={<BotDetail />} />
          <Route path="positions" element={<PositionsHub />} />
          <Route path="tasks" element={<TaskManager />} />
          <Route path="copy-trading" element={<CopyTrading />} />
          <Route path="attribution" element={<PerformanceAttribution />} />
          <Route path="risk-controls" element={<RiskControls />} />
          <Route path="funding-rates" element={<FundingRateMonitor />} />
        </Route>
      </Routes>
    </Suspense>
    </ErrorBoundary>
  )
}
