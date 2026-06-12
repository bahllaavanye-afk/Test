import { lazy, Suspense, useEffect } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { useSelector, useDispatch } from 'react-redux'
import AppShell from './components/layout/AppShell'
import { ErrorBoundary } from './components/ErrorBoundary'
// Public pages — eagerly loaded (smallest possible initial bundle)
import Login from './pages/Login'
import Landing from './pages/Landing'
import GoogleCallback from './pages/GoogleCallback'
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
const AgentDashboard = lazy(() => import('./pages/AgentDashboard'))
const Scanners = lazy(() => import('./pages/Scanners'))
const Promotions = lazy(() => import('./pages/Promotions'))

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
  const isAuth = useSelector(selectIsAuthenticated)
  return isAuth ? <>{children}</> : <Navigate to="/login" replace />
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
        <Route path="/login" element={<Login />} />
        <Route path="/auth/google/callback" element={<GoogleCallback />} />
        <Route path="/" element={<RequireAuth><AppShell /></RequireAuth>}>
          <Route index element={<Dashboard />} />
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
          <Route path="agents" element={<AgentDashboard />} />
          <Route path="scanners" element={<Scanners />} />
          <Route path="promotions" element={<Promotions />} />
        </Route>
      </Routes>
    </Suspense>
    </ErrorBoundary>
  )
}
