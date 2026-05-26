import { Routes, Route, Navigate } from 'react-router-dom'
import { useSelector } from 'react-redux'
import AppShell from './components/layout/AppShell'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import EquityTrading from './pages/EquityTrading'
import CryptoTrading from './pages/CryptoTrading'
import Comparison from './pages/Comparison'
import BacktestLab from './pages/BacktestLab'
import Experiments from './pages/Experiments'
import Analytics from './pages/Analytics'
import RiskManager from './pages/RiskManager'
import { selectIsAuthenticated } from './store/slices/authSlice'

function RequireAuth({ children }: { children: React.ReactNode }) {
  const isAuth = useSelector(selectIsAuthenticated)
  return isAuth ? <>{children}</> : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<RequireAuth><AppShell /></RequireAuth>}>
        <Route index element={<Dashboard />} />
        <Route path="equity" element={<EquityTrading />} />
        <Route path="crypto" element={<CryptoTrading />} />
        <Route path="comparison" element={<Comparison />} />
        <Route path="backtest" element={<BacktestLab />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="risk" element={<RiskManager />} />
      </Route>
    </Routes>
  )
}
