import { useState, useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { useQuery } from '@tanstack/react-query'
import { logout } from '../../store/slices/authSlice'
import { callLogout } from '../../api/client'
import api from '../../api/client'
import { selectTradingMode, setMode } from '../../store/slices/tradingModeSlice'
import { LogOut, Activity } from 'lucide-react'
import { LiveIndicator } from '../ui/LiveIndicator'

export default function TopBar() {
  const dispatch = useDispatch()
  const [clock, setClock] = useState('')
  const [isMarketOpen, setIsMarketOpen] = useState(false)

  const { data: strategies } = useQuery({
    queryKey: ['strategies-count'],
    queryFn: () => api.get('/strategies/').then(r => r.data),
    staleTime: 300_000,
    retry: false,
  })
  const strategyCount = Array.isArray(strategies) ? strategies.length : null

  useEffect(() => {
    function tick() {
      const now = new Date()
      const utc = now.toUTCString().slice(17, 25) // HH:MM:SS
      setClock(utc)
      // NYSE market hours: 14:30-21:00 UTC (Mon-Fri)
      const day = now.getUTCDay()
      const hour = now.getUTCHours()
      const minute = now.getUTCMinutes()
      const totalMinutes = hour * 60 + minute
      const isWeekday = day >= 1 && day <= 5
      setIsMarketOpen(isWeekday && totalMinutes >= 870 && totalMinutes < 1260) // 14:30-21:00
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <>
      <header className="relative h-10 glass-panel border-b border-white/[0.06] flex items-center justify-between px-4 z-10">
        {/* Animated gradient border on bottom */}
        <div
          className="absolute bottom-0 left-0 right-0 h-px animate-gradient"
          style={{
            backgroundImage: 'linear-gradient(90deg, transparent, #00ff88, #00d4ff, #6366f1, #00d4ff, #00ff88, transparent)',
            backgroundSize: '300% 100%',
          }}
        />
        <div className="flex items-center gap-3">
          <Activity size={14} className="text-[#00ff88]" />
          {/* Trading mode badge — paper-only, non-clickable */}
          <span
            className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border border-[#f5a623]/30 bg-[#f5a623]/10 text-[10px] font-bold tracking-widest font-mono text-[#f5a623] cursor-default"
            title="Paper trading mode — live trading not available"
          >
            PAPER
          </span>
          {/* Data feed live badge */}
          <LiveIndicator label="DATA FEED" color="#00ff88" />
          {/* Strategy count badge */}
          {strategyCount !== null && (
            <span
              className="hidden md:inline-flex items-center gap-1 px-2 py-0.5 rounded border border-[#1e1e1e] bg-[#111111] text-[9px] font-mono text-[#888888] tracking-wider"
            >
              <span className="w-1 h-1 rounded-full bg-[#f5a623] inline-block" />
              {strategyCount} strategies
            </span>
          )}
          {/* Pulsing UTC clock */}
          <span className="clock-pulse" style={{fontSize:10,fontFamily:'JetBrains Mono,monospace',color:'var(--muted)',letterSpacing:'0.08em'}}>
            UTC {clock}
          </span>
          <span
            className={isMarketOpen ? 'badge-green' : 'badge-muted'}
            style={{fontSize:9,letterSpacing:'0.1em'}}
          >
            NYSE {isMarketOpen ? 'OPEN' : 'CLOSED'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span
            className="font-bold text-xs text-transparent bg-clip-text"
            style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
          >
            QUANTEDGE
          </span>
          <button
            onClick={async () => {
              await callLogout()  // revoke refresh token on server
              dispatch(logout())
              window.location.href = '/login'
            }}
            className="text-[#888888] hover:text-[#e8e8e8] transition-colors"
            title="Logout"
          >
            <LogOut size={14} />
          </button>
        </div>
      </header>
      <style>{`
        @keyframes topbar-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.3); }
        }
      `}</style>
    </>
  )
}
