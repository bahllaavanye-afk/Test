import { useState, useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { logout } from '../../store/slices/authSlice'
import { callLogout } from '../../api/client'
import { selectTradingMode, setMode } from '../../store/slices/tradingModeSlice'
import { LogOut, Activity } from 'lucide-react'
import { LiveIndicator } from '../ui/LiveIndicator'

function ModeModal({ mode, onClose }: { mode: 'paper' | 'live'; onClose: () => void }) {
  const dispatch = useDispatch()
  const [input, setInput] = useState('')
  const switchingToLive = mode === 'paper'
  const valid = !switchingToLive || input.trim() === 'CONFIRM LIVE'

  function handleSwitch() {
    if (!valid) return
    dispatch(setMode(switchingToLive ? 'live' : 'paper'))
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-[#111111] border rounded-xl p-6 w-full max-w-sm shadow-2xl"
        style={{ borderColor: switchingToLive ? '#ff174440' : '#f5a62340' }}
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 mb-3">
          <span
            className="w-2.5 h-2.5 rounded-full"
            style={{ background: switchingToLive ? '#ff1744' : '#f5a623', boxShadow: switchingToLive ? '0 0 8px #ff1744' : '0 0 8px #f5a623' }}
          />
          <h2 className="font-bold text-sm" style={{ color: switchingToLive ? '#ff1744' : '#f5a623' }}>
            {switchingToLive ? 'Switch to Live Trading' : 'Switch to Paper Trading'}
          </h2>
        </div>

        <p className="text-xs text-[#888] mb-3">
          {switchingToLive
            ? 'Real money will be used. Strategies will trade against live markets with real capital.'
            : 'All orders will be simulated. No real capital will be at risk.'}
        </p>

        {switchingToLive && (
          <>
            <p className="text-xs text-[#888] mb-1.5">Type <span className="font-mono font-bold text-white">CONFIRM LIVE</span> to proceed:</p>
            <input
              autoFocus
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSwitch()}
              className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded px-3 py-2 text-sm font-mono text-white mb-4 focus:outline-none focus:border-[#ff1744]/40"
              placeholder="CONFIRM LIVE"
            />
          </>
        )}

        <div className="flex gap-2 mt-4">
          <button
            onClick={onClose}
            className="flex-1 px-3 py-1.5 rounded bg-[#1e1e1e] text-[#888] text-xs hover:bg-[#2e2e2e] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSwitch}
            disabled={!valid}
            className="flex-1 px-3 py-1.5 rounded text-xs font-bold transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ background: switchingToLive ? '#ff1744' : '#f5a623', color: switchingToLive ? '#fff' : '#000' }}
          >
            {switchingToLive ? 'Go Live' : 'Switch to Paper'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function TopBar() {
  const dispatch = useDispatch()
  const mode = useSelector(selectTradingMode)
  const [showModal, setShowModal] = useState(false)
  const isLive = mode === 'live'
  const [clock, setClock] = useState('')
  const [isMarketOpen, setIsMarketOpen] = useState(false)

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
      {showModal && <ModeModal mode={mode} onClose={() => setShowModal(false)} />}
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
          {/* Trading mode badge */}
          <button
            onClick={() => setShowModal(true)}
            className="flex items-center group transition-all duration-200"
          >
            {isLive ? (
              <LiveIndicator label="LIVE" color="#ff1744" />
            ) : (
              <span
                className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border border-[#f5a623]/30 bg-[#f5a623]/10 text-[10px] font-bold tracking-widest font-mono text-[#f5a623] group-hover:border-[#f5a623]/60 transition-colors"
              >
                PAPER
              </span>
            )}
          </button>
          {/* Data feed live badge */}
          <LiveIndicator label="DATA FEED" color="#00ff88" />
          <span style={{fontSize:10,fontFamily:'JetBrains Mono,monospace',color:'var(--muted)',letterSpacing:'0.08em'}}>
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
