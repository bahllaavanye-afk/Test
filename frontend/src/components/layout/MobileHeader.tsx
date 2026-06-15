import { useDispatch } from 'react-redux'
import { LogOut } from 'lucide-react'
import { logout } from '../../store/slices/authSlice'
import { callLogout } from '../../api/client'
import { LiveIndicator } from '../ui/LiveIndicator'

/**
 * Compact header for mobile — the dense desktop TopBar (UTC clock, NYSE badge,
 * strategy count) is too cramped on a phone. This keeps just brand, live status,
 * the paper-mode badge, and logout. Respects the notch via safe-area inset.
 */
export default function MobileHeader() {
  const dispatch = useDispatch()
  return (
    <header
      role="banner"
      className="md:hidden glass-panel border-b border-white/[0.06] flex items-center justify-between px-4 z-10"
      style={{ paddingTop: 'max(env(safe-area-inset-top), 8px)', height: 'calc(52px + max(env(safe-area-inset-top), 8px))' }}
    >
      <span
        className="font-bold text-base tracking-tight text-transparent bg-clip-text"
        style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
      >
        QUANTEDGE
      </span>
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border border-[#f5a623]/30 bg-[#f5a623]/10 text-[10px] font-bold tracking-widest text-[#f5a623]">
          PAPER
        </span>
        <LiveIndicator label="LIVE" color="#00ff88" />
        <button
          onClick={async () => {
            await callLogout()
            dispatch(logout())
            window.location.href = '/login'
          }}
          className="w-9 h-9 flex items-center justify-center rounded-lg text-[#888] active:bg-white/10"
          aria-label="Logout"
        >
          <LogOut size={18} aria-hidden="true" />
        </button>
      </div>
    </header>
  )
}
