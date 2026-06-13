import { useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { Menu, X } from 'lucide-react'
import { NAV, PRIMARY_NAV } from './navItems'

/**
 * Mobile bottom tab bar — the signature pattern of Robinhood / Coinbase / KuCoin.
 * Shows the four primary destinations plus a "More" button that opens a
 * full-screen sheet with the complete navigation grid. Hidden on md+ (desktop
 * uses the sidebar). Respects iOS safe-area insets so it clears the home bar.
 */
export default function MobileNav() {
  const [sheetOpen, setSheetOpen] = useState(false)
  const location = useLocation()

  const tabClass = (isActive: boolean) =>
    `flex flex-col items-center justify-center gap-0.5 flex-1 min-h-[52px] rounded-xl transition-colors ${
      isActive ? 'text-[#00ff88]' : 'text-[#8a8a9a] active:text-[#e8e8e8]'
    }`

  return (
    <>
      {/* Full-screen "More" sheet */}
      {sheetOpen && (
        <div
          className="md:hidden fixed inset-0 z-50 bg-[#070709]/95 backdrop-blur-xl flex flex-col"
          role="dialog"
          aria-modal="true"
          aria-label="All navigation"
        >
          <div className="flex items-center justify-between px-5 h-14 border-b border-white/[0.06]">
            <span
              className="font-bold text-base text-transparent bg-clip-text"
              style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
            >
              QUANTEDGE
            </span>
            <button
              onClick={() => setSheetOpen(false)}
              aria-label="Close menu"
              className="w-10 h-10 flex items-center justify-center rounded-lg text-[#888] active:bg-white/10"
            >
              <X size={22} />
            </button>
          </div>
          <nav className="grid grid-cols-3 gap-3 p-5 overflow-y-auto" aria-label="All destinations">
            {NAV.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                onClick={() => setSheetOpen(false)}
                className={({ isActive }) =>
                  `flex flex-col items-center justify-center gap-2 aspect-square rounded-2xl border transition-all ${
                    isActive
                      ? 'border-[#00ff88]/40 bg-[#00ff88]/10 text-[#00ff88]'
                      : 'border-white/[0.06] bg-white/[0.02] text-[#bdbdce] active:bg-white/[0.06]'
                  }`
                }
              >
                <Icon size={24} aria-hidden="true" />
                <span className="text-[11px] font-medium text-center leading-tight px-1">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      )}

      {/* Bottom tab bar */}
      <nav
        className="md:hidden fixed bottom-0 left-0 right-0 z-40 glass-panel border-t border-white/[0.08] flex items-stretch px-2 pt-1"
        style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 6px)' }}
        aria-label="Primary"
      >
        {PRIMARY_NAV.map(({ to, icon: Icon, label, short }) => (
          <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => tabClass(isActive)}>
            {({ isActive }) => (
              <>
                <Icon size={21} aria-hidden="true" strokeWidth={isActive ? 2.4 : 1.9} />
                <span className="text-[10px] font-semibold tracking-wide">{short || label}</span>
              </>
            )}
          </NavLink>
        ))}
        <button
          onClick={() => setSheetOpen(true)}
          className={tabClass(
            sheetOpen || !PRIMARY_NAV.some((n) => n.to === location.pathname),
          )}
          aria-label="More navigation"
          aria-expanded={sheetOpen}
        >
          <Menu size={21} strokeWidth={1.9} />
          <span className="text-[10px] font-semibold tracking-wide">More</span>
        </button>
      </nav>
    </>
  )
}
