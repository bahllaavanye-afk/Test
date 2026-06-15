import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { X, ChevronDown } from 'lucide-react'
import { NAV_GROUPS, type NavGroup } from './navItems'

interface MobileDrawerProps {
  open: boolean
  onClose: () => void
}

function MobileNavGroup({ group, onClose }: { group: NavGroup; onClose: () => void }) {
  const [open, setOpen] = useState(true)

  return (
    <div className="mb-1">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-2 rounded-md text-[11px] font-bold tracking-widest text-[#555] hover:text-[#888] hover:bg-white/[0.03] transition-colors"
        aria-expanded={open}
      >
        <span className="flex items-center gap-2">
          <span>{group.emoji}</span>
          <span>{group.label}</span>
        </span>
        <ChevronDown
          size={12}
          className="transition-transform duration-200"
          style={{ transform: open ? 'rotate(0deg)' : 'rotate(-90deg)' }}
          aria-hidden="true"
        />
      </button>

      {open && (
        <div className="mt-0.5 space-y-0.5">
          {group.items.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/dashboard' || to === '/'}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 rounded-md text-sm transition-all duration-150 ${
                  isActive
                    ? 'bg-[#00ff88]/10 text-[#00ff88] font-semibold'
                    : 'text-[#888] hover:text-[#e8e8e8] hover:bg-white/[0.05]'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon size={16} aria-hidden="true" className="flex-shrink-0" />
                  <span>{label}</span>
                  {isActive && <span className="sr-only">(current page)</span>}
                </>
              )}
            </NavLink>
          ))}
        </div>
      )}
    </div>
  )
}

export default function MobileDrawer({ open, onClose }: MobileDrawerProps) {
  if (!open) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="md:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
        aria-hidden="true"
        onClick={onClose}
      />

      {/* Slide-in drawer */}
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
        className="md:hidden fixed top-0 left-0 bottom-0 z-50 w-72 glass-panel border-r border-white/[0.08] flex flex-col"
        style={{ animation: 'drawer-slide-in 0.22s ease-out' }}
      >
        {/* Drawer header */}
        <div className="flex items-center justify-between px-4 h-14 border-b border-white/[0.06] flex-shrink-0">
          <span
            className="font-bold text-sm tracking-tight text-transparent bg-clip-text"
            style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
          >
            QUANTEDGE
          </span>
          <button
            onClick={onClose}
            className="w-9 h-9 flex items-center justify-center rounded-lg text-[#888] hover:text-[#e8e8e8] hover:bg-white/[0.06] transition-colors"
            aria-label="Close navigation menu"
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        {/* Scrollable nav groups */}
        <nav
          aria-label="Main navigation"
          className="flex-1 overflow-y-auto py-3 px-2"
        >
          {NAV_GROUPS.map(group => (
            <MobileNavGroup key={group.label} group={group} onClose={onClose} />
          ))}
        </nav>
      </div>

      <style>{`
        @keyframes drawer-slide-in {
          from { transform: translateX(-100%); opacity: 0.6; }
          to   { transform: translateX(0);     opacity: 1; }
        }
      `}</style>
    </>
  )
}
