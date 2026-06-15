import { useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { ChevronDown } from 'lucide-react'
import { NAV_GROUPS, type NavGroup } from './navItems'

function useActiveGroup(groups: NavGroup[]): string | null {
  const { pathname } = useLocation()
  for (const group of groups) {
    if (group.items.some(item => item.to === pathname)) return group.label
  }
  return null
}

function NavGroupSection({ group, defaultOpen }: { group: NavGroup; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="mb-1">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-1.5 rounded-md text-[10px] font-bold tracking-widest text-[#555] hover:text-[#888] hover:bg-white/[0.03] transition-colors"
        aria-expanded={open}
      >
        <span className="flex items-center gap-1.5">
          <span>{group.emoji}</span>
          <span>{group.label}</span>
        </span>
        <ChevronDown
          size={11}
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
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-3 py-1.5 rounded-md text-xs transition-all duration-150 ${
                  isActive
                    ? 'bg-[#00ff88]/10 text-[#00ff88] font-semibold shadow-[0_0_10px_rgba(0,255,136,0.12)]'
                    : 'text-[#777] hover:text-[#e8e8e8] hover:bg-white/[0.05]'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon size={14} aria-hidden="true" className="flex-shrink-0" />
                  <span className="truncate">{label}</span>
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

export default function Sidebar() {
  const activeGroup = useActiveGroup(NAV_GROUPS)

  return (
    <aside className="hidden md:flex w-52 glass-panel border-r border-white/[0.06] flex-col py-4 relative z-10 overflow-y-auto overflow-x-hidden scrollbar-thin">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:bg-blue-600 focus:text-white focus:px-4 focus:py-2 focus:rounded"
      >
        Skip to main content
      </a>

      {/* Brand */}
      <div className="px-4 mb-5">
        <div
          className="font-bold text-sm tracking-tight text-transparent bg-clip-text"
          style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
        >
          QUANTEDGE
        </div>
      </div>

      <nav aria-label="Main navigation" className="flex-1 px-2">
        {NAV_GROUPS.map(group => (
          <NavGroupSection
            key={group.label}
            group={group}
            defaultOpen={activeGroup === group.label || group.label === 'BOTS'}
          />
        ))}
      </nav>
    </aside>
  )
}
