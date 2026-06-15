import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { ChevronDown } from 'lucide-react'
import { NAV_GROUPS } from './navItems'

export default function Sidebar() {
  // All groups open by default; user can collapse any
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const toggle = (label: string) =>
    setCollapsed(prev => ({ ...prev, [label]: !prev[label] }))

  return (
    <aside className="hidden md:flex w-52 glass-panel border-r border-white/[0.06] flex-col py-4 relative z-10 overflow-y-auto scrollbar-thin">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:bg-blue-600 focus:text-white focus:px-4 focus:py-2 focus:rounded"
      >
        Skip to main content
      </a>

      {/* Logo */}
      <div className="px-4 mb-5">
        <span
          className="font-bold text-lg text-transparent bg-clip-text"
          style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
        >
          QUANTEDGE
        </span>
      </div>

      <nav aria-label="Main navigation" className="flex flex-col gap-0.5 px-2">
        {NAV_GROUPS.map(group => {
          const isCollapsed = collapsed[group.label]
          return (
            <div key={group.label} className="mb-1">
              {/* Group header — clickable to collapse */}
              <button
                onClick={() => toggle(group.label)}
                className="w-full flex items-center justify-between px-2 py-1 mb-0.5 rounded-md text-[#555] hover:text-[#888] transition-colors group"
                aria-expanded={!isCollapsed}
              >
                <span className="text-[10px] font-bold tracking-widest uppercase">
                  {group.label}
                </span>
                <ChevronDown
                  size={12}
                  className={`transition-transform duration-200 ${isCollapsed ? '-rotate-90' : ''}`}
                />
              </button>

              {/* Group items */}
              {!isCollapsed && (
                <div className="flex flex-col gap-0.5">
                  {group.items.map(({ to, icon: Icon, label }) => (
                    <NavLink
                      key={to}
                      to={to}
                      end={to === '/dashboard' || to === '/'}
                      aria-label={label}
                      className={({ isActive }) =>
                        `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-150 ${
                          isActive
                            ? 'bg-[#00ff88]/10 text-[#00ff88]'
                            : 'text-[#777] hover:text-[#e8e8e8] hover:bg-white/[0.04]'
                        }`
                      }
                    >
                      {({ isActive }) => (
                        <>
                          <Icon size={15} aria-hidden="true" className="shrink-0" />
                          <span className="truncate">{label}</span>
                          {isActive && <span className="sr-only">(current)</span>}
                        </>
                      )}
                    </NavLink>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </nav>
    </aside>
  )
}
