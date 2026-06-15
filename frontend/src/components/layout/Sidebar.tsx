import { NavLink } from 'react-router-dom'
import { NAV } from './navItems'

export default function Sidebar() {
  return (
    <aside className="hidden md:flex w-16 glass-panel border-r border-white/[0.06] flex-col items-center py-4 gap-1 relative z-10 overflow-y-auto overflow-x-visible scrollbar-thin">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:bg-blue-600 focus:text-white focus:px-4 focus:py-2 focus:rounded"
      >
        Skip to main content
      </a>
      <div
        className="font-bold text-lg mb-6 text-transparent bg-clip-text"
        style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
      >
        Q
      </div>
      <nav aria-label="Main navigation">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/dashboard' || to === '/'}
            aria-label={label}
            className={({ isActive }) =>
              `sidebar-nav-item w-10 h-10 flex items-center justify-center rounded-lg transition-all duration-200 ${
                isActive
                  ? 'bg-[#00ff88]/10 text-[#00ff88] shadow-[0_0_12px_rgba(0,255,136,0.20)]'
                  : 'text-[#888888] hover:text-[#e8e8e8] hover:bg-white/[0.06] hover:shadow-[0_0_8px_rgba(0,212,255,0.10)]'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon size={18} aria-hidden="true" />
                <span className="sidebar-tooltip">{label}</span>
                {isActive && <span className="sr-only">(current page)</span>}
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
