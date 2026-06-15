import { useState, useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Menu, X } from 'lucide-react'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import MobileDrawer from './MobileDrawer'
import '../../styles/animations.css'

export default function AppShell() {
  const location = useLocation()
  const [drawerOpen, setDrawerOpen] = useState(false)

  // Close drawer on route change
  useEffect(() => {
    setDrawerOpen(false)
  }, [location.pathname])

  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (drawerOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => { document.body.style.overflow = '' }
  }, [drawerOpen])

  return (
    <div className="flex h-[100dvh] overflow-hidden bg-[#0a0a0a] relative">
      {/* Animated grid background */}
      <div
        className="absolute inset-0 pointer-events-none z-0 animated-grid-bg"
        aria-hidden="true"
      />

      {/* Desktop sidebar (hidden on mobile) */}
      <Sidebar />

      {/* Mobile slide-in drawer */}
      <MobileDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />

      <div className="flex flex-col flex-1 overflow-hidden relative z-10">
        {/* Desktop top bar */}
        <div className="hidden md:block">
          <TopBar />
        </div>

        {/* Mobile top bar with hamburger */}
        <header
          className="md:hidden glass-panel border-b border-white/[0.06] flex items-center justify-between px-4 z-10 h-14"
          role="banner"
        >
          <button
            onClick={() => setDrawerOpen(true)}
            className="w-9 h-9 flex items-center justify-center rounded-lg text-[#888] hover:text-[#e8e8e8] hover:bg-white/[0.06] transition-colors"
            aria-label="Open navigation menu"
            aria-expanded={drawerOpen}
          >
            <Menu size={20} aria-hidden="true" />
          </button>
          <span
            className="font-bold text-sm tracking-tight text-transparent bg-clip-text"
            style={{ backgroundImage: 'linear-gradient(135deg, #00ff88, #00d4ff)' }}
          >
            QUANTEDGE
          </span>
          {/* Placeholder to balance the flex row */}
          <div className="w-9" />
        </header>

        <main
          id="main-content"
          role="main"
          key={location.pathname}
          className="flex-1 overflow-auto overflow-x-hidden p-3 sm:p-4"
          style={{ animation: 'page-fade-in 0.18s ease-out' }}
        >
          <Outlet />
        </main>
      </div>

      <style>{`
        @keyframes page-fade-in {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
