import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import MobileNav from './MobileNav'
import MobileHeader from './MobileHeader'
import '../../styles/animations.css'

export default function AppShell() {
  const location = useLocation()
  return (
    <div className="flex h-[100dvh] overflow-hidden bg-[#0a0a0a] relative">
      {/* Animated grid background */}
      <div
        className="absolute inset-0 pointer-events-none z-0 animated-grid-bg"
        aria-hidden="true"
      />
      {/* Desktop sidebar (hidden on mobile) */}
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden relative z-10">
        {/* Dense desktop top bar */}
        <div className="hidden md:block">
          <TopBar />
        </div>
        {/* Compact mobile header */}
        <MobileHeader />
        <main
          id="main-content"
          role="main"
          key={location.pathname}
          className="flex-1 overflow-auto overflow-x-hidden p-3 sm:p-4 pb-24 md:pb-4"
          style={{ animation: 'page-fade-in 0.18s ease-out' }}
        >
          <Outlet />
        </main>
        {/* Mobile bottom tab bar (hidden on desktop) */}
        <MobileNav />
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
