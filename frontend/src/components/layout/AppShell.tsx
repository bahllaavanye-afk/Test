import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import '../../styles/animations.css'

export default function AppShell() {
  const location = useLocation()
  return (
    <div className="flex h-screen overflow-hidden bg-[#0a0d12] relative">
      {/* Animated grid background */}
      <div
        className="absolute inset-0 pointer-events-none z-0 animated-grid-bg"
        aria-hidden="true"
      />
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden relative z-10">
        <TopBar />
        <main
          key={location.pathname}
          className="flex-1 overflow-auto p-4"
          style={{
            animation: 'page-fade-in 0.18s ease-out',
          }}
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
