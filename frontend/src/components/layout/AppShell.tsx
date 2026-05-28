import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopBar from './TopBar'

export default function AppShell() {
  const location = useLocation()
  return (
    <div className="flex h-screen overflow-hidden bg-[#0a0a0a]">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
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
