import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import '../../styles/animations.css'

const theme = {
  colors: {
    background: '#0a0a0a',
    card: '#111111',
    input: '#1a1a1a',
    border: '#1e1e1e',
    hoverBorder: '#2a2a2a',
    accent: '#f5a623',
    textPrimary: '#e8e8e8',
    textMuted: '#888',
    textHighlighted: '#f5a623',
    positive: '#00c853',
    negative: '#ff1744',
    info: '#2196F3',
    purple: '', // not specified, assume variable should be imported
  },
}

export default function AppShell() {
  const location = useLocation()
  return (
    <div className={`flex h-screen overflow-hidden bg-[${theme.colors.background}] relative`}>
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