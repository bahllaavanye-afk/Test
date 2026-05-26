import { useDispatch } from 'react-redux'
import { logout } from '../../store/slices/authSlice'
import { LogOut, Activity } from 'lucide-react'

export default function TopBar() {
  const dispatch = useDispatch()
  return (
    <header className="h-10 bg-[#111111] border-b border-[#1e1e1e] flex items-center justify-between px-4">
      <div className="flex items-center gap-2">
        <Activity size={14} className="text-[#00c853]" />
        <span className="text-[#888888] text-xs">PAPER MODE</span>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-[#f5a623] font-bold text-xs">QUANTEDGE</span>
        <button
          onClick={() => dispatch(logout())}
          className="text-[#888888] hover:text-[#e8e8e8] transition-colors"
          title="Logout"
        >
          <LogOut size={14} />
        </button>
      </div>
    </header>
  )
}
