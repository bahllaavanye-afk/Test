import { useState, useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { useQuery } from '@tanstack/react-query'
import { logout } from '../../store/slices/authSlice'
import { callLogout } from '../../api/client'
import api from '../../api/client'
import { LogOut, Activity } from 'lucide-react'
import { LiveIndicator } from '../ui/LiveIndicator'

export default function TopBar() {
  const dispatch = useDispatch()
  const [clock, setClock] = useState('')
  const [isMarketOpen, setIsMarketOpen] = useState(false)

  const { data: strategies } = useQuery({
    queryKey: ['strategies-count'],
    queryFn: () => api.get('/strategies/').then(r => r.data),
    staleTime: 300_000,
    retry: false,
  })
  const strategyCount = Array.isArray(strategies) ? strategies.length : null

  useEffect(() => {
    function tick() {
      const now = new Date()
      const utc = now.toUTCString().slice(17, 25) // HH:MM:SS
      setClock(utc)
      // NYSE market hours: 14:30-21:00 UTC (Mon-Fri)
      const day = now.getUTCDay()
      const hour = now.getUTCHours()
      const minute = now.getUTCMinutes()
      const totalMinutes = hour * 60 + minute
      const isWeekday = day >= 1 && day <= 5
      setIsMarketOpen(isWeekday && totalMinutes >= 870 && totalMinutes < 1260) // 14:30-21:00
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <>
      <header className="relative h-10 glass-panel border-b border-[#1