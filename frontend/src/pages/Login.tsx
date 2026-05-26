import { useState } from 'react'
import { useDispatch } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import { setCredentials } from '../store/slices/authSlice'
import { login } from '../api/auth'

export default function Login() {
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const data = await login(email, password)
      dispatch(setCredentials(data))
      navigate('/')
    } catch {
      setError('Invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
      <div className="w-96 bg-[#111111] border border-[#1e1e1e] rounded-xl p-8">
        <div className="text-center mb-8">
          <h1 className="text-[#f5a623] text-2xl font-bold">QUANTEDGE</h1>
          <p className="text-[#888888] text-xs mt-1">Institutional Quantitative Trading</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="Email" required className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-[#f5a623] transition-colors" />
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Password" required className="w-full bg-[#0a0a0a] border border-[#1e1e1e] rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-[#f5a623] transition-colors" />
          {error && <p className="text-[#ff1744] text-xs">{error}</p>}
          <button type="submit" disabled={loading} className="w-full bg-[#f5a623] text-black font-bold py-3 rounded-lg hover:bg-[#e09520] transition-colors disabled:opacity-50">
            {loading ? 'Signing in...' : 'SIGN IN'}
          </button>
        </form>
      </div>
    </div>
  )
}
