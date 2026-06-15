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
      setError('Authentication failed — check your credentials')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen mesh-bg flex items-center justify-center px-4 relative overflow-hidden">
      {/* Animated corner accent lines */}
      {/* Top-left */}
      <div style={{position:'fixed',top:0,left:0,width:120,height:120,pointerEvents:'none',zIndex:0}}>
        <div style={{position:'absolute',top:0,left:0,width:80,height:1,background:'linear-gradient(90deg,var(--accent),transparent)'}} />
        <div style={{position:'absolute',top:0,left:0,width:1,height:80,background:'linear-gradient(180deg,var(--accent),transparent)'}} />
      </div>
      {/* Bottom-right */}
      <div style={{position:'fixed',bottom:0,right:0,width:120,height:120,pointerEvents:'none',zIndex:0}}>
        <div style={{position:'absolute',bottom:0,right:0,width:80,height:1,background:'linear-gradient(270deg,var(--green),transparent)'}} />
        <div style={{position:'absolute',bottom:0,right:0,width:1,height:80,background:'linear-gradient(0deg,var(--green),transparent)'}} />
      </div>

      <div className="relative z-10 w-full max-w-sm">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-4">
            <span className="w-2 h-2 rounded-full pulse-green" style={{background:'var(--green)',boxShadow:'0 0 6px var(--green)'}} />
            <span style={{fontSize:10,letterSpacing:'0.2em',color:'var(--muted)',textTransform:'uppercase'}}>QUANTEDGE v2.0</span>
          </div>
          <h1 className="mono-num" style={{fontSize:28,fontWeight:900,letterSpacing:'-0.03em',color:'var(--text)',marginBottom:4}}>
            SYSTEM ACCESS
          </h1>
          <p style={{fontSize:11,color:'var(--muted)',letterSpacing:'0.05em'}}>
            Institutional Quantitative Trading Terminal
          </p>
        </div>

        {/* Login card */}
        <div className="glass-card" style={{padding:'28px 28px',borderRadius:12}}>
          {/* Scan line decoration */}
          <div style={{
            height:1,
            background:'linear-gradient(90deg,transparent,var(--accent),transparent)',
            marginBottom:24,
            opacity:0.6,
          }} />

          <form onSubmit={handleSubmit} style={{display:'flex',flexDirection:'column',gap:14}}>
            <div style={{position:'relative'}}>
              <label style={{fontSize:10,letterSpacing:'0.12em',textTransform:'uppercase',color:'var(--muted)',display:'block',marginBottom:6}}>Email</label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="operator@quantedge.io"
                required
                style={{
                  width:'100%',
                  background:'var(--surface)',
                  border:'1px solid var(--border2)',
                  borderRadius:6,
                  padding:'10px 12px',
                  fontSize:12,
                  color:'var(--text)',
                  outline:'none',
                  fontFamily:'JetBrains Mono, monospace',
                  transition:'border-color 0.15s',
                }}
                onFocus={e => { e.target.style.borderColor = 'rgba(245,166,35,0.4)'; e.target.style.boxShadow = '0 0 8px rgba(245,166,35,0.1)' }}
                onBlur={e => { e.target.style.borderColor = 'var(--border2)'; e.target.style.boxShadow = 'none' }}
              />
            </div>

            <div>
              <label style={{fontSize:10,letterSpacing:'0.12em',textTransform:'uppercase',color:'var(--muted)',display:'block',marginBottom:6}}>Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••••••"
                required
                style={{
                  width:'100%',
                  background:'var(--surface)',
                  border:'1px solid var(--border2)',
                  borderRadius:6,
                  padding:'10px 12px',
                  fontSize:12,
                  color:'var(--text)',
                  outline:'none',
                  fontFamily:'JetBrains Mono, monospace',
                  transition:'border-color 0.15s',
                }}
                onFocus={e => { e.target.style.borderColor = 'rgba(245,166,35,0.4)'; e.target.style.boxShadow = '0 0 8px rgba(245,166,35,0.1)' }}
                onBlur={e => { e.target.style.borderColor = 'var(--border2)'; e.target.style.boxShadow = 'none' }}
              />
            </div>

            {error && (
              <div style={{
                background:'rgba(255,23,68,0.08)',
                border:'1px solid rgba(255,23,68,0.2)',
                borderRadius:6,
                padding:'8px 12px',
                fontSize:11,
                color:'var(--red)',
              }}>
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              style={{
                width:'100%',
                background: loading ? 'var(--surface2)' : 'linear-gradient(135deg,var(--accent),rgba(245,166,35,0.8))',
                color: loading ? 'var(--muted)' : '#000',
                fontWeight:700,
                fontSize:11,
                letterSpacing:'0.15em',
                textTransform:'uppercase',
                padding:'12px',
                borderRadius:6,
                border:'none',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontFamily:'JetBrains Mono, monospace',
                transition:'all 0.15s',
                boxShadow: loading ? 'none' : '0 0 20px rgba(245,166,35,0.2)',
                marginTop:4,
              }}
            >
              {loading ? 'AUTHENTICATING...' : 'ACCESS TERMINAL'}
            </button>
          </form>

          <div style={{
            height:1,
            background:'linear-gradient(90deg,transparent,var(--green),transparent)',
            marginTop:24,
            opacity:0.4,
          }} />
        </div>

        <p style={{textAlign:'center',fontSize:10,color:'var(--muted)',marginTop:16,letterSpacing:'0.05em',opacity:0.6}}>
          QUANTEDGE © 2025 — INSTITUTIONAL USE ONLY
        </p>
      </div>
    </div>
  )
}
