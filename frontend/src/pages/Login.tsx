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

  const handleGoogleLogin = () => {
    const base = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    window.location.href = `${base}/api/v1/auth/google`
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

          {/* Divider */}
          <div style={{display:'flex',alignItems:'center',gap:12,margin:'20px 0'}}>
            <div style={{flex:1,height:1,background:'var(--border2)'}} />
            <span style={{fontSize:10,color:'var(--muted)',letterSpacing:'0.1em'}}>OR</span>
            <div style={{flex:1,height:1,background:'var(--border2)'}} />
          </div>

          {/* Google OAuth button */}
          <button
            onClick={handleGoogleLogin}
            style={{
              width:'100%',
              background:'var(--surface)',
              border:'1px solid var(--border2)',
              borderRadius:6,
              padding:'10px 12px',
              fontSize:11,
              color:'var(--text)',
              cursor:'pointer',
              display:'flex',
              alignItems:'center',
              justifyContent:'center',
              gap:10,
              fontFamily:'JetBrains Mono, monospace',
              letterSpacing:'0.05em',
              transition:'all 0.15s',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border2)'; (e.currentTarget as HTMLButtonElement).style.background = 'var(--surface2)' }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border2)'; (e.currentTarget as HTMLButtonElement).style.background = 'var(--surface)' }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Continue with Google
          </button>

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
