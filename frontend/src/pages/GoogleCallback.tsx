import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useDispatch } from 'react-redux'
import { setCredentials } from '../store/slices/authSlice'

/**
 * Handles the redirect back from Google OAuth.
 *
 * The backend (/auth/google/callback) exchanges the code for tokens and
 * redirects here as: /auth/google/callback?access_token=...&refresh_token=...
 * We read those query params, store them in Redux/session, and forward to the
 * dashboard. On error (e.g. ?error=access_denied) we bounce back to /login.
 */
export default function GoogleCallback() {
  const [params] = useSearchParams()
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const [message, setMessage] = useState('Completing sign-in…')

  useEffect(() => {
    const accessToken = params.get('access_token')
    const refreshToken = params.get('refresh_token') ?? undefined
    const error = params.get('error')

    if (error) {
      setMessage(`Sign-in failed: ${error}`)
      const t = setTimeout(() => navigate('/login', { replace: true }), 2000)
      return () => clearTimeout(t)
    }

    if (accessToken) {
      dispatch(setCredentials({ access_token: accessToken, refresh_token: refreshToken }))
      navigate('/', { replace: true })
      return
    }

    setMessage('No credentials returned. Redirecting to login…')
    const t = setTimeout(() => navigate('/login', { replace: true }), 2000)
    return () => clearTimeout(t)
  }, [params, dispatch, navigate])

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#070709',
        color: '#e8e8e8',
        fontFamily: 'JetBrains Mono, monospace',
        flexDirection: 'column',
        gap: '1rem',
      }}
    >
      <div
        style={{
          width: 40,
          height: 40,
          border: '3px solid #1e1e1e',
          borderTopColor: '#f5a623',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
        }}
      />
      <p>{message}</p>
      <style>{'@keyframes spin{to{transform:rotate(360deg)}}'}</style>
    </div>
  )
}
