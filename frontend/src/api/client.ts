import axios from 'axios'
import { getStoredRefreshToken, setStoredRefreshToken } from '../store/slices/authSlice'

const BASE_URL = import.meta.env.VITE_API_URL || ''

export const api = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 30000,
})

// JWT interceptor — primary store is in-memory, sessionStorage as persistence.
let _accessToken: string | null = sessionStorage.getItem('access_token')

export const setToken = (token: string | null) => {
  _accessToken = token
  if (token) sessionStorage.setItem('access_token', token)
  else sessionStorage.removeItem('access_token')
}
export const getToken = () => _accessToken

api.interceptors.request.use((config) => {
  if (_accessToken) config.headers.Authorization = `Bearer ${_accessToken}`
  return config
})

let _refreshing: Promise<string> | null = null

api.interceptors.response.use(
  (res) => {
    // Any successful response means the backend is reachable.
    window.dispatchEvent(new Event('backendUp'))
    return res
  },
  async (err) => {
    const original = err.config
    // No response object = network error / timeout / DNS = backend unreachable
    // (distinct from an HTTP error, which means the backend answered).
    if (!err.response) {
      // Render's free tier cold-starts in ~30-90s after sleeping, longer than the
      // 30s default timeout — so the first visit after a sleep failed everywhere.
      // Retry idempotent GETs once with a budget that outlasts the cold start.
      const method = (original?.method || '').toLowerCase()
      if (original && method === 'get' && !original._coldStartRetry) {
        original._coldStartRetry = true
        original.timeout = 95000
        window.dispatchEvent(new Event('backendDown'))
        return api(original)
      }
      window.dispatchEvent(new Event('backendDown'))
    }
    if (err.response?.status === 401 && original && !original._retry) {
      original._retry = true
      const refreshToken = getStoredRefreshToken()
      if (refreshToken && !_refreshing) {
        _refreshing = axios
          .post(`${BASE_URL}/api/v1/auth/refresh`, { refresh_token: refreshToken }, { timeout: 10000 })
          .then(r => {
            const { access_token, refresh_token } = r.data
            setToken(access_token)
            setStoredRefreshToken(refresh_token)
            return access_token
          })
          .catch(() => {
            setToken(null)
            setStoredRefreshToken(null)
            window.dispatchEvent(new Event('sessionExpired'))
            return Promise.reject(new Error('session expired'))
          })
          .finally(() => { _refreshing = null })
      }
      if (_refreshing) {
        try {
          const newToken = await _refreshing
          original.headers.Authorization = `Bearer ${newToken}`
          return api(original)
        } catch {
          return Promise.reject(err)
        }
      }
      // No refresh token available — signal session expiry via event
      setToken(null)
      setStoredRefreshToken(null)
      window.dispatchEvent(new Event('sessionExpired'))
    }
    return Promise.reject(err)
  }
)

export const callLogout = async () => {
  const refreshToken = getStoredRefreshToken()
  if (refreshToken) {
    try {
      await axios.post(`${BASE_URL}/api/v1/auth/logout`, { refresh_token: refreshToken })
    } catch {
      // Best-effort — clear locally regardless
    }
  }
  setToken(null)
  setStoredRefreshToken(null)
}

export default api
