import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || ''

export const api = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 30000,
})

// JWT interceptor — primary store is in-memory (_accessToken), with sessionStorage as
// persistence layer so tokens survive page refreshes within the same tab.
let _accessToken: string | null = sessionStorage.getItem('access_token')

export const setToken = (token: string | null) => {
  _accessToken = token
  if (token) {
    sessionStorage.setItem('access_token', token)
  } else {
    sessionStorage.removeItem('access_token')
  }
}
export const getToken = () => _accessToken

api.interceptors.request.use((config) => {
  // Always read from memory (set at login or page-load init above)
  const token = _accessToken
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    if (err.response?.status === 401) {
      setToken(null)
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default api
