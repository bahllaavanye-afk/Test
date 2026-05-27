import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 30000,
})

// JWT interceptor — reads from memory store (not localStorage, for XSS safety)
let _accessToken: string | null = null

export const setToken = (token: string | null) => { _accessToken = token }
export const getToken = () => _accessToken

api.interceptors.request.use((config) => {
  if (_accessToken) {
    config.headers.Authorization = `Bearer ${_accessToken}`
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
