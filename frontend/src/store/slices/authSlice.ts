import { createSlice, PayloadAction } from '@reduxjs/toolkit'
import type { RootState } from '../index'
import { setToken } from '../../api/client'

// Refresh token stored in sessionStorage only (cleared when tab closes).
// Never goes into Redux state to avoid accidental serialization.
const REFRESH_KEY = 'refresh_token'
export const getStoredRefreshToken = () => sessionStorage.getItem(REFRESH_KEY)
export const setStoredRefreshToken = (t: string | null) => {
  if (t) sessionStorage.setItem(REFRESH_KEY, t)
  else sessionStorage.removeItem(REFRESH_KEY)
}

interface AuthState {
  accessToken: string | null
  isAuthenticated: boolean
}

const storedToken = sessionStorage.getItem('access_token')

const initialState: AuthState = {
  accessToken: storedToken,
  isAuthenticated: !!storedToken,
}

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setCredentials(state, action: PayloadAction<{ access_token: string; refresh_token?: string }>) {
      state.accessToken = action.payload.access_token
      state.isAuthenticated = true
      setToken(action.payload.access_token)
      if (action.payload.refresh_token) {
        setStoredRefreshToken(action.payload.refresh_token)
      }
    },
    logout(state) {
      state.accessToken = null
      state.isAuthenticated = false
      setToken(null)
      setStoredRefreshToken(null)
    },
  },
})

export const { setCredentials, logout } = authSlice.actions
export const selectIsAuthenticated = (state: RootState) => state.auth.isAuthenticated
export const selectToken = (state: RootState) => state.auth.accessToken
export default authSlice.reducer
