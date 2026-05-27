import { createSlice, PayloadAction } from '@reduxjs/toolkit'
import type { RootState } from '../index'
import { setToken } from '../../api/client'

interface AuthState {
  accessToken: string | null
  isAuthenticated: boolean
}

// client.ts already initializes _accessToken from sessionStorage at module load,
// so page refreshes preserve auth without re-login.
const storedToken = sessionStorage.getItem('access_token')

const initialState: AuthState = {
  accessToken: storedToken,
  isAuthenticated: !!storedToken,
}

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setCredentials(state, action: PayloadAction<{ access_token: string }>) {
      state.accessToken = action.payload.access_token
      state.isAuthenticated = true
      setToken(action.payload.access_token)
    },
    logout(state) {
      state.accessToken = null
      state.isAuthenticated = false
      setToken(null)
    },
  },
})

export const { setCredentials, logout } = authSlice.actions
export const selectIsAuthenticated = (state: RootState) => state.auth.isAuthenticated
export const selectToken = (state: RootState) => state.auth.accessToken
export default authSlice.reducer
