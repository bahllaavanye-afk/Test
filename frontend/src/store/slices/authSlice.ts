import { createSlice, PayloadAction } from '@reduxjs/toolkit'
import type { RootState } from '../index'

interface AuthState {
  accessToken: string | null
  isAuthenticated: boolean
}

const storedToken = localStorage.getItem('access_token')

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
      localStorage.setItem('access_token', action.payload.access_token)
    },
    logout(state) {
      state.accessToken = null
      state.isAuthenticated = false
      localStorage.removeItem('access_token')
    },
  },
})

export const { setCredentials, logout } = authSlice.actions
export const selectIsAuthenticated = (state: RootState) => state.auth.isAuthenticated
export const selectToken = (state: RootState) => state.auth.accessToken
export default authSlice.reducer
