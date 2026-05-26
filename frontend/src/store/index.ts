import { configureStore } from '@reduxjs/toolkit'
import authReducer from './slices/authSlice'
import pricesReducer from './slices/pricesSlice'
import ordersReducer from './slices/ordersSlice'

export const store = configureStore({
  reducer: {
    auth: authReducer,
    prices: pricesReducer,
    orders: ordersReducer,
  },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
