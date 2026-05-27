import { configureStore } from '@reduxjs/toolkit'
import authReducer from './slices/authSlice'
import pricesReducer from './slices/pricesSlice'
import ordersReducer from './slices/ordersSlice'
import tradingModeReducer from './slices/tradingModeSlice'

export const store = configureStore({
  reducer: {
    auth: authReducer,
    prices: pricesReducer,
    orders: ordersReducer,
    tradingMode: tradingModeReducer,
  },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
