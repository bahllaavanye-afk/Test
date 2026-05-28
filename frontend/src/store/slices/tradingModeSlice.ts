import { createSlice } from '@reduxjs/toolkit'
import type { RootState } from '../index'

type TradingMode = 'paper' | 'live'

interface TradingModeState {
  mode: TradingMode
}

const stored = (sessionStorage.getItem('trading_mode') as TradingMode) || 'paper'

const tradingModeSlice = createSlice({
  name: 'tradingMode',
  initialState: { mode: stored } as TradingModeState,
  reducers: {
    setMode(state, action: { payload: TradingMode }) {
      state.mode = action.payload
      sessionStorage.setItem('trading_mode', action.payload)
    },
  },
})

export const { setMode } = tradingModeSlice.actions
export const selectTradingMode = (state: RootState) => state.tradingMode.mode
export default tradingModeSlice.reducer
