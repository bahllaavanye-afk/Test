import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface PricesState {
  prices: Record<string, number>
}

const initialState: PricesState = { prices: {} }

const pricesSlice = createSlice({
  name: 'prices',
  initialState,
  reducers: {
    updatePrice(state, action: PayloadAction<{ symbol: string; price: number }>) {
      state.prices[action.payload.symbol] = action.payload.price
    },
  },
})

export const { updatePrice } = pricesSlice.actions
export default pricesSlice.reducer
