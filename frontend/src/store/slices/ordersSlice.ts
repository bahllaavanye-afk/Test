import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface Order {
  id: string
  symbol: string
  side: string
  status: string
  quantity: number
  filled_qty: number
}

interface OrdersState {
  orders: Order[]
}

const initialState: OrdersState = { orders: [] }

const ordersSlice = createSlice({
  name: 'orders',
  initialState,
  reducers: {
    setOrders(state, action: PayloadAction<Order[]>) {
      state.orders = action.payload
    },
    addOrder(state, action: PayloadAction<Order>) {
      state.orders.unshift(action.payload)
    },
  },
})

export const { setOrders, addOrder } = ordersSlice.actions
export default ordersSlice.reducer
