import api from './client'

export const getOrders = () => api.get('/orders/').then(r => r.data)
export const submitOrder = (data: any) => api.post('/orders/', data).then(r => r.data)
export const cancelOrder = (id: string) => api.delete(`/orders/${id}`).then(r => r.data)
