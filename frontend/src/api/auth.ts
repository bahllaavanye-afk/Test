import api from './client'

export const login = (email: string, password: string) =>
  api.post('/auth/login', { username: email, password }).then(r => r.data)

export const register = (email: string, password: string) =>
  api.post('/auth/register', { email, password }).then(r => r.data)
