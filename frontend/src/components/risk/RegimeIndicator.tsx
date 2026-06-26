import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

type RegimeType = 'bull' | 'bear' | 'sideways' | 'unknown'

interface RegimeData {
  regime: RegimeType
  confidence: number          // 0–1
  updated_at: string | null
}

const REGIME_CONFIG: Record<
  RegimeType,
  { label: string; dot: string; accent: string; bg: string; border: string }
> = {
  bull: {
    label: 'BULL',
    dot: '',
    accent: '#00c853',
    bg: 'rgba(0,200,83,0.08)',
    border: 'rgba(0,200,83,0.25)',
  },
  bear: {
    label: 'BEAR',
    dot: '',
    accent: '#ff1744',
    bg: 'rgba(255,23,68,0.08)',
    border: 'rgba(255,23,68,0.25)',
  },
  sideways: {
    label: 'SIDEWAYS',
    dot: '',
    accent: '#f5a623',
    bg: 'rgba(245,166,35,0.08)',
    border: 'rgba(245,166,35,0.25)',
  },
  unknown: {
    label: 'UNKNOWN',
    dot: '',
    accent: '#888888',
    bg: 'rgba(136,136,136,0.08)',
    border: 'rgba(136,136,136,0.25)',
  },
}

export const RegimeIndicator = () => {
  const { data, isLoading, isError } = useQuery<RegimeData>({
    queryKey: ['regime', 'current'],
    queryFn: () => api.get('/regime/current').then((r) => r.data),
    refetchInterval: 60_