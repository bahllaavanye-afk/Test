import { type ReactNode } from 'react'

type GlowVariant = 'green' | 'blue' | 'amber' | 'none'

interface GlassCardProps {
  children: ReactNode
  className?: string
  glow?: GlowVariant
  animated?: boolean
}

const glowStyles: Record<GlowVariant, string> = {
  none: '',
  green:
    'hover:shadow-[0_0_0_1px_rgba(0,255,136,0.30),0_0_24px_rgba(0,255,136,0.10)] transition-shadow duration-300',
  blue:
    'hover:shadow-[0_0_0_1px_rgba(0,212,255,0.30),0_0_24px_rgba(0,212,255,0.10)] transition-shadow duration-300',
  amber:
    'hover:shadow-[0_0_0_1px_rgba(255,179,71,0.30),0_0_24px_rgba(255,179,71,0.10)] transition-shadow duration-300',
}

export function GlassCard({
  children,
  className = '',
  glow = 'none',
  animated = false,
}: GlassCardProps) {
  return (
    <div
      className={[
        'backdrop-blur-xl bg-white/[0.03] border border-white/[0.07] rounded-2xl',
        'transition-all duration-300',
        animated ? 'animate-gradient' : '',
        glowStyles[glow],
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      style={
        animated
          ? {
              backgroundImage:
                'linear-gradient(135deg, rgba(0,212,255,0.04), rgba(99,102,241,0.04), rgba(0,255,136,0.04), rgba(99,102,241,0.04))',
              backgroundSize: '300% 300%',
            }
          : undefined
      }
    >
      {children}
    </div>
  )
}

export default GlassCard
