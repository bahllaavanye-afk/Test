import { type ReactNode } from 'react'

type GradientVariant = 'green-blue' | 'amber-pink' | 'blue-purple'

interface GradientTextProps {
  children: ReactNode
  gradient?: GradientVariant
}

const gradientMap: Record<GradientVariant, string> = {
  'green-blue': 'linear-gradient(135deg, #00ff88 0%, #00d4ff 50%, #00ff88 100%)',
  'amber-pink': 'linear-gradient(135deg, #ffb347 0%, #ff6b9d 50%, #ffb347 100%)',
  'blue-purple': 'linear-gradient(135deg, #00d4ff 0%, #6366f1 50%, #00d4ff 100%)',
}

export function GradientText({
  children,
  gradient = 'green-blue',
}: GradientTextProps) {
  return (
    <span
      className="text-transparent bg-clip-text animate-gradient"
      style={{
        backgroundImage: gradientMap[gradient],
        backgroundSize: '300% 300%',
      }}
    >
      {children}
    </span>
  )
}

export default GradientText
