interface LiveIndicatorProps {
  label?: string
  color?: string
}

export function LiveIndicator({
  label = 'LIVE',
  color = '#00ff88',
}: LiveIndicatorProps) {
  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border"
      style={{
        borderColor: `${color}30`,
        background: `${color}10`,
      }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
        style={{
          background: color,
          boxShadow: `0 0 0 0 ${color}`,
          animation: 'live-dot-pulse 2s ease-in-out infinite',
        }}
      />
      <span
        className="text-[10px] font-bold tracking-widest font-mono"
        style={{ color }}
      >
        {label}
      </span>
      <style>{`
        @keyframes live-dot-pulse {
          0%, 100% {
            box-shadow: 0 0 0 0 ${color}80;
            transform: scale(1);
          }
          50% {
            box-shadow: 0 0 0 4px ${color}00;
            transform: scale(1.2);
          }
        }
      `}</style>
    </span>
  )
}

export default LiveIndicator
