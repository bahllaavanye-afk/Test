import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bloomberg: {
          bg: '#0a0a0a',
          surface: '#111111',
          border: '#1e1e1e',
          text: '#e8e8e8',
          muted: '#888888',
          accent: '#f5a623',
          green: '#00c853',
          red: '#ff1744',
          blue: '#2979ff',
        },
        surface: {
          0: '#0a0d12',
          1: '#0f1318',
          2: '#151a21',
        },
        'glow-green': '#00ff88',
        'glow-blue': '#00d4ff',
        'glow-amber': '#ffb347',
        accent: '#6366f1',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      animation: {
        'gradient-flow': 'gradient-flow 4s ease infinite',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        shimmer: 'shimmer 1.5s infinite',
        'counter-up': 'counter-up 0.6s ease-out forwards',
        'spin-slow': 'spin-slow 8s linear infinite',
        'particle-float': 'particle-float 6s ease-in-out infinite',
        'data-stream': 'data-stream 2.5s linear infinite',
      },
      keyframes: {
        'gradient-flow': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 4px 1px #00ff88', opacity: '1' },
          '50%': { boxShadow: '0 0 14px 4px #00ff88', opacity: '0.8' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% center' },
          '100%': { backgroundPosition: '200% center' },
        },
        'counter-up': {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'spin-slow': {
          from: { transform: 'rotate(0deg)' },
          to: { transform: 'rotate(360deg)' },
        },
        'particle-float': {
          '0%, 100%': { transform: 'translateY(0px)', opacity: '0.6' },
          '50%': { transform: 'translateY(-18px)', opacity: '1' },
        },
        'data-stream': {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '10%': { opacity: '1' },
          '90%': { opacity: '1' },
          '100%': { transform: 'translateX(100%)', opacity: '0' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config
