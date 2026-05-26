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
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config
