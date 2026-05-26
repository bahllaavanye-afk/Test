# Frontend Agent Guide

## Stack
React 18 + TypeScript + Vite + Redux Toolkit + TanStack Query + TradingView widgets + Tailwind CSS

## Design System
Bloomberg dark theme: bg=#0a0a0a, surface=#111111, border=#1e1e1e, accent=#f5a623, green=#00c853, red=#ff1744

## Key Directories
- `src/pages/` — one file per route
- `src/components/layout/` — AppShell, Sidebar, TopBar
- `src/components/charts/` — TVAdvancedChart (iframe), LW charts
- `src/components/trading/` — OrderForm, OrdersTable
- `src/store/slices/` — Redux state
- `src/api/` — Axios API client

## Adding a Page
1. Create `src/pages/MyPage.tsx`
2. Add route in `App.tsx`
3. Add nav item in `Sidebar.tsx`

## TradingView Free Widgets
- Advanced Chart: `components/charts/TVAdvancedChart.tsx` (iframe)
- Ticker Tape, Market Overview: use iframe embeds from tradingview.com/widget/

## API Convention
All API calls go through `src/api/client.ts` (Axios with JWT interceptor).
Use TanStack Query for caching: `useQuery({ queryKey: [...], queryFn: () => api.get('/...').then(r => r.data) })`
