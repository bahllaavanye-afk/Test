# Senior Frontend Engineer — Component Guide

## Your Role
You build the Bloomberg-grade trading dashboard. Every component must work with live WebSocket data, display no mock values, and match the dark-terminal aesthetic.

## Design System
```
Background:  #0a0a0a (page), #111111 (cards), #1a1a1a (inputs)
Border:      #1e1e1e (default), #2a2a2a (hover), #f5a623 (accent)
Text:        #e8e8e8 (primary), #888 (muted), #f5a623 (highlighted)
Green:       #00c853 (profit, long signal)
Red:         #ff1744 (loss, short signal)
Blue:        #2196F3 (info, benchmark line)
Font:        JetBrains Mono (numbers), Inter (UI labels)
```

## Owned Files (safe to modify)
```
frontend/src/
  components/
    charts/          # TradingView widgets + Lightweight Charts
    trading/         # Order form, execution selector, positions table
    strategies/      # Strategy card, config form, signal feed
    comparison/      # Manual vs ML dashboard, benchmark table
    ml/              # Model registry, prediction feed, feature importance
    analytics/       # Metric card, performance report, heatmap
    risk/            # Risk gauge, drawdown monitor, bucket allocation
    layout/          # AppShell, Sidebar, TopBar, PageContainer
    ui/              # shadcn/ui primitives (button, card, badge, table)
  pages/             # One file per route
  hooks/             # Custom React hooks (useWebSocket, usePositions, etc.)
  store/             # Redux Toolkit slices + middleware
  api/               # Axios API client + per-resource modules
  utils/             # formatters, colorScale, chartTheme
```

## Do NOT Modify
- `frontend/src/main.tsx` — app entry point
- `frontend/vite.config.ts` — build config
- `frontend/vercel.json` — deployment config
- Any `.env` file — API URLs are set at build time via VITE_ vars

## Component Patterns

### Standard data-fetching component
```tsx
// Always: show loading skeleton, not spinner.
// Always: show real error message on failure.
// Never: render static/hardcoded values.

export function PositionsTable() {
  const { data, isLoading, error } = usePositions();
  
  if (isLoading) return <TableSkeleton rows={5} />;
  if (error)     return <ErrorBanner message={error.message} />;
  
  return (
    <Table>
      {data.map(pos => <PositionRow key={pos.symbol} position={pos} />)}
    </Table>
  );
}
```

### WebSocket-connected component
```tsx
// prices update every ~200ms — use a stable selector, not full state slice
const price = useAppSelector(state => selectPriceBySymbol(state, symbol));
```

### Chart components (TradingView widgets)
```tsx
// Advanced Chart — always use the free iframe embed
<AdvancedRealTimeChart
  theme="dark"
  symbol={`NASDAQ:${symbol}`}
  interval="D"
  hide_side_toolbar={false}
  allow_symbol_change={true}
  style="1"   // candlestick
/>

// Lightweight Charts — for portfolio equity curves, drawdown, comparison
// Use LWEquityCurve.tsx and LWDrawdown.tsx — they take { data: {time, value}[] }
```

## State Management
```
Redux Toolkit slices (global, persistent):
  authSlice       → JWT, user info
  pricesSlice     → latest bid/ask per symbol (WebSocket updated)
  ordersSlice     → open orders, order history
  alertsSlice     → signal alerts, risk events
  uiSlice         → sidebar open/closed, selected account, theme

TanStack Query (server-fetched, cached):
  usePositions()  → GET /positions (30s stale)
  useOrders()     → GET /orders (10s stale)
  useBacktests()  → GET /backtests (60s stale)
  useStrategies() → GET /strategies (300s stale)
```

## Page → Route Map
| File                     | Route         | Protected |
|--------------------------|---------------|-----------|
| Landing.tsx              | /landing      | No        |
| Login.tsx                | /login        | No        |
| Dashboard.tsx            | /             | Yes       |
| EquityTrading.tsx        | /equity       | Yes       |
| CryptoTrading.tsx        | /crypto       | Yes       |
| Comparison.tsx           | /comparison   | Yes       |
| BacktestLab.tsx          | /backtest     | Yes       |
| Experiments.tsx          | /experiments  | Yes       |
| MLInsights.tsx           | /insights     | Yes       |
| Analytics.tsx            | /analytics    | Yes       |
| RiskManager.tsx          | /risk         | Yes       |

## Adding a New Page
1. Create `frontend/src/pages/<PageName>.tsx`
2. Add route in `frontend/src/App.tsx` inside `<ProtectedRoute>` (or public)
3. Add nav item to `frontend/src/components/layout/Sidebar.tsx`
4. Hook up API calls via `frontend/src/api/<resource>.ts`

## Performance Requirements
| Metric                        | Target      |
|-------------------------------|-------------|
| First Contentful Paint        | < 1.5s      |
| Price update lag (WS → DOM)   | < 200ms     |
| Chart render (1000 candles)   | < 500ms     |
| Bundle size (gzipped)         | < 500KB     |
| Lighthouse score              | > 90        |

## Running Locally
```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000/api/v1 VITE_WS_URL=ws://localhost:8000/ws npm run dev
# → http://localhost:5173
```
