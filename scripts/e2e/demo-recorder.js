// Demo recorder: walks every QuantEdge page with live data,
// captures full-page screenshots + a 30-90s video per page.
//
// Run inside GitHub Actions where Alpaca/Supabase/Upstash are reachable.
// Sandbox-side `claude` should NOT run this — sandbox blocks all external APIs.

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const OUT = path.join(__dirname, 'output');
fs.mkdirSync(OUT, { recursive: true });

const PAGES = [
  { name: '01-landing',       path: '/landing',         wait: 'networkidle', settleMs: 4000, requireAuth: false },
  { name: '02-login',         path: '/login',           wait: 'networkidle', settleMs: 3000, requireAuth: false },
  { name: '03-dashboard',     path: '/',                wait: 'networkidle', settleMs: 6000, requireAuth: true  },
  { name: '04-equity',        path: '/equity',          wait: 'networkidle', settleMs: 8000, requireAuth: true  },
  { name: '05-crypto',        path: '/crypto',          wait: 'networkidle', settleMs: 8000, requireAuth: true  },
  { name: '06-options-flow',  path: '/options',         wait: 'networkidle', settleMs: 8000, requireAuth: true  },
  { name: '07-options-chain', path: '/options-chain',   wait: 'networkidle', settleMs: 8000, requireAuth: true  },
  { name: '08-polymarket',    path: '/polymarket',      wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '09-comparison',    path: '/comparison',      wait: 'networkidle', settleMs: 6000, requireAuth: true  },
  { name: '10-backtest',      path: '/backtest',        wait: 'networkidle', settleMs: 6000, requireAuth: true  },
  { name: '11-experiments',   path: '/experiments',     wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '12-ml-insights',   path: '/ml-insights',     wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '13-analytics',     path: '/analytics',       wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '14-risk',          path: '/risk',            wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '15-activity',      path: '/activity',        wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '16-leaderboard',   path: '/leaderboard',     wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '17-pnl',           path: '/pnl',             wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '18-system',        path: '/system',          wait: 'networkidle', settleMs: 5000, requireAuth: true  },
  { name: '19-macro',         path: '/macro',           wait: 'networkidle', settleMs: 5000, requireAuth: true  },
];

// Test credentials seeded on each E2E run (must match seed script)
const DEMO_EMAIL = process.env.DEMO_EMAIL || 'demo@quantedge.local';
const DEMO_PASSWORD = process.env.DEMO_PASSWORD || 'demo-pass-1234';

async function login(page) {
  await page.goto(`${BASE_URL}/login`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(1000);
  try {
    await page.fill('input[type="email"], input[name="email"]', DEMO_EMAIL);
    await page.fill('input[type="password"], input[name="password"]', DEMO_PASSWORD);
    await Promise.all([
      page.waitForLoadState('networkidle', { timeout: 15000 }),
      page.click('button[type="submit"], button:has-text("Sign in"), button:has-text("Login")'),
    ]);
    await page.waitForTimeout(2000);
    console.log(`  ✓ Logged in as ${DEMO_EMAIL}`);
  } catch (e) {
    console.warn(`  ⚠ Login attempt failed: ${e.message.slice(0, 120)}`);
  }
}

(async () => {
  console.log(`Recording ${PAGES.length} pages → ${OUT}`);
  console.log(`BASE_URL: ${BASE_URL}`);

  const browser = await chromium.launch({
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });

  // Health probe
  try {
    const probe = await browser.newContext();
    const probePage = await probe.newPage();
    const resp = await probePage.goto(BASE_URL, { timeout: 15000 });
    console.log(`Probe ${BASE_URL} → ${resp ? resp.status() : 'no-response'}`);
    await probe.close();
  } catch (e) {
    console.error(`Probe failed: ${e.message}`);
  }

  const results = [];
  for (const p of PAGES) {
    const pageOut = path.join(OUT, p.name);
    fs.mkdirSync(pageOut, { recursive: true });

    const context = await browser.newContext({
      viewport: { width: 1600, height: 1000 },
      recordVideo: { dir: pageOut, size: { width: 1600, height: 1000 } },
      colorScheme: 'dark',
    });
    const page = await context.newPage();
    if (p.requireAuth) {
      await login(page);
    }
    page.on('console', (msg) => {
      if (msg.type() === 'error') console.log(`  [console] ${msg.text().slice(0, 200)}`);
    });
    page.on('requestfailed', (req) => {
      console.log(`  [netfail] ${req.method()} ${req.url().slice(0, 120)} - ${req.failure()?.errorText}`);
    });

    const url = `${BASE_URL}${p.path}`;
    const start = Date.now();
    let status = 'ok';
    try {
      const resp = await page.goto(url, { waitUntil: p.wait, timeout: 30000 });
      console.log(`→ ${p.name} ${url} - HTTP ${resp ? resp.status() : '?'}`);
      // Let live queries settle (TanStack Query, websockets)
      await page.waitForTimeout(p.settleMs);
      // Full-page screenshot
      await page.screenshot({ path: path.join(pageOut, 'screenshot.png'), fullPage: true });
      // Above-the-fold viewport screenshot
      await page.screenshot({ path: path.join(pageOut, 'viewport.png'), fullPage: false });
      // Scroll through to capture any lazy-loaded content
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(1500);
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(1500);
    } catch (e) {
      status = `error: ${e.message.slice(0, 200)}`;
      console.error(`✗ ${p.name}: ${status}`);
      try {
        await page.screenshot({ path: path.join(pageOut, 'error.png'), fullPage: true });
      } catch {}
    }
    const elapsed = Date.now() - start;
    await context.close();

    // Rename video to predictable name
    const files = fs.readdirSync(pageOut).filter((f) => f.endsWith('.webm'));
    if (files[0]) {
      fs.renameSync(path.join(pageOut, files[0]), path.join(pageOut, 'video.webm'));
    }

    results.push({ name: p.name, path: p.path, status, elapsedMs: elapsed });
  }

  fs.writeFileSync(path.join(OUT, 'summary.json'), JSON.stringify(results, null, 2));

  console.log('\n=== SUMMARY ===');
  for (const r of results) {
    console.log(`${r.status === 'ok' ? '✓' : '✗'} ${r.name.padEnd(22)} ${r.elapsedMs}ms  ${r.status}`);
  }

  await browser.close();
  const failed = results.filter((r) => r.status !== 'ok').length;
  process.exit(failed > 0 ? 1 : 0);
})();
