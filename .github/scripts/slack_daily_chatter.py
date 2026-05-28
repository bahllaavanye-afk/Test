"""
Daily Slack chatter — randomly post 8-15 messages per run across active channels.

Designed to run on a schedule (every 30-60 min) so the org workspace stays
live. Pulls from a deeper pool of message templates per channel and varies
the personas so no two runs look identical.

Required env:
    SLACK_BOT_TOKEN   xoxb-... with chat:write + channels:join + channels:read

Slack rate limit: chat.postMessage is tier 1 (~1 req/sec). We pace at 0.6s.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone


# ── Personas drawn at random ──────────────────────────────────────────────────
PERSONAS = {
    "engineering": [
        ("Maya Chen", "VP Engineering"),
        ("Priya Subramanian", "Frontend Lead"),
        ("Anna Hoffmann", "Backend Lead"),
        ("Diego Ramírez", "Execution Engineer"),
        ("Jian Wu", "Risk Engineer"),
        ("Sina Hassani", "Data Engineer"),
        ("Karl Nyström", "Junior IC"),
    ],
    "research": [
        ("Sofia Karlsson", "VP Research"),
        ("Aarav Patel", "Alpha Research Director"),
        ("Hugo Bernardes", "Quant Researcher"),
        ("Tomas Lindqvist", "Research Scientist"),
        ("Linh Tran", "ML Modeling Lead"),
        ("Yuki Mori", "Options Researcher"),
        ("Lior Avraham", "Polymarket Researcher"),
    ],
    "ops": [
        ("Kenji Watanabe", "Director of DevOps"),
        ("Aditi Sharma", "Director of QA"),
        ("Cameron Park", "Security Engineer"),
        ("Helena Voss", "Compliance Engineer"),
        ("Wei Chang", "Finance Engineer"),
        ("Ravi Iyer", "ML Infra Engineer"),
    ],
    "exec": [
        ("Laavanye Bahl", "CEO/Founder"),
        ("Marcus Olufemi", "CRO"),
        ("Maya Chen", "VP Engineering"),
        ("Sofia Karlsson", "VP Research"),
    ],
    "bot": [
        ("Pipeline bot", "automated"),
        ("Risk bot", "automated"),
        ("PnL bot", "automated"),
        ("Deploy bot", "automated"),
        ("CI bot", "automated"),
    ],
}


# ── Message templates by channel ──────────────────────────────────────────────
# Each entry: (persona_group, template). Templates use {var} placeholders
# filled at runtime.

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "AMD",
           "NFLX", "JPM", "BAC", "WMT", "XOM", "JNJ", "PFE", "COST", "PEP"]
CRYPTOS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
STRATEGIES = ["momentum", "ml_momentum", "pairs_trading", "tsmom", "options_pcr_reversal",
              "low_volatility", "breakout", "ml_breakout", "triangular_arb", "funding_rate_arb"]


def _rand_ticker() -> str: return random.choice(TICKERS)
def _rand_crypto() -> str: return random.choice(CRYPTOS)
def _rand_strategy() -> str: return random.choice(STRATEGIES)
def _rand_pct(lo=-2.0, hi=2.0) -> str: return f"{random.uniform(lo, hi):+.2f}%"
def _rand_pnl(lo=-500, hi=2000) -> str: return f"${random.uniform(lo, hi):+,.2f}"
def _rand_sharpe(lo=0.4, hi=2.4) -> str: return f"{random.uniform(lo, hi):.2f}"
def _rand_bps(lo=-15, hi=15) -> str: return f"{random.uniform(lo, hi):+.1f} bps"


TEMPLATES: dict[str, list[tuple[str, callable]]] = {
    "pnl-daily": [
        ("bot",
         lambda: f":bar_chart: hourly P&L update — {_rand_strategy()} {_rand_pnl()}, sharpe {_rand_sharpe()} (24h rolling)"),
        ("research",
         lambda: f"{_rand_strategy()} on {_rand_ticker()} just closed {_rand_pnl()} in {random.randint(5, 240)} minutes"),
        ("research",
         lambda: f"day-PnL leaderboard: 1. {_rand_strategy()} {_rand_pnl(100, 1500)}  2. {_rand_strategy()} {_rand_pnl(50, 800)}  3. {_rand_strategy()} {_rand_pnl(0, 400)}"),
    ],
    "risk-alerts": [
        ("bot",
         lambda: f":warning: position concentration: {_rand_ticker()} now {random.uniform(7, 14):.1f}% of NAV (limit 12%)"),
        ("bot",
         lambda: f":warning: VaR(95%, 1d) breach: actual {_rand_pnl(-5000, -3000)} vs limit -$3,500"),
        ("ops",
         lambda: f"Reviewing the breach. Vol regime shifted to *elevated* — recomputing VaR with the wide-tail model."),
        ("bot",
         lambda: f":white_check_mark: VaR within limits. Last breach {random.randint(4, 48)}h ago."),
    ],
    "deploys": [
        ("bot",
         lambda: f":rocket: backend deploy `{random.choice(['da2f4cb','c7a346b','542bf1d','817798c','e1deee9'])}` — {random.choice(['security fix','strategy add','frontend bundle','CI fix','infra bump'])} — ✅ healthy ({random.randint(45, 180)}s)"),
        ("bot",
         lambda: f":rocket: frontend deploy — bundle {random.randint(280, 490)}KB, lighthouse {random.randint(86, 97)}/100 — ✅"),
    ],
    "ci-failures": [
        ("bot",
         lambda: f":warning: CI {random.choice(['flaky','slow','green-on-retry'])} on `claude/advanced-trading-bot-d5Lmw` — {random.choice(['test_pairs_trading','test_kelly','test_features','test_a3c_lstm'])} took {random.uniform(40, 90):.1f}s"),
        ("ops",
         lambda: "Looked at it. Heavy fixture setup. Will move to module-scoped session."),
    ],
    "ml-experiments": [
        ("research",
         lambda: f"LSTM v{random.randint(3, 6)} on {_rand_crypto()} — val_acc {random.uniform(0.55, 0.66):.3f}, test sharpe {_rand_sharpe()}"),
        ("research",
         lambda: f"XGBoost HPO finished — {random.randint(60, 120)} Optuna trials, best params: lr={random.uniform(0.01, 0.1):.3f}, max_depth={random.randint(4, 8)}, n_estimators={random.randint(200, 800)}"),
        ("research",
         lambda: f"Ensemble weight optimization on val set: LSTM {random.uniform(0.2, 0.5):.2f}, XGB {random.uniform(0.2, 0.4):.2f}, Lorentzian {random.uniform(0.1, 0.3):.2f}, TFT {random.uniform(0.1, 0.3):.2f}"),
        ("ops",
         lambda: f"Kaggle GPU quota: {random.randint(8, 30)} T4 hours remaining this week."),
    ],
    "alpha-research": [
        ("research",
         lambda: f"new idea: {random.choice(['cross-sectional skewness premium','vol-of-vol mean reversion','dispersion trade on QQQ vs components','term structure carry on VX futures','overnight drift bias on small-caps'])} — drafting backtest config"),
        ("exec",
         lambda: "Reminder: walk-forward, not in-sample. No exceptions for the new idea."),
        ("research",
         lambda: f"yesterday's idea backtest came in: Sharpe {_rand_sharpe()}, MaxDD {random.uniform(8, 18):.1f}%. Promising — moving to paper next."),
    ],
    "incidents": [
        ("bot",
         lambda: f":green_heart: All systems nominal. Uptime {random.randint(7, 30)} days. No incidents in last 24h."),
        ("ops",
         lambda: f"Chaos test scheduled tomorrow 02:00 UTC — testing strategy_runner under simulated config-reload load."),
    ],
    "wins": [
        ("research",
         lambda: f":tada: {_rand_strategy()} on {_rand_ticker()} hit {random.uniform(2, 7):.1f}% return today, paper-trading"),
        ("engineering",
         lambda: f":tada: bundle size now {random.randint(275, 310)}KB — under the {random.randint(280, 320)}KB target"),
        ("ops",
         lambda: f":tada: {random.randint(14, 45)} days zero unplanned downtime"),
    ],
    "desk-equities": [
        ("research",
         lambda: f"{_rand_ticker()} signal: {random.choice(['breakout','momentum','mean-reversion','factor-tilt'])} — entry {random.uniform(100, 500):.2f}, conf {random.uniform(0.6, 0.85):.2f}"),
        ("engineering",
         lambda: f"filled {_rand_ticker()} at {random.uniform(100, 500):.2f} via {random.choice(['LimitFirst','TWAP','VWAP'])}, slippage {_rand_bps()}"),
    ],
    "desk-crypto": [
        ("research",
         lambda: f"{_rand_crypto()} funding rate: {random.uniform(-0.01, 0.02):.4f}% / 8h — {random.choice(['arb opp','neutral','crowded long'])}"),
        ("research",
         lambda: f"{_rand_crypto()} BTC-dom {random.uniform(48, 58):.1f}%, eth/btc ratio {random.uniform(0.04, 0.07):.4f}"),
    ],
    "desk-options": [
        ("research",
         lambda: f"PCR on {_rand_ticker()} = {random.uniform(0.8, 1.6):.2f} ({random.randint(40, 95)}th percentile, 1yr)"),
        ("research",
         lambda: f"IV30 {_rand_ticker()} {random.uniform(15, 65):.1f}% vs realized {random.uniform(10, 40):.1f}% — {random.choice(['rich','cheap','fair'])}"),
    ],
    "desk-polymarket": [
        ("research",
         lambda: f"poly scan: {random.randint(0, 5)} arb opportunities (YES+NO < $0.97). Top: '{random.choice(['fed-cut-q3','recession-2026','btc-100k-2027','election-margin'])}' at ${random.uniform(0.88, 0.96):.2f}"),
    ],
    "news-feed": [
        ("bot",
         lambda: f":newspaper: *{random.choice(['Reuters','Bloomberg','WSJ','FT','CoinDesk'])}* — {random.choice([_rand_ticker()+' beats earnings, guides higher','Fed signals patience on rate path','BTC ETF flows turn positive','Treasury 10yr +'+str(random.randint(2,8))+'bps on jobs print','Earnings season kicks off — banks first up'])}"),
    ],
    "earnings-watch": [
        ("research",
         lambda: f"{_rand_ticker()} earnings {random.choice(['Mon','Tue','Wed','Thu'])} after close. Implied move {random.uniform(3, 9):.1f}%, hist avg actual {random.uniform(2, 6):.1f}%"),
    ],
    "fed-watch": [
        ("research",
         lambda: f"Fed futures: {random.randint(60, 95)}% no-change, {random.randint(5, 30)}% cut at next FOMC. 2yr {random.uniform(3.8, 5.2):.2f}%"),
    ],
    "papers": [
        ("research",
         lambda: f":books: paper drop: '{random.choice(['Deep RL for portfolio choice','Transformer architecture for limit order books','Robust factor investing under regime change','Cryptocurrency momentum and reversal','Options market microstructure'])}' — adding to queue"),
    ],
    "random": [
        ("engineering",
         lambda: random.choice([
             "anyone else hit the 'sqlite locked' bug yesterday?",
             "coffee #4. send help.",
             "PSA: my dog learned to bark at the kafka consumer lag alert",
             "took 3 hours to find a bug. it was a missing await.",
             "the markets are closed but our bot still has opinions",
         ])),
        ("research",
         lambda: random.choice([
             "I dreamed about a covariance matrix that wouldn't invert.",
             "if you nest async generators inside list comprehensions one more time…",
             "TIL: pandas merge with how='left' on duplicate keys does a cartesian. learned the hard way.",
         ])),
    ],
    "leadership-summary": [
        ("exec",
         lambda: f"*Engineering daily* — shipped: {random.randint(1,6)} PRs. in flight: {random.choice(['code-splitting','LSTM v4','Render deploy','TFT walk-forward','rate-limit hardening'])}. blocked: {random.choice(['none','GPU quota','vendor approval','none'])}"),
        ("exec",
         lambda: f"*Research daily* — backtest sharpe leaderboard top: {_rand_strategy()} {_rand_sharpe()}, {_rand_strategy()} {_rand_sharpe()}, {_rand_strategy()} {_rand_sharpe()}"),
        ("exec",
         lambda: f"*Risk daily* — VaR(95) {_rand_pnl(-4500, -1500)}, current DD {random.uniform(-9, -1):.1f}%, bucket allocation {random.choice(['70/30','65/30/5'])}"),
    ],
    "infra-alerts": [
        ("bot",
         lambda: f":green_heart: Render p99 {random.randint(45, 110)}ms · Supabase queries p99 {random.randint(10, 35)}ms · Upstash hit-rate {random.randint(91, 99)}%"),
    ],
    "security-alerts": [
        ("bot",
         lambda: f":closed_lock_with_key: secret scan: {random.randint(0,0)} leaks · dependabot: {random.randint(0,3)} pending PRs · last rotation: {random.randint(2,28)} days ago"),
    ],
    "okrs": [
        ("exec",
         lambda: f"Q2 OKR check — strategies-live: {random.randint(45, 50)}/50, portfolio sharpe {_rand_sharpe()} (target 2.0), max DD {random.uniform(-12, -4):.1f}% (target -15%)"),
    ],
    "show-and-tell": [
        ("engineering",
         lambda: f"Friday demo: {random.choice(['Comparison page','SHAP feature importance UI','Slippage tracker','Walk-forward results explorer','Order book depth heatmap'])} — 10min, pairs well with espresso"),
    ],
    "competitors": [
        ("exec",
         lambda: f"Tracking {random.choice(['Two Sigma','Renaissance','Citadel','D.E. Shaw','Jane Street','Hudson River'])} — {random.choice(['hiring slowed','public fund down YTD','new strategy paper','infra migration to'])}"),
    ],
}


def slack_call(token: str, method: str, payload: dict) -> dict:
    url = f"https://slack.com/api/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"http_{e.code}", "body": e.read().decode()[:200]}


def list_channels(token: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    cursor = ""
    while True:
        payload: dict = {"types": "public_channel,private_channel", "limit": 200}
        if cursor:
            payload["cursor"] = cursor
        data = slack_call(token, "conversations.list", payload)
        if not data.get("ok"):
            return out
        for ch in data.get("channels", []):
            out[ch["name"]] = ch
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break
    return out


def pick_persona(group: str) -> tuple[str, str]:
    name, role = random.choice(PERSONAS[group])
    return name, role


def post(token: str, channel_id: str, persona_name: str, persona_role: str, text: str) -> dict:
    return slack_call(token, "chat.postMessage", {
        "channel": channel_id,
        "text": f"_{persona_role}_\n{text}",
        "username": persona_name,
        "mrkdwn": True,
    })


def main() -> int:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        print("❌ SLACK_BOT_TOKEN missing or not xoxb-")
        return 1

    auth = slack_call(token, "auth.test", {})
    if not auth.get("ok"):
        print(f"❌ auth.test failed: {auth}")
        return 1
    print(f"✅ Authed as {auth.get('user')} in {auth.get('team')} at {datetime.now(timezone.utc).isoformat()}")

    channels = list_channels(token)

    # Decide how many messages this run: 8-15 total
    total_msgs = random.randint(8, 15)
    print(f"📋 Posting ~{total_msgs} messages")

    # Pick channels weighted toward higher-activity ones
    weights = {
        "pnl-daily": 3, "risk-alerts": 2, "deploys": 2, "ml-experiments": 3,
        "alpha-research": 3, "engineering": 2, "wins": 2, "incidents": 1,
        "desk-equities": 3, "desk-crypto": 2, "desk-options": 2, "desk-polymarket": 1,
        "news-feed": 2, "papers": 1, "random": 2, "leadership-summary": 1,
        "infra-alerts": 1, "security-alerts": 1, "okrs": 1, "show-and-tell": 1,
        "competitors": 1, "ci-failures": 1, "fed-watch": 1, "earnings-watch": 1,
    }
    pool = [c for c, w in weights.items() for _ in range(w) if c in TEMPLATES]
    random.shuffle(pool)
    sampled = pool[:total_msgs]

    posted, errors = 0, 0
    for ch_name in sampled:
        if ch_name not in channels:
            continue
        templates = TEMPLATES.get(ch_name, [])
        if not templates:
            continue
        persona_group, render = random.choice(templates)
        name, role = pick_persona(persona_group)
        text = render()
        ch_id = channels[ch_name]["id"]

        # Try to join if public
        if not channels[ch_name].get("is_private", False):
            slack_call(token, "conversations.join", {"channel": ch_id})

        r = post(token, ch_id, name, role, text)
        if r.get("ok"):
            posted += 1
            print(f"  ✓ #{ch_name}: {name} ({role})")
        else:
            errors += 1
            print(f"  ✗ #{ch_name}: {r.get('error')}")
        time.sleep(0.6)

    print(f"\n✅ Posted {posted}/{total_msgs} messages, {errors} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
