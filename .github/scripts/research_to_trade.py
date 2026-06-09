"""
Research-to-Trade Pipeline — end-to-end autonomous chain.

Chain of events (fully autonomous, no human in loop):
  1. ResearchAgent    → scans market data + calls free LLM for alpha ideas
  2. SignalAgent      → validates ideas against 30d backtest signals
  3. RiskAgent        → Kelly size + drawdown check
  4. TradeAgent       → places paper order via Alpaca REST
  5. MonitorAgent     → tracks fill + P&L
  6. LeadReviewer     → reviews entire chain, flags issues to Slack

All inter-agent communication logged to Slack #desk-research channel.
Runs as GitHub Action every 30 minutes.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# ── Free LLM cascade ──────────────────────────────────────────────────────────

def _free_llm(prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str | None:
    """Try all free LLM providers in cascade. Return first successful response."""
    providers = [
        ("gemini",    os.getenv("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY_1", "")),
         "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini-2.0-flash"),
        ("groq",      os.getenv("GROQ_API_KEY", ""),
         "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
        ("deepseek",  os.getenv("DEEPSEEK_API_KEY", ""),
         "https://api.deepseek.com/v1/chat/completions", "deepseek-chat"),
        ("together",  os.getenv("TOGETHER_API_KEY", ""),
         "https://api.together.xyz/v1/chat/completions", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("cerebras",  os.getenv("CEREBRAS_API_KEY", ""),
         "https://api.cerebras.ai/v1/chat/completions", "llama-3.3-70b"),
        ("sambanova", os.getenv("SAMBANOVA_API_KEY", ""),
         "https://api.sambanova.ai/v1/chat/completions", "Meta-Llama-3.3-70B-Instruct"),
    ]
    import urllib.request
    for name, key, url, model in providers:
        if not key or key in ("", "disabled"):
            continue
        try:
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode()
            req = urllib.request.Request(url, data=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            print(f"  [LLM] Response from {name} ({len(text)} chars)")
            return text
        except Exception as e:
            print(f"  [LLM] {name} failed: {e}")
    return None


# ── Slack ─────────────────────────────────────────────────────────────────────

SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")

def slack(channel: str, text: str, thread_ts: str | None = None) -> str | None:
    """Post to Slack, return thread_ts for threading replies."""
    if not SLACK_TOKEN:
        return None
    try:
        payload: dict = {"channel": channel, "text": text, "mrkdwn": True}
        if thread_ts:
            payload["thread_ts"] = thread_ts
        with httpx.Client(timeout=10) as client:
            r = client.post("https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
                json=payload)
            return r.json().get("ts")
    except Exception as e:
        print(f"  [Slack] {e}")
        return None


# ── Market data ───────────────────────────────────────────────────────────────

def get_price_data(symbol: str, days: int = 30) -> list[dict] | None:
    """Fetch OHLCV from Yahoo Finance (free, no auth)."""
    import urllib.request
    from datetime import timedelta, date
    end = int(time.time())
    start = end - days * 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={start}&period2={end}&interval=1d")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        q = result["indicators"]["quote"][0]
        bars = []
        for i, ts in enumerate(timestamps):
            if q["close"][i] is None:
                continue
            bars.append({
                "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                "open": round(q["open"][i] or 0, 2),
                "high": round(q["high"][i] or 0, 2),
                "low": round(q["low"][i] or 0, 2),
                "close": round(q["close"][i] or 0, 2),
                "volume": int(q["volume"][i] or 0),
            })
        return bars[-20:]  # last 20 bars
    except Exception as e:
        print(f"  [Data] {symbol} fetch failed: {e}")
        return None


def get_crypto_price(symbol: str = "BTCUSDT") -> dict | None:
    import urllib.request
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def get_funding_rate(symbol: str = "BTCUSDT") -> float | None:
    import urllib.request
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
            return float(data.get("lastFundingRate", 0))
    except Exception:
        return None


# ── Agent classes ─────────────────────────────────────────────────────────────

class ResearchAgent:
    """Scans market conditions and proposes actionable trade ideas."""

    def run(self, thread_ts: str | None) -> list[dict]:
        print("[ResearchAgent] Scanning markets...")
        slack("#desk-research", "🔬 *ResearchAgent* scanning markets...", thread_ts)

        # Collect real market data
        market_summary = []
        for sym in ["SPY", "QQQ", "AAPL", "BTC"]:
            bars = get_price_data(sym if sym != "BTC" else "BTC-USD", days=20)
            if bars and len(bars) >= 5:
                last = bars[-1]
                prev = bars[-5]
                roc = (last["close"] / prev["close"] - 1) * 100
                market_summary.append(f"{sym}: ${last['close']:.2f} ({roc:+.1f}% 5d)")

        btc_funding = get_funding_rate("BTCUSDT")
        if btc_funding is not None:
            market_summary.append(f"BTC funding rate: {btc_funding*100:.4f}%")

        context = "\n".join(market_summary) if market_summary else "Market data unavailable"
        ts = datetime.now(timezone.utc).strftime("%H:%M UTC")

        prompt = f"""You are QuantEdge's ResearchAgent. Current time: {ts}

Real-time market data:
{context}

Generate 3 specific, actionable paper trade ideas right now. Each must be:
- Based on the actual price data shown above
- A real symbol (SPY/QQQ/AAPL/MSFT/NVDA/BTC-USD)
- Long or short with clear entry/exit logic
- Have a confidence score 0-100

Respond as JSON array only:
[{{"symbol":"SPY","side":"long","entry_price":450.00,"confidence":75,"rationale":"RSI oversold + EMA support","risk_pct":1.0}}]"""

        response = _free_llm(prompt, max_tokens=600)
        ideas = []
        if response:
            try:
                s, e = response.find("["), response.rfind("]") + 1
                if s >= 0 and e > s:
                    ideas = json.loads(response[s:e])
            except Exception:
                pass

        if ideas:
            summary = "\n".join([f"  • {i.get('symbol')} {i.get('side','?').upper()} — conf={i.get('confidence','?')} — {i.get('rationale','?')}" for i in ideas])
            slack("#desk-research", f"📊 *ResearchAgent found {len(ideas)} ideas:*\n{summary}\n\nMarket data:\n```{context}```", thread_ts)
        else:
            slack("#desk-research", f"⚠️ *ResearchAgent:* No ideas generated (LLM unavailable or parse failed)\nMarket context:\n```{context}```", thread_ts)

        return ideas


class SignalAgent:
    """Validates trade ideas using technical signals."""

    def run(self, ideas: list[dict], thread_ts: str | None) -> list[dict]:
        print(f"[SignalAgent] Validating {len(ideas)} ideas...")
        slack("#desk-research", f"⚡ *SignalAgent* validating {len(ideas)} research ideas...", thread_ts)

        validated = []
        for idea in ideas:
            symbol = idea.get("symbol", "")
            if not symbol:
                continue

            bars = get_price_data(symbol, days=30)
            if not bars or len(bars) < 14:
                idea["signal_validated"] = False
                idea["signal_reason"] = "insufficient data"
                continue

            closes = [b["close"] for b in bars]
            volumes = [b["volume"] for b in bars]

            # RSI
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [max(d, 0) for d in deltas[-14:]]
            losses = [-min(d, 0) for d in deltas[-14:]]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))

            # EMA 21
            ema21 = closes[-1]
            k = 2 / (21 + 1)
            for c in closes[-21:]:
                ema21 = c * k + ema21 * (1 - k)

            # Volume surge
            avg_vol = sum(volumes[-20:]) / 20
            vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1

            side = idea.get("side", "long")
            entry = idea.get("entry_price", closes[-1])

            # Validation rules
            valid = False
            reasons = []
            if side == "long" and rsi < 40:
                valid = True
                reasons.append(f"RSI oversold ({rsi:.0f})")
            if side == "long" and closes[-1] > ema21:
                valid = True
                reasons.append(f"price above EMA21")
            if side == "short" and rsi > 65:
                valid = True
                reasons.append(f"RSI overbought ({rsi:.0f})")
            if vol_ratio > 1.5:
                reasons.append(f"vol surge {vol_ratio:.1f}x")

            idea["signal_validated"] = valid
            idea["signal_reason"] = ", ".join(reasons) if reasons else "no confirming signals"
            idea["current_price"] = closes[-1]
            idea["rsi"] = round(rsi, 1)
            idea["vol_ratio"] = round(vol_ratio, 2)
            if valid:
                validated.append(idea)

        passed = len(validated)
        failed = len(ideas) - passed
        msg = f"✅ *SignalAgent:* {passed}/{len(ideas)} ideas validated\n"
        for i in ideas:
            icon = "✅" if i.get("signal_validated") else "❌"
            msg += f"  {icon} {i.get('symbol')} {i.get('side','?')}: {i.get('signal_reason','?')}\n"
        slack("#desk-research", msg, thread_ts)

        return validated


class RiskAgent:
    """Applies risk rules: Kelly sizing, max position, drawdown gate."""

    MAX_NOTIONAL_USD = 200.0   # max per trade on paper account
    MIN_CONFIDENCE   = 60      # minimum confidence to trade

    def run(self, ideas: list[dict], thread_ts: str | None) -> list[dict]:
        print(f"[RiskAgent] Sizing {len(ideas)} validated signals...")
        slack("#desk-research", f"🛡️ *RiskAgent* applying risk rules to {len(ideas)} signals...", thread_ts)

        approved = []
        msg_lines = []
        for idea in ideas:
            conf = idea.get("confidence", 50)
            if conf < self.MIN_CONFIDENCE:
                msg_lines.append(f"  ❌ {idea.get('symbol')} rejected: confidence {conf} < {self.MIN_CONFIDENCE}")
                continue

            # Fractional Kelly: f = (p - q/b) where p=win_prob, b=reward/risk
            p = min(conf / 100, 0.85)
            b = 2.0  # reward:risk ratio assumption
            q = 1 - p
            kelly_f = max((p - q / b), 0.05)
            fraction = min(kelly_f * 0.25, 0.10)  # quarter-Kelly, cap at 10%
            notional = round(min(fraction * 5000, self.MAX_NOTIONAL_USD), 2)  # $5k paper account

            price = idea.get("current_price", idea.get("entry_price", 100))
            qty = max(1, int(notional / price)) if price > 0 else 1

            idea["approved_qty"] = qty
            idea["approved_notional"] = round(qty * price, 2)
            idea["kelly_fraction"] = round(fraction, 4)
            approved.append(idea)
            msg_lines.append(
                f"  ✅ {idea.get('symbol')} {idea.get('side','?').upper()}: "
                f"qty={qty} @ ${price:.2f} = ${idea['approved_notional']:.0f} "
                f"(Kelly={fraction:.1%} conf={conf})"
            )

        slack("#desk-research",
              f"🛡️ *RiskAgent:* {len(approved)}/{len(ideas)} approved\n" + "\n".join(msg_lines),
              thread_ts)
        return approved


class TradeAgent:
    """Places paper orders via Alpaca REST API."""

    BASE_URL  = "https://paper-api.alpaca.markets"
    API_KEY   = os.getenv("ALPACA_API_KEY", "")
    API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

    def run(self, ideas: list[dict], thread_ts: str | None) -> list[dict]:
        print(f"[TradeAgent] Placing {len(ideas)} paper orders...")
        slack("#desk-research", f"📈 *TradeAgent* placing {len(ideas)} paper orders...", thread_ts)

        if not self.API_KEY or not self.API_SECRET:
            slack("#desk-research", "⚠️ *TradeAgent:* No Alpaca keys — orders simulated (paper only)", thread_ts)
            for idea in ideas:
                idea["order_id"] = f"SIMULATED_{idea['symbol']}_{int(time.time())}"
                idea["order_status"] = "simulated"
            return ideas

        filled = []
        for idea in ideas:
            symbol = idea.get("symbol", "").replace("-USD", "")
            side   = idea.get("side", "long")
            qty    = idea.get("approved_qty", 1)
            order_side = "buy" if side == "long" else "sell"

            # Skip non-equity symbols for Alpaca
            if any(c in symbol for c in ["BTC", "ETH", "SOL"]):
                idea["order_id"] = f"CRYPTO_SKIP_{symbol}"
                idea["order_status"] = "skipped_crypto"
                continue

            try:
                payload = {
                    "symbol": symbol,
                    "qty": str(qty),
                    "side": order_side,
                    "type": "market",
                    "time_in_force": "day",
                }
                with httpx.Client(timeout=10) as client:
                    r = client.post(
                        f"{self.BASE_URL}/v2/orders",
                        headers={"APCA-API-KEY-ID": self.API_KEY,
                                 "APCA-API-SECRET-KEY": self.API_SECRET},
                        json=payload,
                    )
                if r.status_code in (200, 201):
                    order = r.json()
                    idea["order_id"] = order.get("id", "?")
                    idea["order_status"] = order.get("status", "?")
                    filled.append(idea)
                    print(f"  [Trade] {symbol} {order_side} qty={qty}: {order.get('status')}")
                else:
                    idea["order_id"] = None
                    idea["order_status"] = f"error_{r.status_code}: {r.text[:80]}"
            except Exception as e:
                idea["order_id"] = None
                idea["order_status"] = f"exception: {e}"

        msg = f"📈 *TradeAgent orders:*\n"
        for idea in ideas:
            status = idea.get("order_status", "?")
            icon = "✅" if status in ("accepted", "pending_new", "simulated") else ("⏭️" if "skip" in status else "❌")
            msg += f"  {icon} {idea.get('symbol')} {idea.get('side','?')}: {status} (order={idea.get('order_id','?')[:16]})\n"
        slack("#desk-research", msg, thread_ts)
        return ideas


class MonitorAgent:
    """Checks fill status and P&L for orders placed this cycle."""

    BASE_URL   = "https://paper-api.alpaca.markets"
    API_KEY    = os.getenv("ALPACA_API_KEY", "")
    API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

    def run(self, ideas: list[dict], thread_ts: str | None) -> dict:
        print("[MonitorAgent] Checking fills and portfolio P&L...")
        slack("#desk-research", "📊 *MonitorAgent* checking fills + P&L...", thread_ts)

        pnl_summary = {"total_equity": None, "cash": None, "positions": [], "orders_filled": 0}

        if not self.API_KEY or not self.API_SECRET:
            slack("#desk-research", "⚠️ *MonitorAgent:* No Alpaca keys — skipping live P&L", thread_ts)
            return pnl_summary

        try:
            headers = {"APCA-API-KEY-ID": self.API_KEY, "APCA-API-SECRET-KEY": self.API_SECRET}
            with httpx.Client(timeout=10) as client:
                acct = client.get(f"{self.BASE_URL}/v2/account", headers=headers).json()
                positions = client.get(f"{self.BASE_URL}/v2/positions", headers=headers).json()

            equity = float(acct.get("equity", 0))
            cash   = float(acct.get("cash", 0))
            pnl    = float(acct.get("equity", 0)) - float(acct.get("last_equity", acct.get("equity", 0)))

            pnl_summary["total_equity"] = equity
            pnl_summary["cash"] = cash

            pos_lines = []
            for p in (positions if isinstance(positions, list) else [])[:8]:
                sym  = p.get("symbol", "?")
                qty  = p.get("qty", "?")
                upnl = float(p.get("unrealized_pl", 0))
                pct  = float(p.get("unrealized_plpc", 0)) * 100
                icon = "🟢" if upnl >= 0 else "🔴"
                pos_lines.append(f"  {icon} {sym} qty={qty}: ${upnl:+.2f} ({pct:+.1f}%)")
                pnl_summary["positions"].append({"symbol": sym, "upnl": upnl})

            pos_text = "\n".join(pos_lines) if pos_lines else "  (no open positions)"
            pnl_icon = "📈" if pnl >= 0 else "📉"
            msg = (f"📊 *MonitorAgent Portfolio Snapshot:*\n"
                   f"  Equity: *${equity:,.2f}* | Cash: ${cash:,.2f} | Today: {pnl_icon} ${pnl:+.2f}\n"
                   f"*Open Positions:*\n{pos_text}")
            slack("#desk-research", msg, thread_ts)
            pnl_summary["orders_filled"] = len([i for i in ideas if i.get("order_status") in ("accepted", "filled", "simulated")])

        except Exception as e:
            slack("#desk-research", f"⚠️ *MonitorAgent* error: {e}", thread_ts)

        return pnl_summary


class LeadReviewer:
    """Lead agent reviews the entire pipeline cycle and posts verdict."""

    def run(self, ideas_original: int, ideas_validated: int, ideas_traded: int,
            pnl: dict, thread_ts: str | None) -> None:
        print("[LeadReviewer] Reviewing pipeline cycle...")

        context = f"""Pipeline cycle review:
- Research found: {ideas_original} trade ideas
- Signal validated: {ideas_validated} ({100*ideas_validated/max(ideas_original,1):.0f}%)
- Risk approved & traded: {ideas_traded}
- Portfolio equity: ${pnl.get('total_equity') or 'N/A'}
- Open positions: {len(pnl.get('positions', []))}"""

        prompt = f"""You are QuantEdge's Lead Trading Desk Reviewer.

{context}

Review this pipeline cycle (2-3 sentences max):
1. Is the signal quality good? (validated rate should be >30%)
2. Any risk concerns?
3. One specific improvement for next cycle.

Sign off as "Lead Reviewer" if approved, "Lead Reviewer [REVIEW NEEDED]" if issues found."""

        verdict = _free_llm(prompt, max_tokens=200)
        if not verdict:
            verdict = "Lead Reviewer: Pipeline completed. LLM review unavailable — manual check recommended."

        # Post to both research and lead channels
        icon = "✅" if "[REVIEW NEEDED]" not in verdict else "⚠️"
        msg = f"{icon} *Lead Reviewer:*\n_{verdict}_"
        slack("#desk-research", msg, thread_ts)
        slack("#desk-lead-review", f"*Cycle at {datetime.now(timezone.utc).strftime('%H:%M UTC')}*\n{msg}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n=== Research-to-Trade Pipeline | {ts} ===\n")

    # Open a Slack thread for this entire cycle
    thread_ts = slack(
        "#desk-research",
        f"🚀 *Research-to-Trade Pipeline started* — {ts}\n"
        f"Chain: ResearchAgent → SignalAgent → RiskAgent → TradeAgent → Monitor → LeadReview"
    )

    try:
        research  = ResearchAgent()
        signals   = SignalAgent()
        risk      = RiskAgent()
        trade     = TradeAgent()
        monitor   = MonitorAgent()
        lead      = LeadReviewer()

        # Chain of events
        raw_ideas    = research.run(thread_ts)
        validated    = signals.run(raw_ideas, thread_ts)
        approved     = risk.run(validated, thread_ts)
        executed     = trade.run(approved, thread_ts)
        pnl          = monitor.run(executed, thread_ts)
        lead.run(len(raw_ideas), len(validated), len(executed), pnl, thread_ts)

        # Summary
        slack("#desk-research",
              f"✅ *Pipeline complete:* {len(raw_ideas)} researched → "
              f"{len(validated)} validated → {len(approved)} approved → "
              f"{len(executed)} traded",
              thread_ts)

        print(f"\n=== Pipeline done: {len(raw_ideas)} ideas → {len(executed)} trades ===")

    except Exception as e:
        tb = traceback.format_exc()
        slack("#desk-research", f"🔴 *Pipeline ERROR:* {e}\n```{tb[:500]}```", thread_ts)
        print(f"Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
