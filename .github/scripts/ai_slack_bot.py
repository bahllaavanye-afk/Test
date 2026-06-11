"""
QuantEdge AI Slack Bot — every employee can @mention @QuantEdge-AI in any channel.

Triggered by: Slack Events API (app_mention events forwarded to this script).
Also runs as a GitHub Actions scheduled job to post proactive market insights.

Capabilities per employee role:
  Strategy team  → ask about backtest results, signal quality, strategy ideas
  ML team        → model metrics, experiment comparisons, feature importance
  Risk team      → drawdown alerts, circuit breaker status, VaR analysis
  CTO            → system health, deployment status, full platform overview

Usage in Slack:
  @QuantEdge-AI what is the Sharpe ratio of the momentum strategy?
  @QuantEdge-AI explain the current market regime
  @QuantEdge-AI compare ml_momentum vs momentum backtest results
  @QuantEdge-AI what strategies are beating SPY today?

Requires env vars:
  SLACK_BOT_TOKEN     — xoxb-... (chat:write, app_mentions:read, channels:history)
  ANTHROPIC_API_KEY   — for QuantEdge AI API calls
  GH_TOKEN            — to read repo files / experiment results
  GH_REPO             — e.g. bahllaavanye-afk/QuantEdge
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Zero-spend policy: Anthropic API must never be called in CI or automated runs.
if os.environ.get("ALLOW_PAID_APIS", "False").lower() == "true":
    print("ALLOW_PAID_APIS=True is not permitted — exiting.")
    sys.exit(1)

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

import httpx

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GH_TOKEN = os.environ.get("GH_TOKEN", "")
GH_REPO = os.environ.get("GH_REPO", "bahllaavanye-afk/QuantEdge")
REPO_ROOT = Path(__file__).parent.parent.parent

SLACK_API = "https://slack.com/api"

# ── Repo context for QuantEdge AI ──────────────────────────────────────────────────

def _read_experiment_results() -> str:
    """Load latest backtest results for context."""
    results_dir = REPO_ROOT / "experiments" / "results"
    if not results_dir.exists():
        return "No experiment results found."
    files = sorted(results_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
    summaries = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            summaries.append(
                f"Strategy: {data.get('strategy', f.stem)} | "
                f"Symbol: {data.get('symbol', '?')} | "
                f"Sharpe: {data.get('sharpe', '?')} | "
                f"Total Return: {data.get('total_return', '?'):.1%}"
                if isinstance(data.get('total_return'), float) else
                f"Strategy: {data.get('strategy', f.stem)} | {json.dumps(data)[:120]}"
            )
        except Exception:
            continue
    return "\n".join(summaries) if summaries else "No results yet."


def _read_pipeline_state() -> str:
    """Read latest pipeline run state."""
    pipeline_file = REPO_ROOT / "pipeline_runs.json"
    if not pipeline_file.exists():
        return "No pipeline run data."
    try:
        runs = json.loads(pipeline_file.read_text())
        if not runs:
            return "No pipeline runs recorded."
        latest = runs[-1] if isinstance(runs, list) else runs
        return json.dumps(latest, indent=2)[:800]
    except Exception:
        return "Pipeline state unreadable."


SYSTEM_PROMPT = """You are QuantEdge-AI, the institutional AI assistant for the QuantEdge quantitative trading platform.

You have access to:
- 60+ trading strategies across 6 desks: Equities, Crypto, Options, Polymarket, FX/Macro, StatArb
- ML models: LSTM, Transformer (TFT), XGBoost, LightGBM, Lorentzian KNN, SSM ensemble
- Risk management: Kelly sizing, circuit breakers, HRP portfolio optimization
- Backtesting: walk-forward validation, Monte Carlo, benchmark comparison vs SPY/BRK.B/QQQ/All Weather

Your team:
- Strategy Researchers: improve signal quality, backtest new ideas
- ML Engineers: model training, experiment tracking, feature engineering
- Risk Officers: position sizing, drawdown monitoring, correlation limits
- CTO: system architecture, deployment, performance

When answering:
- Be concise and quantitative — give numbers, not vague statements
- Use markdown tables for comparisons
- Reference specific files/functions when relevant
- If asked about a strategy, give its Sharpe, max drawdown, and key signal logic
- Never make up performance numbers — use the provided experiment results

Current platform status:
{status}

Latest experiment results:
{results}
"""


# ── Slack helpers ────────────────────────────────────────────────────────────

def _post(channel: str, text: str, thread_ts: str | None = None) -> dict:
    payload: dict = {"channel": channel, "text": text, "mrkdwn": True}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    try:
        r = httpx.post(
            f"{SLACK_API}/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            json=payload,
            timeout=10,
        )
        result = r.json()
        if not result.get("ok"):
            error_code = result.get("error", "unknown_error")
            print(f"  Slack API error posting to {channel}: {error_code}", flush=True)
            # Fallback: try #general if specific channel fails
            if channel != "#general" and error_code in ("channel_not_found", "not_in_channel", "is_archived"):
                print(f"  Falling back to #general...", flush=True)
                fallback_payload = {**payload, "channel": "#general",
                                    "text": f"[intended for {channel}]\n{text}"}
                fr = httpx.post(
                    f"{SLACK_API}/chat.postMessage",
                    headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
                    json=fallback_payload,
                    timeout=10,
                )
                return fr.json()
        return result
    except httpx.TimeoutException:
        print(f"  Slack API timeout posting to {channel}", flush=True)
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        print(f"  Slack API exception posting to {channel}: {exc}", flush=True)
        return {"ok": False, "error": str(exc)}


def _get_channel_history(channel: str, limit: int = 5) -> list[dict]:
    r = httpx.get(
        f"{SLACK_API}/conversations.history",
        headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
        params={"channel": channel, "limit": limit},
        timeout=10,
    )
    return r.json().get("messages", [])


# ── QuantEdge AI response ──────────────────────────────────────────────────────────

def _ask_free_llm(question: str, system: str) -> str:
    """Fallback to free LLMs (Gemini → Groq → DeepSeek) when Anthropic key is disabled."""
    import urllib.request

    gemini_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY_1", ""))
    if gemini_key:
        try:
            payload = json.dumps({
                "contents": [{"parts": [{"text": f"{system}\n\n{question}"}]}],
                "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.3},
            }).encode()
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"Gemini fallback failed: {e}")

    groq_key = os.environ.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY_1", ""))
    if groq_key:
        try:
            payload = json.dumps({
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
                "max_tokens": 1024,
            }).encode()
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {groq_key}"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Groq fallback failed: {e}")

    return "⚠️ No LLM available (Anthropic disabled, Gemini/Groq keys missing) — QuantEdge AI cannot respond."


def _ask_claude(question: str, channel_context: str = "") -> str:
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "disabled":
        status = _read_pipeline_state()
        results = _read_experiment_results()
        system = SYSTEM_PROMPT.format(status=status, results=results)
        if channel_context:
            system += f"\n\nRecent channel context:\n{channel_context}"
        return _ask_free_llm(question, system)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    status = _read_pipeline_state()
    results = _read_experiment_results()

    system = SYSTEM_PROMPT.format(status=status, results=results)
    if channel_context:
        system += f"\n\nRecent channel context:\n{channel_context}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # fast + cheap for Slack responses
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# ── Proactive daily insights ─────────────────────────────────────────────────

PROACTIVE_PROMPTS = [
    ("What are the top 3 strategy opportunities for today based on current market conditions?",
     "#ml-experiments"),
    ("Summarise the latest experiment results and which ML model is performing best.",
     "#ml-experiments"),
    ("What risk alerts should the team be aware of today?",
     "#risk-alerts"),
    ("Which strategies in the options desk should be prioritised today based on VIX levels?",
     "#desk-options"),
    ("Provide a morning briefing on the crypto desk — funding rates, BTC trend, top signals.",
     "#desk-crypto"),
]


def run_proactive_insights() -> None:
    """Post QuantEdge AI-generated insights to relevant channels."""
    if not SLACK_TOKEN:
        print("No SLACK_BOT_TOKEN — skipping proactive insights")
        return
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "disabled":
        gemini_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY_1", ""))
        groq_key = os.environ.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY_1", ""))
        if not gemini_key and not groq_key:
            print("No LLM keys available — skipping proactive insights")
            return

    hour = datetime.now(timezone.utc).hour
    # Morning (9am UTC): full briefing. Afternoon (13, 17): quick updates. Evening (21): recap.
    if hour == 9:
        prompts = PROACTIVE_PROMPTS
    elif hour in (13, 17):
        prompts = PROACTIVE_PROMPTS[1:3]
    else:
        prompts = PROACTIVE_PROMPTS[:1]

    for question, channel in prompts:
        print(f"  Posting to {channel}: {question[:60]}...", flush=True)
        answer = _ask_claude(question)
        result = _post(channel, f"*QuantEdge-AI insight:*\n{answer}")
        if result.get("ok"):
            print(f"  Posted successfully (ts={result.get('ts')})", flush=True)
        else:
            print(f"  Post failed: {result.get('error', 'unknown')}", flush=True)


def handle_mention(event: dict) -> None:
    """Respond to an @QuantEdge-AI mention."""
    channel = event.get("channel", "")
    ts = event.get("ts")
    text = event.get("text", "").strip()
    # Strip the bot mention prefix <@BOTID>
    if ">" in text:
        text = text[text.index(">") + 1:].strip()

    if not text:
        text = "What can you tell me about the platform status?"

    # Get recent channel context
    history = _get_channel_history(channel, limit=3)
    context = "\n".join(
        f"{m.get('username', 'user')}: {m.get('text', '')[:200]}"
        for m in history if m.get("text")
    )

    print(f"  Responding to: {text[:80]}...", flush=True)
    answer = _ask_claude(text, context)
    _post(channel, answer, thread_ts=ts)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "proactive"

    if mode == "proactive":
        print("Running proactive insights...", flush=True)
        run_proactive_insights()
    elif mode == "mention" and len(sys.argv) > 2:
        event = json.loads(sys.argv[2])
        handle_mention(event)
    else:
        print("Usage: ai_slack_bot.py [proactive | mention '<event_json>']")
