#!/usr/bin/env python3
"""One-time Slack message monitor — sweep every channel, classify every message with a free LLM.

Reads all accessible Slack channels (and their threads), sends each human message
through a free-tier LLM (Groq / Gemini by default, with cascade fallback) and emits:
  - a JSON report with a per-message classification (category, urgency, sentiment,
    action_needed, one-line summary)
  - a Markdown digest grouped by urgency, surfacing anything that needs attention

This is a one-shot analysis tool, not a daemon. Run it whenever you want a fresh
read of the workspace.

Requirements (read from the environment / .env):
  SLACK_BOT_TOKEN   Slack bot token with channels:read, channels:history,
                    groups:read, groups:history (and im/mpim:* for DMs).
  GROQ_API_KEY      and/or GEMINI_API_KEY (or any provider llm_common supports).

Usage:
  python scripts/slack_message_monitor.py
  python scripts/slack_message_monitor.py --channel general --limit 200
  python scripts/slack_message_monitor.py --since-days 7 --provider gemini
  python scripts/slack_message_monitor.py --dry-run        # skip LLM, just dump messages
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "slack_monitor"

# ── Reuse the shared free-LLM cascade if it's importable ──────────────────────
# llm_common lives in .github/scripts/. If present we use llm_with_provider so the
# user's chosen provider (Groq/Gemini) is preferred, with cascade fallback.
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))
try:
    from llm_common import llm_with_provider as _llm_with_provider  # type: ignore

    _HAVE_LLM_COMMON = True
except Exception:  # noqa: BLE001
    _HAVE_LLM_COMMON = False


# ── Minimal fallback LLM caller (Groq + Gemini) ───────────────────────────────
# Used only when llm_common is unavailable (e.g. running outside the repo layout).
_FALLBACK_PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "fmt": "openai",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "key_env": "GEMINI_API_KEY",
        "fmt": "gemini",
    },
}


def _fallback_llm(prompt: str, system: str, provider: str, max_tokens: int) -> str:
    order = [provider] + [p for p in _FALLBACK_PROVIDERS if p != provider]
    for name in order:
        cfg = _FALLBACK_PROVIDERS.get(name)
        if not cfg:
            continue
        key = os.environ.get(cfg["key_env"], "")
        if not key or key == "disabled":
            continue
        try:
            if cfg["fmt"] == "gemini":
                url = f"{cfg['url']}?key={key}"
                body = {
                    "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{prompt}"}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
                }
                headers = {"Content-Type": "application/json"}
            else:
                url = cfg["url"]
                body = {
                    "model": cfg["model"],
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                }
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            if cfg["fmt"] == "gemini":
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] fallback provider {name} failed: {e}", file=sys.stderr)
    return ""


def call_llm(prompt: str, system: str, provider: str, max_tokens: int = 300) -> str:
    """Call the preferred free LLM, falling back across providers."""
    if _HAVE_LLM_COMMON:
        text, _used = _llm_with_provider(
            prompt, provider, system=system, max_tokens=max_tokens,
            temperature=0.2, inject_company_context=False,
        )
        return text
    return _fallback_llm(prompt, system, provider, max_tokens)


# ── Slack API (stdlib only) ───────────────────────────────────────────────────
class SlackError(RuntimeError):
    pass


def _slack_get(method: str, token: str, **params: Any) -> dict:
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
    url = f"https://slack.com/api/{method}?{qs}"
    # Slack rate-limits (HTTP 429 with Retry-After, or ok=false/error=ratelimited).
    # Respect Retry-After and back off exponentially instead of crashing.
    for attempt in range(7):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 6:
                wait = int(e.headers.get("Retry-After", "") or 0) or (2 ** attempt)
                time.sleep(wait + 1)
                continue
            raise
        if not data.get("ok"):
            if data.get("error") == "ratelimited" and attempt < 6:
                time.sleep((2 ** attempt) + 1)
                continue
            raise SlackError(f"{method}: {data.get('error', 'unknown error')}")
        return data
    raise SlackError(f"{method}: exhausted rate-limit retries")


def list_channels(token: str) -> list[dict]:
    """List every channel the bot can see (public + private)."""
    channels: list[dict] = []
    cursor = ""
    while True:
        data = _slack_get(
            "conversations.list", token,
            types="public_channel,private_channel",
            limit=200, cursor=cursor, exclude_archived="true",
        )
        channels.extend(data.get("channels", []))
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break
    return channels


def channel_history(token: str, channel_id: str, limit: int, oldest: float) -> list[dict]:
    """Read a channel's top-level messages (newest first), capped at `limit`."""
    messages: list[dict] = []
    cursor = ""
    while len(messages) < limit:
        data = _slack_get(
            "conversations.history", token, channel=channel_id,
            limit=min(200, limit - len(messages)),
            cursor=cursor, oldest=oldest or None,
        )
        messages.extend(data.get("messages", []))
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break
    return messages[:limit]


def thread_replies(token: str, channel_id: str, thread_ts: str) -> list[dict]:
    try:
        data = _slack_get("conversations.replies", token, channel=channel_id, ts=thread_ts, limit=200)
    except SlackError:
        return []
    # First message is the parent (already captured); return only the replies.
    return data.get("messages", [])[1:]


# ── Classification ────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a Slack message triage assistant at QuantEdge, a quantitative trading firm. "
    "Classify a single message. Respond with STRICT JSON only, no prose, no code fences."
)

_PROMPT_TEMPLATE = """Classify this Slack message. Return JSON with exactly these keys:
  "category": one of [trading, engineering, ops, risk, hr, sales, social, noise, other]
  "urgency": one of [p0, high, medium, low]   (p0 = system down / money at risk)
  "sentiment": one of [positive, neutral, negative]
  "action_needed": true or false
  "summary": a <=15 word plain-language summary

Channel: #{channel}
Author: {user}
Message: {text}
"""


def _is_human_message(m: dict) -> bool:
    if m.get("type") != "message":
        return False
    if m.get("subtype") in {"channel_join", "channel_leave", "bot_message", "channel_topic"}:
        return False
    return bool((m.get("text") or "").strip())


def classify_message(m: dict, channel_name: str, provider: str) -> dict:
    text = (m.get("text") or "").strip()
    prompt = _PROMPT_TEMPLATE.format(
        channel=channel_name, user=m.get("user", "unknown"), text=text[:2000]
    )
    raw = call_llm(prompt, _SYSTEM, provider, max_tokens=200)
    parsed = _safe_json(raw)
    return {
        "ts": m.get("ts"),
        "channel": channel_name,
        "user": m.get("user", "unknown"),
        "text": text[:500],
        "classification": parsed,
    }


def _safe_json(raw: str) -> dict:
    """Best-effort parse of an LLM JSON reply."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"category": "other", "urgency": "low", "sentiment": "neutral",
            "action_needed": False, "summary": raw[:120] or "[unparseable LLM reply]"}


# ── Reporting ─────────────────────────────────────────────────────────────────
def write_reports(results: list[dict], meta: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORTS_DIR / f"report_{stamp}.json"
    md_path = REPORTS_DIR / f"report_{stamp}.md"

    json_path.write_text(json.dumps({"meta": meta, "messages": results}, indent=2))

    by_urgency: dict[str, list[dict]] = {"p0": [], "high": [], "medium": [], "low": []}
    for r in results:
        u = (r.get("classification") or {}).get("urgency", "low")
        by_urgency.setdefault(u, by_urgency["low"] if u not in by_urgency else []).append(r)

    lines = [
        f"# Slack Message Monitor — {stamp}",
        "",
        f"- Channels scanned: **{meta['channels_scanned']}**",
        f"- Messages classified: **{meta['messages_classified']}**",
        f"- LLM provider: **{meta['provider']}**" + ("" if meta["llm_used"] else " (dry-run, not called)"),
        "",
    ]
    for urgency in ("p0", "high", "medium", "low"):
        items = by_urgency.get(urgency, [])
        if not items:
            continue
        lines.append(f"## {urgency.upper()} ({len(items)})")
        for r in items:
            c = r.get("classification") or {}
            flag = " ⚠️ action" if c.get("action_needed") else ""
            lines.append(f"- `#{r['channel']}` — {c.get('summary', '?')} "
                         f"_({c.get('category', '?')}, {c.get('sentiment', '?')})_{flag}")
        lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="One-time Slack sweep + free-LLM classification.")
    ap.add_argument("--channel", help="Only scan this channel (name without #).")
    ap.add_argument("--limit", type=int, default=100, help="Max top-level messages per channel.")
    ap.add_argument("--since-days", type=int, default=0, help="Only messages newer than N days.")
    ap.add_argument("--provider", default="groq", help="Preferred LLM provider (groq|gemini|...).")
    ap.add_argument("--no-threads", action="store_true", help="Skip thread replies.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel classification workers.")
    ap.add_argument("--dry-run", action="store_true", help="Collect messages but skip the LLM.")
    args = ap.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        print("ERROR: SLACK_BOT_TOKEN is not set. Add it to your environment or .env.", file=sys.stderr)
        return 2

    oldest = time.time() - args.since_days * 86400 if args.since_days else 0.0

    print("Listing channels…")
    channels = list_channels(token)
    if args.channel:
        channels = [c for c in channels if c.get("name") == args.channel]
        if not channels:
            print(f"ERROR: channel '{args.channel}' not found or not accessible.", file=sys.stderr)
            return 2
    print(f"  {len(channels)} channel(s) to scan.")

    # Collect every human message across all channels (+ threads).
    collected: list[tuple[dict, str]] = []
    for ch in channels:
        cid = ch["id"]
        name = ch.get("name", cid)
        try:
            msgs = channel_history(token, cid, args.limit, oldest)
        except SlackError as e:
            print(f"  [warn] #{name}: {e}", file=sys.stderr)
            continue
        for m in msgs:
            if _is_human_message(m):
                collected.append((m, name))
            if not args.no_threads and int(float(m.get("reply_count", 0) or 0)) > 0:
                for reply in thread_replies(token, cid, m["ts"]):
                    if _is_human_message(reply):
                        collected.append((reply, name))
        print(f"  #{name}: {len(msgs)} messages read")

    print(f"\nCollected {len(collected)} human message(s).")

    results: list[dict]
    if args.dry_run:
        results = [{"ts": m.get("ts"), "channel": name, "user": m.get("user", "unknown"),
                    "text": (m.get("text") or "")[:500], "classification": None}
                   for m, name in collected]
    else:
        print(f"Classifying with '{args.provider}' (+ fallback)…")

        def _do(item: tuple[dict, str]) -> dict:
            m, name = item
            return classify_message(m, name, args.provider)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            results = list(ex.map(_do, collected))

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "channels_scanned": len(channels),
        "messages_classified": len(results),
        "provider": args.provider,
        "llm_used": not args.dry_run,
        "since_days": args.since_days,
    }
    json_path, md_path = write_reports(results, meta)
    print(f"\nDone.\n  JSON: {json_path}\n  Markdown: {md_path}")

    if not args.dry_run:
        action = sum(1 for r in results if (r.get("classification") or {}).get("action_needed"))
        p0 = sum(1 for r in results if (r.get("classification") or {}).get("urgency") == "p0")
        print(f"  Flagged: {p0} P0, {action} needing action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
