"""
Memory Consolidator — runs nightly.

Problem: agent_memory.json["peer_learnings"] grows to 200 entries and older
knowledge disappears. Agents can't query "what do we know about momentum strategy?"

Solution (no GPU, no vector DB, pure Python):
1. Group peer_learnings by topic using keyword matching
2. For each topic cluster, ask LLM to synthesise into 1-2 bullet points
3. Write a "knowledge_base.json" keyed by topic — agents can query by topic
4. Write a "long_term_skills.json" of validated patterns (IC-tested alpha factors, etc.)

Architecture: Reflexion distillation (Shinn 2023) + knowledge condensation.
All LLM calls use free tier only. No paid APIs.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import requests

ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)


def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v:
            return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v:
                return v
    return ""


GROQ_KEY       = _resolve_key("GROQ_API_KEY")
DEEPSEEK_KEYS  = [k for k in [
    _resolve_key("DEEPSEEK_API_KEY"),
    os.environ.get("DEEPSEEK_API_KEY_2", ""),
    os.environ.get("DEEPSEEK_API_KEY_3", ""),
] if k]
SAMBANOVA_KEY  = _resolve_key("SAMBANOVA_API_KEY")
CEREBRAS_KEY   = _resolve_key("CEREBRAS_API_KEY")
HYPERBOLIC_KEY = _resolve_key("HYPERBOLIC_API_KEY")
TOGETHER_KEY   = _resolve_key("TOGETHER_API_KEY")
GEMINI_KEY     = _resolve_key("GEMINI_API_KEY")

REPO_ROOT  = Path(__file__).resolve().parents[2]
STATE_DIR  = REPO_ROOT / ".github" / "state"
MEMORY_FILE   = STATE_DIR / "agent_memory.json"
SKILL_FILE    = STATE_DIR / "skill_library.json"
KB_FILE       = STATE_DIR / "knowledge_base.json"
BRAIN_FILE    = STATE_DIR / "company_brain.json"


# Topic keywords for bucketing — pure string matching, no GPU needed
TOPICS = {
    "strategies":       ["strategy", "signal", "backtest", "momentum", "arb", "breakout", "pairs", "reversion", "sharpe", "alpha"],
    "ml_models":        ["lstm", "xgboost", "model", "training", "epoch", "loss", "accuracy", "tft", "ssm", "lorentzian", "ensemble"],
    "risk":             ["risk", "drawdown", "kelly", "position", "var", "cvar", "regime", "bear", "bull", "allocation"],
    "code_quality":     ["bug", "fix", "refactor", "import", "syntax", "test", "pytest", "mypy", "ruff", "lint", "commit"],
    "frontend":         ["react", "tsx", "component", "ui", "dashboard", "tailwind", "chart", "page", "route"],
    "infrastructure":   ["github actions", "workflow", "deploy", "render", "supabase", "redis", "upstash", "secrets", "api key"],
    "brokers":          ["alpaca", "binance", "polymarket", "tradestation", "order", "fill", "paper", "execution"],
    "data":             ["ohlcv", "data", "feature", "normalization", "funding rate", "open interest", "microstructure"],
}


def _classify_learning(text: str) -> str:
    """Return best-matching topic for a learning entry."""
    text_lower = text.lower()
    scores = {topic: sum(1 for kw in kws if kw in text_lower) for topic, kws in TOPICS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def call_llm(prompt: str, max_tokens: int = 250) -> str:
    messages = [{"role": "user", "content": prompt}]

    if GROQ_KEY:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": max_tokens},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Groq: {e}")

    for key in DEEPSEEK_KEYS:
        try:
            r = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": messages, "max_tokens": max_tokens},
                timeout=25,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"DeepSeek: {e}")

    if SAMBANOVA_KEY:
        try:
            r = requests.post(
                "https://api.sambanova.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {SAMBANOVA_KEY}", "Content-Type": "application/json"},
                json={"model": "Meta-Llama-3.1-8B-Instruct", "messages": messages, "max_tokens": max_tokens},
                timeout=25,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"SambaNova: {e}")

    if HYPERBOLIC_KEY:
        try:
            r = requests.post(
                "https://api.hyperbolic.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {HYPERBOLIC_KEY}", "Content-Type": "application/json"},
                json={"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": messages, "max_tokens": max_tokens},
                timeout=25,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Hyperbolic: {e}")

    if TOGETHER_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {TOGETHER_KEY}", "Content-Type": "application/json"},
                json={"model": "meta-llama/Llama-3.2-3B-Instruct-Turbo", "messages": messages, "max_tokens": max_tokens},
                timeout=25,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Together: {e}")

    if GEMINI_KEY:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": max_tokens}},
                timeout=25,
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"Gemini: {e}")

    return ""


def consolidate_topic(topic: str, entries: list[str]) -> str:
    """Distil N learning entries into 2-3 bullet points."""
    if not entries:
        return ""

    # If no LLM available, just return the most recent 3 as-is
    sample = entries[-10:]
    numbered = "\n".join(f"{i+1}. {e[:200]}" for i, e in enumerate(sample))

    prompt = (
        f"You are distilling collective knowledge for a quantitative trading AI platform.\n\n"
        f"Topic: {topic}\n\n"
        f"Recent agent learnings ({len(entries)} total, showing latest {len(sample)}):\n{numbered}\n\n"
        f"Synthesise these into 2-3 concise bullet points of the most important, actionable knowledge. "
        f"Use file paths and specific names where available. Be brief."
    )
    result = call_llm(prompt, max_tokens=200)

    if not result:
        return "\n".join(f"• {e[:150]}" for e in entries[-3:])
    return result


def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%H:%M UTC')}] Memory consolidation starting")

    try:
        mem = json.loads(MEMORY_FILE.read_text())
    except Exception:
        mem = {}

    learnings = mem.get("peer_learnings", [])
    if len(learnings) < 10:
        print(f"Only {len(learnings)} learnings — skipping consolidation (need 10+)")
        return 0

    print(f"  Classifying {len(learnings)} learnings into topics…")

    # Classify each learning by topic
    by_topic: dict[str, list[str]] = defaultdict(list)
    for l in learnings:
        topic = _classify_learning(l)
        by_topic[topic].append(l)

    for topic, entries in by_topic.items():
        print(f"  {topic}: {len(entries)} entries")

    # Load existing KB to merge with
    try:
        kb = json.loads(KB_FILE.read_text()) if KB_FILE.exists() else {}
    except Exception:
        kb = {}

    kb.setdefault("consolidated_at", now.isoformat())
    kb["last_updated"] = now.isoformat()
    kb["total_learnings_processed"] = len(learnings)
    kb.setdefault("topics", {})

    # Consolidate each topic (limit LLM calls to 3 topics per run — rate limit)
    topics_sorted = sorted(by_topic.items(), key=lambda x: -len(x[1]))
    llm_calls = 0

    for topic, entries in topics_sorted:
        print(f"  Consolidating {topic} ({len(entries)} entries)…")
        prev = kb["topics"].get(topic, {})
        prev_count = prev.get("source_count", 0)

        # Only re-consolidate if there are significant new entries
        if len(entries) - prev_count < 5 and llm_calls > 0:
            print(f"    Skipping {topic} — not enough new entries ({len(entries) - prev_count} new)")
            continue

        if llm_calls >= 3:
            # Rate limit — just store raw entries for remaining topics
            kb["topics"][topic] = {
                "summary": "\n".join(f"• {e[:150]}" for e in entries[-3:]),
                "source_count": len(entries),
                "updated_at": now.isoformat(),
                "raw_recent": entries[-5:],
            }
            continue

        summary = consolidate_topic(topic, entries)
        kb["topics"][topic] = {
            "summary": summary,
            "source_count": len(entries),
            "updated_at": now.isoformat(),
            "raw_recent": entries[-5:],
        }
        if summary:
            llm_calls += 1
        print(f"    → {summary[:100] if summary else '(no LLM)'}")

    # Write knowledge base
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    KB_FILE.write_text(json.dumps(kb, indent=2))

    # Inject top consolidated knowledge back into skill_library (so all agents get it)
    try:
        skill_data = json.loads(SKILL_FILE.read_text())
    except Exception:
        skill_data = {"version": 1, "skills": []}

    existing_skills = set(skill_data.get("skills", []))
    new_skills = []

    for topic, data in kb["topics"].items():
        summary = data.get("summary", "")
        # Extract bullet points as individual skills
        for line in summary.split("\n"):
            line = line.strip().lstrip("•-").strip()
            if len(line) > 20 and len(line) < 200 and line not in existing_skills:
                new_skills.append(line)
                existing_skills.add(line)

    if new_skills:
        skill_data["skills"] = list(existing_skills)[-200:]
        skill_data["last_updated"] = now.isoformat()
        skill_data["total"] = len(skill_data["skills"])
        SKILL_FILE.write_text(json.dumps(skill_data, indent=2))
        print(f"  Added {len(new_skills)} new skills to skill_library.json")

    # Write consolidated knowledge to shared company_brain.json
    try:
        brain = json.loads(BRAIN_FILE.read_text()) if BRAIN_FILE.exists() else {}
        brain.setdefault("learnings", [])
        brain.setdefault("agent_insights", {})
        # Record each new skill distilled this run
        for skill in new_skills:
            brain["learnings"].append({
                "source": "memory_consolidator",
                "skill": skill,
                "timestamp": now.isoformat(),
            })
        # Keep learnings bounded (last 500)
        brain["learnings"] = brain["learnings"][-500:]
        brain["agent_insights"]["memory_consolidator"] = {
            "last_run": now.isoformat(),
            "topics_consolidated": len(by_topic),
            "llm_calls": llm_calls,
            "new_skills_injected": len(new_skills),
        }
        brain["last_updated"] = now.isoformat()
        BRAIN_FILE.write_text(json.dumps(brain, indent=2))
        print(f"  company_brain.json updated (+{len(new_skills)} skills)")
    except Exception as e:
        print(f"  company_brain write error: {e}")

    print(
        f"✓ Consolidation complete: {len(by_topic)} topics, "
        f"{llm_calls} LLM calls, knowledge_base.json written"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
