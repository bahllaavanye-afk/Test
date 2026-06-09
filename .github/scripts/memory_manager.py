"""
Shared memory manager — the portable context layer that works across ALL LLM providers.

Core principle: model weights cannot be shared across providers.
What CAN be shared is external context injected into every prompt.

This module provides three things:

1. ConversationStore — saves/loads multi-turn dialogue in OpenAI messages format.
   A conversation started by Gemini can be continued by Groq, Cerebras, or any other
   provider — because all of them accept the same messages array.

   Use: store = ConversationStore("strategy_discussion")
        store.add("user", "What's the Sharpe on the momentum strategy?")
        store.add("assistant", "It was 1.4 last backtest period...")
        # Next call to ANY provider:
        messages = store.build_messages(system="You are...")

2. SemanticRetriever — keyword-weighted search over company brain memory.
   Instead of always injecting the 5 most recent episodic entries,
   this retrieves the most RELEVANT entries based on what the current prompt is about.
   Uses TF-IDF style scoring — no external ML libraries, runs anywhere.

   Use: retriever = SemanticRetriever()
        results = retriever.search("momentum strategy drawdown", n=5)

3. ContextBudget — assembles the final context blob within a token budget.
   Priorities: (1) CORE memory, (2) retrieved relevant episodes,
               (3) recent Slack insights, (4) recent trade outcomes.
   Ensures total injected context never exceeds the budget.

All state lives in .github/state/company_brain.json — the single source of truth
readable and writable by every agent regardless of which LLM they call.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

# ── State paths ───────────────────────────────────────────────────────────────

_STATE_DIR = Path(os.environ.get("GITHUB_WORKSPACE", ".")) / ".github" / "state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)
_BRAIN_FILE = _STATE_DIR / "company_brain.json"
_CONV_DIR = _STATE_DIR / "conversations"
_CONV_DIR.mkdir(parents=True, exist_ok=True)

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


# ── 1. Conversation Store ─────────────────────────────────────────────────────

class ConversationStore:
    """
    Persists multi-turn conversation history in OpenAI messages format.
    Any LLM provider can load and continue a conversation started by another.

    Usage:
        store = ConversationStore("momentum_analysis")
        # First turn with Gemini:
        store.add("user", "What is the current Sharpe ratio?")
        reply = llm_call(store.build_messages(system="You are a quant analyst."))
        store.add("assistant", reply)

        # Second turn — Gemini is down, Groq takes over seamlessly:
        store.add("user", "How does that compare to last month?")
        reply = llm_call(store.build_messages(system="You are a quant analyst."))
        # Groq sees full conversation history, continues naturally.
    """

    MAX_TURNS = 20        # max turns to persist (older pruned to save tokens)
    MAX_TOKEN_EST = 4000  # estimated max tokens for history before pruning

    def __init__(self, conversation_id: str) -> None:
        self.id = conversation_id
        self._path = _CONV_DIR / f"{conversation_id}.json"
        self._messages: list[Message] = []
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._messages = data.get("messages", [])
        except Exception:
            self._messages = []

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps({
                "id": self.id,
                "updated_at": time.time(),
                "messages": self._messages,
            }, indent=2))
        except Exception:
            pass

    def add(self, role: str, content: str) -> None:
        """Append a message. Prunes oldest turns if history grows too long."""
        self._messages.append({"role": role, "content": content})
        # Prune: keep most recent MAX_TURNS non-system messages
        non_system = [m for m in self._messages if m["role"] != "system"]
        if len(non_system) > self.MAX_TURNS:
            # Drop oldest pairs to stay under budget
            keep = non_system[-(self.MAX_TURNS):]
            self._messages = keep
        self._save()

    def build_messages(self, system: str) -> list[Message]:
        """
        Build the messages array to send to any OpenAI-compatible endpoint.
        System prompt always comes first. Full history follows.
        """
        return [{"role": "system", "content": system}] + self._messages

    def last_assistant_reply(self) -> str:
        """Get the most recent assistant response."""
        for m in reversed(self._messages):
            if m["role"] == "assistant":
                return m["content"]
        return ""

    def clear(self) -> None:
        self._messages = []
        self._save()

    def summary(self) -> str:
        """Token-efficient summary for injection into other conversations."""
        if not self._messages:
            return ""
        lines = []
        for m in self._messages[-6:]:  # last 3 turns
            prefix = "Q" if m["role"] == "user" else "A"
            lines.append(f"{prefix}: {m['content'][:100]}")
        return f"[Prior conversation '{self.id}']: " + " | ".join(lines)

    @classmethod
    def list_all(cls) -> list[str]:
        return [p.stem for p in _CONV_DIR.glob("*.json")]

    @classmethod
    def load_or_create(cls, conversation_id: str) -> "ConversationStore":
        return cls(conversation_id)


# ── 2. Semantic Retriever (TF-IDF, no ML libraries) ───────────────────────────

class SemanticRetriever:
    """
    Keyword-weighted retrieval over company brain episodic memory.

    Why not a vector DB? Because:
    - GitHub Actions has no GPU, limited RAM
    - sentence-transformers takes 100MB+ and 30s to load
    - TF-IDF gives 80% of the relevance for 0% of the setup cost

    TF-IDF here: score each memory entry by term overlap with query,
    weighted by inverse document frequency (rare terms score higher).
    """

    def __init__(self) -> None:
        self._brain_cache: dict | None = None
        self._idf: dict[str, float] = {}
        self._built = False

    def _load_brain(self) -> dict:
        if self._brain_cache is None:
            try:
                if _BRAIN_FILE.exists():
                    self._brain_cache = json.loads(_BRAIN_FILE.read_text())
                else:
                    self._brain_cache = {}
            except Exception:
                self._brain_cache = {}
        return self._brain_cache

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenizer, lowercased, stop-words stripped."""
        stop = {"the", "a", "an", "is", "in", "on", "at", "to", "for",
                "of", "and", "or", "with", "by", "it", "this", "that",
                "was", "has", "have", "are", "be", "been", "from", "as"}
        words = re.findall(r"[a-z]{3,}", text.lower())
        return [w for w in words if w not in stop]

    def _build_idf(self, documents: list[str]) -> None:
        """Compute inverse document frequency across all memory entries."""
        N = len(documents)
        if N == 0:
            return
        doc_freq: dict[str, int] = {}
        for doc in documents:
            seen = set(self._tokenize(doc))
            for w in seen:
                doc_freq[w] = doc_freq.get(w, 0) + 1
        self._idf = {w: math.log(N / (1 + df)) for w, df in doc_freq.items()}
        self._built = True

    def _score(self, query_terms: list[str], doc_terms: list[str]) -> float:
        if not doc_terms:
            return 0.0
        doc_tf: dict[str, float] = {}
        for w in doc_terms:
            doc_tf[w] = doc_tf.get(w, 0) + 1
        total = len(doc_terms)
        score = 0.0
        for w in query_terms:
            if w in doc_tf:
                tf = doc_tf[w] / total
                idf = self._idf.get(w, 1.0)
                score += tf * idf
        return score

    def search(self, query: str, n: int = 5, categories: list[str] | None = None) -> list[dict]:
        """
        Retrieve the n most relevant memory entries for a given query.

        Args:
            query: Natural language query (e.g. "momentum strategy performance")
            n: Number of results to return
            categories: Which memory categories to search. Default: all.

        Returns:
            List of memory dicts sorted by relevance, most relevant first.
        """
        brain = self._load_brain()
        search_cats = categories or ["episodic", "skills", "slack_insights",
                                     "github_insights", "trade_outcomes", "experiment_results"]

        # Collect all candidate entries with their text
        candidates: list[tuple[float, dict]] = []
        all_texts: list[str] = []
        all_entries: list[dict] = []

        for cat in search_cats:
            for entry in brain.get(cat, []):
                text = (
                    entry.get("lesson", "") or
                    entry.get("summary", "") or
                    entry.get("insight", "") or
                    entry.get("text", "") or
                    str(entry)
                )
                all_texts.append(text)
                all_entries.append({**entry, "_category": cat})

        if not all_texts:
            return []

        # Build IDF on first call
        if not self._built:
            self._build_idf(all_texts)

        query_terms = self._tokenize(query)
        for text, entry in zip(all_texts, all_entries):
            doc_terms = self._tokenize(text)
            score = self._score(query_terms, doc_terms)
            if score > 0:
                candidates.append((score, entry))

        # Sort by score descending, return top n
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in candidates[:n]]

    def search_text(self, query: str, n: int = 5) -> str:
        """Returns retrieved entries as a formatted string for prompt injection."""
        results = self.search(query, n)
        if not results:
            return ""
        lines = []
        for r in results:
            text = (r.get("lesson") or r.get("summary") or r.get("insight") or "")[:150]
            cat = r.get("_category", "?")
            lines.append(f"[{cat}] {text}")
        return "\n".join(lines)


# ── 3. Context Budget Manager ─────────────────────────────────────────────────

class ContextBudget:
    """
    Assembles the context block injected into every prompt, within a token budget.

    Priority order (highest → lowest):
      1. CORE stable facts (regime, top strategies, risk status) — always included
      2. Relevant episodic memory retrieved by semantic search on the current query
      3. Recent Slack insights (last 2 entries)
      4. Recent trade outcomes (last 2 entries)
      5. Relevant skills/solutions

    Estimates tokens as len(text) // 4 (conservative approximation).
    """

    def __init__(self, budget: int = 700) -> None:
        self.budget = budget
        self._retriever = SemanticRetriever()

    def _est_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def build(self, query: str = "") -> str:
        """
        Build a context block relevant to the given query.
        Returns empty string if nothing worth injecting.
        """
        try:
            if not _BRAIN_FILE.exists():
                return ""
            brain = json.loads(_BRAIN_FILE.read_text())
        except Exception:
            return ""

        parts: list[str] = []
        used = 0

        # 1. CORE memory — always first
        core = brain.get("core", {})
        core_lines = []
        if core.get("market_regime", "unknown") != "unknown":
            core_lines.append(f"Regime: {core['market_regime']}")
        if core.get("risk_status", "normal") != "normal":
            core_lines.append(f"Risk: {core['risk_status']}")
        top = core.get("top_strategies", [])
        if top:
            core_lines.append(f"Top strategies: {', '.join(top[:3])}")
        best_model = core.get("best_model")
        if best_model:
            core_lines.append(f"Best model: {best_model}")
        if core_lines:
            core_text = "CORE: " + " | ".join(core_lines)
            parts.append(core_text)
            used += self._est_tokens(core_text)

        # 2. Semantically relevant episodic memory
        if query and used < self.budget:
            relevant = self._retriever.search_text(query, n=4)
            if relevant:
                token_est = self._est_tokens(relevant)
                if used + token_est <= self.budget:
                    parts.append("RELEVANT MEMORY:\n" + relevant)
                    used += token_est

        # 3. Recent Slack insights
        slack_insights = brain.get("slack_insights", [])[-2:]
        if slack_insights and used < self.budget:
            slack_lines = [i.get("summary", "")[:100] for i in slack_insights if i.get("summary")]
            if slack_lines:
                text = "SLACK: " + " | ".join(slack_lines)
                if used + self._est_tokens(text) <= self.budget:
                    parts.append(text)
                    used += self._est_tokens(text)

        # 4. Recent trade outcomes
        trades = brain.get("trade_outcomes", [])[-2:]
        if trades and used < self.budget:
            trade_lines = []
            for t in trades:
                strat = t.get("strategy", "?")
                outcome = t.get("outcome", t.get("pnl_pct", "?"))
                trade_lines.append(f"{strat}: {outcome}")
            if trade_lines:
                text = "TRADES: " + " | ".join(trade_lines)
                if used + self._est_tokens(text) <= self.budget:
                    parts.append(text)
                    used += self._est_tokens(text)

        if not parts:
            return ""

        return "[CONTEXT]\n" + "\n".join(parts) + "\n[/CONTEXT]"


# ── Convenience singletons ────────────────────────────────────────────────────

_retriever: SemanticRetriever | None = None
_budget: ContextBudget | None = None


def get_retriever() -> SemanticRetriever:
    global _retriever
    if _retriever is None:
        _retriever = SemanticRetriever()
    return _retriever


def get_context_budget(max_tokens: int = 700) -> ContextBudget:
    global _budget
    if _budget is None:
        _budget = ContextBudget(max_tokens)
    return _budget


def build_context(query: str = "", max_tokens: int = 700) -> str:
    """One-liner: build relevant context for a prompt."""
    return get_context_budget(max_tokens).build(query)


def retrieve(query: str, n: int = 5) -> list[dict]:
    """One-liner: semantic search over all memory categories."""
    return get_retriever().search(query, n)
