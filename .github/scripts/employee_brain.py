"""Per-employee brain: private memory + history + directed collaboration.

Each QuantEdge employee (vp_eng, alpha_dir, risk_eng, …) gets its OWN
persistent brain instead of sharing one global blob:

  • private history — a rolling log of what this employee did/said/decided
  • private facts   — durable key→value notes the employee keeps about its domain
  • recall          — a compact context block (own history + facts + what peers
                      recently shared) to prepend to the employee's LLM prompt,
                      so it remembers across runs
  • collaboration   — share() posts a one-line learning to a shared bus that
                      OTHER employees read; learn_from_peers() reads theirs

Storage lives in the existing `.github/state/agent_memory.json`:
  • `employee_context[emp]`  — the per-employee brain (this file finally fills
    the slot that was defined in the schema everywhere but always left `{}`)
  • `peer_learnings`         — the shared collaboration bus. Entries stay in the
    legacy string form `"[emp @ iso] text"` so existing readers
    (agent_chat_handler, agent_status_checker) keep working unchanged.

Everything is best-effort and never raises into the caller — a brain problem
must never take an employee down. Stdlib-only.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_STATE = Path(__file__).resolve().parents[1] / "state" / "agent_memory.json"

_HISTORY_CAP = 60        # per-employee history entries kept
_PEER_CAP = 200          # shared bus size (matches the existing cap)
_PEER_PREFIX = re.compile(r"^\[([^\]@]+?)\s*@\s*([^\]]+)\]\s*(.*)$", re.S)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(emp: str) -> str:
    return (emp or "unknown").strip().lower()


class EmployeeBrain:
    """One employee's slice of the shared memory file. Load, mutate, save()."""

    def __init__(self, employee_id: str, path: Path | str = _STATE) -> None:
        self.emp = _norm(employee_id)
        self.path = Path(path)
        self._mem = self._load()
        ctx = self._mem.setdefault("employee_context", {})
        self.brain = ctx.setdefault(self.emp, {"history": [], "facts": {}, "updated_at": None})
        self.brain.setdefault("history", [])
        self.brain.setdefault("facts", {})

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
                return d if isinstance(d, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def save(self) -> bool:
        """Atomically write the whole memory file back. Never raises."""
        self.brain["updated_at"] = _now()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._mem, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
            return True
        except OSError as exc:
            print(f"[brain] save failed for {self.emp}: {str(exc)[:80]}")
            try:
                os.unlink(tmp)  # type: ignore[name-defined]
            except (OSError, NameError, UnboundLocalError):
                pass
            return False

    # ── private memory ───────────────────────────────────────────────────────
    def remember(self, kind: str, text: str, meta: dict | None = None) -> None:
        """Append an entry to this employee's private rolling history."""
        if not text:
            return
        self.brain["history"].append({
            "ts": _now(), "kind": str(kind), "text": str(text)[:1000],
            **({"meta": meta} if meta else {}),
        })
        self.brain["history"] = self.brain["history"][-_HISTORY_CAP:]

    def recent_history(self, n: int = 10, kind: str | None = None) -> list[dict]:
        h = self.brain["history"]
        if kind:
            h = [e for e in h if e.get("kind") == kind]
        return h[-max(0, n):]

    def note(self, key: str, value) -> None:
        """Set a durable fact this employee keeps about its domain."""
        self.brain["facts"][str(key)] = value

    def get_fact(self, key: str, default=None):
        return self.brain["facts"].get(str(key), default)

    def facts(self) -> dict:
        return dict(self.brain["facts"])

    # ── collaboration (shared bus, backward-compatible string form) ──────────
    def share(self, text: str) -> None:
        """Post a one-line learning other employees will read."""
        if not text:
            return
        bus = self._mem.setdefault("peer_learnings", [])
        bus.append(f"[{self.emp} @ {_now()}] {str(text)[:500]}")
        self._mem["peer_learnings"] = bus[-_PEER_CAP:]

    def learn_from_peers(self, n: int = 5, exclude_self: bool = True) -> list[dict]:
        """Most-recent learnings from OTHER employees, newest last.
        Returns parsed {from, ts, text} dicts (author unknown → from=None)."""
        out: list[dict] = []
        for raw in self._mem.get("peer_learnings", []):
            m = _PEER_PREFIX.match(raw) if isinstance(raw, str) else None
            if m:
                author, ts, text = m.group(1).strip().lower(), m.group(2).strip(), m.group(3).strip()
            else:
                author, ts, text = None, None, str(raw)
            if exclude_self and author == self.emp:
                continue
            out.append({"from": author, "ts": ts, "text": text})
        return out[-max(0, n):]

    # ── recall: the context block injected into the employee's prompt ────────
    def context_block(self, history_n: int = 5, peers_n: int = 4, facts_n: int = 6) -> str:
        """Compact memory summary to prepend to this employee's system prompt.
        Empty string when the brain is cold (nothing to recall yet)."""
        parts: list[str] = []

        facts = list(self.brain["facts"].items())[:facts_n]
        if facts:
            parts.append("What you know (your durable notes):")
            parts += [f"  • {k}: {v}" for k, v in facts]

        hist = self.recent_history(history_n)
        if hist:
            parts.append("What you did recently:")
            parts += [f"  • [{e['kind']}] {e['text'][:200]}" for e in hist]

        peers = self.learn_from_peers(peers_n)
        if peers:
            parts.append("What teammates just shared:")
            parts += [f"  • {p['from'] or 'peer'}: {p['text'][:200]}" for p in peers]

        if not parts:
            return ""
        return ("=== YOUR MEMORY (private to you, persists across runs) ===\n"
                + "\n".join(parts)
                + "\n=== end memory ===\n")


def brain_for(employee_id: str, path: Path | str = _STATE) -> EmployeeBrain:
    return EmployeeBrain(employee_id, path)


def record_interaction(employee_id: str, task: str, output: str,
                       share_line: str | None = None, path: Path | str = _STATE) -> bool:
    """Convenience: remember one task→output and optionally share a learning,
    then persist. Fail-open (returns False on any error, never raises)."""
    try:
        b = brain_for(employee_id, path)
        if task:
            b.remember("task", task, )
        if output:
            b.remember("output", output)
        if share_line:
            b.share(share_line)
        return b.save()
    except Exception as exc:  # noqa: BLE001
        print(f"[brain] record_interaction failed: {str(exc)[:80]}")
        return False
