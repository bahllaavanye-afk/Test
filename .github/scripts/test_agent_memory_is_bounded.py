"""`conversations` was unbounded, and it is 47% of the git repository.

`agent_memory.json` has three writers — `claude_conversations.py`,
`employee_conversation_runner.py`, `multi_agent_discussion.py`. All three append
timestamp-keyed entries to `conversations`. None of them removed any.

Measured 2026-08-05:

    conversations      576 KB   915 entries   <- no cap
    employee_context   231 KB    11 entries
    peer_learnings      38 KB   200 entries   <- capped [-100:] / [-200:]
    failure_traces       7 KB    41 entries   <- capped [-200:]
                       ------
    agent_memory.json  933 KB

Growth was ~7 KB per commit (917,453 B at 00:15 → 933,178 B at 02:29), and
**200 of the last 200 commits touching the repo rewrote the whole blob**. Git
therefore stores 2340 historical copies: **59.1 MB of a 125 MB `.git`.**

The retention bought nothing. The only reader is `context_sync.py`:

    convs = mem.get("conversations", {})
    recent = sorted(convs.items())[-20:]      # then displays 10

~900 entries were kept to serve a consumer that reads 20. The file already
capped its other structures — the discipline simply never reached the largest,
fastest-growing one.

This bounds future growth. It does NOT shrink the existing 59 MB of history;
that needs a history rewrite, which is deliberately not done here.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from shared_context import CONVERSATION_CAP, trim_conversations  # noqa: E402


def _mem(n: int, shuffle: bool = True) -> dict:
    """n entries with ISO-8601 keys, inserted in NON-chronological order.

    Shuffling is load-bearing. An earlier version inserted keys already sorted,
    which made `list(convs.items())[-cap:]` and `sorted(convs.items())[-cap:]`
    behave identically — the fixture could not tell insertion order from
    chronological order, and a mutation dropping the sort survived. Real
    agent_memory.json is a JSON round-trip of appends from three unrelated
    processes; its key order is not guaranteed.
    """
    keys = [f"2026-08-05T{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}.000000+00:00"
            for i in range(n)]
    order = list(keys)
    if shuffle:
        import random
        random.Random(1234).shuffle(order)
    return {"conversations": {k: {"speaker": f"e{k}", "message": f"m{k}"} for k in order}}


def test_it_trims_to_the_cap():
    mem = _mem(CONVERSATION_CAP + 250)
    dropped = trim_conversations(mem)
    assert len(mem["conversations"]) == CONVERSATION_CAP
    assert dropped == 250, "the returned count must match what was actually removed"


def test_it_keeps_the_NEWEST_entries():
    """Chronology is the whole point: the reader wants the last 20.

    Keys are ISO-8601 UTC, so lexicographic sort IS chronological. A trim that
    kept the head would leave `context_sync` displaying the oldest conversations
    in the file — stale, and indistinguishable from fresh ones in the output.
    """
    mem = _mem(CONVERSATION_CAP + 100)
    expected = set(sorted(mem["conversations"])[-CONVERSATION_CAP:])
    trim_conversations(mem)
    assert set(mem["conversations"]) == expected, (
        "the survivors are not the chronologically newest entries. The keys are "
        "inserted out of order here on purpose: a trim that slices insertion "
        "order rather than sorted order passes a pre-sorted fixture and fails "
        "this one."
    )


def test_a_file_under_the_cap_is_untouched():
    mem = _mem(10)
    before = dict(mem["conversations"])
    assert trim_conversations(mem) == 0
    assert mem["conversations"] == before


def test_the_cap_exceeds_what_any_reader_asks_for():
    """context_sync reads [-20:]. The cap must not starve it."""
    src = (_DIR / "context_sync.py").read_text()
    assert "sorted(convs.items())[-20:]" in src, (
        "context_sync's conversation window changed; re-check CONVERSATION_CAP "
        "against it before trusting this bound."
    )
    assert CONVERSATION_CAP >= 100, (
        f"CONVERSATION_CAP={CONVERSATION_CAP} leaves little headroom over the "
        "20 the only reader takes, and one run can write ~48 entries."
    )


@pytest.mark.parametrize("bad", [None, [], "text", 42, {"conversations": None}])
def test_malformed_memory_degrades_instead_of_raising(bad):
    """This runs inside three unrelated save paths, one of which is repaired by
    system_watchdog after corruption. Raising here would lose the whole save."""
    mem = {"conversations": bad} if not isinstance(bad, dict) else bad
    assert trim_conversations(mem) == 0


def test_missing_key_is_fine():
    assert trim_conversations({}) == 0
    assert trim_conversations({"peer_learnings": []}) == 0


@pytest.mark.parametrize("module,saver", [
    ("claude_conversations.py", "def save_memory"),
    ("employee_conversation_runner.py", "def _save_memory"),
])
def test_the_named_savers_trim_before_writing(module, saver):
    """A cap only one writer honours is no cap: the others refill the file."""
    src = (_DIR / module).read_text()
    idx = src.index(saver)
    body = src[idx:idx + 900]
    assert "trim_conversations" in body or "_trim_conversations" in body, (
        f"{module}'s {saver} writes agent_memory.json without trimming — the "
        "other writers' caps are undone on its next run."
    )


def test_multi_agent_discussion_trims_before_its_write():
    """It has no save function — the write is inline at the end of main."""
    src = (_DIR / "multi_agent_discussion.py").read_text()
    write = src.index("MEMORY_FILE.write_text")
    assert "trim_conversations" in src[:write], (
        "multi_agent_discussion writes agent_memory.json with no trim before it."
    )


PRODUCERS = {"claude_conversations.py", "employee_conversation_runner.py",
             "multi_agent_discussion.py"}


def test_every_producer_of_conversation_entries_is_covered():
    """Guards against a FOURTH producer quietly reopening the leak.

    Note what this does NOT require. `agent_memory.json` has ~14 writers
    (agent_chat_handler, heartbeat, signal_runner, peer_reviewer, system_watchdog,
    …), and only these three need the trim. Every writer is a separate
    short-lived process that loads the file, mutates, and rewrites it — so a
    non-producer can only preserve the entries it read. **Trimmed entries are
    never resurrected**, and a writer that adds no conversation entries can never
    push the dict past the cap. Patching all 14 would be 11 no-op edits and 11
    chances to regress.

    So the invariant to guard is narrow: whoever ADDS entries must trim.
    Detected by AST (subscript assignment into a `conversations` mapping) rather
    than by grep, so a renamed local still counts.
    """
    producers = set()
    for py in sorted(_DIR.glob("*.py")):
        if py.name.startswith("test_"):
            continue
        text = py.read_text()
        if "conversations" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if not isinstance(t, ast.Subscript):
                    continue
                # Match the EXACT key. A substring match on "conversation"
                # also caught agent_chat_handler's `agent_conversations`, a
                # different structure that is already capped at [-100:] — a
                # false positive that would have sent the next reader to patch
                # code with nothing wrong with it.
                container = ast.unparse(t.value)
                is_conversations_map = (
                    container == "conversations"
                    or container.endswith("['conversations']")
                    or container.endswith('["conversations"]')
                    or "setdefault('conversations'" in container
                    or 'setdefault("conversations"' in container
                )
                if is_conversations_map:
                    producers.add(py.name)
    new = sorted(producers - PRODUCERS)
    assert not new, (
        f"new producer(s) of conversation entries: {new}. Each appends to an "
        "unbounded dict — call shared_context.trim_conversations(mem) before "
        "their save, then add them to PRODUCERS here."
    )
    assert producers == PRODUCERS, (
        f"a known producer stopped producing: {sorted(PRODUCERS - producers)}. "
        "If it was removed, drop it from PRODUCERS; if it was renamed, this "
        "test is no longer watching it."
    )
